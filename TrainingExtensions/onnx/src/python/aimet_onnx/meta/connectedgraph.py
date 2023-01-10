#!/usr/bin/env python3.6

#  =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
#
#  =============================================================================

"""For constructing a uniform representation of the computational graph for a PyTorch model,
that is easy to navigate and stores information for the purpose of AIMET features.
The representation graph consists of nodes that are either 'operation' or 'product';
operations represent a module or a function that generates a tensor, while products represent
the tensors that are either input to the model (input, constant or parameter) or the
result of an operation. Furthermore the graph representation is bi-directional."""


from typing import List, Union
from onnx import onnx_pb
from onnxruntime.quantization.onnx_quantizer import ONNXModel

from aimet_common.connected_graph.connectedgraph import ConnectedGraph as AimetCommonConnectedGraph, get_ordered_ops
from aimet_common.utils import AimetLogger
from aimet_common.model_module import ONNXModelModule
from aimet_onnx.meta.operations import Op
from aimet_onnx.meta.product import Product
from aimet_onnx.utils import ParamUtils

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)

WEIGHT_INDEX = 1
BIAS_INDEX = 2
RUNNING_MEAN_INDEX = 3
RUNNING_VAR_INDEX = 4


class ConnectedGraph(AimetCommonConnectedGraph):
    """
    For construction of a graph that connects operations together (
    either module or functional) as producers and consumers of tensors.
    Note that the graph has two kinds of nodes: operations and products."""

    def __init__(self, model: onnx_pb.ModelProto):
        """
        :param: model: ONNX model to create connected graph from
        """
        super().__init__()
        self.model = model
        if isinstance(self.model, ONNXModel):
            self.model = self.model.model

        # Maps output to consumer node
        self._input_to_node = {}
        self._get_input_to_node()

        self.starting_ops = []
        self._branch_count = 0

        self.fill_op_product_graph()
        # List of ops in the order they are traversed using the forward function
        self.ordered_ops = get_ordered_ops(self.starting_ops)

    def get_op_from_module_name(self, name: str) -> Op:
        """
        Gets CG op given the module name
        :param name: Name of the module
        """
        return self._ops[name]

    def _get_input_to_node(self):
        """
        Maps input names to nodes
        """
        for node in self.model.graph.node:
            for input_name in node.input:
                if input_name in self._input_to_node:
                    self._input_to_node[input_name].append(node)
                else:
                    self._input_to_node[input_name] = [node]

    def _get_input_tensors_names(self) -> List:
        """ Gets list of names of starting nodes"""
        input_tensors_names = []
        for tensor in self.model.graph.input:
            input_tensors_names.append(tensor.name)

        assert input_tensors_names, "The model does not have any input tensors"

        return input_tensors_names

    @staticmethod
    def _create_ir_op(node: onnx_pb.NodeProto) -> Op:
        """
        Creates connected graphs internal representation Op
        :param node: ONNX proto node for which Op needs to be created
        """
        op = Op(name=node.name, dotted_name=node.name, output_shape=None, is_anonymous=False, op_type=node.op_type)
        # Add corresponding node to op
        op.model_module = ONNXModelModule(node)

        if op.type in ['Conv', 'ConvTranspose']:
            op.groups = get_op_groups(node)

        return op

    def _add_children_ops_to_op_queue(self, node: onnx_pb.NodeProto, op_queue: List):
        """
        Utility function for adding all children of op to self._op_queue
        :param node: node whose children will be added to op_queue
        :param op_queue: Queue for performing dfs
        """
        for output_tensor in node.output:
            if output_tensor in self._input_to_node:
                for child_node in self._input_to_node[output_tensor]:
                    op_queue.append((child_node, node, output_tensor))

    def _process_starting_ops(self, op_queue: List):
        """
        Processes input ops
        :param op_queue: Queue for performing dfs
        """
        input_tensors_names = self._get_input_tensors_names()
        for input_tensor_name in input_tensors_names:
            if input_tensor_name in self._input_to_node:
                for node in self._input_to_node[input_tensor_name]:
                    op = self._create_ir_op(node)
                    self._ops[node.name] = op
                    self._create_and_link_product_for_inputs(node.name, input_tensor_name)
                    self._add_children_ops_to_op_queue(node, op_queue)
                    self.starting_ops.append(op)

    def _create_and_link_product_for_inputs(self, consumer_node_name: str, input_tensor_name: str):
        """
        Create products between input and op consuming the input
        """

        assert input_tensor_name, "No inputs present in the model"

        if input_tensor_name + '_to_' + consumer_node_name in self._products:
            logger.debug("%s already exists", input_tensor_name + '_to_' + consumer_node_name)
        else:
            # TODO: figure out a way to add tensor shape. Adding the shape as None for now
            product = Product(input_tensor_name + '_to_' + consumer_node_name, None)
            # add product to self._products dictionary
            self._products[input_tensor_name + '_to_' + consumer_node_name] = product
            logger.debug("Created new product " + input_tensor_name + '_to_' + consumer_node_name)
            product.is_model_input = True

            consumer_op = self._ops[consumer_node_name]
            product.tensor_dict[consumer_op] = input_tensor_name

            # Link parent op, product, and current op
            # Fill in input, output, producer, consumer params as appropriate.
            consumer_op.add_input(product)
            product.add_consumer(consumer_op)

    def _create_op_if_not_exists(self, node: onnx_pb.NodeProto):
        """ Creates a CG op for a node"""
        if node.name not in self._ops:
            op = self._create_ir_op(node)
            self._ops[node.name] = op
            logger.debug("Created new op: %s ", node.name)
        else:
            logger.debug("Op %s already exists", node.name)

    def _create_and_link_product_if_not_exists(self, child_node: onnx_pb.NodeProto, parent_node: onnx_pb.NodeProto,
                                               connecting_tensor_name: str):
        """ Create and link new product if it does not yet exist """
        parent_module_name = parent_node.name
        child_node_name = child_node.name
        if parent_module_name + '_to_' + child_node_name in self._products:
            logger.debug("%s already exists", parent_module_name + '_to_' + child_node_name)
        else:
            # TODO: figure out a way to add tensor shape. Adding the shape as None for now
            product = Product(parent_module_name + '_to_' + child_node_name, None)
            # add product to self._products dictionary
            self._products[parent_module_name + '_to_' + child_node_name] = product
            logger.debug("Created new product " + parent_module_name + '_to_' + child_node_name)

            current_op = self._ops[child_node_name]
            parent_op = self._ops[parent_module_name]

            # find Tensor that corresponds to this product
            if connecting_tensor_name is None:
                logger.error("Could not find corresponding tensor between %s and %s",
                             parent_module_name,
                             child_node_name)
                assert False
            product.tensor_dict[current_op] = connecting_tensor_name

            # Link parent op, product, and current op
            # Fill in input, output, producer, consumer params as appropriate.
            current_op.add_input(product)
            product.producer = parent_op
            product.add_consumer(current_op)
            parent_op.output = product

    def _create_link_for_output_product(self, output_tensor_name: str, producer_node_name: str):
        """ Creates link between nodes and outputs of the model """
        product_name = producer_node_name + '_to_' + output_tensor_name
        if product_name in self._products:
            logger.debug("%s already exists", product_name)
        else:
            # TODO: figure out a way to add tensor shape. Adding the shape as None for now
            product = Product(product_name, None)
            # add product to self._products dictionary
            self._products[product_name] = product
            logger.debug("Created new product " + product_name)

            producer_op = self._ops[producer_node_name]
            product.tensor_dict[producer_node_name] = producer_op

            # Link producer op, product, and current tensor
            producer_op.output = product
            product.producer = producer_op

    def _create_output_products(self):
        """ Create products between last node and output """
        for output in self.model.graph.output:
            for node in self.model.graph.node:
                if output.name in node.output:
                    self._create_link_for_output_product(output.name, node.name)

    def fill_op_product_graph(self):
        """
        - DFS over the graph beginning with input op given as start_op_name
        - Creates op/product graph
        """
        visited_ops = set()
        op_queue = []
        self._process_starting_ops(op_queue)
        # op_queue is treated as a stack, containing operations to traverse. Elements are tuple
        # - Index 0 contains the node to visit.
        # - Index 1 contains the parent node.
        while op_queue:
            child_node, parent_node, connecting_tensor_name = op_queue.pop()
            # new module, create op/product and link to parent
            if child_node.name != parent_node.name:
                self._create_op_if_not_exists(child_node)
                self._create_and_link_product_if_not_exists(child_node, parent_node, connecting_tensor_name)

                # add children to op_queue if not visited
                if child_node.name not in visited_ops:
                    self._add_children_ops_to_op_queue(child_node, op_queue)
                    visited_ops.add(child_node.name)
                    logger.debug("visited op: %s", child_node.name)

        # Add output products
        self._create_output_products()

        # Add parameter products during postprocess
        logger.debug("finished initial pass, num_products is %s", len(self._products))
        self._create_param_products()

        # Identify places where branch Ops need to be inserted
        self._branch_ops_processing()

    def _branch_ops_processing(self):
        """ Identify places in the op/product graph where branch ops need to be inserted, and create them """

        # Dictionary that will map producers (ops) to products
        # Ops that map to multiple products will be ones to create branch ops for
        product_producer_dict = dict()
        for product in self._products.values():
            if product.producer is not None:
                if product.producer not in product_producer_dict:
                    product_producer_dict[product.producer] = [product]
                else:
                    product_producer_dict[product.producer].append(product)

        for op in product_producer_dict:
            if len(product_producer_dict[op]) > 1:
                # check if this has an input node
                is_model_input = False
                for product in product_producer_dict[op]:
                    if product.is_model_input:
                        is_model_input = True
                # Create branch op directly under op
                branch_op = self._create_branch_op()
                # Create product to link op and branch_op
                self._link_previous_op_to_branch_op(op, branch_op)
                # Create product to link branch op with multiple children modules
                self._link_branch_op_to_multiple_ops(branch_op, product_producer_dict[op], is_model_input)

    def _create_branch_op(self):
        """ Create a new branch op in self._ops """

        op = Op(name='branch_' + str(self._branch_count), output_shape=None,
                dotted_name='branch_' + str(self._branch_count),
                is_anonymous=True,
                op_type='branch')
        self._ops[op.name] = op
        self._branch_count += 1
        return op

    def _link_previous_op_to_branch_op(self, original_op: Op, branch_op: Op):
        """ Link the original op to the new branch op by creating a product """

        product = Product(original_op.name + '_to_' + branch_op.name, None)
        product.producer = original_op
        product.add_consumer(branch_op)
        original_op.output = product
        branch_op.add_input(product)
        self._products[product.name] = product

    def _link_branch_op_to_multiple_ops(self, branch_op: Op, product_list: list,
                                        is_model_input: bool = False):
        """ Create new product with multiple consumers, linking branch op with children ops"""

        branch_op_product = Product(branch_op.name + '_to_' + 'multiple_ops', branch_op.output_shape)
        branch_op_product.is_model_input = is_model_input
        branch_op_product.producer = branch_op
        branch_op.output = branch_op_product

        # For each product from original op to multiple children, we must:
        # 1: Add each child op as a consumer of the new branch_op_product
        # 2: Append the old product's corresponding Tensor to the new branch op product's tensor_list
        # 3: Add new branch_op_product as input for each child op
        # 4: Remove product from original op to child in child's inputs
        # 5: Remove product from self._products
        for product in product_list:
            assert len(product.consumers) == 1
            # item 1
            branch_op_product.add_consumer(product.consumers[0])
            # item 2
            assert len(product.tensor_dict.keys()) == 1
            branch_op_product.tensor_dict[product.consumers[0]] = product.tensor_dict[product.consumers[0]]
            # items 3 and 4
            # replace the old product with the new branch product, in the same index as the old product
            index = product.consumers[0].inputs.index(product)
            product.consumers[0].inputs[index] = branch_op_product
            # item 5
            del self._products[product.name]

        self._products[branch_op_product.name] = branch_op_product

    def _create_param_products(self):
        """ Create products for parameters of select modules """

        def create_and_connect_product(param_name: str, product_shape: List, my_op: Op,
                                       param_tensor: onnx_pb.TensorProto, product_type: Union[str, None]):
            """ Create product with given name, shape, and corresponding tensor.  Connect product to my_op. """

            product = Product(param_name, product_shape)
            product.is_parm = True
            product.add_consumer(my_op)
            product.tensor_dict[my_op] = param_tensor
            my_op.add_input(product)
            self._products[product.name] = product
            my_op.add_param(param_name, product, product_type)

        def create_conv2d_dense_type_params(my_op: Op):
            """ Create products for conv2d, dense, depthwise conv2d, and similar """
            op = my_op.get_module()

            weight_tensor = ParamUtils.get_param(self.model, op, WEIGHT_INDEX)
            create_and_connect_product(weight_tensor.name, weight_tensor.dims, my_op, weight_tensor, 'weight')

            bias_tensor = ParamUtils.get_param(self.model, op, BIAS_INDEX)
            if bias_tensor:
                create_and_connect_product(bias_tensor.name, bias_tensor.dims, my_op, bias_tensor, 'bias')

        def create_batchnorm_params(my_op: Op):
            """ Create products for fusedbatchnorm """
            op = my_op.get_module()

            gamma_tensor = ParamUtils.get_param(self.model, op, WEIGHT_INDEX)
            create_and_connect_product(gamma_tensor.name, gamma_tensor.dims, my_op, gamma_tensor, 'weight')

            beta_tensor = ParamUtils.get_param(self.model, op, BIAS_INDEX)
            create_and_connect_product(beta_tensor.name, beta_tensor.dims, my_op, beta_tensor, 'bias')

            moving_mean_tensor = ParamUtils.get_param(self.model, op, RUNNING_MEAN_INDEX)
            create_and_connect_product(moving_mean_tensor.name, moving_mean_tensor.dims, my_op,
                                       moving_mean_tensor, None)

            moving_variance_tensor = ParamUtils.get_param(self.model, op, RUNNING_VAR_INDEX)
            create_and_connect_product(moving_variance_tensor.name, moving_variance_tensor.dims, my_op,
                                       moving_variance_tensor, None)

        def handle_default(my_op: Op):
            """ Handler for other modules """
            logger.debug("Nothing to handle for op %s", my_op.name)

        switcher = {
            "Conv": create_conv2d_dense_type_params,
            "Gemm": create_conv2d_dense_type_params,
            "ConvTranspose": create_conv2d_dense_type_params,
            "BatchNormalization": create_batchnorm_params
        }

        for op in self._ops.values():
            handler = switcher.get(op.type, handle_default)
            handler(op)


def get_op_groups(node: onnx_pb.NodeProto):
    """Gets group information for Conv type node"""
    for attribute in node.attribute:
        if attribute.name == 'group':
            return attribute.i