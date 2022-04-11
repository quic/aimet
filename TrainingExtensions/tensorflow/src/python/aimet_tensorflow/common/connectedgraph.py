# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
# =============================================================================
""" Connected graph class and utilities """
from typing import List
import tensorflow as tf
from aimet_common.utils import AimetLogger
from aimet_common.model_module import TfModelModule
from aimet_common.connected_graph.connectedgraph import ConnectedGraph as aimetCommonConnectedGraph
from aimet_tensorflow.common.module_identifier import StructureModuleIdentifier
from aimet_tensorflow.common.sub_graph_matcher import ModuleIdentifierOpInfo
from aimet_tensorflow.common.operation import Op
from aimet_tensorflow.common.product import Product
from aimet_tensorflow.utils.common import get_valid_ops
from aimet_tensorflow.utils.op.conv import WeightTensorUtils, BiasUtils
from aimet_tensorflow.utils.op.fusedbatchnorm import BNUtils

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)


class ConnectedGraph(aimetCommonConnectedGraph):
    """ Connected Graph class """
    def __init__(self, graph: tf.Graph, starting_op_names: List[str], output_op_names: List[str]):
        """
        :param graph: Tensorflow graph to represent using connected graph
        :param starting_op_names: Ops to start building the connected graph from
        :param output_op_names: Ending ops of the graph.  Used to assist in identifying unnecessary ops like training
        ops.
        """
        super().__init__()
        self._graph = graph
        self._starting_op_names = starting_op_names
        self._branch_count = 0
        self._op_queue = []
        self._num_products_made = 0
        self.starting_ops = []

        # Create the connected graph
        self._valid_ops = get_valid_ops(self._graph, self._starting_op_names, output_op_names)
        self._module_identifier = StructureModuleIdentifier(self._graph, self._starting_op_names, self._valid_ops)
        self.fill_op_product_graph()

    @property
    def graph(self):
        """ Returns the graph that the connected graph is based off of """
        return self._graph

    @property
    def branch_count(self):
        """ Returns the number of branch ops in ops dict """
        return self._branch_count

    def get_op_from_module_name(self, name: str):
        """ Given the name of a tf operation, return the op in ops dict corresponding to the tf operation """

        tf_op = self._graph.get_operation_by_name(name)
        op_info = self._module_identifier.get_op_info(tf_op)
        return self._ops.get(op_info.module_name, None)

    def fill_op_product_graph(self):
        """
        - DFS over the graph beginning with input op given as start_op_name
        - Graph was pre-analyzed to pick out known module names
        - Creates op/product graph
        """

        visited_ops = set()
        self._process_starting_ops(self._starting_op_names)

        # op_queue is treated as a stack, containing operations to traverse.  Elements are tuples.
        # - Index 0 contains the tf operation to visit.
        # - Index 1 contains the parent operation.
        # - Index 2 contains the shape of the tensor whose output is the operation to visit.
        while self._op_queue:
            current_tf_op, parent_tf_op, product_shape = self._op_queue.pop()
            current_op_info = self._module_identifier.get_op_info(current_tf_op)
            parent_op_info = self._module_identifier.get_op_info(parent_tf_op)

            # new module, create op/product and link to parent
            if current_op_info.module_name != parent_op_info.module_name:
                self._create_op_if_not_exists(current_op_info)
                from_input = (parent_op_info.tf_op.name in self._starting_op_names)
                self._create_and_link_product_if_not_exists(current_op_info.module_name,
                                                            parent_op_info.module_name,
                                                            current_tf_op,
                                                            parent_tf_op,
                                                            product_shape,
                                                            from_input)

                # switching modules means marking previous op as output of previous module,
                # and current op as input of new current module
                self._ops[parent_op_info.module_name].output_op_node = parent_tf_op

            # add children to op_queue if not visited
            if current_tf_op.name not in visited_ops:
                num_children = self._add_children_ops_to_op_queue(current_tf_op)
                if not num_children:
                    # No children added to the queue, mark this op as the output op of the current connected graph Op
                    self._ops[current_op_info.module_name].output_op_node = current_tf_op
                visited_ops.add(current_tf_op.name)
                logger.debug("visited op: %s", current_tf_op.name)

        # Add parameter products during postprocess
        logger.debug("finished initial pass, num_products is %s", len(self._products))
        for op in self._ops.values():
            self._create_param_products(op, self._products)
            _fill_flatten_shape_if_needed(op)
            _reorder_multi_input_op_products(op)

        # Identify places where branch Ops need to be inserted
        self._branch_ops_processing()

    def _add_children_ops_to_op_queue(self, op: tf.Operation) -> int:
        """
        Utility function for adding all children of op to self._op_queue
        :param op: Op whose children will be added to op_queue
        :return: Number of child ops added to the queue
        """
        num_ops_added = 0
        for output_tensor in op.outputs:
            for output_op in output_tensor.consumers():
                if output_op in self._valid_ops:
                    self._op_queue.append((output_op, op, output_tensor.shape))
                    num_ops_added += 1
        return num_ops_added

    def _create_op_if_not_exists(self, current_op_info: ModuleIdentifierOpInfo):
        """ Create new op if it does not yet exist """

        if current_op_info.module_name not in self._ops:
            op = Op(name=current_op_info.module_name,
                    dotted_name=current_op_info.module_name,
                    output_shape=None,
                    is_anonymous=False,
                    op_type=current_op_info.op_type,
                    pattern_type=current_op_info.pattern_type,
                    internal_ops=current_op_info.internal_ops)
            fill_op_info(op, current_op_info)
            self._ops[current_op_info.module_name] = op
            logger.debug("Created new op: %s ", current_op_info.module_name)
        else:
            logger.debug("Op %s already exists", current_op_info.module_name)

    def _create_and_link_product_if_not_exists(self, current_module_name: str, parent_module_name: str,
                                               current_tf_op: tf.Operation, parent_tf_op: tf.Operation,
                                               product_shape: tf.TensorShape, from_input: bool):
        """ Create and link new product if it does not yet exist """

        if parent_module_name + '_to_' + current_module_name in self._products:
            logger.debug("%s already exists", parent_module_name + '_to_' + current_module_name)
        else:
            product = Product(parent_module_name + '_to_' + current_module_name, product_shape)
            product.is_model_input = from_input

            # add product to self._products dictionary
            self._products[parent_module_name + '_to_' + current_module_name] = product
            logger.debug("Created new product " + parent_module_name + '_to_' + current_module_name)

            current_op = self._ops[current_module_name]
            parent_op = self._ops[parent_module_name]

            # find tf.Tensor that corresponds to this product
            connecting_tensor, _, _ = get_connecting_tensor_and_indices(parent_tf_op, current_tf_op)
            if connecting_tensor is None:
                logger.error("Could not find corresponding tensor between %s and %s",
                             parent_tf_op.name,
                             current_tf_op.name)
                assert False
            product.tensor_dict[current_op] = connecting_tensor

            # Link parent op, product, and current op
            # Fill in input, output, producer, consumer params as appropriate.
            current_op.add_input(product)
            product.producer = parent_op
            product.add_consumer(current_op)
            parent_op.output = product
            parent_op.output_shape = product_shape

    def _process_starting_ops(self, starting_op_names):
        """ Given name of the starting op, create the op in self._ops and add its children to the queue """

        for start_op_name in starting_op_names:
            starting_op = self._graph.get_operation_by_name(start_op_name)
            if starting_op in self._valid_ops:
                starting_op_info = self._module_identifier.get_op_info(starting_op)
                op = Op(name=starting_op_info.module_name,
                        dotted_name=starting_op_info.module_name,
                        output_shape=None,
                        is_anonymous=False,
                        op_type=starting_op_info.op_type,
                        pattern_type=starting_op_info.pattern_type,
                        internal_ops=starting_op_info.internal_ops)
                fill_op_info(op, starting_op_info)
                self._ops[starting_op_info.module_name] = op
                self._add_children_ops_to_op_queue(starting_op)
                self.starting_ops.append(op)

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
                branch_op = self._create_branch_op(op.output_shape)
                # Create product to link op and branch_op
                self._link_previous_op_to_branch_op(op, branch_op, op.output_shape)
                # Create product to link branch op with multiple children modules
                self._link_branch_op_to_multiple_ops(branch_op, product_producer_dict[op], is_model_input)

    def _create_branch_op(self, output_shape: tf.TensorShape):
        """ Create a new branch op in self._ops """

        op = Op(name='branch_' + str(self._branch_count),
                dotted_name='branch_' + str(self._branch_count),
                output_shape=output_shape,
                is_anonymous=True,
                op_type='branch',
                pattern_type=None,
                internal_ops=None)
        self._ops[op.name] = op
        self._branch_count += 1
        return op

    def _link_previous_op_to_branch_op(self, original_op: Op, branch_op: Op,
                                       output_shape: tf.TensorShape):
        """ Link the original op to the new branch op by creating a product """

        product = Product(original_op.name + '_to_' + branch_op.name, output_shape)
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
        # 2: Append the old product's corresponding tf.Tensor to the new branch op product's tensor_list
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

    def _create_param_products(self, op: Op, products_dict: dict):
        """ Create products for parameters of select modules """

        def create_and_connect_product(param_name: str, product_shape: tf.TensorShape, my_op: Op,
                                       param_tensor: tf.Tensor):
            """ Create product with given name, shape, and corresponding tensor.  Connect product to my_op. """

            product = Product(my_op.name + '/' + param_name, product_shape)
            product.is_parm = True
            product.add_consumer(my_op)
            product.tensor_dict[my_op] = param_tensor
            my_op.add_input(product)
            products_dict[product.name] = product
            my_op.add_param(param_name, product)
            self._num_products_made += 1

        def create_conv2d_dense_type_params(my_op: Op):
            """ Create products for conv2d, dense, depthwise conv2d, and similar """
            tf_op = my_op.get_module()

            weight_op = WeightTensorUtils.get_read_op(tf_op)
            create_and_connect_product('kernel', weight_op.outputs[0].shape, my_op, weight_op.outputs[0])

            if not BiasUtils.is_bias_none(tf_op):
                bias_op = BiasUtils.get_bias_read_op(tf_op)
                create_and_connect_product('bias', bias_op.outputs[0].shape, my_op, bias_op.outputs[0])

        def create_batchnorm_params(my_op: Op):
            """ Create products for fusedbatchnorm """
            tf_op = my_op.get_module()

            beta_tensor = BNUtils.get_beta_read_var_op_tensor(self._graph, tf_op)
            create_and_connect_product('beta', beta_tensor.shape, my_op, beta_tensor)

            gamma_tensor = BNUtils.get_gamma_read_var_op_tensor(self._graph, tf_op)
            create_and_connect_product('gamma', gamma_tensor.shape, my_op, gamma_tensor)

            moving_mean_tensor = BNUtils.get_moving_mean_read_var_op_tensor(self._graph, tf_op)
            create_and_connect_product('moving_mean', moving_mean_tensor.shape, my_op,
                                       moving_mean_tensor)

            moving_variance_tensor = BNUtils.get_moving_variance_read_var_op_tensor(self._graph, tf_op)
            create_and_connect_product('moving_variance', moving_variance_tensor.shape, my_op,
                                       moving_variance_tensor)

        def create_layernorm_params(layernorm_op: Op):
            """
            Create products for layernorm
            :param layernorm_op: Connected Graph operation to find and insert parameter operations
            """

            tf_op = layernorm_op.get_module()

            beta_tensor = BNUtils.get_beta_read_var_op_tensor_using_structure(tf_op)
            create_and_connect_product('beta', beta_tensor.shape, layernorm_op, beta_tensor)

            gamma_tensor = BNUtils.get_gamma_read_var_op_tensor_using_structure(tf_op)
            create_and_connect_product('gamma', gamma_tensor.shape, layernorm_op, gamma_tensor)

        def handle_default(my_op: Op):
            """ Handler for other modules """
            logger.debug("Nothing to handle for op %s", my_op.name)

        switcher = {
            "Conv2D": create_conv2d_dense_type_params,
            "Dense": create_conv2d_dense_type_params,
            "DepthwiseConv2dNative": create_conv2d_dense_type_params,
            "BatchNorm": create_batchnorm_params,
            "FusedBatchNormV3": create_batchnorm_params,
            "LayerNorm": create_layernorm_params,
            "Conv2DTranspose": create_conv2d_dense_type_params
        }

        handler = switcher.get(op.type, handle_default)
        handler(op)


def _fill_flatten_shape_if_needed(op: Op):
    """
    Tensorflow flatten doesn't know its output size.  This poses a problem for Mask Propagator, so we try
    to deduce the size ourselves by looking at the input and multiplying all dimensions together.
    To ensure this is only done on flatten and not another reshape, check if the last dimension is unknown.
    """

    # If flatten op is last, it will have no output and thus no output shape
    if op.type == "Flatten" and op.output_shape:
        dims = op.output_shape.as_list()
        if dims:
            if dims[-1] is None:
                output_size = 1
                input_shape = op.inputs[0].shape.as_list()
                for dim in input_shape:
                    if dim is not None:
                        output_size *= dim
                new_output_shape = tf.TensorShape([tf.compat.v1.Dimension(None), tf.compat.v1.Dimension(output_size)])
                op.output_shape = new_output_shape
                op.output.shape = new_output_shape


def _reorder_multi_input_op_products(op: Op):
    """
    Ops with multiple input products need to have the input products arranged in the same order as in the tf graph,
    so mask propagation will work correctly.
    """

    if op.type in ['Add', 'AddN', 'ConcatV2', 'Merge', 'Mul']:
        # Create new product list with the same length as the old input products list
        # When looking at inputs to the op in tf graph, there may be inputs that were not seen during DFS.
        # Currently, handle this by first creating a product list with the length of the op inputs, filling in entries
        # corresponding to products which we have seen, then remove all indices that are not filled in.
        old_products = op.get_input_products()
        tf_op_input_list = list(op.get_module().inputs)

        new_product_list = [None] * len(tf_op_input_list)
        for product in old_products:
            index = tf_op_input_list.index(product.tensor_dict[op])
            new_product_list[index] = product
        op.inputs = [i for i in new_product_list if i]


def fill_op_info(op: Op, tf_op_info: ModuleIdentifierOpInfo):
    """ Fill in op information """
    if op.type == 'Conv2D':
        op.groups = 1
    elif op.type == 'DepthwiseConv2dNative':
        # Set number of groups to be the number of input channels
        # Format is expected to be NHWC, so channels is the last index in shape array
        # This is not actually guaranteed to be correct, need to figure a way to check for sure
        op.groups = tf_op_info.tf_op.inputs[0].shape.as_list()[-1]

    op.model_module = TfModelModule(tf_op_info.tf_op)
    for attribute_name, attribute in tf_op_info.get_attributes().items():
        op.add_attribute(attribute_name, attribute)


def get_connecting_tensor_and_indices(parent_op: tf.Operation, child_op: tf.Operation):
    """
    If a connecting tensor between parent_op and child_op is found, returns the following:
    - tf.Tensor connecting parent_op to child_op
    - output index of the tf.Tensor in parent_op.outputs
    - input index of the tf.Tensor in child_op.inputs
    If no tensor is found linking the two ops, returns None
    """
    for i in range(len(parent_op.outputs)):
        for consumer in parent_op.outputs[i].consumers():
            if consumer == child_op:
                connecting_tensor = parent_op.outputs[i]
                for j in range(len(child_op.inputs)):
                    if child_op.inputs[j] == connecting_tensor:
                        return connecting_tensor, i, j
                logger.error("No inputs match the connecting tensor")
                assert False

    return None, None, None
