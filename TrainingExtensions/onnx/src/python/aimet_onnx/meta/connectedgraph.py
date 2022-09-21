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


from typing import Dict, List
from onnx import onnx_pb
from onnxruntime.quantization.onnx_quantizer import ONNXModel

from aimet_common.connected_graph.connectedgraph import ConnectedGraph as AimetCommonConnectedGraph
from aimet_common.utils import AimetLogger
from aimet_common.connected_graph.operation import Op

from aimet_onnx.meta.product import Product

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)


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

        self._split_count = 0  # Use it in the name of split Ops getting added to the connected graph.

        # List of ops in the order they are traversed using the forward function
        self.ordered_ops = []

        self.starting_ops = []
        self.fill_op_product_graph()

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

    def _get_starting_nodes(self) -> Dict:
        """ Gets list of names of starting nodes"""
        input_nodes = {}
        for tensor in self.model.graph.input:
            input_nodes[tensor.name] = tensor

        assert input_nodes, "The model does not have any input tensors"

        return input_nodes

    @staticmethod
    def _create_ir_op(node: onnx_pb.NodeProto) -> Op:
        """
        Creates connected graphs internal representation Op
        :param node: ONNX proto node for which Op needs to be created
        """
        op = Op(name=node.name, dotted_name=node.name, output_shape=None, is_anonymous=False, op_type=node.op_type)
        return op

    def _add_children_ops_to_op_queue(self, node: onnx_pb.NodeProto, op_queue: List) -> int:
        """
        Utility function for adding all children of op to self._op_queue
        :param node: node whose children will be added to op_queue
        :param op_queue: Queue for performing dfs
        :return: Number of child ops added to the queue
        """
        num_ops_added = 0
        for output_tensor in node.output:
            if output_tensor in self._input_to_node:
                for child_node in self._input_to_node[output_tensor]:
                    op_queue.append((child_node, node, output_tensor))
                    num_ops_added += 1
        return num_ops_added

    def _process_starting_ops(self, op_queue: List):
        """
        Processes input ops
        :param op_queue: Queue for performing dfs
        """
        input_nodes = self._get_starting_nodes()
        for input_name in input_nodes:
            if input_name in self._input_to_node:
                for node in self._input_to_node[input_name]:
                    op = self._create_ir_op(node)
                    self._ops[node.name] = op
                    self._create_and_link_product_for_inputs(node.name, input_name)
                    self._add_children_ops_to_op_queue(node, op_queue)
                    self.starting_ops.append(op)

    def _create_and_link_product_for_inputs(self, node_name: str, input_name: str):
        """
        Create products between input and op consuming the input
        """

        assert input_name, "No inputs present in the model"

        if input_name + '_to_' + node_name in self._products:
            logger.debug("%s already exists", input_name + '_to_' + node_name)
        else:
            # TODO: figure out a way to add tensor shape. Adding the shape as None for now
            product = Product(input_name + '_to_' + node_name, None)
            # add product to self._products dictionary
            self._products[input_name + '_to_' + node_name] = product
            logger.debug("Created new product " + input_name + '_to_' + node_name)

            current_op = self._ops[node_name]
            product.tensor_dict[current_op] = input_name

            # Link parent op, product, and current op
            # Fill in input, output, producer, consumer params as appropriate.
            current_op.add_input(product)
            product.add_consumer(current_op)

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
