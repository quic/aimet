# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Utilities to load and save onnx models """

from typing import Union, List, Tuple, Dict, Set

import torch
import torch.nn as nn
import onnx

from aimet_common.utils import AimetLogger
import aimet_torch.utils
from aimet_torch.defs import PassThroughOp


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


# This is a dict that maps a PyTorch module type to the corresponding ONNX op type (as a string)
map_torch_types_to_onnx = {
    nn.Conv2d: ['Conv'],
    nn.BatchNorm2d: ['BatchNormalization'],
    nn.ReLU: ['Relu'],
    nn.ReLU6: ['Clip'],
    nn.MaxPool2d: ['MaxPool'],
    nn.Linear: ['Gemm'],
    nn.AdaptiveAvgPool2d: ['GlobalAveragePool', 'AveragePool'],
    nn.AvgPool2d: ['AveragePool'],
    nn.LogSoftmax: ['LogSoftmax']
}

torch_types_to_ignore = (nn.Dropout, nn.Dropout2d, PassThroughOp)

# List of associations between onnx types and pytorch connected graph types.
# Multiple onnx types may be associated with a pytorch connected graph type, and vice versa.
onnx_pytorch_conn_graph_type_pairs = [
    [["Conv"], ["convolution"]],
    [["BatchNormalization"], ["batch_norm"]],
    [["MaxPool"], ["max_pool2d"]],
    [["AveragePool"], ["avg_pool2d"]],
    [["Relu"], ["relu"]],
    [["Gemm"], ["addmm", "matmul"]],
    [["Add"], ["add"]]
]


class OnnxSaver:
    """
    Utilities to save/load onnx models
    """

    @classmethod
    def set_node_names(cls, onnx_model_path: str, pytorch_model: torch.nn.Module,
                       input_shape: Union[Tuple, List[Tuple]]):
        """
        This utility loads a given onnx model file and set the names of all the nodes (ops) to equivalent
        pytorch module names given the corresponding pytorch model.
        :param onnx_model_path: Path to the ONNX model file
        :param pytorch_model: Equivalent PyTorch model instance
        :param input_shape: Shape of the input to the model
                            (a tuple for 1 input and a list of tuples for multiple inputs)
        :return:
        """

        # Load the model
        onnx_model = onnx.load(onnx_model_path)

        # Parse the ONNX model and create mapping from input and output tensors to corresponding nodes
        map_output_tensor_to_node, map_input_tensor_to_node = cls.create_map_of_tensor_to_node(onnx_model)

        # Find the nodes of the ONNX model that are input nodes (no preceeding nodes)
        input_nodes = cls.find_model_input_nodes(onnx_model, map_output_tensor_to_node)

        for node in input_nodes:
            _logger.debug("Input Node: %r -> %r", node.op_type, node.output)

        # Find a ordinal ordering of the nodes - all nodes before node of index x preceed the node in the graph
        visited_nodes = set()
        ordered_list_of_nodes = []

        for node in input_nodes:
            cls.append_ordered_list_of_onnx_nodes(node, map_input_tensor_to_node, map_output_tensor_to_node,
                                                  ordered_list_of_nodes, visited_nodes)

        # Find corresponding pytorch nodes for every ONNX node
        # and set the name of the ONNX nodes to the names of the corresponding PyTorch modules
        cls.map_onnx_nodes_to_pytorch(pytorch_model, input_shape, ordered_list_of_nodes)

        # Save back the onnx model file
        onnx.save(onnx_model, onnx_model_path)

    @staticmethod
    def create_map_of_tensor_to_node(onnx_model: onnx.ModelProto) -> Tuple[Dict[str, List[onnx.NodeProto]],
                                                                           Dict[str, onnx.NodeProto]]:
        """
        Create and return two dicts
            1. Tensor -> list of nodes that consume this tensor
            2. Tensor -> node that produces this tensor
        :param onnx_model: ONNX model object
        :return: The two dicts described above

        Note: The list in #1 is ordered exactly in the order that pytorch trace reaches these nodes. This is important
        because later on we will use pytorch layer hooks to match these nodes with the equivalent PyTorch modules.
        The expectation is that PyTorch trace and PyTorch hooks follow the same execution sequence
        """
        map_input_tensor_to_node = {}
        map_output_tensor_to_node = {}
        for node in onnx_model.graph.node:
            for in_tensor in node.input:
                if in_tensor in map_input_tensor_to_node:
                    map_input_tensor_to_node[in_tensor].append(node)
                else:
                    map_input_tensor_to_node[in_tensor] = [node]

            for output in node.output:
                assert output not in map_output_tensor_to_node, 'More than one node produces the same tensor'
                map_output_tensor_to_node[output] = node

        return map_output_tensor_to_node, map_input_tensor_to_node

    @classmethod
    def append_ordered_list_of_onnx_nodes(cls, node: onnx.NodeProto,
                                          map_input_tensor_to_node: Dict[str, List[onnx.NodeProto]],
                                          map_output_tensor_to_node: Dict[str, onnx.NodeProto],
                                          running_list: List[onnx.NodeProto],
                                          visited_nodes: Set[int]):
        """
        Recursive function that returns an ordered list of nodes in a ONNX model
        :param node: Starting node to order from
        :param map_input_tensor_to_node: Dict of (tensor->nodes that consume this tensor)
        :param map_output_tensor_to_node: Dict of (tensor->node that produces this tensor)
        :param running_list: Running ordered list
        :param visited_nodes: Running set of visited nodes (to short-circuit the graph search)
        :return:
        """

        # Need to use id(node) since NodeProto is unfortunately unhashable
        if id(node) in visited_nodes:
            return
        visited_nodes.add(id(node))

        running_list.append(node)

        for output in node.output:
            if output in map_input_tensor_to_node:
                for downstream_node in map_input_tensor_to_node[output]:
                    # check if all preceeding nodes leading to the current node have been visited
                    # Else, we return. IOW, don't traverse downstream till all nodes leading to me have been reached
                    preceeding_nodes = cls.find_preceeding_nodes(downstream_node, map_output_tensor_to_node)
                    all_preceeding_visited = True
                    for pnode in preceeding_nodes:
                        if id(pnode) not in visited_nodes:
                            all_preceeding_visited = False

                    if all_preceeding_visited:
                        OnnxSaver.append_ordered_list_of_onnx_nodes(downstream_node, map_input_tensor_to_node,
                                                                    map_output_tensor_to_node, running_list,
                                                                    visited_nodes)

    @staticmethod
    def map_onnx_nodes_to_pytorch(torch_model: nn.Module, input_shape: Union[Tuple, List[Tuple]],
                                  onnx_ordered_list: List[onnx.NodeProto]):
        """
        Find the ONNX node that corresponds to each PyTorch module. And sets the name of the ONNX mode to that of the
        PyTorch module
        :param torch_model: PyTorch model instance
        :param input_shape: Shape of input(s) to the model
        :param onnx_ordered_list: An ordinally-ordered list of ONNX nodes
        :return:
        """
        torch_ordered_list = aimet_torch.utils.get_ordered_list_of_modules(torch_model, input_shape)

        torch_index = 0
        onnx_index = 0

        while torch_index < len(torch_ordered_list):
            name, module = torch_ordered_list[torch_index]

            if isinstance(module, torch_types_to_ignore):
                torch_index += 1
                continue

            if onnx_ordered_list[onnx_index].op_type in map_torch_types_to_onnx[type(module)]:
                _logger.debug('Found a match: %r -> %r', onnx_ordered_list[onnx_index].op_type, name)
                onnx_ordered_list[onnx_index].name = name
                torch_index += 1

            onnx_index += 1

    @classmethod
    def find_model_input_nodes(cls, onnx_model: onnx.ModelProto,
                               map_output_tensor_to_node: Dict[str, onnx.NodeProto]) -> List[onnx.NodeProto]:
        """
        Given a ONNX model find all the nodes that are inputs to the model - meaning nodes who have no
        preceeding nodes
        :param onnx_model: ONNX model instance
        :param map_output_tensor_to_node: Map of tensor->node that produces this tensor
        :return: List of input nodes
        """
        input_nodes = []
        for node in onnx_model.graph.node:
            preceeding_nodes = cls.find_preceeding_nodes(node, map_output_tensor_to_node)
            if not preceeding_nodes:
                input_nodes.append(node)

        return input_nodes

    @staticmethod
    def find_preceeding_nodes(node: onnx.NodeProto,
                              map_output_tensor_to_node: Dict[str, onnx.NodeProto]) -> List[onnx.NodeProto]:
        """
        Given an ONNX node, find the nodes that directly feed into this node
        :param node: ONNX node
        :param map_output_tensor_to_node: Map of tensor->node that produces this tensor
        :return:
        """

        nodes = []
        for in_tensor in node.input:
            if in_tensor in map_output_tensor_to_node:
                nodes.append(map_output_tensor_to_node[in_tensor])

        return nodes
