# /usr/bin/env python3.8
# -*- mode: python -*-
# =============================================================================
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
# =============================================================================
""" Utility functions for ONNX """
from typing import Dict, List

import onnx
from onnx import onnx_pb


OP_TYPES_WITH_PARAMS = ['Conv', 'Gemm', 'ConvTranspose', 'BatchNormalization']


def remove_nodes_with_type(node_type: str, onnx_graph: onnx.onnx_pb.GraphProto):
    """
    Remove specific type of nodes from graph

    :param node_type: string, type of node to be removed
    :param onnx_graph: onnx graph to modify

    """
    input_output_pairs = {}
    for node in onnx_graph.node:
        if node.op_type == node_type:
            input_output_pairs[node.output[0]] = node.input[0]
            onnx_graph.node.remove(node)
    for node in onnx_graph.node:
        if node.input[0] in input_output_pairs.keys():
            node.input[0] = input_output_pairs[node.input[0]]
        for outputs in onnx_graph.output:
            if outputs.name in input_output_pairs.keys() and \
                    node.output[0] == input_output_pairs[outputs.name]:
                node.output[0] = outputs.name


def replace_node_with_op(node_type: str, new_type: str, onnx_graph: onnx.onnx_pb.GraphProto):
    """
    Replace the given op type of nodes to new op type

    :param node_type: string, type of node to be replaced
    :param new_type: string, type of node to substitute for
    :param onnx_graph: onnx graph to modify

    """
    for node in onnx_graph.node:
        if node.op_type == node_type:
            node.op_type = new_type


def get_weights(name: str, onnx_graph: onnx.onnx_pb.GraphProto) -> bytes:
    """
    Return the weights by given name
    :param name, name of the weights to find
    :param onnx_graph, onnx graph to find the corresponding weight data
    :return onnx tensor
    """
    for param in onnx_graph.initializer:
        if param.name == name:
            return param.raw_data
    assert Exception("Couldn't find weights by the given name")
    return None


def get_ordered_dict_of_nodes(onnx_graph: onnx.onnx_pb.GraphProto) -> Dict:
    """
    Return the ordered list of nodes

    :param onnx_graph: onnx graph to provide node info
    :return dict of ordered nodes with name as key

    """
    ordered_dict = {}
    for node in onnx_graph.node:
        ordered_dict[node.name] = node
    return ordered_dict


class ParamUtils:
    """ Param utilities """
    @staticmethod
    def get_shape(model: onnx_pb.ModelProto, node: onnx_pb.NodeProto, param_index: int) -> List:
        """
        Returns a list of shape for the param specifies
        :param model: ONNX model
        :param node: ONNX node to which the param feeds to
        :param param_index: Index at which param feeds to the ONNX node
        """
        if node.op_type in OP_TYPES_WITH_PARAMS:
            if len(node.input) >= param_index + 1:
                param_name = node.input[param_index]
                for param in model.graph.initializer:
                    if param.name == param_name:
                        return param.dims
            assert "Param not present in the node"
        else:
            assert "Node type not in allowed op types with param list"
        return None

    @staticmethod
    def get_param(model: onnx_pb.ModelProto, node: onnx_pb.NodeProto, param_index: int) -> onnx_pb.TensorProto:
        """
        Returns the param tensor
        :param model: ONNX model
        :param node: ONNX node to which the param feeds to
        :param param_index: Index at which param feeds to the ONNX node
        """
        if node.op_type in OP_TYPES_WITH_PARAMS:
            if len(node.input) >= param_index + 1:
                param_name = node.input[param_index]
                for param in model.graph.initializer:
                    if param.name == param_name:
                        return param
            assert "Param not present in the node"
        else:
            assert "Node type not in allowed op types with param list"
        return None


def get_product_name_from_quantized_name(quantized_name: str):
    """
    Gets product's name from quantized name
    :param quantized_name: Quantized name
    """
    if '_updated' in quantized_name:
        return quantized_name[:quantized_name.index('_updated')]
    if '_qdq' in quantized_name:
        return quantized_name[:quantized_name.index('_qdq')]
    assert "Incorrect quantized name"
    return None
