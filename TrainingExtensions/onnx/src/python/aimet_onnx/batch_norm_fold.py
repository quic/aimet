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
""" ONNX Code to fold batch-norm layers """

from typing import Dict, List, Tuple

import onnx
from onnx import helper


from aimet_common.bias_correction import ConvBnPatternHandler
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.graph_searcher import GraphSearcher
from aimet_common.connected_graph.connectedgraph_utils import get_ordered_ops

from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.operations import Op


ConvType = ['Conv', 'ConvTranspose']
LinearType = ['Gemm']
BatchNormType = ['BatchNormalization']


def _find_conv_bn_pairs(connected_graph: ConnectedGraph) -> Dict:
    """
    Uses searcher to find preceding and next bn layers for a conv/linear layer
    :param connected_graph: ConnectedGraph object.
    :return: dictionary of conv/linear Op with associated bn op / activation info
    """

    # initialize all patterns to be matched and associated call back functions
    patterns_with_callbacks = []
    layer_select_handler = ConvBnPatternHandler()
    preceding_linear_op_types = ['Flatten', 'Reshape']

    # Linear layer combinations
    for preceding_linear_op_type in preceding_linear_op_types:
        # BN -> Linear
        patterns_with_callbacks.append(PatternType(pattern=['BatchNormalization', preceding_linear_op_type, 'Gemm'],
                                                   action=layer_select_handler))

    for op_type in ConvType + LinearType:
        patterns_with_callbacks.append(PatternType(pattern=['BatchNormalization', op_type],
                                                   action=layer_select_handler))
        patterns_with_callbacks.append(PatternType(pattern=[op_type, 'BatchNormalization'],
                                                   action=layer_select_handler))

    # create graph searcher instance with connected graph and patterns to search
    graph_searcher = GraphSearcher(connected_graph, patterns_with_callbacks)

    # get all conv/linear and bn info
    graph_searcher.find_all_patterns_in_graph_apply_actions()
    convs_bn_activation_dict = layer_select_handler.get_conv_linear_bn_info_dict()

    return convs_bn_activation_dict


def find_all_batch_norms_to_fold(connected_graph: ConnectedGraph) -> Tuple[List[Tuple[Op, Op]],
                                                                           List[Tuple[Op, Op]]]:
    """
    Find all possible batch norm layers that can be folded. Returns a list of pairs such that (bn, layer)
    means bn will be forward-folded into layer and (layer, bn) means bn will be backward-folded into layer
    :param connected_graph: connected graph model to search
    :return: A list of (layer, bn) pairs and a list of (bn, layer) pairs,
             where `bn` can be folded into to `layer`.
    """
    conv_linear_bn_activation_info_dict = _find_conv_bn_pairs(connected_graph)

    # To mark BN's already picked for backward folding
    bn_picked_for_folding = set()

    ordered_conv_fc_nodes = get_ordered_conv_linears(connected_graph)
    # Layer input channels are needed to determine if conv is standard, depth-wise, or grouped
    infer_input_output_channels(connected_graph, ordered_conv_fc_nodes)
    conv_bn_pairs = []
    # Backward fold is given priority over Forward fold
    for node in ordered_conv_fc_nodes:
        # Filter out combinations that are not supported
        if node in conv_linear_bn_activation_info_dict.keys() and is_valid_bn_fold(node, True):
            bn_info = conv_linear_bn_activation_info_dict[node]
            if bn_info.output_bn and bn_info.output_bn not in bn_picked_for_folding:
                conv_bn_pairs.append((node, bn_info.output_bn))
                bn_picked_for_folding.add(bn_info.output_bn)

    bn_conv_pairs = []
    for node in ordered_conv_fc_nodes:
        # Filter out combinations that are not supported
        if node in conv_linear_bn_activation_info_dict.keys() and is_valid_bn_fold(node, False):
            bn_info = conv_linear_bn_activation_info_dict[node]
            if bn_info.input_bn and bn_info.input_bn not in bn_picked_for_folding:
                bn_conv_pairs.append((bn_info.input_bn, node))
                bn_picked_for_folding.add(bn_info.input_bn)

    return conv_bn_pairs, bn_conv_pairs


def get_ordered_conv_linears(conn_graph: ConnectedGraph) -> List[Op]:
    """
    helper to select a list of candidate layers for BatchNorm folding
    :param conn_graph: connected graph to search
    :return: List of conv/linear layers
    """
    # get ordered operations list from the connected graph
    list_of_ordered_ops = get_ordered_ops(conn_graph.starting_ops)

    # look for conv/linear layers
    ordered_convs = []
    for op in list_of_ordered_ops:
        if op.type in ConvType + LinearType:
            ordered_convs.append(op)
    return ordered_convs


def is_valid_bn_fold(conv: Op, fold_backward: bool) -> bool:
    """
    Determine if a given layer can successfully absorb a BatchNorm given the layer type and parameters
    :param conv: The Conv/Linear layer to fold a BatchNorm into.
    :param fold_backward: True if BatchNorm comes after Conv/Linear layer
    :return: True if a BatchNorm layer can be folded without causing output error.
    """
    # Convert the layer's attribute list to dictionary
    conv_attr_dict = {attr.name: helper.get_attribute_value(attr) for attr in conv.get_module().attribute}
    valid = True
    if not fold_backward:
        # Cannot fold BN -> Conv with padding. AIMET does not support forward folding to grouped or DW Conv
        if conv.type == 'Conv':
            print(conv_attr_dict["pads"])
            valid &= all(item == 0 for item in conv_attr_dict["pads"])
            valid &= conv_attr_dict["group"] == 1
        # AIMET does not support forward folding to ConvTranspose
        elif conv.type == 'ConvTranspose':
            valid = False
    else:
        # AIMET does not support backwards folding to grouped ConvTranspose
        if conv.type == 'ConvTranspose':
            valid &= (conv_attr_dict["group"] == 1 or conv_attr_dict["group"] == conv.num_in_channels)
    return valid


def infer_input_output_channels(model: ConnectedGraph, op_list: List[Op]):
    """
    Find the input and output channels of the layers specified in op_list and set the
    layer.num_in_channels and layer.num_out_channels for each layer accordingly
    :param model: The connected graph to which the layers in op_list belong
    :param op_list: List of the layers for which to find the input and output channels
    """
    # Find all intermediate activation shapes
    shape_info = onnx.shape_inference.infer_shapes(model.model).graph.value_info
    # Create dictionary of tensor.name : tensor.shape
    shape_dict = {item.name: item.type.tensor_type.shape.dim for item in shape_info}

    # Add the model's input and output shapes to the shape_dict
    for item in model.model.graph.input:
        inp_type = item.type
        if hasattr(inp_type, "tensor_type"):
            shape_dict[item.name] = inp_type.tensor_type.shape.dim
    for item in model.model.graph.output:
        inp_type = item.type
        if hasattr(inp_type, "tensor_type"):
            shape_dict[item.name] = inp_type.tensor_type.shape.dim

    for op in op_list:
        op_inputs = op.get_module().input
        op_outputs = op.get_module().output
        found_in_shape = False
        found_out_shape = False
        for item in op_inputs:
            if item in shape_dict.keys():
                found_in_shape = True
                # Take n_channels from shape (batches, n_channels, H, W)
                op.num_in_channels = shape_dict[item][1].dim_value

        for item in op_outputs:
            if item in shape_dict.keys():
                found_out_shape = True
                # Take n_channels from shape (batches, n_channels, H, W)
                op.num_out_channels = shape_dict[item][1].dim_value

        assert found_in_shape and found_out_shape
