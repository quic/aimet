# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Optimization code to fold batch-norm layers """

from typing import List, Tuple, Union, Dict
import numpy as np
import torch
import torch.nn

import libpymo

from aimet_common.bias_correction import ConvBnPatternHandler
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.graph_searcher import GraphSearcher

from aimet_torch.defs import PassThroughOp
from aimet_torch import utils
from aimet_torch.meta.connectedgraph import ConnectedGraph


def _delete_bn_from_model(model: torch.nn.Module, bn_layer_list: List[torch.nn.BatchNorm2d]):
    utils.replace_modules_with_instances_of_new_type(model, bn_layer_list, PassThroughOp)


LayerType = Union[torch.nn.Conv2d, torch.nn.Linear]
PairType = Union[Tuple[LayerType, torch.nn.BatchNorm2d],
                 Tuple[torch.nn.BatchNorm2d, LayerType]]


def call_mo_batch_norm_fold(conv_linear: Union[torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d],
                            bn: torch.nn.BatchNorm2d, is_batch_norm_second: bool) -> [torch.nn.Parameter,
                                                                                      torch.nn.Parameter]:
    """
    Calls Model optimization batch norm fold code and returns updated bias and weight

    :param conv_linear: Conv or Linear layer. For Conv layers Conv2D and TransposedConv2D are supported currently
    :param bn: Batch Norm layer
    :param is_batch_norm_second: True if BatchNorm comes after Conv/Linear layer
    :return: Updated bias and weight
    """
    bn_params = libpymo.BNParams()
    bn_params.gamma = bn.weight.detach().numpy().reshape(-1)
    bn_params.beta = bn.bias.detach().numpy().reshape(-1)
    bn_params.runningMean = bn.running_mean.detach().numpy().reshape(-1)
    sigma = torch.sqrt(bn.running_var + bn.eps)
    bn_params.runningVar = sigma.detach().numpy().reshape(-1)

    weight_tensor = libpymo.TensorParams()
    weight = conv_linear.weight

    # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
    # However depthwise conv layers are always N, 1, H, W whether transposed-conv or not, so no need to transpose
    if isinstance(conv_linear, torch.nn.ConvTranspose2d) and conv_linear.groups == 1:
        weight = weight.permute(1, 0, 2, 3)
    weight_tensor.data = weight.detach().numpy().reshape(-1)

    weight_shape = np.array(weight.shape)
    if len(conv_linear.weight.shape) == 2:
        weight_shape = np.append(weight_shape, [1, 1])

    weight_tensor.shape = weight_shape

    bias_tensor = libpymo.TensorParams()
    is_bias_valid = False
    if conv_linear.bias is not None:
        bias_tensor.data = conv_linear.bias.detach().numpy().reshape(-1)
        bias_tensor.shape = np.array(conv_linear.bias.shape)
        is_bias_valid = True

    bias = libpymo.fold(bn_params, weight_tensor, bias_tensor, is_bias_valid, is_batch_norm_second)
    return bias, weight_tensor


def fold_given_batch_norms(model, layer_pairs: List[PairType]):
    """
    Fold a given set of batch_norm layers into conv layers

    :param model: Model
    :param layer_pairs: Pairs of conv and batch_norm layers to use for folding
    :return: None
    """

    # Assuming that the entire model in on one device
    device = next(model.parameters()).device
    model.to('cpu')

    list_of_bn_layers = []
    for pair in layer_pairs:

        if isinstance(pair[0], (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            is_batch_norm_second = False
            bn = pair[0]
            conv_linear = pair[1]
        else:
            is_batch_norm_second = True
            bn = pair[1]
            conv_linear = pair[0]

        assert isinstance(conv_linear, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d))

        list_of_bn_layers.append(bn)

        bias, weight_tensor = call_mo_batch_norm_fold(conv_linear, bn, is_batch_norm_second)

        if isinstance(conv_linear, torch.nn.Linear):
            weight_tensor.shape = np.array([weight_tensor.shape[0], weight_tensor.shape[1]])

        conv_linear.bias = torch.nn.Parameter(torch.Tensor(bias))
        conv_linear.weight.data = torch.from_numpy(np.reshape(weight_tensor.data,
                                                              np.array(weight_tensor.shape)))

        conv_linear.weight.data = conv_linear.weight.data.type(torch.FloatTensor)

        # Transpose weight back to N, C, H, W for transposed Conv2D, for non-depthwise layers
        if isinstance(conv_linear, torch.nn.ConvTranspose2d) and conv_linear.groups == 1:
            conv_linear.weight.data = conv_linear.weight.data.permute(1, 0, 2, 3)

    _delete_bn_from_model(model, list_of_bn_layers)
    model.to(device)


def find_all_batch_norms_to_fold(model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]]) -> List[PairType]:
    """
    Find all possible batch norm layers that can be folded. And returns a list of pairs such that (bn, layer)
    means bn will be forward-folded into layer and (layer, bn) means bn will be backward-folded into layer
    :param model: Model to search
    :param input_shapes: Input shapes to use for the model (can be one or multiple inputs)
    :return: List of pairs of bn and layers to fold bn into
    """

    conv_linear_bn_activation_info_dict = find_all_conv_bn_with_activation(model, input_shapes)

    # To mark BN's already picked for backward folding
    bn_picked_for_folding = set()

    ordered_conv_fc_nodes = utils.get_ordered_lists_of_conv_fc(model, input_shapes)

    bn_conv_linear_pairs = []
    # Backward fold is given priority over Forward fold
    for _, module in ordered_conv_fc_nodes:
        if module in conv_linear_bn_activation_info_dict.keys():
            bn_info = conv_linear_bn_activation_info_dict[module]
            if bn_info.output_bn and bn_info.output_bn not in bn_picked_for_folding:
                bn_conv_linear_pairs.append((module, bn_info.output_bn.get_module()))
                bn_picked_for_folding.add(bn_info.output_bn)

    for _, module in ordered_conv_fc_nodes:
        if module in conv_linear_bn_activation_info_dict.keys():
            bn_info = conv_linear_bn_activation_info_dict[module]
            if bn_info.input_bn and bn_info.input_bn not in bn_picked_for_folding:
                bn_conv_linear_pairs.append((bn_info.input_bn.get_module(), module))
                bn_picked_for_folding.add(bn_info.input_bn)

    return bn_conv_linear_pairs


def fold_all_batch_norms(model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]]) -> \
        List[Tuple[torch.nn.Module, torch.nn.BatchNorm2d]]:
    """
    Fold all batch_norm layers in a model into corresponding conv layers

    :param model: Model
    :param input_shapes: Input shapes for the model (can be one or multiple inputs)
    :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
    """
    # Find whether model is on GPU
    device = utils.get_device(model)

    # If model is not on CPU, convert it to CPU
    model.cpu()

    bn_conv_linear_pairs = find_all_batch_norms_to_fold(model, input_shapes)

    fold_given_batch_norms(model, bn_conv_linear_pairs)

    # When returning the pairs, we want the second element of the pair to be the BN
    pairs_to_return = []
    for pair in bn_conv_linear_pairs:
        if isinstance(pair[0], torch.nn.BatchNorm2d):
            pairs_to_return.append((pair[1], pair[0]))
        else:
            pairs_to_return.append(pair)

    model.to(device=device)

    return pairs_to_return


def find_all_conv_bn_with_activation(model: torch.nn.Module, input_shape: Tuple) -> Dict:
    """
    Uses searcher to find preceding and next bn layers for a conv/linear layer
    :param model: PyTorch model
    :param input_shape: shape of input to the model
    :return: dictionary of conv/linear layers with associated bn op / activation info
    """

    # initialize all patterns to be matched and associated call back functions
    patterns_with_callbacks = []
    layer_select_handler = ConvBnPatternHandler()

    patterns_with_callbacks.append(PatternType(pattern=['batch_norm', 'convolution'],
                                               action=layer_select_handler))
    patterns_with_callbacks.append(PatternType(pattern=['convolution', 'batch_norm'],
                                               action=layer_select_handler))
    linear_types = ['addmm', 'matmul']
    for linear_type in linear_types:
        patterns_with_callbacks.append(PatternType(pattern=['batch_norm', linear_type],
                                                   action=layer_select_handler))
        patterns_with_callbacks.append(PatternType(pattern=[linear_type, 'batch_norm'],
                                                   action=layer_select_handler))

    inp_tensor_list = utils.create_rand_tensors_given_shapes(input_shape)
    connected_graph = ConnectedGraph(model, inp_tensor_list)

    # create graph searcher instance with connected graph and patterns to search
    graph_searcher = GraphSearcher(connected_graph, patterns_with_callbacks)

    # get all conv/linear and bn info
    graph_searcher.find_all_patterns_in_graph_apply_actions()
    convs_bn_activation_dict = layer_select_handler.get_conv_linear_bn_info_dict()

    return convs_bn_activation_dict
