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

import contextlib
from typing import List, Tuple, Union, Dict, Iterable
import numpy as np
import torch
import torch.nn
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d

import libpymo

from aimet_common.bias_correction import ConvBnPatternHandler
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.graph_searcher import GraphSearcher

# pylint: disable=unused-import
from aimet_torch.defs import PassThroughOp
from aimet_torch import utils
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.quantsim import QuantizationSimModel


LayerType = Union[
    torch.nn.Linear,
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.ConvTranspose2d,
]
_supported_layers = LayerType.__args__

BatchNormType = Union[BatchNorm1d, BatchNorm2d]
_supported_batchnorms = BatchNormType.__args__


def _delete_bn_from_model(model: torch.nn.Module, bn_layer_list: Iterable[BatchNormType]):
    utils.replace_modules_with_instances_of_new_type(model, bn_layer_list, torch.nn.Identity)


@contextlib.contextmanager
def _expand_shape_to_4d(weight_tensor: libpymo.TensorParams):
    """ Expand the shape of the weight into 4d.  """
    dims = len(weight_tensor.shape)

    if dims > 4:
        raise RuntimeError

    if dims == 4:
        yield weight_tensor

    else:
        orig_shape = weight_tensor.shape
        _4d_shape = np.append(orig_shape, [1 for _ in range(4-dims)]).astype(int)

        try:
            weight_tensor.shape = _4d_shape
            yield weight_tensor
        finally:
            weight_tensor.shape = orig_shape


def _call_mo_batch_norm_fold(weight: torch.Tensor,
                             bias: torch.Tensor,
                             bn: BatchNormType,
                             fold_backward: bool):
    """
    Calls C++ batch norm folding API.

    :param weight: Weight or scale tensor to fold BN into.
    :param bias: Bias tensor to fold BN into.
    :param bn: Batch Norm layer
    :param fold_backward: True if BatchNorm comes after Conv/Linear layer
    """
    with torch.no_grad():
        bn_params = libpymo.BNParams()
        bn_params.gamma = bn.weight.detach().numpy().reshape(-1)
        bn_params.beta = bn.bias.detach().numpy().reshape(-1)
        bn_params.runningMean = bn.running_mean.detach().numpy().reshape(-1)
        sigma = torch.sqrt(bn.running_var + bn.eps)
        bn_params.runningVar = sigma.detach().numpy().reshape(-1)

        weight_tensor = libpymo.TensorParams()

        weight_tensor.data = weight.detach().numpy().reshape(-1)
        weight_tensor.shape = np.array(weight.shape)

        bias_tensor = libpymo.TensorParams()

        bias_tensor.data = bias.detach().numpy().reshape(-1)
        bias_tensor.shape = np.array(bias.shape)
        is_bias_valid = True

        with _expand_shape_to_4d(weight_tensor):
            _bias = libpymo.fold(bn_params, weight_tensor, bias_tensor, is_bias_valid, fold_backward)

        bias.copy_(
            torch.tensor(_bias, device=bias.device, dtype=bias.dtype).reshape_as(bias)
        )

        weight.copy_(
            torch.tensor(weight_tensor.data,
                         device=weight.device,
                         dtype=weight.dtype).reshape_as(weight)
        )


def _fold_to_scale(conv_linear, bn):
    """
    Fold BatchNorm into the scale and bias of the given layer.

    :param conv_linear: Layer to fold BatchNorm into.
    :param bn: BatchNorm to be folded.
    """
    assert conv_linear.bias is not None
    raise NotImplementedError


def _fold_to_weight(conv_linear, bn, fold_backward: bool):
    """
    Fold BatchNorm into the weight and bias of the given layer.

    :param conv_linear: Layer to fold BatchNorm into.
    :param bn: BatchNorm to be folded.
    :param fold_backward: If True, perform backward folding.
                          Otherwise, perform forwawrd folding.
    """
    assert conv_linear.bias is not None

    # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
    # However depthwise conv layers are always N, 1, H, W whether transposed-conv or not, so no need to transpose
    if isinstance(conv_linear, torch.nn.ConvTranspose2d) and conv_linear.groups == 1:
        conv_linear.weight.data = conv_linear.weight.data.permute(1, 0, 2, 3)

    _call_mo_batch_norm_fold(conv_linear.weight, conv_linear.bias, bn, fold_backward=fold_backward)

    # Transpose weight back to N, C, H, W for transposed Conv2D, for non-depthwise layers
    if isinstance(conv_linear, torch.nn.ConvTranspose2d) and conv_linear.groups == 1:
        conv_linear.weight.data = conv_linear.weight.data.permute(1, 0, 2, 3)


def _fold(conv_linear, bn, fold_backward: bool, fold_to_scale: bool):
    """
    Fold BatchNorm into the given layer.

    :param conv_linear: Layer to fold BatchNorm into.
    :param bn: BatchNorm to be folded.
    :param fold_backward: If True, perform backward folding.
                          Otherwise, perform forwawrd folding.
    :param fold_to_scale: If True, fold BatchNorms to quantization scale parameter.
    :return: None
    """
    if not fold_backward and fold_to_scale:
        raise RuntimeError("Forward folding to scale is not possible.")

    assert isinstance(conv_linear, _supported_layers)

    if conv_linear.bias is None:
        out_channels = conv_linear.out_features if isinstance(conv_linear, torch.nn.Linear)\
                       else conv_linear.out_channels
        bias_data = torch.zeros(out_channels,
                                device=conv_linear.weight.device,
                                dtype=conv_linear.weight.dtype)
        conv_linear.bias = torch.nn.Parameter(bias_data)

    if fold_to_scale:
        _fold_to_scale(conv_linear, bn)
    else:
        _fold_to_weight(conv_linear, bn, fold_backward=fold_backward)


def fold_given_batch_norms(model, layer_pairs):
    """
    Fold a given set of batch_norm layers into conv layers

    :param model: Model
    :param layer_pairs: Pairs of conv and batch_norm layers to use for folding
    :return: None
    """
    conv_bn_pairs = []
    bn_conv_pairs = []
    for x, y in layer_pairs:
        if isinstance(x, _supported_batchnorms):
            assert isinstance(y, _supported_layers)
            bn = x
            conv = y
            bn_conv_pairs.append((bn, conv))
        else:
            assert isinstance(x, _supported_layers)
            assert isinstance(y, _supported_batchnorms)
            conv = x
            bn = y
            conv_bn_pairs.append((conv, bn))

    _fold_given_batch_norms(model, conv_bn_pairs, bn_conv_pairs)


def _fold_given_batch_norms(model,
                            conv_bn_pairs: List[Tuple[LayerType, BatchNormType]],
                            bn_conv_pairs: List[Tuple[BatchNormType, LayerType]],
                            fold_to_scale: bool = False):
    """
    Fold a given set of batch_norm layers into conv layers

    :param model: Model
    :param conv_bn_pairs: List of (conv, bn) pairs to fold
    :param bn_conv_pairs: List of (bn, conv) pairs to fold
    :param fold_to_scale: If True, fold BatchNorms to quantization scale parameter.
    :return: None
    """

    with utils.in_eval_mode(model), torch.no_grad():
        device = utils.get_device(model)

        try:
            # If model is not on CPU, convert it to CPU
            model.cpu()

            for conv, bn in conv_bn_pairs:
                _fold(conv, bn, fold_backward=True, fold_to_scale=fold_to_scale)

            for bn, conv in bn_conv_pairs:
                _fold(conv, bn, fold_backward=False, fold_to_scale=fold_to_scale)

            bn_modules = [bn for _, bn in conv_bn_pairs] + [bn for bn, _ in bn_conv_pairs]
            _delete_bn_from_model(model, bn_modules)

        finally:
            model.to(device)


def find_all_batch_norms_to_fold(model, input_shapes):
    """
    Find all possible batch norm layers that can be folded. And returns a list of pairs such that (bn, layer)
    means bn will be forward-folded into layer and (layer, bn) means bn will be backward-folded into layer
    :param model: Model to search
    :param input_shapes: Input shapes to use for the model (can be one or multiple inputs)
    :return: List of pairs of bn and layers to fold bn into
    """
    connected_graph = ConnectedGraph(model,
                                     utils.create_rand_tensors_given_shapes(input_shapes))
    conv_bn_pairs, bn_conv_pairs = _find_all_batch_norms_to_fold(model, input_shapes, connected_graph)
    return conv_bn_pairs + bn_conv_pairs


def _find_all_batch_norms_to_fold(
        model: torch.nn.Module,
        input_shapes: Union[Tuple, List[Tuple]],
        connected_graph: ConnectedGraph,
) -> Tuple[List[Tuple[LayerType, BatchNormType]],
           List[Tuple[BatchNormType, LayerType]]]:
    """
    Find all possible batch norm layers that can be folded. And returns a list of pairs such that (bn, layer)
    means bn will be forward-folded into layer and (layer, bn) means bn will be backward-folded into layer
    :param model: Model to search
    :param input_shapes: Input shapes to use for the model (can be one or multiple inputs)
    :return: A list of (layer, bn) pairs and a list of (bn, layer) pairs,
             where `bn` can be folded into to `layer`.
    """
    conv_linear_bn_activation_info_dict = _find_all_conv_bn_with_activation(connected_graph)

    # To mark BN's already picked for backward folding
    bn_picked_for_folding = set()

    ordered_conv_fc_nodes = utils.get_ordered_lists_of_conv_fc(model, input_shapes)

    conv_bn_pairs = []
    # Backward fold is given priority over Forward fold
    for _, module in ordered_conv_fc_nodes:
        if module in conv_linear_bn_activation_info_dict.keys():
            bn_info = conv_linear_bn_activation_info_dict[module]
            if bn_info.output_bn and bn_info.output_bn not in bn_picked_for_folding:
                conv_bn_pairs.append((module, bn_info.output_bn.get_module()))
                bn_picked_for_folding.add(bn_info.output_bn)

    bn_conv_pairs = []
    for _, module in ordered_conv_fc_nodes:
        if module in conv_linear_bn_activation_info_dict.keys():
            bn_info = conv_linear_bn_activation_info_dict[module]
            if bn_info.input_bn and bn_info.input_bn not in bn_picked_for_folding:
                bn_conv_pairs.append((bn_info.input_bn.get_module(), module))
                bn_picked_for_folding.add(bn_info.input_bn)

    return conv_bn_pairs, bn_conv_pairs


def _fold_all_batch_norms(model: torch.nn.Module,
                          input_shapes,
                          connected_graph: ConnectedGraph,
                          fold_to_scale: bool) -> List[Tuple[LayerType, BatchNormType]]:
    """
    Fold all batch_norm layers in a model.

    :param model: Model
    :param input_shapes: Input shapes for the model (can be one or multiple inputs)
    :param fold_to_scale: If True, fold BatchNorms to quantization scale parameter.
    :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
    """
    with utils.in_eval_mode(model), torch.no_grad():
        device = utils.get_device(model)

        try:
            # If model is not on CPU, convert it to CPU
            model.cpu()
            conv_bn_pairs, bn_conv_pairs = _find_all_batch_norms_to_fold(model, input_shapes, connected_graph)
            _fold_given_batch_norms(model, conv_bn_pairs, bn_conv_pairs, fold_to_scale=fold_to_scale)
            return conv_bn_pairs + [(conv, bn) for bn, conv in bn_conv_pairs]
        finally:
            model.to(device=device)


def fold_all_batch_norms_to_weight(
        model: torch.nn.Module,
        input_shapes: Union[Tuple, List[Tuple]],
) -> List[Tuple[LayerType, BatchNormType]]:
    """
    Fold all batch_norm layers in a model into the weight of the corresponding conv layers

    :param model: Model
    :param input_shapes: Input shapes for the model (can be one or multiple inputs)
    :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
    """
    inp_tensor_list = utils.create_rand_tensors_given_shapes(input_shapes)
    connected_graph = ConnectedGraph(model, inp_tensor_list)
    return _fold_all_batch_norms(model, input_shapes, connected_graph, fold_to_scale=False)


fold_all_batch_norms = fold_all_batch_norms_to_weight


def fold_all_batch_norms_to_scale(
        sim: QuantizationSimModel,
        input_shapes: Union[Tuple, List[Tuple]],
) -> List[Tuple[LayerType, BatchNormType]]:
    """
    Fold all batch_norm layers in a model into the quantization scale parameter
    of the corresponding conv layers

    :param model: Model
    :param input_shapes: Input shapes for the model (can be one or multiple inputs)
    :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
    """
    return _fold_all_batch_norms(sim.model, input_shapes, sim.connected_graph, fold_to_scale=True)


def find_all_conv_bn_with_activation(model: torch.nn.Module, input_shape: Tuple) -> Dict:
    """
    Uses searcher to find preceding and next bn layers for a conv/linear layer
    :param model: PyTorch model
    :param input_shape: shape of input to the model
    :return: dictionary of conv/linear layers with associated bn op / activation info
    """
    inp_tensor_list = utils.create_rand_tensors_given_shapes(input_shape)
    connected_graph = ConnectedGraph(model, inp_tensor_list)
    return _find_all_conv_bn_with_activation(connected_graph)


def _find_all_conv_bn_with_activation(connected_graph: ConnectedGraph) -> Dict:
    """
    Uses searcher to find preceding and next bn layers for a conv/linear layer
    :param connected_graph: ConnectedGraph object.
    :return: dictionary of conv/linear layers with associated bn op / activation info
    """

    # initialize all patterns to be matched and associated call back functions
    patterns_with_callbacks = []
    layer_select_handler = ConvBnPatternHandler()
    conv_types = ['Conv1d', 'Conv', 'ConvTranspose']
    linear_types = ['Gemm']

    for op_type in conv_types + linear_types:
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
