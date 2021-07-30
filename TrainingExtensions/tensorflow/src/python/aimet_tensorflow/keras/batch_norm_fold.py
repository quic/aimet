# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Utility for batch norm fold in tf 2.x """

from typing import Tuple, Union, List
from collections import defaultdict
import  numpy as np
import tensorflow as tf
import libpymo
from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.utils import common
from aimet_tensorflow.keras.utils.op.batchnorm import BNUtils

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

LayerType = Union[tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.Conv2DTranspose,
                  tf.keras.layers.DepthwiseConv2D]

PairType = Union[Tuple[LayerType, tf.keras.layers.BatchNormalization, bool],
                 Tuple[tf.keras.layers.BatchNormalization, LayerType, bool]]

bn_type = tf.keras.layers.BatchNormalization
linear_type = tf.keras.layers.Dense
conv_type = tf.keras.layers.Conv2D
flatten_type = Union[tf.keras.layers.Flatten, tf.keras.layers.Reshape]


def check_layer_to_find_pattern(cur_layer: tf.keras.layers.Layer, conv_linear_with_bn_dict: dict, layer_out_node_ref: dict, has_seen: list):
    """
    find all paths in the model considering all inputs
    :param cur_layer: layer to investigate for finding a pattern
    :param conv_linear_with_bn_dict: dictionary to store possible conv_bn pairs, key: Dense or Conv layer
             & Value: list of BNS; first index in this list shows bn_in and the second index shows bn_out
    :param layer_out_node_ref: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param has_seen: for storing the layer which is useful for finding pattern in the next layers; index 0 is for conv op,
        index 2 is for bn op and index 3 is for storing flatten/reshape op
    """

    # pylint: disable=too-many-branches
    if isinstance(cur_layer, conv_type):
        if has_seen[1] is not None:
            conv_linear_with_bn_dict[cur_layer] = [has_seen[1], None]
            has_seen[1] = None
        if (cur_layer.activation is tf.keras.activations.linear) and (cur_layer in layer_out_node_ref) and len(layer_out_node_ref[cur_layer]) == 1:
            has_seen[0] = cur_layer
    elif isinstance(cur_layer, bn_type):
        if has_seen[0] is not None:
            if has_seen[0] in conv_linear_with_bn_dict:
                conv_linear_with_bn_dict[has_seen[0]][1] = cur_layer
            else:
                conv_linear_with_bn_dict[has_seen[0]] = [None, cur_layer]
            has_seen[0] = None
        if (cur_layer in layer_out_node_ref) and len(layer_out_node_ref[cur_layer]) == 1:
            has_seen[1] = cur_layer
    elif isinstance(cur_layer, (tf.keras.layers.Flatten, tf.keras.layers.Reshape)):
        if (cur_layer in layer_out_node_ref) and len(layer_out_node_ref[cur_layer]) == 1:
            if has_seen[1]:
                has_seen[2] = cur_layer
            else:
                has_seen[1] = None
        if has_seen[0]:
            has_seen[0] = None
    elif isinstance(cur_layer, linear_type):
        if has_seen[1] is not None and has_seen[2] is not None:
            conv_linear_with_bn_dict[cur_layer] = [has_seen[1], None]
        has_seen[2] = None
        has_seen[1] = None
        # visited[0] = None
    else:
        has_seen[0] = None
        has_seen[1] = None
        has_seen[2] = None

def find_all_batch_norms_to_fold(model: (tf.keras.Model, tf.keras.layers.Layer)) -> List[PairType]:
    """
    uses searcher to choose layers for bias correction
    :param model: model to obtain conv_linear pairs for
    :return: List of conv/linear layers with associated bn op / activation info
    """
    node_layer_map = common.create_node_to_layer_map(model)
    layer_out_node_map = common.create_layer_to_out_node_map(model)

    convs_linears_bn_activation_info_dict = find_possible_convs_linears_bn(node_layer_map, layer_out_node_map)

    # get all ordered convs/ linears layers
    ordered_conv_linears = get_ordered_conv_linears(node_layer_map, layer_out_node_map)

    # get the in out tensor for bns found, we need this on TF to remove the bns after fold.
    bn_conv_linear_pairs = []

    # track BNs added for fold
    marked_bn_set = set()

    for conv_linear_layer in ordered_conv_linears:
        if conv_linear_layer in convs_linears_bn_activation_info_dict.keys():
            bn_info = convs_linears_bn_activation_info_dict[conv_linear_layer]
            if bn_info[1]:
                if bn_info[1] not in marked_bn_set:
                    bn_conv_linear_pairs.append((conv_linear_layer, bn_info[1], True))
                    marked_bn_set.add(bn_info.output_bn)
            elif bn_info[0]:
                if bn_info[0] not in marked_bn_set:
                    bn_conv_linear_pairs.append((conv_linear_layer, bn_info[0], False))
                    marked_bn_set.add(bn_info.input_bn)

    return bn_conv_linear_pairs

def add_children_layer_before_parent_layer(cur_layer: tf.keras.layers.Layer, node_layer_map: dict, layer_out_node_map:dict, visited_layers: set, reversed_ordered_layers: list) :
    """
    Function to do topological sorting for finding all the layers  which are accessible from the specific input_layer in order of occurrence
    :param cur_layer:
    :param node_layer_map:
    :param layer_out_node_map:
    :param visited_layers: Set of all layers that have been visited
    :param ordered_layers: List of all layers in the opposite of order of occurrence for the layers that we have visited so far
    :return:
    """

    # Mark the current layer as visited.
    visited_layers.add(cur_layer)

    if cur_layer in layer_out_node_map:
        # Recur for all the layers adjacent to this layer
        for next_node in layer_out_node_map[cur_layer]:
            next_layer = node_layer_map[next_node][1]
            if next_layer not in visited_layers:
                add_children_layer_before_parent_layer(next_layer, node_layer_map, layer_out_node_map, visited_layers, reversed_ordered_layers)
            reversed_ordered_layers.append(cur_layer)
    else:
        reversed_ordered_layers.append(cur_layer)


def get_ordered_layers(node_layer_map: dict, layer_out_node_map: dict) -> List[tf.keras.layers.Layer]:
    """
    Function to return the list with all the layers in which layers come before parent layer
    :param node_layer_map:
    :param layer_out_node_map:
    :return: ordered_layers: List of all layers in order of occurrence
    """
    # to find the input layers of the model
    input_layers = common.find_input_layers(node_layer_map)

    #  Set of all layers that have been visited (to cut short duplicate traversals)
    visited_layers = set()

    # List of all layers in the opposite of order of occurrence
    reversed_ordered_layers = []

    for input_layer in input_layers:
        add_children_layer_before_parent_layer(input_layer, node_layer_map, layer_out_node_map, visited_layers, reversed_ordered_layers)

    # reverse the list because layers are in reverse order
    ordered_layers = reversed_ordered_layers[::-1]

    # # filter ordered ops for only valid ops
    # ordered_ops = [op for op in ordered_ops if op in valid_ops]

    return ordered_layers

def get_ordered_conv_linears(node_layer_map: dict, layer_out_node_map: dict) -> List:
    """
    helper to select a list of candidate layers for bias correction
    :param node_layer_map:
    :param layer_out_node_map:
    :return: List of conv/linear layer refs
    """
    # get ordered layers list in node_layer map dictionary
    list_of_ordered_layers = get_ordered_layers(node_layer_map, layer_out_node_map)

    # look for conv layers
    ordered_conv_linears = []
    for layer in list_of_ordered_layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            ordered_conv_linears.append(layer)
    return ordered_conv_linears

def find_paths_specific_input(cur_layer: tf.keras.layers.Layer, node_layer_ref: dict, layer_out_node_ref: dict,
                              has_seen: list, visited_layer: dict, conv_linear_with_bn_dict: dict) -> dict:
    """
    find all paths in the model for the specific input
    :param cur_layer: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param node_layer_ref: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param layer_out_node_ref: dictionary includes layer_ref as a key, outbound nodes as value
    :paramm has_seen: for storing the layer which is useful for finding pattern in the next layers; index 0 is for conv op,
        index 2 is for bn op and index 3 is for storing flatten/reshape op
    :param visited_layer: to store all the layers that have been visited so far in the dictionary
    :param conv_linear_with_bn_dict:
    :return: dictionary includes all possible conv-bn pairs for the path starting with the specific input
    """

    # Mark the current layer as visited to prevent passing from one layer more than once
    visited_layer[cur_layer] = True

    check_layer_to_find_pattern(cur_layer, conv_linear_with_bn_dict, layer_out_node_ref, has_seen)

    if cur_layer in layer_out_node_ref:
        for next_node in layer_out_node_ref[cur_layer]:
            next_layer = node_layer_ref[next_node][1]
            if next_layer not in visited_layer:
                find_paths_specific_input(next_layer, node_layer_ref, layer_out_node_ref, has_seen, visited_layer, conv_linear_with_bn_dict)
            else:
                has_seen[0] = None
                has_seen[1] = None
                has_seen[2] = None

    return conv_linear_with_bn_dict

def find_possible_convs_linears_bn(node_layer_map: dict, layer_out_node_map: dict) -> dict:
    """
    find all possible convs_linears_bn_activations by traversing all paths in the model considering all inputs
    :param node_layer_map:  dictionary includes node_ref as a key, in_layers and out_layer as value
    :param layer_out_node_map: dictionary includes layer_ref as a key, outbound nodes as value
    :return: return dictionary of all possible conv_bn pairs, key: Dense or Conv layer
             & Value: list of BNS; first index in this list shows bn_in and the second index shows bn_out
    """

    input_layers = common.find_input_layers(node_layer_map)
    visited_layer = defaultdict()
    conv_linear_with_bn_dict = dict()

    for input_layer in input_layers:
        conv_linear_with_bn_dict = find_paths_specific_input(input_layer, node_layer_map, layer_out_node_map, [None, None, None], visited_layer, conv_linear_with_bn_dict)

    return conv_linear_with_bn_dict

def _get_bn_params(bn: tf.keras.layers.BatchNormalization) -> libpymo.BNParams():
    """
    helper to populate BN params from given BN Layer, required for fold
    :param bn: BatchNorm Layer
    :return: return bn params in libpymo.TensorParams() format.
    """

    bn_params = libpymo.BNParams()

    bn_params.gamma = bn.gamma.numpy().reshape(-1)
    bn_params.beta = bn.beta.numpy().reshape(-1)
    bn_params.runningMean = bn.moving_mean.numpy().reshape(-1)
    bn_params.runningVar = bn.moving_variance.numpy().reshape(-1)
    epsilon = bn.epsilon
    var = bn.moving_variance.numpy()
    var_with_epsilon = var + epsilon
    sigma = np.sqrt(var_with_epsilon)
    bn_params.runningVar = sigma

    return bn_params

def _get_bias_tensor(conv_linear: LayerType) -> libpymo.TensorParams():
    """
    Get bias tensor in given conv layer.
    Packs bias in the format required for BN fold
    (libpymo.TensorParams()).
    :param conv: conv Layer
    :return: return bias param in libpymo.TensorParams() format.
    """
    bias_tensor = libpymo.TensorParams()
    if conv_linear.bias is not None:
        bias_tensor.data = conv_linear.bias.numpy().reshape(-1)
        bias_tensor.shape = np.array(conv_linear.bias.shape)

    return bias_tensor

def _get_weight_tensor_transpose_reshape(conv_linear: LayerType) -> libpymo.TensorParams():
    """
    Get weight tensor from conv layer
    Converts to right format - performs transpose and reshape.
    Packs it to the format required for BN fold (libpymo.TensorParams()).
    :param conv: conv layer
    :return: return weight tensor in libpymo.TensorParams() format.
    """

    # Weight tensor libpymo format
    weight_tensor = libpymo.TensorParams()

    # linear array to be sent for bn fold
    weight = conv_linear.kernel.numpy()
    shape = conv_linear.kernel.shape


    if isinstance(conv_linear, tf.keras.layers.DepthwiseConv2D):
        # Depthwise conv layers in TF have outputs(Noc) set to 1.
        # we will use format [Nic, Noc, kh, kw] -
        # to be compatible with cpp backend.
        weight = np.transpose(weight, (2, 3, 0, 1))
        # [Nic, Noc, kh, kw]
        shape = np.array([shape[2], shape[3], shape[0], shape[1]])

    elif isinstance(conv_linear, tf.keras.layers.Dense):
        shape = np.concatenate((np.array([1, 1]), shape))
        weight = np.transpose(weight, (1, 0))
        # [Noc, Nic, kh, kw]
        shape = np.array([shape[3], shape[2], shape[0], shape[1]])
    elif isinstance(conv_linear, tf.keras.layers.Conv2D):
        weight = np.transpose(weight, (3, 2, 0, 1))
        # [Noc, Nic, kh, kw]
        shape = np.array([shape[3], shape[2], shape[0], shape[1]])
    else:
        logger.error("_get_weight_tensor_transpose_reshape(): Operation type unsupported")

    weight_tensor.data = weight.reshape(-1)
    weight_tensor.shape = shape

    return weight_tensor

class PassThroughOp(tf.keras.layers.Layer):
    """
    This is a pass-through op, used for purpose of making an op a no-op
    """
    # pylint: disable=arguments-differ
    @staticmethod
    def call(inputs):
        """
        This is a function to return input as an output
        :param inputs: input to pass through
        """
        return inputs

def _remove_bn_from_sequential(layer: tf.keras.layers.Layer, bn: tf.keras.layers.BatchNormalization):

    """
    This is the function for removing batch normalization layers that are layers of sequential model
    layer: model to obtain bn_layer that we want to remove
    bn: batch normalization layer that needs to be removed
    """
    layers_after_bn = []
    visited = False
    idx = None
    # pylint: disable=protected-access
    for index, inner_layer in enumerate(layer._layers):
        if visited:
            layers_after_bn.append(inner_layer)

        elif inner_layer == bn:
            visited = True
            idx = index

        elif inner_layer.submodules:
            _delete_bn_from_model(inner_layer, bn)

    if visited and idx is not None:
        # pylint: disable=protected-access
        for _ in range(len(layer._layers) - idx):
            layer.pop()
        for layer_to_add in layers_after_bn:
            layer.add(layer_to_add)

def _delete_bn_from_model(model: tf.keras.Model, bn_layer: tf.keras.layers.BatchNormalization):
    """
    Remove bn layer
    :param model
    :param bn_layer: bn layer that should be removed
    """
    if isinstance(model, tf.keras.Sequential):
        _remove_bn_from_sequential(model, bn_layer)

    # We are expecting to handle functional model or model subclassing in the elif statement
    elif isinstance(model, (tf.keras.layers.Layer, tf.keras.Model)):
        # pylint: disable=protected-access
        for layer in model._layers:
            if layer.submodules:
                _delete_bn_from_model(layer, bn_layer)

def _delete_all_bns_from_model(model: tf.keras.Model, bn_layers: List[tf.keras.layers.BatchNormalization]):
    '''
    Remove bn layer
    :param model
    :param bn_layers: bn layers that should be removed
    '''

    ref_name = common.module_to_name_map(model)

    # if bn is a layer in model subclassing api, will exist in ref_name so we will replace it with passthrough op otherwise
    # it would be a layer inside functional or sequential model which will be handled in delete_bn_from_model function

    for bn in bn_layers:
        if bn in ref_name:
            parent_ref, module_name = ref_name[bn]
            op = PassThroughOp()
            setattr(parent_ref, module_name, op)
        else:
            _delete_bn_from_model(model, bn)

def _fold_given_auto_selected_batch_norms(model: tf.keras.Model, layer_pairs: List[PairType]):
    """
    Fold a given set of batch_norm layers into conv layers
    :param model
    :param layer_pairs: Tuple of conv, bn layers and is_batch_norm_second flag
    """

    list_of_bn_layers = []
    for pair in layer_pairs:
        if pair[2]:
            conv_linear, batchnorm, is_batch_norm_second = pair
        else:
            batchnorm, conv_linear, is_batch_norm_second = pair

        assert isinstance(conv_linear, (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.Conv2DTranspose, tf.keras.layers.DepthwiseConv2D))

        list_of_bn_layers.append(batchnorm)

        #  check flag
        is_bias_valid = False

        if conv_linear.bias is not None:
            is_bias_valid = True

        bn_params = _get_bn_params(batchnorm)
        weight_tensor = _get_weight_tensor_transpose_reshape(conv_linear)
        bias_tensor = _get_bias_tensor(conv_linear)

        #Updated weight and bias
        bias = libpymo.fold(bn_params, weight_tensor, bias_tensor, is_bias_valid, is_batch_norm_second)

        if isinstance(conv_linear, tf.keras.layers.DepthwiseConv2D):
            # Depthwise conv layers in TF have outputs(Noc) set to 1.
            # we send in format [Nic, Noc, kh, kw]
            numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape).transpose((2, 3, 0, 1))
        elif isinstance(conv_linear, tf.keras.layers.Dense):
            # o, i - convert to i , o
            numpy_weight_reshaped = np.reshape(weight_tensor.data, [weight_tensor.shape[0], weight_tensor.shape[1]]).transpose(1, 0)
        else:
            # conv2D case
            # we sent in format [Noc, Nic, kh, kw]
            numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape).transpose((2, 3, 1, 0))

        # update bias tensor, even in case there was no existing bias add op in given conv2D op.
        bias_tensor_shape = [weight_tensor.shape[0]]
        numpy_bias_reshaped = np.reshape(bias, bias_tensor_shape)
        conv_linear.set_weights([numpy_weight_reshaped.data, numpy_bias_reshaped])

        BNUtils.modify_bn_params_to_make_as_passthrough(batchnorm)

    _delete_all_bns_from_model(model, list_of_bn_layers)
