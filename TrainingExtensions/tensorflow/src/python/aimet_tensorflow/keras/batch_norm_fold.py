# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

from collections import OrderedDict
from typing import Optional, Tuple, Union, List, Dict, Set
from enum import IntEnum
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.layers.core import TFOpLambda

import aimet_common.libpymo as libpymo
from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.utils import common
from aimet_tensorflow.keras.utils.op.batchnorm import BNUtils
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

LAYER_TYPE = Union[tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.Conv2DTranspose,
                   tf.keras.layers.DepthwiseConv2D]

PAIR_TYPE = Union[Tuple[LAYER_TYPE, tf.keras.layers.BatchNormalization, bool],
                  Tuple[tf.keras.layers.BatchNormalization, LAYER_TYPE, bool]]

BN_TYPE = tf.keras.layers.BatchNormalization
# Todo: search for more types of convolution
LINEAR_TYPE = tf.keras.layers.Dense
CONV_TYPE = tf.keras.layers.Conv2D
FLATTEN_TYPE = Union[tf.keras.layers.Flatten, tf.keras.layers.Reshape]

class PerChannelQuantizerType(IntEnum):
    """
    Enumeration of ConvLinear Input/Output/Param PerChannel Quantizers
    """
    INPUTS = 0
    OUPUTS = 1
    PARAMS = 2

class ConvLinearParamType(IntEnum):
    """
    Enumeration of ConvLinear Param Type
    """
    WEIGHTS = 0
    BIAS = 1


def _check_layer_to_find_pattern(cur_layer: tf.keras.layers.Layer,
                                 conv_linear_with_bn_dict: Dict[Union[CONV_TYPE, LINEAR_TYPE],
                                                                List[Union[None, BN_TYPE]]],
                                 layer_out_node_ref: Dict,
                                 has_seen: List[Union[None, CONV_TYPE, BN_TYPE, FLATTEN_TYPE]]):
    """
    find all paths in the model considering all inputs.

    :param cur_layer: layer to investigate for finding a pattern
    :param conv_linear_with_bn_dict: dictionary to store possible conv_bn pairs,
        key: Dense or Conv layer & Value: list of BNS;
        first index in this list shows bn_in and the second index shows bn_out
    :param layer_out_node_ref: dictionary includes layer_ref as a key, outbound nodes as value
    :param has_seen: for storing the layer which is useful for finding pattern in the next layers;
        index 0 is for conv op, index 2 is for bn op and index 3 is for storing flatten/reshape op
    """

    # pylint: disable=too-many-branches
    if isinstance(cur_layer, CONV_TYPE):
        if has_seen[1] is not None:
            conv_linear_with_bn_dict[cur_layer] = [has_seen[1], None]
            has_seen[1] = None
        if (cur_layer.activation is tf.keras.activations.linear) and \
                (cur_layer in layer_out_node_ref) and len(layer_out_node_ref[cur_layer]) == 1:
            has_seen[0] = cur_layer
    elif isinstance(cur_layer, BN_TYPE):
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
    elif isinstance(cur_layer, LINEAR_TYPE):
        if has_seen[1] is not None and has_seen[2] is not None:
            conv_linear_with_bn_dict[cur_layer] = [has_seen[1], None]
        has_seen[2] = None
        has_seen[1] = None
    else:
        has_seen[0] = None
        has_seen[1] = None
        has_seen[2] = None


def _add_children_layer_before_parent_layer(cur_layer: tf.keras.layers.Layer, node_layer_map: Dict,
                                            layer_out_node_map: Dict,
                                            visited_layers: Set[tf.keras.layers.Layer],
                                            reversed_ordered_layers: List[tf.keras.layers.Layer]):
    """
    Function to use topological sorting for finding all the layers which are accessible
    from the specific input_layer in the opposite order of occurrence.

    :param cur_layer:layer that we want to find path from
    :param node_layer_map: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param layer_out_node_map: dictionary includes layer_ref as a key, outbound nodes as value
    :param visited_layers: Set of all layers that have been visited
    :param reversed_ordered_layers: List of layers in the opposite order of occurrence
        for the layers that we have visited so far
    """

    # Mark the current layer as visited.
    visited_layers.add(cur_layer)

    if cur_layer in layer_out_node_map:
        # Recur for all the layers adjacent to this layer
        for next_node in layer_out_node_map[cur_layer]:
            next_layer = node_layer_map[next_node][1]
            if next_layer not in visited_layers:
                _add_children_layer_before_parent_layer(next_layer, node_layer_map,
                                                        layer_out_node_map, visited_layers,
                                                        reversed_ordered_layers)
            reversed_ordered_layers.append(cur_layer)
    else:
        reversed_ordered_layers.append(cur_layer)


def _get_ordered_layers(node_layer_map: Dict,
                        layer_out_node_map: Dict) -> List[tf.keras.layers.Layer]:
    """
    Function to return the list with all the layers in which layers come before parent layer.

    :param node_layer_map: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param layer_out_node_map: dictionary includes layer_ref as a key, outbound nodes as value
    :return: ordered_layers: List of all layers in the order of occurrence
    """
    # to find the input layers of the model
    input_layers = common.find_input_layers(node_layer_map)

    #  Set of all layers that have been visited (to cut short duplicate traversals)
    visited_layers = set()

    # List of all layers in the opposite of order of occurrence
    reversed_ordered_layers = []

    for input_layer in input_layers:
        _add_children_layer_before_parent_layer(input_layer, node_layer_map, layer_out_node_map,
                                                visited_layers, reversed_ordered_layers)

    # reverse the list because layers are in reverse order
    ordered_layers = reversed_ordered_layers[::-1]

    # # filter ordered ops for only valid ops
    # ordered_ops = [op for op in ordered_ops if op in valid_ops]

    return ordered_layers


def _get_ordered_conv_linears(node_layer_map: Dict,
                              layer_out_node_map: Dict) -> List[Union[CONV_TYPE, LINEAR_TYPE]]:
    """
    helper to select a list of conv_linears in the order of occurence

    :param node_layer_map: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param layer_out_node_map: dictionary includes layer_ref as a key, outbound nodes as value
    :return: return List of conv/linear layer refs
    """
    # get ordered layers list in node_layer map dictionary
    list_of_ordered_layers = _get_ordered_layers(node_layer_map, layer_out_node_map)

    # look for conv layers
    ordered_conv_linears = []
    for layer in list_of_ordered_layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            ordered_conv_linears.append(layer)
    return ordered_conv_linears


def _fill_conv_linear_bn_dict(cur_layer: tf.keras.layers.Layer, node_layer_ref: Dict,
                              layer_out_node_ref: Dict,
                              has_seen: List[Union[None, CONV_TYPE, BN_TYPE, FLATTEN_TYPE]],
                              visited_layer: Set[tf.keras.layers.Layer],
                              conv_linear_with_bn_dict: Dict[Union[CONV_TYPE, LINEAR_TYPE],
                                                             List[Union[None, BN_TYPE]]]):
    """
    fill conv_linear_bn_dict for the model

    :param cur_layer: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param node_layer_ref: dictionary includes node_ref as a key, in_layers and out_layer as value
    :param layer_out_node_ref: dictionary includes layer_ref as a key, outbound nodes as value
    :paramm has_seen: for storing the layer which is useful for finding pattern in the next layers;
        index 0 is for conv op, index 2 is for bn op and index 3 is for storing flatten/reshape op
    :param visited_layer: to store all the layers that have been visited so far in the dictionary
    :param conv_linear_with_bn_dict: dictionary of all possible conv_bn pairs,
        key: Dense or Conv layer & Value: list of BNS;
        first index in this list shows bn_in and the second index shows bn_out
    """

    # Mark the current layer as visited to prevent passing from one layer more than once
    visited_layer.add(cur_layer)

    _check_layer_to_find_pattern(cur_layer, conv_linear_with_bn_dict, layer_out_node_ref, has_seen)

    if cur_layer in layer_out_node_ref:
        for next_node in layer_out_node_ref[cur_layer]:
            next_layer = node_layer_ref[next_node][1]
            if next_layer not in visited_layer:
                _fill_conv_linear_bn_dict(next_layer, node_layer_ref, layer_out_node_ref, has_seen,
                                          visited_layer, conv_linear_with_bn_dict)
            else:
                has_seen[0] = None
                has_seen[1] = None
                has_seen[2] = None


def _find_possible_convs_linears_bn(node_layer_map: Dict, layer_out_node_map: Dict)\
        -> Dict[Union[CONV_TYPE, LINEAR_TYPE], List[Union[None, BN_TYPE]]]:
    """
    find all possible convs_linears_bn by traversing all paths in the model considering all inputs

    :param node_layer_map:  dictionary includes node_ref as a key, in_layers and out_layer as value
    :param layer_out_node_map: dictionary includes layer_ref as a key, outbound nodes as value
    :return: return dictionary of all possible conv_bn pairs,
        key: Dense or Conv layer & Value: list of BNS;
        first index in this list shows bn_in and the second index shows bn_out
    """

    input_layers = common.find_input_layers(node_layer_map)
    visited_layer = set()
    conv_linear_with_bn_dict = {}

    for input_layer in input_layers:
        _fill_conv_linear_bn_dict(input_layer, node_layer_map, layer_out_node_map,
                                  [None, None, None], visited_layer, conv_linear_with_bn_dict)

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


def _get_bias_tensor(conv_linear: LAYER_TYPE) -> libpymo.TensorParams():
    """
    Get bias tensor in given conv layer.

    Packs bias in the format required for BN fold
    (libpymo.TensorParams()).
    :param conv_linear: conv Layer
    :return: return bias param in libpymo.TensorParams() format.
    """

    bias_tensor = libpymo.TensorParams()
    if conv_linear.bias is not None:
        bias_tensor.data = conv_linear.bias.numpy().reshape(-1)
        bias_tensor.shape = np.array(conv_linear.bias.shape)

    return bias_tensor


def _get_weight_tensor_transpose_reshape(conv_linear: LAYER_TYPE) -> libpymo.TensorParams():
    """
    Get weight tensor from conv layer.

    Converts to right format - performs transpose and reshape.
    Packs it to the format required for BN fold (libpymo.TensorParams()).
    :param conv_linear: conv layer
    :return: return weight tensor in libpymo.TensorParams() format.
    """

    # Weight tensor libpymo format
    weight_tensor = libpymo.TensorParams()

    # linear array to be sent for bn fold
    weight = conv_linear.get_weights()[0]
    shape = weight.shape

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
    elif isinstance(conv_linear, tf.keras.layers.Conv2DTranspose):
        weight = np.transpose(weight, (2, 3, 0, 1))
        # [Noc, Nic, kh, kw]
        shape = np.array([shape[2], shape[3], shape[0], shape[1]])
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

# pylint: disable=too-many-branches
def _delete_bn_from_functional(model: tf.keras.Model,
                               bn_layers_to_remove: List[tf.keras.layers.BatchNormalization]) -> tf.keras.Model:
    """
    This function is used to remove ALL batch normalization layers from a functional model passed via the
    bn_layers_to_remove parameter. Removing in place is not possible for functional models as the layers inbound and
    outbound connections are immutable. This function returns a new model with the batch normalization layers removed.
    param model: model to remove bn_layers from
    param bn_layers_to_remove: list of batch normalization layers to remove from the model
    """

    # In order to do this, we first need to know the original models inbound and outbound connections to each layer.
    # We then need to create a new model with the same inbound and outbound connections, but with the batch normalization
    # layers removed. This is done by rerouting the inbound nodes of the batch normalization layers to the inbound nodes
    # of the next layer. This can be seen in the following diagram:
    #
    # Original model flow ------------------------->
    #   ______________        ______________        ______________
    #  |             |       |             |       |             |
    #  |    Conv     |  -X-> |  Batch Norm |  -X-> |    ReLU     |
    #  |_____________|       |_____________|     ^ |_____________|
    #  New model flow   \                       /
    #                    \                     /
    #                     \___________________/
    #
    #
    #

    INBOUND_NODES = 'inbound_nodes'
    OUTBOUND_TENSORS = 'outbound_tensors'

    # Step 1: Get the inbound and outbound connections for each layer in the model
    model_layer_connections = OrderedDict()
    model_layer_connections[INBOUND_NODES] = OrderedDict()
    model_layer_connections[OUTBOUND_TENSORS] = OrderedDict()

    for current_layer in model.layers:
        for outbound_node in current_layer.outbound_nodes:
            outbound_layers_name = outbound_node.outbound_layer.name

            model_layer_connections[INBOUND_NODES][outbound_layers_name] = [
                *model_layer_connections[INBOUND_NODES].get(outbound_layers_name, []), current_layer.name
            ]

    if isinstance(model.input, list):
        # If the model has multiple inputs, we need to set the output tensor of each input layer
        for inp in model.input:
            model_layer_connections[OUTBOUND_TENSORS].update({inp.name: inp})
    else:
        model_layer_connections[OUTBOUND_TENSORS].update({model.layers[0].name: model.input})

    # Step 2: Create a new model with the batch normalization layers removed by iterating through the layers in the model
    # and using the inbound and outbound connections to rerouting around the batch normalization layers.

    model_outputs = []
    for current_layer in model.layers:
        if isinstance(current_layer, tf.keras.layers.InputLayer):
            continue

        # Determine input tensors of the given layer
        layer_input = [model_layer_connections[OUTBOUND_TENSORS][layer_aux]
                       for layer_aux in model_layer_connections[INBOUND_NODES][current_layer.name]]


        layer_input = layer_input[0] if len(layer_input) == 1 else layer_input

        # Reroute around batch normalization layers if the layer is in the list of layers to remove
        if current_layer in bn_layers_to_remove:
            logger.debug("Removing Batch Normalization layer %s", current_layer.name)

            for outbound_node in current_layer._outbound_nodes:  # pylint: disable=protected-access
                # Find and replace the Batch Normalization output layers input that holds the Batch Normalization layer
                # node and replace it with the input layers of the Batch Normalization layer.
                # For example, if ReLU's inputs are [conv1_bn] and conv1_bn's inputs are [conv1], then we replace
                # ReLU's inputs with [conv1]

                all_batch_norms_inbound_layers_names = \
                    [inbound_node.inbound_layers.name for inbound_node in current_layer._inbound_nodes]  # pylint: disable=protected-access

                # Go through all the outbound layers of the batch normalization layer and replace the batch normalization
                # layer name with the input layer names of the batch normalization layer.
                batch_norms_outbound_layers_new_inbound_layers_names = \
                    [outlayer.replace(current_layer.name, *all_batch_norms_inbound_layers_names)
                     for outlayer in model_layer_connections[INBOUND_NODES][outbound_node.outbound_layer.name]]

                model_layer_connections[INBOUND_NODES].update(
                    {outbound_node.outbound_layer.name: batch_norms_outbound_layers_new_inbound_layers_names})

                # The above updates our dict for the mapping of the inputs but we need to also update what Keras thinks
                # the inputs are. This is done by updating the inbound nodes of the output layer of the Batch Normalization.
                # THIS IS ONLY FOR MAPPING THE INPUTS TO BUILD A NEW MODEL. The original models underlinig structure is
                # not changed.
                outbound_node.outbound_layer._inbound_nodes = current_layer.inbound_nodes  # pylint: disable=protected-access

        # Otherwise, treat like a normal layer
        else:
            # Since we are rerouting around the batch normalization layers, we need to temporarily remove the inbound and
            # outbound nodes of the batch normalization layers so that the model can be built correctly and not duplicate
            # the non batch normalization layers inbound/outbound nodes.
            current_layer._inbound_nodes = []  # pylint: disable=protected-access
            # Special case for when there is a Lambda opertaion with multiple inputs. For example, x = y + z.
            if isinstance(current_layer, TFOpLambda) and isinstance(layer_input, List):
                x = current_layer(*layer_input)
            else:
                x = current_layer(layer_input)
            current_layer._outbound_nodes = [] # pylint: disable=protected-access

            # Set new output tensor (in this case, it will be the same as the original model)
            model_layer_connections[OUTBOUND_TENSORS].update({current_layer.name: x})

        # Save tensor in output list if it is output in the initial model
        if current_layer.name in model.output_names:
            model_outputs.append(x)

    tf.keras.backend.clear_session() # clear session to not have tensor name conflicts
    return tf.keras.Model(inputs=model.inputs, outputs=model_outputs)


def _delete_bn_from_sequential(layer: tf.keras.layers.Layer,
                               bn: tf.keras.layers.BatchNormalization):

    """
    This is the function for removing batch normalization layers that are layers of sequential model

    :param layer: model to obtain bn_layer that we want to remove
    :param bn: batch normalization layer that needs to be removed
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
            _delete_bn_for_non_subclassed_model(inner_layer, bn)

    if visited and idx is not None:
        # pylint: disable=protected-access
        for _ in range(len(layer._layers) - idx):
            layer.pop()
        for layer_to_add in layers_after_bn:
            layer.add(layer_to_add)


def _delete_bn_for_non_subclassed_model(model: (tf.keras.Model, tf.keras.layers.Layer),
                                        bn_layer: tf.keras.layers.BatchNormalization):
    """
    Remove bn layer for those model which are not part of model subclassing

    :param model: model to delete bn layers from
    :param bn_layer: bn layer that should be removed
    """

    if isinstance(model, tf.keras.Sequential):
        _delete_bn_from_sequential(model, bn_layer)

    # We are expecting to find sequential model in functional model
    # or model subclassing in the elif statement
    elif isinstance(model, (tf.keras.layers.Layer, tf.keras.Model)):
        # pylint: disable=protected-access
        for layer in model._layers:
            if layer.submodules:
                _delete_bn_for_non_subclassed_model(layer, bn_layer)


def _delete_bn_from_model_subclassing(module_to_name_map: Dict[tf.keras.layers.Layer,
                                                               Tuple[tf.keras.Model, str]],
                                      bn_layer: tf.keras.layers.BatchNormalization):
    """
    Remove bn layer which is part of model subclassing api
    or model inheriting from tf.keras.layers.Layer

    :param module_to_name_map: model to remove bn from
    :param bn_layer: bn layer that should be removed
    """

    parent_ref, module_name = module_to_name_map[bn_layer]
    op = PassThroughOp()
    setattr(parent_ref, module_name, op)

# pylint: disable=inconsistent-return-statements
def _delete_all_bns_from_model(model: (tf.keras.Model, tf.keras.layers.Layer),
                               bn_layers: List[tf.keras.layers.BatchNormalization]) -> Optional[tf.keras.Model]:
    """
    Remove all bn layers

    :param model
    :param bn_layers: bn layers that should be removed
    :return: new model with bn layers removed, if model is functional else None
    """

    if isinstance(model, Functional) and not isinstance(model, tf.keras.Sequential) and bn_layers:
        return _delete_bn_from_functional(model, bn_layers)

    module_to_name_map = common.module_to_name_map(model)

    for bn_layer in bn_layers:
        if bn_layer in module_to_name_map:
            _delete_bn_from_model_subclassing(module_to_name_map, bn_layer)
        else:
            _delete_bn_for_non_subclassed_model(model, bn_layer)

def _find_all_batch_norms_to_fold(model: tf.keras.Model) -> List[PAIR_TYPE]:
    """
    uses searcher to choose layers for bias correction

    :param model: model to obtain conv_linear pairs for
    :return: List of conv/linear layers with associated bn op / activation info
    """

    node_layer_map = common.create_node_to_layer_map(model)
    layer_out_node_map = common.create_layer_to_out_node_map(model)

    possible_convs_linears_bn = _find_possible_convs_linears_bn(node_layer_map, layer_out_node_map)

    # get all ordered convs/ linears layers
    ordered_conv_linears = _get_ordered_conv_linears(node_layer_map, layer_out_node_map)

    valid_bn_conv_linear_pairs = []

    # track BNs added for fold
    marked_bn_set = set()

    for conv_linear_layer in ordered_conv_linears:
        if conv_linear_layer in possible_convs_linears_bn.keys():
            bn_info = possible_convs_linears_bn[conv_linear_layer]
            if bn_info[1]:
                if bn_info[1] not in marked_bn_set:
                    valid_bn_conv_linear_pairs.append((conv_linear_layer, bn_info[1], True))
                    marked_bn_set.add(bn_info[1])
            elif bn_info[0]:
                if bn_info[0] not in marked_bn_set:
                    valid_bn_conv_linear_pairs.append((conv_linear_layer, bn_info[0], False))
                    marked_bn_set.add(bn_info[0])

    return valid_bn_conv_linear_pairs


def fold_all_batch_norms(model: tf.keras.Model) \
        -> Tuple[List[Tuple[tf.keras.layers.BatchNormalization, LAYER_TYPE]], tf.keras.Model]:
    """
    Fold all batch_norm layers in a model into corresponding conv/linear layers

    :param model: model to find all batch norms for
    :return: A tuple of List of conv/linear layers with associated bn op / activation info and a new model with the
    Batch Normalization layers folded
    """

    bn_conv_linear_pairs = _find_all_batch_norms_to_fold(model)

    # Potential new model is returned in case the model is a functional model
    potential_new_model = fold_given_batch_norms(model, bn_conv_linear_pairs)
    model = potential_new_model if potential_new_model else model

    # When returning the pairs, we want the second element of the pair to be the BN
    pairs_to_return = []
    for bn, conv, _ in bn_conv_linear_pairs:
        pairs_to_return.append((bn, conv))

    logger.warning("A new model is returned with the Batch Normalization layers removed for Keras models. "
                   "Please use this new model for the rest of the AIMET flow.")

    return pairs_to_return, model

#pylint: disable=protected-access
def fold_all_batch_norms_to_scale(sim: QuantizationSimModel):
    """
    Fold all batch_norm layers in a model into corresponding conv/linear layers

    :param sim: quantized keras model to fold all batch norms
    """

    assert isinstance(sim, QuantizationSimModel)

    bn_conv_linear_pairs = _find_all_batch_norms_to_fold(sim._model_without_wrappers)

    fold_given_batch_norms(sim, bn_conv_linear_pairs, is_fold_to_scale=True)

    # When returning the pairs, we want the second element of the pair to be the BN
    pairs_to_return = []
    for bn, conv, _ in bn_conv_linear_pairs:
        pairs_to_return.append((bn, conv))

    return pairs_to_return

# pylint: disable=too-many-locals
# pylint: disable=inconsistent-return-statements
def fold_given_batch_norms(model: Union[tf.keras.Model, QuantizationSimModel], layer_pairs: List[PAIR_TYPE], is_fold_to_scale: bool = False) -> Optional[tf.keras.Model]:
    """
    Fold a given set of batch_norm layers into conv_linear layers

    :param model: keras fp32 model/quantized model to fold selected batchnorms
    :param layer_pairs: Tuple of conv, bn layers and is_batch_norm_second flag
    :param is_fold_to_scale: default is False,  when it is True, fold BN scaling factor into the per-channel quantization scaling factor of the preceding convolution
    :return: new model with batch norm layers folded if model is a functional model, else None
    """
    if is_fold_to_scale:
        assert isinstance(model, QuantizationSimModel)
    else:
        assert isinstance(model, tf.keras.Model)

    list_of_bn_layers = []
    for pair in layer_pairs:

        conv_linear, batchnorm, is_batch_norm_second = pair
        assert isinstance(
            conv_linear,
            (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.DepthwiseConv2D))

        list_of_bn_layers.append(batchnorm)

        #  check flag
        is_bias_valid = False

        if conv_linear.bias is not None:
            is_bias_valid = True

        bn_params = _get_bn_params(batchnorm)
        weight_tensor = _get_weight_tensor_transpose_reshape(conv_linear)
        bias_tensor = _get_bias_tensor(conv_linear)

        # Updated weight and bias
        bias = libpymo.fold(bn_params, weight_tensor, bias_tensor, is_bias_valid,
                            is_batch_norm_second)

        if isinstance(conv_linear, tf.keras.layers.DepthwiseConv2D):
            # Depthwise conv layers in TF have outputs(Noc) set to 1.
            # we send in format [Nic, Noc, kh, kw]
            numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape)\
                .transpose((2, 3, 0, 1))
        elif isinstance(conv_linear, tf.keras.layers.Dense):
            # o, i - convert to i , o
            numpy_weight_reshaped = np.reshape(
                weight_tensor.data,
                [weight_tensor.shape[0], weight_tensor.shape[1]]).transpose(1, 0)
        elif isinstance(conv_linear, tf.keras.layers.Conv2DTranspose):
            # we sent in format [Noc, Nic, kh, kw]
            numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape) \
                .transpose((2, 3, 0, 1))
        else:
            # conv2D case
            # we sent in format [Noc, Nic, kh, kw]
            numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape)\
                .transpose((2, 3, 1, 0))

        # update bias tensor, even in case there was no existing bias add op in given conv2D op.
        bias_tensor_shape = [weight_tensor.shape[0]]
        numpy_bias_reshaped = np.reshape(bias, bias_tensor_shape)

        if not is_bias_valid:
            conv_linear.use_bias = True
            conv_linear.bias = conv_linear.add_weight(name="bias",
                                                      shape=(weight_tensor.shape[0],),
                                                      dtype=conv_linear.dtype,
                                                      trainable=True)
        conv_linear.set_weights([numpy_weight_reshaped.data, numpy_bias_reshaped])

        if is_fold_to_scale:
            sim = model
            conv_wrapper = sim.get_quant_wrapper_for_layer_name(conv_linear.name)
            # check no bias
            if hasattr(conv_wrapper, 'param_quantizers[1]'):
                conv_wrapper.param_quantizers[1].disable()
            conv_wrapper.output_quantizers[0].disable()

            # Disable quantizers of batchnorms
            bn_wrapper = sim.get_quant_wrapper_for_layer_name(batchnorm.name)
            bn_wrapper.param_quantizers[0].disable()
            bn_wrapper.param_quantizers[1].disable()
            bn_wrapper.param_quantizers[2].disable()
            bn_wrapper.param_quantizers[3].disable()
            bn_wrapper.input_quantizers[0].disable()
            bn_wrapper.output_quantizers[0].disable()

            _fold_pair_scale(sim, conv_linear, bn_params)

        BNUtils.modify_bn_params_to_make_as_passthrough(batchnorm)

    if is_fold_to_scale:
        return
    return _delete_all_bns_from_model(model, list_of_bn_layers)

def _fold_pair_scale(sim: QuantizationSimModel, conv_linear: tf.keras.layers, bn_params: libpymo.BNParams()):
    """
     Fold a batch_norm layer into conv_linear's scale
    :param sim: keras quantized model
    :param conv_linear: conv or Linear layer
    :param bn_params: bn_params
    """
    assert isinstance(sim, QuantizationSimModel)
    conv_linear_quantizer_w = sim._get_quantizers_by_layer(conv_linear)[PerChannelQuantizerType.PARAMS][ConvLinearParamType.WEIGHTS]
    encodings = conv_linear_quantizer_w.encoding
    new_encodings = []
    for old_encoding, bn_gamma_to_runningvar_ratio in zip(encodings, np.array(bn_params.gamma) * (1.0 / np.array(bn_params.runningVar))):
        new_encoding = libpymo.TfEncoding()
        if bn_gamma_to_runningvar_ratio >= 0:
            new_encoding.max = old_encoding.max * bn_gamma_to_runningvar_ratio
            new_encoding.min = old_encoding.min * bn_gamma_to_runningvar_ratio
        else:
            new_encoding.max = old_encoding.min * bn_gamma_to_runningvar_ratio
            new_encoding.min = old_encoding.max * bn_gamma_to_runningvar_ratio
        new_encoding.delta = old_encoding.delta * abs(bn_gamma_to_runningvar_ratio)
        new_encoding.offset = new_encoding.min/new_encoding.delta
        new_encoding.bw = old_encoding.bw

        new_encodings.append(new_encoding)

    conv_linear_quantizer_w.encoding = new_encodings
