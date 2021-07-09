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
import  numpy as np
import tensorflow as tf
import libpymo
from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.utils import common_tf2

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

LayerType = Union[tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.Conv2DTranspose,
                  tf.keras.layers.DepthwiseConv2D]

PairType = Union[Tuple[LayerType, tf.keras.layers.BatchNormalization, bool],
                 Tuple[tf.keras.layers.BatchNormalization, LayerType, bool]]

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
    layers_after_bn  = []
    visited = False
    idx = None
    for index, layer2 in enumerate(layer.layers):
        if visited:
            layers_after_bn .append(layer2)

        elif layer2 == bn:
            visited = True
            idx = index

        elif layer2.submodules and isinstance(layer2, tf.keras.Sequential):
            _remove_bn_from_sequential(layer2, bn)

    if visited and idx is not None:
        for _ in range(len(layer.layers) - idx):
            layer.pop()
        for layer_to_add in layers_after_bn :
            layer.add(layer_to_add)


def _delete_bn_from_model(model: tf.keras.Model, bn_layers: List[tf.keras.layers.BatchNormalization]):
    """
    Remove bn layer
    :param model
    :param bn_layers: bn layers that should be removed
    """

    ref_name = common_tf2.module_to_name_map(model)

    for bn in bn_layers:
        if bn in ref_name.keys():
            parent_ref, module_name = ref_name[bn]
            op = PassThroughOp()
            setattr(parent_ref, module_name, op)
        else:
            _remove_bn_from_sequential(model, bn)


def _fold_given_auto_selected_batch_norms(model: tf.keras.Model, layer_pairs: List[PairType]):
    """
    Fold a given set of batch_norm layers into conv layers
    :param layer_pairs: Tuple of conv, bn layers and is_batch_norm_second flag
    :param model
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


    _delete_bn_from_model(model, list_of_bn_layers)
