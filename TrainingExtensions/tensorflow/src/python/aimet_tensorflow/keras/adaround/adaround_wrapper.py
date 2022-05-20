# /usr/bin/env python3.6
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

""" Adaround wrapper """

from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import AIMET specific modules
import aimet_common.libpymo as libpymo
from aimet_common.defs import AdaroundConstants
from aimet_common.defs import QuantScheme
from aimet_tensorflow.adaround.adaround_wrapper import AdaroundWrapper as TfAdaroundWrapper


class AdaroundWrapper(keras.layers.Layer):
    """
    Adaround Wrapper base class
    """
    def __init__(self, layer: tf.keras.layers.Layer, param_bw: int, quant_scheme: QuantScheme, is_symmetric: bool):
        """
        :param layer: Tf keras layer.
        :param param_bw: Bitwidth for weight quantization
        :param quant_scheme: Quantization scheme
        :param is_symmetric: Symmetric vs Asymmetric encodings
        """
        super(AdaroundWrapper, self).__init__()

        self._layer = layer
        weights = layer.get_weights()
        self._weight_tensor = weights[0]
        self._bias_tensor = None
        if len(weights) > 1:
            self._bias_tensor = weights[1]

        self.use_soft_rounding = self.add_weight(layer.name + '_use_soft_rounding', dtype=tf.bool,
                                                 initializer=tf.constant_initializer(True), trainable=False)

        self.encoding = self.compute_encodings(self._weight_tensor, param_bw, quant_scheme, is_symmetric,
                                               strict_symmetric=False, unsigned_symmetric=True)
        alpha = self._calculate_alpha(self._weight_tensor, self.encoding)
        self.alpha = self.add_weight(self._layer.name + '_alpha', trainable=True, shape=alpha.shape)
        self.alpha.assign(alpha)

    def adaround_weights(self) -> tf.Tensor:
        """
        Adaround the weight tensor
        :return: AdaRounded weight tensor
        """
        return TfAdaroundWrapper.get_adarounded_weight(self.alpha, self._weight_tensor, self.encoding,
                                                       self.use_soft_rounding, enable_per_channel=False,
                                                       ch_axis=len(list(self._weight_tensor.shape)) - 1)

    def _compute_output_with_adarounded_weights(self, inp_tensor: tf.Tensor, adaround_weight_tensor: tf.Tensor) -> \
            tf.Tensor:
        """
        Compute output of AdaroundSupportedModules with adarounded weights
        :param inp_tensor: The input tensor to be used for computing the output
        :param adaround_weight_tensor: The adarounded weight
        :return: output of the op computed with AdaRounded weights
        """
        if isinstance(self._layer, tf.keras.layers.Conv2D):
            kwargs = self._get_conv_args(self._layer)
            if isinstance(self._layer, tf.keras.layers.DepthwiseConv2D):
                adaround_out_tensor = tf.nn.depthwise_conv2d(inp_tensor, adaround_weight_tensor, **kwargs)
            else:
                adaround_out_tensor = tf.nn.conv2d(inp_tensor, adaround_weight_tensor, **kwargs)

        elif isinstance(self._layer, tf.keras.layers.Dense):
            adaround_out_tensor = tf.matmul(inp_tensor, adaround_weight_tensor)

        else:
            raise ValueError('Op type not supported')

        return adaround_out_tensor

    def call(self, inputs, **kwargs): # pylint: disable=unused-argument
        """
        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments
        :return: Adarounded output tensor
        """
        adaround_weight_tensor = self.adaround_weights()
        adaround_out_tensor = self._compute_output_with_adarounded_weights(inputs, adaround_weight_tensor)

        if self._bias_tensor is not None:
            adaround_out_tensor = adaround_out_tensor + self._bias_tensor

        return adaround_out_tensor

    @staticmethod
    def _calculate_alpha(weight: np.ndarray, encoding: libpymo.TfEncoding) -> np.ndarray:
        """
        Calculate alpha parameter, same shape as the weight tensor
        :param weight: The weight tensor to be ada rounded
        """
        tensor_floor = tf.floor(weight / encoding.delta)
        tensor = (weight / encoding.delta) - tensor_floor

        # pylint: disable=invalid-unary-operand-type
        alpha = -tf.math.log((AdaroundConstants.ZETA - AdaroundConstants.GAMMA) / (tensor - AdaroundConstants.GAMMA) -
                             1)
        return alpha

    @staticmethod
    def compute_encodings(weight_data: np.ndarray, param_bw: int, quant_scheme: QuantScheme, is_symmetric: bool,
                          strict_symmetric: bool, unsigned_symmetric: bool) \
            -> libpymo.TfEncoding:
        """
        :param weight_data: Weight data of Adaround supported ops
        :param param_bw: bitwidth (4-31) to use for quantizing weight data
        :param quant_scheme: Quantization scheme
        :param is_symmetric: True if symmetric encodings is used, else asymmetric encodings.
        :param strict_symmetric: If true, and if is_symmetric is true, calculate encodings exactly centered
        around 0. E.g. if bw==8, then this results in quantized int values (-127:127). If this is not set, then
        quantized int values would be (-128:127) to use the entire range.
        :param unsigned_symmetric: If true, and if is_symmetric is true, check if the entire statistics we
        have collected are for +ve numbers. If yes, use quantized int values (0:255). This is a special case,
        where we have double the resolution for the computed encodings while still preserving the zero-point to
        be absolute 0.
        :return: Encodings (max, min, delta and offset)
        """
        return TfAdaroundWrapper.compute_encodings(weight_data, param_bw, quant_scheme, is_symmetric,
                                                   strict_symmetric, unsigned_symmetric, enable_per_channel=False,
                                                   ch_axis=len(list(weight_data.shape)) - 1)

    @staticmethod
    def _get_conv_args(layer: tf.keras.layers.Conv2D) -> Dict:
        """
        :param op: Tf op of type Conv2d, Depthwise_Conv2d
        :return: keyword arguments
        """

        if layer.data_format == 'channels_last':
            data_format = 'NHWC'
            strides = [1, layer.strides[0], layer.strides[1], 1]
        else:
            data_format = 'NCHW'
            strides = [1, 1, layer.strides[0], layer.strides[1]]

        if layer.padding == 'valid':
            padding = 'VALID'
        else:
            padding = 'SAME'

        kwargs = {'data_format': data_format,
                  'strides': strides,
                  'padding': padding,
                  'dilations': layer.dilation_rate}
        return kwargs
