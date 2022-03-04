# /usr/bin/env python3.6
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

""" Adaround wrapper """

from typing import Union, Dict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from packaging import version
import libpymo

# Import AIMET specific modules
from aimet_common.defs import AdaroundConstants
from aimet_common.defs import QuantScheme
from aimet_tensorflow.utils.op.conv import WeightTensorUtils, BiasUtils

QUANT_SCHEME_TO_PYMO = {QuantScheme.post_training_tf_enhanced: libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                        QuantScheme.post_training_tf: libpymo.QuantizationMode.QUANTIZATION_TF}


class AdaroundWrapper(keras.layers.Layer):
    """
    Adaround Wrapper base class
    """
    def __init__(self, session: tf.compat.v1.Session, op: tf.Operation, param_bw: int, is_symmetric: bool,
                 quant_scheme: QuantScheme):
        """
        :param session: Tf session
        :param op: Tf op
        :param param_bw: Bitwidth for weight quantization
        :param is_symmetric: Symmetric vs Asymmetric encodings
        :param quant_scheme: Quantization scheme
        """
        super(AdaroundWrapper, self).__init__()

        self._op = op
        weight, bias = self._get_weight_bias(session, op)

        self._weight_tensor = tf.convert_to_tensor(weight, dtype='float32')
        self._bias_tensor = None
        if bias is not None:
            self._bias_tensor = tf.convert_to_tensor(bias, dtype='float32')

        self.use_soft_rounding = tf.compat.v1.placeholder_with_default(True, shape=[])

        self.encoding = self.compute_encodings(weight, param_bw, is_symmetric, quant_scheme)
        self.alpha = self._initialize_alpha(self._weight_tensor, self.encoding)

    def adaround_weights(self) -> tf.Tensor:
        """
        Adaround the weight tensor
        :return: AdaRounded weight tensor
        """
        return self.get_adarounded_weight(self.alpha, self._weight_tensor, self.encoding, self.use_soft_rounding)

    @staticmethod
    def get_adarounded_weight(alpha, weight_tensor, encoding, use_soft_rounding) -> tf.Tensor:
        """
        Get the adarounded weight
        :param alpha: Alpha parameter
        :param weight_tensor: Weight to adaround
        :param encoding: Encodings corresponding to weights
        :param use_soft_rounding: True if soft rounding is to be used, False if hard rounding is to be used
        :return: Adarounded weight tensor
        """
        # Soft rounding maps alpha parameter between zero and one using rectified sigmoid function
        def compute_soft_rounding():
            return tf.clip_by_value(tf.sigmoid(alpha) * (AdaroundConstants.ZETA - AdaroundConstants.GAMMA) +
                                    AdaroundConstants.GAMMA, 0, 1)

        # Hard rounding maps alpha to exact zero or one
        def compute_hard_rounding():
            return tf.cast(alpha > 0, dtype=alpha.dtype)

        # Scale the tensor
        tensor = tf.floor(weight_tensor / encoding.delta)

        # Compute h_alpha depending on soft or hard rounding
        h_alpha = tf.cond(use_soft_rounding, compute_soft_rounding, compute_hard_rounding)

        # Adaround the tensor
        tensor = tf.add(tensor, h_alpha)

        # Quantize and de-quantize the tensor
        tensor_quant = tf.clip_by_value(tensor - encoding.offset, 0, 2 ** encoding.bw - 1)
        tensor_dequant = (tensor_quant + encoding.offset) * encoding.delta

        return tensor_dequant

    def _compute_output_with_adarounded_weights(self, inp_tensor: tf.Tensor, adaround_weight_tensor: tf.Tensor) ->\
            tf.Tensor:
        """
        Compute output of AdaroundSupportedModules with adarounded weights
        :param inp_tensor: The input tensor to be used for computing the output
        :param adaround_weight_tensor: The adarounded weight
        :return: output of the op computed with AdaRounded weights
        """
        if self._op.type == 'Conv2D':
            kwargs = self._get_conv_args(self._op)
            adaround_out_tensor = tf.nn.conv2d(inp_tensor, adaround_weight_tensor, **kwargs)

        elif self._op.type == 'DepthwiseConv2dNative':
            kwargs = self._get_conv_args(self._op)
            adaround_out_tensor = tf.nn.depthwise_conv2d(inp_tensor, adaround_weight_tensor, **kwargs)

        elif self._op.type == 'MatMul':
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
    def _initialize_alpha(tensor: tf.Tensor, encoding: libpymo.TfEncoding) -> tf.Variable:
        """
        Initializes alpha parameter, same shape as the weight tensor
        :param tensor: The weight tensor to be ada rounded
        """
        tensor_floor = tf.floor(tensor / encoding.delta)
        tensor = (tensor / encoding.delta) - tensor_floor

        # pylint: disable=invalid-unary-operand-type
        alpha = -tf.math.log((AdaroundConstants.ZETA - AdaroundConstants.GAMMA) / (tensor - AdaroundConstants.GAMMA) - 1)

        # pylint: disable=unexpected-keyword-arg
        # Resource variable is default in TF2.x
        if version.parse(tf.version.VERSION) >= version.parse("2.0"):
            alpha_var = tf.Variable(alpha, trainable=True, name='alpha')
        else:
            alpha_var = tf.Variable(alpha, trainable=True, use_resource=True, name='alpha')

        return alpha_var

    @staticmethod
    def _get_weight_bias(session: tf.compat.v1.Session, op: tf.Operation) -> (np.ndarray, Union[None, np.ndarray]):
        """
        :param session: Tf session
        :param op: Tf op
        :return: weight and bias
        """
        # Get weight tensor of an op as numpy data
        weight = WeightTensorUtils.get_tensor_as_numpy_data(session, op)

        bias = None
        if not BiasUtils.is_bias_none(op):
            bias = BiasUtils.get_bias_as_numpy_data(session, op)

        return weight, bias

    @staticmethod
    def compute_encodings(weight_data: np.ndarray, param_bw: int, is_symmetric: bool, quant_scheme: QuantScheme) \
            -> libpymo.TfEncoding:
        """
        :param weight_data: Weight data of Adaround supported ops
        :param param_bw: bitwidth (4-31) to use for quantizing weight data
        :param quant_scheme: Quantization scheme
        :param is_symmetric: True if symmetric encodings is used, else asymmetric encodings.
        :return: Encodings (max, min, delta and offset)
        """
        quant_scheme = QUANT_SCHEME_TO_PYMO[quant_scheme]

        # Create Encodings Analyzer and collect statistical data to compute encodings
        # Since the weight data is numpy and on CPU memory, useCuda is False
        analyzer = libpymo.EncodingAnalyzerForPython(quant_scheme)
        analyzer.updateStats(weight_data, False)

        # Compute the encodings for the weight data using collected stats
        encoding, _ = analyzer.computeEncoding(param_bw, is_symmetric, False, True)

        return encoding

    @staticmethod
    def _get_conv_args(op: tf.Operation) -> Dict:
        """
        :param op: Tf op of type Conv2d, Depthwise_Conv2d
        :return: keyword arguments
        """
        kwargs = dict(
            data_format=op.get_attr("data_format").decode('utf-8'),
            strides=op.get_attr("strides"),
            padding=op.get_attr('padding').decode('utf-8'),
            dilations=op.get_attr('dilations'))
        return kwargs
