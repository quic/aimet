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
""" Qc Quantize op for tf 2 keras """

from typing import Union, List
import tensorflow as tf
import numpy as np

@tf.custom_gradient
def round_with_gradient(x):
    """ Round with gradient implemented """
    def grad(upstream):
        return upstream
    return tf.math.round(x), grad

class QcQuantizeLayer(tf.keras.layers.Layer):
    """ Wrapper for simulating quantization noise """

    # pylint: disable=arguments-differ
    def call(self, inputs):
        """
        Forward-pass routine. This quantizes the weights before delegating to the wrapped module and then quantizes the
        output before returning the same
        :param inputs: Inputs passed to the module in the forward pass
        :return: Quantized output from the wrapped module
        """
        outputs = self._quantize_activation(inputs)
        return outputs

    @staticmethod
    def _quantize_activation(activation: Union[tf.Tensor, List]):
        """
        Quantize activation
        :param activation: Activation tensor to quantize
        """
        if isinstance(activation, tf.Tensor):
            activation = [activation]
        quantized_activations = []
        for tensor in activation:
            quantized_activations.append(round_with_gradient(tensor))
            # quantized_activations.append(tf.add(tensor, 1))
        if len(quantized_activations) == 1:
            quantized_activations = quantized_activations[0]
        return quantized_activations

class QcQuantizeParamWrapper(tf.keras.layers.Layer):
    """ Wrapper for simulating quantization noise """
    def __init__(self, layer_to_wrap: tf.keras.layers.Layer):
        super(QcQuantizeParamWrapper, self).__init__()
        self._layer_to_wrap = layer_to_wrap

    # pylint: disable=arguments-differ
    def call(self, inputs):
        """
        Forward-pass routine. This quantizes the weights before delegating to the wrapped module and then quantizes the
        output before returning the same
        :param inputs: Inputs passed to the module in the forward pass
        :return: Quantized output from the wrapped module
        """
        shadow_params = [tf.keras.backend.get_value(param) for param in self._layer_to_wrap.weights]
        self._quantize_params()
        outputs = self._layer_to_wrap(inputs)
        self._restore_shadow_params(shadow_params)
        return outputs

    def get_config(self):
        """ Override get_config """
        return {"layer_to_wrap": self._layer_to_wrap}

    def _quantize_params(self):
        """ Quantize parameters """
        for idx, param in enumerate(self._layer_to_wrap.weights):
            param_val = tf.keras.backend.get_value(param)
            quantized_param = np.round(param_val)
            self._layer_to_wrap.weights[idx].assign(quantized_param)

    def _restore_shadow_params(self, shadow_params):
        """
        Restore saved parameters
        :param shadow_params: Original parameters to restore
        """
        for idx, param in enumerate(shadow_params):
            self._layer_to_wrap.weights[idx].assign(param)
