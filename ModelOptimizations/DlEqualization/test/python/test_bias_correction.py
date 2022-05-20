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

import unittest
from enum import Enum
import numpy as np
from scipy.stats import norm

import aimet_common.libpymo as libpymo


class TestBiasCorrection(unittest.TestCase):

    def test_bias_correction(self):
        # Generating random numbers from a normal distribution for the weights and biases of the current and prev layer
        np.random.seed(1)
        shape = (2, 3, 2, 2)

        # output 1
        o1 = np.random.randn(*shape)

        # output 2
        o2 = np.random.randn(*shape)

        biasCorrection = libpymo.BiasCorrection()
        biasCorrection.storePreActivationOutput(o1)
        biasCorrection.storePreActivationOutput(o1)

        biasCorrection.storeQuantizedPreActivationOutput(o2)
        biasCorrection.storeQuantizedPreActivationOutput(o2)

        bias_tensor = libpymo.TensorParamBiasCorrection()
        bias = np.array(np.random.randn(shape[1]))
        bias_tensor.data = bias

        biasCorrection.correctBias(bias_tensor)

        bias_python = correct_bias(o1, o2, bias)

        print(bias_tensor.data)
        print(bias_python)
        assert np.allclose(bias_tensor.data, bias_python)

    def test_bias_correction_bn_params_no_activation(self):
        np.random.seed(1)
        shape = (3, 3, 2, 2)

        weight = np.random.randn(*shape)

        quantized_weight = np.random.randn(*shape)

        bn_params = libpymo.BnParamsBiasCorr()
        gamma = np.array(np.random.randn(3))
        beta = np.array(np.random.randn(3))
        bn_params.gamma = gamma
        bn_params.beta = beta

        bias_tensor = libpymo.TensorParamBiasCorrection()
        bias = np.array(np.random.randn(shape[1]))
        bias_copy = bias.copy()
        bias_tensor.data = bias

        activation = libpymo.ActivationType.noActivation
        biasCorrection = libpymo.BnBasedBiasCorrection()
        biasCorrection.correctBias(bias_tensor, quantized_weight, weight, bn_params, activation)
        bias_python = bn_based_bias_correction(weight, quantized_weight, bias_copy, beta,
                                               gamma, ActivationType.no_activation)
        assert (np.allclose(bias_python, bias_tensor.data))

    def test_bias_correction_bn_params_relu_activation(self):
        np.random.seed(1)
        shape = (3, 3, 2, 2)

        weight = np.random.randn(*shape)

        quantized_weight = np.random.randn(*shape)

        bn_params = libpymo.BnParamsBiasCorr()
        gamma = np.array(np.random.randn(3))
        beta = np.array(np.random.randn(3))
        bn_params.gamma = gamma
        bn_params.beta = beta

        bias_tensor = libpymo.TensorParamBiasCorrection()
        bias = np.array(np.random.randn(shape[1]))
        bias_copy = bias.copy()
        bias_tensor.data = bias

        activation = libpymo.ActivationType.relu
        biasCorrection = libpymo.BnBasedBiasCorrection()
        biasCorrection.correctBias(bias_tensor, quantized_weight, weight, bn_params, activation)
        bias_python = bn_based_bias_correction(weight, quantized_weight, bias_copy, beta,
                                               gamma, ActivationType.relu)
        assert (np.allclose(bias_python, bias_tensor.data))

    def test_bias_correction_bn_params_relu6_activation(self):
        np.random.seed(1)
        shape = (3, 3, 2, 2)

        weight = np.random.randn(*shape)

        quantized_weight = np.random.randn(*shape)

        bn_params = libpymo.BnParamsBiasCorr()
        gamma = np.array(np.random.randn(3))
        beta = np.array(np.random.randn(3))
        bn_params.gamma = gamma
        bn_params.beta = beta

        bias_tensor = libpymo.TensorParamBiasCorrection()
        bias = np.array(np.random.randn(shape[1]))
        bias_copy = bias.copy()
        bias_tensor.data = bias

        activation = libpymo.ActivationType.relu6
        biasCorrection = libpymo.BnBasedBiasCorrection()
        biasCorrection.correctBias(bias_tensor, quantized_weight, weight, bn_params, activation)
        bias_python = bn_based_bias_correction(weight, quantized_weight, bias_copy, beta,
                                               gamma, ActivationType.relu6)
        assert (np.allclose(bias_python, bias_tensor.data))

    def test_bias_correction_bn_params_relu_activation_depthwise_layer(self):
        np.random.seed(1)
        shape = (3, 1, 2, 2)

        weight = np.random.randn(*shape)

        quantized_weight = np.random.randn(*shape)

        bn_params = libpymo.BnParamsBiasCorr()
        gamma = np.array(np.random.randn(3))
        beta = np.array(np.random.randn(3))
        bn_params.gamma = gamma
        bn_params.beta = beta

        bias_tensor = libpymo.TensorParamBiasCorrection()
        bias = np.array(np.random.randn(3))
        bias_copy = bias.copy()
        bias_tensor.data = bias

        activation = libpymo.ActivationType.relu
        biasCorrection = libpymo.BnBasedBiasCorrection()
        biasCorrection.correctBias(bias_tensor, quantized_weight, weight, bn_params, activation)
        bias_python = bn_based_bias_correction(weight, quantized_weight, bias_copy, beta,
                                               gamma, ActivationType.relu)
        assert (np.allclose(bias_python, bias_tensor.data))


def correct_bias(y1, y2, bias):
    y2 = y2 - y1
    y2 = y2.mean(3).mean(2).mean(0)
    return bias - y2


# Batch Norm Based bias correction
class ActivationType(Enum):
    no_activation = 1
    relu = 2
    relu6 = 3


def bn_based_bias_correction(weight, quantized_weight, bias, beta, gamma, activation_type):
    subtracted_weights = quantized_weight - weight
    epsilon = subtracted_weights.sum(3).sum(2)

    if activation_type is ActivationType.no_activation:
        e_x = beta
    elif activation_type is ActivationType.relu:
        e_x = beta * (1 - norm.cdf(-beta / gamma)) + gamma * norm.pdf(-beta / gamma)
    else:
        b = 6
        z = norm.pdf(-beta / gamma) - norm.pdf((b - beta) / gamma)
        Z = norm.cdf((b - beta) / gamma) - norm.cdf(-beta / gamma)
        e_x = gamma * z + beta * Z + b * (1 - norm.cdf((b - beta) / gamma))

    if epsilon.shape[1] == 1:
        ep = epsilon.reshape(epsilon.shape[0])
        error = np.multiply(ep, e_x)
    else:
        error = np.matmul(epsilon, e_x)

    return bias - error
