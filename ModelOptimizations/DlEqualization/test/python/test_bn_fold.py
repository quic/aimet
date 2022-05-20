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
import numpy as np
import copy

import aimet_common.libpymo as libpymo


class TestBNFold(unittest.TestCase):

    def test_bn_fold(self):
        # Generating random numbers from a normal distribution for the weights and biases of the current and prev layer
        np.random.seed(1)
        total = 2 * 3 * 2 * 2

        beta = np.array(np.random.randn(2))
        gamma = np.array(np.random.randn(2))
        running_mean = np.array(np.random.randn(2))
        running_var = np.array(np.random.randn(2))

        bn_params = libpymo.BNParams()
        bn_params.beta = beta
        bn_params.gamma = gamma
        bn_params.runningMean = running_mean
        bn_params.runningVar = running_var

        weight_tensor = libpymo.TensorParams()
        weight = np.array(np.random.randn(total))
        weight_sz = np.array([2, 3, 2, 2])
        weight_tensor.data = weight
        weight_tensor.shape = weight_sz

        random_bias = np.array(np.random.rand(2))
        bias_tensor = libpymo.TensorParams()
        bias_tensor.data = random_bias
        bias_tensor.shape = np.array([2])

        w, b = bn_fold_prev_layer(weight.reshape(weight_sz), random_bias,
                                  beta, gamma, running_mean, running_var)

        bias = libpymo.fold(bn_params, weight_tensor, bias_tensor, True, True)

        assert (np.allclose(w.flatten(), weight_tensor.data))
        assert (np.allclose(b.flatten(), bias))

    def test_bn_fold_linear_layer(self):
        # Generating random numbers from a normal distribution for the weights and biases of the current and prev layer
        np.random.seed(1)
        total = 2 * 3

        weight = np.array(np.random.randn(total))
        weight_sz = np.array([2, 3, 1, 1])

        beta = np.array(np.random.randn(2))
        gamma = np.array(np.random.randn(2))
        running_mean = np.array(np.random.randn(2))
        running_var = np.array(np.random.randn(2))

        bn_params = libpymo.BNParams()
        bn_params.beta = beta
        bn_params.gamma = gamma
        bn_params.runningMean = running_mean
        bn_params.runningVar = running_var

        layer_weight_params = libpymo.TensorParams()
        layer_weight_params.data = weight
        layer_weight_params.shape = weight_sz

        random_bias = np.array(np.random.rand(2))
        bias_tensor = libpymo.TensorParams()
        bias_tensor.data = random_bias
        bias_tensor.shape = np.array([2])

        w, b = bn_fold_prev_layer(weight.reshape(weight_sz), random_bias,
                                  beta, gamma, running_mean, running_var)
        bias = libpymo.fold(bn_params, layer_weight_params, bias_tensor, True, True)

        print(w.flatten())
        print("******")
        print(layer_weight_params.data)

        assert (np.allclose(layer_weight_params.data, w.flatten()))
        assert (np.allclose(bias, b.flatten()))

    def test_bn_fold_to_next_conv_layer(self):
        np.random.seed(1)
        total = 4 * 2 * 3 * 1

        weight = np.array(np.random.randn(total))
        weight_sz = np.array([4, 2, 3, 1])

        beta = np.array(np.random.randn(2))
        gamma = np.array(np.random.randn(2))
        running_mean = np.array(np.random.randn(2))
        running_var = np.array(np.random.randn(2))

        bn_params = libpymo.BNParams()
        bn_params.beta = beta
        bn_params.gamma = gamma
        bn_params.runningMean = running_mean
        bn_params.runningVar = running_var

        layer_weight_params = libpymo.TensorParams()
        layer_weight_params.data = weight
        layer_weight_params.shape = weight_sz

        random_bias = np.array(np.random.rand(4))
        bias_tensor = libpymo.TensorParams()
        bias_tensor.data = random_bias
        bias_tensor.shape = np.array([4])

        w, b = bn_fold_next_layer(weight.reshape(weight_sz), random_bias,
                                  beta, gamma, running_mean, running_var)
        bias = libpymo.fold(bn_params, layer_weight_params, bias_tensor, True, False)

        print(w)
        print("*******")
        print(layer_weight_params.data)
        assert (np.allclose(w.flatten(), layer_weight_params.data))
        assert (np.allclose(b.flatten(), bias))

    def test_bn_fold_to_next_linear_layer(self):
        np.random.seed(1)
        total = 4 * 2

        weight = np.array(np.random.randn(total))
        weight_sz = np.array([4, 2, 1, 1])

        beta = np.array(np.random.randn(2))
        gamma = np.array(np.random.randn(2))
        running_mean = np.array(np.random.randn(2))
        running_var = np.array(np.random.randn(2))

        bn_params = libpymo.BNParams()
        bn_params.beta = beta
        bn_params.gamma = gamma
        bn_params.runningMean = running_mean
        bn_params.runningVar = running_var

        layer_weight_params = libpymo.TensorParams()
        layer_weight_params.data = weight
        layer_weight_params.shape = weight_sz

        random_bias = np.array(np.random.rand(4))
        bias_tensor = libpymo.TensorParams()
        bias_tensor.data = random_bias
        bias_tensor.shape = np.array([4])

        w, b = bn_fold_next_layer(weight.reshape(weight_sz), random_bias,
                                  beta, gamma, running_mean, running_var)
        bias = libpymo.fold(bn_params, layer_weight_params, bias_tensor, True, False)

        assert (np.allclose(w.flatten(), layer_weight_params.data))
        assert (np.allclose(b.flatten(), bias))


def bn_fold_prev_layer(weight, bias, beta, gamma, mu, sigma):
    scale = gamma / sigma
    b = beta - (mu - bias) * scale
    w = copy.deepcopy(weight)
    for i in range(len(beta)):
        w[i, :, :, :] = w[i, :, :, :] * scale[i]

    return w, b


def bn_fold_next_layer(weight, bias, beta, gamma, mu, sigma):
    w = copy.deepcopy(weight)
    weight_matrix = weight.sum(3).sum(2)
    mu_hat = np.matmul(weight_matrix, mu * gamma / sigma)
    beta_hat = np.matmul(weight_matrix, beta)
    for i in range(len(beta)):
        w[:, i, :, :] = w[:, i, :, :] * gamma[i] / sigma[i]

    b = beta_hat - mu_hat + bias
    return w, b
