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

import aimet_common.libpymo as libpymo


class TestHighBiasFold(unittest.TestCase):

    def test_high_bias_fold(self):
        # Generating random numbers from a normal distribution for the weights and biases of the current and prev layer
        np.random.seed(1)
        total = 2 * 2 * 2 * 2

        weight = np.array(np.random.randn(total))
        weight_sz = np.array([2, 2, 2, 2])

        prev_layer_bias = np.array(np.random.randn(2))
        gamma = np.array(np.random.randn(2))
        beta = np.array(np.random.randn(2))
        curr_layer_bias = np.array(np.random.randn(2))

        prev_layer_params = libpymo.LayerParams()
        curr_layer_params = libpymo.LayerParams()
        prev_layer_bn_params = libpymo.BNParamsHighBiasFold()

        prev_layer_params.bias = prev_layer_bias
        prev_layer_params.activationIsRelu = True

        prev_layer_bn_params.gamma = gamma
        prev_layer_bn_params.beta = beta

        curr_layer_params.bias = curr_layer_bias
        curr_layer_params.weight = weight
        curr_layer_params.weightShape = weight_sz

        b_i_1, b_i = fold_high_bias_next_conv_qt(True, beta, gamma,
                                                 weight.reshape(weight_sz), curr_layer_bias,
                                                 prev_layer_bias)
        libpymo.updateBias(prev_layer_params, curr_layer_params, prev_layer_bn_params)
        assert (np.allclose(b_i_1, prev_layer_params.bias))
        assert (np.allclose(b_i, curr_layer_params.bias))

    def test_high_bias_fold_depthwise_layer(self):
        # Generating random numbers from a normal distribution for the weights and biases of the current and prev layer
        np.random.seed(1)
        total = 2 * 1 * 2 * 2

        weight = np.array(np.random.randn(total))
        weight_sz = np.array([2, 1, 2, 2])

        prev_layer_bias = np.array(np.random.randn(2))
        gamma = np.array(np.random.randn(2))
        beta = np.array(np.random.randn(2))
        curr_layer_bias = np.array(np.random.randn(2))

        prev_layer_params = libpymo.LayerParams()
        curr_layer_params = libpymo.LayerParams()
        prev_layer_bn_params = libpymo.BNParamsHighBiasFold()

        prev_layer_params.bias = prev_layer_bias
        prev_layer_params.activationIsRelu = True

        prev_layer_bn_params.gamma = gamma
        prev_layer_bn_params.beta = beta

        curr_layer_params.bias = curr_layer_bias
        curr_layer_params.weight = weight
        curr_layer_params.weightShape = weight_sz

        b_i_1, b_i = fold_high_bias_next_conv_qt(True, beta, gamma,
                                                 weight.reshape(weight_sz), curr_layer_bias,
                                                 prev_layer_bias)
        libpymo.updateBias(prev_layer_params, curr_layer_params, prev_layer_bn_params)
        assert (np.allclose(b_i_1, prev_layer_params.bias))
        assert (np.allclose(b_i, curr_layer_params.bias))


def fold_high_bias_next_conv_qt(activation_is_relu, beta, gamma, weight, bias_curr_layer, bias_prev_layer):
    curr_layer_bias = bias_curr_layer
    prev_layer_bias = bias_prev_layer
    if not activation_is_relu:
        # No activation function, absorbe whole bias
        absorb_bias = beta
    else:
        # Only absorb bias part that is more than 'min_std' standard deviations
        abs_gamma = np.abs(gamma)
        absorb_bias = np.maximum(0, beta - 3 * abs_gamma)
    # Calculate correction term for next layer
    weight_matrix = weight.sum(3).sum(2)
    if weight_matrix.shape[1] == 1:
        weight_matrix1 = weight_matrix.reshape(weight_matrix.shape[0])
        bias_correction = np.multiply(weight_matrix1, absorb_bias)
    else:
        bias_correction = np.matmul(weight_matrix, absorb_bias)

    # Update next layers
    curr_layer_bias += bias_correction
    prev_layer_bias = prev_layer_bias - absorb_bias
    return prev_layer_bias, curr_layer_bias
