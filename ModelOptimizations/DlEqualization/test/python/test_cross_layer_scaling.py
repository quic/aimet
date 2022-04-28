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


class TestCrossLayerScaling(unittest.TestCase):

    def test_cross_layer_scaling_equalize_params(self):
        print("starting python model optimization cross layer scaling test for non depthwise")

        # Generating random numbers from a normal distribution for the weights and biases of the current and prev layer
        np.random.seed(1)
        total = 2 * 3 * 2 * 2
        weight1 = np.array(np.random.randn(total))
        bias1 = np.array(np.random.randn(2))
        weight2 = np.array(np.random.randn(total))

        weight_sz1 = np.array([2, 3, 2, 2])
        weight_sz2 = np.array([3, 2, 2, 2])

        # Initializing the struct EqualizationParams
        prev_layer_params = libpymo.EqualizationParams()
        curr_layer_params = libpymo.EqualizationParams()

        prev_layer_params.weight = weight1
        prev_layer_params.weightShape = weight_sz1
        prev_layer_params.bias = bias1
        prev_layer_params.isBiasNone = False

        curr_layer_params.weight = weight2
        curr_layer_params.weightShape = weight_sz2

        w1, w2, b1, scale_factor = cross_layer_scaling_python_implementation(weight1.reshape(weight_sz1),
                                                                             weight2.reshape(weight_sz2),
                                                                             bias1)

        rescaling_vector = libpymo.scaleLayerParams(prev_layer_params, curr_layer_params)
        assert (np.allclose(w1.flatten(), prev_layer_params.weight))
        assert (np.allclose(w2.flatten(), curr_layer_params.weight))
        assert (np.allclose(b1, prev_layer_params.bias))
        assert (np.allclose(scale_factor, rescaling_vector))

    def test_cross_layer_scaling_equalize_params_depthwise(self):
        np.random.seed(1)
        weight1 = np.array(np.random.randn(2 * 2 * 3 * 2))
        bias1 = np.array(np.random.randn(2))
        weight2 = np.array(np.random.randn(2 * 2 * 4))
        bias2 = np.array(np.random.randn(2))
        weight3 = np.array(np.random.randn(2 * 2 * 4 * 5))

        weight_sz1 = np.array([2, 2, 3, 2])
        weight_sz2 = np.array([2, 2, 4, 1])
        weight_sz3 = np.array([2, 2, 4, 5])

        # Initializing the struct EqualizationParams
        prev_layer_params = libpymo.EqualizationParams()
        curr_layer_params = libpymo.EqualizationParams()
        next_layer_params = libpymo.EqualizationParams()

        prev_layer_params.weight = weight1
        prev_layer_params.weightShape = weight_sz1
        prev_layer_params.bias = bias1
        prev_layer_params.isBiasNone = False

        curr_layer_params.weight = weight2
        curr_layer_params.weightShape = weight_sz2
        curr_layer_params.bias = bias2
        curr_layer_params.isBiasNone = False

        next_layer_params.weight = weight3
        next_layer_params.weightShape = weight_sz3

        w1, w2, w3, b1, b2, s_12, s_23 = cross_layer_scaling_depthwise_separable_layers(weight1.reshape(weight_sz1),
                                                                                        weight2.reshape(weight_sz2),
                                                                                        weight3.reshape(weight_sz3),
                                                                                        bias1, bias2)

        scaling_params = libpymo.scaleDepthWiseSeparableLayer(prev_layer_params, curr_layer_params, next_layer_params)
        assert (np.allclose(w1.flatten(), prev_layer_params.weight))
        assert (np.allclose(w2.flatten(), curr_layer_params.weight))
        assert (np.allclose(w3.flatten(), next_layer_params.weight))
        assert (np.allclose(b1, prev_layer_params.bias))
        assert (np.allclose(b2, curr_layer_params.bias))
        assert (np.allclose(scaling_params.scalingMatrix12, s_12))
        assert (np.allclose(s_23, scaling_params.scalingMatrix23))


def cross_layer_scaling_python_implementation(weight1, weight2, bias1):
    w1 = weight1
    w2 = weight2

    b1 = bias1
    range1 = np.amax(np.abs(w1), axis=(1, 2, 3))
    range2 = np.amax(np.abs(w2), axis=(0, 2, 3))
    scale_factor = range1 / np.power(range1 * range2, 1. / 2)
    for i in range(len(scale_factor)):
        w1[i, :, :, :] = w1[i, :, :, :] * (1.0 / scale_factor[i])
        w2[:, i, :, :] = w2[:, i, :, :] * scale_factor[i]
        b1[i] = b1[i] * (1.0 / scale_factor[i])
    return w1, w2, b1, scale_factor


def cross_layer_scaling_depthwise_separable_layers(weight1, weight2, weight3, bias1, bias2):
    w1 = weight1
    w2 = weight2
    w3 = weight3
    b1 = bias1
    b2 = bias2
    range1 = np.amax(np.abs(w1), axis=(1, 2, 3))
    range2 = np.amax(np.abs(w2), axis=(1, 2, 3))
    range3 = np.amax(np.abs(w3), axis=(0, 2, 3))

    s_12 = range1 / np.power(range1 * range2 * range3, 1.0 / 3)
    s_23 = np.power(range1 * range2 * range3, 1.0 / 3) / range3

    for i in range(len(s_12)):
        w1[i, :, :, :] = w1[i, :, :, :] * (1.0 / s_12[i])
        w2[i, :, :, :] = w2[i, :, :, :] * s_12[i] * (1.0 / s_23[i])
        b1[i] = b1[i] * (1.0 / s_12[i])
        b2[i] = b2[i] * (1.0 / s_23[i])
        w3[:, i, :, :] = w3[:, i, :, :] * s_23[i]

    return w1, w2, w3, b1, b2, s_12, s_23
