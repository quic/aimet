# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""
Unit test for separable conv to depthwise pointwise replacement
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from aimet_tensorflow.keras.utils.model_transform_utils import replace_separable_conv_with_depthwise_pointwise

def _simple_separable_conv_model(model_type="functional"):
    if model_type == "functional":
        inp = layers.Input((32, 32, 3))
        x = layers.SeparableConv2D(filters=32, kernel_size=3)(inp)
        x = layers.ReLU()(x)
        x = layers.SeparableConv2D(filters=5, kernel_size=3, activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=10, activation="relu")(x)
        out = layers.Dense(units=2, activation="softmax")(x)
        return tf.keras.Model(inp, out)
    elif model_type == "sequential":
        model = tf.keras.Sequential()
        model.add(layers.SeparableConv2D(filters=32, kernel_size=2, input_shape=(32, 32, 3)))
        model.add(layers.ReLU())
        model.add(layers.SeparableConv2D(filters=5, kernel_size=2, activation="relu"))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=10, activation="relu"))
        model.add(layers.Dense(units=2, activation="softmax"))
        return model

    raise ValueError("Unknown model type")

def _get_layers(model, model_type="functional"):
    if model_type == "functional":
        return model.layers
    elif model_type == "sequential":
        return [None, *model.layers]

    raise ValueError("Unknown model type")

class TestSeparableConvReplacement:
    """
    Test class for separable conv to depthwise pointwise replacement
    """
    @pytest.mark.parametrize("model_type", ["functional", "sequential"])
    def test_separable_conv_replacement(self, model_type):
        """
        Test separable conv to depthwise pointwise replacement
        """

        original_model = _simple_separable_conv_model(model_type)
        inp = tf.random.uniform((1, *original_model.input_shape[1:]))
        original_model_out = original_model(inp)

        transformed_model, _ = replace_separable_conv_with_depthwise_pointwise(original_model)

        depthwise_conv1, pointwise_conv1 = _get_layers(transformed_model, model_type)[1:3]
        depthwise_conv2, pointwise_conv2 = _get_layers(transformed_model, model_type)[4:6]

        assert isinstance(depthwise_conv1, layers.DepthwiseConv2D)
        assert isinstance(pointwise_conv1, layers.Conv2D)

        assert isinstance(depthwise_conv2, layers.DepthwiseConv2D)
        assert isinstance(pointwise_conv2, layers.Conv2D)

        # Check that the weights are the same
        assert depthwise_conv1.get_weights()[0].shape == _get_layers(original_model, model_type)[1].get_weights()[0].shape
        assert pointwise_conv1.get_weights()[0].shape == _get_layers(original_model, model_type)[1].get_weights()[1].shape

        assert depthwise_conv2.get_weights()[0].shape == _get_layers(original_model, model_type)[3].get_weights()[0].shape
        assert pointwise_conv2.get_weights()[0].shape == _get_layers(original_model, model_type)[3].get_weights()[1].shape

        # Check that the bias is the same
        assert pointwise_conv1.get_weights()[1].shape == (32,)

        assert pointwise_conv2.get_weights()[1].shape == (5,)

        # Check that the weights are the same
        assert np.array_equal(depthwise_conv1.get_weights()[0], _get_layers(original_model, model_type)[1].get_weights()[0]), \
            "Depthwise 1 kernel weights are not the same"
        assert np.array_equal(pointwise_conv1.get_weights()[0], _get_layers(original_model, model_type)[1].get_weights()[1]), \
            "Pointwise 1 kernel weights are not the same"
        assert np.array_equal(pointwise_conv1.get_weights()[1], _get_layers(original_model, model_type)[1].get_weights()[2]), \
            "Pointwise 1 bias weights are not the same"

        assert np.array_equal(depthwise_conv2.get_weights()[0], _get_layers(original_model, model_type)[3].get_weights()[0]), \
            "Depthwise 2 kernel weights are not the same"
        assert np.array_equal(pointwise_conv2.get_weights()[0], _get_layers(original_model, model_type)[3].get_weights()[1]), \
            "Pointwise 2 kernel weights are not the same"
        assert np.array_equal(pointwise_conv2.get_weights()[1], _get_layers(original_model, model_type)[3].get_weights()[2]), \
            "Pointwise 2 bias weights are not the same"

        # Check that the output is the same
        transformed_model_out = transformed_model(inp)
        assert np.array_equal(original_model_out, transformed_model_out)

        # Check that number of parameters is the same
        assert original_model.count_params() == transformed_model.count_params()
