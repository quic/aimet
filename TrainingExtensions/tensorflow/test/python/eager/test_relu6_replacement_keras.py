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
"""
Unit test about ReLU6 replacement
"""
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import layers

from aimet_tensorflow.keras.utils.model_transform_utils import replace_relu6_with_relu


def _simple_conv_model(model_type="functional"):
    if model_type == "functional":
        inp = layers.Input((32, 32, 3))
        x = layers.Conv2D(filters=32, kernel_size=2)(inp)
        x = layers.ReLU(max_value=6.0)(x)
        x = layers.Conv2D(filters=32, kernel_size=2, activation=tf.nn.relu6)(x)
        x = layers.Activation(tf.nn.relu6)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=10, activation=tf.nn.relu6)(x)
        out = layers.Dense(units=2, activation=tf.nn.softmax)(x)
        return tf.keras.Model(inp, out)
    elif model_type == "sequential":
        return tf.keras.Sequential(
            [
                layers.Conv2D(filters=32, kernel_size=2, input_shape=(32, 32, 3)),
                layers.ReLU(max_value=6.0),
                layers.Conv2D(filters=32, kernel_size=2, activation=tf.nn.relu6),
                layers.Activation(tf.nn.relu6),
                layers.Flatten(),
                layers.Dense(units=10, activation=tf.nn.relu6),
                layers.Dense(units=2, activation=tf.nn.softmax)
            ]
        )


def _get_layers(model, model_type="functional"):
    # Drop first layer (Input layer) of Functional model
    if model_type == "functional":
        return model.layers[1:]
    elif model_type == "sequential":
        return model.layers


class TestReluReplacement:
    @pytest.mark.parametrize("model_type", ["sequential", "functional"])
    def test_relu6_replacement_remain_fusing(self, model_type):
        model = _simple_conv_model(model_type)

        transformed_model, _ = replace_relu6_with_relu(model, True)
        conv1, activation1, conv2, activation2, _, dense1, dense2 = _get_layers(
            transformed_model, model_type
        )

        # conv1 doesn't change because it's not a pattern that fits (Conv2D + ReLU6)
        assert conv1.activation == tf.keras.activations.linear

        # activation1.max_value is transformed from 6 to None
        assert activation1.max_value is None

        # fused Conv2D(filters=..., activation=tf.nn.relu6) is transformed to
        #   Conv2D(filters=..., activation=tf.nn.relu')
        assert conv2.activation == tf.keras.activations.relu

        # activation2.max_value is transformed from 6 to None
        assert activation2.max_value is None

        # dense1.activation is transformed from relu6 to linear
        assert dense1.activation == tf.keras.activations.relu

        # dense2.activation shouldn't be transformed
        assert dense2.activation == tf.keras.activations.softmax
        assert dense2.activation != tf.keras.activations.linear

        # Even If the layer of the model changes, the weights should not change
        for original_layer, transformed_layer in zip(model.layers, transformed_model.layers):
            if isinstance(original_layer, (layers.Conv2D, layers.Dense)):
                original_weight, original_bias = original_layer.get_weights()
                transformed_weight, transformed_bias = transformed_layer.get_weights()

                assert np.array_equal(original_weight, transformed_weight)
                assert np.array_equal(original_bias, transformed_bias)

    @pytest.mark.parametrize("model_type", ["sequential", "functional"])
    def test_relu6_replacement_separate_fusing(self, model_type):
        model = _simple_conv_model(model_type)

        transformed_model, _ = replace_relu6_with_relu(model, False)

        # Conv2D(filters=..., activation=tf.nn.relu6) and Dense(units=..., activation=tf.nn.relu6) is separated to
        # 1-1) Conv2D(filters=..., activation='linear') and 1-2) ReLU()
        # 2-1) Dense(units=..., activation='linear') and 2-2) ReLU()
        assert len(model.layers) + 2 == len(transformed_model.layers)

        # relu1 is newly added from fused Conv2D
        conv1, activation1, conv2, relu1, activation2, _, dense1, relu2, dense2 = _get_layers(
            transformed_model, model_type
        )
        assert conv1.activation == tf.keras.activations.linear

        # activation1.max_value is transformed from 6 to None
        assert activation1.max_value is None

        # conv2.activation is transformed from relu6 to linear
        # relu1 is newly added from conv2
        assert conv2.activation == tf.keras.activations.linear
        assert relu1.max_value is None

        # activation2.max_value is transformed from 6 to None
        assert activation2.max_value is None

        # dense1.activation is transformed from relu6 to linear
        # relu2 is newly added from conv2
        assert dense1.activation == tf.keras.activations.linear
        assert relu2.max_value is None

        # dense2.activation shouldn't be transformed
        assert dense2.activation == tf.keras.activations.softmax
        assert dense2.activation != tf.keras.activations.linear

        # Even If the fused layer of the model separates, the weights should not change
        original_model_layers = _get_layers(model, model_type)
        transformed_model_layers = _get_layers(transformed_model, model_type)

        origin_conv2_layer = original_model_layers[2]
        transformed_conv2_layer = transformed_model_layers[2]

        assert np.array_equal(origin_conv2_layer.get_weights()[0], transformed_conv2_layer.get_weights()[0])
        assert np.array_equal(origin_conv2_layer.get_weights()[1], transformed_conv2_layer.get_weights()[1])

        origin_dense1_layer = original_model_layers[5]
        transformed_dense1_layer = transformed_model_layers[6]

        assert np.array_equal(origin_dense1_layer.get_weights()[0], transformed_dense1_layer.get_weights()[0])
        assert np.array_equal(origin_dense1_layer.get_weights()[1], transformed_dense1_layer.get_weights()[1])

