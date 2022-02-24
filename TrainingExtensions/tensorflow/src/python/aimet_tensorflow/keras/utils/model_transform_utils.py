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
Classes and utilities to replace ReLU6 with ReLU
"""
import typing

import tensorflow as tf
from tensorflow.keras import layers
from packaging import version

# Remove version check when we upgrade to tf 2.0
if version.parse(tf.version.VERSION) >= version.parse("2.00"):
    # pylint: disable=no-name-in-module
    from tensorflow_model_optimization.python.core.api.quantization.keras.graph_transformations import (
        transforms,
        model_transformer,
    )


    def _create_layer_by_config(
            class_name: str, layer_config: typing.Dict
    ) -> tf.keras.layers.Layer:
        """
        Factory method to create layer by class name and its configuration

        :param class_name: Class name inherited from tf.keras.layers.Layer
        :param layer_config: Layer configuration dictionary
        :return: Return a layer that meets the condition or raise exception
        """
        if class_name == "Conv2D":
            return layers.Conv2D(**layer_config)
        if class_name == "Dense":
            return layers.Dense(**layer_config)

        # Raise exception for the case of not supported layers
        raise NotImplementedError


    class ReplaceRelu6WithRelu(transforms.Transform):
        """
        Transform class for the case like tf.keras.layers.ReLU(max_value=6)
        """

        def pattern(self):
            """
            Pattern to transform
            """
            return transforms.LayerPattern("ReLU", {"max_value": 6.0})

        def replacement(self, match_layer):
            """
            Replacement logic for match_layer
            :param match_layer: matched layer by pattern
            """
            replace_layer = layers.serialize(layers.ReLU())
            replace_layer["name"] = replace_layer["config"]["name"]
            return transforms.LayerNode(replace_layer)


    class ReplaceActivationWithRelu(transforms.Transform):
        """
        Transform class for the case like tf.keras.layers.Activation(tf.nn.relu6)
        """

        def pattern(self):
            """
            Pattern to transform
            """
            return transforms.LayerPattern("Activation", {"activation": "relu6"})

        def replacement(self, match_layer):
            """
            Replacement logic for match_layer
            :param match_layer: matched layer by pattern
            """
            replace_layer = layers.serialize(layers.ReLU())
            replace_layer["name"] = replace_layer["config"]["name"]
            return transforms.LayerNode(replace_layer)


    class ReplaceFusedRelu6WithFusedRelu(transforms.Transform):
        """
        Transform class for the case like tf.keras.layers.Conv2D(..., activation=tf.nn.relu6)
        Result is still fused such as tf.keras.layers.Conv2D(..., activation=tf.nn.relu)
        """
        def __init__(self, class_name):
            self.class_name = class_name

        def pattern(self):
            """
            Pattern to transform
            """
            return transforms.LayerPattern(self.class_name, {"activation": "relu6"})

        def replacement(self, match_layer):
            """
            Replacement logic for match_layer
            :param match_layer: matched layer by pattern
            """
            match_layer_config = match_layer.layer["config"]
            match_layer_config["activation"] = "relu"

            replace_layer = _create_layer_by_config(self.class_name, match_layer_config)
            replace_layer_config = layers.serialize(replace_layer)
            replace_layer_config["name"] = replace_layer_config["config"]["name"]

            return transforms.LayerNode(replace_layer_config, match_layer.weights)


    class ReplaceFusedRelu6WithSeparateLayers(transforms.Transform):
        """
        Transform class for the case like tf.keras.layers.Conv2D(..., activation=tf.nn.relu6)
        Result is separated such as layers.Conv2D(..., activation='linear') and layers.ReLU()
        """
        def __init__(self, class_name):
            self.class_name = class_name

        def pattern(self):
            """
            Pattern to transform
            """
            return transforms.LayerPattern(self.class_name, {"activation": "relu6"})

        def replacement(self, match_layer):
            """
            Replacement logic for match_layer
            :param match_layer: matched layer by pattern
            """
            activation_layer = layers.ReLU()
            activation_layer_config = layers.serialize(activation_layer)
            activation_layer_config["name"] = activation_layer.name

            match_layer_config = match_layer.layer["config"]
            match_layer_config["activation"] = "linear"

            replace_layer = _create_layer_by_config(self.class_name, match_layer_config)
            replace_layer = layers.serialize(replace_layer)
            replace_layer["name"] = replace_layer["config"]["name"]

            return transforms.LayerNode(
                activation_layer_config,
                input_layers=[transforms.LayerNode(replace_layer, match_layer.weights)]
            )


    def replace_relu6_with_relu(
            model: tf.keras.Model, remain_fusing: bool = False
    ) -> typing.Tuple[tf.keras.Model, typing.Dict]:
        """
        Replace ReLU6 with ReLU in tf.keras.Model

        :param model: tf.keras.Model
        :param remain_fusing:
            If remain_fusing is True, Fused Conv2D remained fused Conv2D
            e.g., Conv2D(activation="relu6") -> Conv2D(activation="relu")

            If remain_fusing is False, Fused Conv2D is separated to Conv2D and ReLU
            e.g., Conv2D(activation="relu6") -> Conv2D(activation="linear") and ReLU()
        """
        if remain_fusing:
            transform_list = [
                ReplaceRelu6WithRelu(),
                ReplaceActivationWithRelu(),
                ReplaceFusedRelu6WithFusedRelu("Conv2D"),
                ReplaceFusedRelu6WithFusedRelu("Dense")
            ]
        else:
            transform_list = [
                ReplaceRelu6WithRelu(),
                ReplaceActivationWithRelu(),
                ReplaceFusedRelu6WithSeparateLayers("Conv2D"),
                ReplaceFusedRelu6WithSeparateLayers("Dense"),
            ]

        return model_transformer.ModelTransformer(model, transform_list).transform()
