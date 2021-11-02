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
import pytest
import tensorflow as tf

from aimet_tensorflow.keras.cross_layer_equalization import GraphSearchUtils


class TestTrainingExtensionsCrossLayerScaling:

    def test_find_layer_groups_in_vgg16(self):
        """
        VGG16 is a typical sequential structure
        Input -> Conv -> Conv -> Pooling -> Conv -> Conv -> Pooling
               -> Conv -> Conv -> Conv -> Pooling
               -> Conv -> Conv -> Conv -> Pooling
               -> Conv -> Conv -> Conv -> Pooling
               -> Dense -> Dense -> Dense
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=2),
            tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=2),
            tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=2),
            tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=2),
            tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10)
        ])

        # TODO: Implement layer group searching logic and replace mock_layer_groups with its result
        layers = model.layers
        mock_layer_groups = [
            [layers[0], layers[1]],
            [layers[3], layers[4]],
            [layers[6], layers[7], layers[8]],
            [layers[10], layers[11], layers[12]],
            [layers[14], layers[15], layers[16]]
        ]
        assert len(mock_layer_groups) == 5

    def test_find_layer_groups_in_network_with_residual(self):
        """
        Sample network that has residual connection (branching)
        Input -> Conv -> BN -> ReLU -> MaxPool ->
               -> (In Residual) Conv -> BN -> ReLU -> Conv -> BN -> ReLU
               -> (In Residual) Conv -> BN -> ReLU -> Conv -> BN -> ReLU
               -> Dense
        """

        class Residual(tf.keras.Model):  # pylint: disable=too-many-ancestors
            def __init__(self):
                super().__init__()
                self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3)
                self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=3)

                self.bn1 = tf.keras.layers.BatchNormalization()
                self.bn2 = tf.keras.layers.BatchNormalization()

                self.relu1 = tf.keras.layers.ReLU()
                self.relu2 = tf.keras.layers.ReLU()

            def call(self, inputs, training=None, mask=None):
                outputs = self.relu1(self.bn1(self.conv1))
                outputs = self.bn2(self.conv2(outputs))

                outputs += inputs
                return self.relu2(outputs)

            def get_config(self):
                pass

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=7),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=3),
            Residual(),
            Residual(),
            tf.keras.layers.Dense(units=10)
        ])

        # TODO: Implement layer group searching logic and replace mock_layer_groups with its result
        #   Also need to implement graph parsing in future,
        #   model.layers can't guarantee actual ordering and structure
        layers = model.layers
        mock_layer_groups = [
            [layers[4].conv1, layers[4].conv2],
            [layers[5].conv1, layers[5].conv2]
        ]
        assert len(mock_layer_groups) == 2

    def test_find_layer_groups_in_network_with_multiple_inputs(self):
        """
        Sample network that has multiple inbounds
        Left Input -> Conv -> Conv -> MaxPool -> Conv -> Conv -> Flatten
        Right Input -> Conv -> Conv -> Conv -> Flatten
        Output -> Concat([Left Output, Right Output])
        """

        def generate_model():
            x1 = tf.keras.layers.Input(shape=(28, 28, 1))
            left_input = tf.keras.layers.Conv2D(16, 3, activation="relu")(x1)
            left_input = tf.keras.layers.Conv2D(32, 3, activation="relu")(left_input)
            left_input = tf.keras.layers.MaxPool2D(3)(left_input)
            left_input = tf.keras.layers.Conv2D(32, 3, activation="relu")(left_input)
            left_input = tf.keras.layers.Conv2D(16, 3, activation="relu")(left_input)
            left_input = tf.keras.layers.Flatten()(left_input)

            x2 = tf.keras.layers.Input(shape=(28, 28, 1))
            right_input = tf.keras.layers.Conv2D(64, 3, activation="relu")(x2)
            right_input = tf.keras.layers.Conv2D(32, 3, activation="relu")(right_input)
            right_input = tf.keras.layers.Conv2D(16, 3, activation="relu")(right_input)
            right_input = tf.keras.layers.Flatten()(right_input)

            y = tf.keras.layers.Concatenate()([left_input, right_input])
            return tf.keras.models.Model([x1, x2], y)

        model = generate_model()
        # TODO: Implement layer group searching logic and replace mock_layer_groups with its result
        #   Also need to implement graph parsing in future,
        #   model.layers can't guarantee actual ordering and structure
        layers = model.layers
        mock_layer_groups = [
            [layers[1], layers[2]],
            [layers[5], layers[6]],
            [layers[7], layers[8], layers[9]]
        ]

        assert len(mock_layer_groups) == 3

    def test_convert_layer_group_to_cls_sets_for_consecutive_convolution(self):
        """
        Layer group (Consecutive Convolution)
        """

        conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3)
        conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5)
        conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=5)
        conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=7)
        layer_group = [conv1, conv2, conv3, conv4]

        actual = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
        expected = [(conv1, conv2), (conv2, conv3), (conv3, conv4)]

        assert actual == expected

    def test_convert_layer_group_to_cls_sets_for_depthwise_separable_convolution(self):
        """
        Layer group (Typical Depthwise Separable Convolution)
        """

        conv = tf.keras.layers.Conv2D(filters=8, kernel_size=3)
        depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3)
        pointwise_conv = tf.keras.layers.Conv2D(filters=16, kernel_size=1)
        layer_group = [conv, depthwise_conv, pointwise_conv]

        actual = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
        expected = [(conv, depthwise_conv, pointwise_conv)]

        assert actual == expected

    def test_convert_layer_group_to_cls_sets_for_part_of_mobile_net_v1(self):
        """
        Layer group from the earlier part of MobileNetV1
        """

        conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same")
        depthwise_conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=32, strides=1, padding="same")
        pointwise_conv1 = tf.keras.layers.Conv2D(64, kernel_size=1, strides=1, padding="same")
        depthwise_conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=64, strides=2, padding="same")
        pointwise_conv2 = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same")
        depthwise_conv3 = tf.keras.layers.DepthwiseConv2D(kernel_size=128, padding="same")
        pointwise_conv3 = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same")

        layer_group = [
            conv1,
            depthwise_conv1, pointwise_conv1,
            depthwise_conv2, pointwise_conv2,
            depthwise_conv3, pointwise_conv3
        ]

        actual = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
        expected = [
            (conv1, depthwise_conv1, pointwise_conv1),
            (pointwise_conv1, depthwise_conv2, pointwise_conv2),
            (pointwise_conv2, depthwise_conv3, pointwise_conv3)
        ]

        assert actual == expected

    def test_convert_layer_group_to_cls_sets_for_consecutive_depthwise_convolution(self):
        """
        Layer group (Consecutive Depthwise Separable Convolution)
        """

        conv = tf.keras.layers.Conv2D(filters=8, kernel_size=3)
        depthwise_conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3)
        depthwise_conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=5)
        layer_group = [conv, depthwise_conv1, depthwise_conv2]

        with pytest.raises(NotImplementedError):
            _ = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
