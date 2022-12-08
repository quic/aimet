# /usr/bin/env python3.8
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

import unittest
from unittest.mock import MagicMock
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from packaging import version

from aimet_tensorflow.keras.layer_database import Layer
from aimet_tensorflow.keras.layer_selector import ConvFcLayerSelector, ConvNoDepthwiseLayerSelector

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)


def get_model():
    """
    Returns a model with two regular conv layers, one depthwise conv and one dense layer
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28 * 28,)),
        tf.keras.layers.Conv2D(32, 5, name='conv1', padding='same'),
        tf.keras.layers.Conv2D(64, 32, name='conv2', padding='same'),
        tf.keras.layers.DepthwiseConv2D(64, 64, name='conv3', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64 * 28 * 28, name='linear')
    ])
    return model

class TestLayerSelector(unittest.TestCase):

    def test_select_all_conv_layers(self):
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            tf.keras.backend.clear_session()

            # Load the  model to test layer selection
            model = get_model()

            conv1_op = model.layers[1]
            conv2_op = model.layers[2]
            conv3_op = model.layers[3]
            matmul1_op = model.layers[5]

            conv1_op_output_shape = conv1_op.output_shape
            conv2_op_output_shape = conv2_op.output_shape
            conv3_op_output_shape = conv3_op.output_shape
            matmul1_op_output_shape = matmul1_op.output_shape

            layer1 = Layer(conv1_op, conv1_op.name, output_shape=conv1_op_output_shape)
            layer2 = Layer(conv2_op, conv2_op.name, output_shape=conv2_op_output_shape)
            layer3 = Layer(conv3_op, conv3_op.name, output_shape=conv3_op_output_shape)
            layer4 = Layer(matmul1_op, matmul1_op.name, output_shape=matmul1_op_output_shape)

            layer_db = MagicMock()
            layer_db.__iter__.return_value = [layer1, layer2, layer3, layer4]

            layer_selector = ConvNoDepthwiseLayerSelector()
            layer_selector.select(layer_db, [])
            layer_db.mark_picked_layers.assert_called_once_with([layer1, layer2])

            # Two regular conv layers - one in ignore list
            layer_db.mark_picked_layers.reset_mock()
            layer_selector.select(layer_db, [layer2.module])
            layer_db.mark_picked_layers.assert_called_once_with([layer1])

    def test_select_all_conv_and_fc_layers(self):
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            tf.keras.backend.clear_session()

            # Load the  model to test layer selection
            model = get_model()

            conv1_op = model.layers[1]
            conv2_op = model.layers[2]
            conv3_op = model.layers[3]
            matmul1_op = model.layers[5]

            conv1_op_output_shape = conv1_op.output_shape
            conv2_op_output_shape = conv2_op.output_shape
            conv3_op_output_shape = conv3_op.output_shape
            matmul1_op_output_shape = matmul1_op.output_shape

            layer1 = Layer(conv1_op, conv1_op.name, output_shape=conv1_op_output_shape)
            layer2 = Layer(conv2_op, conv2_op.name, output_shape=conv2_op_output_shape)
            layer3 = Layer(conv3_op, conv3_op.name, output_shape=conv3_op_output_shape)
            layer4 = Layer(matmul1_op, matmul1_op.name, output_shape=matmul1_op_output_shape)

            layer_db = MagicMock()
            layer_db.__iter__.return_value = [layer1, layer2, layer3, layer4]

            layer_selector = ConvFcLayerSelector()
            layer_selector.select(layer_db, [])
            layer_db.mark_picked_layers.assert_called_once_with([layer1, layer2, layer4])

            # Two regular conv layers - one in ignore list
            layer_db.mark_picked_layers.reset_mock()
            layer_selector.select(layer_db, [layer2.module])
            layer_db.mark_picked_layers.assert_called_once_with([layer1, layer4])

