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

# import unittest
import pytest

import numpy as np
from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.layer_database import *
from aimet_tensorflow.keras.svd_spiltter import SpatialSvdModuleSplitter
from aimet_tensorflow.keras.svd_pruner import SpatialSvdPruner

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)

def get_model(model_type = "Sequential"):
    tf.keras.backend.clear_session()
    if model_type == "Sequential":
        return  tf.keras.Sequential([
            tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28 * 28,)),
            tf.keras.layers.Conv2D(32, 5, strides=(2, 2), name='conv1', padding='same'),
            tf.keras.layers.Conv2D(64, 32, name='conv2', padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, name='linear')
        ])
    elif model_type == "Functional":
        inp = tf.keras.Input((28*28))
        x = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(inp)
        x = tf.keras.layers.Conv2D(32, 5, strides=(2, 2), name='conv1', padding='same')(x)
        x = tf.keras.layers.Conv2D(64, 32, name='conv2', padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        out = tf.keras.layers.Dense(10, name = 'linear')(x)

        return tf.keras.Model(inp, out)

def _get_layers(model, model_type="Sequential"):
    # Drop first layer (Input layer) of Functional model
    if model_type == "Functional":
        return model.layers[1:]
    elif model_type == "Sequential":
        return model.layers

class TestSpatialSvdLayerSplitandSVDPrunner:


    @pytest.mark.parametrize("model_type", ["Sequential", "Functional"])
    @pytest.mark.parametrize("rank", [1024, 512])
    def test_split_layer(self, model_type, rank):
        """
        test the output after and before the split_module call
        """
        model = get_model(model_type)
        orig_conv_op = _get_layers(model, model_type)[2]

        org_conv_op_shape = orig_conv_op.output_shape

        layer1 = Layer(orig_conv_op, orig_conv_op.name, output_shape=org_conv_op_shape)

        split_conv_op1, split_conv_op2 = SpatialSvdModuleSplitter.split_module(model, layer=layer1, rank=rank)

        split_conv_output = split_conv_op2.output_shape

        assert org_conv_op_shape == split_conv_output

        # check the bias value after split.

        assert len(split_conv_op2.get_weights()) == len(orig_conv_op.get_weights())
        if len(orig_conv_op.get_weights()) > 1:

            orig_bias_out = orig_conv_op.get_weights()[1]
            split_bias_out = split_conv_op2.get_weights()[1]

            assert np.allclose(orig_bias_out, split_bias_out, atol=1e-4)

        # First split conv op should not have bias
        assert len(split_conv_op1.get_weights()) == 1

    @pytest.mark.parametrize("model_type", ["Sequential", "Functional"])
    def test_split_layer_with_stride(self, model_type):
        """
        test the conv2d split after and before split_module call with stride
        """
        model = get_model(model_type)
        orig_conv_op = _get_layers(model, model_type)[1]

        org_conv_op_shape = orig_conv_op.output_shape

        layer1 = Layer(orig_conv_op, orig_conv_op.name, output_shape=org_conv_op_shape)

        split_conv_op1, split_conv_op2 = SpatialSvdModuleSplitter.split_module(model, layer=layer1, rank=5)

        split_conv_output = split_conv_op2.output_shape

        assert org_conv_op_shape == split_conv_output

        # check the bias value after split.

        assert len(split_conv_op2.get_weights()) == len(orig_conv_op.get_weights())
        if len(orig_conv_op.get_weights()) > 1:

            orig_bias_out = orig_conv_op.get_weights()[1]
            split_bias_out = split_conv_op2.get_weights()[1]

            assert np.allclose(orig_bias_out, split_bias_out, atol=1e-4)

        # First split conv op should not have bias
        assert len(split_conv_op1.get_weights()) == 1


    @pytest.mark.parametrize("model_type", ["Sequential", "Functional"])
    def test_perform_svd_and_split_layer(self, model_type):

        model = get_model(model_type)
        layer_db = LayerDatabase(model)
        layer = layer_db.find_layer_by_name(_get_layers(model, model_type)[2].name)
        org_count = len(list(layer_db._compressible_layers.values()))
        splitter = SpatialSvdPruner()
        splitter._perform_svd_and_split_layer(layer, 1024, layer_db)

        assert layer not in list(layer_db._compressible_layers.values())

        after_split_count = len(list(layer_db._compressible_layers.values()))
        assert (org_count + 1) == after_split_count

