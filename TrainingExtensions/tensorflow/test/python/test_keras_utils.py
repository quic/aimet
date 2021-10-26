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

""" Unit tests for keras utils """

import os

import pytest
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from aimet_tensorflow.keras.utils.common import replace_layer_for_non_subclassed_model

def dense_relu_functional():
    inp = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Dense(units=1)(inp)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ReLU()(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name="dense_relu_functional")
    return model

@pytest.mark.skip(reason="Only run this test if eager execution is enabled")
def test_replace_layer():
    inp = np.array([[1, 2]])
    model = dense_relu_functional()
    new_layer = tf.keras.layers.Dense(units=2)
    old_relu = model.layers[2]
    replace_layer_for_non_subclassed_model(model, old_relu, new_layer)
    tf.keras.models.save_model(model, './data/saved_model')
    new_model = tf.keras.models.load_model('./data/saved_model')
    out = new_model(inp)
    assert out.shape == [1, 2]
    assert isinstance(new_model.layers[2], tf.keras.layers.Dense)
    assert new_model.layers[1].outbound_nodes[0] == new_model.layers[2].inbound_nodes[0]
    assert new_model.layers[2].outbound_nodes[0] == new_model.layers[3].inbound_nodes[0]
    assert new_model.layers[4].inbound_nodes == []
    assert new_model.layers[4].outbound_nodes == []
