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
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper

# Uncomment below to run unit tests. Cannot include always since it will affect all other tensorflow unit tests.
# tf.compat.v1.enable_eager_execution()

def dense_relu_functional():
    inp = tf.keras.layers.Input(shape=(5,))
    x = tf.keras.layers.Dense(units=2)(inp)
    x= tf.keras.layers.ReLU()(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name="dense_relu_functional")
    return model

def dense_relu_sequential():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=2, input_shape=(5,)))
    model.add(tf.keras.layers.ReLU())
    return model

class DenseReluSubclassing(tf.keras.Model):
    def __init__(self):
        super(DenseReluSubclassing, self).__init__()
        self.linear1 = tf.keras.layers.Dense(units=2)
        self.relu1 = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, mask=None):
        x = self.linear1(inputs)
        x = self.relu1(x)
        return x

@pytest.mark.skip(reason="Only run this test if eager execution is enabled")
def test_qc_quantize_wrapper():
    """
    Test basic functionality of qc quantize wrapper
    """
    inp = np.random.randn(1, 8)
    rand_weights = np.random.randn(8, 2)

    dense_layer = tf.keras.layers.Dense(units=2, activation='relu')
    # Run one forward pass to initialize weights
    dense_layer(inp)
    dense_layer.set_weights([rand_weights, np.array([1.2, 1.5])])
    wrapped_layer = QcQuantizeWrapper(dense_layer)
    orig_weights = [weight for weight in dense_layer.get_weights()]
    dense_out = dense_layer(inp)
    wrapped_out = wrapped_layer(inp)
    end_weights = [weight for weight in wrapped_layer.get_weights()]
    for idx, _ in enumerate(orig_weights):
        assert np.array_equal(orig_weights[idx], end_weights[idx])
    assert not np.array_equal(dense_out, wrapped_out)
