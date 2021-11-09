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
from aimet_tensorflow.keras.quant_sim.qc_quantize_layer import QcQuantizeLayer, QcQuantizeParamWrapper
from aimet_tensorflow.keras.utils.common import replace_layer_in_functional_model

# Uncomment below to run unit tests. Cannot include always since it will affect all other tensorflow unit tests.
# tf.compat.v1.enable_eager_execution()

def dense_functional():
    inp = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Dense(units=2)(inp)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name="dense_functional")
    return model

class DenseSubclassing(tf.keras.Model):
    def __init__(self):
        super(DenseSubclassing, self).__init__()
        self.linear1 = tf.keras.layers.Dense(units=2)

    def call(self, inputs, training=None, mask=None):
        x = self.linear1(inputs)
        return x

@pytest.mark.skip(reason="Enable with TF 2.4")
def test_qc_quantize_param_wrapper():
    """ Test qc quantize param wrapper """
    inp = np.abs(np.random.randn(100, 2))
    model = dense_functional()
    model.layers[1].set_weights([np.ones((2, 2)), np.array([1.2, 1.5])])
    orig_out = model.predict(inp)
    replace_layer_in_functional_model(model, model.layers[1], QcQuantizeParamWrapper(model.layers[1]))
    tf.keras.models.save_model(model, './data/saved_model', save_format='tf')
    model = tf.keras.models.load_model('./data/saved_model')
    print('before quant out')
    quant_out = model.predict(inp)
    assert not np.array_equal(orig_out, quant_out)

@pytest.mark.skip(reason="Enable with TF 2.4")
def test_qc_quantize_standalone_layer():
    """ Test qc quantize layer """
    inp = np.abs(np.random.randn(100, 2))
    model = dense_functional()
    model.layers[1].set_weights([np.ones((2, 2)), np.array([1.2, 1.5])])
    orig_out = model.predict(inp)
    replace_layer_in_functional_model(model, model.layers[1], [model.layers[1], QcQuantizeLayer()])
    tf.keras.models.save_model(model, './data/saved_model', save_format='')
    model = tf.keras.models.load_model('./data/saved_model')
    print('before quant out')
    quant_out = model.predict(inp)
    assert not np.array_equal(orig_out, quant_out)

@pytest.mark.skip(reason="Only run this test if eager execution is enabled")
def test_qc_quantize_layer_with_param_wrapper_training():
    """
    Test qc quantize layer and param wrapper
    """
    inp = np.abs(np.random.randn(100, 2))
    model = dense_functional()
    model.layers[1].set_weights([np.ones((2, 2)), np.array([1.2, 1.5])])
    orig_out = model.predict(inp)
    replace_layer_in_functional_model(model, model.layers[1], [QcQuantizeParamWrapper(model.layers[1]),
                                                               QcQuantizeLayer()])
    tf.keras.models.save_model(model, './data/saved_model', save_format='tf')
    model = tf.keras.models.load_model('./data/saved_model')
    quant_out = model.predict(inp)
    assert not np.array_equal(orig_out, quant_out)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    starting_weights = [weight for weight in model.layers[1]._layer_to_wrap.get_weights()]
    y = np.random.randn(100, 2)
    model.fit(x=inp, y=y, batch_size=1)
    ending_weights = [weight for weight in model.layers[1]._layer_to_wrap.get_weights()]
    for idx, weight in enumerate(starting_weights):
        assert not np.array_equal(weight, ending_weights[idx])

@pytest.mark.skip(reason="Enable with TF 2.4")
def test_qc_quantize_layer_and_param_wrapper_subclass_model():
    """
        Test basic functionality of qc quantize wrapper
        """
    model = DenseSubclassing()

    rand_inp = np.random.randn(1, 2)
    orig_out = model.predict(rand_inp)
    quantize_layer_seq = tf.keras.Sequential()
    quantize_layer_seq.add(QcQuantizeParamWrapper(model.linear1))
    quantize_layer_seq.add(QcQuantizeLayer())
    model.linear1 = quantize_layer_seq
    tf.keras.models.save_model(model, './data/saved_model', save_format='tf')
    new_model = tf.keras.models.load_model('./data/saved_model')
    quant_out = new_model.predict(rand_inp)
    assert not np.array_equal(orig_out, quant_out)
