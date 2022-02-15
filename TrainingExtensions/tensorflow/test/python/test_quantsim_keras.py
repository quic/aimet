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

import json
import pytest
pytestmark = pytest.mark.skip("Disable tests that requires eager execution")
from packaging import version
import numpy as np
import tensorflow as tf

from aimet_common.defs import QuantScheme
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
import libpymo

def dense_functional():
    inp = tf.keras.layers.Input(shape=(5,))
    x = tf.keras.layers.Dense(units=2)(inp)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name="dense_functional")
    return model

def dense_sequential():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=2, input_shape=(5,)))
    model.add(tf.keras.layers.Softmax())
    return model

class DenseSubclassing(tf.keras.Model):
    def __init__(self):
        super(DenseSubclassing, self).__init__()
        self.linear1 = tf.keras.layers.Dense(units=2)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        x = self.linear1(inputs)
        x = self.softmax(x)
        return x

def model_with_lambda_operators():
    inp = tf.keras.layers.Input(shape=(5,))
    inp_2 = tf.keras.layers.Input(shape=(3,))
    x1 = tf.keras.layers.Dense(units=2)(inp)
    x2 = tf.keras.layers.Dense(units=2)(inp_2)
    x = x1 + x2
    x = x - 1.0
    x = x * x1
    x = x / x2
    model = tf.keras.Model(inputs=(inp, inp_2), outputs=x, name="model_with_lambda_operators")
    return model

def model_with_reused_layer():
    relu = tf.keras.layers.ReLU()
    inp = tf.keras.layers.Input(shape=(5,))
    x = relu(inp)
    x = tf.keras.layers.Dense(units=2)(x)
    x = relu(x)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name="model_with_reused_layer")
    return model

class DenseReluLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DenseReluLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(units=2)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.relu(x)
        return x

def test_quantsim_basic():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = dense_functional()
        rand_inp = np.random.randn(100, 5)
        orig_out = model.predict(rand_inp)

        qsim = QuantizationSimModel(model, quant_scheme='tf')
        quant_wrappers = [quant_wrapper for quant_wrapper in qsim.quant_wrappers()]
        assert len(quant_wrappers) == 2
        assert len(quant_wrappers[0].param_quantizers) == 2
        for quant_wrapper in quant_wrappers:
            assert quant_wrapper.input_quantizers[0].quant_scheme == QuantScheme.post_training_tf
            assert quant_wrapper.input_quantizers[0].round_mode == libpymo.RoundingMode.ROUND_NEAREST
            assert quant_wrapper.output_quantizers[0].quant_scheme == QuantScheme.post_training_tf
            assert quant_wrapper.output_quantizers[0].round_mode == libpymo.RoundingMode.ROUND_NEAREST
        assert len(qsim.model.layers[1].input_quantizers) == 1
        assert len(qsim.model.layers[1].output_quantizers) == 1
        assert len(qsim.model.layers[1].param_quantizers) == 2
        assert len(qsim.model.layers[2].input_quantizers) == 1
        assert len(qsim.model.layers[2].output_quantizers) == 1
        assert len(qsim.model.layers[2].param_quantizers) == 0

        # Test that model output remains same prior to compute encodings
        # Disable param quantizers first, otherwise one shot quant/dequant will affect output
        qsim.model.layers[1].param_quantizers[0].disable()
        qsim.model.layers[1].param_quantizers[1].disable()
        quant_out = qsim.model.predict(rand_inp)
        assert np.array_equal(orig_out, quant_out)

        qsim.model.layers[1].param_quantizers[0].enable()
        qsim.model.layers[1].param_quantizers[1].enable()

        # Run one more forward pass after enabling param quantizers
        qsim.compute_encodings(lambda m, _: m(rand_inp), None)

        assert qsim.model.layers[1].param_quantizers[0].encoding is not None
        quant_out = qsim.model.predict(rand_inp)
        assert not np.array_equal(orig_out, quant_out)

        qsim.export('./data', 'test_export')

def test_model_with_lambda_operators():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = model_with_lambda_operators()
        rand_inp_1 = np.random.randn(10, 5)
        rand_inp_2 = np.random.randn(10, 3)
        _ = model.predict((rand_inp_1, rand_inp_2))

        qsim = QuantizationSimModel(model, quant_scheme='tf')
        qsim.compute_encodings(lambda m, _: m((rand_inp_1, rand_inp_2)), None)
        qsim.export('./data', 'model_with_lambda_operators')
        assert len(list(qsim.quant_wrappers())) == 6

        with open("./data/model_with_lambda_operators.encodings", "r") as encodings_file:
            encodings = json.load(encodings_file)

        assert len(encodings['activation_encodings']) == 8
        assert len(encodings['param_encodings']) == 4

def test_qat():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = dense_functional()
        rand_inp = np.random.randn(10, 5)
        rand_out = np.random.randn(10, 2)
        qsim = QuantizationSimModel(model, quant_scheme='tf', default_param_bw=8, default_output_bw=8)
        qsim.compute_encodings(lambda m, _: m.predict(rand_inp), None)
        qsim.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss=tf.keras.losses.MeanSquaredError())
        # Track weights for dense layer to check that they are updated during fit
        running_weights = [tf.keras.backend.get_value(param) for
                              param in qsim.model.layers[1]._layer_to_wrap.weights]
        # Track encoding max for dense output quantizer to check that it is not updated during fit
        running_dense_output_quantizer_encoding_max = \
            tf.keras.backend.get_value(qsim.model.layers[1].output_quantizers[0]._encoding_max)

        for i in range(10):
            _ = qsim.model.fit(x=rand_inp, y=rand_out, batch_size=1)
            ending_weights = [tf.keras.backend.get_value(param) for
                              param in qsim.model.layers[1]._layer_to_wrap.weights]
            new_dense_output_quantizer_encoding_max = \
                tf.keras.backend.get_value(qsim.model.layers[1].output_quantizers[0]._encoding_max)
            for idx, weight in enumerate(running_weights):
                assert not np.array_equal(weight, ending_weights[idx])
            assert np.array_equal(new_dense_output_quantizer_encoding_max,
                                  running_dense_output_quantizer_encoding_max)
            running_weights = ending_weights
            running_dense_output_quantizer_encoding_max = new_dense_output_quantizer_encoding_max

def test_range_learning():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = dense_functional()
        rand_inp = np.random.randn(10, 5)
        rand_out = np.random.randn(10, 2)
        qsim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                    default_param_bw=8, default_output_bw=8)
        for wrapper in qsim.quant_wrappers():
            wrapper.input_quantizers[0].disable()

        qsim.compute_encodings(lambda m, _: m.predict(rand_inp), None)
        qsim.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss=tf.keras.losses.MeanSquaredError())
        # Track weights for dense layer to check that they are updated during fit
        running_weights = [tf.keras.backend.get_value(param) for
                              param in qsim.model.layers[1]._layer_to_wrap.weights]
        # Track encoding max for dense output quantizer to check that it is updated during fit
        running_dense_output_quantizer_encoding_max = \
            tf.keras.backend.get_value(qsim.model.layers[1].output_quantizers[0]._encoding_max)

        for i in range(10):
            _ = qsim.model.fit(x=rand_inp, y=rand_out, batch_size=1)
            ending_weights = [tf.keras.backend.get_value(param) for
                              param in qsim.model.layers[1]._layer_to_wrap.weights]
            new_dense_output_quantizer_encoding_max = \
                tf.keras.backend.get_value(qsim.model.layers[1].output_quantizers[0]._encoding_max)
            for idx, weight in enumerate(running_weights):
                assert not np.array_equal(weight, ending_weights[idx])
            assert not np.array_equal(new_dense_output_quantizer_encoding_max,
                                      running_dense_output_quantizer_encoding_max)
            running_weights = ending_weights
            running_dense_output_quantizer_encoding_max = new_dense_output_quantizer_encoding_max

        # Check that exporting encodings exports the learned encodings
        qsim.export('./data', 'dense_functional')
        with open("./data/dense_functional.encodings", "r") as encodings_file:
            encodings = json.load(encodings_file)
        assert encodings['activation_encodings']['dense/BiasAdd:0']['max'] == \
               running_dense_output_quantizer_encoding_max

def test_assert_on_reused_layer():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = model_with_reused_layer()
        rand_inp = np.random.randn(100, 5)
        _ = model.predict(rand_inp)

        with pytest.raises(NotImplementedError):
            _ = QuantizationSimModel(model, quant_scheme='tf')
