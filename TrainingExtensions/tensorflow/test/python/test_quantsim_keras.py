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
from packaging import version
import numpy as np
import tensorflow as tf

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

def test_quantsim_basic():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = dense_functional()
        rand_inp = np.random.randn(100, 5)
        qsim = QuantizationSimModel(model, quant_scheme='tf')
        orig_out = qsim.model.predict(rand_inp)
        quant_wrappers = [quant_wrapper for quant_wrapper in qsim.quant_wrappers()]
        assert len(quant_wrappers) == 2
        assert len(quant_wrappers[0].param_quantizers) == 2
        for quant_wrapper in quant_wrappers:
            assert quant_wrapper.input_quantizers[0].quant_scheme == libpymo.QuantizationMode.QUANTIZATION_TF
            assert quant_wrapper.input_quantizers[0].round_mode == libpymo.RoundingMode.ROUND_NEAREST
            assert quant_wrapper.output_quantizers[0].quant_scheme == libpymo.QuantizationMode.QUANTIZATION_TF
            assert quant_wrapper.output_quantizers[0].round_mode == libpymo.RoundingMode.ROUND_NEAREST
        qsim.compute_encodings(lambda model, _: model.predict(rand_inp), None)
        quant_out = qsim.model.predict(rand_inp)
        assert not np.array_equal(orig_out, quant_out)

def test_qat():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = dense_functional()
        rand_inp = np.random.randn(10, 5)
        rand_out = np.random.randn(10, 2)
        qsim = QuantizationSimModel(model, quant_scheme='tf', default_param_bw=8, default_output_bw=8)
        qsim.compute_encodings(lambda m, _: m.predict(rand_inp), None)
        qsim.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss=tf.keras.losses.MeanSquaredError())
        running_weights = [tf.keras.backend.get_value(param) for
                              param in qsim.model.layers[1]._layer_to_wrap.weights]
        running_loss = None
        for i in range(10):
            history = qsim.model.fit(x=rand_inp, y=rand_out, batch_size=1)
            loss = history.history['loss'][0]
            if running_loss is not None:
                assert loss <= running_loss
            running_loss = loss
            ending_weights = [tf.keras.backend.get_value(param) for
                              param in qsim.model.layers[1]._layer_to_wrap.weights]
            for idx, weight in enumerate(running_weights):
                assert not np.array_equal(weight, ending_weights[idx])
            running_weights = ending_weights
