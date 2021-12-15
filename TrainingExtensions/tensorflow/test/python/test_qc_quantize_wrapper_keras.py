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
import tensorflow as tf
import numpy as np
from packaging import version
if version.parse(tf.version.VERSION) >= version.parse("2.00"):
    from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer

    from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper, QuantizeWrapperTransform, \
        QuantizerSettings
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

def test_wrapper():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        inp = tf.keras.layers.Input(shape=(2,))
        x = QcQuantizeWrapper(tf.keras.layers.Lambda(lambda x: x),
                              QuantizerSettings(8, 'nearest', 'tf', False, False, False),
                              QuantizerSettings(8, 'nearest', 'tf', False, False, False))(inp)
        model = tf.keras.Model(inputs=inp, outputs=x)

        rand_inp = np.random.randn(100, 2) * 10.0
        orig_out = model.predict(rand_inp)
        encoding = model.layers[1].input_quantizers[0].compute_encoding()
        model.layers[1].input_quantizers[0]._encoding_min.assign(encoding.min)
        model.layers[1].input_quantizers[0]._encoding_max.assign(encoding.max)
        assert model.layers[1].input_quantizers[0].tensor_quantizer.isEncodingValid

        # Check that before configuring op mode var to quantizeDequantize, the model output remains same
        quant_out = model.predict(rand_inp)
        assert np.array_equal(orig_out, quant_out)

        model.layers[1].input_quantizers[0]._quantizer_mode.assign(
            int(libpymo.TensorQuantizerOpMode.quantizeDequantize))
        quant_out = model.predict(rand_inp)
        assert not np.array_equal(orig_out, quant_out)

def test_functional_model_with_wrapper():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        rand_inp = np.random.randn(100, 2)
        inp = tf.keras.layers.Input(shape=(2,))
        out = tf.keras.layers.Dense(units=2)(inp)
        out = tf.keras.layers.Softmax()(out)
        model = tf.keras.Model(inputs=inp, outputs=out, name="dense_functional")
        orig_out = model.predict(rand_inp)

        name_to_layer_map = {}
        for layer in model.layers:
            name_to_layer_map[layer.name] = layer

        transforms = [QuantizeWrapperTransform('Softmax',
                                               QuantizerSettings(8, 'nearest', 'tf', False, False, False),
                                               QuantizerSettings(8, 'nearest', 'tf', False, False, False),
                                               name_to_layer_map),
                      QuantizeWrapperTransform('Dense',
                                               QuantizerSettings(8, 'nearest', 'tf', False, False, False),
                                               QuantizerSettings(8, 'nearest', 'tf', False, False, False),
                                               name_to_layer_map)]
        new_model, _ = model_transformer.ModelTransformer(model, transforms).transform()
        assert len(new_model.layers[1].input_quantizers) == 1
        assert len(new_model.layers[1].output_quantizers) == 1
        assert len(new_model.layers[1].param_quantizers) == 2
        assert len(new_model.layers[2].input_quantizers) == 1
        assert len(new_model.layers[2].output_quantizers) == 1
        assert len(new_model.layers[2].param_quantizers) == 0

        # Test that model output remains same prior to compute encodings
        # Disable param quantizers first, otherwise one shot quant/dequant will affect output
        new_model.layers[1].param_quantizers[0]._quantizer_mode.assign(3)
        new_model.layers[1].param_quantizers[1]._quantizer_mode.assign(3)
        quant_out = new_model.predict(rand_inp)
        assert np.array_equal(orig_out, quant_out)

        new_model.layers[1].param_quantizers[0]._quantizer_mode.assign(1)
        new_model.layers[1].param_quantizers[1]._quantizer_mode.assign(1)
        encoding = new_model.layers[1].input_quantizers[0].compute_encoding()
        new_model.layers[1].input_quantizers[0]._quantizer_mode.assign(
            int(libpymo.TensorQuantizerOpMode.quantizeDequantize))
        new_model.layers[1].input_quantizers[0]._encoding_min.assign(encoding.min)
        new_model.layers[1].input_quantizers[0]._encoding_max.assign(encoding.max)
        quant_out = new_model.predict(rand_inp)
        assert not np.array_equal(orig_out, quant_out)
