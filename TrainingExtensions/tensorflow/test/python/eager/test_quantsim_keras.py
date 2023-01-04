# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
import os
from typing import List

import aimet_common.libpymo as libpymo
import numpy as np
import pytest
import tensorflow as tf
from aimet_common.libpymo import TfEncoding
from packaging import version
from tensorflow import keras

from aimet_common.defs import QuantScheme
from aimet_tensorflow.keras.cross_layer_equalization import equalize_model
from aimet_tensorflow.keras.quant_sim.qc_mha_wrapper import QcQuantizableMultiHeadAttention
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from test_models_keras import tiny_conv_net


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

def test_quantsim_export_quantizer_args():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = dense_functional()
        rand_inp = np.random.randn(100, 5)

        qsim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced, default_param_bw=16, default_output_bw=16 )

        qsim.export('./data', 'test_export_with_quant_args')

        with open('./data/test_export_with_quant_args.encodings') as json_file:
            encoding_data = json.load(json_file)

        assert "quantizer_args" in encoding_data
        quantizer_args = encoding_data["quantizer_args"]
        assert quantizer_args["activation_bitwidth"] == 16
        assert quantizer_args["param_bitwidth"] == 16
        assert not quantizer_args["per_channel_quantization"]
        assert quantizer_args["quant_scheme"] == QuantScheme.post_training_tf_enhanced.name
        assert quantizer_args["dtype"] == "int"
        assert quantizer_args["is_symmetric"]

def test_quantsim_with_custom_config_file():
    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "False"
            },
            "params": {
                "is_quantized": "False",
                "is_symmetric": "False"
            }
        },
        "params": {},
        "op_type": {},
        "supergroups": [
            {
                "op_list": ["Conv", "BatchNormalization"]
            },
            {
                "op_list": ["Relu", "MaxPool"]
            },
            {
                "op_list": ["Conv", "Relu", "AveragePool"]
            }
        ],
        "model_input": {},
        "model_output": {}
    }
    with open("./data/quantsim_config.json", "w") as f:
        json.dump(quantsim_config, f)

    model = tiny_conv_net()
    qsim = QuantizationSimModel(model, quant_scheme='tf', config_file="./data/quantsim_config.json")

    layers = qsim.model.layers
    conv1, relu1, conv2, conv3 = layers[1], layers[3], layers[5], layers[8]
    relu3 = layers[9]
    bn1, maxpool, bn2, avgpool = layers[2], layers[4], layers[6], layers[10]
    for layer in layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        # Check configs for starts of supergroups
        if layer in [conv1, relu1, conv2, conv3]:
            for q in layer.output_quantizers:
                assert not q.is_enabled()
        # Check configs for middle ops in supergroups
        elif layer == relu3:
            for q in layer.input_quantizers:
                assert not q.is_enabled()
            for q in layer.output_quantizers:
                assert not q.is_enabled()
        # Check configs for ends of supergroups
        elif layer in [bn1, maxpool, bn2, avgpool]:
            for q in layer.input_quantizers:
                assert not q.is_enabled()
            for q in layer.output_quantizers:
                assert q.is_enabled()
        else:
            for q in layer.input_quantizers:
                assert not q.is_enabled()
            for q in layer.output_quantizers:
                assert q.is_enabled()

    if os.path.exists("./data/quantsim_config.json"):
        os.remove("./data/quantsim_config.json")


def test_quantsim_handling_folded_bn_layer():
    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "False"
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": "False"
            }
        },
        "params": {},
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }
    with open("./data/quantsim_config.json", "w") as f:
        json.dump(quantsim_config, f)

    model = tiny_conv_net()
    cle_applied_model = equalize_model(model)
    qsim = QuantizationSimModel(cle_applied_model, quant_scheme='tf', config_file="./data/quantsim_config.json")

    layers = qsim.model.layers
    bn1, bn2 = layers[2], layers[6]
    for layer in layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        # Folded batch normalization layer
        if layer in [bn1, bn2]:
            for q in layer.output_quantizers:
                assert not q.is_enabled()
            for q in layer.param_quantizers:
                assert not q.is_enabled()
        # other layers follows default quantsim config rule
        else:
            for q in layer.output_quantizers:
                assert q.is_enabled()
            for q in layer.param_quantizers:
                assert q.is_enabled()

    if os.path.exists("./data/quantsim_config.json"):
        os.remove("./data/quantsim_config.json")


def test_quantsim_with_specific_op_type_per_channel_quantization() -> None:
    """
    Test whether TfEncoding is set correctly when specific op type has per_channel_quantization property
    """
    quantsim_config = {
        "defaults": {
            "ops": {"is_output_quantized": "True"},
            "params": {
                "is_quantized": "True",
                "is_symmetric": "True"
            },
            "strict_symmetric": "False",
            "unsigned_symmetric": "True",
            "per_channel_quantization": "True"
        },
        "params": {
            "bias": {"is_quantized": "False"}
        },
        "op_type": {
            "Gemm": {"per_channel_quantization": "False"}
        },
        "supergroups": [],
        "model_input": {"is_input_quantized": "True"},
        "model_output": {}
    }
    with open("./data/quantsim_config.json", "w") as f:
        json.dump(quantsim_config, f)

    model = tiny_conv_net()
    cle_applied_model = equalize_model(model)
    qsim = QuantizationSimModel(cle_applied_model, quant_scheme='tf',
                                config_file="./data/quantsim_config.json")

    dummy_input = np.random.randn(4, 32, 32, 3)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)

    layers = qsim.model.layers
    conv1, conv2, conv3, dense = layers[1], layers[5], layers[8], layers[12]

    for conv_layer in [conv1, conv2, conv3]:
        # Conv type will follow default per_channel_quantization=True
        for q in conv_layer.param_quantizers:
            if q.is_enabled():
                assert isinstance(q.encoding, List)

    for q in dense.param_quantizers:
        # Gemm type will follow op_type per_channel_quantization=False
        if q.is_enabled():
            assert isinstance(q.encoding, TfEncoding)

    if os.path.exists("./data/quantsim_config.json"):
        os.remove("./data/quantsim_config.json")


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
        # Note: Disable bias quantization in default_config.json
        assert len(encodings['param_encodings']) == 2

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
        tf.keras.backend.clear_session()

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
        assert encodings['activation_encodings']['dense/BiasAdd:0'][0]['max'] == \
               running_dense_output_quantizer_encoding_max

def test_assert_on_reused_layer():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        model = model_with_reused_layer()
        rand_inp = np.random.randn(100, 5)
        _ = model.predict(rand_inp)

        with pytest.raises(NotImplementedError):
            _ = QuantizationSimModel(model, quant_scheme='tf')


def test_quantizable_mha_basic():
    B = 5
    T = 8
    S = 4

    # STAGE 1 MODEL - model created with layers.MultiHeadAttention
    stage_1_q_inputs = keras.Input(shape=(T, 16))
    stage_1_v_inputs = keras.Input(shape=(S, 16))
    stage_1_output = keras.layers.MultiHeadAttention(key_dim=2, num_heads=2)(stage_1_q_inputs, stage_1_v_inputs)
    stage_1_model = keras.Model(inputs=[stage_1_q_inputs, stage_1_v_inputs], outputs=stage_1_output)

    # STAGE 2 MODEL - model manually created with QcQuantizeableMultiHeadAttention
    stage_2_q_inputs = keras.Input(shape=(T, 16))
    stage_2_v_inputs = keras.Input(shape=(S, 16))
    stage_2_output = QcQuantizableMultiHeadAttention(key_dim=2, num_heads=2)(stage_2_q_inputs, stage_2_v_inputs)
    stage_2_model = keras.Model(inputs=[stage_2_q_inputs, stage_2_v_inputs], outputs=stage_2_output)

    # STAGE 3 MODEL - model created using QuantSim
    stage_3_model = QuantizationSimModel(stage_1_model)

    query = np.ones([B, T, 16])
    value = np.ones([B, S, 16])

    output_1_tensor = stage_1_model([query, value])
    output_2_tensor = stage_2_model([query, value])
    output_3_tensor = stage_3_model.model([query, value])

    for layer in stage_3_model.model.layers:
        if isinstance(layer, QcQuantizableMultiHeadAttention): layer.deactivate_quantizers()
    output_3_tensor_without_quantizers = stage_3_model.model([query, value])
    for layer in stage_3_model.model.layers:
        if isinstance(layer, QcQuantizableMultiHeadAttention): layer.reactivate_quantizers()

    # check that all output tensors have the same shape
    assert output_1_tensor.shape == output_2_tensor.shape == output_3_tensor.shape

    # check that QcQuantizableMultiHeadAttention does not exist in original model.layers
    assert not any(isinstance(layer, QcQuantizableMultiHeadAttention) for layer in stage_1_model.layers)

    # check that QcQuantizableMultiHeadAttention exists in QuantSim model.layers
    assert any(isinstance(layer, QcQuantizableMultiHeadAttention) for layer in stage_3_model.model.layers)

    # check that QuantSim generated model has same output as original model when quantizers are disabled
    assert tf.equal(output_1_tensor, output_3_tensor_without_quantizers).numpy().flatten().all()


def test_quantizable_mha_with_value():
    B = 5
    T = 8
    S = 4

    q_inputs = keras.Input(shape=(T, 16))
    v_inputs = keras.Input(shape=(S, 16))
    k_inputs = keras.Input(shape=(S, 16))
    model_output = keras.layers.MultiHeadAttention(key_dim=2, num_heads=2)(q_inputs, v_inputs, k_inputs)
    unquantized_model = keras.Model(inputs=[q_inputs, v_inputs, k_inputs], outputs=model_output)

    quantized_model = QuantizationSimModel(unquantized_model)

    query = np.ones([B, T, 16])
    value = np.ones([B, S, 16])
    key = np.ones([B, S, 16])

    unquantized_model_tensor = unquantized_model([query, value, key])
    quantized_model_tensor = quantized_model.model([query, value, key])

    for layer in quantized_model.model.layers:
        if isinstance(layer, QcQuantizableMultiHeadAttention): layer.deactivate_quantizers()
    quantized_model_tensor_without_quantizers = quantized_model.model([query, value, key])
    for layer in quantized_model.model.layers:
        if isinstance(layer, QcQuantizableMultiHeadAttention): layer.reactivate_quantizers()

    # check that all output tensors have the same shape
    assert unquantized_model_tensor.shape == quantized_model_tensor.shape == \
           quantized_model_tensor_without_quantizers.shape

    # check that QuantSim generated model has same output as original model when quantizers are disabled
    assert tf.equal(unquantized_model_tensor, quantized_model_tensor_without_quantizers).numpy().flatten().all()

    # check that QcQuantizableMultiHeadAttention does not exist in original model.layers
    assert not any(isinstance(layer, QcQuantizableMultiHeadAttention) for layer in unquantized_model.layers)

    # check that QcQuantizableMultiHeadAttention exists in QuantSim model.layers
    assert any(isinstance(layer, QcQuantizableMultiHeadAttention) for layer in quantized_model.model.layers)


def test_quantizable_mha_with_mask():
    B = 5
    T = 8
    S = 4

    q_inputs = keras.Input(shape=(T, 16))
    v_inputs = keras.Input(shape=(S, 16))
    k_inputs = keras.Input(shape=(S, 16))
    m_inputs = keras.Input(shape=(T, S))
    model_output = keras.layers.MultiHeadAttention(key_dim=2, num_heads=2)(q_inputs, v_inputs, k_inputs, m_inputs)
    unquantized_model = keras.Model(inputs=[q_inputs, v_inputs, k_inputs, m_inputs], outputs=model_output)

    quantized_model = QuantizationSimModel(unquantized_model)

    query = np.ones([B, T, 16])
    value = np.ones([B, S, 16])
    key = np.ones([B, S, 16])
    mask = np.zeros([B, T, S])

    unquantized_model_tensor = unquantized_model([query, value, key, mask])
    quantized_model_tensor = quantized_model.model([query, value, key, mask])

    for layer in quantized_model.model.layers:
        if isinstance(layer, QcQuantizableMultiHeadAttention): layer.deactivate_quantizers()
    quantized_model_tensor_without_quantizers = quantized_model.model([query, value, key, mask])
    for layer in quantized_model.model.layers:
        if isinstance(layer, QcQuantizableMultiHeadAttention): layer.reactivate_quantizers()

    # check that all output tensors have the same shape
    assert unquantized_model_tensor.shape == quantized_model_tensor.shape == \
           quantized_model_tensor_without_quantizers.shape

    # check that QuantSim generated model has same output as original model when quantizers are disabled
    assert tf.equal(unquantized_model_tensor, quantized_model_tensor_without_quantizers).numpy().flatten().all()

    # check that QcQuantizableMultiHeadAttention does not exist in original model.layers
    assert not any(isinstance(layer, QcQuantizableMultiHeadAttention) for layer in unquantized_model.layers)

    # check that QcQuantizableMultiHeadAttention exists in QuantSim model.layers
    assert any(isinstance(layer, QcQuantizableMultiHeadAttention) for layer in quantized_model.model.layers)

def test_quantizable_mha_encodings():
    B = 5
    T = 8
    S = 4

    q_inputs = keras.Input(shape=(T, 16))
    v_inputs = keras.Input(shape=(S, 16))
    k_inputs = keras.Input(shape=(S, 16))
    m_inputs = keras.Input(shape=(T, S))
    model_output = keras.layers.MultiHeadAttention(key_dim=2, num_heads=2)(q_inputs, v_inputs, k_inputs, m_inputs)
    unquantized_model = keras.Model(inputs=[q_inputs, v_inputs, k_inputs, m_inputs], outputs=model_output)

    quantized_model = QuantizationSimModel(unquantized_model)

    rng = np.random.default_rng(seed=42)
    query = rng.random([B, T, 16])
    value = rng.random([B, S, 16])
    key = rng.random([B, S, 16])
    mask = np.zeros([B, T, S])

    quantized_model.compute_encodings(lambda m, _: m([query, value, key, mask]), None)

    query = query * 10
    value = value * 10
    key = key * 10

    unquantized_model_tensor = unquantized_model([query, value, key, mask]).numpy().flatten()
    quantized_model_tensor = quantized_model.model([query, value, key, mask]).numpy().flatten()

    output_encoding_min = quantized_model.model.layers[-1]._wrapped_layers[-1].output_quantizers[0]._encoding_min
    output_encoding_max = quantized_model.model.layers[-1]._wrapped_layers[-1].output_quantizers[0]._encoding_max

    # checking to make sure all outputs fall within the limits set by the output quantizer
    FLOAT_DELTA = 0.0001
    assert all((quantized_model_tensor >= output_encoding_min - FLOAT_DELTA) &
               (quantized_model_tensor <= output_encoding_max + FLOAT_DELTA))

def test_quantizable_mha_export_encodings():
    B = 5
    T = 8
    S = 4

    # STAGE 1 MODEL - model created with layers.MultiHeadAttention
    stage_1_q_inputs = keras.Input(shape=(T, 16))
    stage_1_v_inputs = keras.Input(shape=(S, 16))
    stage_1_output = keras.layers.MultiHeadAttention(key_dim=2, num_heads=2)(stage_1_q_inputs, stage_1_v_inputs)
    stage_1_model = keras.Model(inputs=[stage_1_q_inputs, stage_1_v_inputs], outputs=stage_1_output)

    # STAGE 3 MODEL - model created using QuantSim
    stage_3_model = QuantizationSimModel(stage_1_model)

    rng = np.random.default_rng(seed=42)
    query = rng.random([B, T, 16]) * 100
    value = rng.random([B, S, 16]) * 100

    stage_3_model.compute_encodings(lambda m, _: m([query, value]), None)
    stage_3_model.export('./data', 'mha')

    with open("./data/mha.encodings", "r") as encodings_file:
        encodings = json.load(encodings_file)

    for wrapper in stage_3_model.model.layers[2]._wrapped_layers:
        for io_quantizer in wrapper.input_quantizers + wrapper.output_quantizers:
            if io_quantizer.encoding is not None:
                tensor_name = "multi_head_attention/" + wrapper.name + "/" + io_quantizer.name
                encoding_dict = QuantizationSimModel._get_encoding_dict_for_quantizer(io_quantizer)
                assert tensor_name in encodings['activation_encodings']
                assert encodings['activation_encodings'][tensor_name] == encoding_dict
        for idx, param_quantizer in enumerate(wrapper.param_quantizers):
            if param_quantizer.encoding is not None:
                param_name = wrapper._layer_to_wrap.weights[idx].name
                encoding_dict = QuantizationSimModel._get_encoding_dict_for_quantizer(param_quantizer)
                assert param_name in encodings['param_encodings']
                assert encodings['param_encodings'][param_name] == encoding_dict


