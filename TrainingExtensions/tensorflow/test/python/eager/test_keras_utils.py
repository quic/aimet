# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
import json
import os
import tempfile

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.utils.common import convert_h5_model_to_pb_model
from aimet_tensorflow.keras.utils.common import replace_layer_in_functional_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def conv_functional():
    input_shape = (128, 28, 28, 1)
    inp = tf.keras.Input(shape=input_shape[1:])
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inp)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5, trainable=False)(x)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs=x, name='conv_functional')
    return model


# Not used for testing at the moment. This is placed here for future testing.
class ConvTimesThree(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConvTimesThree, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(32,
                                           kernel_size=(3, 3),
                                           activation='relu',
                                           name='class_conv')
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(64,
                                                              kernel_size=(3, 3),
                                                              activation='relu',
                                                              name='class_conv_transpose')
        self.depth_conv = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1,
                                                          kernel_size=(3, 3),
                                                          activation='relu',
                                                          name='class_conv_depth')

    def call(self, x):
        x = self.conv(x)
        x = self.conv_transpose(x)
        x = self.depth_conv(x)
        return x


# See comment above ConvTimesThree Class
def conv_sub_class():
    input_shape = (128, 28, 28, 1)
    inp = tf.keras.Input(shape=input_shape[1:])
    x = ConvTimesThree()(inp)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x, training=False)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs=x, name='conv_classes')
    return model


def test_replace_middle_layers():
    # Create model
    inp = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Dense(units=1)(inp)
    x = tf.keras.layers.Dense(units=2, kernel_initializer=tf.keras.initializers.Constant(2.))(x)
    x = tf.keras.layers.Dense(units=3)(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name="replace_middle_layers_model")

    test_inp = np.array([[1, 2]])
    _ = model.predict(test_inp)

    # Define new layers to substitute for old layer
    new_layers = [tf.keras.layers.Dense(units=4),
                  tf.keras.layers.Dense(units=2, kernel_initializer=tf.keras.initializers.Constant(5.))]
    old_dense = model.layers[2]

    replace_layer_in_functional_model(model, old_dense, new_layers)
    tf.keras.models.save_model(model, './data/saved_model')
    new_model = tf.keras.models.load_model('./data/saved_model')

    out = new_model(test_inp)
    assert out.shape == [1, 3]
    assert isinstance(new_model.layers[2], tf.keras.layers.Dense)
    assert new_model.layers[1].outbound_nodes[0] == new_model.layers[2].inbound_nodes[0]
    assert new_model.layers[2].outbound_nodes[0] == new_model.layers[3].inbound_nodes[0]
    assert new_model.layers[3].outbound_nodes[0] == new_model.layers[4].inbound_nodes[0]

    # layers[5] should contain old dense layer that is not connected to anything
    assert new_model.layers[5].inbound_nodes == []
    assert new_model.layers[5].outbound_nodes == []
    assert new_model.layers[5].weights[0].shape == [1, 2]
    assert new_model.layers[5].weights[0][0][0] == 2.

    # check that new layers were inserted correctly
    assert new_model.layers[2].weights[0].shape == [1, 4]
    assert new_model.layers[3].weights[0].shape == [4, 2]
    assert new_model.layers[3].weights[0][0][0] == 5.


def test_replace_output_layer():
    inp = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Dense(units=1)(inp)
    x = tf.keras.layers.Dense(units=2)(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name="replace_output_layer_model")

    test_inp = np.array([[1, 2]])
    _ = model.predict(test_inp)

    new_layer = tf.keras.layers.Dense(units=3)
    old_dense = model.layers[2]
    replace_layer_in_functional_model(model, old_dense, new_layer)
    tf.keras.models.save_model(model, './data/saved_model')
    new_model = tf.keras.models.load_model('./data/saved_model')

    out = new_model(test_inp)
    assert out.shape == [1, 3]


def test_replace_multi_input_layer():
    inp = tf.keras.layers.Input(shape=(2,))
    inp2 = tf.keras.layers.Input(shape=(2,))
    x = inp + inp2
    model = tf.keras.Model(inputs=[inp, inp2], outputs=x, name="replace_multi_input_layer_model")

    test_inp = np.array([[1, 2]])
    test_inp2 = np.array([[2, 3]])
    _ = model.predict([test_inp, test_inp2])

    new_layer = tf.keras.layers.Subtract()
    old_add = model.layers[2]
    replace_layer_in_functional_model(model, old_add, new_layer)
    tf.keras.models.save_model(model, './data/saved_model')
    new_model = tf.keras.models.load_model('./data/saved_model')

    out = new_model([test_inp, test_inp2])
    assert np.array_equal(out, np.array([[-1, -1]]))


def test_replace_layer_with_multiple_children():
    inp = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.ReLU()(inp)
    out1 = tf.keras.layers.Dense(units=2)(x)
    out2 = tf.keras.layers.Dense(units=4)(x)
    model = tf.keras.Model(inputs=inp, outputs=[out1, out2], name="replace_multiple_children_layer_model")

    test_inp = np.array([[1, 2]])
    _, _ = model.predict(test_inp)

    new_layer = tf.keras.layers.PReLU()
    old_relu = model.layers[1]
    replace_layer_in_functional_model(model, old_relu, new_layer)
    tf.keras.models.save_model(model, './data/saved_model')
    new_model = tf.keras.models.load_model('./data/saved_model')

    assert isinstance(new_model.layers[1], tf.keras.layers.PReLU)
    assert new_model.layers[0].outbound_nodes[0] == new_model.layers[1].inbound_nodes[0]
    assert new_model.layers[2].inbound_nodes[0] in new_model.layers[1].outbound_nodes
    assert new_model.layers[3].inbound_nodes[0] in new_model.layers[1].outbound_nodes
    assert isinstance(new_model.layers[4], tf.keras.layers.ReLU)
    assert not new_model.layers[4].inbound_nodes
    assert not new_model.layers[4].outbound_nodes


def test_replace_layer_in_internal_model():
    inp = tf.keras.layers.Input(shape=(2,))
    out = tf.keras.layers.PReLU(alpha_initializer='ones')(inp)
    inner_model = tf.keras.Model(inputs=inp, outputs=out, name="internal_model")

    outer_model = tf.keras.Sequential()
    outer_model.add(tf.keras.layers.PReLU(alpha_initializer='ones', input_shape=(2,)))
    outer_model.add(inner_model)
    outer_model.add(tf.keras.layers.PReLU(alpha_initializer='ones'))

    test_inp = np.array([[-1, -2]])
    _ = outer_model.predict(test_inp)

    old_prelu = outer_model.layers[1].layers[1]
    new_relu = tf.keras.layers.ReLU()
    replace_layer_in_functional_model(outer_model, old_prelu, new_relu)
    tf.keras.models.save_model(outer_model, './data/saved_model')
    new_model = tf.keras.models.load_model('./data/saved_model')

    out = new_model.predict(test_inp)
    assert np.array_equal(np.array([[0., 0.]]), out)


def check_conversion_tensor_names(model, custom_objects=None):
    """
    Driving function for testing conversion script. Takes a model as input to run QuantSim on and then convert the
    exported h5 model to a frozen pb model for SNPE/QNN consumtion. This function checks if the h5 and the pb weight
    names are valid.
    :param model:
    :param custom_objects:
    :return:
    """
    tf.keras.backend.clear_session()

    def get_converted_models_weight_names(converted_model_path) -> set:
        """
        Helper function to read a converted pb model and return all the weight names
        :param converted_model_path: path to the converted model
        :return: a set of the weight names
        """
        converted_weight_names = set()
        with tf.compat.v1.Session() as persisted_sess:
            with gfile.FastGFile(converted_model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                persisted_sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
                for op in persisted_sess.graph.get_operations():
                    converted_weight_names.add(op.name)
        return converted_weight_names

    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True"
            },
            "params": {
                "is_symmetric": "True",
                "is_quantized": "True"
            },
            "per_channel_quantization": "False",
        },
        "params": {},
        "op_type": {
            "Conv": {
                "is_input_quantized": "True",
                "is_output_quantized": "True"
            },
            "ConvTranspose": {
                "is_input_quantized": "True",
                "is_output_quantized": "True"
            },
            "Gemm": {
                "is_input_quantized": "True",
                "is_output_quantized": "True"
            },
            "MatMul": {
                "is_input_quantized": "True",
                "is_output_quantized": "True"
            },
            "MaxPooling2D": {
                "is_input_quantized": "True",
                "is_output_quantized": "True"
            }
        },
        "supergroups": [],
        "model_input": {},
        "model_output": {
            "is_output_quantized": "True"
        }
    }

    with open('./config.json', 'w') as f:
        json.dump(quantsim_config, f)

    random_input_data = tf.random.normal(shape=(128, *model.input_shape[1:]))
    # run forward pass on dense to generate weights
    _ = model(random_input_data)

    sim = QuantizationSimModel(model, quant_scheme='tf', config_file='./config.json')
    sim.compute_encodings(lambda m, _: m.predict(random_input_data), None)
    # convert_h5_model_to_pb_model is called during export.
    sim.export('./tmp', model.name)

    # Get all encodings names (param_encodings and activation encodings) and put their respective keys
    # (which represent the weights names) into a set for fast checking.
    encodings = sim.get_encodings_dict()
    encoding_weight_names = {*encodings['param_encodings'].keys(), *encodings['activation_encodings'].keys()}
    original_weight_names = {
        weight_name.split(':')[0]
        for weight_name in encoding_weight_names
        if 'dropout' not in weight_name
    }

    # Convert h5 model that was exported from QuantSim to a pb model to be used with encodings that were exported
    converted_weight_names = get_converted_models_weight_names(f'./tmp/{model.name}_converted.pb')

    # Check to see if all the original weight names can be found in the converted pb model
    missing_weight_names = original_weight_names.difference(converted_weight_names)
    assert not missing_weight_names, f"Weight name(s): {missing_weight_names} are missing"


def test_convert_h5_to_pb_file_does_not_exist():
    with pytest.raises(FileNotFoundError):
        convert_h5_model_to_pb_model('NA_FILE.h5')


def test_convert_h5_to_pb_not_h5_file():
    incorrect_filename = tempfile.NamedTemporaryFile(suffix='.pb', delete=True)
    with pytest.raises(ValueError):
        convert_h5_model_to_pb_model(incorrect_filename.name)


def test_convert_h5_to_pb_functional_model():
    check_conversion_tensor_names(conv_functional())


@pytest.mark.skip(
    reason="Subclassed Keras models are not currently supported. "
           "Created for future testing.")
def test_convert_h5_to_pb_subclass_model():
    check_conversion_tensor_names(conv_sub_class())


@pytest.mark.skip(reason="This test takes a long time. Only used during development.")
def test_convert_h5_to_pb_pretrained_keras():
    model = tf.keras.applications.ResNet50(weights="imagenet",
                                           input_shape=(224, 224, 3))
    check_conversion_tensor_names(model)
