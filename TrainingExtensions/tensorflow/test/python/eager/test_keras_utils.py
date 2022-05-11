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
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from aimet_tensorflow.keras.utils.common import replace_layer_in_functional_model

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
    new_layers = [tf.keras.layers.Dense(units=4), tf.keras.layers.Dense(units=2, kernel_initializer=tf.keras.initializers.Constant(5.))]
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
