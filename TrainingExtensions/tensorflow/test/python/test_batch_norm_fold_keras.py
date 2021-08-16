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
"""
This file contains unit tests for testing batch norm folding
"""
import unittest
import tensorflow as tf
import numpy as np
from packaging import version

from aimet_tensorflow.keras.utils import common
from aimet_tensorflow.keras.batch_norm_fold import _delete_all_bns_from_model, find_possible_convs_linears_bn, get_ordered_conv_linears, find_all_batch_norms_to_fold, fold_all_batch_norms
from aimet_tensorflow.keras.utils.op.batchnorm import BNUtils

class TestBatchNormFold(unittest.TestCase):
    """ Test methods for BatchNormFold"""

    def test_bn_replacement_model(self):

        class Block2(tf.keras.Model):
            def __init__(self):
                super(Block2, self).__init__()
                # define all layers in init
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.relu = tf.keras.layers.ReLU()

            def call(self, x):
                # forward pass
                x = self.dn1(x)
                x = self.bn1(x)
                x = self.relu(x)
                return x

        class Block1(tf.keras.Model):
            def __init__(self):
                super(Block1, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block2 = Block2()

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block2(x)
                return x

        class MyModel(tf.keras.Model):
            def __init__(self, kernel_num, kernel_size, strides):
                super(MyModel, self).__init__()
                # define all layers in init
                self.conv1 = tf.keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.block1 = Block1()
                self.block2 = Block2()

            def call(self, input_tensor, training=False):
                # forward pass
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.block1(x)
                x = self.block2(x)
                return x

        model = MyModel(3, (3, 3), 1)

        bn_layers = [model.bn1, model.block1.block2.bn1, model.block2.bn1]

        _delete_all_bns_from_model(model, bn_layers)

        self.assertFalse(isinstance(model.bn1,tf.keras.layers.BatchNormalization))
        self.assertFalse(isinstance(model.block1.block2.bn1,tf.keras.layers.BatchNormalization))
        self.assertFalse(isinstance(model.block2.bn1,tf.keras.layers.BatchNormalization))
        self.assertTrue(isinstance(model.block1.bn1, tf.keras.layers.BatchNormalization))

    def test_bn_replacement_layers(self):

        class Block2(tf.keras.layers.Layer):
            def __init__(self):
                super(Block2, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.relu = tf.keras.layers.ReLU()

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.relu(x)
                return x

        class Block1(tf.keras.Model):
            def __init__(self):
                super(Block1, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.block2 = Block2()

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.block2(x)
                return x

        class MyModel(tf.keras.Model):
            def __init__(self, kernel_num, kernel_size, strides):
                super(MyModel, self).__init__()
                # define all layers in init
                self.conv1 = tf.keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block1 = Block1()
                self.block2 = Block2()

            def call(self, input_tensor, training=False):
                # forward pass
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block1(x)
                x = self.block2(x)
                return x


        model = MyModel(3, (3, 3), 1)
        model.build((1, 6, 6, 3))

        bn_layers = [model.bn1, model.block1.block2.bn1, model.block2.bn1]

        # ref_name = {}
        # module_name_reference(ref_name, model)

        _delete_all_bns_from_model(model, bn_layers)

        self.assertFalse(isinstance(model.bn1,tf.keras.layers.BatchNormalization))
        self.assertFalse(isinstance(model.block1.block2.bn1,tf.keras.layers.BatchNormalization))
        self.assertFalse(isinstance(model.block2.bn1,tf.keras.layers.BatchNormalization))
        self.assertTrue(isinstance(model.block1.bn1, tf.keras.layers.BatchNormalization))

    def test_bn_replacement_sequential(self):

        Block3 = tf.keras.Sequential()
        Block3.add(tf.keras.layers.BatchNormalization(fused=True))
        Block3.add(tf.keras.layers.ReLU())

        Block1 = tf.keras.Sequential()
        Block1.add(tf.keras.layers.BatchNormalization(fused=True))
        Block1.add(Block3)

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(Block1)

        chain_model = tf.keras.Sequential()
        chain_model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1))
        chain_model.add(tf.keras.layers.BatchNormalization(fused=True))
        chain_model.add(Block2)


        # model.build()

        bn_layers = []
        len_layers = []
        len_layers.append(len(chain_model.layers[2].layers))
        len_layers.append(len(chain_model.layers[2].layers[3].layers[1].layers))
        bn_layers.append(chain_model.layers[2].layers[3].layers[1].layers[0])
        bn_layers.append(chain_model.layers[2].layers[0])

        _delete_all_bns_from_model(chain_model, bn_layers)

        self.assertTrue(len(chain_model.layers[2].layers) == len_layers[0]-1)
        self.assertTrue(len(chain_model.layers[2].layers[2].layers[1].layers) == len_layers[1]-1)

    def test_bn_replacement_combined_seq_model(self):

        class Block3(tf.keras.Model):
            def __init__(self):
                super(Block3, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.relu = tf.keras.layers.ReLU()

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.relu(x)
                return x

        class Block1(tf.keras.Model):
            def __init__(self):
                super(Block1, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.block3 = Block3()

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.block3(x)
                return x

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(Block1())

        class chain_model(tf.keras.Model):
            def __init__(self, kernel_num, kernel_size, strides):
                super(chain_model, self).__init__()
                # define all layers in init
                self.conv1 = tf.keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block2 = Block2

            def call(self, input_tensor, training=False):
                # forward pass
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block2(x)
                return x

        model = chain_model(3, (3, 3), 1)

        bn_layers = []
        bn_layers.append(model.block2.get_layer(index=0))
        bn_layers.append(model.block2.get_layer(index=3).bn1)
        bn_layers.append(model.block2.get_layer(index=3).block3.bn1)

        len_layers = []
        len_layers.append(len(model.block2.layers))

        _delete_all_bns_from_model(model, bn_layers)


        self.assertTrue(len(model.block2.layers) == len_layers[0]-1)
        self.assertFalse(isinstance(model.block2.get_layer(index=2).bn1, tf.keras.layers.BatchNormalization))
        self.assertFalse(isinstance(model.block2.get_layer(index=2).block3.bn1, tf.keras.layers.BatchNormalization))

    def test_bn_replacement_combined2_seq_model(self):

        Block3 = tf.keras.Sequential()
        Block3.add(tf.keras.layers.BatchNormalization(fused=True))
        Block3.add(tf.keras.layers.ReLU())
        Block3.add(tf.keras.layers.Conv2D(3, 3))


        class Block1(tf.keras.Model):
            def __init__(self):
                super(Block1, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.block3 = Block3

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.block3(x)
                return x

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(Block1())

        class chain_model(tf.keras.Model):
            def __init__(self, kernel_num, kernel_size, strides):
                super(chain_model, self).__init__()
                # define all layers in init
                self.conv1 = tf.keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block2 = Block2

            def call(self, input_tensor, training=False):
                # forward pass
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block2(x)
                return x

        model = chain_model(3, (3, 3), 1)

        bn_layers = list()
        bn = model.block2.get_layer(index =3).block3.get_layer(index=0)
        bn_layers.append(bn)
        bn = model.bn1
        bn_layers.append(bn)

        len_layers = []
        len_layers.append(len(model.block2.get_layer(index =3).block3.layers))

        _delete_all_bns_from_model(model,bn_layers)

        self.assertTrue(len(model.block2.get_layer(index =3).block3.layers) == len_layers[0]-1)
        self.assertFalse(isinstance(model.bn1, tf.keras.layers.BatchNormalization))

    def test_bn_replacement_combined3_seq_model(self):
        Block3 = tf.keras.Sequential()
        Block3.add(tf.keras.layers.BatchNormalization(fused=True))
        Block3.add(tf.keras.layers.ReLU())
        Block3.add(tf.keras.layers.Conv2D(3, 3))


        class Block1(tf.keras.Model):
            def __init__(self):
                super(Block1, self).__init__()
                # define all layers in init
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.block3 = Block3

            def call(self, x):
                # forward pass
                x = self.bn1(x)
                x = self.block3(x)
                return x



        class chain_model(tf.keras.Model):
            def __init__(self, kernel_num, kernel_size, strides):
                super(chain_model, self).__init__()
                # define all layers in init
                self.conv1 = tf.keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block1 = Block1()

            def call(self, input_tensor, training=False):
                # forward pass
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block1(x)
                return x

        model = chain_model(3, (3, 3), 1)

        bn_layers = list()
        bn = model.block1.block3.get_layer(index=0)
        bn_layers.append(bn)
        bn = model.block1.bn1
        bn_layers.append(bn)

        len_layers = []
        len_layers.append(len(model.block1.block3.layers))

        _delete_all_bns_from_model(model, bn_layers)

        self.assertTrue(len(model.block1.block3.layers) == len_layers[0]-1)
        self.assertFalse(isinstance(model.block1.bn1, tf.keras.layers.BatchNormalization))

    def test_bn_replacement_combined_all(self):

        Block3 = tf.keras.Sequential()
        Block3.add(tf.keras.layers.BatchNormalization(fused=True))
        Block3.add(tf.keras.layers.ReLU())
        Block3.add(tf.keras.layers.Conv2D(3, 3))

        inputs = tf.keras.Input((28, 28, 64))
        conv2d = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(inputs)
        block3 = Block3(conv2d)
        outputs = tf.keras.layers.BatchNormalization(fused=True)(block3)
        Block1 = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model3")

        Block2 = tf.keras.Sequential()
        Block2.add(tf.keras.layers.BatchNormalization(fused=True))
        Block2.add(tf.keras.layers.ReLU())
        Block2.add(tf.keras.layers.Conv2D(3, 3))
        Block2.add(Block1)

        class chain_model(tf.keras.Model):
            def __init__(self, kernel_num, kernel_size, strides):
                super(chain_model, self).__init__()
                # define all layers in init
                self.conv1 = tf.keras.layers.Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides)
                self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                self.dn1 = tf.keras.layers.Dense(units=32)
                self.block2 = Block2

            def call(self, input_tensor, training=False):
                # forward pass
                x = self.conv1(input_tensor)
                x = self.bn1(x)
                x = self.dn1(x)
                x = self.block2(x)
                return x

        model = chain_model(3, (3, 3), 1)

        bn_layers = list()
        bn = model.block2.get_layer(index=3).get_layer(index=2).get_layer(index=0)
        bn_layers.append(bn)
        bn = model.block2.get_layer(index=3).get_layer(index=3)
        bn_layers.append(bn)
        bn = model.bn1
        bn_layers.append(bn)

        len_layers = []
        len_layers.append(len(model.block2.get_layer(index=3).get_layer(index=2).layers))

        _delete_all_bns_from_model(model, bn_layers)

        self.assertTrue(len(model.block2.get_layer(index=3).get_layer(index=2).layers) == len_layers[0]-1)
        self.assertTrue(isinstance(model.block2.get_layer(index=3).get_layer(index=3), tf.keras.layers.BatchNormalization))
        self.assertFalse(isinstance(model.bn1, tf.keras.layers.BatchNormalization))

    def test_modify_bn_params_to_make_as_passthrough(self):
        if version.parse(tf.version.VERSION ) >= version.parse("2.00"):
            inputs = tf.keras.Input(shape=(1,))
            add_ = tf.keras.layers.Lambda(lambda x: x + 10)(inputs)
            outputs = tf.keras.layers.BatchNormalization(epsilon=0, beta_initializer='random_uniform',
                                                         gamma_initializer='random_uniform',
                                                         moving_mean_initializer='random_uniform',
                                                         moving_variance_initializer='ones')(add_)
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_model")

            BNUtils.modify_bn_params_to_make_as_passthrough(model.layers[2])

            var = tf.constant([[5.0]])
            out = model(var)

            self.assertTrue(out.numpy(), 15.0)

    def test_find_conv_bn_pairs_combined_model(self):
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            Block3 = tf.keras.Sequential()
            Block3.add(tf.keras.layers.BatchNormalization(fused=True))
            Block3.add(tf.keras.layers.Conv2D(3, 3))

            inputs = tf.keras.Input((60, 60, 3))
            conv2d = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(inputs)
            block3 = Block3(conv2d)
            outputs = tf.keras.layers.BatchNormalization(fused=True)(block3)
            Block1 = tf.keras.Model(inputs=inputs, outputs=outputs)

            Block2 = tf.keras.Sequential()
            Block2.add(tf.keras.layers.BatchNormalization(fused=True))
            Block2.add(tf.keras.layers.Conv2D(3, 3))
            Block2.add(tf.keras.layers.ReLU())
            Block2.add(Block1)

            class MyModelMLP(tf.keras.Model):
                def __init__(self, input_shape):
                    super(MyModelMLP, self).__init__()

                    self.conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=1, activation='relu')
                    self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                    self.dn1 = tf.keras.layers.Dense(units=32)
                    self.block2 = Block2
                    self.relu = tf.keras.layers.ReLU()
                    # # Parameters of the model
                    self.input_layer = tf.keras.layers.Input(input_shape)
                    self.out = self.call(self.input_layer)
                    super().__init__(inputs=self.input_layer, outputs=self.out)

                # Define forward passing of model
                def call(self, input_tensor):
                    x = self.conv1(input_tensor)
                    x = self.bn1(x)
                    x = self.dn1(x)
                    x = self.block2(x)
                    x = self.relu(x)
                    return x

            model = MyModelMLP((64,64,64))

            node_layer_map = common.create_node_to_layer_map(model)
            layer_out_node_map = common.create_layer_to_out_node_map(model)
            conv_linear_with_bn_dict = find_possible_convs_linears_bn(node_layer_map, layer_out_node_map)

            self.assertFalse(model.conv1 in conv_linear_with_bn_dict)
            self.assertFalse(model.bn1 in conv_linear_with_bn_dict)
            self.assertTrue(len(conv_linear_with_bn_dict) == 3)

    def test_find_conv_bn_pairs_functional(self):

        # tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        input2 = tf.keras.Input(name='input2', shape=(12, 12, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a')(input1)
        y = tf.keras.layers.BatchNormalization()(x1)
        x2 = tf.keras.layers.BatchNormalization()(input2)
        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b')(x2)
        x = tf.keras.layers.add([y, x2])
        x = tf.keras.layers.Conv2D(4, (1, 1), name='conv2')(x)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(x)
        output = tf.nn.relu(bn_op)
        model = tf.keras.Model([input1, input2], output)

        node_layer_map = common.create_node_to_layer_map(model)
        layer_out_node_map = common.create_layer_to_out_node_map(model)
        conv_linear_with_bn_dict = find_possible_convs_linears_bn(node_layer_map, layer_out_node_map)

        self.assertEqual(3, len(conv_linear_with_bn_dict))

    def test_find_conv_bn_pairs_sequential(self):
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            Block1 = tf.keras.Sequential()
            Block1.add(tf.keras.layers.ReLU())
            Block1.add(tf.keras.layers.BatchNormalization())
            Block1.add(tf.keras.layers.Conv2D(3, 3))

            Block2 = tf.keras.Sequential()
            Block2.add(tf.keras.Input((28, 28, 64)))
            Block2.add(tf.keras.layers.BatchNormalization(fused=True))
            Block2.add(tf.keras.layers.ReLU())
            Block2.add(tf.keras.layers.Conv2D(3, 3))
            Block2.add(Block1)
            Block2.add(tf.keras.layers.ReLU())

            node_layer_map = common.create_node_to_layer_map(Block2)
            layer_out_node_map = common.create_layer_to_out_node_map(Block2)
            conv_linear_with_bn_dict = find_possible_convs_linears_bn(node_layer_map, layer_out_node_map)

            self.assertTrue(Block1._layers[3] in conv_linear_with_bn_dict)
            self.assertEqual(1, len(conv_linear_with_bn_dict))

    def test_ordered_conv_linears_functional(self):
        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a')(input1)
        y = tf.keras.layers.BatchNormalization()(x1)
        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b')(y)
        x3 = tf.keras.layers.Conv2D(8, (3, 3), name='conv2b')(y)
        y2 = tf.keras.layers.BatchNormalization()(x2)
        y3 = tf.keras.layers.BatchNormalization()(x3)
        x = tf.keras.layers.add([y3, y2])
        x = tf.keras.layers.Conv2D(4, (1, 1), name='conv3')(x)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(x)
        output = tf.nn.relu(bn_op)
        model = tf.keras.Model(input1, output)

        node_layer_map = common.create_node_to_layer_map(model)
        layer_out_node_map = common.create_layer_to_out_node_map(model)
        ordered_conv_linears = get_ordered_conv_linears(node_layer_map, layer_out_node_map)

        for layer in model._layers:
            if layer.name == 'conv1a':
                self.assertTrue(layer == ordered_conv_linears[0])
            if layer.name == 'conv3':
                self.assertTrue(layer == ordered_conv_linears[3])

    def test_ordered_conv_linears_sequential(self):
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            Block1 = tf.keras.Sequential()
            Block1.add(tf.keras.layers.ReLU())
            Block1.add(tf.keras.layers.BatchNormalization())
            Block1.add(tf.keras.layers.Conv2D(3, 3))

            Block2 = tf.keras.Sequential()
            Block2.add(tf.keras.Input((28, 28, 64)))
            Block2.add(tf.keras.layers.BatchNormalization(fused=True))
            Block2.add(tf.keras.layers.ReLU())
            Block2.add(tf.keras.layers.Conv2D(3, 3))
            Block2.add(Block1)
            Block2.add(tf.keras.layers.ReLU())

            node_layer_map = common.create_node_to_layer_map(Block2)
            layer_out_node_map = common.create_layer_to_out_node_map(Block2)
            ordered_conv_linears = get_ordered_conv_linears(node_layer_map, layer_out_node_map)
            self.assertEqual(Block2._layers[3], ordered_conv_linears[0])

    def test_ordered_conv_linears_combined_model(self):
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            Block3 = tf.keras.Sequential()
            Block3.add(tf.keras.layers.BatchNormalization(fused=True))
            Block3.add(tf.keras.layers.Conv2D(3, 3))

            inputs = tf.keras.Input((60, 60, 3))
            conv2d = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(inputs)
            block3 = Block3(conv2d)
            outputs = tf.keras.layers.BatchNormalization(fused=True)(block3)
            Block1 = tf.keras.Model(inputs=inputs, outputs=outputs)

            Block2 = tf.keras.Sequential()
            Block2.add(tf.keras.layers.BatchNormalization(fused=True))
            Block2.add(tf.keras.layers.Conv2D(3, 3))
            Block2.add(tf.keras.layers.ReLU())
            Block2.add(Block1)

            class MyModelMLP(tf.keras.Model):
                def __init__(self, input_shape):
                    super(MyModelMLP, self).__init__()

                    self.conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=1, activation='linear')
                    self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                    self.dn1 = tf.keras.layers.Dense(units=32)
                    self.block2 = Block2
                    self.relu = tf.keras.layers.ReLU()
                    # # Parameters of the model
                    self.input_layer = tf.keras.layers.Input(input_shape)
                    self.out = self.call(self.input_layer)
                    super().__init__(inputs=self.input_layer, outputs=self.out)

                # Define forward passing of model
                def call(self, input_tensor):
                    x = self.conv1(input_tensor)
                    x = self.bn1(x)
                    x = self.dn1(x)
                    x = self.block2(x)
                    x = self.relu(x)
                    return x

            model = MyModelMLP((64,64,64))

            node_layer_map = common.create_node_to_layer_map(model)
            layer_out_node_map = common.create_layer_to_out_node_map(model)
            ordered_conv_linears = get_ordered_conv_linears(node_layer_map, layer_out_node_map)

            self.assertEqual(model.conv1, ordered_conv_linears[0])
            self.assertEqual(model.dn1, ordered_conv_linears[1])
            self.assertEqual(model.block2._layers[2], ordered_conv_linears[2])
            self.assertEqual(model.block2._layers[4]._layers[1], ordered_conv_linears[3])
            self.assertEqual(model.block2._layers[4]._layers[2]._layers[2], ordered_conv_linears[4])

    def test_find_all_bns_to_fold_sequential(self):

        Block1 = tf.keras.Sequential()
        Block1.add(tf.keras.Input((60, 60, 3)))
        Block1.add(tf.keras.layers.ReLU())
        Block1.add(tf.keras.layers.BatchNormalization())
        Block1.add(tf.keras.layers.Conv2D(3, 3))
        Block1.add(tf.keras.layers.BatchNormalization())
        Block1.add(tf.keras.layers.Conv2D(6, 6))
        Block1.add(tf.keras.layers.ReLU())

        conv_bn_pairs = find_all_batch_norms_to_fold(Block1)
        self.assertEqual(1, len(conv_bn_pairs))
        self.assertEqual((Block1._layers[3], Block1._layers[4], True), conv_bn_pairs[0])

    def test_find_all_bns_to_fold_functional(self):
        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a')(input1)
        y = tf.keras.layers.BatchNormalization()(x1)
        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b')(y)
        x3 = tf.keras.layers.Conv2D(8, (3, 3), name='conv2b')(y)
        y2 = tf.keras.layers.BatchNormalization()(x2)
        y3 = tf.keras.layers.BatchNormalization()(x3)
        x = tf.keras.layers.add([y3, y2])
        x = tf.keras.layers.Conv2D(4, (1, 1), name='conv3')(x)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(x)
        output = tf.keras.layers.ReLU()(bn_op)
        model = tf.keras.Model(input1, output)

        conv_bn_pairs = find_all_batch_norms_to_fold(model)
        self.assertEqual(4, len(conv_bn_pairs))

    def test_find_all_bns_to_fold_combined_model(self):
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            Block3 = tf.keras.Sequential()
            Block3.add(tf.keras.layers.BatchNormalization(fused=True))
            Block3.add(tf.keras.layers.Conv2D(3, 3))

            inputs = tf.keras.Input((60, 60, 3))
            conv2d = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(inputs)
            block3 = Block3(conv2d)
            outputs = tf.keras.layers.BatchNormalization(fused=True)(block3)
            Block1 = tf.keras.Model(inputs=inputs, outputs=outputs)

            Block2 = tf.keras.Sequential()
            Block2.add(tf.keras.layers.BatchNormalization(fused=True))
            Block2.add(tf.keras.layers.Conv2D(3, 3))
            Block2.add(tf.keras.layers.ReLU())
            Block2.add(Block1)

            class MyModelMLP(tf.keras.Model):
                def __init__(self, input_shape):
                    super(MyModelMLP, self).__init__()

                    self.conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=1, activation='relu')
                    self.bn1 = tf.keras.layers.BatchNormalization(fused=True)
                    self.dn1 = tf.keras.layers.Dense(units=32)
                    self.block2 = Block2
                    self.relu = tf.keras.layers.ReLU()
                    # # Parameters of the model
                    self.input_layer = tf.keras.layers.Input(input_shape)
                    self.out = self.call(self.input_layer)
                    super().__init__(inputs=self.input_layer, outputs=self.out)

                # Define forward passing of model
                def call(self, input_tensor):
                    x = self.conv1(input_tensor)
                    x = self.bn1(x)
                    x = self.dn1(x)
                    x = self.block2(x)
                    x = self.relu(x)
                    return x

            model = MyModelMLP((64,64,64))

            conv_bn_pairs = find_all_batch_norms_to_fold(model)

            self.assertEqual(3, len(conv_bn_pairs))
            self.assertEqual((model._layers[4]._layers[2], model._layers[4]._layers[1], False), conv_bn_pairs[0])
            self.assertEqual((model._layers[4]._layers[4]._layers[1], model._layers[4]._layers[4]._layers[2]._layers[1], True), conv_bn_pairs[1])
            self.assertEqual((model._layers[4]._layers[4]._layers[2]._layers[2], model._layers[4]._layers[4]._layers[3], True), conv_bn_pairs[2])

    def test_bn_fold_auto_rules_bn_after_conv(self):
        """
        Test batch norm fold layer selection when conv layer is followed by a BN layer
        """
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(conv_op)
        relu = tf.nn.relu(bn_op)
        model = tf.keras.Model(inputs = inputs, outputs = relu)

        bn_conv_linear_pairs = find_all_batch_norms_to_fold(model)
        self.assertEqual(1, len(bn_conv_linear_pairs))

    def test_bn_fold_layer_selection_looped_network(self):

        """
        Test layer selection with looped network
        """
        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a',
                                    kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                    bias_initializer='random_uniform')(input1)

        bn_op_1 = tf.keras.layers.BatchNormalization(fused=True)(x1)
        bn_op_2 = tf.keras.layers.BatchNormalization(fused=True)(x1)

        add = tf.keras.layers.add([bn_op_1, bn_op_2])

        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b',
                                   kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                   bias_initializer='random_uniform')(add)

        model = tf.keras.Model(inputs=input1, outputs=x2)

        bn_conv_linear_pairs = find_all_batch_norms_to_fold(model)

        self.assertEqual(0, len(bn_conv_linear_pairs))

    def test_bn_fold_auto_rules_bn_before_conv(self):
        """
        Test batch norm fold layer selection when BN layer is followed by a conv layer
        """
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(inputs)
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(bn_op)
        relu = tf.nn.relu(conv_op)
        model = tf.keras.Model(inputs=inputs, outputs=relu)

        bn_conv_linear_pairs = find_all_batch_norms_to_fold(model)
        self.assertEqual(1, len(bn_conv_linear_pairs))

    def test_bn_fold_find_layers_model_with_multi_input(self):
        """
        Test bn fold with multiple input nodes
        """

        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        input2 = tf.keras.Input(name='input2', shape=(12, 12, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a')(input1)
        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b')(input2)
        x = tf.keras.layers.add([x1, x2])
        x = tf.keras.layers.Conv2D(4, (1, 1), name='conv2')(x)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(x)
        relu = tf.nn.relu(bn_op)
        model = tf.keras.Model(inputs=[input1, input2], outputs=relu)

        bn_conv_linear_pairs = find_all_batch_norms_to_fold(model)
        self.assertEqual(1, len(bn_conv_linear_pairs))

    def test_bn_fold_auto_rules_conv_bn_conv(self):
        """
        Test batch norm fold layer selection with pattern conv1 - bn - conv2
        bn folds into conv1
        """
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv = tf.keras.layers.Conv2D(32, (3, 3), name ='conv1')(inputs)
        bn = tf.keras.layers.BatchNormalization(fused=True)(conv)
        conv2 = tf.keras.layers.Conv2D(32, (3, 3), name ='conv2')(bn)
        relu = tf.nn.relu(conv2)
        model = tf.keras.Model(inputs=inputs, outputs=relu)

        bn_conv_linear_pairs = find_all_batch_norms_to_fold(model)
        self.assertEqual(1, len(bn_conv_linear_pairs))
        conv_linear, batchnorm, is_batch_norm_second = bn_conv_linear_pairs[0]
        self.assertEqual('conv1', conv_linear.name)
        # add additional check to verify backward fold is picked over forward in case both are available
        self.assertEqual(True, is_batch_norm_second)

    def test_batch_norm_fold(self):
        """
        Test batch norm fold custom model
        """
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            inputs = tf.keras.Input(shape=(32, 32, 3,))
            conv = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
            bn = tf.keras.layers.BatchNormalization(fused=True)(conv, training=False)
            relu = tf.nn.relu(bn)
            model = tf.keras.Model(inputs=inputs, outputs=relu)

            np.random.seed(0)
            w_shape = model._layers[0].input.shape
            numpy_data = np.random.rand(1, w_shape[1], w_shape[2], w_shape[3]).astype(np.float32)

            baseline_output = model(numpy_data)

            fold_all_batch_norms(model)
            output_after_fold = model(numpy_data)

            self.assertTrue(np.allclose(baseline_output, output_after_fold, atol=1.e-4))

    def test_batch_norm_fold_with_random_data(self):
        """
        Test batch norm fold custom model with randomly initialized kernel, bias and bn params,
        """
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            inputs = tf.keras.Input(shape=(32, 32, 3,))
            conv = tf.keras.layers.Conv2D(32, (3, 3),
                                             kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                             bias_initializer='random_uniform')(inputs)
            bn = tf.keras.layers.BatchNormalization(fused=True,
                                                       beta_initializer='random_uniform',
                                                       gamma_initializer='random_uniform',
                                                       moving_mean_initializer='random_uniform',
                                                       moving_variance_initializer='ones')(conv, training=False)
            relu = tf.nn.relu(bn)

            model = tf.keras.Model(inputs=inputs, outputs=relu)

            np.random.seed(0)
            w_shape = model._layers[0].input.shape
            numpy_data = np.random.rand(1, w_shape[1], w_shape[2], w_shape[3]).astype(np.float32)
            baseline_output = model(numpy_data)

            fold_all_batch_norms(model)

            output_after_fold = model(numpy_data)

            self.assertFalse(np.allclose(baseline_output, output_after_fold, atol=0))
            self.assertTrue(np.allclose(baseline_output, output_after_fold, atol=1e-4))

    def test_bn_fold_with_linear_layer(self):
        """
        Custom Model where BN layer is followed by Dense layer
        :return:
        """
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            inputs = tf.keras.Input(shape=(1, 1, 4,))
            bn = tf.keras.layers.BatchNormalization(fused=True)(inputs, training=False)
            x = tf.keras.layers.Flatten()(bn)
            dense = tf.keras.layers.Dense(2, activation=tf.nn.relu, name="linear_layer")(x)
            model = tf.keras.Model(inputs=inputs, outputs=dense)

            # get baseline output
            np.random.seed(0)
            w_shape = model._layers[0].input.shape
            numpy_data = np.random.rand(1, w_shape[1], w_shape[2], w_shape[3]).astype(np.float32)
            baseline_output = model(numpy_data)
            weight_before_fold = model._layers[3].kernel.numpy()

            fold_all_batch_norms(model)
            after_fold_output = model(numpy_data)
            weight_after_fold = model._layers[3].kernel.numpy()

            # check that weight got updated
            self.assertFalse(np.allclose(weight_before_fold, weight_after_fold, atol=1e-4))

            # check outputs are close
            self.assertTrue(np.allclose(baseline_output, after_fold_output, atol=1e-4))

    def test_find_conv_bn_pairs_functional_nested(self):
        """
        Test case for finding conv-bn pairs when the inner model has 2 layers connected to the input

        """
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            inputs = tf.keras.Input((26, 26, 3))
            conv2d_1 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(inputs)
            bn = tf.keras.layers.BatchNormalization(fused=True)(inputs)
            conv2d_2 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1)(bn)
            outputs = tf.keras.layers.add([conv2d_1, conv2d_2])
            Block1 = tf.keras.Model(inputs=inputs, outputs=outputs)

            inputs2 = tf.keras.Input((28, 28, 64))
            bn1 = tf.keras.layers.BatchNormalization(fused=True)(inputs2)
            relu = tf.keras.layers.ReLU()(bn1)
            conv2d_0 = tf.keras.layers.Conv2D(3, 3)(relu)
            block1 = Block1(conv2d_0)
            outputs = tf.keras.layers.ReLU()(block1)
            model = tf.keras.Model(inputs=inputs2, outputs=outputs)


            node_layer_map = common.create_node_to_layer_map(model)
            layer_out_node_map = common.create_layer_to_out_node_map(model)
            conv_linear_with_bn_dict = find_possible_convs_linears_bn(node_layer_map, layer_out_node_map)

            self.assertEqual(10,len(node_layer_map))
            self.assertEqual(9, len(layer_out_node_map))
            self.assertEqual(1, len(conv_linear_with_bn_dict))
