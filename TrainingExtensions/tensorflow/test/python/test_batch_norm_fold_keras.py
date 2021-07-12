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

import unittest
import tensorflow as tf
from packaging import version

from aimet_tensorflow.keras.batch_norm_fold import _delete_bn_from_model

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

        model = MyModel(3,(3,3),1)

        bn_layers = [model.bn1, model.block1.block2.bn1, model.block2.bn1]

        _delete_bn_from_model(model, bn_layers)

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

        model = MyModel(3,(3,3),1)
        model.build((1,6,6,3))

        bn_layers = [model.bn1, model.block1.block2.bn1, model.block2.bn1]

        # ref_name = {}
        # module_name_reference(ref_name, model)

        _delete_bn_from_model(model, bn_layers)

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
        _delete_bn_from_model(chain_model, bn_layers)

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

        _delete_bn_from_model(model, bn_layers)

        self.assertTrue(len(model.block2.layers) == len_layers[0]-1)
        self.assertFalse(isinstance(model.block2.get_layer(index=2).bn1, tf.keras.layers.BatchNormalization))
        self.assertFalse(isinstance(model.block2.get_layer(index=2).block3.bn1, tf.keras.layers.BatchNormalization))

