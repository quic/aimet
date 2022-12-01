# /usr/bin/env python3.5
# -*- mode: python -*-
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

import unittest
import numpy as np
import copy
import tensorflow as tf
from packaging import version

from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.layer_database import LayerDatabase, Layer
from aimet_tensorflow.examples import mnist_tf_model


class TestTensorFlowLayerDatabase(unittest.TestCase):

    def test_layer_database_with_mnist(self):
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            tf.keras.backend.clear_session()
            
            # Get the basic mnist model
            model = mnist_tf_model.create_model(data_format='channels_last')

            layer_db = LayerDatabase(model=model)

            layers = list(layer_db._compressible_layers.values())

            # check output shapes
            self.assertEqual(layers[0].output_shape, [None, 32, 28, 28])
            self.assertEqual(layers[1].output_shape, [None, 64, 14, 14])
            self.assertEqual(layers[2].output_shape, [None, 1024, 1, 1])
            self.assertEqual(layers[3].output_shape, [None, 10, 1, 1])

            # check weight shapes
            # layer weight_shape is in common format [Noc, Nic, k_h, k_w]
            self.assertEqual(layers[0].weight_shape, (32, 1, 5, 5))
            self.assertEqual(layers[1].weight_shape, (64, 32, 5, 5))
            self.assertEqual(layers[2].weight_shape, (1024, 3136, 1, 1))
            self.assertEqual(layers[3].weight_shape, (10, 1024, 1, 1))

            # tensorflow weights are stored in [k_h, k_w, Nic, Noc]
            self.assertEqual(layers[0].module.get_weights()[0].shape, (5, 5, 1, 32))
            self.assertEqual(layers[1].module.get_weights()[0].shape, (5, 5, 32, 64))
            self.assertEqual(layers[2].module.get_weights()[0].shape, (3136, 1024))
            self.assertEqual(layers[3].module.get_weights()[0].shape, (1024, 10))

    def test_layer_database_deepcopy(self):
    
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            tf.keras.backend.clear_session()


            # Get the basic mnist model
            model = mnist_tf_model.create_model(data_format='channels_last')

            layer_db = LayerDatabase(model=model)
            layers = list(layer_db._compressible_layers.values())

            # set the picked_for_compression for first two layers
            layers[0].picked_for_compression = True
            layers[1].picked_for_compression = True

            layer_db_copy = copy.deepcopy(layer_db)
            layers_copy = list(layer_db_copy._compressible_layers.values())

            # should be True
            self.assertTrue(layers_copy[0].picked_for_compression)
            self.assertTrue(layers_copy[1].picked_for_compression)

            # should be False
            self.assertFalse(layers_copy[2].picked_for_compression)
            self.assertFalse(layers_copy[3].picked_for_compression)

            # op reference should be different
            self.assertNotEqual(layers[0].module, layers_copy[0].module)
            self.assertNotEqual(layers[1].module, layers_copy[1].module)
            self.assertNotEqual(layers[2].module, layers_copy[2].module)
            self.assertNotEqual(layers[3].module, layers_copy[3].module)

            # layer reference should be different
            self.assertNotEqual(layers[0], layers_copy[0])
            self.assertNotEqual(layers[1], layers_copy[1])
            self.assertNotEqual(layers[2], layers_copy[2])
            self.assertNotEqual(layers[3], layers_copy[3])

            # op name should be same for both original and copy
            self.assertEqual(layers[0].name, layers_copy[0].name)
            self.assertEqual(layers[1].name, layers_copy[1].name)
            self.assertEqual(layers[2].name, layers_copy[2].name)
            self.assertEqual(layers[3].name, layers_copy[3].name)

            # session should be different
            self.assertNotEqual(layer_db.model, layer_db_copy.model)

            # check output shapes for layers in layer_db_copy
            self.assertEqual(layers_copy[0].output_shape, [None, 32, 28, 28])
            self.assertEqual(layers_copy[1].output_shape, [None, 64, 14, 14])
            self.assertEqual(layers_copy[2].output_shape, [None, 1024, 1, 1])
            self.assertEqual(layers_copy[3].output_shape, [None, 10, 1, 1])

            # check weight shapes for layers in layer_db_copy
            self.assertEqual(layers_copy[0].weight_shape, (32, 1, 5, 5))
            self.assertEqual(layers_copy[1].weight_shape, (64, 32, 5, 5))
            self.assertEqual(layers_copy[2].weight_shape, (1024, 3136, 1, 1))
            self.assertEqual(layers_copy[3].weight_shape, (10, 1024, 1, 1))

            # check the weight elements are equal in both data bases
            self.assertTrue(np.array_equal(layers[0].module.get_weights()[0].shape,
                                           layers_copy[0].module.get_weights()[0].shape))
            self.assertTrue(np.array_equal(layers[1].module.get_weights()[0].shape,
                                           layers_copy[1].module.get_weights()[0].shape))
            self.assertTrue(np.array_equal(layers[2].module.get_weights()[0].shape,
                                           layers_copy[2].module.get_weights()[0].shape))
            self.assertTrue(np.array_equal(layers[3].module.get_weights()[0].shape,
                                           layers_copy[3].module.get_weights()[0].shape))


    def test_layer_database_destroy(self):
    
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            tf.keras.backend.clear_session()

            model = mnist_tf_model.create_model(data_format='channels_last')

            layer_db = LayerDatabase(model=model)

            layer_db.destroy()

            self.assertEqual(layer_db.model, None)
            self.assertEqual(len(layer_db._compressible_layers.values()), 0)


    def test_replace_layer_with_sequential_of_two_layers(self):
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            tf.keras.backend.clear_session()
            
            # Load the basic mnist model
            model = mnist_tf_model.create_model(data_format='channels_last')

            layer_db = LayerDatabase(model)


            # Get the second conv layer to split and replcae
            layer_to_replace = list(layer_db._compressible_layers.values())[1]
            
            # Create a new layers split which will replcae the original second conv layer

            split_a = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 1),
                                             data_format='channels_last',
                                             activation=None, padding='same',
                                             name=layer_to_replace.module.name + '_a', use_bias=False)

            split_b = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 5),
                                             data_format='channels_last',
                                             activation='relu', padding='same',
                                             name=layer_to_replace.module.name + '_b', use_bias=True)

            split_a_op = split_a(model.layers[2].output)
            _ = split_b(split_a_op)
            
            
            # Create Layer for new conv layers and replace them with original layer
            layer_a = Layer(layer=split_a,name=split_a.name, output_shape=split_a.output_shape)
            layer_b = Layer(layer=split_b,name=split_b.name, output_shape=split_b.output_shape)

            layer_db.replace_layer_with_sequential_of_two_layers(layer_to_replace, layer_a, layer_b)

            layers = list(layer_db._compressible_layers.values())

            self.assertFalse(layer_to_replace in layers)
            self.assertTrue(layer_a in layers)
            self.assertTrue(layer_b in layers)
