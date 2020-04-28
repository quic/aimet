# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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

import os
import copy
import shutil
import unittest
import numpy as np
import tensorflow as tf

from aimet_common.utils import AimetLogger
from aimet_tensorflow.layer_database import LayerDatabase
from aimet_tensorflow.examples import mnist_tf_model

from tensorflow.keras.applications.resnet50 import ResNet50

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class TestTensorFlowLayerDatabase(unittest.TestCase):

    def test_layer_database_with_mnist(self):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # model will be constructed in default graph
            _ = mnist_tf_model.create_model(data_format='channels_last')
            # initialize the weights and biases with appropriate initializer
            sess.run(tf.global_variables_initializer())

        layer_db = LayerDatabase(model=sess, working_dir=None)

        layers = list(layer_db._compressible_layers.values())

        # check output shapes
        self.assertEqual(layers[0].output_shape, [None, 32, 28, 28])
        self.assertEqual(layers[1].output_shape, [None, 64, 14, 14])
        self.assertEqual(layers[2].output_shape, [None, 1024, 1, 1])
        self.assertEqual(layers[3].output_shape, [None, 10, 1, 1])

        # check weight shapes
        # layer weight_shape is in common format [Noc, Nic, k_h, k_w]
        self.assertEqual(layers[0].weight_shape, [32, 1, 5, 5])
        self.assertEqual(layers[1].weight_shape, [64, 32, 5, 5])
        self.assertEqual(layers[2].weight_shape, [1024, 3136])
        self.assertEqual(layers[3].weight_shape, [10, 1024])

        # tensorflow weights are stored in [k_h, k_w, Nic, Noc]
        self.assertEqual(layers[0].module.inputs[1].eval(session=layer_db.model).shape, (5, 5, 1, 32))
        self.assertEqual(layers[1].module.inputs[1].eval(session=layer_db.model).shape, (5, 5, 32, 64))
        self.assertEqual(layers[2].module.inputs[1].eval(session=layer_db.model).shape, (3136, 1024))
        self.assertEqual(layers[3].module.inputs[1].eval(session=layer_db.model).shape, (1024, 10))

        tf.reset_default_graph()
        sess.close()
        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))

    def test_layer_database_deepcopy(self):

        # create tf.Session and initialize the weights and biases with zeros
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # by default, model will be constructed in default graph
            _ = mnist_tf_model.create_model(data_format='channels_last')
            sess.run(tf.global_variables_initializer())

        layer_db = LayerDatabase(model=sess, working_dir=None)
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
        self.assertEqual(layers_copy[0].weight_shape, [32, 1, 5, 5])
        self.assertEqual(layers_copy[1].weight_shape, [64, 32, 5, 5])
        self.assertEqual(layers_copy[2].weight_shape, [1024, 3136])
        self.assertEqual(layers_copy[3].weight_shape, [10, 1024])

        # check the weight elements are equal in both data bases
        self.assertTrue(np.array_equal(layers[0].module.inputs[1].eval(session=layer_db.model),
                                       layers_copy[0].module.inputs[1].eval(session=layer_db_copy.model)))
        self.assertTrue(np.array_equal(layers[1].module.inputs[1].eval(session=layer_db.model),
                                       layers_copy[1].module.inputs[1].eval(session=layer_db_copy.model)))
        self.assertTrue(np.array_equal(layers[2].module.inputs[1].eval(session=layer_db.model),
                                       layers_copy[2].module.inputs[1].eval(session=layer_db_copy.model)))
        self.assertTrue(np.array_equal(layers[3].module.inputs[1].eval(session=layer_db.model),
                                       layers_copy[3].module.inputs[1].eval(session=layer_db_copy.model)))

        tf.reset_default_graph()
        sess.close()
        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))

    def test_layer_database_working_dir(self):

        # create tf.Session and initialize the weights and biases with zeros
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # by default, model will be constructed in default graph
            _ = mnist_tf_model.create_model(data_format='channels_last')
            sess.run(tf.global_variables_initializer())

        meta_path = str('./temp_working_dir/')

        if not os.path.exists(meta_path):
            os.mkdir(meta_path)

        layer_db = LayerDatabase(model=sess, working_dir=meta_path)
        copy_layer_db = copy.deepcopy(layer_db)

        shutil.rmtree(meta_path)

        tf.reset_default_graph()

        layer_db.model.close()
        copy_layer_db.model.close()

    @unittest.skip
    def test_layer_database_cpu_memory_Leak(self):

        # create tf.Session and initialize the weights and biases with zeros
        import psutil
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # by default, model will be constructed in default graph
            _ = ResNet50(weights=None)
            init = tf.global_variables_initializer()

        sess.run(init)
        layer_db = LayerDatabase(model=sess, working_dir=None)

        mem_before = psutil.virtual_memory().available

        for i in range(5):

            copy_layer_db = copy.deepcopy(layer_db)
            copy_layer_db.model.close()

        mem_after = psutil.virtual_memory().available

        memory_consumed = (mem_before - mem_after) / (1024 * 1024)

        print('Memory consumed in MB:', memory_consumed, 'for iterations : ', 5)

        self.assertTrue(memory_consumed < 200)
        layer_db.model.close()

    def test_layer_database_destroy(self):

        # create tf.Session and initialize the weights and biases with zeros
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # by default, model will be constructed in default graph
            _ = mnist_tf_model.create_model(data_format='channels_last')
            init = tf.global_variables_initializer()

        sess.run(init)
        layer_db = LayerDatabase(model=sess, working_dir=None)

        layer_db.destroy()

        self.assertRaises(RuntimeError, lambda: sess.run(init))
        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))



