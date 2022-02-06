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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import unittest
import unittest.mock
import math
import shutil
from decimal import Decimal
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Reshape, MaxPool2D, Conv2D, Flatten, Dense

from aimet_common.utils import AimetLogger
from aimet_common import cost_calculator as cc
from aimet_common.defs import CostMetric, LayerCompRatioPair
from aimet_tensorflow.layer_database import LayerDatabase, Layer
from aimet_tensorflow.examples import mnist_tf_model, test_models
from aimet_tensorflow.channel_pruning.channel_pruner import InputChannelPruner, ChannelPruningCostCalculator

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()


def mnist(data_format):

    # pylint: disable=no-member

    if data_format == 'channels_first':
        input_shape = [1, 28, 28]
    else:
        assert data_format == 'channels_last'
        input_shape = [28, 28, 1]

    return Sequential(
        [
            Reshape(target_shape=input_shape, input_shape=(28 * 28,)),
            Conv2D(32, 5, padding='same', data_format=data_format, activation=tf.nn.relu,
                   kernel_initializer='random_uniform'),
            MaxPool2D((2, 2), (2, 2), padding='same', data_format=data_format),
            Conv2D(64, 5, padding='same', data_format=data_format, activation=tf.nn.relu,
                   kernel_initializer='random_uniform'),
            MaxPool2D((2, 2), (2, 2), padding='same', data_format=data_format),
            Flatten(),
            Dense(1024, activation=tf.nn.relu, kernel_initializer='random_uniform'),
            Dense(10, kernel_initializer='random_uniform')
        ])


class TestTrainingExtensionsCostCalculator(unittest.TestCase):

    def test_compute_layer_cost(self):

        logger.debug(self.id())

        # data format NCHW
        inp_tensor = tf.Variable(tf.random.normal([1, 1, 28, 28]))
        filter_tensor = tf.Variable(tf.random.normal([5, 5, 1, 32]))
        conv1 = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 1, 1, 1], padding='SAME',
                             data_format="NCHW", name='Conv2D_1')

        conv_op1 = tf.compat.v1.get_default_graph().get_operation_by_name('Conv2D_1')

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        shape = conv_op1.outputs[0].get_shape().as_list()
        # data format NCHW
        self.assertEqual(shape, [1, 32, 28, 28])

        layer1 = Layer(model=sess, op=conv_op1, output_shape=shape)

        cost1 = cc.CostCalculator.compute_layer_cost(layer1)

        self.assertEqual(32 * 1 * 5 * 5, cost1.memory)
        self.assertEqual(32 * 1 * 5 * 5 * 28 * 28, cost1.mac)

        tf.compat.v1.reset_default_graph()
        sess.close()

        # second conv2 op with strides
        # data format NCHW
        inp_tensor = tf.Variable(tf.random.normal([1, 32, 28, 28]))
        filter_tensor = tf.Variable(tf.random.normal([5, 5, 32, 64]))

        conv2 = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 1, 2, 2], padding='SAME',
                             data_format="NCHW", name='Conv2D_2')

        conv_op2 = tf.compat.v1.get_default_graph().get_operation_by_name('Conv2D_2')

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        shape = conv_op2.outputs[0].get_shape().as_list()
        # data format NCHW
        self.assertEqual(shape, [1, 64, 14, 14])

        layer2 = Layer(model=sess, op=conv_op2, output_shape=shape)

        cost1 = cc.CostCalculator.compute_layer_cost(layer2)

        self.assertEqual(64 * 32 * 5 * 5, cost1.memory)
        self.assertEqual(64 * 32 * 5 * 5 * 14 * 14, cost1.mac)

        tf.compat.v1.reset_default_graph()
        sess.close()

    def test_total_model_cost(self):

        # create tf.compat.v1.Session and initialize the weights and biases with zeros
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # by default, model will be constructed in default graph
            _ = mnist_tf_model.create_model(data_format='channels_last')
            sess.run(tf.compat.v1.global_variables_initializer())

        layer_database = LayerDatabase(model=sess, input_shape=(1, 28, 28, 1), working_dir=None)

        cost_calc = cc.CostCalculator()
        network_cost = cost_calc.compute_model_cost(layer_database)

        self.assertEqual(800 + 51200 + 3211264 + 10240, network_cost.memory)
        self.assertEqual(627200 + 10035200 + 3211264 + 10240, network_cost.mac)

        tf.compat.v1.reset_default_graph()
        sess.close()
        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))


class TestTrainingExtensionsSpatialSvdCostCalculator(unittest.TestCase):

    def test_calculate_spatial_svd_cost(self):

        # data format NCHW
        inp_tensor = tf.Variable(tf.random.normal([1, 32, 28, 28]))
        filter_tensor = tf.Variable(tf.random.normal([5, 5, 32, 64]))
        conv = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 1, 1, 1], padding='SAME',
                            data_format="NCHW", name='Conv2D')

        conv_op = tf.compat.v1.get_default_graph().get_operation_by_name('Conv2D')

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        shape = conv_op.outputs[0].get_shape().as_list()
        # data format NCHW
        self.assertEqual(shape, [1, 64, 28, 28])

        layer = Layer(model=sess, op=conv_op, output_shape=shape)

        self.assertEqual(32 * 5, cc.SpatialSvdCostCalculator.calculate_max_rank(layer))

        comp_ratios_to_check = [0.8, 0.75, 0.5, 0.25, 0.125]

        original_cost = cc.CostCalculator.compute_layer_cost(layer)

        for comp_ratio in comp_ratios_to_check:

            rank = cc.SpatialSvdCostCalculator.calculate_rank_given_comp_ratio(layer, comp_ratio, CostMetric.mac)
            print('Rank = {}, for compression_ratio={}'.format(rank, comp_ratio))
            compressed_cost = cc.SpatialSvdCostCalculator.calculate_cost_given_rank(layer, rank)

            self.assertTrue(math.isclose(compressed_cost.mac/original_cost.mac, comp_ratio, abs_tol=0.01))

        # Higher level API
        for comp_ratio in comp_ratios_to_check:

            compressed_cost = cc.SpatialSvdCostCalculator.calculate_per_layer_compressed_cost(layer, comp_ratio,
                                                                                              CostMetric.mac)

            self.assertTrue(math.isclose(compressed_cost.mac/original_cost.mac, comp_ratio, abs_tol=0.01))

        tf.compat.v1.reset_default_graph()
        sess.close()

    def test_calculate_spatial_svd_cost_with_stride(self):

        # data format : NHWC
        inp_tensor = tf.Variable(tf.random.normal([1, 28, 28, 32]))
        filter_tensor = tf.Variable(tf.random.normal([5, 5, 32, 64]))
        conv = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 2, 2, 1], padding='SAME',
                            data_format="NHWC", name='Conv2D')

        conv_op = tf.compat.v1.get_default_graph().get_operation_by_name('Conv2D')

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        shape = conv_op.outputs[0].get_shape().as_list()
        # data format : NHWC
        self.assertEqual(shape, [1, 14, 14, 64])

        # but layer  expects output shape in NCHW format similar to PyTorch
        shape = (shape[0], shape[3], shape[1], shape[2])
        layer = Layer(model=sess, op=conv_op, output_shape=shape)

        original_cost = cc.CostCalculator.compute_layer_cost(layer)
        compressed_cost = cc.SpatialSvdCostCalculator.calculate_cost_given_rank(layer, 40)

        self.assertEqual(10035200, original_cost.mac)
        self.assertEqual(5017600, compressed_cost.mac)

        tf.compat.v1.reset_default_graph()
        sess.close()

    def test_calculate_spatial_svd_cost_all_layers(self):

        # create tf.compat.v1.Session and initialize the weights and biases with zeros
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # by default, model will be constructed in default graph
            _ = mnist_tf_model.create_model(data_format='channels_last')
            sess.run(tf.compat.v1.global_variables_initializer())

        layer_database = LayerDatabase(model=sess, input_shape=(1, 28, 28, 1), working_dir=None)

        # Compress all layers by 50%

        # Create a list of tuples of (layer, comp_ratio)
        layer_ratio_list = []

        for layer in layer_database:

            if layer.module.type == 'Conv2D':
                layer_ratio_list.append(LayerCompRatioPair(layer, Decimal(0.5)))
            else:
                layer_ratio_list.append(LayerCompRatioPair(layer, None))

        compressed_cost = cc.SpatialSvdCostCalculator.calculate_compressed_cost(layer_database,
                                                                                layer_ratio_list, CostMetric.mac)

        self.assertEqual(8466464, compressed_cost.mac)

        tf.compat.v1.reset_default_graph()
        sess.close()
        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))


class TestTrainingExtensionsChannelPruningCostCalculator(unittest.TestCase):

    def test_calculate_channel_pruning_cost_all_layers(self):

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # model will be constructed in default graph
            _ = mnist(data_format='channels_last')
            # initialize the weights and biases with appropriate initializer
            sess.run(tf.compat.v1.global_variables_initializer())

        meta_path = str('./temp_working_dir/')
        if not os.path.exists(meta_path):
            os.mkdir(meta_path)

        layer_db = LayerDatabase(model=sess, input_shape=(1, 28, 28, 1), working_dir=meta_path)

        # Compress all layers by 50%

        # Create a list of tuples of (layer, comp_ratio)
        layer_ratio_list = []

        # Unfortunately in mnist we can only input channel prune conv2d_1/Conv2D
        for layer in layer_db:
            if layer.module.name == 'conv2d_1/Conv2D':
                layer_ratio_list.append(LayerCompRatioPair(layer, Decimal('0.5')))
            else:
                layer_ratio_list.append(LayerCompRatioPair(layer, None))

        inp_op_names = ['reshape_input']
        output_op_names = ['dense_1/BiasAdd']

        data_set = unittest.mock.MagicMock()
        batch_size = unittest.mock.MagicMock()
        num_reconstruction_samples = unittest.mock.MagicMock()

        pruner = InputChannelPruner(input_op_names=inp_op_names, output_op_names=output_op_names, data_set=data_set,
                                    batch_size=batch_size, num_reconstruction_samples=num_reconstruction_samples,
                                    allow_custom_downsample_ops=True)

        cost_calculator = ChannelPruningCostCalculator(pruner)

        compressed_cost = cost_calculator.calculate_compressed_cost(layer_db,
                                                                    layer_ratio_list, CostMetric.mac)

        self.assertEqual(8552704, compressed_cost.mac)
        self.assertEqual(3247504, compressed_cost.memory)

        # delete the meta and the checkpoint files
        shutil.rmtree(meta_path)

        layer_db.model.close()

    def test_calculate_channel_pruning_cost_two_layers(self):
        """
        test compressed model cost using two layers
        :return:
        """
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # model will be constructed in default graph
            test_models.single_residual()
            init = tf.compat.v1.global_variables_initializer()

        # initialize the weights and biases with appropriate initializer
        sess.run(init)

        meta_path = str('./temp_working_dir/')

        if not os.path.exists(meta_path):
            os.mkdir(meta_path)

        layer_db = LayerDatabase(model=sess, input_shape=None, working_dir=meta_path)

        # Create a list of tuples of (layer, comp_ratio)
        layer_ratio_list = []

        layer_names = ['conv2d_2/Conv2D', 'conv2d_3/Conv2D']
        for layer in layer_db:
            if layer.module.name in layer_names:
                layer_ratio_list.append(LayerCompRatioPair(layer, 0.5))
            else:
                layer_ratio_list.append(LayerCompRatioPair(layer, None))

        input_op_names = ['input_1']
        output_op_names = ['single_residual/Softmax']
        data_set = unittest.mock.MagicMock()
        batch_size = unittest.mock.MagicMock()
        num_reconstruction_samples = unittest.mock.MagicMock()

        pruner = InputChannelPruner(input_op_names=input_op_names, output_op_names=output_op_names, data_set=data_set,
                                    batch_size=batch_size, num_reconstruction_samples=num_reconstruction_samples,
                                    allow_custom_downsample_ops=True)

        cost_calculator = ChannelPruningCostCalculator(pruner)

        compressed_cost = cost_calculator.calculate_compressed_cost(layer_db, layer_ratio_list, CostMetric.mac)

        self.assertEqual(108544, compressed_cost.mac)
        self.assertEqual(1264, compressed_cost.memory)

        # delete the meta and the checkpoint files
        shutil.rmtree(meta_path)

        layer_db.model.close()
