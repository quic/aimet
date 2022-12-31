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

import pytest
import unittest
import unittest.mock
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import itertools
import copy
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50

import aimet_tensorflow.utils.graph_saver
import aimet_tensorflow.utils.op.conv
from aimet_common.defs import CostMetric, LayerCompRatioPair
from aimet_common.utils import AimetLogger
from aimet_common.input_match_search import InputMatchSearch
from aimet_tensorflow.channel_pruning.data_subsampler import DataSubSampler
from aimet_tensorflow.channel_pruning.channel_pruner import InputChannelPruner
from aimet_tensorflow.channel_pruning.weight_reconstruction import WeightReconstructor
from aimet_tensorflow.layer_database import Layer, LayerDatabase

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
tf.compat.v1.disable_eager_execution()


class TestTrainingExtensionsChannelPruning(unittest.TestCase):

    def test_get_activation_data_keras_vgg16(self):
        """
        Test to collect activations and compare for Keras model
        """
        tf.compat.v1.reset_default_graph()

        g = tf.Graph()
        with g.as_default():
            _ = VGG16(weights=None, input_shape=(224, 224, 3))
            init = tf.compat.v1.global_variables_initializer()

        inp_op_names = ['input_1']
        conv = g.get_operation_by_name('block1_conv1/Conv2D')

        batch_size = 2
        input_data = np.random.rand(100, 224, 224, 3)

        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.batch(batch_size=batch_size)

        sess = tf.compat.v1.Session(graph=g)
        sess.run(init)

        # Mock out layer_db and layer
        layer_db_mock = unittest.mock.MagicMock()
        layer_db_mock.model = sess
        layer_mock = unittest.mock.MagicMock()
        layer_mock.module = conv

        inp_data, out_data = DataSubSampler.get_sub_sampled_data(layer_mock, layer_mock, inp_op_names,
                                                                 layer_db_mock, layer_db_mock, dataset, batch_size,
                                                                 num_reconstruction_samples=1000)

        self.assertEqual(list(inp_data.shape), [1000, 3, 3, 3])
        self.assertEqual(list(out_data.shape), [1000, 64])

        sess.close()

    def test_get_activation_data_keras_vgg16_with_additional_samples(self):
        """
        Test to collect activations and compare for Keras model with additional num_reconstruction_samples
        """
        tf.compat.v1.reset_default_graph()

        g = tf.Graph()
        with g.as_default():
            _ = VGG16(weights=None, input_shape=(224, 224, 3))
            init = tf.compat.v1.global_variables_initializer()

        inp_op_names = ['input_1']
        conv = g.get_operation_by_name('block1_conv1/Conv2D')

        batch_size = 2
        input_data = np.random.rand(100, 224, 224, 3)

        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.batch(batch_size=batch_size)

        sess = tf.compat.v1.Session(graph=g)
        sess.run(init)

        # Mock out layer_db and layer
        layer_db_mock = unittest.mock.MagicMock()
        layer_db_mock.model = sess
        layer_mock = unittest.mock.MagicMock()
        layer_mock.module = conv
        # num_reconstruction_samples=1010 > possible samples (100 * 10)
        self.assertRaises(StopIteration,
                          lambda: DataSubSampler.get_sub_sampled_data(layer_mock, layer_mock,
                                                                      inp_op_names, layer_db_mock,
                                                                      layer_db_mock, dataset,
                                                                      batch_size, num_reconstruction_samples=1010))
        sess.close()

    # Need to mark this for CUDA because TF CPU Conv does not support NCHW
    @pytest.mark.cuda
    @pytest.mark.skip('Skip while investigating intermittent failure')
    def test_find_input_match_for_pixel_from_output_data_baseline_channels_first(self):
        """
        Test find input match for output pixel implementation with channels_first (NCHW) format
        """
        tf.compat.v1.reset_default_graph()

        strides = [[1, 1], [2, 2], [1, 2], [2, 1]]
        kernel_size_options = [[1, 1], [2, 2], [3, 3], [1, 3], [3, 1]]
        padding_options = ['SAME', 'VALID']
        # test middle and border values
        size_options = [[5, 5], [0, 0], [3, 3]]

        all_options = [kernel_size_options, padding_options, size_options, strides]

        for kernel_size, padding, size_opt, stride in itertools.product(*all_options):

            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]

            if isinstance(stride, list) and len(stride) == 2:
                height, width = [size_opt[0]//stride[0], size_opt[1] // stride[1]]

            else:
                height, width = [size // stride for size in size_opt]

            output_data_pixel = (height, width)

            input_data = np.array(range(8 * 8)).reshape([1, 1, 8, 8])
            filter_data = np.ones([kernel_size[0], kernel_size[1], 1, 1], dtype=np.float32)

            g = tf.Graph()
            with g.as_default():
                inp_tensor = tf.Variable(initial_value=input_data, name='inp_tensor', dtype=tf.float32)
                filter_tensor = tf.Variable(initial_value=filter_data, name='filter_tensor', dtype=tf.float32)
                _ = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 1, stride[0], stride[1]],
                                 padding=padding, data_format="NCHW", name='Conv2D_1')
                init = tf.compat.v1.global_variables_initializer()

            conv1_op = g.get_operation_by_name('Conv2D_1')
            layer_attributes = aimet_tensorflow.utils.op.conv.get_layer_attributes(sess=None, op=conv1_op,
                                                                                   input_op_names=None,
                                                                                   input_shape=None)

            input_match = InputMatchSearch._find_input_match_for_output_pixel(input_data[0], layer_attributes,
                                                                              output_data_pixel)

            sess = tf.compat.v1.Session(graph=g)
            sess.run(init)

            conv2d_out = sess.run(conv1_op.outputs[0])

            predicted_output = int(np.sum(input_match))
            generated_output = int(conv2d_out[0, 0, height, width])
            print('generated output: ', generated_output)
            print('predicted output: ', predicted_output)

            self.assertEqual(generated_output, predicted_output)
            self.assertTrue(np.prod(input_match.shape) == kernel_size[0] * kernel_size[1])

            sess.close()

    @pytest.mark.cuda
    @pytest.mark.skip('Skip while investigating TF GPU unit test failure')
    def test_find_input_match_for_pixel_from_output_data_baseline_channels_last(self):
        """
        Test find input match for output pixel implementation with channels_last (NHWC) format
        """
        tf.compat.v1.reset_default_graph()

        strides = [[1, 1], [2, 2], [1, 2], [2, 1]]
        kernel_size_options = [[1, 1], [2, 2], [3, 3], [1, 3], [3, 1]]
        padding_options = ['SAME', 'VALID']
        # test middle and border values
        size_options = [[5, 5], [0, 0], [3, 3]]

        all_options = [kernel_size_options, padding_options, size_options, strides]

        for kernel_size, padding, size_opt, stride in itertools.product(*all_options):

            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]

            if isinstance(stride, list) and len(stride) == 2:
                height, width = [size_opt[0]//stride[0], size_opt[1] // stride[1]]

            else:
                height, width = [size // stride for size in size_opt]

            output_data_pixel = (height, width)

            input_data = np.array(range(8 * 8)).reshape([1, 8, 8, 1])
            filter_data = np.ones([kernel_size[0], kernel_size[1], 1, 1], dtype=np.float32)

            g = tf.Graph()
            with g.as_default():
                inp_tensor = tf.Variable(initial_value=input_data, name='inp_tensor', dtype=tf.float32)
                filter_tensor = tf.Variable(initial_value=filter_data, name='filter_tensor', dtype=tf.float32)
                _ = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, stride[0], stride[1], 1],
                                 padding=padding, data_format="NHWC", name='Conv2D_1')
                init = tf.compat.v1.global_variables_initializer()

            conv1_op = g.get_operation_by_name('Conv2D_1')
            layer_attributes = aimet_tensorflow.utils.op.conv.get_layer_attributes(sess=None, op=conv1_op,
                                                                                   input_op_names=None,
                                                                                   input_shape=None)

            # reshape input_data, output_data function expects activations in channels_first format
            input_data = input_data.reshape(1, 1, 8, 8)
            input_match = InputMatchSearch._find_input_match_for_output_pixel(input_data[0], layer_attributes,
                                                                              output_data_pixel)

            sess = tf.compat.v1.Session(graph=g)
            sess.run(init)

            conv2d_out = sess.run(conv1_op.outputs[0])

            predicted_output = int(np.sum(input_match))
            generated_output = int(conv2d_out[0, height, width, 0])
            print('generated output: ', generated_output)
            print('predicted output: ', predicted_output)

            self.assertEqual(generated_output, predicted_output)
            self.assertTrue(np.prod(input_match.shape) == kernel_size[0] * kernel_size[1])

            sess.close()

    @unittest.mock.patch('numpy.random.choice')
    def test_subsample_data_channels_first(self, np_choice_function):
        """
        Test to subsample input match for random output pixel (1, 1) and corresponding input match
        """
        tf.compat.v1.reset_default_graph()

        # randomly selected output pixel (height, width) is fixed here and it is (1, 1)
        np_choice_function.return_value = [1]

        # input_data and output_data are in channels_first format, similar to Pytorch format
        input_data = np.arange(0, 1440).reshape((2, 5, 12, 12))
        output_data = np.arange(0, 1280).reshape((2, 10, 8, 8))

        g = tf.Graph()

        with g.as_default():
            inp_tensor = tf.Variable(initial_value=input_data, name='inp_tensor', dtype=tf.float32)
            filter_tensor = tf.compat.v1.get_variable('filter_tensor', shape=[5, 5, 5, 10],
                                            initializer=tf.random_normal_initializer())
            conv1 = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 1, 1, 1], padding='VALID',
                                 data_format="NCHW", name='Conv2D_1')

        conv1_op = g.get_operation_by_name('Conv2D_1')

        layer_attributes = aimet_tensorflow.utils.op.conv.get_layer_attributes(sess=None, op=conv1_op,
                                                                               input_op_names=None,
                                                                               input_shape=None)
        sub_sample_input, sub_sample_output = InputMatchSearch.subsample_data(layer_attributes=layer_attributes,
                                                                              input_data=input_data,
                                                                              output_data=output_data,
                                                                              samples_per_image=1)

        # compare the inputs for both batches
        self.assertEqual(sub_sample_input.shape, (2, 5, 5, 5))
        self.assertTrue(np.array_equal(sub_sample_input[0, :, :, :], input_data[0, :, 1:6, 1:6]))
        self.assertTrue(np.array_equal(sub_sample_input[1, :, :, :], input_data[1, :, 1:6, 1:6]))

        # compare the output for batches
        output_pixel = (1, 1)
        self.assertEqual(sub_sample_output.shape, (2, 10))
        self.assertTrue(np.array_equal(sub_sample_output, output_data[:, :, output_pixel[0], output_pixel[1]]))

    @unittest.mock.patch('numpy.random.choice')
    @pytest.mark.cuda
    def test_subsample_data_channels_first_dynamic_shape(self, np_choice_function):
        """
        Test to subsample input match for random output pixel (1, 1) and corresponding input match
        using dynamic input shape
        """
        tf.compat.v1.reset_default_graph()

        # randomly selected output pixel (height, width) is fixed here and it is (1, 1)
        np_choice_function.return_value = [1]

        # input_data and output_data are in channels_first format, similar to Pytorch format
        input_data = np.arange(0, 1440).reshape((2, 5, 12, 12))
        output_data = np.arange(0, 1280).reshape((2, 10, 8, 8))

        g = tf.Graph()

        with g.as_default():
            inp_tensor = tf.compat.v1.placeholder(tf.float32, [None, None, None, None], 'inp_tensor')
            filter_tensor = tf.compat.v1.get_variable('filter_tensor', shape=[5, 5, 5, 10],
                                            initializer=tf.random_normal_initializer())
            conv1 = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 1, 1, 1], padding='VALID',
                                 data_format="NCHW", name='Conv2D_1')
            init = tf.compat.v1.global_variables_initializer()

        sess = tf.compat.v1.Session(graph=g)
        sess.run(init)

        conv1_op = g.get_operation_by_name('Conv2D_1')

        layer_attributes = aimet_tensorflow.utils.op.conv.get_layer_attributes(sess=sess, op=conv1_op,
                                                                               input_op_names=['inp_tensor'],
                                                                               input_shape=(2, 5, 12, 12))
        sub_sample_input, sub_sample_output = InputMatchSearch.subsample_data(layer_attributes=layer_attributes,
                                                                              input_data=input_data,
                                                                              output_data=output_data,
                                                                              samples_per_image=1)
        # compare the inputs for both batches
        self.assertEqual(sub_sample_input.shape, (2, 5, 5, 5))
        self.assertTrue(np.array_equal(sub_sample_input[0, :, :, :], input_data[0, :, 1:6, 1:6]))
        self.assertTrue(np.array_equal(sub_sample_input[1, :, :, :], input_data[1, :, 1:6, 1:6]))

        # compare the output for batches
        output_pixel = (1, 1)
        self.assertEqual(sub_sample_output.shape, (2, 10))
        self.assertTrue(np.array_equal(sub_sample_output, output_data[:, :, output_pixel[0], output_pixel[1]]))

        sess.close()

    @unittest.mock.patch('numpy.random.choice')
    def test_subsample_data_channels_last(self, np_choice_function):
        """
        Test to subsample input match for random output pixel (1, 1) and corresponding input match
        """
        tf.compat.v1.reset_default_graph()

        # randomly selected output pixel (height, width) is fixed here and it is (1, 1)
        np_choice_function.return_value = [1]

        # input_data and output_data are in channels_first format, similar to Pytorch format
        input_data = np.arange(0, 1440).reshape((2, 12, 12, 5))
        output_data = np.arange(0, 1280).reshape((2, 8, 8, 10))

        g = tf.Graph()
        with g.as_default():
            inp_tensor = tf.Variable(initial_value=input_data, name='inp_tensor', dtype=tf.float32)
            filter_tensor = tf.compat.v1.get_variable('filter_tensor', shape=[5, 5, 5, 10],
                                            initializer=tf.random_normal_initializer())
            conv1 = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 1, 1, 1], padding='VALID',
                                 data_format="NHWC", name='Conv2D_1')

        conv1_op = g.get_operation_by_name('Conv2D_1')
        layer_attributes = aimet_tensorflow.utils.op.conv.get_layer_attributes(sess=None, op=conv1_op,
                                                                               input_op_names=None,
                                                                               input_shape=None)

        # reshape input_data, output_data function expects activations in channels_first format
        input_data = input_data.reshape(2, 5, 12, 12)
        output_data = output_data.reshape(2, 10, 8, 8)

        sub_sample_input, sub_sample_output = InputMatchSearch.subsample_data(layer_attributes=layer_attributes,
                                                                              input_data=input_data,
                                                                              output_data=output_data,
                                                                              samples_per_image=1)
        # compare the inputs for both batches
        self.assertEqual(sub_sample_input.shape, (2, 5, 5, 5))
        self.assertTrue(np.array_equal(sub_sample_input[0, :, :, :], input_data[0, :, 1:6, 1:6]))
        self.assertTrue(np.array_equal(sub_sample_input[1, :, :, :], input_data[1, :, 1:6, 1:6]))

        # compare the output for batches
        output_pixel = (1, 1)
        self.assertEqual(sub_sample_output.shape, (2, 10))
        self.assertTrue(np.array_equal(sub_sample_output, output_data[:, :, output_pixel[0], output_pixel[1]]))

    @unittest.mock.patch('numpy.random.choice')
    def test_subsample_data_channels_last_dynamic_shape(self, np_choice_function):
        """
        Test to subsample input match for random output pixel (1, 1) and corresponding input match
        using dynamic input shape
        """
        tf.compat.v1.reset_default_graph()

        # randomly selected output pixel (height, width) is fixed here and it is (1, 1)
        np_choice_function.return_value = [1]

        # input_data and output_data are in channels_first format, similar to Pytorch format
        input_data = np.arange(0, 1440).reshape((2, 12, 12, 5))
        output_data = np.arange(0, 1280).reshape((2, 8, 8, 10))

        g = tf.Graph()
        with g.as_default():
            inp_tensor = tf.compat.v1.placeholder(tf.float32, [None, None, None, None], 'inp_tensor')
            filter_tensor = tf.compat.v1.get_variable('filter_tensor', shape=[5, 5, 5, 10],
                                            initializer=tf.random_normal_initializer())
            conv1 = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 1, 1, 1], padding='VALID',
                                 data_format="NHWC", name='Conv2D_1')
            init = tf.compat.v1.global_variables_initializer()

        sess = tf.compat.v1.Session(graph=g)
        sess.run(init)
        conv1_op = g.get_operation_by_name('Conv2D_1')

        layer_attributes = aimet_tensorflow.utils.op.conv.get_layer_attributes(sess=sess, op=conv1_op,
                                                                               input_op_names=['inp_tensor'],
                                                                               input_shape=(2, 12, 12, 5))

        # reshape input_data, output_data function expects activations in channels_first format
        input_data = input_data.reshape(2, 5, 12, 12)
        output_data = output_data.reshape(2, 10, 8, 8)

        sub_sample_input, sub_sample_output = InputMatchSearch.subsample_data(layer_attributes=layer_attributes,
                                                                              input_data=input_data,
                                                                              output_data=output_data,
                                                                              samples_per_image=1)

        # compare the inputs for both batches
        self.assertEqual(sub_sample_input.shape, (2, 5, 5, 5))
        self.assertTrue(np.array_equal(sub_sample_input[0, :, :, :], input_data[0, :, 1:6, 1:6]))
        self.assertTrue(np.array_equal(sub_sample_input[1, :, :, :], input_data[1, :, 1:6, 1:6]))

        # compare the output for batches
        output_pixel = (1, 1)
        self.assertEqual(sub_sample_output.shape, (2, 10))
        self.assertTrue(np.array_equal(sub_sample_output, output_data[:, :, output_pixel[0], output_pixel[1]]))

        sess.close()

    def test_select_inp_channels(self):
        """
        Test select input channels
        """
        tf.compat.v1.reset_default_graph()

        data_set = unittest.mock.MagicMock()
        number_of_batches = unittest.mock.MagicMock()
        input_op_names = unittest.mock.MagicMock()
        output_op_names = unittest.mock.MagicMock()
        number_of_reconstruction_samples = unittest.mock.MagicMock()
        num_examples = 2000

        g = tf.Graph()
        with g.as_default():
            x1 = tf.range(5.0 * 5.0 * 32.0 * 64.0)
            print("X1", x1)
            x2 = tf.reshape(tensor=x1, shape=(5, 5, 32, 64))
            print("x2 shape", x2.shape)
            inp_tensor = tf.compat.v1.get_variable('inp_tensor', shape=[num_examples, 32, 5, 5],
                                         initializer=tf.random_normal_initializer())
            filter_tensor = tf.compat.v1.get_variable('filter_tensor', initializer=x2)
            conv1 = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 1, 1, 1], padding='VALID',
                                 data_format="NCHW", name='Conv2D_1')
            bias_tensor = tf.compat.v1.get_variable('bias_tensor', shape=[64],
                                                    initializer=tf.random_normal_initializer())
            bias = tf.nn.bias_add(value=conv1, bias=bias_tensor, data_format="NCHW")
            init = tf.compat.v1.global_variables_initializer()

        conv1_op = g.get_operation_by_name('Conv2D_1')
        # output shape in NCHW format
        output_shape = conv1_op.outputs[0].shape

        shape = conv1_op.outputs[0].get_shape().as_list()
        self.assertEqual(shape, [num_examples, 64, 1, 1])

        sess = tf.compat.v1.Session(graph=g)
        sess.run(init)

        conv_layer = Layer(model=sess, op=conv1_op, output_shape=output_shape)

        cp = InputChannelPruner(input_op_names=input_op_names, output_op_names=output_op_names, data_set=data_set,
                                batch_size=number_of_batches,
                                num_reconstruction_samples=number_of_reconstruction_samples,
                                allow_custom_downsample_ops=True)

        # in_channels = 32 and calculate remaining channels
        # 1) 32 * 0.25 = 8
        # 3) 32 * 0.50 = 16
        # 4) 32 * 0.75 = 24
        # 5) 32 * 1 = 32

        input_channels_indices = list(range(32))

        comp_ratio_prune_inp_channels_list = [(0.25, 24), (0.50, 16), (0.75, 8), (1, 0)]

        for comp_ratio, remaining_channels in comp_ratio_prune_inp_channels_list:

            prune_indices = cp._select_inp_channels(layer=conv_layer, comp_ratio=comp_ratio)

            self.assertTrue(isinstance(prune_indices, list))
            expected_indices = input_channels_indices[:remaining_channels]
            self.assertEqual(len(prune_indices), len(expected_indices))
            self.assertEqual(prune_indices, expected_indices)

        sess.close()

    # Need to mark this for CUDA because TF CPU Conv does not support NCHW
    @pytest.mark.cuda
    def test_reconstruct_weight_for_layer(self):
        """
        Test the reconstruction of weight
        """
        tf.compat.v1.reset_default_graph()

        # input shape should be [Ns, Nic, k_h, k_w]
        number_of_images = 500
        num_in_channels = 5
        num_out_channels = 10
        input_data = np.random.rand(number_of_images, num_in_channels, 5, 5)
        g = tf.Graph()

        with g.as_default():
            inp_tensor = tf.Variable(initial_value=input_data, name='inp_tensor', dtype=tf.float32)
            filter_tensor = tf.compat.v1.get_variable('filter_tensor', shape=[5, 5, num_in_channels, num_out_channels],
                                            initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 1, 1, 1], padding='VALID',
                                data_format="NCHW", name='Conv2D_1')
            init = tf.compat.v1.global_variables_initializer()

        conv_op = g.get_operation_by_name('Conv2D_1')
        sess = tf.compat.v1.Session(graph=g)
        sess.run(init)

        conv_out = sess.run(conv_op.outputs[0])
        # output shape in NCHW format
        output_shape = conv_op.outputs[0].shape

        conv2d_reshaped_out = conv_out.reshape([conv_out.shape[0], np.prod(conv_out.shape[1:4])])
        # create layer
        layer = Layer(model=sess, op=conv_op, output_shape=output_shape)

        WeightReconstructor.reconstruct_params_for_conv2d(layer=layer, input_data=input_data,
                                                          output_data=conv2d_reshaped_out,
                                                          output_mask=[1]*num_out_channels)

        meta_path = str('./temp_working_dir')

        if not os.path.exists(meta_path):
            os.mkdir(meta_path)

        # get the same op output after reconstruction
        conv_op = sess.graph.get_operation_by_name('Conv2D_1')
        updated_conv_out = sess.run(conv_op.outputs[0])

        # if data is increased, choose tolerance wisely
        self.assertTrue(np.allclose(conv_out, updated_conv_out, atol=1e-5))

        # they should not be exactly same
        self.assertFalse(np.array_equal(conv_out, updated_conv_out))

        # delete the directory
        shutil.rmtree(meta_path)

        sess.close()

    # Need to mark this for CUDA because TF CPU Conv does not support NCHW
    @pytest.mark.cuda
    def test_reconstruct_weight_and_bias_for_layer(self):
        """
        Test the reconstruction of weight and bias
        """
        tf.compat.v1.reset_default_graph()
        # input shape should be [Ns, Nic, k_h, k_w]
        number_of_images = 500
        num_in_channels = 5
        num_out_channels = 10
        input_data = np.random.rand(number_of_images, 5, 5, 5)
        g = tf.Graph()

        with g.as_default():
            inp_tensor = tf.Variable(initial_value=input_data, name='inp_tensor', dtype=tf.float32)
            filter_tensor = tf.compat.v1.get_variable('filter_tensor', shape=[5, 5, num_in_channels, num_out_channels],
                                            initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 1, 1, 1], padding='VALID',
                                data_format="NCHW", name='Conv2D_1')
            bias_tensor = tf.compat.v1.get_variable('bias_tensor', shape=[num_out_channels],
                                          initializer=tf.random_normal_initializer())
            tf.nn.bias_add(conv, bias_tensor, data_format="NCHW")

            init = tf.compat.v1.global_variables_initializer()

        sess = tf.compat.v1.Session(graph=g)
        sess.run(init)

        conv_op = g.get_operation_by_name('Conv2D_1')
        # output shape in NCHW format
        output_shape = conv_op.outputs[0].shape

        # create layer
        layer = Layer(model=sess, op=conv_op, output_shape=output_shape)

        bias_op = g.get_operation_by_name('BiasAdd')
        bias_out = sess.run(bias_op.outputs[0])

        bias_reshaped_out = bias_out.reshape([bias_out.shape[0], np.prod(bias_out.shape[1:4])])

        WeightReconstructor.reconstruct_params_for_conv2d(layer=layer, input_data=input_data,
                                                          output_data=bias_reshaped_out,
                                                          output_mask=[1]*num_out_channels)

        meta_path = str('./temp_working_dir')

        if not os.path.exists(meta_path):
            os.mkdir(meta_path)

        # get the updated output after reconstruction
        bias_op = sess.graph.get_operation_by_name('BiasAdd')
        updated_bias_out = sess.run(bias_op.outputs[0])

        # if data is increased, choose tolerance wisely
        # compare output after bias op
        self.assertTrue(np.allclose(bias_out, updated_bias_out, atol=1e-5))

        # they should not be exactly same
        self.assertFalse(np.array_equal(bias_out, updated_bias_out))

        # delete the directory
        shutil.rmtree(meta_path)

        sess.close()

    def test_datasampling_and_reconstruction(self):
        """
        Test data sampling and reconstruction logic
        """
        tf.compat.v1.reset_default_graph()
        batch_size = 1
        input_data = np.random.rand(100, 224, 224, 3)
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.batch(batch_size=batch_size)

        orig_g = tf.Graph()

        with orig_g.as_default():

            _ = VGG16(weights=None, input_shape=(224, 224, 3), include_top=False)
            orig_init = tf.compat.v1.global_variables_initializer()

        input_op_names = ['input_1']
        output_op_names = ['block5_pool/MaxPool']
        # create sess with graph
        orig_sess = tf.compat.v1.Session(graph=orig_g)
        # initialize all the variables in VGG16
        orig_sess.run(orig_init)

        # create layer database
        layer_db = LayerDatabase(model=orig_sess, input_shape=(1, 224, 224, 3), working_dir=None)
        conv_layer = layer_db.find_layer_by_name('block1_conv1/Conv2D')

        comp_layer_db = copy.deepcopy(layer_db)
        comp_conv_layer = comp_layer_db.find_layer_by_name('block1_conv1/Conv2D')

        # get the weights before reconstruction in original model
        before_recon_weights_orig_model = layer_db.model.run(conv_layer.module.inputs[1])

        # get the weights before reconstruction in pruned  model
        before_recon_weights_pruned_model = comp_layer_db.model.run(comp_conv_layer.module.inputs[1])

        # weight should be exactly same before reconstruction in original and pruned layer database
        self.assertTrue(np.array_equal(before_recon_weights_orig_model, before_recon_weights_pruned_model))

        cp = InputChannelPruner(input_op_names=input_op_names, output_op_names=output_op_names, data_set=dataset,
                                batch_size=batch_size, num_reconstruction_samples=50, allow_custom_downsample_ops=True)

        num_in_channels = comp_conv_layer.weight_shape[0]
        cp._data_subsample_and_reconstruction(orig_layer=conv_layer, pruned_layer=comp_conv_layer,
                                              output_mask=[1]*num_in_channels, orig_layer_db=layer_db,
                                              comp_layer_db=comp_layer_db)

        # get the weights after reconstruction
        after_recon_weights_pruned_model = comp_layer_db.model.run(comp_conv_layer.module.inputs[1])

        # weight should not be same before and after reconstruction
        self.assertFalse(np.array_equal(before_recon_weights_orig_model, after_recon_weights_pruned_model))

        layer_db.model.close()
        comp_layer_db.model.close()
        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))

    def test_prune_model(self):
        """
        Test end-to-end prune_model with VGG16-imagenet
        """
        AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Winnow, logging.INFO)
        tf.compat.v1.reset_default_graph()

        batch_size = 1
        input_data = np.random.rand(100, 224, 224, 3)
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.batch(batch_size=batch_size)

        orig_g = tf.Graph()

        with orig_g.as_default():
            _ = VGG16(weights=None, input_shape=(224, 224, 3), include_top=False)
            orig_init = tf.compat.v1.global_variables_initializer()

        input_op_names = ['input_1']
        output_op_names = ['block5_pool/MaxPool']
        # create sess with graph
        orig_sess = tf.compat.v1.Session(graph=orig_g)
        # initialize all the variables in VGG16
        orig_sess.run(orig_init)

        # create layer database
        layer_db = LayerDatabase(model=orig_sess, input_shape=(1, 224, 224, 3), working_dir=None)

        block1_conv2 = layer_db.model.graph.get_operation_by_name('block1_conv2/Conv2D')
        block2_conv1 = layer_db.model.graph.get_operation_by_name('block2_conv1/Conv2D')
        block2_conv2 = layer_db.model.graph.get_operation_by_name('block2_conv2/Conv2D')

        # output shape in NCHW format
        block1_conv2_output_shape = block1_conv2.outputs[0].shape
        block2_conv1_output_shape = block2_conv1.outputs[0].shape
        block2_conv2_output_shape = block2_conv2.outputs[0].shape

        # keeping compression ratio = 0.5 for all layers
        layer_comp_ratio_list = [LayerCompRatioPair(Layer(model=layer_db.model, op=block1_conv2,
                                                          output_shape=block1_conv2_output_shape), 0.5),
                                 LayerCompRatioPair(Layer(model=layer_db.model, op=block2_conv1,
                                                          output_shape=block2_conv1_output_shape), 0.5),
                                 LayerCompRatioPair(Layer(model=layer_db.model, op=block2_conv2,
                                                          output_shape=block2_conv2_output_shape), 0.5)
                                 ]

        cp = InputChannelPruner(input_op_names=input_op_names, output_op_names=output_op_names, data_set=dataset,
                                batch_size=batch_size, num_reconstruction_samples=20, allow_custom_downsample_ops=True)

        comp_layer_db = cp.prune_model(layer_db=layer_db, layer_comp_ratio_list=layer_comp_ratio_list,
                                       cost_metric=CostMetric.mac, trainer=None)

        pruned_block1_conv2 = comp_layer_db.find_layer_by_name('reduced_reduced_block1_conv2/Conv2D')
        pruned_block2_conv1 = comp_layer_db.find_layer_by_name('reduced_reduced_block2_conv1/Conv2D')
        pruned_block2_conv2 = comp_layer_db.find_layer_by_name('reduced_block2_conv2/Conv2D')

        # input channels = 64 * 0.5 = 32
        # output channels = 64 * 0.5 = 32
        self.assertEqual(pruned_block1_conv2.weight_shape[1], 32)
        self.assertEqual(pruned_block1_conv2.weight_shape[0], 32)

        # input channels = 64 * 0.5 = 32
        # output channels = 128 * 0.5 = 64
        self.assertEqual(pruned_block2_conv1.weight_shape[1], 32)
        self.assertEqual(pruned_block2_conv1.weight_shape[0], 64)

        # input channels = 128 * 0.5 = 64
        # output channels = 128
        self.assertEqual(pruned_block2_conv2.weight_shape[1], 64)
        self.assertEqual(pruned_block2_conv2.weight_shape[0], 128)

        layer_db.model.close()
        comp_layer_db.model.close()
        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))

    def test_sort_on_occurrence(self):
        """
        Test sorting of ops based on occurrence
        """
        AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Winnow, logging.INFO)
        tf.compat.v1.reset_default_graph()

        orig_g = tf.Graph()
        with orig_g.as_default():
            _ = VGG16(weights=None, input_shape=(224, 224, 3), include_top=False)
            orig_init = tf.compat.v1.global_variables_initializer()

        # create sess with graph
        orig_sess = tf.compat.v1.Session(graph=orig_g)
        orig_sess.run(orig_init)

        # create layer database
        layer_db = LayerDatabase(model=orig_sess, input_shape=(1, 224, 224, 3), working_dir=None)

        block1_conv2 = layer_db.model.graph.get_operation_by_name('block1_conv2/Conv2D')
        block2_conv1 = layer_db.model.graph.get_operation_by_name('block2_conv1/Conv2D')
        block2_conv2 = layer_db.model.graph.get_operation_by_name('block2_conv2/Conv2D')
        block5_conv3 = layer_db.model.graph.get_operation_by_name('block5_conv3/Conv2D')

        # output shape in NCHW format
        block1_conv2_output_shape = block1_conv2.outputs[0].shape
        block2_conv1_output_shape = block2_conv1.outputs[0].shape
        block2_conv2_output_shape = block2_conv2.outputs[0].shape
        block5_conv3_output_shape = block5_conv3.outputs[0].shape

        # keeping compression ratio = None for all layers
        layer_comp_ratio_list = [
                                 LayerCompRatioPair(Layer(model=layer_db.model, op=block5_conv3,
                                                          output_shape=block5_conv3_output_shape), None),
                                 LayerCompRatioPair(Layer(model=layer_db.model, op=block2_conv2,
                                                          output_shape=block2_conv2_output_shape), None),
                                 LayerCompRatioPair(Layer(model=layer_db.model, op=block1_conv2,
                                                          output_shape=block1_conv2_output_shape), None),
                                 LayerCompRatioPair(Layer(model=layer_db.model, op=block2_conv1,
                                                          output_shape=block2_conv1_output_shape), None)
                                 ]

        input_op_names = ['input_1']
        output_op_names = ['block5_pool/MaxPool']
        dataset = unittest.mock.MagicMock()
        batch_size = unittest.mock.MagicMock()
        num_reconstruction_samples = unittest.mock.MagicMock()

        cp = InputChannelPruner(input_op_names=input_op_names, output_op_names=output_op_names, data_set=dataset,
                                batch_size=batch_size, num_reconstruction_samples=num_reconstruction_samples,
                                allow_custom_downsample_ops=True)

        sorted_layer_comp_ratio_list = cp._sort_on_occurrence(layer_db.model, layer_comp_ratio_list)

        self.assertEqual(sorted_layer_comp_ratio_list[0].layer.module, block1_conv2)
        self.assertEqual(sorted_layer_comp_ratio_list[1].layer.module, block2_conv1)
        self.assertEqual(sorted_layer_comp_ratio_list[2].layer.module, block2_conv2)
        self.assertEqual(sorted_layer_comp_ratio_list[3].layer.module, block5_conv3)

        self.assertEqual(len(sorted_layer_comp_ratio_list), 4)
        layer_db.model.close()
        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))

    def test_sort_on_occurrence_resnet50(self):
        """
        Test sorting of ops based on occurrence
        """
        AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Winnow, logging.INFO)
        tf.compat.v1.reset_default_graph()

        orig_g = tf.Graph()
        with orig_g.as_default():
            _ = ResNet50(weights=None, input_shape=(224, 224, 3), include_top=False)
            orig_init = tf.compat.v1.global_variables_initializer()

        # create sess with graph
        orig_sess = tf.compat.v1.Session(graph=orig_g)
        orig_sess.run(orig_init)

        # create layer database
        layer_db = LayerDatabase(model=orig_sess, input_shape=(1, 224, 224, 3), working_dir=None)

        res2b_branch2a = layer_db.model.graph.get_operation_by_name('conv1_conv/Conv2D')
        res2a_branch1 = layer_db.model.graph.get_operation_by_name('conv2_block1_0_conv/Conv2D')
        res2b_branch2b = layer_db.model.graph.get_operation_by_name('conv2_block1_1_conv/Conv2D')
        res2c_branch2a = layer_db.model.graph.get_operation_by_name('conv2_block1_2_conv/Conv2D')

        # output shape in NCHW format
        res2b_branch2a_output_shape = res2b_branch2a.outputs[0].shape
        res2a_branch1_output_shape = res2a_branch1.outputs[0].shape
        res2b_branch2b_output_shape = res2b_branch2b.outputs[0].shape
        res2c_branch2a_output_shape = res2c_branch2a.outputs[0].shape

        # keeping compression ratio = None for all layers
        layer_comp_ratio_list = [
            LayerCompRatioPair(Layer(model=layer_db.model, op=res2c_branch2a,
                                     output_shape=res2c_branch2a_output_shape), None),
            LayerCompRatioPair(Layer(model=layer_db.model, op=res2b_branch2b,
                                     output_shape=res2b_branch2b_output_shape), None),
            LayerCompRatioPair(Layer(model=layer_db.model, op=res2a_branch1,
                                     output_shape=res2a_branch1_output_shape), None),
            LayerCompRatioPair(Layer(model=layer_db.model, op=res2b_branch2a,
                                     output_shape=res2b_branch2a_output_shape), None)
        ]

        input_op_names = ['input_1']
        output_op_names = ['conv5_block3_out/Relu']
        dataset = unittest.mock.MagicMock()
        batch_size = unittest.mock.MagicMock()
        num_reconstruction_samples = unittest.mock.MagicMock()

        cp = InputChannelPruner(input_op_names=input_op_names, output_op_names=output_op_names, data_set=dataset,
                                batch_size=batch_size, num_reconstruction_samples=num_reconstruction_samples,
                                allow_custom_downsample_ops=True)

        sorted_layer_comp_ratio_list = cp._sort_on_occurrence(layer_db.model, layer_comp_ratio_list)

        self.assertEqual(sorted_layer_comp_ratio_list[0].layer.module, res2b_branch2a)
        self.assertEqual(sorted_layer_comp_ratio_list[1].layer.module, res2b_branch2b)
        self.assertEqual(sorted_layer_comp_ratio_list[2].layer.module, res2c_branch2a)
        self.assertEqual(sorted_layer_comp_ratio_list[3].layer.module, res2a_branch1)

        self.assertEqual(len(sorted_layer_comp_ratio_list), 4)
        layer_db.model.close()

        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))
