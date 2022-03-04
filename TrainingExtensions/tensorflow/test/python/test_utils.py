# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Module to test TF utils """

import itertools
import pytest
import unittest
from packaging import version
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
if not version.parse(tf.version.VERSION) >= version.parse("2.0"):
    import tensorflow.contrib.slim as slim
from aimet_tensorflow import graph_editor
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50

from aimet_common.utils import AimetLogger
from aimet_tensorflow.utils.common import get_ordered_ops, create_input_feed_dict, \
    iter_first_x, get_ordered_conv_linears, get_training_tensors,\
    iterate_tf_dataset, _tf_dataset_iterables
from aimet_tensorflow.utils.graph_saver import wrapper_func
from aimet_tensorflow.examples.test_models import single_residual, multiple_input_model, \
    model_with_multiple_training_tensors, keras_model_functional, keras_model_functional_with_non_fused_batchnorms,\
    keras_model_functional_for_tf2, keras_model_functional_with_non_fused_batchnorms_for_tf2
from aimet_tensorflow.utils.op.conv import WeightTensorUtils, BiasUtils, get_output_activation_shape
from aimet_tensorflow.utils.op.fusedbatchnorm import BNUtils
from aimet_tensorflow.utils.graph_saver import save_and_load_graph

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


def get_bn_params_keras_layer(layer):
    """
    helper to get param values from Keras BN layer
    :param layer: BN layer
    :return: param values extracted from layer
    """
    gamma = layer.weights[0].eval()
    beta = layer.weights[1].eval()
    mean = layer.weights[2].eval()
    variance = layer.weights[3].eval()

    return [beta, gamma, mean, variance]


def get_bn_params_aimet_api(sess, bn_op):
    """
    Helper to get param values from BN layer using AIMET api(s)
    :param bn_op: BN layer
    :return: beta, gamma, mean and vairance values extracted from BN layer
    """
    beta = BNUtils.get_beta_as_numpy_data(sess, bn_op)
    gamma = BNUtils.get_gamma_as_numpy_data(sess, bn_op)
    moving_mean = BNUtils.get_moving_mean_as_numpy_data(sess, bn_op)
    moving_var = BNUtils.get_moving_variance_as_numpy_data(sess, bn_op)

    return [beta, gamma, moving_mean, moving_var]


class TestTrainingExtensionsTfUtils(unittest.TestCase):
    """ Unittest class for testing Tf Utils """

    def setUp(self):
        _tf_dataset_iterables.clear()

    def test_wrapper_func_second_arg_without_args(self):
        """
        test wrapper_func without any arguments, expect ValueError
        """
        def dummy_eval_func():
            return 1

        dummy_eval_func = wrapper_func(dummy_eval_func)

        # calling dummy_eval_func without any arguments
        with self.assertRaises(ValueError):
            dummy_eval_func()

    def test_wrapper_func_second_arg_with_sess(self):
        """
        test wrapper_func with second argument tf.compat.v1.Session, expect ValueError
        """
        def dummy_eval_func(model, _):
            return model

        tf.compat.v1.reset_default_graph()
        g = tf.Graph()
        with g.as_default():
            _ = VGG16(weights=None, input_shape=(224, 224, 3))
            init = tf.compat.v1.global_variables_initializer()

        sess = tf.compat.v1.Session(graph=g)
        sess.run(init)

        dummy_eval_func = wrapper_func(dummy_eval_func)

        # calling dummy_eval_func with first random argument, and second argument tf.compat.v1.Session
        self.assertRaises(ValueError, lambda: dummy_eval_func('test', sess))

        sess.close()

    def test_wrapper_func_first_arg_with_sess(self):
        """
        test wrapper_func with first argument tf.compat.v1.Session
        test to see if the provides session and updated session should be different or not
        """
        def dummy_eval_func(model, _):
            return model

        tf.compat.v1.reset_default_graph()
        g = tf.Graph()
        with g.as_default():
            _ = VGG16(weights=None, input_shape=(224, 224, 3))
            init = tf.compat.v1.global_variables_initializer()

        sess = tf.compat.v1.Session(graph=g)
        sess.run(init)

        dummy_eval_func = wrapper_func(dummy_eval_func)

        # calling dummy_eval_func with tf.compat.v1.Session first argument
        updated_sess = dummy_eval_func(sess, 'test')
        self.assertNotEqual(sess, updated_sess)

        sess.close()

    def test_get_ordered_ops_with_single_residual(self):
        """
        test get_op with simple single residual model
        """
        tf.compat.v1.reset_default_graph()
        g = tf.Graph()

        with g.as_default():
            single_residual()

        ordered_ops = get_ordered_ops(g, ['input_1'], ['single_residual/Softmax'])

        self.assertTrue(ordered_ops.index(g.get_operation_by_name('conv2d_4/Conv2D')) >
                        ordered_ops.index(g.get_operation_by_name('conv2d_1/Conv2D')))

    @unittest.skip
    def test_get_ordered_ops_with_resnet50(self):
        """
        test get_ordered_operations with Resnet50 model
        """
        tf.compat.v1.reset_default_graph()
        g = tf.Graph()

        with g.as_default():
            _ = ResNet50(weights=None)
            filter_tensor = tf.compat.v1.get_variable('filter_tensor', shape=[5, 5, 3, 50],
                                            initializer=tf.random_normal_initializer())

            # add dangling conv, which is not a valid op
            # pylint: disable=no-member
            _ = tf.nn.conv2d(g.get_tensor_by_name('input_1:0'), filter_tensor, strides=[1, 1, 1, 1],
                             padding='VALID', data_format="NHWC", name='dangling/Conv2D')

        ordered_ops = get_ordered_ops(g, ['input_1'], ['probs/Softmax'])

        self.assertTrue(ordered_ops.index(g.get_operation_by_name('res2a_branch2b/convolution')) >
                        ordered_ops.index(g.get_operation_by_name('res2a_branch1/convolution')))

        self.assertTrue(ordered_ops.index(g.get_operation_by_name('activation_4/Relu')) >
                        ordered_ops.index(g.get_operation_by_name('add_1/add')))

        self.assertTrue(ordered_ops.index(g.get_operation_by_name('res2a_branch2a/BiasAdd')) >
                        ordered_ops.index(g.get_operation_by_name('res2a_branch2a/convolution')))

        self.assertTrue(g.get_operation_by_name('dangling/Conv2D') not in ordered_ops)

    def test_get_ordered_ops_with_multiple_inputs(self):
        """
        test get_ordered_operations with multiple inputs
        """
        tf.compat.v1.reset_default_graph()
        g = tf.Graph()

        with g.as_default():
            multiple_input_model()

        ordered_ops = get_ordered_ops(g, ['input2', 'input1'], ['multiple_input_model/Softmax'])

        self.assertTrue(ordered_ops.index(g.get_operation_by_name('conv1b/Conv2D')) >
                        ordered_ops.index(g.get_operation_by_name('input2')))

        self.assertTrue(ordered_ops.index(g.get_operation_by_name('conv1a/Conv2D')) >
                        ordered_ops.index(g.get_operation_by_name('input1')))

        self.assertTrue(ordered_ops.index(g.get_operation_by_name('add/add')) >
                        ordered_ops.index(g.get_operation_by_name('input1')))

        self.assertTrue(ordered_ops.index(g.get_operation_by_name('add/add')) >
                        ordered_ops.index(g.get_operation_by_name('input2')))

    @pytest.mark.tf1
    def test_create_input_feed_dict(self):
        """
        test create_input_feed_dict
        """
        tf.compat.v1.reset_default_graph()
        # 1) input_batch_data numpy array
        g = tf.Graph()
        with g.as_default():
            _ = single_residual()

        input_data = np.random.rand(1, 16, 16, 3)
        feed_dict = create_input_feed_dict(graph=g, input_op_names_list=['input_1'], input_data=input_data)
        self.assertEqual(feed_dict[g.get_tensor_by_name('input_1:0')].shape, input_data.shape)

        tf.compat.v1.reset_default_graph()

        # 2) input_batch_data List of numpy array
        g = tf.Graph()
        with g.as_default():
            multiple_input_model()

        input_data = list()
        input_data.append(np.random.rand(10, 10, 3))
        input_data.append(np.random.rand(12, 12, 3))
        feed_dict = create_input_feed_dict(graph=g, input_op_names_list=['input1', 'input2'],
                                           input_data=input_data)

        self.assertEqual(feed_dict[g.get_tensor_by_name('input1:0')].shape, input_data[0].shape)
        self.assertEqual(feed_dict[g.get_tensor_by_name('input2:0')].shape, input_data[1].shape)

        tf.compat.v1.reset_default_graph()

        # 3) input_batch_data Tuple of numpy array
        g = tf.Graph()
        with g.as_default():
            multiple_input_model()

        input_data = (np.random.rand(10, 10, 3), np.random.rand(12, 12, 3))

        feed_dict = create_input_feed_dict(graph=g, input_op_names_list=['input1', 'input2'],
                                           input_data=input_data)

        self.assertEqual(feed_dict[g.get_tensor_by_name('input1:0')].shape, input_data[0].shape)
        self.assertEqual(feed_dict[g.get_tensor_by_name('input2:0')].shape, input_data[1].shape)
        tf.compat.v1.reset_default_graph()

        # 3) input_batch_data and input_op_names mismatch
        g = tf.Graph()
        with g.as_default():
            multiple_input_model()

        input_data = (np.random.rand(10, 10, 3))

        self.assertRaises(ValueError, lambda: create_input_feed_dict(graph=g,
                                                                     input_op_names_list=['input1', 'input2'],
                                                                     input_data=input_data))
        tf.compat.v1.reset_default_graph()

        g = tf.Graph()
        with g.as_default():
            model_with_multiple_training_tensors()
        input_data = (np.random.rand(32, 32, 3))
        feed_dict = create_input_feed_dict(graph=g, input_op_names_list=['input_1'],
                                           input_data=input_data, training=True)
        keras_learning_phase_tensor = g.get_tensor_by_name('keras_learning_phase:0')
        is_training_tensor = g.get_tensor_by_name('is_training:0')
        is_training_2_tensor = g.get_tensor_by_name('is_training_2:0')
        self.assertEqual(feed_dict[keras_learning_phase_tensor], True)
        self.assertEqual(feed_dict[is_training_tensor], True)
        self.assertEqual(feed_dict[is_training_2_tensor], True)
        tf.compat.v1.reset_default_graph()

    def test_iter_first_x(self):
        """ Test iter_first_x generator for creating a dataset generator """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            dataset = tf.data.Dataset.from_tensor_slices([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            dataset_iterator = iter_first_x(dataset, num_batches=5)

        for i, data in enumerate(dataset_iterator):
            self.assertEqual(i, data)       # Data has not been batched, so each element should be returned individually
            self.assertTrue(i < 5)          # Check that iterator stops at the correct point

        with sess.graph.as_default():
            dataset = tf.data.Dataset.from_tensor_slices([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            dataset = dataset.batch(2)
            dataset_iterator = iter_first_x(dataset, num_batches=5)

        for i, data in enumerate(dataset_iterator):
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0], 2*i)
            self.assertEqual(data[1], 2*i+1)

        # Test that trying to extract more data than possible from the dataset is handled
        # since tensorflow OutOfRangeError is converted to StopIteration
        with sess.graph.as_default():
            dataset_iterator = iter_first_x(dataset, num_batches=6)

        for i, data in enumerate(dataset_iterator):
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0], 2*i)
            self.assertEqual(data[1], 2*i+1)

        sess.close()

    def test_update_to_weight_tensor_with_load_var(self):
        """
        tests update to weight tensor of conv op using tf variable load api
        """
        # create conv op
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        _ = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer=tf.random_uniform_initializer(-1, 2))(inputs)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')

        original_weights = WeightTensorUtils.get_tensor_as_numpy_data(sess, conv_op)

        # add dummy weight tensor data
        np.random.seed(0)
        w_shape = WeightTensorUtils.get_tensor_shape(conv_op)
        numpy_data = np.random.rand(3, w_shape[1], w_shape[2], w_shape[3])

        # send in numpy data to overwrite previous value
        WeightTensorUtils.update_tensor_for_op(sess, conv_op, numpy_data)

        updated_weight_tensor = WeightTensorUtils.get_tensor_as_numpy_data(sess, conv_op)

        # validate they are not the same
        self.assertFalse(np.allclose(original_weights, updated_weight_tensor))
        self.assertTrue(np.allclose(numpy_data, updated_weight_tensor))
        sess.close()

    def test_update_to_bias_with_load_var(self):
        """
        tests update to bias param of conv op using tf variable load api
        """
        # create conv op
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3),
                                   kernel_initializer=tf.random_uniform_initializer(-1, 2))(inputs)

        bn_op = tf.keras.layers.BatchNormalization(fused=True)(conv_op)
        _ = tf.nn.relu(bn_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')

        original_bias = BiasUtils.get_bias_as_numpy_data(sess, conv_op)

        # add dummy weight tensor data
        np.random.seed(0)
        b_shape = BiasUtils.get_shape(conv_op)
        numpy_data = np.random.rand(b_shape[0])

        # send in numpy data to overwrite previous value
        BiasUtils.update_bias_for_op(sess, conv_op, numpy_data)

        updated_bias = BiasUtils.get_bias_as_numpy_data(sess, conv_op)

        # validate they are not the same
        self.assertFalse(np.allclose(original_bias, updated_bias))
        self.assertTrue(np.allclose(numpy_data, updated_bias))
        sess.close()

    def test_bias_add_with_conv(self):
        """
        Test bias add on conv op
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        # create a conv without bias param
        conv_op = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(inputs)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(conv_op)
        # pylint: disable=no-member
        _ = tf.nn.relu(bn_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        self.assertTrue(BiasUtils.is_bias_none(conv_op))

        # new_sess = BiasUtils.initialize_model_with_bias(sess)
        shape = BiasUtils.get_shape(conv_op)
        numpy_data = np.random.rand(shape[0])
        BiasUtils.update_bias_for_op(sess, conv_op, bias_as_numpy_array=numpy_data)
        new_sess = save_and_load_graph('./temp_bn_fold', sess)
        conv_op = new_sess.graph.get_operation_by_name('conv2d/Conv2D')
        bias_as_numpy_data = BiasUtils.get_bias_as_numpy_data(new_sess, conv_op)

        assert(not BiasUtils.is_bias_none(conv_op))
        sess.close()
        new_sess.close()

    def test_bias_update_to_dense(self):
        """
        test bias correction on matmul layer
        """
        tf.compat.v1.reset_default_graph()

        inputs = tf.keras.Input(shape=(32, 32, 3,))
        x = tf.keras.layers.Flatten()(inputs)
        dense = tf.keras.layers.Dense(2, use_bias=False, activation=tf.nn.softmax, name="single_residual")(x)
        # pylint: disable=no-member
        _ = tf.nn.relu(dense)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        sess.run(init)

        dense_op = sess.graph.get_operation_by_name('single_residual/MatMul')
        self.assertTrue(BiasUtils.is_bias_none(dense_op))

        new_sess = BiasUtils.initialize_model_with_bias(sess, ['input_1'], ['Relu'])

        dense_op = new_sess.graph.get_operation_by_name('single_residual/MatMul')
        self.assertTrue(not BiasUtils.is_bias_none(dense_op))
        sess.close()
        new_sess.close()

    def test_get_ordered_conv_linears(self):
        """
        Test get_ordered_conv_linears
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))

        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        # pylint: disable=no-member
        relu_1 = tf.nn.relu(conv_op)

        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(relu_1)
        _ = tf.nn.relu(conv2_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')

        # check if we get ordered list
        input_op = conv_op.inputs[0].op.name
        selected_ops = get_ordered_conv_linears(sess, [input_op], ['Relu_1'])

        self.assertEqual(2, len(selected_ops))
        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        conv_1_op = sess.graph.get_operation_by_name('conv2d_1/Conv2D')
        self.assertEqual(selected_ops[0], conv_op)
        self.assertEqual(selected_ops[1], conv_1_op)

    @pytest.mark.tf1
    def test_get_training_tensors(self):
        """ Test for obtaining all training tensors in a graph """
        tf.compat.v1.reset_default_graph()
        _ = model_with_multiple_training_tensors()
        training_tensors = get_training_tensors(tf.compat.v1.get_default_graph())
        self.assertEqual(3, len(training_tensors))

    @pytest.mark.cuda
    def test_get_output_activation_shape(self):
        """Test for getting output activation shapes"""
        """Conv NCHW not supported on the CPU"""
        tf.compat.v1.reset_default_graph()
        # 1) dynamic shape
        graph = tf.Graph()
        filter_data = np.ones([5, 5, 3, 32], dtype=np.float32)

        with graph.as_default():
            input_tensor = tf.compat.v1.placeholder(tf.float32, [1, None, None, None], 'input')
            filter_tensor = tf.Variable(initial_value=filter_data, name='filter_tensor', dtype=tf.float32)

            _ = tf.nn.conv2d(input_tensor, filter_tensor, padding='SAME', strides=[1, 1, 1, 1],
                             data_format="NCHW", name='Conv2D_1')

            init = tf.compat.v1.global_variables_initializer()

        sess = tf.compat.v1.Session(graph=graph)
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('Conv2D_1')
        output_shape = get_output_activation_shape(sess=sess, op=conv_op, input_op_names=['input'],
                                                   input_shape=(1, 3, 10, 10))

        batch_size, channels, activations_h, activations_w = output_shape

        self.assertEqual(activations_h, 10)
        self.assertEqual(activations_w, 10)
        self.assertEqual(channels, 32)

        sess.close()

        # 2) static shape

        graph = tf.Graph()
        input_data = np.ones([1, 3, 10, 10], dtype=np.float32)
        filter_data = np.ones([5, 5, 3, 32], dtype=np.float32)

        with graph.as_default():
            input_tensor = tf.Variable(initial_value=input_data, name='input', dtype=tf.float32)
            filter_tensor = tf.Variable(initial_value=filter_data, name='filter_tensor', dtype=tf.float32)

            _ = tf.nn.conv2d(input_tensor, filter_tensor, padding='SAME', strides=[1, 1, 1, 1],
                             data_format="NCHW", name='Conv2D_1')

            init = tf.compat.v1.global_variables_initializer()

        sess = tf.compat.v1.Session(graph=graph)
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('Conv2D_1')
        output_shape = get_output_activation_shape(sess=sess, op=conv_op, input_op_names=['input'],
                                                   input_shape=(1, 3, 10, 10))

        batch_size, channels, activations_h, activations_w = output_shape
        self.assertEqual(activations_h, 10)
        self.assertEqual(activations_w, 10)
        self.assertEqual(channels, 32)

        sess.close()

    def test_get_output_activation_shape_channels_last(self):
        """Test for getting output activation shapes for channels_last format"""
        tf.compat.v1.reset_default_graph()
        # 1) dynamic shape
        graph = tf.Graph()
        filter_data = np.ones([5, 5, 3, 32], dtype=np.float32)

        with graph.as_default():
            input_tensor = tf.compat.v1.placeholder(tf.float32, [1, None, None, None], 'input')
            filter_tensor = tf.Variable(initial_value=filter_data, name='filter_tensor', dtype=tf.float32)

            _ = tf.nn.conv2d(input_tensor, filter_tensor, padding='SAME', strides=[1, 1, 1, 1],
                             data_format="NHWC", name='Conv2D_1')

            init = tf.compat.v1.global_variables_initializer()

        sess = tf.compat.v1.Session(graph=graph)
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('Conv2D_1')
        output_shape = get_output_activation_shape(sess=sess, op=conv_op, input_op_names=['input'],
                                                   input_shape=(1, 10, 10, 3))

        batch_size, channels, activations_h, activations_w = output_shape

        self.assertEqual(activations_h, 10)
        self.assertEqual(activations_w, 10)
        self.assertEqual(channels, 32)

        sess.close()

        # 2) static shape

        graph = tf.Graph()
        # channels_last format
        input_data = np.ones([1, 10, 10, 3], dtype=np.float32)
        filter_data = np.ones([5, 5, 3, 32], dtype=np.float32)

        with graph.as_default():
            input_tensor = tf.Variable(initial_value=input_data, name='input', dtype=tf.float32)
            filter_tensor = tf.Variable(initial_value=filter_data, name='filter_tensor', dtype=tf.float32)

            _ = tf.nn.conv2d(input_tensor, filter_tensor, padding='SAME', strides=[1, 1, 1, 1],
                             data_format="NHWC", name='Conv2D_1')
            init = tf.compat.v1.global_variables_initializer()

        sess = tf.compat.v1.Session(graph=graph)
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('Conv2D_1')
        output_shape = get_output_activation_shape(sess=sess, op=conv_op, input_op_names=['input'],
                                                   input_shape=(1, 10, 10, 3))

        batch_size, channels, activations_h, activations_w = output_shape
        self.assertEqual(activations_h, 10)
        self.assertEqual(activations_w, 10)
        self.assertEqual(channels, 32)

        sess.close()


class TestBNUtils(unittest.TestCase):
    """
    Unittest class for testing BN Utils
    """
    def test_with_tf_bn_op(self):
        """
        Test with TF BN op
        """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        inp = tf.compat.v1.placeholder(tf.float32, [1, 32, 32, 3])
        net = tf.compat.v1.layers.conv2d(inp, 32, [3, 3])
        _ = tf.compat.v1.layers.batch_normalization(net)

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        with sess.as_default():
            bn_op = sess.graph.get_operation_by_name('batch_normalization/FusedBatchNormV3')
            moving_mean = BNUtils.get_moving_mean_as_numpy_data(sess, bn_op)
            moving_var = BNUtils.get_moving_variance_as_numpy_data(sess, bn_op)
            beta = BNUtils.get_beta_as_numpy_data(sess, bn_op)
            gamma = BNUtils.get_gamma_as_numpy_data(sess, bn_op)

        # check the values read are equal to init values
        expected_beta = np.zeros_like(beta)
        expected_gamma = np.ones_like(gamma)
        expected_mean = np.zeros_like(moving_mean)
        expected_variance = np.ones_like(moving_var)

        self.assertTrue(np.allclose(expected_beta, beta))
        self.assertTrue(np.allclose(expected_gamma, gamma))
        self.assertTrue(np.allclose(expected_mean, moving_mean))
        self.assertTrue(np.allclose(expected_variance, moving_var))
        sess.close()

    @pytest.mark.tf1
    def test_with_slim_bn_op(self):
        """
        Test with Tf Slim BN op
        :return:
        """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        inp = tf.compat.v1.placeholder(tf.float32, [1, 32, 32, 3])
        net = slim.conv2d(inp, 32, [3, 3])
        _ = slim.batch_norm(net, decay=.7, epsilon=.65, is_training=True)

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        with sess.graph.as_default():
            bn_op = sess.graph.get_operation_by_name('BatchNorm/FusedBatchNormV3')
            moving_mean = BNUtils.get_moving_mean_as_numpy_data(sess, bn_op)
            moving_var = BNUtils.get_moving_variance_as_numpy_data(sess, bn_op)
            beta = BNUtils.get_beta_as_numpy_data(sess, bn_op)
            gamma = BNUtils.get_gamma_as_numpy_data(sess, bn_op)

        # check the values read are equal to init values
        expected_beta = np.zeros_like(beta)
        expected_gamma = np.ones_like(gamma)
        expected_mean = np.zeros_like(moving_mean)
        expected_variance = np.ones_like(moving_var)

        self.assertTrue(np.allclose(expected_beta, beta))
        self.assertTrue(np.allclose(expected_gamma, gamma))
        self.assertTrue(np.allclose(expected_mean, moving_mean))
        self.assertTrue(np.allclose(expected_variance, moving_var))

    @pytest.mark.tf1
    def test_param_read_keras_model_with_fused_batchnorms(self):
        """
        Test to validate fused BN op param read AIMET api(s) on Keras layers.
        This test also reproduces SFTI issue
        tensorflow.python.framework.errors_impl.InvalidArgumentError
        """
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        with tf.device('/cpu:0'):
            model = keras_model_functional()
            model.summary()

        sess = tf.compat.v1.keras.backend.get_session()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        # layer 1 ,3 and 5 are fused BN of different types
        with sess.as_default():
            # read weights ( beta, gamma, mean, variance)
            bn_1 = model.layers[2]
            bn_2 = model.layers[4]
            bn_3 = model.layers[6]
            keras_bn_1_params = get_bn_params_keras_layer(bn_1)
            keras_bn_2_params = get_bn_params_keras_layer(bn_2)
            keras_bn_3_params = get_bn_params_keras_layer(bn_3)

            bn_op_1 = sess.graph.get_operation_by_name('batch_normalization/FusedBatchNormV3')
            bn_op_2 = sess.graph.get_operation_by_name('scope_1/batch_normalization_1/cond/FusedBatchNormV3_1')
            bn_op_3 = sess.graph.get_operation_by_name('scope_1/batch_normalization_2/FusedBatchNormV3')
            bn_1_params = get_bn_params_aimet_api(sess, bn_op_1)
            bn_2_params = get_bn_params_aimet_api(sess, bn_op_2)
            bn_3_params = get_bn_params_aimet_api(sess, bn_op_3)

            self.assertTrue(np.allclose(keras_bn_1_params, bn_1_params))
            self.assertTrue(np.allclose(keras_bn_2_params, bn_2_params))
            self.assertTrue(np.allclose(keras_bn_3_params, bn_3_params))

        sess.close()

    def test_param_read_keras_model_with_fused_batchnorms_for_tf2(self):
        """
        Test to validate fused BN op param read AIMET api(s) on Keras layers.
        This test also reproduces SFTI issue
        tensorflow.python.framework.errors_impl.InvalidArgumentError
        """
        if version.parse(tf.version.VERSION) >= version.parse("2.0"):
            tf.compat.v1.reset_default_graph()
            tf.keras.backend.clear_session()
            with tf.device('/cpu:0'):
                model = keras_model_functional_for_tf2()
                model.summary()

            sess = tf.compat.v1.keras.backend.get_session()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

            # layer 1 ,3 and 5 are fused BN of different types
            with sess.as_default():
                # read weights ( beta, gamma, mean, variance)
                bn_1 = model.layers[2]
                bn_2 = model.layers[4]
                bn_3 = model.layers[6]
                keras_bn_1_params = get_bn_params_keras_layer(bn_1)
                keras_bn_2_params = get_bn_params_keras_layer(bn_2)
                keras_bn_3_params = get_bn_params_keras_layer(bn_3)

                bn_op_1 = sess.graph.get_operation_by_name('batch_normalization/FusedBatchNormV3')
                bn_op_2 = sess.graph.get_operation_by_name('scope_1/batch_normalization_1/FusedBatchNormV3')
                bn_op_3 = sess.graph.get_operation_by_name('scope_1/batch_normalization_2/FusedBatchNormV3')
                bn_1_params = get_bn_params_aimet_api(sess, bn_op_1)
                bn_2_params = get_bn_params_aimet_api(sess, bn_op_2)
                bn_3_params = get_bn_params_aimet_api(sess, bn_op_3)

                self.assertTrue(np.allclose(keras_bn_1_params, bn_1_params))
                self.assertTrue(np.allclose(keras_bn_2_params, bn_2_params))
                self.assertTrue(np.allclose(keras_bn_3_params, bn_3_params))

            sess.close()

    @pytest.mark.tf1
    def test_training_batchnorm(self):
        """
        Test BNUtils get_training() with both fused and non fused batchnorms, with all three training modes
        """
        tf.compat.v1.reset_default_graph()

        # Model with fused batchnorms
        _ = keras_model_functional()
        fused_bn_training_true_op = tf.compat.v1.get_default_graph().get_operation_by_name('batch_normalization/FusedBatchNormV3')
        self.assertTrue(BNUtils.get_training(fused_bn_training_true_op))
        self.assertTrue(isinstance(BNUtils.get_training(fused_bn_training_true_op), bool))

        fused_bn_training_tensor_op = tf.compat.v1.get_default_graph().get_operation_by_name('scope_1/batch_normalization_1/cond/'
                                                                                   'FusedBatchNormV3_1')
        training_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('is_training:0')
        self.assertEqual(BNUtils.get_training(fused_bn_training_tensor_op), training_tensor)

        fused_bn_training_false_op = tf.compat.v1.get_default_graph().get_operation_by_name('scope_1/batch_normalization_2/'
                                                                                  'FusedBatchNormV3')
        self.assertFalse(BNUtils.get_training(fused_bn_training_false_op))

        tf.compat.v1.reset_default_graph()

        # Model with non fused batchnorms
        _ = keras_model_functional_with_non_fused_batchnorms()
        bn_training_true_op = tf.compat.v1.get_default_graph().get_operation_by_name('batch_normalization/batchnorm/mul_1')
        self.assertTrue(BNUtils.get_training(bn_training_true_op))
        self.assertTrue(isinstance(BNUtils.get_training(bn_training_true_op), bool))

        bn_training_tensor_op = tf.compat.v1.get_default_graph().get_operation_by_name('scope_1/batch_normalization_1/batchnorm/'
                                                                             'mul_1')
        training_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('is_training:0')
        self.assertEqual(BNUtils.get_training(bn_training_tensor_op), training_tensor)

        bn_training_false_op = tf.compat.v1.get_default_graph().get_operation_by_name('scope_1/batch_normalization_2/batchnorm/'
                                                                            'mul_1')
        self.assertFalse(BNUtils.get_training(bn_training_false_op))

        tf.compat.v1.reset_default_graph()

    def test_training_batchnorm_for_tf2(self):
        """
        Test BNUtils get_training() with both fused and non fused batchnorms, with all three training modes
        """
        if version.parse(tf.version.VERSION) >= version.parse("2.0"):
            tf.compat.v1.reset_default_graph()

            # Model with fused batchnorms
            _ = keras_model_functional_for_tf2()
            fused_bn_training_true_op = tf.compat.v1.get_default_graph().get_operation_by_name('batch_normalization/FusedBatchNormV3')
            self.assertTrue(BNUtils.get_training(fused_bn_training_true_op))
            self.assertTrue(isinstance(BNUtils.get_training(fused_bn_training_true_op), bool))

            fused_bn_training_tensor_op = tf.compat.v1.get_default_graph().get_operation_by_name('scope_1/batch_normalization_1/FusedBatchNormV3')
            assert not BNUtils.get_training(fused_bn_training_tensor_op)
            self.assertTrue(isinstance(BNUtils.get_training(fused_bn_training_tensor_op), bool))

            fused_bn_training_false_op = tf.compat.v1.get_default_graph().get_operation_by_name('scope_1/batch_normalization_2/FusedBatchNormV3')
            self.assertFalse(BNUtils.get_training(fused_bn_training_false_op))

            tf.compat.v1.reset_default_graph()
            # Model with non fused batchnorms
            _ = keras_model_functional_with_non_fused_batchnorms_for_tf2()

            bn_training_true_op = tf.compat.v1.get_default_graph().get_operation_by_name('batch_normalization/batchnorm/mul_1')
            self.assertTrue(BNUtils.get_training(bn_training_true_op))
            self.assertTrue(isinstance(BNUtils.get_training(bn_training_true_op), bool))

            bn_training_tensor_op = tf.compat.v1.get_default_graph().get_operation_by_name('scope_1/batch_normalization_1/batchnorm/'
                                                                                 'mul_1')
            assert not BNUtils.get_training(bn_training_tensor_op)
            self.assertTrue(isinstance(BNUtils.get_training(bn_training_tensor_op), bool))

            bn_training_false_op = tf.compat.v1.get_default_graph().get_operation_by_name('scope_1/batch_normalization_2/batchnorm/'
                                                                                'mul_1')
            self.assertFalse(BNUtils.get_training(bn_training_false_op))

            tf.compat.v1.reset_default_graph()

    def test_initialize_with_bias_with_detached_ops(self):
        """
        Test that initialize with bias only affects valid ops
        """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()

        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(inputs)
        _ = tf.keras.layers.Conv2D(16, (2, 2), activation=tf.nn.tanh, use_bias=False)(conv1)
        _ = tf.keras.layers.Conv2D(8, (2, 2), activation=tf.nn.tanh)(conv1)
        graph_editor.detach_inputs(sess.graph.get_operation_by_name('conv2d_1/Conv2D'))
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        # Check that outputs of conv2d and conv2d_1 have no biases
        self.assertTrue(sess.graph.get_operation_by_name('conv2d/Conv2D').outputs[0].consumers()[0].type != 'BiasAdd')
        self.assertTrue(sess.graph.get_operation_by_name('conv2d_1/Conv2D').outputs[0].consumers()[0].type != 'BiasAdd')

        sess = BiasUtils.initialize_model_with_bias(sess, ['input_1'], ['conv2d_2/BiasAdd'])

        # Check that conv2d has a bias inserted but not conv2d_1
        self.assertTrue(sess.graph.get_operation_by_name('conv2d/Conv2D').outputs[0].consumers()[0].type == 'BiasAdd')
        self.assertTrue(sess.graph.get_operation_by_name('conv2d_1/Conv2D').outputs[0].consumers()[0].type != 'BiasAdd')

        sess.close()

    def test_tf_dataset_iterator(self):
        dataset_content = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataset = tf.compat.v1.data.Dataset.from_tensor_slices(dataset_content)
        iterator = iterate_tf_dataset(dataset)
        _assert_lists_equal(list(iterator), dataset_content)

    def test_tf_dataset_iterator_one_after_another(self):
        dataset_content = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataset = tf.compat.v1.data.Dataset.from_tensor_slices(dataset_content)

        iter_1 = iterate_tf_dataset(dataset)
        iter_1_outputs = list(iter_1)

        iter_2 = iterate_tf_dataset(dataset)
        iter_2_outputs = list(iter_2)

        assert iter_1 is not iter_2
        _assert_lists_equal(iter_1_outputs, iter_2_outputs)

    def test_tf_dataset_iterator_interleave(self):
        dataset_content = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataset = tf.compat.v1.data.Dataset.from_tensor_slices(dataset_content)

        iter_1 = iterate_tf_dataset(dataset)
        iter_2 = iterate_tf_dataset(dataset)

        iter_1_outputs = list(iter_1)
        iter_2_outputs = list(iter_2)

        assert iter_1 is not iter_2
        _assert_lists_equal(iter_1_outputs, iter_2_outputs)

    def test_tf_dataset_iterator_reuse(self):
        dataset_content = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataset = tf.compat.v1.data.Dataset.from_tensor_slices(dataset_content)
        outputs_1 = list(
            itertools.islice(iterate_tf_dataset(dataset), 5)
        )
        assert outputs_1 == dataset_content[:5]

        outputs_2 = list(
            itertools.islice(iterate_tf_dataset(dataset), 5)
        )
        assert outputs_2 == dataset_content[:5]

        # Should have instantiated and reused only one iterable
        iterables = _tf_dataset_iterables[dataset]
        assert len(iterables) == 1
        # All the associated iterators are destructed;
        # the iterable should be free by now.
        iterable = iterables[0]
        assert not iterable.is_busy()


def _assert_lists_equal(iter_1, iter_2):
    iter_1_outputs = list(iter_1)
    iter_2_outputs = list(iter_2)
    assert len(iter_1_outputs) == len(iter_2_outputs)
    for x, y in zip(iter_1_outputs, iter_2_outputs):
        assert x == y
