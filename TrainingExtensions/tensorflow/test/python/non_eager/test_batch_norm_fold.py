#!/usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019,2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" This file contains unit tests for testing cross layer scaling feature of CLE """

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import copy
import unittest
import pytest
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten, AvgPool2D, MaxPool2D

import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme

from aimet_tensorflow.utils import graph_saver
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms, find_all_batch_norms_to_fold, \
    fold_all_batch_norms_to_scale, _get_weight_tensor_transpose_reshape, _get_bn_params
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.examples.test_models import tf_slim_basic_model
from aimet_tensorflow.utils.graph import update_keras_bn_ops_trainable_flag
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils.op.conv import WeightTensorUtils
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()


class TestBatchNormFold(unittest.TestCase):
    """ Test methods for BatchNormFold"""

    def test_batch_norm_fold(self):
        """
        Test batch norm fold custom model
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(inputs)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(conv_op, training=False)
        _ = tf.nn.relu(bn_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        np.random.seed(0)
        w_shape = conv_op.inputs[0].shape
        numpy_data = np.random.rand(1, w_shape[1], w_shape[2], w_shape[3])

        relu_op = sess.graph.get_operation_by_name('Relu')
        baseline_output = sess.run(relu_op.outputs[0], feed_dict={conv_op.inputs[0]:numpy_data})

        new_sess, pairs = fold_all_batch_norms(sess, "input_1", 'Relu')

        new_conv_op = new_sess.graph.get_operation_by_name('conv2d/Conv2D')
        w2 = new_conv_op.inputs[0]
        feed_dict ={w2:numpy_data}

        new_relu_op = new_sess.graph.get_operation_by_name('Relu')
        output_after_fold = new_sess.run(new_relu_op.outputs[0], feed_dict= feed_dict)

        self.assertTrue(np.allclose(baseline_output, output_after_fold, atol=1.e-4))
        sess.close()
        new_sess.close()

    def test_bn_fold_auto_rules_bn_after_conv(self):
        """
        Test batch norm fold layer selection when conv layer is followed by a BN layer
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(conv_op)
        _ = tf.nn.relu(bn_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)
        start_op = ["inputs"]
        bn_conv_linear_pairs, _ = find_all_batch_norms_to_fold(sess, start_op, ['Relu'])
        self.assertEqual(1, len(bn_conv_linear_pairs))
        sess.close()

    def test_bn_fold_layer_selection_looped_network(self):
        """
        Test layer selection with looped network
        """
        tf.compat.v1.reset_default_graph()
        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a',
                                    kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                    bias_initializer='random_uniform')(input1)

        bn_op_1 = tf.keras.layers.BatchNormalization(fused=True)(x1)
        bn_op_2 = tf.keras.layers.BatchNormalization(fused=True)(x1)

        add = tf.keras.layers.add([bn_op_1, bn_op_2])
        _ = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b',
                                   kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                   bias_initializer='random_uniform')(add)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        start_op_name = 'input1'
        output_op_name = 'conv1b/Conv2D'

        bn_conv_linear_pairs, _ = find_all_batch_norms_to_fold(sess, [start_op_name], [output_op_name])

        self.assertTrue(0 == len(bn_conv_linear_pairs))
        sess.close()

    def test_bn_fold_auto_rules_bn_before_conv(self):
        """
        Test batch norm fold layer selection when BN layer is followed by a conv layer
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(inputs)
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(bn_op)
        _ = tf.nn.relu(conv_op)
        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        start_op = ["inputs"]
        bn_conv_linear_pairs, _ = find_all_batch_norms_to_fold(sess, start_op, ['Relu'])

        self.assertEqual(1, len(bn_conv_linear_pairs))
        sess.close()

    def test_bn_fold_find_layers_model_with_multi_input(self):
        """
        Test bn fold with multiple input nodes
        """
        tf.compat.v1.reset_default_graph()
        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        input2 = tf.keras.Input(name='input2', shape=(12, 12, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a')(input1)
        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b')(input2)
        x = tf.keras.layers.add([x1, x2])
        x = tf.keras.layers.Conv2D(4, (1, 1), name='conv2')(x)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(x)
        _ = tf.nn.relu(bn_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        start_ops = ['input1', 'input2']
        bn_conv_linear_pairs, _ = find_all_batch_norms_to_fold(sess, start_ops, ['Relu'])

        assert(1 == len(bn_conv_linear_pairs))
        sess.close()

    def test_bn_fold_find_layers_model_with_multi_input_and_training_ops(self):
        """
        Test bn fold with multiple input nodes and training_ops added
        """
        tf.compat.v1.reset_default_graph()
        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        input2 = tf.keras.Input(name='input2', shape=(12, 12, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a')(input1)
        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b')(input2)
        x = tf.keras.layers.add([x1, x2])
        x = tf.keras.layers.Conv2D(4, (1, 1), name='conv2')(x)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(x)
        output = tf.nn.relu(bn_op)

        # add training ops
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3, name='Adam_new')
        _ = optimizer.minimize(loss=output, name='train_step_new')

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        #_ = tf.compat.v1.summary.FileWriter('./multi_input', sess.graph)

        start_ops = ['input1', 'input2']
        output_op = [output.op.name]
        bn_conv_linear_pairs, _ = find_all_batch_norms_to_fold(sess, start_ops, output_op)

        assert 1 == len(bn_conv_linear_pairs)
        sess.close()

    def test_bn_fold_auto_rules_conv_bn_conv(self):
        """
        Test batch norm fold layer selection with pattern conv1 - bn - conv2
        bn folds into conv1
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(conv_op)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(bn_op)
        _ = tf.nn.relu(conv2_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)
        start_op = ["inputs"]
        bn_conv_linear_pairs, _ = find_all_batch_norms_to_fold(sess, start_op, ['Relu'])
        self.assertEqual(1, len(bn_conv_linear_pairs))
        conv_linear, batchnorm, is_batch_norm_second = bn_conv_linear_pairs[0]
        first_conv = sess.graph.get_operation_by_name('conv2d/Conv2D')
        assert first_conv == conv_linear
        # add additional check to verify backward fold is picked over forward in case both are available
        assert is_batch_norm_second is True
        sess.close()

    def test_bn_fold_with_linear_layer(self):
        """
        test bn fold on matmul layer
        Custom Model where BN layer is followed by MatMul layer
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(1, 1, 4,))
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(inputs, training=False)
        x = tf.keras.layers.Flatten()(bn_op)
        _ = tf.keras.layers.Dense(2, activation=tf.nn.relu, name="linear_layer")(x)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph())
        sess.run(init)

        op_list = sess.graph.get_operations()
        linear_layer = sess.graph.get_operation_by_name('linear_layer/MatMul')
        weight_before_fold  = WeightTensorUtils.get_tensor_as_numpy_data(sess, linear_layer)
        input_op_name = 'input_1'

        # get baseline output
        np.random.seed(0)
        input_tensor = sess.graph.get_tensor_by_name('input_1:0')
        w_shape = input_tensor.shape
        # tf 1.14 we do not have fused_batchnorm_1 in this case
        bn_layer = sess.graph.get_operation_by_name('batch_normalization/FusedBatchNormV3')
        numpy_data = np.random.rand(1, w_shape[1], w_shape[2], w_shape[3])
        relu_op = sess.graph.get_operation_by_name('linear_layer/Relu')
        baseline_output = sess.run(relu_op.outputs[0], feed_dict={bn_layer.inputs[0]:numpy_data})

        new_sess, pairs = fold_all_batch_norms(sess, input_op_name, 'linear_layer/Relu')
        linear_layer = new_sess.graph.get_operation_by_name('linear_layer/MatMul')
        weight_after_fold  = WeightTensorUtils.get_tensor_as_numpy_data(new_sess, linear_layer)

        # check that weight got updated
        self.assertFalse(np.allclose(weight_before_fold, weight_after_fold, atol=1e-4))

        # check outputs are close
        linear_layer = new_sess.graph.get_operation_by_name('linear_layer/MatMul')
        relu_op = new_sess.graph.get_operation_by_name('linear_layer/Relu')
        # after bn removal,  linear layer input is from flatten layer, that gets from input_1
        after_fold_output = new_sess.run(relu_op.outputs[0], feed_dict={linear_layer.inputs[0].op.inputs[0]:numpy_data})

        self.assertTrue(np.allclose(baseline_output, after_fold_output, atol=1e-4))
        sess.close()
        new_sess.close()

    def test_batch_norm_fold_with_random_data(self):
        """
        Test batch norm fold custom model with randomly initialized kernel, bias and bn params,
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3),
                                         kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                         bias_initializer='random_uniform')(inputs)
        bn_op = tf.keras.layers.BatchNormalization(fused=True,
                                                   beta_initializer='random_uniform',
                                                   gamma_initializer='random_uniform',
                                                   moving_mean_initializer='random_uniform',
                                                   moving_variance_initializer='ones')(conv_op, training=False)
        # @todo check why moving var with random_uniform init fails on 1.15
        _ = tf.nn.relu(bn_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        np.random.seed(0)
        w_shape = conv_op.inputs[0].shape
        numpy_data = np.random.rand(1, w_shape[1], w_shape[2], w_shape[3])

        relu_op = sess.graph.get_operation_by_name('Relu')
        baseline_output = sess.run(relu_op.outputs[0], feed_dict={conv_op.inputs[0]:numpy_data})

        new_sess, pairs = fold_all_batch_norms(sess, "input_1", 'Relu')

        new_conv_op = new_sess.graph.get_operation_by_name('conv2d/Conv2D')
        w2 = new_conv_op.inputs[0]
        feed_dict ={w2:numpy_data}

        new_relu_op = new_sess.graph.get_operation_by_name('Relu')
        output_after_fold = new_sess.run(new_relu_op.outputs[0], feed_dict= feed_dict)

        self.assertTrue(np.allclose(baseline_output, output_after_fold, atol=1.e-4))
        sess.close()
        new_sess.close()

    def test_batch_norm_conversiion(self):
        """
        Test batch norm conversion.
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3),
                                         kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                         bias_initializer='random_uniform')(inputs)
        bn_op = tf.keras.layers.BatchNormalization(fused=True,
                                                   beta_initializer='random_uniform',
                                                   gamma_initializer='random_uniform',
                                                   moving_mean_initializer='random_uniform',
                                                   moving_variance_initializer='ones')(conv_op)
        # @todo check why moving var with random_uniform init fails on 1.15
        relu_op = tf.nn.relu(bn_op)
        bn_op = tf.keras.layers.BatchNormalization(fused=True,
                                                   beta_initializer='random_uniform',
                                                   gamma_initializer='random_uniform',
                                                   moving_mean_initializer='random_uniform',
                                                   moving_variance_initializer='ones')(relu_op)
        # @todo check why moving var with random_uniform init fails on 1.15
        _ = tf.nn.relu(bn_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        np.random.seed(0)
        w_shape = conv_op.inputs[0].shape
        numpy_data = np.random.rand(1, w_shape[1], w_shape[2], w_shape[3])

        relu_op = sess.graph.get_operation_by_name('Relu_1')
        baseline_output = sess.run(relu_op.outputs[0], feed_dict={conv_op.inputs[0]:numpy_data})

        new_sess, pairs = fold_all_batch_norms(sess, "input_1", 'Relu_1')

        new_conv_op = new_sess.graph.get_operation_by_name('conv2d/Conv2D')
        w2 = new_conv_op.inputs[0]
        feed_dict ={w2:numpy_data}

        new_relu_op = new_sess.graph.get_operation_by_name('Relu_1')
        output_after_fold = new_sess.run(new_relu_op.outputs[0], feed_dict= feed_dict)

        self.assertTrue(len(pairs) == 1)
        self.assertTrue(np.allclose(baseline_output, output_after_fold, atol=1.e-4))
        sess.close()
        new_sess.close()

    def test_modify_bn_params_to_weight_bias_form(self):
        """
        Test Modify BN Parameters utility.
        """

        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(1, 1, 4,))
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(inputs, training=False)
        x = tf.keras.layers.Flatten()(bn_op)
        _ = tf.keras.layers.Dense(2, activation=tf.nn.relu, name="linear_layer")(x)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph())
        sess.run(init)

        bn_op = sess.graph.get_operation_by_name('batch_normalization/FusedBatchNormV3')

        bn_params = _get_bn_params(sess, bn_op)
        weight = np.array(bn_params.gamma) / np.array(bn_params.runningVar)
        bias = np.array(bn_params.beta) - np.array(bn_params.runningMean) * weight
        BNUtils.modify_bn_params_to_weight_bias_form(sess, bn_op, weight, bias)


        bn_params_after_conversion = _get_bn_params(sess, bn_op)

        self.assertTrue(np.allclose(np.array(bn_params_after_conversion.gamma), weight, atol=1.e-4))
        self.assertTrue(np.allclose(np.array(bn_params_after_conversion.beta), bias, atol=1.e-4))
        self.assertTrue(np.allclose(np.array(bn_params_after_conversion.runningMean),
                        np.zeros(np.array(bn_params.runningMean).shape, dtype=np.array(bn_params.runningMean).dtype), atol=1.e-4))
        self.assertTrue(np.allclose(np.array(bn_params_after_conversion.runningVar),
                                    np.ones(np.array(bn_params.runningVar).shape, dtype=np.array(bn_params.runningVar).dtype), atol=1.e-3))


        sess.close()


    @pytest.mark.tf1
    def test_removing_bn_ops_from_update_ops(self):
        """
        Test that folding batch norms also removes associated ops from update_ops, if present.
        """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        x = tf.compat.v1.placeholder(tf.float32, [1, 32, 32, 3])
        _ = tf_slim_basic_model(x)
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        # check that update_ops list is not empty
        with sess.graph.as_default():
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            self.assertEqual(4, len(update_ops))

        new_sess, pairs = fold_all_batch_norms(sess, "Placeholder", 'tf_slim_model/Softmax')

        self.assertEqual(3, len(pairs))
        # check that update_ops list is empty
        with new_sess.graph.as_default():
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            self.assertEqual(0, len(update_ops))

        sess.close()
        new_sess.close()

    def test_bn_fold_with_no_bias(self):
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(inputs)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(conv_op, training=False)
        _ = tf.nn.relu(bn_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        np.random.seed(0)
        w_shape = conv_op.inputs[0].shape
        numpy_data = np.random.rand(1, w_shape[1], w_shape[2], w_shape[3])

        relu_op = sess.graph.get_operation_by_name('Relu')
        baseline_output = sess.run(relu_op.outputs[0], feed_dict={conv_op.inputs[0]:numpy_data})
        old_conn_graph = ConnectedGraph(sess.graph, starting_op_names=['input_1'], output_op_names=['Relu'])

        new_sess, pairs = fold_all_batch_norms(sess, "input_1", 'Relu')

        new_conv_op = new_sess.graph.get_operation_by_name('conv2d/Conv2D')
        w2 = new_conv_op.inputs[0]
        feed_dict = {w2: numpy_data}

        new_relu_op = new_sess.graph.get_operation_by_name('Relu')
        output_after_fold = new_sess.run(new_relu_op.outputs[0], feed_dict= feed_dict)
        new_conn_graph = ConnectedGraph(new_sess.graph, starting_op_names=['input_1'], output_op_names=['Relu'])

        self.assertTrue(np.allclose(baseline_output, output_after_fold, atol=1.e-4))
        # New connected graph should have one less op since bn was removed
        self.assertTrue(len(old_conn_graph.get_all_ops()), len(new_conn_graph.get_all_ops()) - 1)

        sess.close()
        new_sess.close()

    def test_bn_fold_model_zoo_videnn_pose_estimation(self):
        """
        create a smaller network with connections as in pose estimation model and ViDeNN model
        Test BN fold
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(None, None, 2), name="inputs")

        x = tf.keras.layers.Conv2D(2, kernel_size=3, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization(trainable=False)(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(2, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization(trainable=False)(x)
        z = tf.keras.layers.Add()([inputs, x])
        x = tf.nn.relu(z)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        new_sess, folded_bn_conv_pairs = fold_all_batch_norms(sess, "inputs", 'Relu_1')
        self.assertEqual(len(folded_bn_conv_pairs), 2)

        sess.close()
        new_sess.close()

    def test_bn_fold_model_zoo_sr_gan(self):
        """
        create a smaller network with connections as in SR-GAN model
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(None, None, 2), name="inputs")
        x = tf.keras.layers.Conv2D(2, kernel_size=3, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization(trainable=False)(x)
        y = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(2, kernel_size=3, padding='same')(y)
        x = tf.keras.layers.BatchNormalization(trainable=False)(x)
        x = tf.keras.layers.Add()([y, x])
        _ = tf.nn.relu(x)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        new_sess, folded_bn_conv_pairs = fold_all_batch_norms(sess, "inputs", 'Relu')

        # there should be two pairs of BN- Conv picked for fold
        self.assertEqual(len(folded_bn_conv_pairs), 2)

        sess.close()
        new_sess.close()

    def test_keras_bn_op_trainable_flag_config(self):
        """
        Test to check keras bn op trainable flag update
        :return:
        """
        tf.compat.v1.reset_default_graph()
        model = Sequential([
            Conv2D(8, (2, 2), input_shape=(16, 16, 3,)),
            BatchNormalization(momentum=.3, epsilon=.65),
            AvgPool2D(),
            MaxPool2D(),
            BatchNormalization(momentum=.4, epsilon=.25),
            Conv2D(4, (2, 2), activation=tf.nn.tanh, kernel_regularizer=tf.keras.regularizers.l2(0.5)),
            Flatten(),
            Dense(2, activation='softmax', name="keras_model")])

        _ = update_keras_bn_ops_trainable_flag(model, False, "./data")
        sess = tf.compat.v1.keras.backend.get_session()
        new_sess, folded_bn_conv_pairs = fold_all_batch_norms(sess, "conv2d_input", 'keras_model/Softmax')
        self.assertTrue(len(folded_bn_conv_pairs) == 2)

    def test_fold_forward(self):
        """ test to check fold_backward flag """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        x = tf.keras.layers.BatchNormalization()(inputs)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="keras_model_functional")(x)
        tf.keras.Model(inputs=inputs, outputs=outputs)
        graph = tf.compat.v1.get_default_graph()
        sess = tf.compat.v1.Session(graph=graph)
        initialize_uninitialized_vars(sess)
        start_op_names = ["input_1"]
        output_op_names = ["keras_model_functional/Softmax"]
        pairs = find_all_batch_norms_to_fold(sess, start_op_names, output_op_names, return_bn_conn_op=True)
        conv, bn, fold_backward = pairs[0]
        assert not fold_backward
        assert bn.type == 'FusedBatchNormV3'
        sess.close()


def get_sim_model_conv2d_FusedBatchNormV3():
    tf.compat.v1.reset_default_graph()
    inputs = tf.keras.Input(shape=(32, 32, 3,))
    conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
    bn_op = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=1.0,
                                               moving_mean_initializer=tf.compat.v1.random_uniform_initializer(),
                                               moving_variance_initializer=tf.compat.v1.random_uniform_initializer(),
                                               beta_initializer=tf.compat.v1.random_uniform_initializer(),
                                               gamma_initializer=tf.compat.v1.random_uniform_initializer(),
                                               fused=True)(conv_op, training=False)
    rulu_op = tf.nn.relu(bn_op)
    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init)


    quantsim_config = {
        "defaults": {
            "ops": {"is_output_quantized": "True"},
            "params": {"is_quantized": "True"},
            "strict_symmetric": "False",
            "unsigned_symmetric": "True",
            "per_channel_quantization": "True"
        },
        "params": {
            "bias": {"is_quantized": "False"}
        },
        "op_type": {},
        "supergroups": [
            {"op_list": ["Conv", "BatchNormalization"]},
            {"op_list": ["Gemm", "BatchNormalization"]}

        ],
        "model_input": {"is_input_quantized": "True"},
        "model_output": {}
    }

    with open('./quantsim_config.json', 'w') as f:
        json.dump(quantsim_config, f)


    sim = QuantizationSimModel(sess, ["input_1"], ['Relu'], use_cuda=True,
                               quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                               config_file='./quantsim_config.json')

    fp32_conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
    w_shape = fp32_conv_op.inputs[0].shape
    numpy_data = np.random.rand(128, w_shape[1], w_shape[2], w_shape[3])

    def dummy_forward_pass(sess, args):
        model_input = sess.graph.get_tensor_by_name('input_1:0')
        model_output = sess.graph.get_tensor_by_name('Relu:0')
        sess.run(model_output, feed_dict={model_input: numpy_data})

    sim.compute_encodings(dummy_forward_pass, None)

    return sim, numpy_data


symmetric_quantsim_config ={
    "defaults": {
        "ops": { "is_output_quantized": "True" },
        "params": { "is_quantized": "True", "is_symmetric": "True"},
        "strict_symmetric": "False",
        "unsigned_symmetric": "True",
        "per_channel_quantization": "True"
    },
    "params": {
        "bias": { "is_quantized": "False" }
    },
    "op_type": {},
    "supergroups": [
        { "op_list": ["Conv", "Relu"] },
        { "op_list": ["Conv", "Clip"] },
        { "op_list": ["Add", "Relu"] },
        { "op_list": ["Gemm", "Relu"] },
    ],
    "model_input": { "is_input_quantized": "True" },
    "model_output": {}
}

asymmetric_quantsim_config = copy.deepcopy(symmetric_quantsim_config)
asymmetric_quantsim_config["defaults"]["params"]["is_symmetric"] = "False"

strict_symmetric_quantsim_config = copy.deepcopy(symmetric_quantsim_config)
strict_symmetric_quantsim_config["defaults"]["strict_symmetric"] = "True"

quantsim_config_map = {
    "symmetric": symmetric_quantsim_config,
    "asymmetric": asymmetric_quantsim_config,
}


def quantsim(session, start_op_names, output_op_names, dummy_input, quantsim_config=None):
    config_file_path = "/tmp/quantsim_config.json"
    quantsim_config = quantsim_config or symmetric_quantsim_config
    try:
        with open(config_file_path, 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(session,
                                   start_op_names,
                                   output_op_names,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   config_file=config_file_path)
        def forward_pass_callback(sess, _):
            input_tensor = sess.graph.get_tensor_by_name(start_op_names[0] + ':0')
            output_tensor = sess.graph.get_tensor_by_name(output_op_names[0] + ':0')
            sess.run(output_tensor, feed_dict={input_tensor: dummy_input})

        sim.compute_encodings(forward_pass_callback, None)
        return sim

    finally:
        try:
            os.remove(config_file_path)
        except FileNotFoundError:
            pass

def _get_inp_out_tensor(session, inp_tensor_name, out_tensor_name):
    inp_tensor = session.graph.get_tensor_by_name(inp_tensor_name[0] + ':0')
    out_tensor = session.graph.get_tensor_by_name(out_tensor_name[0] + ':0')
    return inp_tensor, out_tensor


class TestTrainingExtensionBnFoldToScale:
    """ Test methods for BatchNormFold with QuantizationSimModel"""

    @pytest.mark.cuda
    @pytest.mark.parametrize("is_training_variable", [True, False])
    def test_fold_conv_with_bias_bn_relu(self, is_training_variable):
        """
        test conv (no bias) + bn + relu sequence
        """
        np.random.seed(43)
        tf.compat.v1.reset_default_graph()
        tf.random.set_seed(43)

        def model():
            """ Model with conv + bn + relu sequence """
            training = tf.Variable(False, name='bn_training_var') if is_training_variable else False
            inputs = tf.keras.Input(shape=(24, 24, 10,))
            x = tf.keras.layers.Conv2D(10, (3, 3))(inputs)
            x = tf.keras.layers.BatchNormalization(beta_initializer=tf.random_normal_initializer(),
                                                   gamma_initializer=tf.random_normal_initializer(),
                                                   moving_mean_initializer=tf.random_normal_initializer(),
                                                   moving_variance_initializer=tf.random_uniform_initializer(0)) \
                (x, training)
            x = tf.nn.relu(x)
            outputs = tf.keras.layers.Flatten()(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        device = '/gpu:0'
        with tf.device(device):
            _ = model()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        start_op_names = ["input_1"]
        output_op_names = ["flatten/Reshape"]
        dummy_input = np.random.randn(1, 24, 24, 10)
        sim = quantsim(sess, start_op_names, output_op_names, dummy_input)

        # Check quantizers are enabled/disabled properly
        conv_a_quantizer = sim.quantizer_config('conv2d/BiasAdd_quantized')
        conv_w_quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')
        if is_training_variable:
            bn_a_quantizer = sim.quantizer_config('batch_normalization/cond/Identity_quantized')
        else:
            bn_a_quantizer = sim.quantizer_config('batch_normalization/FusedBatchNormV3_quantized')
        relu_a_quantizer = sim.quantizer_config('Relu_quantized')

        assert not conv_a_quantizer.enabled
        assert conv_w_quantizer.enabled
        assert not bn_a_quantizer.enabled
        assert relu_a_quantizer.enabled

        input_tensor, output_tensor = _get_inp_out_tensor(sim.session, start_op_names, output_op_names)
        baseline_output = sim.session.run(output_tensor, feed_dict={input_tensor: dummy_input})

        fold_all_batch_norms_to_scale(sim, start_op_names, output_op_names)

        input_tensor, output_tensor = _get_inp_out_tensor(sim.session, start_op_names, output_op_names)
        output_after_fold = sim.session.run(output_tensor, feed_dict={input_tensor: dummy_input})

        # Check quantizers are enabled/disabled properly
        assert not conv_a_quantizer.enabled
        assert conv_w_quantizer.enabled
        assert not bn_a_quantizer.enabled
        assert relu_a_quantizer.enabled

        relu_output_encoding = relu_a_quantizer.get_encoding()
        delta = float((relu_output_encoding.max - relu_output_encoding.min) / 255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference

        sess.close()
        sim.session.close()

    @pytest.mark.cuda
    @pytest.mark.parametrize("is_training_variable", [True, False])
    def test_fold_conv_no_bias_bn_relu(self, is_training_variable):
        """
        test conv (no bias) + bn + relu sequence
        """
        np.random.seed(43)
        tf.compat.v1.reset_default_graph()
        tf.random.set_seed(43)

        def model():
            """ Model with conv + bn + relu sequence """
            training = tf.Variable(False, name='bn_training_var') if is_training_variable else False
            inputs = tf.keras.Input(shape=(24, 24, 10,))
            x = tf.keras.layers.Conv2D(10, (3, 3), use_bias=False)(inputs)
            x = tf.keras.layers.BatchNormalization(beta_initializer=tf.random_normal_initializer(),
                                                   gamma_initializer=tf.random_normal_initializer(),
                                                   moving_mean_initializer=tf.random_normal_initializer(),
                                                   moving_variance_initializer=tf.random_uniform_initializer(0)) \
                (x, training)
            x = tf.nn.relu(x)
            outputs = tf.keras.layers.Flatten()(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        device = '/gpu:0'
        with tf.device(device):
            _ = model()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        start_op_names = ["input_1"]
        output_op_names = ["flatten/Reshape"]
        dummy_input = np.random.randn(1, 24, 24, 10)
        sim = quantsim(sess, start_op_names, output_op_names, dummy_input)

        # Check quantizers are enabled/disabled properly
        conv_a_quantizer = sim.quantizer_config('conv2d/Conv2D_quantized')
        conv_w_quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')
        if is_training_variable:
            bn_a_quantizer = sim.quantizer_config('batch_normalization/cond/Identity_quantized')
        else:
            bn_a_quantizer = sim.quantizer_config('batch_normalization/FusedBatchNormV3_quantized')
        relu_a_quantizer = sim.quantizer_config('Relu_quantized')

        assert not conv_a_quantizer.enabled
        assert conv_w_quantizer.enabled
        assert not bn_a_quantizer.enabled
        assert relu_a_quantizer.enabled

        input_tensor, output_tensor = _get_inp_out_tensor(sim.session, start_op_names, output_op_names)
        baseline_output = sim.session.run(output_tensor, feed_dict={input_tensor: dummy_input})

        fold_all_batch_norms_to_scale(sim, start_op_names, output_op_names)

        input_tensor, output_tensor = _get_inp_out_tensor(sim.session, start_op_names, output_op_names)
        output_after_fold = sim.session.run(output_tensor, feed_dict={input_tensor: dummy_input})

        # Check quantizers are enabled/disabled properly
        assert not conv_a_quantizer.enabled
        assert conv_w_quantizer.enabled
        assert not bn_a_quantizer.enabled
        assert relu_a_quantizer.enabled

        relu_output_encoding = relu_a_quantizer.get_encoding()
        delta = float((relu_output_encoding.max - relu_output_encoding.min) / 255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference

        sess.close()
        sim.session.close()

    @pytest.mark.cuda
    @pytest.mark.parametrize("is_training_variable", [True, False])
    def test_fold_depthwise_conv_with_bias_bn_relu(self, is_training_variable):
        """
        test depthwise (with bias) + bn + relu sequence
        """
        np.random.seed(43)
        tf.compat.v1.reset_default_graph()
        tf.random.set_seed(43)

        def model():
            """ Model with depthwise (with bias) + bn + relu sequence """
            training = tf.Variable(False, name='bn_training_var') if is_training_variable else False
            inputs = tf.keras.Input(shape=(24, 24, 10,))
            x = tf.keras.layers.DepthwiseConv2D(10, (3, 3))(inputs)
            x = tf.keras.layers.BatchNormalization(beta_initializer=tf.random_normal_initializer(),
                                                   gamma_initializer=tf.random_normal_initializer(),
                                                   moving_mean_initializer=tf.random_normal_initializer(),
                                                   moving_variance_initializer=tf.random_uniform_initializer(0)) \
                (x, training)
            x = tf.nn.relu(x)
            outputs = tf.keras.layers.Flatten()(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        device = '/gpu:0'
        with tf.device(device):
            _ = model()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        start_op_names = ["input_1"]
        output_op_names = ["flatten/Reshape"]
        dummy_input = np.random.randn(1, 24, 24, 10)
        sim = quantsim(sess, start_op_names, output_op_names, dummy_input)

        # Check quantizers are enabled/disabled properly
        depthwise_conv_a_quantizer = sim.quantizer_config('depthwise_conv2d/BiasAdd_quantized')
        depthwise_conv_w_quantizer = sim.quantizer_config('depthwise_conv2d/depthwise/ReadVariableOp_quantized')
        if is_training_variable:
            bn_a_quantizer = sim.quantizer_config('batch_normalization/cond/Identity_quantized')
        else:
            bn_a_quantizer = sim.quantizer_config('batch_normalization/FusedBatchNormV3_quantized')
        relu_a_quantizer = sim.quantizer_config('Relu_quantized')

        assert not depthwise_conv_a_quantizer.enabled
        assert depthwise_conv_w_quantizer.enabled
        assert not bn_a_quantizer.enabled
        assert relu_a_quantizer.enabled

        input_tensor, output_tensor = _get_inp_out_tensor(sim.session, start_op_names, output_op_names)
        baseline_output = sim.session.run(output_tensor, feed_dict={input_tensor: dummy_input})

        fold_all_batch_norms_to_scale(sim, start_op_names, output_op_names)

        input_tensor, output_tensor = _get_inp_out_tensor(sim.session, start_op_names, output_op_names)
        output_after_fold = sim.session.run(output_tensor, feed_dict={input_tensor: dummy_input})

        # Check quantizers are enabled/disabled properly
        assert not depthwise_conv_a_quantizer.enabled
        assert depthwise_conv_w_quantizer.enabled
        assert not bn_a_quantizer.enabled
        assert relu_a_quantizer.enabled

        relu_output_encoding = relu_a_quantizer.get_encoding()
        delta = float((relu_output_encoding.max - relu_output_encoding.min) / 255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference

        sess.close()
        sim.session.close()

    @pytest.mark.cuda
    @pytest.mark.parametrize("is_training_variable", [True, False])
    def test_fold_depthwise_conv_no_bias_bn_relu(self, is_training_variable):
        """
        test depthwise (no bias) + bn + relu sequence
        """
        np.random.seed(43)
        tf.compat.v1.reset_default_graph()
        tf.random.set_seed(43)

        def model():
            """ Model with depthwise (no bias) + bn + relu sequence """
            training = tf.Variable(False, name='bn_training_var') if is_training_variable else False
            inputs = tf.keras.Input(shape=(24, 24, 10,))
            x = tf.keras.layers.DepthwiseConv2D(10, (3, 3), use_bias=False)(inputs)
            x = tf.keras.layers.BatchNormalization(beta_initializer=tf.random_normal_initializer(),
                                                   gamma_initializer=tf.random_normal_initializer(),
                                                   moving_mean_initializer=tf.random_normal_initializer(),
                                                   moving_variance_initializer=tf.random_uniform_initializer(0)) \
                (x, training)
            x = tf.nn.relu(x)
            outputs = tf.keras.layers.Flatten()(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        device = '/gpu:0'
        with tf.device(device):
            _ = model()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        start_op_names = ["input_1"]
        output_op_names = ["flatten/Reshape"]
        dummy_input = np.random.randn(1, 24, 24, 10)
        sim = quantsim(sess, start_op_names, output_op_names, dummy_input)

        # Check quantizers are enabled/disabled properly
        depthwise_conv_a_quantizer = sim.quantizer_config('depthwise_conv2d/depthwise_quantized')
        depthwise_conv_w_quantizer = sim.quantizer_config('depthwise_conv2d/depthwise/ReadVariableOp_quantized')
        if is_training_variable:
            bn_a_quantizer = sim.quantizer_config('batch_normalization/cond/Identity_quantized')
        else:
            bn_a_quantizer = sim.quantizer_config('batch_normalization/FusedBatchNormV3_quantized')
        relu_a_quantizer = sim.quantizer_config('Relu_quantized')

        assert not depthwise_conv_a_quantizer.enabled
        assert depthwise_conv_w_quantizer.enabled
        assert not bn_a_quantizer.enabled
        assert relu_a_quantizer.enabled

        input_tensor, output_tensor = _get_inp_out_tensor(sim.session, start_op_names, output_op_names)
        baseline_output = sim.session.run(output_tensor, feed_dict={input_tensor: dummy_input})

        fold_all_batch_norms_to_scale(sim, start_op_names, output_op_names)

        input_tensor, output_tensor = _get_inp_out_tensor(sim.session, start_op_names, output_op_names)
        output_after_fold = sim.session.run(output_tensor, feed_dict={input_tensor: dummy_input})

        # Check quantizers are enabled/disabled properly
        assert not depthwise_conv_a_quantizer.enabled
        assert depthwise_conv_w_quantizer.enabled
        assert not bn_a_quantizer.enabled
        assert relu_a_quantizer.enabled

        relu_output_encoding = relu_a_quantizer.get_encoding()
        delta = float((relu_output_encoding.max - relu_output_encoding.min) / 255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference

        sess.close()
        sim.session.close()

    @pytest.mark.cuda
    @pytest.mark.parametrize("is_training_variable", [True, False])
    def test_fold_conv_with_bias_bn(self, is_training_variable):
        """
        test conv + bn sequence
        """
        np.random.seed(43)
        tf.compat.v1.reset_default_graph()
        tf.random.set_seed(43)
        def model():
            """ Model with conv (with bias) + bn sequence """
            training = tf.Variable(False, name='bn_training_var') if is_training_variable else False
            inputs = tf.keras.Input(shape=(24, 24, 10,))
            x = tf.keras.layers.Conv2D(10, (3, 3))(inputs)
            x = tf.keras.layers.BatchNormalization(beta_initializer=tf.random_normal_initializer(),
                                                   gamma_initializer=tf.random_normal_initializer(),
                                                   moving_mean_initializer=tf.random_normal_initializer(),
                                                   moving_variance_initializer=tf.random_uniform_initializer(0)) \
                (x, training)
            outputs = tf.keras.layers.Flatten()(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        device = '/gpu:0'
        with tf.device(device):
            _ = model()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        start_op_names = ["input_1"]
        output_op_names = ["flatten/Reshape"]
        dummy_input = np.random.randn(1, 24, 24, 10)
        sim = quantsim(sess, start_op_names, output_op_names, dummy_input)

        # Check quantizers are enabled/disabled properly
        conv_a_quantizer = sim.quantizer_config('conv2d/BiasAdd_quantized')
        conv_w_quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')
        if is_training_variable:
            bn_a_quantizer = sim.quantizer_config('batch_normalization/cond/Identity_quantized')
        else:
            bn_a_quantizer = sim.quantizer_config('batch_normalization/FusedBatchNormV3_quantized')

        assert not conv_a_quantizer.enabled
        assert conv_w_quantizer.enabled
        assert bn_a_quantizer.enabled
        bn_output_encoding = bn_a_quantizer.get_encoding()

        input_tensor, output_tensor = _get_inp_out_tensor(sim.session, start_op_names, output_op_names)
        baseline_output = sim.session.run(output_tensor, feed_dict={input_tensor: dummy_input})

        fold_all_batch_norms_to_scale(sim, start_op_names, output_op_names)

        input_tensor, output_tensor = _get_inp_out_tensor(sim.session, start_op_names, output_op_names)
        output_after_fold = sim.session.run(output_tensor, feed_dict={input_tensor: dummy_input})

        # Check quantizers are enabled/disabled properly
        assert conv_a_quantizer.enabled
        assert conv_w_quantizer.enabled
        assert not bn_a_quantizer.enabled

        # Check batchnorm's output encoding is copied to conv's output encoding
        conv_output_encoding = conv_a_quantizer.get_encoding()
        assert conv_output_encoding.max == bn_output_encoding.max and \
               conv_output_encoding.min == bn_output_encoding.min and \
               conv_output_encoding.delta == bn_output_encoding.delta and \
               conv_output_encoding.offset == bn_output_encoding.offset and \
               conv_output_encoding.bw == bn_output_encoding.bw

        conv_output_encoding = conv_a_quantizer.get_encoding()
        delta = float((conv_output_encoding.max - conv_output_encoding.min) / 255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference

        sess.close()
        sim.session.close()

    @pytest.mark.cuda
    @pytest.mark.parametrize("is_training_variable", [True, False])
    def test_fold_conv_no_bias_bn(self, is_training_variable):
        """
        test conv (no bias) + bn sequence
        """
        np.random.seed(43)
        tf.compat.v1.reset_default_graph()
        tf.random.set_seed(43)

        def model():
            """ Model with conv (no bias) + bn sequence """
            training = tf.Variable(False, name='bn_training_var') if is_training_variable else False
            inputs = tf.keras.Input(shape=(24, 24, 10,))
            x = tf.keras.layers.Conv2D(10, (3, 3), use_bias=False)(inputs)
            x = tf.keras.layers.BatchNormalization(beta_initializer=tf.random_normal_initializer(),
                                                   gamma_initializer=tf.random_normal_initializer(),
                                                   moving_mean_initializer=tf.random_normal_initializer(),
                                                   moving_variance_initializer=tf.random_uniform_initializer(0)) \
                (x, training)
            outputs = tf.keras.layers.Flatten()(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        device = '/gpu:0'
        with tf.device(device):
            _ = model()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        start_op_names = ["input_1"]
        output_op_names = ["flatten/Reshape"]
        dummy_input = np.random.randn(1, 24, 24, 10)
        sim = quantsim(sess, start_op_names, output_op_names, dummy_input)

        # Check quantizers are enabled/disabled properly
        conv_a_quantizer = sim.quantizer_config('conv2d/Conv2D_quantized')
        conv_w_quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')
        if is_training_variable:
            bn_a_quantizer = sim.quantizer_config('batch_normalization/cond/Identity_quantized')
        else:
            bn_a_quantizer = sim.quantizer_config('batch_normalization/FusedBatchNormV3_quantized')

        assert not conv_a_quantizer.enabled
        assert conv_w_quantizer.enabled
        assert bn_a_quantizer.enabled
        bn_output_encoding = bn_a_quantizer.get_encoding()

        input_tensor, output_tensor = _get_inp_out_tensor(sim.session, start_op_names, output_op_names)
        baseline_output = sim.session.run(output_tensor, feed_dict={input_tensor: dummy_input})

        fold_all_batch_norms_to_scale(sim, start_op_names, output_op_names)

        input_tensor, output_tensor = _get_inp_out_tensor(sim.session, start_op_names, output_op_names)
        output_after_fold = sim.session.run(output_tensor, feed_dict={input_tensor: dummy_input})

        # Check quantizers are enabled/disabled properly
        assert conv_a_quantizer.enabled
        assert conv_w_quantizer.enabled
        assert not bn_a_quantizer.enabled

        # Check batchnorm's output encoding is copied to conv's output encoding
        conv_output_encoding = conv_a_quantizer.get_encoding()
        assert conv_output_encoding.max == bn_output_encoding.max and \
               conv_output_encoding.min == bn_output_encoding.min and \
               conv_output_encoding.delta == bn_output_encoding.delta and \
               conv_output_encoding.offset == bn_output_encoding.offset and \
               conv_output_encoding.bw == bn_output_encoding.bw

        delta = float((conv_output_encoding.max - conv_output_encoding.min) / 255)
        assert np.allclose(baseline_output, output_after_fold, atol=delta)  # Allow 1-tick difference

        sess.close()
        sim.session.close()
