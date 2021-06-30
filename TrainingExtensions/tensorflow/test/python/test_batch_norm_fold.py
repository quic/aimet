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
import unittest
import tensorflow as tf
import numpy as np

from aimet_tensorflow.batch_norm_fold import  fold_all_batch_norms, find_all_batch_norms_to_fold
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.examples.test_models import tf_compat_v1_layers_basic_model
from aimet_tensorflow.utils.op.conv import WeightTensorUtils

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class TestBatchNormFold(unittest.TestCase):
    """ Test methods for BatchNormFold"""

    def test_batch_norm_fold(self):
        """
        Test batch norm fold custom model
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn_op = tf.compat.v1.layers.batch_normalization(conv_op, fused=False, training=False)
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
        bn_op = tf.compat.v1.layers.batch_normalization(conv_op, fused=True)
        _ = tf.nn.relu(bn_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)
        bn_conv_linear_pairs = find_all_batch_norms_to_fold(sess, ["inputs"], ['Relu'])
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
        bn_op_1 = tf.compat.v1.layers.batch_normalization(x1, fused=True)
        bn_op_2 = tf.compat.v1.layers.batch_normalization(x1, fused=True)
        add = tf.keras.layers.add([bn_op_1, bn_op_2])
        _ = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b',
                                   kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                   bias_initializer='random_uniform')(add)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        start_op_name = ['input1']
        output_op_name = ['conv1b/Conv2D']

        bn_conv_linear_pairs = find_all_batch_norms_to_fold(sess, start_op_name, output_op_name)

        self.assertTrue(0 == len(bn_conv_linear_pairs))
        sess.close()

    def test_bn_fold_auto_rules_bn_before_conv(self):
        """
        Test batch norm fold layer selection when BN layer is followed by a conv layer
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        bn_op = tf.compat.v1.layers.batch_normalization(inputs, fused=True)
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(bn_op)
        _ = tf.nn.relu(conv_op)
        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        bn_conv_linear_pairs = find_all_batch_norms_to_fold(sess, ["inputs"], ['Relu'])

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
        bn_op = tf.compat.v1.layers.batch_normalization(x, fused=True)
        _ = tf.nn.relu(bn_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        start_ops = ['input1', 'input2']
        bn_conv_linear_pairs = find_all_batch_norms_to_fold(sess, start_ops, ['Relu'])

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
        bn_op = tf.compat.v1.layers.batch_normalization(x, fused=True)
        output = tf.nn.relu(bn_op)

        # add training ops
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3, name='Adam_new')
        _ = optimizer.minimize(loss=output, name='train_step_new')

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        start_ops = ['input1', 'input2']
        output_op = [output.op.name]
        bn_conv_linear_pairs = find_all_batch_norms_to_fold(sess, start_ops, output_op)

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
        bn_op = tf.compat.v1.layers.batch_normalization(conv_op, fused=True)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(bn_op)
        _ = tf.nn.relu(conv2_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)
        start_op = ["inputs"]
        bn_conv_linear_pairs = find_all_batch_norms_to_fold(sess, start_op, ['Relu'])
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
        bn_op = tf.compat.v1.layers.batch_normalization(inputs, fused=True, training=False)
        x = tf.keras.layers.Flatten()(bn_op)
        _ = tf.keras.layers.Dense(2, activation=tf.nn.relu, name="linear_layer")(x)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph())
        sess.run(init)

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
        bn_op = tf.compat.v1.layers.batch_normalization(conv_op, fused=True, training=False,
                                                        beta_initializer='random_uniform',
                                                        gamma_initializer='random_uniform',
                                                        moving_mean_initializer='random_uniform',
                                                        moving_variance_initializer='ones')
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

    def test_removing_bn_ops_from_update_ops(self):
        """
        Test that folding batch norms also removes associated ops from update_ops, if present.
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        _ = tf_compat_v1_layers_basic_model(inputs)
        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        # check that update_ops list is not empty
        with sess.graph.as_default():
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            self.assertEqual(4, len(update_ops))

        new_sess, pairs = fold_all_batch_norms(sess, "input_1", 'dense/Softmax')

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
        bn_op = tf.compat.v1.layers.batch_normalization(conv_op, fused=True, training=False)
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
        x = tf.compat.v1.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(2, kernel_size=3, padding='same')(x)
        x = tf.compat.v1.layers.batch_normalization(x)
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
        x = tf.compat.v1.layers.batch_normalization(x)
        y = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(2, kernel_size=3, padding='same')(y)
        x = tf.compat.v1.layers.batch_normalization(x)
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
