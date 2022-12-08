# /usr/bin/env python3.6
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
import pytest
import logging
import os

import logging
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.layers import normalization as normalization_layers

from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_common.utils import AimetLogger
from aimet_tensorflow.utils.op.fusedbatchnorm import BNUtils
from aimet_tensorflow.utils.op.bn_mutable import modify_model_bn_mutable, modify_sess_bn_mutable
from aimet_tensorflow.batch_norm_fold import find_all_batch_norms_to_fold
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
AimetLogger.set_level_for_all_areas(logging.DEBUG)

tf.compat.v1.disable_eager_execution()
np.random.seed(0)
tf.compat.v1.set_random_seed(0)

def slim_mutable_bn_model(training_as_placeholder=False):
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.variable_scope("foo"):

        if training_as_placeholder:
            bn_training = tf.compat.v1.placeholder_with_default(True, shape=[], name='bn_training_placeholder')
        else:
            # bn_training = False
            bn_training = tf.compat.v1.placeholder_with_default(False, shape=[], name='bn_training_placeholder')

        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)

        layer = normalization_layers.BatchNormalization(name="old1")
        bn_op = layer.apply(conv_op, training=bn_training)

        relu0 = tf.nn.relu(bn_op)
        conv_op1 = tf.keras.layers.Conv2D(32, (3, 3))(relu0)


        bn_op1 = tf.compat.v1.layers.batch_normalization(conv_op1, name="old2", training=bn_training,fused=True)

        relu1 = tf.nn.relu(bn_op1)

def v1_bn_model(training_as_placeholder=False):
    tf.compat.v1.reset_default_graph()

    if training_as_placeholder:
        bn_training = tf.compat.v1.placeholder_with_default(False, shape=[], name='bn_training_placeholder')
    else:
        bn_training = False

    inputs = tf.keras.Input(shape=(32, 32, 3,))
    conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)

    bn_op = tf.compat.v1.layers.batch_normalization(conv_op, name="old1", training=bn_training,
                                                    beta_initializer=tf.compat.v1.random_uniform_initializer(),
                                                    gamma_initializer=tf.compat.v1.random_uniform_initializer(),
                                                    moving_mean_initializer=tf.compat.v1.random_uniform_initializer(),
                                                    moving_variance_initializer=tf.compat.v1.random_uniform_initializer(),
                                                    fused=True)
    relu0 = tf.nn.relu(bn_op)

    conv_op1 = tf.keras.layers.Conv2D(32, (3, 3))(relu0)

    bn_op1 = tf.compat.v1.layers.batch_normalization(conv_op1, name="old2", training=bn_training,
                                                     beta_initializer=tf.compat.v1.random_uniform_initializer(),
                                                     gamma_initializer=tf.compat.v1.random_uniform_initializer(),
                                                     moving_mean_initializer=tf.compat.v1.random_uniform_initializer(),
                                                     moving_variance_initializer=tf.compat.v1.random_uniform_initializer(),
                                                     fused=True)

    relu1 = tf.nn.relu(bn_op1)


def keras_bn_model(training_as_placeholder=False):
    tf.compat.v1.reset_default_graph()

    if training_as_placeholder:
        bn_training = tf.compat.v1.placeholder_with_default(False, shape=[], name='bn_training_placeholder')

    inputs = tf.keras.Input(shape=(32, 32, 3,))
    conv_output = tf.keras.layers.Conv2D(32, (3, 3))(inputs)

    bn_op = tf.keras.layers.BatchNormalization(name="old1",
                                               beta_initializer=tf.compat.v1.random_uniform_initializer(),
                                               gamma_initializer=tf.compat.v1.random_uniform_initializer(),
                                               moving_mean_initializer=tf.compat.v1.random_uniform_initializer(),
                                               moving_variance_initializer=tf.compat.v1.random_uniform_initializer(),
                                               fused=True) ## , fused=True  _fused=False get_all_tf_bn_vars_list_without_momentum   ValueError: not enough

    if training_as_placeholder:
        bn_output = bn_op(conv_output, training=bn_training)
    else:
        bn_output = bn_op(conv_output)

    # values to unpack (expected 2, got 0)
    relu = tf.nn.relu(bn_output)
    conv1_output = tf.keras.layers.Conv2D(32, (3, 3))(relu)
    bn1_op = tf.keras.layers.BatchNormalization(name="old2",
                                                beta_initializer=tf.compat.v1.random_uniform_initializer(),
                                                gamma_initializer=tf.compat.v1.random_uniform_initializer(),
                                                moving_mean_initializer=tf.compat.v1.random_uniform_initializer(),
                                                moving_variance_initializer=tf.compat.v1.random_uniform_initializer(),
                                                fused=True)

    if training_as_placeholder:
        bn1_output = bn1_op(conv1_output, training=bn_training)
    else:
        bn1_output = bn1_op(conv1_output)

    relu1 = tf.nn.relu(bn1_output)
    model = tf.keras.Model(inputs=inputs, outputs=relu1)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def keras_applications_mobilenet_v2():
    tf.compat.v1.reset_default_graph()
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=None, input_shape=(32, 32, 3))
    return model


class TestBnMutable(unittest.TestCase):
    def test_modify_sess_bn_mutable_with_v1_slim_bn_model(self):
        slim_mutable_bn_model()
        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        input_op_names = ['foo/input_1']
        output_op_names = ['foo/Relu_1']
        dummy_val = np.random.randn(1, 32, 32, 3)

        conn_graph_old = ConnectedGraph(sess.graph, input_op_names, output_op_names)

        self.assertEqual(len(conn_graph_old.get_all_ops()), 7)

        # Fetch conn graph op of batchnorm layers
        bn1_old = conn_graph_old.get_op_from_module_name('foo/old1/cond/Identity')
        bn2_old = conn_graph_old.get_op_from_module_name('foo/old2/cond/Identity')

        self.assertEqual(bn1_old.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_old.type, 'FusedBatchNormV3')

        # Fetch outputs of batchnorms
        bn1_old_tensor = bn1_old.get_module().outputs[0]
        bn2_old_tensor = bn2_old.get_module().outputs[0]

        input_old_tensor = sess.graph.get_tensor_by_name('foo/input_1:0')
        relu_old_tensor = sess.graph.get_tensor_by_name('foo/Relu_1:0')
        bn1_old_output, bn2_old_output, relu_old_output = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                                                   feed_dict={input_old_tensor: dummy_val})


        # get old batchnorms status
        bn_old1_op = sess.graph.get_operation_by_name(bn1_old.get_module().name)
        bn_old2_op = sess.graph.get_operation_by_name(bn2_old.get_module().name)

        old1_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_old1_op)
        old2_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_old2_op)

        old1_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_old1_op)
        old2_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_old2_op)

        old1_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_old1_op)
        old2_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_old2_op)

        old1_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_old1_op)
        old2_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_old2_op)

        old1_beta, old2_beta, old1_gamma, old2_gamma, old1_mean, old2_mean, old1_var, old2_var = \
            sess.run([old1_beta_read_var, old2_beta_read_var, old1_gamma_read_var, old2_gamma_read_var,
                      old1_mean_read_var, old2_mean_read_var, old1_var_read_var, old2_var_read_var])

        # Modify batchnorm of sess
        modify_sess_bn_mutable(sess, input_op_names, output_op_names)


        conn_graph_new = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_new.get_all_ops()), 7)

        # Check if bn layer with modified_bn_old1 and modified_bn_old2 is created correctly
        bn1_new = conn_graph_new.get_op_from_module_name('foo/old1_modified/cond/Identity')
        bn2_new = conn_graph_new.get_op_from_module_name('foo/old2_modified/cond/Identity')

        self.assertEqual(bn1_new.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_new.type, 'FusedBatchNormV3')

        # get new batchnorms status
        bn1_new_op = sess.graph.get_operation_by_name(bn1_new.get_module().name)
        bn2_new_op = sess.graph.get_operation_by_name(bn2_new.get_module().name)

        new1_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn1_new_op)
        new2_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn2_new_op)
        new1_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn1_new_op)
        new2_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn2_new_op)
        new1_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn1_new_op)
        new2_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn2_new_op)
        new1_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn1_new_op)
        new2_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn2_new_op)


        new1_beta, new2_beta, new1_gamma, new2_gamma, new1_mean, new2_mean, new1_var, new2_var = \
            sess.run([new1_beta_read_var, new2_beta_read_var, new1_gamma_read_var, new2_gamma_read_var,
                     new1_mean_read_var, new2_mean_read_var, new1_var_read_var, new2_var_read_var])

        # Compare parameters of old and new batchnorms
        self.assertTrue(np.allclose(new1_beta, old1_beta))
        self.assertTrue(np.allclose(new2_beta, old2_beta))
        self.assertTrue(np.allclose(new1_gamma, old1_gamma))
        self.assertTrue(np.allclose(new2_gamma, old2_gamma))
        self.assertTrue(np.allclose(new1_mean, old1_mean))
        self.assertTrue(np.allclose(new2_mean, old2_mean))
        self.assertTrue(np.allclose(new1_var, old1_var))
        self.assertTrue(np.allclose(new2_var, old2_var))

        training_tf_placeholder = set()
        for bn in [bn1_new, bn2_new]:
            # Check if training flag of batchnorm is replaced to placeholder
            bn_training = BNUtils.get_training(bn.get_module())
            self.assertEqual(bn_training.op.type, 'PlaceholderWithDefault')
            training_tf_placeholder.add(bn_training)

            # Check if momentum is replaced to tf.Variable
            bn_momentum = BNUtils.get_momentum(bn.get_module())
            self.assertEqual(bn_momentum.type, 'VarHandleOp')

            # Check if assign moving avg exists
            assign_moving_avg_op = BNUtils.get_assign_moving_avg_op(bn.get_module())
            assert assign_moving_avg_op.type == 'AssignSubVariableOp'
            assign_moving_avg_1_op = BNUtils.get_assign_moving_avg_1_op(bn.get_module())
            assert assign_moving_avg_1_op.type == 'AssignSubVariableOp'

        # Make sure there is only one training tf placeholder
        self.assertEqual(len(training_tf_placeholder), 1)

        bn1_new_tensor = bn1_new.get_module().outputs[0]
        bn2_new_tensor = bn2_new.get_module().outputs[0]
        input_new_tensor = sess.graph.get_tensor_by_name('foo/input_1:0')
        relu_new_tensor = sess.graph.get_tensor_by_name('foo/Relu_1:0')

        # Set training placeholder to False
        feed_dict = {input_new_tensor: dummy_val, training_tf_placeholder.pop(): False}
        # Fetch outputs of new batchnorms
        bn1_new_output, bn2_new_output, relu_new_output = sess.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor],
                                                                   feed_dict=feed_dict)

        # Compare if outputs of old and new batchnorms are equivalent
        self.assertTrue(np.allclose(bn1_new_output, bn1_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(bn2_new_output, bn2_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(relu_new_output, relu_old_output, atol=1.e-4))


    def test_modify_sess_bn_mutable_with_v1_bn_model(self):
        v1_bn_model()
        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        input_op_names = ['input_1']
        output_op_names = ['Relu_1']
        dummy_val = np.random.randn(1, 32, 32, 3)

        conn_graph_old = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_old.get_all_ops()), 7)

        # Fetch conn graph op of batchnorm layers
        bn1_old = conn_graph_old.get_op_from_module_name('old1/FusedBatchNormV3')
        bn2_old = conn_graph_old.get_op_from_module_name('old2/FusedBatchNormV3')

        self.assertEqual(bn1_old.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_old.type, 'FusedBatchNormV3')

        # Fetch outputs of batchnorms
        bn1_old_tensor = bn1_old.get_module().outputs[0]
        bn2_old_tensor = bn2_old.get_module().outputs[0]
        input_old_tensor = sess.graph.get_tensor_by_name('input_1:0')
        relu_old_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
        bn1_old_output, bn2_old_output, relu_old_output = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                                                   feed_dict={input_old_tensor: dummy_val})

        # Modify batchnorm of sess
        modify_sess_bn_mutable(sess, input_op_names, output_op_names)

        conn_graph_new = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_new.get_all_ops()), 7)

        # Check if bn layer with modified_bn_old1 and modified_bn_old2 is created correctly
        bn1_new = conn_graph_new.get_op_from_module_name('old1_modified/cond/Identity')
        bn2_new = conn_graph_new.get_op_from_module_name('old2_modified/cond/Identity')

        self.assertEqual(bn1_new.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_new.type, 'FusedBatchNormV3')

        training_tf_placeholder = set()
        for bn in [bn1_new, bn2_new]:
            # Check if training flag of batchnorm is replaced to placeholder
            bn_training = BNUtils.get_training(bn.get_module())
            self.assertEqual(bn_training.op.type, 'PlaceholderWithDefault')
            training_tf_placeholder.add(bn_training)

            # Check if momentum is replaced to tf.Variable
            bn_momentum = BNUtils.get_momentum(bn.get_module())
            self.assertEqual(bn_momentum.type, 'VarHandleOp')

            # Check if assign moving avg exists
            assign_moving_avg_op = BNUtils.get_assign_moving_avg_op(bn.get_module())
            assert assign_moving_avg_op.type == 'AssignSubVariableOp'
            assign_moving_avg_1_op = BNUtils.get_assign_moving_avg_1_op(bn.get_module())
            assert assign_moving_avg_1_op.type == 'AssignSubVariableOp'

        # Make sure there is only one training tf placeholder
        self.assertEqual(len(training_tf_placeholder), 1)

        bn1_new_tensor = bn1_new.get_module().outputs[0]
        bn2_new_tensor = bn2_new.get_module().outputs[0]
        input_new_tensor = sess.graph.get_tensor_by_name('input_1:0')
        relu_new_tensor = sess.graph.get_tensor_by_name('Relu_1:0')

        # Set training placeholder to False
        feed_dict = {input_new_tensor: dummy_val, training_tf_placeholder.pop(): False}
        # Fetch outputs of new batchnorms
        bn1_new_output, bn2_new_output, relu_new_output = sess.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor],
                                                                   feed_dict=feed_dict)

        # Compare if outputs of old and new batchnorms are equivalent
        self.assertTrue(np.allclose(bn1_new_output, bn1_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(bn2_new_output, bn2_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(relu_new_output, relu_old_output, atol=1.e-4))

        # Compare parameters of old and new batchnorms
        for bn_new, bn_old in zip([bn1_new.get_module().name, bn2_new.get_module().name],
                                  [bn1_old.get_module().name, bn2_old.get_module().name]):
            bn_new_op = sess.graph.get_operation_by_name(bn_new)
            bn_old_op = sess.graph.get_operation_by_name(bn_old)

            new_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_new_op)
            old_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_old_op)
            new_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_new_op)
            old_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_old_op)
            new_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_new_op)
            old_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_old_op)
            new_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_new_op)
            old_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_old_op)

            new_beta, old_beta, new_gamma, old_gamma, new_mean, old_mean, new_var, old_var = \
                sess.run([new_beta_read_var, old_beta_read_var, new_gamma_read_var, old_gamma_read_var,
                         new_mean_read_var, old_mean_read_var, new_var_read_var, old_var_read_var])

            self.assertTrue(np.allclose(new_beta, old_beta))
            self.assertTrue(np.allclose(new_gamma, old_gamma))
            self.assertTrue(np.allclose(new_mean, old_mean))
            self.assertTrue(np.allclose(new_var, old_var))

    def test_modify_sess_bn_mutable_with_keras_bn_model(self):
        keras_bn_model()
        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        input_op_names = ['input_1']
        output_op_names = ['Relu_1']
        dummy_val = np.random.randn(1, 32, 32, 3)

        conn_graph_old = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_old.get_all_ops()), 7)

        # Fetch conn graph op of batchnorm layers
        bn1_old = conn_graph_old.get_op_from_module_name('old1/cond/Identity')
        bn2_old = conn_graph_old.get_op_from_module_name('old2/cond/Identity')

        self.assertEqual(bn1_old.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_old.type, 'FusedBatchNormV3')

        # Fetch outputs of batchnorms
        training_tensor = sess.graph.get_tensor_by_name('keras_learning_phase:0')
        bn1_old_tensor = bn1_old.get_module().outputs[0]
        bn2_old_tensor = bn2_old.get_module().outputs[0]
        input_old_tensor = sess.graph.get_tensor_by_name('input_1:0')
        relu_old_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
        bn1_old_output, bn2_old_output, relu_old_output = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                                                   feed_dict={input_old_tensor: dummy_val,
                                                                              training_tensor: False})

        # Modify batchnorm of sess
        modify_sess_bn_mutable(sess, input_op_names, output_op_names)

        conn_graph_new = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_new.get_all_ops()), 7)

        # Check if bn layer with modified_bn_old1 and modified_bn_old2 is created correctly
        bn1_new = conn_graph_new.get_op_from_module_name('old1_modified/cond/Identity')
        bn2_new = conn_graph_new.get_op_from_module_name('old2_modified/cond/Identity')

        self.assertEqual(bn1_new.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_new.type, 'FusedBatchNormV3')

        training_tf_placeholder = set()
        for bn in [bn1_new, bn2_new]:
            # Check if training flag of batchnorm is replaced to placeholder
            bn_training = BNUtils.get_training(bn.get_module())
            self.assertEqual(bn_training.op.type, 'PlaceholderWithDefault')
            training_tf_placeholder.add(bn_training)

            # Check if momentum is replaced to tf.Variable
            bn_momentum = BNUtils.get_momentum(bn.get_module())
            self.assertEqual(bn_momentum.type, 'VarHandleOp')

            # Check if assign moving avg exists
            assign_moving_avg_op = BNUtils.get_assign_moving_avg_op(bn.get_module())
            assert assign_moving_avg_op.type == 'AssignSubVariableOp'
            assign_moving_avg_1_op = BNUtils.get_assign_moving_avg_1_op(bn.get_module())
            assert assign_moving_avg_1_op.type == 'AssignSubVariableOp'

        # Make sure there is only one training tf placeholder
        self.assertEqual(len(training_tf_placeholder), 1)

        bn1_new_tensor = bn1_new.get_module().outputs[0]
        bn2_new_tensor = bn2_new.get_module().outputs[0]
        input_new_tensor = sess.graph.get_tensor_by_name('input_1:0')
        relu_new_tensor = sess.graph.get_tensor_by_name('Relu_1:0')

        # Set training placeholder to False
        feed_dict = {input_new_tensor: dummy_val, training_tf_placeholder.pop(): False}
        # Fetch outputs of new batchnorms
        bn1_new_output, bn2_new_output, relu_new_output = sess.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor],
                                                                   feed_dict=feed_dict)

        # Compare if outputs of old and new batchnorms are equivalent
        self.assertTrue(np.allclose(bn1_new_output, bn1_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(bn2_new_output, bn2_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(relu_new_output, relu_old_output, atol=1.e-4))

        # Compare parameters of old and new batchnorms
        for bn_new, bn_old in zip([bn1_new.get_module().name, bn2_new.get_module().name],
                                  [bn1_old.get_module().name, bn2_old.get_module().name]):
            bn_new_op = sess.graph.get_operation_by_name(bn_new)
            bn_old_op = sess.graph.get_operation_by_name(bn_old)

            new_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_new_op)
            old_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_old_op)
            new_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_new_op)
            old_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_old_op)
            new_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_new_op)
            old_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_old_op)
            new_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_new_op)
            old_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_old_op)

            new_beta, old_beta, new_gamma, old_gamma, new_mean, old_mean, new_var, old_var = \
                sess.run([new_beta_read_var, old_beta_read_var, new_gamma_read_var, old_gamma_read_var,
                          new_mean_read_var, old_mean_read_var, new_var_read_var, old_var_read_var])

            self.assertTrue(np.allclose(new_beta, old_beta))
            self.assertTrue(np.allclose(new_gamma, old_gamma))
            self.assertTrue(np.allclose(new_mean, old_mean))
            self.assertTrue(np.allclose(new_var, old_var))

    def test_modify_sess_bn_mutable_bn_training_is_tfvar_with_v1_bn_model(self):
        v1_bn_model()
        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        input_op_names = ['input_1']
        output_op_names = ['Relu_1']
        dummy_val = np.random.randn(1, 32, 32, 3)

        conn_graph_old = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_old.get_all_ops()), 7)

        # Fetch conn graph op of batchnorm layers
        bn1_old = conn_graph_old.get_op_from_module_name('old1/FusedBatchNormV3')
        bn2_old = conn_graph_old.get_op_from_module_name('old2/FusedBatchNormV3')

        self.assertEqual(bn1_old.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_old.type, 'FusedBatchNormV3')

        # Fetch outputs of batchnorms
        bn1_old_tensor = bn1_old.get_module().outputs[0]
        bn2_old_tensor = bn2_old.get_module().outputs[0]
        input_old_tensor = sess.graph.get_tensor_by_name('input_1:0')
        relu_old_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
        bn1_old_output, bn2_old_output, relu_old_output = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                                                   feed_dict={input_old_tensor: dummy_val})

        # Modify batchnorm of sess
        modify_sess_bn_mutable(sess, input_op_names, output_op_names, training_tf_placeholder=False)

        conn_graph_new = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_new.get_all_ops()), 7)

        # Check if bn layer with modified_bn_old1 and modified_bn_old2 is created correctly
        bn1_new = conn_graph_new.get_op_from_module_name('old1_modified/cond/Identity')
        bn2_new = conn_graph_new.get_op_from_module_name('old2_modified/cond/Identity')

        self.assertEqual(bn1_new.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_new.type, 'FusedBatchNormV3')

        training_tf_var = set()
        for bn in [bn1_new, bn2_new]:
            # Check if training flag of batchnorm is replaced to placeholder
            bn_training = BNUtils.get_training(bn.get_module())
            self.assertEqual(bn_training.op.type, 'ReadVariableOp')
            training_tf_var.add(bn_training.op.inputs[0])

            # Check if momentum is replaced to tf.Variable
            bn_momentum = BNUtils.get_momentum(bn.get_module())
            self.assertEqual(bn_momentum.type, 'VarHandleOp')

            # Check if assign moving avg exists
            assign_moving_avg_op = BNUtils.get_assign_moving_avg_op(bn.get_module())
            assert assign_moving_avg_op.type == 'AssignSubVariableOp'
            assign_moving_avg_1_op = BNUtils.get_assign_moving_avg_1_op(bn.get_module())
            assert assign_moving_avg_1_op.type == 'AssignSubVariableOp'

        # Make sure there is only one training tf var
        self.assertEqual(len(training_tf_var), 1)

        bn1_new_tensor = bn1_new.get_module().outputs[0]
        bn2_new_tensor = bn2_new.get_module().outputs[0]
        input_new_tensor = sess.graph.get_tensor_by_name('input_1:0')
        relu_new_tensor = sess.graph.get_tensor_by_name('Relu_1:0')

        feed_dict = {input_new_tensor: dummy_val}
        # Set training var to False
        training_tf_var = training_tf_var.pop()
        for var in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
            if var.name == training_tf_var.name:
                feed_dict[var] = False
        # Fetch outputs of new batchnorms
        bn1_new_output, bn2_new_output, relu_new_output = sess.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor],
                                                                   feed_dict=feed_dict)

        # Compare if outputs of old and new batchnorms are equivalent
        self.assertTrue(np.allclose(bn1_new_output, bn1_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(bn2_new_output, bn2_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(relu_new_output, relu_old_output, atol=1.e-4))

        # Compare parameters of old and new batchnorms
        for bn_new, bn_old in zip([bn1_new.get_module().name, bn2_new.get_module().name],
                                  [bn1_old.get_module().name, bn2_old.get_module().name]):
            bn_new_op = sess.graph.get_operation_by_name(bn_new)
            bn_old_op = sess.graph.get_operation_by_name(bn_old)

            new_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_new_op)
            old_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_old_op)
            new_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_new_op)
            old_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_old_op)
            new_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_new_op)
            old_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_old_op)
            new_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_new_op)
            old_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_old_op)

            new_beta, old_beta, new_gamma, old_gamma, new_mean, old_mean, new_var, old_var = \
                sess.run([new_beta_read_var, old_beta_read_var, new_gamma_read_var, old_gamma_read_var,
                          new_mean_read_var, old_mean_read_var, new_var_read_var, old_var_read_var])

            self.assertTrue(np.allclose(new_beta, old_beta))
            self.assertTrue(np.allclose(new_gamma, old_gamma))
            self.assertTrue(np.allclose(new_mean, old_mean))
            self.assertTrue(np.allclose(new_var, old_var))

    def test_modify_sess_bn_mutable_bn_training_is_tfvar_with_keras_bn_model(self):
        keras_bn_model()
        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        input_op_names = ['input_1']
        output_op_names = ['Relu_1']
        dummy_val = np.random.randn(1, 32, 32, 3)

        conn_graph_old = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_old.get_all_ops()), 7)

        # Fetch conn graph op of batchnorm layers
        bn1_old = conn_graph_old.get_op_from_module_name('old1/cond/Identity')
        bn2_old = conn_graph_old.get_op_from_module_name('old2/cond/Identity')

        self.assertEqual(bn1_old.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_old.type, 'FusedBatchNormV3')

        # Fetch outputs of batchnorms
        training_tensor = sess.graph.get_tensor_by_name('keras_learning_phase:0')
        bn1_old_tensor = bn1_old.get_module().outputs[0]
        bn2_old_tensor = bn2_old.get_module().outputs[0]
        input_old_tensor = sess.graph.get_tensor_by_name('input_1:0')
        relu_old_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
        bn1_old_output, bn2_old_output, relu_old_output = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                                                   feed_dict={input_old_tensor: dummy_val,
                                                                              training_tensor: False})

        # Modify batchnorm of sess
        modify_sess_bn_mutable(sess, input_op_names, output_op_names, training_tf_placeholder=False)

        conn_graph_new = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_new.get_all_ops()), 7)

        # Check if bn layer with modified_bn_old1 and modified_bn_old2 is created correctly
        bn1_new = conn_graph_new.get_op_from_module_name('old1_modified/cond/Identity')
        bn2_new = conn_graph_new.get_op_from_module_name('old2_modified/cond/Identity')

        self.assertEqual(bn1_new.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_new.type, 'FusedBatchNormV3')

        training_tf_var = set()
        for bn in [bn1_new, bn2_new]:
            # Check if training flag of batchnorm is replaced to placeholder
            bn_training = BNUtils.get_training(bn.get_module())
            self.assertEqual(bn_training.op.type, 'ReadVariableOp')
            training_tf_var.add(bn_training.op.inputs[0])

            # Check if momentum is replaced to tf.Variable
            bn_momentum = BNUtils.get_momentum(bn.get_module())
            self.assertEqual(bn_momentum.type, 'VarHandleOp')

            # Check if assign moving avg exists
            assign_moving_avg_op = BNUtils.get_assign_moving_avg_op(bn.get_module())
            assert assign_moving_avg_op.type == 'AssignSubVariableOp'
            assign_moving_avg_1_op = BNUtils.get_assign_moving_avg_1_op(bn.get_module())
            assert assign_moving_avg_1_op.type == 'AssignSubVariableOp'

        # Make sure there is only one training tf var
        self.assertEqual(len(training_tf_var), 1)

        bn1_new_tensor = bn1_new.get_module().outputs[0]
        bn2_new_tensor = bn2_new.get_module().outputs[0]
        input_new_tensor = sess.graph.get_tensor_by_name('input_1:0')
        relu_new_tensor = sess.graph.get_tensor_by_name('Relu_1:0')

        feed_dict = {input_new_tensor: dummy_val}
        # Set training var to False
        training_tf_var = training_tf_var.pop()
        for var in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
            if var.name == training_tf_var.name:
                feed_dict[var] = False
        # Fetch outputs of new batchnorms
        bn1_new_output, bn2_new_output, relu_new_output = sess.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor],
                                                                   feed_dict=feed_dict)

        # Compare if outputs of old and new batchnorms are equivalent
        self.assertTrue(np.allclose(bn1_new_output, bn1_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(bn2_new_output, bn2_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(relu_new_output, relu_old_output, atol=1.e-4))

        # Compare parameters of old and new batchnorms
        for bn_new, bn_old in zip([bn1_new.get_module().name, bn2_new.get_module().name],
                                  [bn1_old.get_module().name, bn2_old.get_module().name]):
            bn_new_op = sess.graph.get_operation_by_name(bn_new)
            bn_old_op = sess.graph.get_operation_by_name(bn_old)

            new_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_new_op)
            old_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_old_op)
            new_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_new_op)
            old_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_old_op)
            new_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_new_op)
            old_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_old_op)
            new_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_new_op)
            old_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_old_op)

            new_beta, old_beta, new_gamma, old_gamma, new_mean, old_mean, new_var, old_var = \
                sess.run([new_beta_read_var, old_beta_read_var, new_gamma_read_var, old_gamma_read_var,
                          new_mean_read_var, old_mean_read_var, new_var_read_var, old_var_read_var])

            self.assertTrue(np.allclose(new_beta, old_beta))
            self.assertTrue(np.allclose(new_gamma, old_gamma))
            self.assertTrue(np.allclose(new_mean, old_mean))
            self.assertTrue(np.allclose(new_var, old_var))

    def test_modify_sess_bn_mutable_with_v1_bn_model_with_is_training_as_placeholder(self):
        v1_bn_model(training_as_placeholder=True)
        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        input_op_names = ['input_1']
        output_op_names = ['Relu_1']
        dummy_val = np.random.randn(1, 32, 32, 3)

        conn_graph_old = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_old.get_all_ops()), 7)

        # Fetch conn graph op of batchnorm layers
        bn1_old = conn_graph_old.get_op_from_module_name('old1/cond/Identity')
        bn2_old = conn_graph_old.get_op_from_module_name('old2/cond/Identity')

        self.assertEqual(bn1_old.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_old.type, 'FusedBatchNormV3')

        # Fetch outputs of batchnorms
        bn1_old_tensor = bn1_old.get_module().outputs[0]
        bn2_old_tensor = bn2_old.get_module().outputs[0]
        input_old_tensor = sess.graph.get_tensor_by_name('input_1:0')
        relu_old_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
        bn1_old_output, bn2_old_output, relu_old_output = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                                                   feed_dict={input_old_tensor: dummy_val})

        # Modify batchnorm of sess
        modify_sess_bn_mutable(sess, input_op_names, output_op_names)

        conn_graph_new = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_new.get_all_ops()), 7)

        # Check if bn layer with modified_bn_old1 and modified_bn_old2 is created correctly
        bn1_new = conn_graph_new.get_op_from_module_name('old1_modified/cond/Identity')
        bn2_new = conn_graph_new.get_op_from_module_name('old2_modified/cond/Identity')

        self.assertEqual(bn1_new.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_new.type, 'FusedBatchNormV3')

        training_tf_placeholder = set()
        for bn in [bn1_new, bn2_new]:
            # Check if training flag of batchnorm is replaced to placeholder
            bn_training = BNUtils.get_training(bn.get_module())
            self.assertEqual(bn_training.op.type, 'PlaceholderWithDefault')
            training_tf_placeholder.add(bn_training)

            # Check if momentum is replaced to tf.Variable
            bn_momentum = BNUtils.get_momentum(bn.get_module())
            self.assertEqual(bn_momentum.type, 'VarHandleOp')

            # Check if assign moving avg exists
            assign_moving_avg_op = BNUtils.get_assign_moving_avg_op(bn.get_module())
            assert assign_moving_avg_op.type == 'AssignSubVariableOp'
            assign_moving_avg_1_op = BNUtils.get_assign_moving_avg_1_op(bn.get_module())
            assert assign_moving_avg_1_op.type == 'AssignSubVariableOp'

        # Make sure there is only one training tf placeholder
        self.assertEqual(len(training_tf_placeholder), 1)

        bn1_new_tensor = bn1_new.get_module().outputs[0]
        bn2_new_tensor = bn2_new.get_module().outputs[0]
        input_new_tensor = sess.graph.get_tensor_by_name('input_1:0')
        relu_new_tensor = sess.graph.get_tensor_by_name('Relu_1:0')

        # Set training placeholder to False
        feed_dict = {input_new_tensor: dummy_val, training_tf_placeholder.pop(): False}
        # Fetch outputs of new batchnorms
        bn1_new_output, bn2_new_output, relu_new_output = sess.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor], feed_dict=feed_dict)

        # Compare if outputs of old and new batchnorms are equivalent
        self.assertTrue(np.allclose(bn1_new_output, bn1_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(bn2_new_output, bn2_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(relu_new_output, relu_old_output, atol=1.e-4))

        # Compare parameters of old and new batchnorms
        for bn_new, bn_old in zip([bn1_new.get_module().name, bn2_new.get_module().name],
                                  [bn1_old.get_module().name, bn2_old.get_module().name]):
            bn_new_op = sess.graph.get_operation_by_name(bn_new)
            bn_old_op = sess.graph.get_operation_by_name(bn_old)

            new_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_new_op)
            old_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_old_op)
            new_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_new_op)
            old_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_old_op)
            new_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_new_op)
            old_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_old_op)
            new_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_new_op)
            old_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_old_op)

            new_beta, old_beta, new_gamma, old_gamma, new_mean, old_mean, new_var, old_var = \
                sess.run([new_beta_read_var, old_beta_read_var, new_gamma_read_var, old_gamma_read_var,
                          new_mean_read_var, old_mean_read_var, new_var_read_var, old_var_read_var])

            self.assertTrue(np.allclose(new_beta, old_beta))
            self.assertTrue(np.allclose(new_gamma, old_gamma))
            self.assertTrue(np.allclose(new_mean, old_mean))
            self.assertTrue(np.allclose(new_var, old_var))

    def test_modify_sess_bn_mutable_with_keras_model_with_is_training_as_placeholder(self):
        keras_bn_model(training_as_placeholder=True)
        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        input_op_names = ['input_1']
        output_op_names = ['Relu_1']
        dummy_val = np.random.randn(1, 32, 32, 3)

        conn_graph_old = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_old.get_all_ops()), 7)

        # Fetch conn graph op of batchnorm layers
        bn1_old = conn_graph_old.get_op_from_module_name('old1/cond/Identity')
        bn2_old = conn_graph_old.get_op_from_module_name('old2/cond/Identity')

        self.assertEqual(bn1_old.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_old.type, 'FusedBatchNormV3')

        # Fetch outputs of batchnorms
        bn1_old_tensor = bn1_old.get_module().outputs[0]
        bn2_old_tensor = bn2_old.get_module().outputs[0]
        input_old_tensor = sess.graph.get_tensor_by_name('input_1:0')
        relu_old_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
        bn1_old_output, bn2_old_output, relu_old_output = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                                                   feed_dict={input_old_tensor: dummy_val})

        # Modify batchnorm of sess
        modify_sess_bn_mutable(sess, input_op_names, output_op_names)

        conn_graph_new = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self.assertEqual(len(conn_graph_new.get_all_ops()), 7)

        # Check if bn layer with modified_bn_old1 and modified_bn_old2 is created correctly
        bn1_new = conn_graph_new.get_op_from_module_name('old1_modified/cond/Identity')
        bn2_new = conn_graph_new.get_op_from_module_name('old2_modified/cond/Identity')

        self.assertEqual(bn1_new.type, 'FusedBatchNormV3')
        self.assertEqual(bn2_new.type, 'FusedBatchNormV3')

        training_tf_placeholder = set()
        for bn in [bn1_new, bn2_new]:
            # Check if training flag of batchnorm is replaced to placeholder
            bn_training = BNUtils.get_training(bn.get_module())
            self.assertEqual(bn_training.op.type, 'PlaceholderWithDefault')
            training_tf_placeholder.add(bn_training)

            # Check if momentum is replaced to tf.Variable
            bn_momentum = BNUtils.get_momentum(bn.get_module())
            self.assertEqual(bn_momentum.type, 'VarHandleOp')

            # Check if assign moving avg exists
            assign_moving_avg_op = BNUtils.get_assign_moving_avg_op(bn.get_module())
            assert assign_moving_avg_op.type == 'AssignSubVariableOp'
            assign_moving_avg_1_op = BNUtils.get_assign_moving_avg_1_op(bn.get_module())
            assert assign_moving_avg_1_op.type == 'AssignSubVariableOp'

        # Make sure there is only one training tf placeholder
        self.assertEqual(len(training_tf_placeholder), 1)

        bn1_new_tensor = bn1_new.get_module().outputs[0]
        bn2_new_tensor = bn2_new.get_module().outputs[0]
        input_new_tensor = sess.graph.get_tensor_by_name('input_1:0')
        relu_new_tensor = sess.graph.get_tensor_by_name('Relu_1:0')

        # Set training placeholder to False
        feed_dict = {input_new_tensor: dummy_val, training_tf_placeholder.pop(): False}
        # Fetch outputs of new batchnorms
        bn1_new_output, bn2_new_output, relu_new_output = sess.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor],
                                                                   feed_dict=feed_dict)

        # Compare if outputs of old and new batchnorms are equivalent
        self.assertTrue(np.allclose(bn1_new_output, bn1_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(bn2_new_output, bn2_old_output, atol=1.e-4))
        self.assertTrue(np.allclose(relu_new_output, relu_old_output, atol=1.e-4))

        # Compare parameters of old and new batchnorms
        for bn_new, bn_old in zip([bn1_new.get_module().name, bn2_new.get_module().name],
                                  [bn1_old.get_module().name, bn2_old.get_module().name]):
            bn_new_op = sess.graph.get_operation_by_name(bn_new)
            bn_old_op = sess.graph.get_operation_by_name(bn_old)

            new_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_new_op)
            old_beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_old_op)
            new_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_new_op)
            old_gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_old_op)
            new_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_new_op)
            old_mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_old_op)
            new_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_new_op)
            old_var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_old_op)

            new_beta, old_beta, new_gamma, old_gamma, new_mean, old_mean, new_var, old_var = \
                sess.run([new_beta_read_var, old_beta_read_var, new_gamma_read_var, old_gamma_read_var,
                          new_mean_read_var, old_mean_read_var, new_var_read_var, old_var_read_var])

            self.assertTrue(np.allclose(new_beta, old_beta))
            self.assertTrue(np.allclose(new_gamma, old_gamma))
            self.assertTrue(np.allclose(new_mean, old_mean))
            self.assertTrue(np.allclose(new_var, old_var))

    def test_modify_model_bn_mutable_with_keras_bn_model(self):
        model = keras_bn_model()

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                assert isinstance(layer.momentum, float)

        modify_model_bn_mutable(model)
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                assert not isinstance(layer.momentum, float)
                assert layer.momentum.op.type == 'VarHandleOp'

        bn_momentum_tf_var_list_tf2 = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                bn_momentum_tf_var_list_tf2.append(layer.momentum)

        graph = bn_momentum_tf_var_list_tf2[0].op.graph
        init = tf.compat.v1.global_variables_initializer()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(graph=graph, config=config)
        initialize_uninitialized_vars(sess)
        with graph.as_default():
            tf_global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        # Sanity check all items of bn_momentum_tf_var_list_tf2 in tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
        assert all(bn_momentum_tf_var in tf_global_vars for bn_momentum_tf_var in bn_momentum_tf_var_list_tf2)
        training_tensor = sess.graph.get_tensor_by_name('keras_learning_phase:0')
        assert training_tensor is not None
        with sess.graph.as_default():
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        assert update_ops == []
        return

    def test_modify_model_bn_mutable_keras_mobilenetv2_model(self):
        model = keras_applications_mobilenet_v2()

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                assert isinstance(layer.momentum, float)

        modify_model_bn_mutable(model)
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                assert not isinstance(layer.momentum, float)
                assert layer.momentum.op.type == 'VarHandleOp'

        bn_momentum_tf_var_list_tf2 = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                bn_momentum_tf_var_list_tf2.append(layer.momentum)

        graph = bn_momentum_tf_var_list_tf2[0].op.graph
        init = tf.compat.v1.global_variables_initializer()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(graph=graph, config=config)
        initialize_uninitialized_vars(sess)
        with graph.as_default():
            tf_global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        # Sanity check all items of bn_momentum_tf_var_list_tf2 in tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
        assert all(bn_momentum_tf_var in tf_global_vars for bn_momentum_tf_var in bn_momentum_tf_var_list_tf2)

        training_tensor = sess.graph.get_tensor_by_name('keras_learning_phase:0')
        assert training_tensor is not None
        with sess.graph.as_default():
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        assert update_ops == []
        return
