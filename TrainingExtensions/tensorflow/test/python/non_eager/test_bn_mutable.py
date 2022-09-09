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
import pytest
import logging
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import logging
from typing import List, Callable, Any
import json
import numpy as np
import tensorflow as tf

from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_common.utils import AimetLogger
from aimet_tensorflow.utils.op.bn_mutable import modify_model_bn_mutable, modify_sess_bn_mutable
from aimet_tensorflow.batch_norm_fold import find_all_batch_norms_to_fold
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
AimetLogger.set_level_for_all_areas(logging.DEBUG)

tf.compat.v1.disable_eager_execution()
np.random.seed(0)
tf.compat.v1.set_random_seed(0)


def sessions_tf1_pre_trained_model(device):
    with tf.device(device):
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn_op = tf.compat.v1.layers.batch_normalization(conv_op, name="old1/",
                                                        beta_initializer=tf.compat.v1.random_uniform_initializer(),
                                                        gamma_initializer=tf.compat.v1.random_uniform_initializer(),
                                                        moving_mean_initializer=tf.compat.v1.random_uniform_initializer(),
                                                        moving_variance_initializer=tf.compat.v1.random_uniform_initializer(),
                                                        fused=True)
        relu0 = tf.nn.relu(bn_op)

        conv_op1 = tf.keras.layers.Conv2D(32, (3, 3))(relu0)

        bn_op1 = tf.compat.v1.layers.batch_normalization(conv_op1, name="old2/",
                                                         beta_initializer=tf.compat.v1.random_uniform_initializer(),
                                                         gamma_initializer=tf.compat.v1.random_uniform_initializer(),
                                                         moving_mean_initializer=tf.compat.v1.random_uniform_initializer(),
                                                         moving_variance_initializer=tf.compat.v1.random_uniform_initializer(),
                                                         fused=True)

        relu1 = tf.nn.relu(bn_op1)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        return sess


def sessions_tf1_pre_trained_model_with_is_traing_as_placeholder(device):
    with tf.device(device):
        tf.compat.v1.reset_default_graph()
        bn_training = tf.compat.v1.placeholder_with_default(False, shape=[], name='bn_is_training_placehoder')
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn_op = tf.compat.v1.layers.batch_normalization(conv_op, name="old1/", training=bn_training,
                                                        beta_initializer=tf.compat.v1.random_uniform_initializer(),
                                                        gamma_initializer=tf.compat.v1.random_uniform_initializer(),
                                                        moving_mean_initializer=tf.compat.v1.random_uniform_initializer(),
                                                        moving_variance_initializer=tf.compat.v1.random_uniform_initializer(),
                                                        fused=True)
        relu0 = tf.nn.relu(bn_op)

        conv_op1 = tf.keras.layers.Conv2D(32, (3, 3))(relu0)

        bn_op1 = tf.compat.v1.layers.batch_normalization(conv_op1, name="old2/", training=bn_training,
                                                         beta_initializer=tf.compat.v1.random_uniform_initializer(),
                                                         gamma_initializer=tf.compat.v1.random_uniform_initializer(),
                                                         moving_mean_initializer=tf.compat.v1.random_uniform_initializer(),
                                                         moving_variance_initializer=tf.compat.v1.random_uniform_initializer(),
                                                         fused=True)

        relu1 = tf.nn.relu(bn_op1)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, 'tf1_placeholder')


@pytest.fixture
def gpu_sessions_tf1_pre_trained_model_with_is_traing_as_placehoder():
    return sessions_tf1_pre_trained_model_with_is_traing_as_placeholder('/gpu:0')


@pytest.fixture
def gpu_sessions_tf1_pre_trained_model():
    return sessions_tf1_pre_trained_model('/gpu:0')


def sessions_pretrained_tf2_model(device):
    with tf.device(device):
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn_op = tf.keras.layers.BatchNormalization(name="old1/",
                                                   beta_initializer=tf.compat.v1.random_uniform_initializer(),
                                                   gamma_initializer=tf.compat.v1.random_uniform_initializer(),
                                                   moving_mean_initializer=tf.compat.v1.random_uniform_initializer(),
                                                   moving_variance_initializer=tf.compat.v1.random_uniform_initializer(),
                                                   fused=True)(
            conv_op)  ## , fused=True  _fused=False get_all_tf_bn_vars_list_without_momentum   ValueError: not enough
        # values to unpack (expected 2, got 0)
        relu = tf.nn.relu(bn_op)
        conv_op1 = tf.keras.layers.Conv2D(32, (3, 3))(relu)
        bn_op1 = tf.keras.layers.BatchNormalization(name="old2/",
                                                    beta_initializer=tf.compat.v1.random_uniform_initializer(),
                                                    gamma_initializer=tf.compat.v1.random_uniform_initializer(),
                                                    moving_mean_initializer=tf.compat.v1.random_uniform_initializer(),
                                                    moving_variance_initializer=tf.compat.v1.random_uniform_initializer(),
                                                    fused=True)(conv_op1)
        relu1 = tf.nn.relu(bn_op1)
        model = tf.keras.Model(inputs=inputs, outputs=relu1)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model


@pytest.fixture
def cpu_sessions_pretrained_tf2_model():
    return sessions_pretrained_tf2_model('/cpu:0')


@pytest.fixture
def gpu_sessions_pretrained_tf2_model():
    return sessions_pretrained_tf2_model('/gpu:0')


def sessions_tf_keras_applications_mobilenet_v2(device):
    with tf.device(device):
        tf.compat.v1.reset_default_graph()
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=None, input_shape=(32, 32, 3))
        return model


@pytest.fixture
def gpu_sessions_tf_keras_applications_mobilenet_v2():
    return sessions_tf_keras_applications_mobilenet_v2('/gpu:0')


def sessions_tf2_pre_trained_model_with_is_traing_as_placehoder(device):
    with tf.device(device):
        tf.compat.v1.reset_default_graph()
        bn_training = tf.compat.v1.placeholder_with_default(False, shape=[], name='bn_is_training_placehoder')
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn_op = tf.keras.layers.BatchNormalization(name="old1/", momentum=tf.Variable(0.9, name="momentum_mutable1"),
                                                   fused=True)(conv_op, training=bn_training)
        relu = tf.nn.relu(bn_op)
        conv_op1 = tf.keras.layers.Conv2D(32, (3, 3))(relu)
        bn_op1 = tf.keras.layers.BatchNormalization(name="old2/", momentum=tf.Variable(0.9, name="momentum_mutable2"),
                                                    fused=True)(conv_op1, training=bn_training)
        relu1 = tf.nn.relu(bn_op1)
        model = tf.keras.Model(inputs=inputs, outputs=relu1)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, 'tf2_placeholder')


@pytest.fixture
def gpu_sessions_tf2_pre_trained_model_with_is_traing_as_placehoder():
    return sessions_tf2_pre_trained_model_with_is_traing_as_placehoder('/gpu:0')


class TestBnMutable:

    @pytest.mark.cuda
    def test_modify_sess_bn_mutable_with_tf1_pre_trained_model_gpu(self, gpu_sessions_tf1_pre_trained_model):
        tf.compat.v1.reset_default_graph()
        sess = gpu_sessions_tf1_pre_trained_model
        input_op_names = ["input_1"]
        output_op_names = ['Relu_1']
        dummy_val = np.random.randn(1, 32, 32, 3)

        model_input_old = sess.graph.get_tensor_by_name('input_1:0')
        bn1_old_tensor = sess.graph.get_tensor_by_name("old1/FusedBatchNormV3:0")
        bn2_old_tensor = sess.graph.get_tensor_by_name('old2/FusedBatchNormV3:0')
        relu_old_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
        bn1_old, bn2_old, relu_old = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                              feed_dict={model_input_old: dummy_val})  # training_tensor: False})

        modify_sess_bn_mutable(sess, input_op_names,
                               output_op_names)
        sess_new = sess
        conn_graph_new = ConnectedGraph(sess_new.graph, input_op_names, output_op_names)
        assert len(conn_graph_new._ops) == 7
        ops_name_list_new = [op for op in conn_graph_new._ops]
        # Check modify_sess_bn_mutable results in ops_list
        assert "modified_bn_old1/FusedBatchNormV3" in ops_name_list_new
        assert "modified_bn_old1/FusedBatchNormV3" in ops_name_list_new
        bn_conv_linear_pairs_new = find_all_batch_norms_to_fold(sess_new, input_op_names, output_op_names)
        _, bn1_new, _ = bn_conv_linear_pairs_new[0]
        _, bn2_new, _ = bn_conv_linear_pairs_new[1]
        # Check modify_sess_bn_mutable results in Bn pairs
        assert "modified_bn_old1/FusedBatchNormV3/cond/Identity" in bn1_new.op.name
        assert "modified_bn_old2/FusedBatchNormV3/cond/Identity" in bn2_new.op.name

        with sess_new.graph.as_default():
            is_training_tf_placehodler_list = []
            for _, bn, _ in bn_conv_linear_pairs_new:
                assert bn.op.type in ['Identity']
                bn_cond1_tf_op = sess_new.graph.get_operation_by_name("/".join(bn.op.name.split("/")[0:-2]) + "/cond_1")
                bn_training = bn_cond1_tf_op.outputs[0].op.inputs[0]
                # Check bn_training is tf.compat.v1.PlaceholderWithDefault
                assert bn_training.op.type in ['PlaceholderWithDefault']
                is_training_tf_placehodler_list.append(bn_training)

        bn_momentum_tf_var_list = []
        with sess_new.graph.as_default():
            tf_global_vars_new = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
            for var in tf_global_vars_new:
                if "momentum_mutable_modified_bn_" in var.name:
                    bn_momentum_tf_var_list.append(var)
            # Check modify_sess_bn_mutable results(bn_momentum) in tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
            assert len(bn_momentum_tf_var_list) == 2

        model_input_new = sess_new.graph.get_tensor_by_name('input_1:0')
        bn1_new_tensor = sess_new.graph.get_tensor_by_name("modified_bn_old1/FusedBatchNormV3/cond/Identity:0")
        bn2_new_tensor = sess_new.graph.get_tensor_by_name('modified_bn_old2/FusedBatchNormV3/cond/Identity:0')
        relu_new_tensor = sess_new.graph.get_tensor_by_name('Relu_1:0')

        feed_dict = {model_input_new: dummy_val}
        for training_tensor in is_training_tf_placehodler_list:
            feed_dict[training_tensor] = False
        bn1_new, bn2_new, relu_new = sess_new.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor],
                                                  feed_dict=feed_dict)  # training_tensor: False})

        # Compare “original BN and modified BN “ are equivalent
        assert np.allclose(bn1_new, bn1_old, rtol=1.e-1)
        assert np.allclose(bn2_new, bn2_old, rtol=1.e-1)

        # check update_ops
        with sess_new.graph.as_default():
            for _, bn, _ in bn_conv_linear_pairs_new:
                bn_mean_tf_var_name = bn.op.inputs[0].op.inputs[3].name
                bn_var_tf_var_name = bn.op.inputs[0].op.inputs[4].name
                bn_cond1_tf_op = sess_new.graph.get_operation_by_name(
                    "/".join(bn.op.name.split("/")[0:-2]) + "/cond_1")

                bn_mean_update = \
                    bn_cond1_tf_op.outputs[0].consumers()[0].outputs[0].consumers()[0].outputs[0].consumers()[
                        0].outputs[
                        0].consumers()[0]
                bn_var_update = \
                    bn_cond1_tf_op.outputs[0].consumers()[0].outputs[0].consumers()[1].outputs[0].consumers()[
                        0].outputs[
                        0].consumers()[0]
                assert bn_mean_update.type in ['AssignSubVariableOp'] and bn_var_update.type in ['AssignSubVariableOp']


    @pytest.mark.cuda
    def test_modify_sess_bn_mutable_with_pretrained_tf2_model_gpu(self, gpu_sessions_pretrained_tf2_model):
        tf.compat.v1.reset_default_graph()
        model = gpu_sessions_pretrained_tf2_model
        graph = model.inputs[0].graph
        with graph.as_default():
            init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph=graph)
        initialize_uninitialized_vars(sess)

        input_op_names = ["input_1"]
        output_op_names = ['Relu_1']
        dummy_val = np.random.randn(1, 32, 32, 3)
        training_tensor = sess.graph.get_tensor_by_name('keras_learning_phase:0')
        model_input_old = sess.graph.get_tensor_by_name('input_1:0')
        bn1_old_tensor = sess.graph.get_tensor_by_name("old1/cond/Identity:0")
        bn2_old_tensor = sess.graph.get_tensor_by_name('old2/cond/Identity:0')
        relu_old_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
        bn1_old, bn2_old, relu_old = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                              feed_dict={model_input_old: dummy_val,
                                                         training_tensor: False})  # training_tensor: False})

        modify_sess_bn_mutable(sess, input_op_names,
                               output_op_names)
        sess_new = sess
        conn_graph_new = ConnectedGraph(sess_new.graph, input_op_names, output_op_names)
        assert len(conn_graph_new._ops) == 7
        ops_name_list_new = [op for op in conn_graph_new._ops]
        # Check modify_sess_bn_mutable results in ops_list
        assert "modified_bn_old1/cond/Identity" in ops_name_list_new
        assert "modified_bn_old2/cond/Identity" in ops_name_list_new
        bn_conv_linear_pairs_new = find_all_batch_norms_to_fold(sess_new, input_op_names, output_op_names)
        _, bn1_new, _ = bn_conv_linear_pairs_new[0]
        _, bn2_new, _ = bn_conv_linear_pairs_new[1]
        # Check modify_sess_bn_mutable results in Bn pairs
        assert "modified_bn_old1/cond/Identity/cond/Identity" == bn1_new.op.name
        assert "modified_bn_old2/cond/Identity/cond/Identity" == bn2_new.op.name

        with sess_new.graph.as_default():
            is_training_tf_placehodler_list = []
            for _, bn, _ in bn_conv_linear_pairs_new:
                assert bn.op.type in ['Identity']
                bn_cond1_tf_op = sess_new.graph.get_operation_by_name("/".join(bn.op.name.split("/")[0:-2]) + "/cond_1")
                bn_training = bn_cond1_tf_op.outputs[0].op.inputs[0]
                # Check bn_training is tf.compat.v1.PlaceholderWithDefault
                assert bn_training.op.type in ['PlaceholderWithDefault']
                is_training_tf_placehodler_list.append(bn_training)

        bn_momentum_tf_var_list = []
        with sess_new.graph.as_default():
            tf_global_vars_new = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
            for var in tf_global_vars_new:
                if "momentum_mutable_modified_bn_" in var.name:
                    bn_momentum_tf_var_list.append(var)
            # Check modify_sess_bn_mutable results(bn_momentum) in tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
            assert len(bn_momentum_tf_var_list) == 2

        model_input_new = sess_new.graph.get_tensor_by_name('input_1:0')
        training_tensor = sess_new.graph.get_tensor_by_name('keras_learning_phase:0')
        bn1_new_tensor = sess_new.graph.get_tensor_by_name("modified_bn_old1/cond/Identity/cond/Identity:0")
        bn2_new_tensor = sess_new.graph.get_tensor_by_name('modified_bn_old2/cond/Identity/cond/Identity:0')
        relu_new_tensor = sess_new.graph.get_tensor_by_name('Relu_1:0')

        feed_dict = {model_input_new: dummy_val, training_tensor: False}
        for training_tensor in is_training_tf_placehodler_list:
            feed_dict[training_tensor] = False
        bn1_new, bn2_new, relu_new = sess_new.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor],
                                                  feed_dict=feed_dict)  # training_tensor: False})

        # Compare “original BN and modified BN “ are equivalent
        assert np.allclose(bn1_new, bn1_old, rtol=1.e-4)
        assert np.allclose(bn2_new, bn2_old, rtol=1.e-4)

        # check update_ops
        with sess_new.graph.as_default():
            for _, bn, _ in bn_conv_linear_pairs_new:
                bn_mean_tf_var_name = bn.op.inputs[0].op.inputs[3].name
                bn_var_tf_var_name = bn.op.inputs[0].op.inputs[4].name
                bn_cond1_tf_op = sess_new.graph.get_operation_by_name(
                    "/".join(bn.op.name.split("/")[0:-2]) + "/cond_1")

                bn_mean_update = \
                    bn_cond1_tf_op.outputs[0].consumers()[0].outputs[0].consumers()[0].outputs[0].consumers()[
                        0].outputs[
                        0].consumers()[0]
                bn_var_update = \
                    bn_cond1_tf_op.outputs[0].consumers()[0].outputs[0].consumers()[1].outputs[0].consumers()[
                        0].outputs[
                        0].consumers()[0]
                assert bn_mean_update.type in ['AssignSubVariableOp'] and bn_var_update.type in ['AssignSubVariableOp']

    @pytest.mark.cuda
    def test_modify_sess_bn_mutable_bn_training_is_tfvar_with_tf1_pre_trained_model_gpu(self,
                                                                                        gpu_sessions_tf1_pre_trained_model):
        tf.compat.v1.reset_default_graph()
        sess = gpu_sessions_tf1_pre_trained_model
        input_op_names = ["input_1"]
        output_op_names = ['Relu_1']
        dummy_val = np.random.randn(1, 32, 32, 3)

        model_input_old = sess.graph.get_tensor_by_name('input_1:0')
        bn1_old_tensor = sess.graph.get_tensor_by_name("old1/FusedBatchNormV3:0")
        bn2_old_tensor = sess.graph.get_tensor_by_name('old2/FusedBatchNormV3:0')
        relu_old_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
        bn1_old, bn2_old, relu_old = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                              feed_dict={model_input_old: dummy_val})  # training_tensor: False})

        modify_sess_bn_mutable(sess, input_op_names,
                               output_op_names, trainin_is_tf_placeholder=False)
        sess_new = sess
        conn_graph_new = ConnectedGraph(sess_new.graph, input_op_names, output_op_names)
        assert len(conn_graph_new._ops) == 7
        ops_name_list_new = [op for op in conn_graph_new._ops]
        # Check modify_sess_bn_mutable results in ops_list
        assert "modified_bn_old1/FusedBatchNormV3" in ops_name_list_new
        assert "modified_bn_old1/FusedBatchNormV3" in ops_name_list_new
        bn_conv_linear_pairs_new = find_all_batch_norms_to_fold(sess_new, input_op_names, output_op_names)
        _, bn1_new, _ = bn_conv_linear_pairs_new[0]
        _, bn2_new, _ = bn_conv_linear_pairs_new[1]
        # Check modify_sess_bn_mutable results in Bn pairs
        assert "modified_bn_old1/FusedBatchNormV3/cond/Identity" in bn1_new.op.name
        assert "modified_bn_old2/FusedBatchNormV3/cond/Identity" in bn2_new.op.name

        with sess_new.graph.as_default():
            is_training_tf_var_list = []
            for _, bn, _ in bn_conv_linear_pairs_new:
                assert bn.op.type in ['Identity']
                bn_cond1_tf_op = sess_new.graph.get_operation_by_name("/".join(bn.op.name.split("/")[0:-2]) + "/cond_1")
                bn_training = bn_cond1_tf_op.outputs[0].op.inputs[0]
                # Check bn_training is tf.compat.v1.Variable
                assert bn_training.op.type in['ReadVariableOp']
                is_training_tf_var_list.append(bn_training)

        bn_momentum_tf_var_list = []
        with sess_new.graph.as_default():
            tf_global_vars_new = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
            for var in tf_global_vars_new:
                if "momentum_mutable_modified_bn_" in var.name:
                    bn_momentum_tf_var_list.append(var)
            # Check modify_sess_bn_mutable results(bn_momentum) in tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
            assert len(bn_momentum_tf_var_list) == 2

        model_input_new = sess_new.graph.get_tensor_by_name('input_1:0')
        bn1_new_tensor = sess_new.graph.get_tensor_by_name("modified_bn_old1/FusedBatchNormV3/cond/Identity:0")
        bn2_new_tensor = sess_new.graph.get_tensor_by_name('modified_bn_old2/FusedBatchNormV3/cond/Identity:0')
        relu_new_tensor = sess_new.graph.get_tensor_by_name('Relu_1:0')

        feed_dict = {model_input_new: dummy_val}
        for training_tensor in is_training_tf_var_list:
            feed_dict[training_tensor] = False
        bn1_new, bn2_new, relu_new = sess_new.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor],
                                                  feed_dict=feed_dict)  # training_tensor: False})

        # Compare “original BN and modified BN “ are equivalent
        assert np.allclose(bn1_new, bn1_old, rtol=1.e-1)
        assert np.allclose(bn2_new, bn2_old, rtol=1.e-1)

        # check update_ops
        with sess_new.graph.as_default():
            for _, bn, _ in bn_conv_linear_pairs_new:
                bn_mean_tf_var_name = bn.op.inputs[0].op.inputs[3].name
                bn_var_tf_var_name = bn.op.inputs[0].op.inputs[4].name
                bn_cond1_tf_op = sess_new.graph.get_operation_by_name(
                    "/".join(bn.op.name.split("/")[0:-2]) + "/cond_1")

                bn_mean_update = \
                    bn_cond1_tf_op.outputs[0].consumers()[0].outputs[0].consumers()[0].outputs[0].consumers()[
                        0].outputs[
                        0].consumers()[0]
                bn_var_update = \
                    bn_cond1_tf_op.outputs[0].consumers()[0].outputs[0].consumers()[1].outputs[0].consumers()[
                        0].outputs[
                        0].consumers()[0]
                assert bn_mean_update.type in ['AssignSubVariableOp'] and bn_var_update.type in ['AssignSubVariableOp']

    @pytest.mark.cuda
    def test_modify_sess_bn_mutable_bn_training_is_tfvar_with_pretrained_tf2_model_gpu(self, gpu_sessions_pretrained_tf2_model):
        tf.compat.v1.reset_default_graph()
        model = gpu_sessions_pretrained_tf2_model
        graph = model.inputs[0].graph
        with graph.as_default():
            init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph=graph)
        initialize_uninitialized_vars(sess)

        input_op_names = ["input_1"]
        output_op_names = ['Relu_1']
        dummy_val = np.random.randn(1, 32, 32, 3)
        training_tensor = sess.graph.get_tensor_by_name('keras_learning_phase:0')
        model_input_old = sess.graph.get_tensor_by_name('input_1:0')
        bn1_old_tensor = sess.graph.get_tensor_by_name("old1/cond/Identity:0")
        bn2_old_tensor = sess.graph.get_tensor_by_name('old2/cond/Identity:0')
        relu_old_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
        bn1_old, bn2_old, relu_old = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                              feed_dict={model_input_old: dummy_val,
                                                         training_tensor: False})  # training_tensor: False})

        modify_sess_bn_mutable(sess, input_op_names,
                               output_op_names, trainin_is_tf_placeholder=False)
        sess_new = sess
        conn_graph_new = ConnectedGraph(sess_new.graph, input_op_names, output_op_names)
        assert len(conn_graph_new._ops) == 7
        ops_name_list_new = [op for op in conn_graph_new._ops]
        # Check modify_sess_bn_mutable results in ops_list
        assert "modified_bn_old1/cond/Identity" in ops_name_list_new
        assert "modified_bn_old2/cond/Identity" in ops_name_list_new
        bn_conv_linear_pairs_new = find_all_batch_norms_to_fold(sess_new, input_op_names, output_op_names)
        _, bn1_new, _ = bn_conv_linear_pairs_new[0]
        _, bn2_new, _ = bn_conv_linear_pairs_new[1]
        # Check modify_sess_bn_mutable results in Bn pairs
        assert "modified_bn_old1/cond/Identity/cond/Identity" == bn1_new.op.name
        assert "modified_bn_old2/cond/Identity/cond/Identity" == bn2_new.op.name

        with sess_new.graph.as_default():
            is_training_tf_var_list = []
            for _, bn, _ in bn_conv_linear_pairs_new:
                assert bn.op.type in ['Identity']
                bn_cond1_tf_op = sess_new.graph.get_operation_by_name("/".join(bn.op.name.split("/")[0:-2]) + "/cond_1")
                bn_training = bn_cond1_tf_op.outputs[0].op.inputs[0]
                # Check bn_training is tf.compat.v1.Variable
                assert bn_training.op.type in ['ReadVariableOp']
                is_training_tf_var_list.append(bn_training)

        bn_momentum_tf_var_list = []
        with sess_new.graph.as_default():
            tf_global_vars_new = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
            for var in tf_global_vars_new:
                if "momentum_mutable_modified_bn_" in var.name:
                    bn_momentum_tf_var_list.append(var)
            # Check modify_sess_bn_mutable results(bn_momentum) in tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
            assert len(bn_momentum_tf_var_list) == 2

        model_input_new = sess_new.graph.get_tensor_by_name('input_1:0')
        training_tensor = sess_new.graph.get_tensor_by_name('keras_learning_phase:0')
        bn1_new_tensor = sess_new.graph.get_tensor_by_name("modified_bn_old1/cond/Identity/cond/Identity:0")
        bn2_new_tensor = sess_new.graph.get_tensor_by_name('modified_bn_old2/cond/Identity/cond/Identity:0')
        relu_new_tensor = sess_new.graph.get_tensor_by_name('Relu_1:0')

        feed_dict = {model_input_new: dummy_val, training_tensor: False}
        for training_tensor in is_training_tf_var_list:
            feed_dict[training_tensor] = False
        bn1_new, bn2_new, relu_new = sess_new.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor],
                                                  feed_dict=feed_dict)  # training_tensor: False})

        # Compare “original BN and modified BN “ are equivalent
        assert np.allclose(bn1_new, bn1_old, rtol=1.e-4)
        assert np.allclose(bn2_new, bn2_old, rtol=1.e-4)

        # check update_ops
        with sess_new.graph.as_default():
            for _, bn, _ in bn_conv_linear_pairs_new:
                bn_mean_tf_var_name = bn.op.inputs[0].op.inputs[3].name
                bn_var_tf_var_name = bn.op.inputs[0].op.inputs[4].name
                bn_cond1_tf_op = sess_new.graph.get_operation_by_name(
                    "/".join(bn.op.name.split("/")[0:-2]) + "/cond_1")

                bn_mean_update = \
                    bn_cond1_tf_op.outputs[0].consumers()[0].outputs[0].consumers()[0].outputs[0].consumers()[
                        0].outputs[
                        0].consumers()[0]
                bn_var_update = \
                    bn_cond1_tf_op.outputs[0].consumers()[0].outputs[0].consumers()[1].outputs[0].consumers()[
                        0].outputs[
                        0].consumers()[0]
                assert bn_mean_update.type in ['AssignSubVariableOp'] and bn_var_update.type in ['AssignSubVariableOp']


    def test_modify_model_bn_mutable_cpu(self, cpu_sessions_pretrained_tf2_model):
        model = cpu_sessions_pretrained_tf2_model
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

    @pytest.mark.cuda
    def test_modify_model_bn_mutable_gpu(self, gpu_sessions_pretrained_tf2_model):
        model = gpu_sessions_pretrained_tf2_model
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
        sess = tf.compat.v1.Session(graph=graph)
        sess.run(tf.compat.v1.global_variables_initializer())
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

    @pytest.mark.cuda
    def test_modify_model_bn_mutable_keras_mobilenetv2_model_gpu(self, gpu_sessions_tf_keras_applications_mobilenet_v2):
        model = gpu_sessions_tf_keras_applications_mobilenet_v2

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

    @pytest.mark.skip(
        reason=" new_bn = tf.compat.v1.layers.batch_normalization(batchnorm.in_tensor), depends on Compatible BNUtils "
               "on both TF1&TF2 runtime")
    @pytest.mark.cuda
    def test_modify_sess_bn_mutable_with_tf1_pre_trained_model_is_traing_as_placehoder_gpu(
            gpu_sessions_tf1_pre_trained_model_with_is_traing_as_placehoder):
        tf.compat.v1.reset_default_graph()
        _ = gpu_sessions_tf1_pre_trained_model_with_is_traing_as_placehoder

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.import_meta_graph('tf1_placeholder.meta')
            saver.restore(sess, tf.compat.v1.train.latest_checkpoint('./'))

            input_op_names = ["input_1"]
            output_op_names = ['Relu_1']
            dummy_val = np.random.randn(1, 32, 32, 3)

            # training_tensor = sess.graph.get_tensor_by_name('keras_learning_phase:0')
            model_input_old = sess.graph.get_tensor_by_name('input_1:0')
            bn1_old_tensor = sess.graph.get_tensor_by_name("old1/cond/Identity:0")
            bn2_old_tensor = sess.graph.get_tensor_by_name('old2/cond/Identity:0')
            relu_old_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
            bn1_old, bn2_old, relu_old = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                                  feed_dict={model_input_old: dummy_val})

            modify_sess_bn_mutable(sess, input_op_names, output_op_names)
            sess_new = sess
            conn_graph_new = ConnectedGraph(sess_new.graph, input_op_names, output_op_names)
            assert len(conn_graph_new._ops) == 7
            ops_name_list_new = [op for op in conn_graph_new._ops]
            assert "modified_bn_old1/cond/Identity" in ops_name_list_new
            assert "modified_bn_old2/cond/Identity" in ops_name_list_new
            bn_conv_linear_pairs_new = find_all_batch_norms_to_fold(sess_new, input_op_names, output_op_names)
            _, bn1_new, _ = bn_conv_linear_pairs_new[0]
            _, bn2_new, _ = bn_conv_linear_pairs_new[1]
            assert "modified_bn_old1/cond/Identity/cond/Identity" in bn1_new.op.name
            assert "modified_bn_old2/cond/Identity/cond/Identity" in bn2_new.op.name

            with sess_new.graph.as_default():
                is_training_tf_placehodler_list = []  # tf1.x style Golden BN layer API
                for _, bn, _ in bn_conv_linear_pairs_new:
                    assert bn.op.type in ['Identity']
                    # (mutable ) tf1.x style Golden BN layer API session
                    bn_cond1_tf_op = sess_new.graph.get_operation_by_name(
                        "/".join(bn.op.name.split("/")[0:-2]) + "/cond_1")
                    is_training_tf_placehodler = bn_cond1_tf_op.outputs[0].op.inputs[0]
                    is_training_tf_placehodler_list.append(is_training_tf_placehodler)

            bn_momentum_tf_var_list = []
            with sess_new.graph.as_default():
                tf_global_vars_new = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
                for var in tf_global_vars_new:
                    if "momentum_mutable_modified_bn_" in var.name:
                        bn_momentum_tf_var_list.append(var)
                assert len(bn_momentum_tf_var_list) == 2

            model_input_new = sess_new.graph.get_tensor_by_name('input_1:0')
            bn1_new_tensor = sess_new.graph.get_tensor_by_name("modified_bn_old1/cond/Identity/cond/Identity:0")
            bn2_new_tensor = sess_new.graph.get_tensor_by_name('modified_bn_old2/cond/Identity/cond/Identity:0')
            relu_new_tensor = sess_new.graph.get_tensor_by_name('Relu_1:0')

            feed_dict = {model_input_new: dummy_val}
            for training_tensor in is_training_tf_placehodler_list:
                feed_dict[training_tensor] = False
            bn1_new, bn2_new, relu_new = sess_new.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor],
                                                      feed_dict=feed_dict)  # training_tensor: False})

            assert np.allclose(bn1_new, bn1_old, rtol=1.e-4)
            assert np.allclose(bn2_new, bn2_old, rtol=1.e-4)
            assert np.allclose(relu_new, relu_old, rtol=1.e-4)

    @pytest.mark.skip(reason="fold_all_batch_norms type error depends on Compatible BNUtils on both TF1&TF2 runtime")
    @pytest.mark.cuda
    def test_modify_sess_bn_mutable_with_tf2_pre_trained_model_is_traing_as_placehoder_gpu(
            gpu_sessions_tf2_pre_trained_model_with_is_traing_as_placehoder):
        tf.compat.v1.reset_default_graph()
        _ = gpu_sessions_tf2_pre_trained_model_with_is_traing_as_placehoder

        with tf.compat.v1.Session() as sess:
            saver = tf.compat.v1.train.import_meta_graph('tf2_placeholder.meta')
            saver.restore(sess, tf.compat.v1.train.latest_checkpoint('./'))
            input_op_names = ["input_1"]
            output_op_names = ['Relu_1']
            dummy_val = np.random.randn(1, 32, 32, 3)

            # training_tensor = sess.graph.get_tensor_by_name('keras_learning_phase:0')
            model_input_old = sess.graph.get_tensor_by_name('input_1:0')
            bn1_old_tensor = sess.graph.get_tensor_by_name("old1/cond/Identity:0")
            bn2_old_tensor = sess.graph.get_tensor_by_name('old2/cond/Identity:0')
            relu_old_tensor = sess.graph.get_tensor_by_name('Relu_1:0')
            bn1_old, bn2_old, relu_old = sess.run([bn1_old_tensor, bn2_old_tensor, relu_old_tensor],
                                                  feed_dict={model_input_old: dummy_val})
            modify_sess_bn_mutable(sess, input_op_names,
                                   output_op_names)
            sess_new = sess
            conn_graph_new = ConnectedGraph(sess_new.graph, input_op_names, output_op_names)
            assert len(conn_graph_new._ops) == 7
            ops_name_list_new = [op for op in conn_graph_new._ops]
            assert "modified_bn_old1/cond/Identity" in ops_name_list_new
            assert "modified_bn_old2/cond/Identity" in ops_name_list_new
            bn_conv_linear_pairs_new = find_all_batch_norms_to_fold(sess_new, input_op_names, output_op_names)
            _, bn1_new, _ = bn_conv_linear_pairs_new[0]
            _, bn2_new, _ = bn_conv_linear_pairs_new[1]
            assert "modified_bn_old1/cond/Identity/cond/Identity" in bn1_new.op.name
            assert "modified_bn_old2/cond/Identity/cond/Identity" in bn2_new.op.name

            with sess_new.graph.as_default():
                is_training_tf_placehodler_list = []  # tf1.x style Golden BN layer API
                for _, bn, _ in bn_conv_linear_pairs_new:
                    assert bn.op.type in ['Identity']
                    # (mutable ) tf1.x style Golden BN layer API session
                    bn_cond1_tf_op = sess_new.graph.get_operation_by_name(
                        "/".join(bn.op.name.split("/")[0:-2]) + "/cond_1")
                    is_training_tf_placehodler = bn_cond1_tf_op.outputs[0].op.inputs[0]
                    is_training_tf_placehodler_list.append(is_training_tf_placehodler)

            bn_momentum_tf_var_list = []
            with sess_new.graph.as_default():
                tf_global_vars_new = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
                for var in tf_global_vars_new:
                    if "momentum_mutable_modified_bn_" in var.name:
                        bn_momentum_tf_var_list.append(var)
                assert len(bn_momentum_tf_var_list) == 2

            model_input_new = sess_new.graph.get_tensor_by_name('input_1:0')
            bn1_new_tensor = sess_new.graph.get_tensor_by_name("modified_bn_old1/cond/Identity/cond/Identity:0")
            bn2_new_tensor = sess_new.graph.get_tensor_by_name('modified_bn_old2/cond/Identity/cond/Identity:0')
            relu_new_tensor = sess_new.graph.get_tensor_by_name('Relu_1:0')

            feed_dict = {model_input_new: dummy_val}
            for training_tensor in is_training_tf_placehodler_list:
                feed_dict[training_tensor] = False
            bn1_new, bn2_new, relu_new = sess_new.run([bn1_new_tensor, bn2_new_tensor, relu_new_tensor],
                                                      feed_dict=feed_dict)  # training_tensor: False})

            assert np.allclose(bn1_new, bn1_old, rtol=1.e-4)
            assert np.allclose(bn2_new, bn2_old, rtol=1.e-4)
            assert np.allclose(relu_new, relu_old, rtol=1.e-4)

            from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms
            fold_sess, pairs = fold_all_batch_norms(sess_new, input_op_names, output_op_names)
