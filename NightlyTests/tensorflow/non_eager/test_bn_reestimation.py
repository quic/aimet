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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import json
import numpy as np
import tensorflow as tf

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.bn_reestimation import reestimate_bn_stats, _get_all_tf_bn_vars_list
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms_to_scale
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.utils.op.bn_mutable import modify_sess_bn_mutable
from aimet_tensorflow.utils.op.fusedbatchnorm import BNUtils
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.utils.op.bn_mutable import get_active_bn_ops, set_mutable_bn_is_training_var

#from Examples.tensorflow.utils.add_computational_nodes_in_graph import add_image_net_computational_nodes_in_graph
# currently multiple notebook example is using the above utilty function, making a copy,
# TODO refactor to common TF util

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
AimetLogger.set_level_for_all_areas(logging.DEBUG)
tf.compat.v1.disable_eager_execution()
np.random.seed(0)
tf.compat.v1.set_random_seed(0)

def add_image_net_computational_nodes_in_graph(session: tf.compat.v1.Session, logits_name: str, num_classes: int):
    """
    :param session: Tensorflow session to operate on
    :param logits_name: Output tensor name of session graph
    :param num_classes: No of classes in model data
    """
    with session.graph.as_default():
        # predicted value of the model
        y_hat = session.graph.get_tensor_by_name(logits_name)
        y_hat_argmax = tf.compat.v1.argmax(y_hat, axis=1)

        # placeholder for the labels
        y = tf.compat.v1.placeholder(tf.compat.v1.int64, shape=[None, num_classes], name='labels')
        y_argmax = tf.compat.v1.argmax(y, axis=1)

        # prediction Op
        correct_prediction = tf.compat.v1.equal(y_hat_argmax, y_argmax)

        # pylint: disable-msg=unused-variable
        # accuracy Op: top1
        top1_acc = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.compat.v1.float32), name='top1-acc')

        # accuracy Op: top5
        top5_acc = tf.compat.v1.reduce_mean(tf.compat.v1.cast(tf.compat.v1.nn.in_top_k(predictions=y_hat,
                                                         targets=tf.compat.v1.cast(y_argmax, tf.compat.v1.int32),
                                                         k=5),
                                          tf.compat.v1.float32),
                                  name='top5-acc')

        # loss Op: loss
        loss = tf.compat.v1.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=y, logits=y_hat))

def get_all_status(
        sess,
        bn_mean_var_tf_var_list, bn_momentum_tf_var_list, bn_training_tf_var_list
):
    """
    get current all stats (momentum,training, mean,var) for debug and unit test
   ."""
    with sess.graph.as_default():
        bn_stats_dict = dict(zip(bn_mean_var_tf_var_list, sess.run([v for v in bn_mean_var_tf_var_list])))
        bn_momentum_dict = dict(zip(bn_momentum_tf_var_list, sess.run([v for v in bn_momentum_tf_var_list])))
        bn_traning_dict = dict(zip(bn_training_tf_var_list, sess.run([v for v in bn_training_tf_var_list])))
    return bn_stats_dict, bn_momentum_dict, bn_traning_dict


def is_dict_close_numpy_array_zeros(dict1):
    for k in dict1.keys():
        np_zeros = np.zeros(dict1[k].shape)
        if not (np.allclose(dict1[k], np_zeros)):
            return False
    return True


def is_two_dict_close_numpy_array(dict1, dict2):
    for k in dict1.keys():
        if not (np.allclose(dict1[k], dict2[k])):
            return False
    return True


def is_two_dict_close_bool(dict1, dict2):
    for k in dict1.keys():
        if not (dict1[k] == dict2[k]):
            return False
    return True


def is_two_dict_close_float(dict1, dict2):
    for k in dict1.keys():
        if not (dict1[k] == pytest.approx(dict2[k])):
            return False
    return True


@pytest.fixture(scope="session")
def bn_num_batches():
    return 4


@pytest.fixture(scope="session")
def batch_size():
    return 2


@pytest.fixture
def bn_re_estimation_dataset(bn_num_batches, batch_size):
    graph = tf.Graph()
    with graph.as_default():
        dummy_inputs = tf.random.normal((bn_num_batches * batch_size, 32, 32, 3))
        dataset = tf.compat.v1.data.Dataset.from_tensor_slices(dummy_inputs)
        dataset = dataset.batch(batch_size)
        return dataset

class TestBNReEstimation:

    def test_modle_rewriter_ptq_reestimation_fold(self, bn_re_estimation_dataset, bn_num_batches):
        tf.compat.v1.reset_default_graph()
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=None, input_shape=(32, 32, 3))
        graph = model.inputs[0].graph
        sess = tf.compat.v1.Session(graph=graph)
        initialize_uninitialized_vars(sess)

        # model rewriter
        start_op_names = ["input_1"]
        end_op_names = ["predictions/Softmax"]
        sess = modify_sess_bn_mutable(sess, start_op_names, end_op_names, training_tf_placeholder=False)

        # PTQ

        default_config_per_channel = {
            "defaults":
                {
                    "ops":
                        {
                            "is_output_quantized": "True"
                        },
                    "params":
                        {
                            "is_quantized": "True",
                            "is_symmetric": "True"
                        },
                    "strict_symmetric": "False",
                    "unsigned_symmetric": "True",
                    "per_channel_quantization": "True"
                },

            "params":
                {
                    "bias":
                        {
                            "is_quantized": "False"
                        }
                },

            "op_type":
                {
                    "Squeeze":
                        {
                            "is_output_quantized": "False"
                        },
                    "Pad":
                        {
                            "is_output_quantized": "False"
                        },
                    "Mean":
                        {
                            "is_output_quantized": "False"
                        }
                },

            "supergroups":
                [
                    {
                        "op_list": ["Conv", "Relu"]
                    },
                    {
                        "op_list": ["Conv", "Clip"]
                    },
                    {
                        "op_list": ["Conv", "BatchNormalization", "Relu"]
                    },
                    {
                        "op_list": ["Add", "Relu"]
                    },
                    {
                        "op_list": ["Gemm", "Relu"]
                    }
                ],

            "model_input":
                {
                    "is_input_quantized": "True"
                },

            "model_output":
                {}
        }

        config_file_path = "/tmp/default_config_per_channel.json"
        with open(config_file_path, "w") as f:
            json.dump(default_config_per_channel, f)

        sim = QuantizationSimModel(sess, start_op_names, end_op_names, use_cuda=True,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   config_file=config_file_path)



        def dummy_forward_pass(sess, args):
            model_input = sess.graph.get_tensor_by_name("input_1:0")
            model_output = sess.graph.get_tensor_by_name('predictions/Softmax:0')
            dummy_val = np.random.randn(1, *model_input.shape[1:])
            sess.run(model_output, feed_dict={model_input: dummy_val})
        sim.compute_encodings(dummy_forward_pass, None)

        # check bn_re_estimation
        self._reestimate_and_compare_results(sim, sess, bn_re_estimation_dataset, bn_num_batches, "input_1", "predictions/Softmax")

        # check bn_fold
        model_input = sim.session.graph.get_tensor_by_name("input_1:0")
        model_output = sim.session.graph.get_tensor_by_name('predictions/Softmax:0')
        dummy_val = np.random.randn(128, *model_input.shape[1:])

        output_baseline = sim.session.run(model_output, feed_dict={model_input: dummy_val})

        fold_all_batch_norms_to_scale(sim, start_op_names, end_op_names)

        model_input_after_fold = sim.session.graph.get_tensor_by_name("input_1:0")
        model_output_after_fold = sim.session.graph.get_tensor_by_name('predictions/Softmax:0')

        output_fold_after_fold = sim.session.run(model_output_after_fold, feed_dict={model_input_after_fold: dummy_val})

        assert np.allclose(output_baseline, output_fold_after_fold, atol=1e-2)
        sim.session.close()

    def test_remove_bn_update_ops_with_training_ops(self):
        """ verify that the BNs UPDATE_OPS are removed correctly after training ops are added (QAT) """
        tf.compat.v1.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            tf.keras.applications.mobilenet_v2.MobileNetV2(weights=None, input_shape=(224, 224, 3))
        sess = tf.compat.v1.Session(graph=graph)
        initialize_uninitialized_vars(sess)

        start_op_names = ["input_1"]
        output_op_names = ["predictions/Softmax"]
        validation_inputs = ["labels"]
        add_image_net_computational_nodes_in_graph(sess, logits_name=output_op_names[0] + ':0', num_classes=1000)

        # Update BNs with mutable BNs.
        updated_sess = modify_sess_bn_mutable(sess, start_op_names, output_op_names)

        with updated_sess.graph.as_default():
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            assert len(update_ops) == 104

        # set all mutable BNs in training mode.
        set_mutable_bn_is_training_var(updated_sess, True)
        self._training_loop(updated_sess, update_ops, start_op_names, validation_inputs)

        # Find BNs UPDATE_OPS programmatically.
        update_ops_programmatically = []
        conn_graph = ConnectedGraph(updated_sess.graph, start_op_names, output_op_names)
        bn_conn_graph_ops = tuple(get_active_bn_ops(conn_graph))
        for bn_conn_graph_op in bn_conn_graph_ops:
            bn_tf_op = bn_conn_graph_op.get_tf_op_with_io_tensor().op
            assign_moving_avg_op = BNUtils.get_assign_moving_avg_op(bn_tf_op)
            assign_moving_avg_op_1 = BNUtils.get_assign_moving_avg_1_op(bn_tf_op)
            update_ops_programmatically.append(assign_moving_avg_op)
            update_ops_programmatically.append(assign_moving_avg_op_1)

        with updated_sess.graph.as_default():
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        assert len(update_ops_programmatically) == len(update_ops)

        # Remove BNs UPDATE_OPS
        for bn_conn_graph_op in bn_conn_graph_ops:
            bn_tf_op = bn_conn_graph_op.get_tf_op_with_io_tensor().op
            BNUtils.remove_bn_op_from_update_ops(updated_sess, bn_tf_op)
        with updated_sess.graph.as_default():
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        # check that UPDATE_OPS list is empty
        assert not update_ops

        sess.close()
        updated_sess.close()

    @staticmethod
    def _training_loop(session, update_ops, data_inputs, validation_inputs):
        """ utility to add training ops """
        dummy_input = np.random.randn(1, 224, 224, 3)
        dummy_labels = np.random.randn(1, 1000)

        with session.graph.as_default():
            loss_op = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOSSES)[0]
            global_step_op = tf.compat.v1.train.create_global_step()

            # Define an optimizer
            optimizer_op = tf.compat.v1.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
            with tf.control_dependencies(update_ops):
                train_op = optimizer_op.minimize(loss_op, global_step=global_step_op)

            initialize_uninitialized_vars(session)
            input_label_tensors = [session.graph.get_tensor_by_name(input_label + ':0')
                                   for input_label in tuple(data_inputs) + tuple(validation_inputs)]
            input_label_tensors_dict = {input_label_tensors[0]: dummy_input,
                                        input_label_tensors[1]: dummy_labels}
            feed_dict = {**input_label_tensors_dict}
            for i in range(2):
                batch_loss_val, _ = session.run([loss_op, train_op], feed_dict=feed_dict)

    def _reestimate_and_compare_results(self, sess_sim, sess_fp32, bn_re_restimation_dataset, bn_num_batches, input_op, output_op):
        bn_mean_var_tf_var_list, bn_momentum_tf_var_list, bn_training_tf_var_list = _get_all_tf_bn_vars_list(sess_sim)

        model_input = sess_sim.session.graph.get_tensor_by_name(input_op + ':0')
        model_output = sess_sim.session.graph.get_tensor_by_name(output_op + ':0')
        dummy_val = np.random.randn(1, *model_input.shape[1:])

        feed_dict_data = {model_input: dummy_val}
        for bn_training in bn_training_tf_var_list:
            feed_dict_data[bn_training]: True
        sess_sim.session.run(model_output, feed_dict=feed_dict_data)

        bn_mean_var_ori, bn_momentum_ori, bn_training_ori = get_all_status(sess_sim.session, bn_mean_var_tf_var_list,
                                                                           bn_momentum_tf_var_list, bn_training_tf_var_list)

        with reestimate_bn_stats(sim=sess_sim, start_op_names=[input_op], output_op_names=[output_op],
                                 bn_re_estimation_dataset=bn_re_restimation_dataset,
                                 bn_num_batches=bn_num_batches):
            bn_mean_var_est, bn_momentum_est, bn_training_est = get_all_status(sess_sim.session, bn_mean_var_tf_var_list,
                                                                               bn_momentum_tf_var_list,
                                                                               bn_training_tf_var_list)
            # Sanity check(apply_bn_re_estimation):  re-estimation , update runing mean &var, set training with False for
            # eval(), momentum  no change
            assert not is_two_dict_close_numpy_array(bn_mean_var_ori, bn_mean_var_est)
            assert not is_dict_close_numpy_array_zeros(bn_mean_var_est)
            assert not is_two_dict_close_float(bn_momentum_ori, bn_momentum_est)
            assert not is_two_dict_close_bool(bn_training_ori, bn_training_est)

        bn_mean_var_restored, bn_momentum_restored, bn_training_restored = get_all_status(sess_sim.session,
                                                                                          bn_mean_var_tf_var_list,
                                                                                          bn_momentum_tf_var_list,
                                                                                          bn_training_tf_var_list)
        # Sanity check(train_mode): restore  mean &var, set training with True for train(), momentum no change
        assert is_two_dict_close_numpy_array(bn_mean_var_ori, bn_mean_var_restored)
        assert is_two_dict_close_float(bn_momentum_ori, bn_momentum_restored)
        assert is_two_dict_close_bool(bn_training_ori, bn_training_restored)
