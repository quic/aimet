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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import logging
from typing import List, Callable, Any

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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
AimetLogger.set_level_for_all_areas(logging.DEBUG)
tf.compat.v1.disable_eager_execution()
np.random.seed(0)
tf.compat.v1.set_random_seed(0)

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
        modify_sess_bn_mutable(sess, start_op_names, end_op_names, training_tf_placeholder=False)

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


    def _reestimate_and_compare_results(self, sess_sim, sess_fp32, bn_re_restimation_dataset, bn_num_batches, input_op, output_op):
        bn_mean_var_tf_var_list, bn_momentum_tf_var_list, bn_training_tf_var_list = _get_all_tf_bn_vars_list(
            sess_sim, [input_op], [output_op])

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
