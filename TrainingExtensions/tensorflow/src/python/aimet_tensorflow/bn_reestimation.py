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

"""BatchNorm Reestimation"""
from typing import List, Tuple, Dict
import numpy as np
import tensorflow as tf
from aimet_common.utils import Handle, AimetLogger
from aimet_tensorflow.utils.op.fusedbatchnorm import BNUtils
from aimet_tensorflow.batch_norm_fold import find_all_batch_norms_to_fold
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils.common import create_input_feed_dict, iterate_tf_dataset

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# pylint: disable=not-an-iterable
# pylint: disable=too-many-locals
def _get_all_tf_bn_vars_list(sim: QuantizationSimModel, start_op_names: List[str],
                             output_op_names: List[str]) -> Tuple:
    """
    find tf varaible list to access
    :param sim: tf quantized model
    :param start_op_names: List of starting op names of the model
    :param output_op_names: List of output op names of the model
    :param momentum_names: BN's momentum name list
    :param training_names: BN's training_names list
    :return: tf.variable lists to access bn layers's mean,var,momentum,training
    """
    sim.export("/tmp/", "sim_model")
    config = tf.compat.v1.ConfigProto()
    new_sess = tf.compat.v1.Session(graph=tf.compat.v1.Graph(), config=config)
    with new_sess.graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph("/tmp/sim_model" + '.meta')
    saver.restore(new_sess, "/tmp/sim_model")
    bn_conv_linear_pairs = find_all_batch_norms_to_fold(new_sess, start_op_names, output_op_names)
    with sim.session.graph.as_default():
        tf_global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        mean_var_tf_var_name_list = []
        training_tf_var_name_list = []
        momentum_tf_var_name_list = []

        for _, batchnorm, _ in bn_conv_linear_pairs:
            assert batchnorm.op.type in ['Identity']
            bn_mean_tf_var_name = batchnorm.op.inputs[0].op.inputs[3].name
            bn_var_tf_var_name = batchnorm.op.inputs[0].op.inputs[4].name

            bn_cond_1_tf_op = BNUtils.get_cond_1_identity_op(batchnorm.op)
            bn_momentum_tf_var_name = bn_cond_1_tf_op.inputs[0].op.inputs[1].name
            bn_training_tf_var_name = batchnorm.op.inputs[0].op.inputs[0].op.inputs[0].name

            mean_var_tf_var_name_list.append(bn_mean_tf_var_name)
            mean_var_tf_var_name_list.append(bn_var_tf_var_name)
            momentum_tf_var_name_list.append(bn_momentum_tf_var_name)
            training_tf_var_name_list.append(bn_training_tf_var_name)

        mean_var_tf_var_list = []
        training_tf_var_list = []
        momentum_tf_var_list = []
        for v in tf_global_vars:

            for tf_var_name in mean_var_tf_var_name_list:
                if v.name == tf_var_name:
                    mean_var_tf_var_list.append(v)

            for tf_momentum_name in momentum_tf_var_name_list:
                if v.name == tf_momentum_name:
                    momentum_tf_var_list.append(v)

            for tf_traning_name in training_tf_var_name_list:
                if v.name == tf_traning_name:
                    training_tf_var_list.append(v)

    return mean_var_tf_var_list, momentum_tf_var_list, training_tf_var_list


# pylint: disable=not-an-iterable
# pylint: disable=unsubscriptable-object
def _reset_bn_stats(sess: tf.compat.v1.Session, bn_mean_var_checkpoints: Dict, bn_momentum_checkpoints: Dict,
                    bn_training_checkpoints: Dict) -> Handle:
    """
    reset bn stats
    :param sess: tf session
    :param bn_mean_var_checkpoints: Dict for original mean&var
    :param bn_momentum_checkpoints: Dict for original bn momentum
    :param bn_training_checkpoints: Dict for original bn training
    :return:
    """

    def cleanup():
        """
        Restore Bn stats
        """
        with sess.graph.as_default():
            sess.run([tf.compat.v1.assign(k, bn_mean_var_checkpoints[k]) for k in bn_mean_var_checkpoints.keys()])
            sess.run([tf.compat.v1.assign(k, bn_momentum_checkpoints[k]) for k in bn_momentum_checkpoints.keys()])
            sess.run([tf.compat.v1.assign(k, bn_training_checkpoints[k]) for k in bn_training_checkpoints.keys()])

    try:
        with sess.graph.as_default():
            sess.run([tf.compat.v1.assign(k, 0.0) for k in bn_momentum_checkpoints.keys()])
            sess.run([tf.compat.v1.assign(k, tf.compat.v1.constant(True)) for k in bn_training_checkpoints.keys()])
        return Handle(cleanup)
    except:
        cleanup()
        raise


# pylint: disable=too-many-arguments
# pylint: disable=not-an-iterable
# pylint: disable=unsubscriptable-object
# pylint: disable=too-many-locals
def reestimate_bn_stats(sim: QuantizationSimModel, start_op_names: List[str],
                        output_op_names: List[str], bn_re_estimation_dataset: tf.compat.v1.data.Dataset,
                        bn_num_batches: int = 100) -> Handle:
    """
    top level api for end user directly call for eval()
    :param sim: tf quantized model
    :param start_op_names: List of starting op names of the model
    :param output_op_names: List of output op names of the model
    :param bn_re_estimation_dataset: Training dataset
    :param bn_num_batches: The number of batches to be used for reestimation
    :returns: Handle that undos the effect of BN reestimation upon handle.remove()
    """
    # setup tf varaible list to access

    bn_mean_var_tf_var_list, bn_momentum_tf_var_list, bn_training_tf_var_list = \
        _get_all_tf_bn_vars_list(sim, start_op_names, output_op_names)

    sess = sim.session
    with sess.graph.as_default():
        # save checkpoints
        bn_mean_var_checkpoints = dict(zip(bn_mean_var_tf_var_list, sess.run([v for v in bn_mean_var_tf_var_list])))
        bn_momentum_checkpoints = dict(zip(bn_momentum_tf_var_list, sess.run([v for v in bn_momentum_tf_var_list])))
        bn_training_checkpoints = dict(zip(bn_training_tf_var_list, sess.run([v for v in bn_training_tf_var_list])))

    # 1. switch to re-estimation mode and setup remove
    handle = _reset_bn_stats(sess, bn_mean_var_checkpoints, bn_momentum_checkpoints, bn_training_checkpoints)
    # 2 per batch forward and BN re-estimation
    with sess.graph.as_default():
        output_ops = [sess.graph.get_operation_by_name(name) for name in output_op_names]
        output_tensors = [sess.graph.get_tensor_by_name(output_op.name + ':0') for output_op in output_ops]
        bn_dataset_iterator = iterate_tf_dataset(bn_re_estimation_dataset)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.compat.v1.control_dependencies(update_ops):
            output_tensors_dependencies = []
            for output_tensor in output_tensors:
                output_tensor = tf.compat.v1.identity(output_tensor)
                output_tensors_dependencies.append(output_tensor)
        initialize_uninitialized_vars(sess)
        # (1)intilization
        sum_dict = {v: np.zeros(v.shape, dtype=v.dtype.as_numpy_dtype) for v in bn_mean_var_tf_var_list}
        # (2)forward and accumulate mean and var
    for batch_index in range(bn_num_batches):
        try:
            batch_data = next(bn_dataset_iterator)
            feed_dict = create_input_feed_dict(sess.graph, start_op_names, batch_data)
            sess.run(output_tensors_dependencies, feed_dict=feed_dict)
            for v in bn_mean_var_tf_var_list:
                sum_dict[v] += sess.run(v)
            if batch_index == bn_num_batches - 1:
                break
        except tf.errors.OutOfRangeError:
            logger.info("tf.errors.OutOfRangeError:: no data from BN dataset.")  # ==> "End of dataset"
            break
    # (3) average mean&var
    for k in sum_dict.keys():
        sum_dict[k] = sum_dict[k] / bn_num_batches
    # (4) apply result: update BN stats
    with sess.graph.as_default():
        sess.run([tf.compat.v1.assign(k, sum_dict[k]) for k in bn_mean_var_checkpoints.keys()])
    return handle
