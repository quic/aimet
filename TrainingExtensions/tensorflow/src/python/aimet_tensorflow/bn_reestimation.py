# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""BatchNorm Re-estimation"""

from typing import List, Tuple, Dict
import numpy as np
import tensorflow as tf

from aimet_common.utils import Handle, AimetLogger
from aimet_tensorflow.utils.op.fusedbatchnorm import BNUtils
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils.common import create_input_feed_dict, iterate_tf_dataset
from aimet_tensorflow.utils.op.bn_mutable import get_active_bn_ops

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# pylint: disable=too-many-locals
def _get_all_tf_bn_vars_list(sim: QuantizationSimModel) -> Tuple[List[tf.Variable],
                                                                 List[tf.Variable],
                                                                 List[tf.Variable],
                                                                 List[tf.Variable]]:
    """
    find tf variables list to access BNs mean, variance, momentum and is_training

    :param sim: tf quantized model
    :return: tf.variable lists to access bn layers's mean, var, momentum, is_training
    """
    conn_graph = sim.connected_graph
    bn_conn_graph_ops = tuple(get_active_bn_ops(conn_graph))

    with sim.session.graph.as_default():
        tf_global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

    mean_tf_var_names = []
    variance_tf_var_names = []
    is_training_tf_var_names = []
    momentum_tf_var_names = []

    for bn_conn_graph_op in bn_conn_graph_ops:
        tf_op = bn_conn_graph_op.internal_ops[0]
        assert tf_op.type in ['Identity'], 'Fused Batch Norm with training tensor is only supported.'
        bn_mean_tf_var_name = tf_op.inputs[0].op.inputs[3].name
        bn_var_tf_var_name = tf_op.inputs[0].op.inputs[4].name

        bn_cond_1_tf_op = BNUtils.get_cond_1_identity_op(tf_op)
        bn_momentum_tf_var_name = bn_cond_1_tf_op.inputs[0].op.inputs[1].name
        bn_training_tf_var_name = tf_op.inputs[0].op.inputs[0].op.inputs[0].name

        mean_tf_var_names.append(bn_mean_tf_var_name)
        variance_tf_var_names.append(bn_var_tf_var_name)
        momentum_tf_var_names.append(bn_momentum_tf_var_name)
        is_training_tf_var_names.append(bn_training_tf_var_name)

    mean_tf_vars = []
    variance_tf_vars = []
    is_training_tf_vars = []
    momentum_tf_vars = []

    for v in tf_global_vars:
        if v.name in mean_tf_var_names:
            mean_tf_vars.append(v)

        if v.name in variance_tf_var_names:
            variance_tf_vars.append(v)

        if v.name in momentum_tf_var_names:
            momentum_tf_vars.append(v)

        if v.name in is_training_tf_var_names:
            is_training_tf_vars.append(v)

    return mean_tf_vars, variance_tf_vars, momentum_tf_vars, is_training_tf_vars


def _reset_bn_stats(sess: tf.compat.v1.Session,
                    bn_mean_checkpoints: Dict[tf.Variable, np.ndarray],
                    bn_variance_checkpoints: Dict[tf.Variable, np.ndarray]) -> Handle:
    """
    Reset all BNs statistics to the initial values.

    :param sess: tf session
    :param bn_mean_checkpoints: Dict for original BN mean
    :param bn_variance_checkpoints: Dict for original BN variance
    :return: Handle that restores the original BN statistics upon handle.remove().
    """
    def cleanup():
        """
        Restore all BNs stats
        """
        with sess.graph.as_default():
            sess.run([tf.compat.v1.assign(v, bn_mean_checkpoints[v]) for v in bn_mean_checkpoints])
            sess.run([tf.compat.v1.assign(v, bn_variance_checkpoints[v]) for v in bn_variance_checkpoints])

    try:
        with sess.graph.as_default():
            sess.run([tf.compat.v1.assign(v, np.zeros(v.shape, dtype=v.dtype.as_numpy_dtype))
                      for v in bn_mean_checkpoints])
            sess.run([tf.compat.v1.assign(v, np.ones(v.shape, dtype=v.dtype.as_numpy_dtype))
                      for v in bn_variance_checkpoints])
        return Handle(cleanup)
    except:
        cleanup()
        raise


def _reset_momentum(sess: tf.compat.v1.Session,
                    momentum_checkpoints: Dict[tf.Variable, np.float32]) -> Handle:
    """
    Set all BNs momentum to 0.0.

    :param sess: tf session
    :param momentum_checkpoints: Dict for original BN momentum[tf.Variable --> original_values]
    :return: Handle that restores the original BN statistics upon handle.remove().
    """
    def cleanup():
        """
        Restore BNs momentum
        """
        with sess.graph.as_default():
            sess.run([tf.compat.v1.assign(v, momentum_checkpoints[v]) for v in momentum_checkpoints])
    try:
        with sess.graph.as_default():
            sess.run([tf.compat.v1.assign(v, 0.0) for v in momentum_checkpoints])
        return Handle(cleanup)
    except:
        cleanup()
        raise


def _set_bn_in_train_mode(sess: tf.compat.v1.Session,
                          is_training_checkpoints: Dict[tf.Variable, bool]) -> Handle:
    """
    Set BNs in training mode.

    :param sess: tf session
    :param is_training_checkpoints: Dict for original BNs is_training flag.
    :return: Handle that sets all mutable BNs to eval mode upon handle.remove().
    """
    def cleanup():
        """
        Set all the BNs to eval mode.
        """
        with sess.graph.as_default():
            sess.run([tf.compat.v1.assign(k, False) for k in is_training_checkpoints])

    try:
        # Set all the BNs to train mode
        with sess.graph.as_default():
            sess.run([tf.compat.v1.assign(k, True) for k in is_training_checkpoints])
        return Handle(cleanup)
    except:
        cleanup()
        raise


def _get_tf_vars_and_orig_values(sim: QuantizationSimModel) -> Tuple[Dict[tf.Variable, np.ndarray],
                                                                     Dict[tf.Variable, np.ndarray],
                                                                     Dict[tf.Variable, np.float32],
                                                                     Dict[tf.Variable, bool]]:
    """
    save original values for all BNs mean, variance, momentum and is_training tf Variables.

    :param sim: QuantizationSimModel object.
    :return: Dictionary [tf.Variable] --> original_value for all BNs mean, variance, momentum and is_training.
    """
    # setup tf variable list to access
    mean_tf_vars, variance_tf_vars, momentum_tf_vars, is_training_tf_vars = _get_all_tf_bn_vars_list(sim)

    with sim.session.graph.as_default():
        mean_checkpoints = dict(zip(mean_tf_vars, sim.session.run(list(mean_tf_vars))))
        variance_checkpoints = dict(zip(variance_tf_vars, sim.session.run(list(variance_tf_vars))))
        momentum_checkpoints = dict(zip(momentum_tf_vars, sim.session.run(list(momentum_tf_vars))))
        is_training_checkpoints = dict(zip(is_training_tf_vars, sim.session.run(list(is_training_tf_vars))))

    return mean_checkpoints, variance_checkpoints, momentum_checkpoints, is_training_checkpoints


DEFAULT_NUM_BATCHES = 100


def reestimate_bn_stats(sim: QuantizationSimModel,
                        start_op_names: List[str],
                        output_op_names: List[str],
                        dataset: tf.compat.v1.data.Dataset,
                        num_batches: int = DEFAULT_NUM_BATCHES) -> Handle:
    """
    Reestimate BatchNorm statistics (running mean and var).

    :param sim: QuantizationSimModel object.
    :param start_op_names: List of starting op names of the model
    :param output_op_names: List of output op names of the model
    :param dataset: Training dataset
    :param num_batches: The number of batches to be used for reestimation
    :returns: Handle that undos the effect of BN reestimation upon handle.remove()
    """
    # setup tf variable list to access
    mean_checkpoints, variance_checkpoints, momentum_checkpoints, is_training_checkpoints = \
        _get_tf_vars_and_orig_values(sim)

    sess = sim.session
    # Set all the BNs in training mode
    with _set_bn_in_train_mode(sess, is_training_checkpoints), _reset_momentum(sess, momentum_checkpoints):
        handle = _reset_bn_stats(sess, mean_checkpoints, variance_checkpoints)
        try:
            with sess.graph.as_default():
                output_tensors = [sess.graph.get_tensor_by_name(name + ':0') for name in output_op_names]
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                assert update_ops, "GraphKeys.UPDATE_OPS can not be empty."

                # GraphKeys.UPDATE_OPS is collection of moving mean and variance for BN layers. During training mode
                # moving mean and variance need to be updated and added as a control dependency.
                with tf.compat.v1.control_dependencies(update_ops):
                    output_tensors_dependencies = []
                    for output_tensor in output_tensors:
                        output_tensor = tf.compat.v1.identity(output_tensor)
                        output_tensors_dependencies.append(output_tensor)
                initialize_uninitialized_vars(sess)

            # BN statistics accumulation buffer
            sum_mean = {v: np.zeros(v.shape, dtype=v.dtype.as_numpy_dtype) for v in mean_checkpoints}
            sum_var = {v: np.zeros(v.shape, dtype=v.dtype.as_numpy_dtype) for v in variance_checkpoints}

            batches = 0
            iterator = iterate_tf_dataset(dataset)
            for _ in range(num_batches):
                try:
                    data = next(iterator)
                    batches += 1
                except StopIteration:
                    break
                feed_dict = create_input_feed_dict(sess.graph, start_op_names, data)
                sess.run(output_tensors_dependencies, feed_dict=feed_dict)
                for v in mean_checkpoints:
                    sum_mean[v] += sess.run(v)
                for v in variance_checkpoints:
                    sum_var[v] += sess.run(v)

            # Override BN stats with the reestimated stats.
            with sess.graph.as_default():
                sess.run([tf.compat.v1.assign(v, sum_mean[v] / batches) for v in mean_checkpoints])
                sess.run([tf.compat.v1.assign(v, sum_var[v] / batches) for v in variance_checkpoints])

            return handle
        except:
            handle.remove()
            raise
