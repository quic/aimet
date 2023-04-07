# /usr/bin/env python3.5
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
""" Utilities to modify batchnorm layer's momentum of pre-traind tf2 model as mutable TF variable """

from typing import List, Union, Tuple
import numpy as np
import tensorflow as tf

from aimet_tensorflow import graph_editor
from aimet_tensorflow.utils.graph_saver import save_and_load_graph
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.common.operation import Op
from aimet_tensorflow.utils.constants import BNOpParamType

_DEFAULT_BN_MOMENTUM = 0.99
_DEFAULT_BN_EPSILON = 0.001
_BN_MOMENTUM_NAME = '/mutable_momentum'


def modify_model_bn_mutable(model: tf.keras.Model):
    """
    Utilities to modify batchnorm layer's momentum of keras model as mutable tf.Variable

    :param model: keras model to modify batchnorms
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            momentum = layer.momentum
            bn_momentum_var = tf.Variable(momentum, trainable=False, name=layer.name + _BN_MOMENTUM_NAME)
            layer.momentum = bn_momentum_var


# pylint: disable=too-many-locals
def modify_sess_bn_mutable(sess: tf.compat.v1.Session,
                           start_op_names: Union[List[str], str],
                           output_op_names: Union[List[str], str],
                           training_tf_placeholder: bool = False) -> tf.compat.v1.Session:
    """
    Utilities to modify Batch norm layer's momentum and training flag.

    NOTE: For is_training flag, single placeholder/variable (default False) is attached to all the Batch norm layers.

    :param sess: active tf.compat.v1.Session
    :param start_op_names: Name of the starting op in the given graph or a list of names in case of multi-input model
    :param output_op_names: List of output op names of the model, used to help ConnectedGraph determine valid ops
           (to ignore training ops for example).  If None, all ops in the model are considered valid.
    :param training_tf_placeholder: Use tf.placeholder as training arg when set to True, else use tf.Variable
    :return: new session with modifiable momentum and training flag for Batch norm layers.
    """
    conn_graph = ConnectedGraph(sess.graph, start_op_names, output_op_names)
    bn_ops = tuple(get_active_bn_ops(conn_graph))
    graph_def = sess.graph.as_graph_def()

    with sess.graph.as_default():
        if training_tf_placeholder:
            bn_training = tf.compat.v1.placeholder_with_default(False, shape=[], name='bn_training_placeholder')
        else:
            bn_training = tf.Variable(tf.compat.v1.constant(False), trainable=False, name='bn_training_var')

        for bn_op in bn_ops:
            new_bn_name = bn_op.name + '_modified'

            # Get Batch norm layer statistics (mean and variance) and params (gamma and beta)
            mean, var, gamma, beta = _get_bn_stats_and_params(sess, bn_op)
            mean_init = tf.compat.v1.constant_initializer(mean, dtype=tf.float32)
            var_init = tf.compat.v1.constant_initializer(var, dtype=tf.float32)
            beta_init = tf.compat.v1.constant_initializer(beta, dtype=tf.float32)
            gamma_init = tf.compat.v1.constant_initializer(gamma, dtype=tf.float32)

            # Get momentum and epsilon. Since both are compiled inside the graph, use graph_def.
            epsilon = _get_bn_epsilon(graph_def, bn_op)
            momentum = _get_bn_momentum(graph_def, bn_op)
            momentum_var = tf.Variable(momentum, trainable=False, name=new_bn_name + _BN_MOMENTUM_NAME)

            tf_op = bn_op.get_tf_op_with_io_tensor()
            new_bn = tf.compat.v1.layers.batch_normalization(tf_op.in_tensor,
                                                             epsilon=epsilon,
                                                             momentum=momentum_var,
                                                             beta_initializer=beta_init,
                                                             gamma_initializer=gamma_init,
                                                             moving_mean_initializer=mean_init,
                                                             moving_variance_initializer=var_init,
                                                             name=new_bn_name,
                                                             training=bn_training,
                                                             )
            graph_editor.reroute_ts(ts0=new_bn, ts1=tf_op.out_tensor)
            graph_editor.detach_inputs(tf_op.op)

    initialize_uninitialized_vars(sess)
    after_bn_mutable_sess = save_and_load_graph('./temp_bn_mutable', sess)
    return after_bn_mutable_sess


def get_active_bn_ops(connected_graph: ConnectedGraph):
    """
    Get all the active Batch norm ops for given Connected graph object.

    :param connected_graph: Connected graph.
    :return:
    """
    for conn_graph_op in connected_graph.get_all_ops().values():
        if conn_graph_op.type == 'FusedBatchNormV3':
            bn_conn_graph_op = conn_graph_op
            yield bn_conn_graph_op


def _get_bn_stats_and_params(session: tf.compat.v1.Session, bn_op: Op) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get BN stats (mean and variance) and params (gamma and beta).

    :param session: TensorFlow session
    :param bn_op: BN op.
    :return: (mean, variance, gamma, beta)
    """
    bn_stats_and_params = []
    bn_stats_and_params.extend(_get_bn_stats_var(session, bn_op))
    bn_stats_and_params.extend(_get_bn_params_var(session, bn_op))
    assert len(bn_stats_and_params) == 4, "Unable to get the BN stats and params for given BN op: %s" % bn_op.name
    with session.graph.as_default():
        mean, var, gamma, beta = session.run(bn_stats_and_params)
    return mean, var, gamma, beta

def _get_bn_stats_var(session: tf.compat.v1.Session, bn_conn_graph_op: Op) -> List[tf.Variable]:
    """
    Get BN statistics (mean, variance) variables for given batch norm op.

    :param session: TensorFlow session.
    :param bn_conn_graph_op: Batch norm op.
    :return: BN statistics (mean, variance)
    """
    moving_mean = str(bn_conn_graph_op.inputs[BNOpParamType.moving_mean]) + ':0'
    moving_var = str(bn_conn_graph_op.inputs[BNOpParamType.moving_variance]) + ':0'

    with session.graph.as_default():
        var = [var for var in tf.compat.v1.global_variables() if var.name == moving_mean or var.name == moving_var]
    assert len(var) == 2, "Unable to get the BN statics (mean, var) variables for given BN op: %s" % bn_conn_graph_op.name
    return var


def _get_bn_params_var(session: tf.compat.v1.Session, bn_conn_graph_op: Op) -> List[tf.Variable]:
    """
    Get BN params (weight, bias) variables for given batch norm op.

    :param session: TensorFlow session.
    :param bn_conn_graph_op: Batch norm op.
    :return: BN statistics
    """
    weight = str(bn_conn_graph_op.inputs[BNOpParamType.gamma]) + ':0'
    bias = str(bn_conn_graph_op.inputs[BNOpParamType.beta]) + ':0'

    with session.graph.as_default():
        var = [var for var in tf.compat.v1.global_variables() if var.name == weight or var.name == bias]
    assert len(var) == 2, "Unable to get the BN params (weight, bias) variables for given BN op: %s" % bn_conn_graph_op.name
    return var


def _get_bn_momentum_var(session: tf.compat.v1.Session, bn_conn_graph_op: Op) -> tf.Variable:
    """
    Get BN momentum variable for given batch norm op.

    NOTE: By default, momentum is compiled inside the graph.
     invoke modify_sess_bn_mutable() API before invoking this API.

    :param session: TensorFlow session.
    :param bn_conn_graph_op: Batch norm op.
    :return: BN statistics
    """
    momentum_var_name = bn_conn_graph_op.name + _BN_MOMENTUM_NAME + ':0'
    with session.graph.as_default():
        momentum_var = [var for var in tf.compat.v1.global_variables() if var.name == momentum_var_name]

    assert len(momentum_var) == 1, "Unable to get BN momentum variable, convert to variable using" \
                                   " modify_sess_bn_mutable() API."
    return momentum_var[0]


def _get_bn_is_training_var(session: tf.compat.v1.Session, bn_conn_graph_op: Op) -> tf.Variable:
    """
    Get BN is_training variable for given batch norm op.

    NOTE: invoke modify_sess_bn_mutable() API before invoking this API.

    :param session: TensorFlow session.
    :param bn_conn_graph_op: Batch norm op.
    :return: BN statistics
    """
    is_training_tensor = bn_conn_graph_op.internal_ops[1].inputs[0].op.inputs[0]

    with session.graph.as_default():
        is_training_var = [var for var in tf.compat.v1.global_variables() if var.name == is_training_tensor.name]

    assert len(is_training_var) == 1, "Unable to get BN is_training variable, convert to variable using" \
                                      " modify_sess_bn_mutable() API."
    return is_training_var[0]


def _set_var(session: tf.compat.v1.Session, var: tf.Variable, updated_value: float):
    """
    Set updated_value for given variable.

    :param session: TensorFlow session
    :param var: Variable to be updated.
    :param updated_value: Updated value.
    """
    assign_op = tf.compat.v1.assign(var, updated_value)
    session.run(assign_op)


def _get_bn_momentum(graph_def: tf.Graph, bn_conn_graph_op: Op) -> float:
    """
    Get the BN momentum from graph_def protobuf.

    NOTE: momentum is only needed during training and not during inference.

    :param graph_def: A protobuf containing the graph operations.
    :param bn_conn_graph_op: BN op.
    :return: momentum
    """
    momentum = None

    bn_op_name = bn_conn_graph_op.name.replace("/", "_") + '_cond_1_true'
    for func in graph_def.library.function:
        if func.signature.name.startswith(bn_op_name):
            momentum = func.node_def[0].attr['value'].tensor.float_val[0]
            break

    bn_op_name = bn_conn_graph_op.name + '/Const'
    for node in graph_def.node:
        if bn_op_name == node.name:
            momentum = node.attr['value'].tensor.float_val[0]
            break

    if momentum is None:
        momentum = _DEFAULT_BN_MOMENTUM
    return momentum


def _get_bn_epsilon(graph_def: tf.Graph, bn_conn_graph_op: Op) -> float:
    """
    Get the epsilon from graph_def protobuf.

    :param graph_def: A protobuf containing the graph operations.
    :param bn_conn_graph_op: BN op.
    :return: epsilon.
    """
    epsilon = None

    bn_op_name = bn_conn_graph_op.name.replace("/", "_") + '_cond_true'
    for func in graph_def.library.function:
        if func.signature.name.startswith(bn_op_name):
            epsilon = [node.attr['epsilon'].f for node in func.node_def if node.op == "FusedBatchNormV3"][0]
            break

    bn_op_name = bn_conn_graph_op.name + '/FusedBatchNormV3'
    for node in graph_def.node:
        if bn_op_name == node.name and node.op == "FusedBatchNormV3":
            epsilon = node.attr['epsilon'].f
            break

    if epsilon is None:
        epsilon = _DEFAULT_BN_EPSILON
    return epsilon
