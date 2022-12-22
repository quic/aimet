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
from typing import List, Union
import tensorflow as tf
from aimet_tensorflow import graph_editor
from aimet_tensorflow.utils.op.fusedbatchnorm import BNUtils
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.batch_norm_fold import find_all_batch_norms_to_fold

_DEFAULT_TF_BN_MOMENTUM = 0.99

def modify_model_bn_mutable(model: tf.keras.Model):
    """
    Utilities to modify batchnorm layer's momentum of keras model as mutable tf.Variable

    :param model: keras model to modify batchnorms
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            momentum = layer.momentum
            bn_momentum_var = tf.Variable(momentum, trainable=False, name=layer.name + "/momentum_mutable")
            layer.momentum = bn_momentum_var


# pylint: disable=too-many-locals
def modify_sess_bn_mutable(sess: tf.compat.v1.Session, start_op_names: Union[List[str], str],
                           output_op_names: Union[List[str], str], training_tf_placeholder: bool = True):
    """
    Utilities to modify batchnorm layer's momentum and training argument of tf session model as tf.Variable and/or tf.placeholder
    :param sess: active tf.compat.v1.Session
    :param start_op_names: Name of the starting op in the given graph or a list of names in case of multi-input model
    :param output_op_names: List of output op names of the model, used to help ConnectedGraph determine valid ops
           (to ignore training ops for example).  If None, all ops in the model are considered valid.
    :param training_tf_placeholder: Use tf.placeholder as training arg when set to True, else use tf.Variable

    """
    bn_conv_linear_pairs = find_all_batch_norms_to_fold(sess, start_op_names, output_op_names, return_bn_conn_op=True)
    with sess.graph.as_default():
        if training_tf_placeholder:
            bn_training = tf.compat.v1.placeholder_with_default(False, shape=[], name='bn_training_placeholder')
        else:
            bn_training = tf.compat.v1.Variable(tf.compat.v1.constant(False), name='bn_training_var')

        for pair in bn_conv_linear_pairs:
            batchnorm = pair[1]
            modified_name = batchnorm.name + '_modified'
            batchnorm_tensor = batchnorm.get_tf_op_with_io_tensor()

            beta_read_var = BNUtils.get_beta_read_var_op_tensor(sess.graph, batchnorm_tensor.op)
            gamma_read_var = BNUtils.get_gamma_read_var_op_tensor(sess.graph, batchnorm_tensor.op)
            mean_read_var = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, batchnorm_tensor.op)
            var_read_var = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, batchnorm_tensor.op)

            beta, gamma, mean, var = sess.run([beta_read_var, gamma_read_var, mean_read_var, var_read_var])

            beta_init = tf.compat.v1.constant_initializer(beta, dtype=tf.float32, verify_shape=True)
            gamma_init = tf.compat.v1.constant_initializer(gamma, dtype=tf.float32, verify_shape=True)
            mean_init = tf.compat.v1.constant_initializer(mean, dtype=tf.float32, verify_shape=True)
            var_init = tf.compat.v1.constant_initializer(var, dtype=tf.float32, verify_shape=True)
            momentum = tf.Variable(_DEFAULT_TF_BN_MOMENTUM, trainable=False, name=modified_name + "/momentum_mutable")
            new_bn = tf.compat.v1.layers.batch_normalization(batchnorm_tensor.in_tensor, beta_initializer=beta_init,
                                                             gamma_initializer=gamma_init,
                                                             moving_mean_initializer=mean_init,
                                                             moving_variance_initializer=var_init,
                                                             name=modified_name, momentum=momentum, training=bn_training,
                                                             fused=True)

            graph_editor.reroute_ts(ts0=[new_bn], ts1=batchnorm_tensor.out_tensor)
            graph_editor.detach_inputs(batchnorm_tensor.op)

    initialize_uninitialized_vars(sess)
