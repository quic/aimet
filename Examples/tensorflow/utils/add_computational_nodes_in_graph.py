# !/usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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


"""
Adds labels placeholder and accuracy, loss ops into a graph of a TF session
"""
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def add_image_net_computational_nodes_in_graph(session: tf.Session, logits: tf.python.ops.Tensor, num_classes: int):
    """
    :param session: Tensorflow session to operate on
    :param logits: Output tensor of session graph
    :param num_classes: No of classes in model data
    """
    with session.graph.as_default():
        # predicted value of the model
        y_hat = logits
        y_hat_argmax = tf.argmax(y_hat, axis=1)

        # place holder for the labels
        y = tf.placeholder(tf.int64, shape=[None, num_classes], name='labels')
        y_argmax = tf.argmax(y, axis=1)

        # prediction Op
        correct_prediction = tf.equal(y_hat_argmax, y_argmax)

        # pylint: disable-msg=unused-variable
        # accuracy Op: top1
        top1_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='top1-acc')

        # accuracy Op: top5
        top5_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=y_hat,
                                                         targets=tf.cast(y_argmax, tf.int32),
                                                         k=5),
                                          tf.float32),
                                  name='top5-acc')

        # loss Op: loss
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_hat))
