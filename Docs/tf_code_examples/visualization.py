# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

# Quantization visualization imports
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from aimet_tensorflow import plotting_utils
# End of import statements

def visualizing_weight_ranges_for_single_layer():
    # load a model
    tf.keras.backend.clear_session()
    _ = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
    sess = tf.compat.v1.keras.backend.get_session()

    results_dir = 'artifacts'

    with sess.as_default():
        # Getting a layer for visualizaing its weight ranges
        conv_op = sess.graph.get_operation_by_name('conv1_conv/Conv2D')

        plotting_utils.visualize_weight_ranges_single_layer(sess=sess, layer=conv_op, results_dir=results_dir)
    sess.close()


def visualizing_relative_weight_ranges_for_single_layer():
    # load a model
    tf.keras.backend.clear_session()
    _ = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
    sess = tf.compat.v1.keras.backend.get_session()

    results_dir = 'artifacts'

    with sess.as_default():
        # Getting a layer for visualizaing its weight ranges
        conv_op = sess.graph.get_operation_by_name('conv1_conv/Conv2D')

        plotting_utils.visualize_relative_weight_ranges_single_layer(sess=sess, layer=conv_op,
                                                                     results_dir=results_dir)
    sess.close()