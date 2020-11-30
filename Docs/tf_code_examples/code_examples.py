# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=missing-docstring
""" These are code examples to be used when generating AIMET documentation via Sphinx """

from decimal import Decimal

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import VGG16

# Compression-related imports
from aimet_common.defs import GreedySelectionParameters
from aimet_common.defs import CostMetric, CompressionScheme
from aimet_tensorflow.defs import SpatialSvdParameters, ChannelPruningParameters, ModuleCompRatioPair
from aimet_tensorflow.compress import ModelCompressor


def evaluate_model(sess: tf.compat.v1.Session, eval_iterations: int, use_cuda: bool) -> float:
    """
    This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's eval function does not
    match this signature, please create a simple wrapper.

    Note: Honoring the number of iterations is not absolutely necessary.
    However if all evaluations run over an entire epoch of validation data,
    the runtime for AIMET compression will obviously be higher.

    :param sess: Tensorflow session
    :param eval_iterations: Number of iterations to use for evaluation.
            None for entire epoch.
    :param use_cuda: If true, evaluate using gpu acceleration
    :return: single float number (accuracy) representing model's performance
    """

    # Evaluate model should run data through the model and return an accuracy score.
    # If the model does not have nodes to measure accuracy, they will need to be added to the graph.
    return .5


def spatial_svd_auto_mode():

    sess = tf.compat.v1.Session()
    # Construct graph
    with sess.graph.as_default():
        _ = VGG16(weights=None, input_shape=(224, 224, 3))
        init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # ignore first Conv2D op
    conv2d = sess.graph.get_operation_by_name('block1_conv1/Conv2D')
    modules_to_ignore = [conv2d]

    greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                              num_comp_ratio_candidates=10,
                                              use_monotonic_fit=True,
                                              saved_eval_scores_dict=None)

    auto_params = SpatialSvdParameters.AutoModeParams(greedy_select_params=greedy_params,
                                                      modules_to_ignore=modules_to_ignore)

    params = SpatialSvdParameters(input_op_names=['input_1'], output_op_names=['predictions/Softmax'],
                                  mode=SpatialSvdParameters.Mode.auto, params=auto_params, multiplicity=8)
    input_shape = (1, 3, 224, 224)

    # Single call to compress the model
    compr_model_sess, stats = ModelCompressor.compress_model(sess=sess,
                                                             working_dir=str('./'),
                                                             eval_callback=evaluate_model,
                                                             eval_iterations=10,
                                                             input_shape=input_shape,
                                                             compress_scheme=CompressionScheme.spatial_svd,
                                                             cost_metric=CostMetric.mac,
                                                             parameters=params,
                                                             trainer=None)

    print(stats)    # Stats object can be pretty-printed easily


def spatial_svd_manual_mode():

    sess = tf.compat.v1.Session()
    # Construct graph
    with sess.graph.as_default():
        _ = VGG16(weights=None, input_shape=(224, 224, 3))
        init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # Pick two convs to compress as examples
    conv2d = sess.graph.get_operation_by_name('block1_conv1/Conv2D')
    conv2d_1 = sess.graph.get_operation_by_name('block1_conv2/Conv2D')

    # Specify the necessary parameters
    manual_params = SpatialSvdParameters.ManualModeParams([ModuleCompRatioPair(module=conv2d, comp_ratio=0.5),
                                                           ModuleCompRatioPair(module=conv2d_1, comp_ratio=0.4)])

    params = SpatialSvdParameters(input_op_names=['input_1'], output_op_names=['predictions/Softmax'],
                                  mode=SpatialSvdParameters.Mode.manual, params=manual_params)

    input_shape = (1, 3, 224, 224)

    # Single call to compress the model
    compr_model_sess, stats = ModelCompressor.compress_model(sess=sess,
                                                             working_dir=str('./'),
                                                             eval_callback=evaluate_model,
                                                             eval_iterations=10,
                                                             input_shape=input_shape,
                                                             compress_scheme=CompressionScheme.spatial_svd,
                                                             cost_metric=CostMetric.mac,
                                                             parameters=params,
                                                             trainer=None)

    print(stats)    # Stats object can be pretty-printed easily


def channel_pruning_auto_mode():

    sess = tf.compat.v1.Session()
    # Construct graph
    with sess.graph.as_default():
        _ = VGG16(weights=None, input_shape=(224, 224, 3))
        init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # ignore first Conv2D op
    conv2d = sess.graph.get_operation_by_name('block1_conv1/Conv2D')
    modules_to_ignore = [conv2d]

    greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                              num_comp_ratio_candidates=2,
                                              use_monotonic_fit=True,
                                              saved_eval_scores_dict=None)

    auto_params = ChannelPruningParameters.AutoModeParams(greedy_select_params=greedy_params,
                                                          modules_to_ignore=modules_to_ignore)

    # Create random dataset
    batch_size = 1
    input_data = np.random.rand(100, 224, 224, 3)
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)

    params = ChannelPruningParameters(input_op_names=['input_1'],
                                      output_op_names=['predictions/Softmax'],
                                      data_set=dataset,
                                      batch_size=32,
                                      num_reconstruction_samples=50,
                                      allow_custom_downsample_ops=False,
                                      mode=ChannelPruningParameters.Mode.auto,
                                      params=auto_params,
                                      multiplicity=8)

    # Single call to compress the model
    results = ModelCompressor.compress_model(sess,
                                             working_dir=None,
                                             eval_callback=evaluate_model,
                                             eval_iterations=10,
                                             input_shape=(32, 224, 224, 3),
                                             compress_scheme=CompressionScheme.channel_pruning,
                                             cost_metric=CostMetric.mac,
                                             parameters=params)

    compressed_model, stats = results
    print(compressed_model)
    print(stats)     # Stats object can be pretty-printed easily


def channel_pruning_manual_mode():

    sess = tf.compat.v1.Session()

    # Construct graph
    with sess.graph.as_default():
        _ = VGG16(weights=None, input_shape=(224, 224, 3))
        init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # Create random dataset
    batch_size = 1
    input_data = np.random.rand(100, 224, 224, 3)
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)

    #  Pick two convs to compress as examples
    block1_conv2_op = sess.graph.get_operation_by_name('block1_conv2/Conv2D')
    block2_conv2_op = sess.graph.get_operation_by_name('block2_conv2/Conv2D')

    list_of_module_comp_ratio_pairs = [ModuleCompRatioPair(block1_conv2_op, 0.5),
                                       ModuleCompRatioPair(block2_conv2_op, 0.5)]

    manual_params = ChannelPruningParameters.ManualModeParams(list_of_module_comp_ratio_pairs=
                                                              list_of_module_comp_ratio_pairs)

    params = ChannelPruningParameters(input_op_names=['input_1'],
                                      output_op_names=['predictions/Softmax'],
                                      data_set=dataset,
                                      batch_size=32,
                                      num_reconstruction_samples=50,
                                      allow_custom_downsample_ops=False,
                                      mode=ChannelPruningParameters.Mode.
                                      manual,
                                      params=manual_params,
                                      multiplicity=8)

    # Single call to compress the model
    results = ModelCompressor.compress_model(sess,
                                             working_dir=None,
                                             eval_callback=evaluate_model,
                                             eval_iterations=10,
                                             input_shape=(32, 224, 224, 3),
                                             compress_scheme=CompressionScheme.channel_pruning,
                                             cost_metric=CostMetric.mac,
                                             parameters=params)

    compressed_model, stats = results
    print(compressed_model)
    print(stats)     # Stats object can be pretty-printed easily


if __name__ == '__main__':
    spatial_svd_auto_mode()
    spatial_svd_manual_mode()
    channel_pruning_auto_mode()
    channel_pruning_manual_mode()
