# /usr/bin/env python2.7
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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
Load a trained network and modify it to enable BFS. Add in beta switches after
the relu layers. Create a new solver which includes an architecture loss.
"""

from itertools import compress
import tensorflow as tf

from aimet_common.utils import AimetLogger

log = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def initialize_uninitialized_vars(sess):
    """
    Some graphs have variables created after training that need to be initialized (eg SVD/Quantization).
    However, in pre-trained graphs we don't want to reinitialize variables that are already
    which would overwrite the values obtained during training. Therefore search for all
    uninitialized variables and initialize ONLY those variables.
    :param sess: tf.compat.v1.Session
    :return:
    """
    with sess.graph.as_default():
        global_vars = tf.compat.v1.global_variables()
        is_not_initialized = sess.run([~(tf.compat.v1.is_variable_initialized(var)) for var in global_vars])
        uninitialized_vars = list(compress(global_vars, is_not_initialized))

        if uninitialized_vars:
            log.info('Initializing uninitialized variables')
            sess.run(tf.compat.v1.variables_initializer(uninitialized_vars))


def default_eval_func(data):
    """
    Evaluates the graph for accuracy. Returns the accuracy based on the current
    data iteration. The default "accuracy" should always be the first entry in the list
    provided to eval_names.
    :param data:
    :return:
    """
    if len(data) > 1:
        print('default evaluation function only expected 1 output, accuracy. Using first datum')

    # Return the accuracy
    return data[0][1]


def evaluate_graph(session, generator, eval_names, eval_func, iterations):
    """
    Evaluates the graph's performance by running data through the network
    and calling an evaluation function to generate the performance metric.
    :param session: The tensorflow session that contains the graph
    :param generator: The data generator providing the network with batch data
    :param eval_names: The names providing the nodes on which the network's performance should be judged
    :param eval_func: The customized function to evaluate the performance of the network
    :param iterations: The number of iterations (batches) to run through the network
    :return:
    """

    # Ensure any uninitialized variables are initialized
    initialize_uninitialized_vars(session)

    # Get the first batch and ue it to create the tensor map
    t_map = _create_map_of_input_tensors(generator, session)

    eval_outputs = []
    for name in eval_names:
        op = session.graph.get_operation_by_name(name)
        eval_outputs.append(op.outputs[0])

    # Run the graph and verify the data is being updated properly for each iteration
    avg_metric = 0
    log.info("Evaluating graph for %i iterations", iterations)
    for _, batch in zip(range(iterations), generator):
        # Setup the feed dictionary
        feed_dict = {}
        for name, data in batch.items():
            feed_dict[t_map[name]] = data

        output_data = session.run(eval_outputs, feed_dict=feed_dict)
        avg_metric += eval_func(list(zip(eval_names, output_data)))

    log.info("Completed graph evaluation for %i iterations", iterations)
    return avg_metric / iterations


def _create_map_of_input_tensors(generator, session):
    t_map = {}
    inputs = generator.get_data_inputs() + generator.get_validation_inputs()
    for name in inputs:
        t_map[name] = session.graph.get_tensor_by_name(name + ':0')
    return t_map
