# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" utilities for Simulation of recurrent models running on Quantized hardware """
from typing import List, Tuple

import tensorflow as tf
from aimet_common.utils import AimetLogger
from aimet_tensorflow.common import core
from aimet_tensorflow.utils.quantsim import get_op_input_indices, is_op_quantizable

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


# types of RNN Module that are supported
SUPPORTED_RECURRENT_TYPES = ['SimpleRNN', 'LSTM']

# op types inside SimpleRNN module for activation quantization
basic_recurrent_inner_op_types_to_quantize = {} #'Tanh'

# op types inside LSTM module for activation quantization
lstm_recurrent_inner_op_types_to_quantize = {} #'Tanh', 'sigmoid'


def _get_internal_ops_to_quantize_params_for(graph: tf.Graph, internal_ops: List[tf.Operation]) \
        -> Tuple[List[str], List[int]]:
    """
    Fetches op names with param input indices for ops with quantizable params
    :param graph: TensorFlow graph as tf.Graph
    :param internal_ops:list of TensorFlow ops within a module
    :return: Tuple consisting of list of op names with params to insert quantize ops for as well as list of indices
    of parameters for each op, within recurrent block
    """

    query = core.OpQuery(graph, ops_to_ignore=None)
    valid_tf_ops = [op for op in query.get_weight_ops(ops=internal_ops)]
    ops_with_param_names = set()
    for tf_op in valid_tf_ops:
        ops_with_param_names.add(tf_op.name)
    input_indices = []
    if ops_with_param_names:
        input_indices = get_op_input_indices(graph, ops_with_param_names)
    else:
        _logger.info("No ops with params detected in this recurrent module")
    return list(ops_with_param_names), input_indices


def _get_input_ops_to_quantize(internal_ops: List[tf.Operation]) -> List[str]:
    """
    Searches the names of input ops to MatMuls in recurrent layers
    :param internal_ops: List of internal ops of Recurrent Layer
    :return: names of unique input ops to quantize in a recurrent layer
    """
    input_op_names = set()
    for op in internal_ops:
        if op.type in['MatMul']:
            # get input op names
            input_op_names.add(op.inputs[0].op.name)

    return list(input_op_names)


def _get_internal_ops_to_quantize_activations_for(internal_ops: List[tf.Operation],
                                                  inner_op_types_to_quantize: List[str]) -> List[str]:
    """
    Get names of ops to insert activation quantizers for
    :param internal_ops: list of TensorFlow ops within a module
    :param inner_op_types_to_quantize: list of TensorFlow op types within recurrent module that are to be quantized
    This list is customizable per recurrent module type such as SimpleRNN, LSTM etc.
    :return: List of unique op names to insert activation quantize ops for within recurrent block
    """

    valid_tf_ops = [op for op in internal_ops if op.type in inner_op_types_to_quantize]
    # append only unique names
    op_names_to_quantize = set()
    for tf_op in valid_tf_ops:
        output_act_name = tf_op.outputs[0].op.name
        if is_op_quantizable(tf_op):
            op_names_to_quantize.add(output_act_name)

    # get inputs to recurrent MatMuls
    input_op_names = _get_input_ops_to_quantize(internal_ops)
    if input_op_names:
        op_names_to_quantize.update(input_op_names)

    return list(op_names_to_quantize)


# internal ops to be picked for quantization can be different per recurrent module type
def _select_simple_rnn_internal_ops_to_quantize(graph: tf.Graph, internal_ops: List[tf.Operation]) \
        -> Tuple[List[str], List[int], List[str]]:
    """
    selects params and activations from internal ops in a SimpleRNN module, for quantization
    :param graph: tf.Graph active TensorFlow graph
    :param internal_ops: internal ops in given recurrent module
    :return: Tuple[List[str], List[int], List[str]], selected param op names, input indices and activation op names
    """

    # Get list of ops with params to insert quantizers for, as well as the input indices to insert on.
    curr_module_ops_with_param_names, curr_module_input_indices = _get_internal_ops_to_quantize_params_for(graph,
                                                                                                           internal_ops)

    # Get list of activation ops to insert quantizers for
    curr_module_activation_op_names = _get_internal_ops_to_quantize_activations_for(
        internal_ops, basic_recurrent_inner_op_types_to_quantize)

    return curr_module_ops_with_param_names, curr_module_input_indices, curr_module_activation_op_names


def _select_lstm_internal_ops_to_quantize(graph: tf.Graph, internal_ops: List[tf.Operation]) \
        -> Tuple[List[str], List[int], List[str]]:
    """
    selects params and activations from internal ops in lstm module, for quantization
    :param graph: tf.Graph active TensorFlow graph
    :param internal_ops: internal ops in given recurrent module
    :return: Tuple[List[str], List[int], List[str]], selected param op names, input indices and activation op names
    """

    # Get list of ops with params to insert quantizers for, as well as the input indices to insert on.
    curr_module_ops_with_param_names, curr_module_input_indices = _get_internal_ops_to_quantize_params_for(graph,
                                                                                                           internal_ops)

    # Get list of activation ops to insert quantizers for, and the connected graph used to obtain these ops
    curr_module_activation_op_names = _get_internal_ops_to_quantize_activations_for(
        internal_ops, lstm_recurrent_inner_op_types_to_quantize)

    return curr_module_ops_with_param_names, curr_module_input_indices, curr_module_activation_op_names
