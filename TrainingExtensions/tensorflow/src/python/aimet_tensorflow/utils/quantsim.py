# /usr/bin/env python3.5
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
""" utilities for quantsim """

from typing import List

import tensorflow as tf

from aimet_common.utils import AimetLogger
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.common.operation import Op
from aimet_tensorflow.quantsim_config.quantsim_config import OpToQuantOpsDictType

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

def create_op_to_quant_ops_dict(graph: tf.Graph, conn_graph: ConnectedGraph, ops_with_param_names: List[str],
                                indices: List[int], activation_op_names: List[str]) -> OpToQuantOpsDictType:
    """
    Create an op to quant ops dictionary mapping connected graph ops to a list consisting of the activation quantizer
    and a dictionary mapping param type string to param quantizers.
    :param graph: Tensorflow graph containing inserted quantizers
    :param conn_graph: Connected graph of the original unquantized model
    :param ops_with_param_names: List of tf operation names for which parameter quantizers were inserted for
    :param indices: Indices of tf operations of which parameter quantizers were inserted for
    :param activation_op_names: List of tf operation names for which activation quantizers were inserted for
    :return: Dictionary mapping connected graph ops to a list consisting of the activation quantizer and a dictionary
    mapping param type string to param quantizers.
    """

    op_to_quant_ops_dict = {}
    for op_with_param_name, index in zip(ops_with_param_names, indices):
        op_with_param = graph.get_operation_by_name(op_with_param_name)
        conn_graph_op = conn_graph.get_op_from_module_name(op_with_param_name)
        param_type = 'weight'
        if op_with_param.type == 'BiasAdd':
            param_type = 'bias'
        param_quantizer = op_with_param.inputs[index].op
        assert param_quantizer.type == 'QcQuantize'
        add_op_to_quant_ops_dict_entry(param_quantizer, conn_graph_op, True, param_type, op_to_quant_ops_dict)
    for activation_op_name in activation_op_names:
        activation_op = graph.get_operation_by_name(activation_op_name)
        conn_graph_op = conn_graph.get_op_from_module_name(activation_op_name)
        activation_quantizer = \
            [consumer for consumer in activation_op.outputs[0].consumers() if consumer.type == 'QcQuantize']
        if len(activation_quantizer) != 1:
            _logger.error('Expected one activation quantizer but found %s', len(activation_quantizer))
            raise AssertionError
        add_op_to_quant_ops_dict_entry(activation_quantizer[0], conn_graph_op, False, '', op_to_quant_ops_dict)
    return op_to_quant_ops_dict


def add_op_to_quant_ops_dict_entry(qc_quantize_op: tf.Operation, conn_graph_op: Op, is_param: bool, param_type: str,
                                   op_to_quant_ops_dict: OpToQuantOpsDictType):
    """
    Add an entry to the op_to_quant_ops_dict
    :param qc_quantize_op: Qc quantize op to add to the dictionary
    :param conn_graph_op: Connected graph Op associated with the qc quantize op
    :param is_param: True if the qc quantize op was created for a parameter, False otherwise
    :param param_type: Type of parameter (unused for activation quantizers)
    :param op_to_quant_ops_dict: Dictionary mapping connected graph op to a two item list consisting of a dictionary
    of param types to param qc quantize ops, and activation qc quantize op
    """
    if is_param:
        if conn_graph_op in op_to_quant_ops_dict:
            param_quant_op_dict, _ = op_to_quant_ops_dict[conn_graph_op]
            if param_type in param_quant_op_dict:
                param_quant_op_dict[param_type].add(qc_quantize_op)
            else:
                param_quant_op_dict[param_type] = {qc_quantize_op}
        else:
            param_quant_op_dict = {param_type: {qc_quantize_op}}
            op_to_quant_ops_dict[conn_graph_op] = [param_quant_op_dict, None]
    else:
        if conn_graph_op in op_to_quant_ops_dict:
            op_to_quant_ops_dict[conn_graph_op][1] = qc_quantize_op
        else:
            op_to_quant_ops_dict[conn_graph_op] = [dict(), qc_quantize_op]
