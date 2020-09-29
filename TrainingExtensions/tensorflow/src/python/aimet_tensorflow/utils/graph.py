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
""" utilities for tf graph related operations """

import tensorflow as tf


def op_not_in_loop_control_flow_context(graph: tf.Graph, input_op: tf.Operation) -> bool:
    """
    checks if the  op is not in loop control flow context or not
    :param graph: tf.Graph is the active graph
    :param input_op: op as tf.Operation
    :return: True if op is not in a loop control flow context, False otherwise.
    """
    # pylint: disable=protected-access
    active_ctxt = graph._get_control_flow_context()
    input_ctxt = input_op._get_control_flow_context()

    if not input_ctxt or input_ctxt is active_ctxt:
        # input_op isn't in 'a' loop control flow context or
        # input_op is in the same context as op.
        return True

    return False


def updated_graph_flow_context_to_loop_context(graph: tf.Graph, preceeding_tensor: tf.Tensor):
    """
    updates graph flow context to loop context
    :param graph: TensorFlow Graph (tf.Graph)
    :param preceeding_tensor: TF tensor that feeds into the op which needs modification
    :return: old graph context object
    """

    # pylint: disable=protected-access
    old_graph_context = graph._get_control_flow_context()
    graph._set_control_flow_context(preceeding_tensor.op._get_control_flow_context())

    return old_graph_context


def set_graph_flow_context(graph: tf.Graph, active_context):
    """
    sets graph context to active context provided
    :param graph: TensorFlow Graph (tf.Graph)
    :param active_context: context object to be set as current graph's context
    :return:
    """

    # pylint: disable=protected-access
    graph._set_control_flow_context(active_context)
