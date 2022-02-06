# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" utilities for Relu op """

import tensorflow as tf
from aimet_tensorflow import graph_editor
from aimet_tensorflow.common.operation import Op
from aimet_common.utils import AimetLogger
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def replace_relu6_with_relu(sess: tf.compat.v1.Session, relu6_op: tf.Operation):
    """
    replaces existing Relu6 op with a Relu.
    :param sess : active tf.compat.v1.Session
    :param relu6_op: Relu6 op to be replaced with Relu
    :return:
    """
    with sess.graph.as_default():
        assert len(relu6_op.inputs) == 1
        new_tensor = tf.nn.relu(relu6_op.inputs[0])     # pylint: disable=no-member
        relu_op = new_tensor.op

        relu_outputs = list(relu_op.outputs)
        relu6_outputs = list(relu6_op.outputs)

        # swap the two tensors using reroute
        graph_editor.reroute_ts(ts0=relu_outputs,
                                ts1=relu6_outputs)

        graph_editor.detach_inputs(relu6_op)

def does_conv_have_relu_activation(input_op: Op)-> bool:
    """
    check if a given operation of type conv2d or depthwise conv2d
    has Relu activations or not.
    :param input_op: Conv or Depthwise conv 2D ops
    :return: True if Relu activation present - False otherwise.
    """

    if input_op.type not in ['Conv2D', 'DepthwiseConv2dNative']:
        raise ValueError('Op type: '+input_op.type+' is not CONV2D!')

    # conv --> bias add / add --> relu
    if (len(input_op.output.consumers) == 1) and \
        (input_op.output.consumers[0].type == 'Relu' or input_op.output.consumers[0].type == 'PReLU'):
        return True

    return False
