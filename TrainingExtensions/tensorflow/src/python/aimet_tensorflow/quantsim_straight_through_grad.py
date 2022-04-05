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

""" implements straight through graident computation for Quantize Op"""

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from aimet_tensorflow.utils.constants import QuantizeOpIndices


def _compute_dloss_by_dx(op, grad):
    x = tf.cast(op.inputs[0], tf.float32)
    encoding_min = tf.cast(op.inputs[int(QuantizeOpIndices.encoding_min)], tf.float32)
    encoding_max = tf.cast(op.inputs[int(QuantizeOpIndices.encoding_max)], tf.float32)
    op_mode = tf.cast(op.inputs[int(QuantizeOpIndices.op_mode)], tf.int8)

    inner_cond = tf.compat.v2.where(tf.less_equal(x, encoding_max),  # condition to check per value
                                    1.0,  # execute if true
                                    0.0)  # execute if false

    dloss_by_dx = (tf.compat.v2.where(tf.less_equal(encoding_min, x),  # condition to check per value
                                      inner_cond,  # execute if true
                                      0.0)) * grad

    # Pass through gradient for skipped ops
    dloss_by_dx = tf.cond(tf.equal(op_mode, 3), lambda: grad, lambda: dloss_by_dx)

    return dloss_by_dx


# by default we will have this registered for Qc Quantize op.
@tf_ops.RegisterGradient("QcQuantize")
def _qc_straight_through_estimator_grad(op, grad):
    # pylint: disable=unused-argument
    """
    straight through estimator logic used to compute gradient for Quantize Op.
    :param op: quantize op
    :param grad: gradient
    :return: gradients computed per input
    """

    dloss_by_dx = _compute_dloss_by_dx(op, grad)
    return dloss_by_dx, None, None, None, None, None, None, None


@tf_ops.RegisterGradient("QcQuantizeRecurrentParam")
def _qc_recurrent_param_straight_through_estimator_grad(op, grad):
    # pylint: disable=unused-argument
    """
    straight through estimator logic used to compute gradient for Quantize Op.
    :param op: quantize op
    :param grad: gradient
    :return: gradients computed per input
    """
    dloss_by_dx = _compute_dloss_by_dx(op, grad)

    return dloss_by_dx, None, None, None, None, None, None, None


def _get_n_and_p(bitwidth, use_symmetric_encoding):
    """
    compute bounds n and p params given bitwidth and use_symmetric_encoding flag
    :param bitwidth: tf tensor with bitwidth configured
    :param use_symmetric_encoding: boolean flag tf tensor that indicates symmetric/ asymmetric encoding
    :return: n and p params computed
    """

    bitwidth = tf.cast(bitwidth, tf.float32)
    two_pow_bw = tf.cast(tf.pow(tf.cast(tf.constant(2), tf.float32), bitwidth), tf.float32)
    two_pow_bw_minus_1 = tf.cast(tf.pow(tf.cast(tf.constant(2), tf.float32), (bitwidth - 1)), tf.float32)
    minus_one_as_float32 = tf.cast(tf.constant(-1), tf.float32)
    one_as_float32 = tf.cast(tf.constant(1), tf.float32)

    # symmetric case : n = (- 2 ** (bw -1 )) + 1 , p = (2 ** (bw -1) - 1)
    def n_symmetric_encoding():
        return tf.add(tf.multiply(minus_one_as_float32, two_pow_bw_minus_1), one_as_float32)

    def p_symmetric_encoding():
        return tf.subtract(two_pow_bw_minus_1, one_as_float32)

    # asymmetric case : n = 0 , p = 2 ** (bw) - 1
    def n_asymmetric_encoding():
        return tf.cast(0, tf.float32)

    def p_asymmetric_encoding():
        return tf.cast(two_pow_bw - 1, tf.float32)

    n = tf.cond(use_symmetric_encoding, n_symmetric_encoding, n_asymmetric_encoding)

    p = tf.cond(use_symmetric_encoding, p_symmetric_encoding, p_asymmetric_encoding)

    return n, p


def _compute_dloss_by_dmin_using_dmax(dloss_by_dmax):
    """
    compute derivative of loss w.r.t min, it is sign flipped version of derivative w.r.t max
    :param dq_by_dmax derivative w.r.t max
    :return: derivative w.r.t min
    """

    return tf.negative(dloss_by_dmax)


def _compute_dloss_by_dmax(x, grad, scaling, offset, bitwidth, use_symmetric_encoding):
    """
    helper function to compute derivative of loss w.r.t encoding max
    :param grad: gradient param
    :param scaling: scaling factor as tf tensor computed for given encoding min/max
    :param offset: offset computed as tf tensor
    :param bitwidth: bit-width as tf tensor
    :param use_symmetric_encoding: bool flag to indicate symmetric/asymmetric encoding
    :return: computed derivative w.r.t encoding max
    """

    steps = tf.cast(tf.pow(tf.cast(tf.constant(2), tf.float32), bitwidth) - 1, tf.float32)
    r_x_by_s_plus_round_o = tf.round(x / scaling) + tf.round(offset)
    # R(x/s)-(x/s)
    r_x_by_s_minus_x_by_s = tf.round(x / scaling) - (x / scaling)

    # compute dq_by_dmax
    # expr to be used if r_x_by_s_plus_round_o < n or > p
    n, p = _get_n_and_p(bitwidth, use_symmetric_encoding)
    false_expr = tf.multiply(tf.clip_by_value(r_x_by_s_plus_round_o, n, p), tf.cast(1 / steps, tf.float32))
    inner_cond = tf.where(tf.less_equal(r_x_by_s_plus_round_o, p), tf.multiply(r_x_by_s_minus_x_by_s,
                                                                               tf.cast(1 / steps, tf.float32)),
                          false_expr)

    # we need a scalar value for dq_by_dmax, so reduce 4d value computed above
    # to single value before returning gradient
    # this uses chain rule, multiply by loss and sum it to get scalar.
    dq_by_dmax = tf.where(tf.less_equal(n, r_x_by_s_plus_round_o), inner_cond, false_expr)

    # If per channel is active, scaling tensor will be rank 1 (an array instead of a singular value).
    # In case of per channel, we reduce by all but the last dimension. Otherwise, we reduce all dimensions.
    dloss_by_dmax = tf.cond(tf.equal(tf.rank(scaling), 0), lambda: tf.reduce_sum(dq_by_dmax * grad),
                            lambda: tf.reduce_sum(dq_by_dmax * grad, axis=tf.range(0, tf.rank(x) - 1)))
    return dloss_by_dmax


# pylint: disable=too-many-locals
def _compute_dloss_by_dmin_dmax_and_dx(inputs: tf.Tensor, bitwidth: tf.Tensor, op_mode: tf.Tensor,
                                       encoding_min: tf.Tensor, encoding_max: tf.Tensor, is_symmetric: tf.Tensor,
                                       grad: tf.Tensor):
    """
    Return tensors for dloss_by_dmin, dloss_by_dmax, and dloss_by_dx.
    :param inputs: Inputs to op
    :param bitwidth: Bitwidth used to quantize
    :param op_mode: Op mode (if passthrough, gradient is returned as is)
    :param encoding_min: Encoding min value(s), will be more than one if per channel is active
    :param encoding_max: Encoding max value(s), will be more than one if per channel is active
    :param is_symmetric: True if symmetric encodings are used, False otherwise
    :param grad: Gradient from child layer
    :return: Tensors for dloss_by_dmin, dloss_by_dmax, and dloss_by_dx
    """
    x = tf.cast(inputs, tf.float32)
    bitwidth = tf.cast(bitwidth, tf.float32)
    op_mode = tf.cast(op_mode, tf.int8)
    encoding_min = tf.cast(encoding_min, tf.float32)
    encoding_max = tf.cast(encoding_max, tf.float32)
    # handle min == max to avoid divide by zero
    epsilon = tf.constant(1e-5, dtype=tf.float32)
    encoding_max = tf.math.maximum(encoding_max, tf.add(encoding_min, epsilon))

    # compute n, p, scaling and offset params
    # choose n based on symmetric or asymmetric flag
    # symmetric : -two_pow_bw + 1
    # asymmetric : 0
    n, p = _get_n_and_p(bitwidth, is_symmetric)
    steps = tf.cast(tf.pow(tf.cast(tf.constant(2), tf.float32), bitwidth) - 1, tf.float32)
    scaling = tf.cast(((encoding_max - encoding_min) / steps), tf.float32)
    rounded_offset = tf.round(-encoding_min / scaling)  # pylint: disable=invalid-unary-operand-type
    # R(x/s) + R(o)
    r_x_by_s_plus_round_o = tf.round(x / scaling) + rounded_offset

    # compute dQ(x,s)/dx , dQ(x,s)/dmin and dQ(x,s)/dmax
    # compute dq_by_dx, dq_by_dmax, dq_by_dmin
    inner_cond = tf.where(tf.less_equal(r_x_by_s_plus_round_o, p),  # condition to check per value
                          tf.ones_like(r_x_by_s_plus_round_o),  # execute if true
                          tf.zeros_like(r_x_by_s_plus_round_o))  # execute if false
    dloss_by_dx = (tf.where(tf.less_equal(n, r_x_by_s_plus_round_o),  # condition to check per value
                            inner_cond,  # execute if true
                            tf.zeros_like(r_x_by_s_plus_round_o))) * grad

    dloss_by_dmax = tf.cast(_compute_dloss_by_dmax(x, grad, scaling, rounded_offset, bitwidth, is_symmetric),
                            tf.float64)
    dloss_by_dmin = tf.cast(_compute_dloss_by_dmin_using_dmax(dloss_by_dmax), tf.float64)

    # Pass through gradient for skipped ops
    dloss_by_dx = tf.cond(tf.equal(op_mode, 3), lambda: grad, lambda: dloss_by_dx)
    return dloss_by_dmin, dloss_by_dmax, dloss_by_dx


@tf_ops.RegisterGradient("QcQuantizeRangeLearningCustomGradient")
def quantsim_custom_grad_learned_grid(op, grad):
    """
    Performs custom gradient calculations for trained Quantize op
    :param op: Tf operation for which gradients are to be computed
    :param grad: Gradient flowing through
    """
    dloss_by_dmin, dloss_by_dmax, dloss_by_dx = \
        _compute_dloss_by_dmin_dmax_and_dx(op.inputs[0],
                                           op.inputs[int(QuantizeOpIndices.bit_width)],
                                           op.inputs[int(QuantizeOpIndices.op_mode)],
                                           op.inputs[int(QuantizeOpIndices.encoding_min)],
                                           op.inputs[int(QuantizeOpIndices.encoding_max)],
                                           op.inputs[int(QuantizeOpIndices.use_symmetric_encoding)],
                                           grad)
    return dloss_by_dx, None, None, dloss_by_dmin, dloss_by_dmax, None, None, None


@tf_ops.RegisterGradient("QcQuantizePerChannelRangeLearningCustomGradient")
def quantsim_per_channel_custom_grad_learned_grid(op, grad):
    """
    Performs custom gradient calculations for trained QcQuantizePerChannel op
    :param op: Tf operation for which gradients are to be computed
    :param grad: Gradient flowing through
    """
    dloss_by_dmin, dloss_by_dmax, dloss_by_dx = \
        _compute_dloss_by_dmin_dmax_and_dx(op.inputs[0],
                                           op.inputs[int(QuantizeOpIndices.bit_width)],
                                           op.inputs[int(QuantizeOpIndices.op_mode)],
                                           op.inputs[int(QuantizeOpIndices.encoding_min)],
                                           op.inputs[int(QuantizeOpIndices.encoding_max)],
                                           op.inputs[int(QuantizeOpIndices.use_symmetric_encoding)],
                                           grad)
    return dloss_by_dx, None, None, dloss_by_dmin, dloss_by_dmax, None, None, None, None
