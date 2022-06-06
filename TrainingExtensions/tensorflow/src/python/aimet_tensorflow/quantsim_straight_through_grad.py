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
from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops

from aimet_tensorflow.defs import AxisHandling
from aimet_tensorflow.utils.constants import QuantizeOpIndices


@dataclass
class LearnedGridParams:
    """
    Data carrier containing parameters for learned grid
    """
    scaling: tf.Tensor
    offset: tf.Tensor
    n: tf.Tensor
    p: tf.Tensor


@dataclass
class IntermediateResultForLearnedGrid:
    """
    Data carrier containing intermediate result for learned grid backward computation

    forward_result: Round(x / scaling) + Round(offset)
    rounding_error_q: Round(x / scaling) - (x / scaling)
    rounding_error_o: Round(offset) - offset
    """
    forward_result: tf.Tensor
    rounding_error_q: tf.Tensor
    rounding_error_o: tf.Tensor


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

    dloss_by_dx = tf.cond(op.inputs[int(QuantizeOpIndices.is_int_data_type)], lambda: _compute_dloss_by_dx(op, grad),
                          lambda: grad)
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


def compute_intermediate_result_for_learned_grid(x: tf.Tensor,
                                                 scaling: tf.Tensor,
                                                 offset: tf.Tensor) -> IntermediateResultForLearnedGrid:
    """
    helper function to compute forward result and rounding error before derivative
    :param x: input
    :param scaling: scaling factor computed for given encoding min/max
    :param offset: offset computed
    :return: forward result, rounding error of quantizer, rounding error of offset tuple
    """
    forward_result = tf.round(x / scaling) + tf.round(offset)
    rounding_error_q = tf.round(x / scaling) - (x / scaling)
    rounding_error_o = tf.ones_like(x) * (tf.round(offset) - offset)

    return IntermediateResultForLearnedGrid(forward_result, rounding_error_q, rounding_error_o)


def _compute_derivative_of_loss_function(x: tf.Tensor,
                                         derivative_of_quantizer: tf.Tensor,
                                         grad: tf.Tensor,
                                         scaling: tf.Tensor) -> tf.Tensor:
    """
    Compute derivative of the loss function like dloss_by_dmin or dloss_by_dmax

    :param x: input
    :param derivative_of_quantizer: derivative of the quantizer function like dq_by_dmin or dq_by_dmax
    :param grad: gradient
    :param scaling: scaling factor computed for given encoding min/max
    :return: computed derivative of loss w.r.t derivative of quantizer
    """

    # If per channel is active, scaling tensor will be rank 1 (an array instead of a singular value).
    # In case of per channel, we reduce by all but the last dimension. Otherwise, we reduce all dimensions.
    derivative_of_loss_function = tf.cond(tf.equal(tf.rank(scaling), 0),
                                          lambda: tf.reduce_sum(derivative_of_quantizer * grad),
                                          lambda: tf.reduce_sum(derivative_of_quantizer * grad,
                                                                axis=tf.range(0, tf.rank(x) - 1)))
    return derivative_of_loss_function


def _compute_dloss_by_dmin(x: tf.Tensor,
                           grad: tf.Tensor,
                           intermediate_result: IntermediateResultForLearnedGrid,
                           grid_params: LearnedGridParams) -> tf.Tensor:
    """
    helper function to compute derivative of loss w.r.t encoding min
    Implementation based on LSQ+ ( https://arxiv.org/pdf/2004.09576.pdf )

    Inner condition ( n <= fw <= p ):
        dq_by_dmin = (round(x/s) - x/s) / -p
    Outer condition ( fw < n ):
        dq_by_dmin = -n/p + 1 + (round(o) - o)/p
    Outer condition ( p < fw ):
        dq_by_dmin = (round(o) - o)/p

    :param x: input
    :param grad: gradient
    :param intermediate_result: data carrier containing intermediate result (forward result, rounding error q and o)
    :param grid_params: data carrier containing parameters for learned grid (scale, offset, n, p)
    :return: computed derivative of loss w.r.t encoding min
    """
    scaling, _, n, p = grid_params.scaling, grid_params.offset, grid_params.n, grid_params.p
    forward_result = intermediate_result.forward_result
    rounding_error_q = intermediate_result.rounding_error_q
    rounding_error_o = intermediate_result.rounding_error_o

    dq_by_dmin = tf.where(tf.less_equal(forward_result, p),
                          -rounding_error_q / p, rounding_error_o / p)
    dq_by_dmin = tf.where(tf.less_equal(n, forward_result),
                          dq_by_dmin, -n/p + 1 + rounding_error_o / p)

    dloss_by_dmin = _compute_derivative_of_loss_function(x, dq_by_dmin, grad, scaling)
    return dloss_by_dmin


def _compute_dloss_by_dmax(x: tf.Tensor,
                           grad: tf.Tensor,
                           intermediate_result: IntermediateResultForLearnedGrid,
                           grid_params: LearnedGridParams) -> tf.Tensor:
    """
    helper function to compute derivative of loss w.r.t encoding max
    Implementation based on LSQ+ ( https://arxiv.org/pdf/2004.09576.pdf )

    Inner condition ( n <= fw <= p ):
        dq_by_dmax = (round(x/s) - x/s) / p
    Outer condition ( fw < n ):
        dq_by_dmax = n/p - (round(o) - o)/p
    Outer condition ( p < fw ):
        dq_by_dmax = 1 - (round(o) - o)/p

    :param x: input
    :param grad: gradient
    :param intermediate_result: data carrier containing intermediate result tensors (forward result, rounding errors)
    :param grid_params: data carrier containing parameters for learned grid (scale, offset, n, p)
    :return: computed derivative of loss w.r.t encoding max
    """
    scaling, _, n, p = grid_params.scaling, grid_params.offset, grid_params.n, grid_params.p
    forward_result = intermediate_result.forward_result
    rounding_error_q = intermediate_result.rounding_error_q
    rounding_error_o = intermediate_result.rounding_error_o

    dq_by_dmax = tf.where(tf.less_equal(forward_result, p),
                          rounding_error_q / p, tf.ones_like(p) - rounding_error_o / p)
    dq_by_dmax = tf.where(tf.less_equal(n, forward_result),
                          dq_by_dmax, n / p - rounding_error_o / p)

    dloss_by_dmax = _compute_derivative_of_loss_function(x, dq_by_dmax, grad, scaling)
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

    grid_params = LearnedGridParams(scaling, rounded_offset, n, p)
    intermediate_result = compute_intermediate_result_for_learned_grid(x, scaling, rounded_offset)
    dloss_by_dmax = tf.cast(_compute_dloss_by_dmax(x, grad, intermediate_result, grid_params), tf.float64)
    dloss_by_dmin = tf.cast(_compute_dloss_by_dmin(x, grad, intermediate_result, grid_params), tf.float64)

    # Pass through gradient for skipped ops
    dloss_by_dx = tf.cond(tf.equal(op_mode, 3), lambda: grad, lambda: dloss_by_dx)
    return dloss_by_dmin, dloss_by_dmax, dloss_by_dx


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
def _compute_dloss_by_dmin_dmax_and_dx_for_per_channel(inputs: tf.Tensor, bitwidth: tf.Tensor, op_mode: tf.Tensor,
                                                       encoding_min: tf.Tensor, encoding_max: tf.Tensor,
                                                       is_symmetric: tf.Tensor, is_int_data_type: tf.Tensor,
                                                       axis_handling: tf.Tensor, grad: tf.Tensor) -> \
                                                       Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Return tensors for dloss_by_dmin, dloss_by_dmax, and dloss_by_dx in the case of per channel.
    :param inputs: Inputs to op
    :param bitwidth: Bitwidth used to quantize
    :param op_mode: Op mode (if passthrough, gradient is returned as is)
    :param encoding_min: Encoding min value(s), will be more than one if per channel is active
    :param encoding_max: Encoding max value(s), will be more than one if per channel is active
    :param is_symmetric: True if symmetric encodings are used, False otherwise
    :param is_int_data_type: True if op needs to operate with int data type, else False
    :param axis_handling: Determines behavior for reshaping inputs and gradients based on axis handling value.
    :param grad: Gradient from child layer
    :return: Tensors for dloss_by_dmin, dloss_by_dmax, and dloss_by_dx
    """
    @tf.function
    def reshape_input_and_grad_for_axis_handling(inputs, grad, axis_handling) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Reshape input and grad tensors from (H, W, channels, depth multiplier) to (H, W, channels * depth multiplier) in
        the case of axis_handling = LAST_TWO_AXES to get all channel elements in last dimension only.
        :param inputs: inputs to reshape
        :param grad: gradient to reshape
        :param axis_handling: Axis handling to determine reshape behavior
        :return: reshaped inputs and grad tensors
        """
        if tf.equal(axis_handling, tf.constant([AxisHandling.LAST_TWO_AXES.value])):
            # Even when in the case of inputs being a bias tensor, and axis handling will not be LAST_TWO_AXES, TF will
            # still execute both paths of the conditional branch to construct the graph. When doing so, if there are not
            # 4 dimensions to the tensor, the below code will fail, even though during session run we would not be going
            # down this path.
            # To fix this, add 3 dummy dimensions to the left side dimensions of the tensor such that we are guaranteed
            # to have at least 4 dimensions. Then continue with taking the rightmost 4 dimensions for the shape to
            # reshape to.
            inputs = tf.expand_dims(inputs, axis=0)
            inputs = tf.expand_dims(inputs, axis=0)
            inputs = tf.expand_dims(inputs, axis=0)
            orig_shape = tf.shape(inputs)
            inputs = tf.reshape(inputs, [orig_shape[-4], orig_shape[-3], orig_shape[-2] * orig_shape[-1]])
            grad = tf.reshape(grad, [orig_shape[-4], orig_shape[-3], orig_shape[-2] * orig_shape[-1]])
        return inputs, grad

    @tf.function
    def reshape_dloss_by_dx_for_axis_handling(inputs, dloss_by_dx, axis_handling) -> tf.Tensor:
        """
        Reshape dloss_by_dx tensor from (H, W, channels * depth multiplier) to (H, W, channels, depth multiplier) in
        the case of axis_handling = LAST_TWO_AXES to match shape with that of the weight tensor to update.
        :param inputs: inputs tensor to get original shape from
        :param dloss_by_dx: dloss_by_dx tensor to reshape
        :param axis_handling: Axis handling to determine reshape behavior
        :return: reshaped dloss_by_dx tensor
        """
        if tf.equal(axis_handling, tf.constant([AxisHandling.LAST_TWO_AXES.value])):
            # Even when in the case of inputs being a bias tensor, and axis handling will not be LAST_TWO_AXES, TF will
            # still execute both paths of the conditional branch to construct the graph. When doing so, if there are not
            # 4 dimensions to the tensor, the below code will fail, even though during session run we would not be going
            # down this path.
            # To fix this, add 3 dummy dimensions to the left side dimensions of the tensor such that we are guaranteed
            # to have at least 4 dimensions. Then continue with taking the rightmost 4 dimensions for the shape to
            # reshape to.
            inputs = tf.expand_dims(inputs, axis=0)
            inputs = tf.expand_dims(inputs, axis=0)
            inputs = tf.expand_dims(inputs, axis=0)
            orig_shape = tf.shape(inputs)
            dloss_by_dx = tf.reshape(dloss_by_dx, [orig_shape[-4], orig_shape[-3], orig_shape[-2], orig_shape[-1]])
        return dloss_by_dx

    reshaped_inputs, grad = reshape_input_and_grad_for_axis_handling(inputs, grad, axis_handling)
    dloss_by_dmin, dloss_by_dmax, dloss_by_dx = \
        _compute_dloss_by_dmin_dmax_and_dx(reshaped_inputs, bitwidth, op_mode, encoding_min, encoding_max, is_symmetric,
                                           grad)
    dloss_by_dx = reshape_dloss_by_dx_for_axis_handling(inputs, dloss_by_dx, axis_handling)

    #return grad in case of floating-point mode
    dloss_by_dx = tf.cond(is_int_data_type, lambda: dloss_by_dx, lambda: grad)
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
        _compute_dloss_by_dmin_dmax_and_dx_for_per_channel(op.inputs[0],
                                                           op.inputs[int(QuantizeOpIndices.bit_width)],
                                                           op.inputs[int(QuantizeOpIndices.op_mode)],
                                                           op.inputs[int(QuantizeOpIndices.encoding_min)],
                                                           op.inputs[int(QuantizeOpIndices.encoding_max)],
                                                           op.inputs[int(QuantizeOpIndices.use_symmetric_encoding)],
                                                           op.inputs[int(QuantizeOpIndices.is_int_data_type)],
                                                           op.inputs[int(QuantizeOpIndices.axis_handling)],
                                                           grad)
    return dloss_by_dx, None, None, dloss_by_dmin, dloss_by_dmax, None, None, None, None, None
