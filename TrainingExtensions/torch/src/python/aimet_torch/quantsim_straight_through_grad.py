# /usr/bin/env python3.8
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Implements straight through gradient computation for Quant op"""
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
from torch.autograd import Variable

from aimet_torch.tensor_factory_utils import constant_tensor_factory

if TYPE_CHECKING:
    from aimet_torch.tensor_quantizer import LearnedGridTensorQuantizer


@dataclass
class IntermediateResult:
    """
    Data carrier containing intermediate result for learned grid backward computation
    """
    x_quant: torch.Tensor
    encoding_min: torch.nn.Parameter
    encoding_max: torch.nn.Parameter
    delta: torch.Tensor
    offset: torch.Tensor
    mask_tensor: torch.Tensor
    num_steps: torch.Tensor
    is_symmetric: bool
    is_unsigned: bool


def broadcast_to_tensor(tensor, encoding, ch_axis):
    """
    This helper method takes n-dimension tensor and a 1-dimension encoding. And the encoding is broad-casted to
    match the n-dimensional tensor
    :param tensor: Tensor to use as target for the broadcasting operation
    :param encoding: Encoding 1-dimensional tensor to broadcast
    :param ch_axis: Channel axis along which broadcasting happens
    :return: Broad-casted tensor
    """
    if not isinstance(encoding, torch.Tensor):
        encoding = torch.Tensor(encoding).to(tensor.device)  # convert encoding to a tensor

    # Original tensor shape is OIHW/IOHW, we change the shape to IHWO. Encoding (which is of shape O) can naturally
    # broadcast to this shape
    # This will work if the original tensor shape was any dimensions as long as the first dimension matches the
    # encoding tensor shape
    shape = list(tensor.shape)
    num_channels = shape.pop(ch_axis)
    encoding = encoding * torch.ones(shape + [num_channels]).to(tensor.device)

    # we permute the resulting tensor back to OIHW/IOHW shape
    permute_dims = list(range(len(shape)))
    permute_dims.insert(ch_axis, len(shape))
    encoding = encoding.permute(permute_dims)

    return encoding


def compute_dloss_by_dx(x, grad, encoding_min, encoding_max, ch_axis=0):
    """
    compute derivative w.r.t input using straight through estimator.
    :param x: input tensor
    :param grad: gradient flowing
    :param encoding_min: encoding min grid param used on forward pass
    :param encoding_max: encoding max grid param used on forward pass
    :param ch_axis: Channel axis to use for per-channel quant
    :return: gradient w.r.t input
    """

    # Broadcast the encoding min and max tensors if they were single dimensioned. If they were scalars, the
    # broadcast is automatic and more optimal in runtime, so we skip calling the helper above
    if isinstance(encoding_max, list) and len(x.shape) > 1:
        encoding_max = broadcast_to_tensor(x, encoding_max, ch_axis)

    if isinstance(encoding_min, list) and len(x.shape) > 1:
        encoding_min = broadcast_to_tensor(x, encoding_min, ch_axis)
    else:
        encoding_min = torch.Tensor([encoding_min]).to(x.device)

    # compute dloss_by_dx = dq_by_dx * grad
    inner_cond = torch.where(torch.le(x, encoding_max),  # condition to check per value
                             torch.ones_like(x),  # execute if true
                             torch.zeros_like(x))  # execute if false

    dloss_by_dx = torch.where(torch.le(encoding_min, x),  # condition to check per value
                              inner_cond,  # execute if true
                              torch.zeros_like(x)) * grad

    return dloss_by_dx


def get_computed_encodings(bitwidth: int,
                           encoding_min: torch.nn.Parameter,
                           encoding_max: torch.nn.Parameter,
                           use_symmetric_encodings: bool,
                           use_strict_symmetric: bool,
                           is_unsigned_symmetric: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute delta and offset, and number of steps given quantization parameters
    Followed the flow of C++ compute encoding function (quantization_utils::getComputedEncodings)
    :param bitwidth: Bitwidth
    :param encoding_min: Encoding min
    :param encoding_max: Encoding max
    :param use_symmetric_encodings: True if symmetric encoding is used. False otherwise
    :param use_strict_symmetric: True if strict symmetric encoding is used. False otherwise
    :param is_unsigned_symmetric: Whether to use signed/unsigned in symmetric case
    :return: Tuple of delta and offset and num_steps
    """
    device = encoding_min.device
    num_steps = 2 ** bitwidth - 1
    if use_symmetric_encodings and use_strict_symmetric:
        num_steps -= 1
    half_num_steps = num_steps / 2

    num_steps_tensor = constant_tensor_factory(num_steps, device)
    # NOTE: This assumes that the use_* flags reflect true condition (regardless of the encoding_* values)
    if use_symmetric_encodings and (not is_unsigned_symmetric):
        # signed symmetric
        absmax = torch.max(torch.abs(encoding_min), torch.abs(encoding_max))
        delta = absmax / constant_tensor_factory(math.floor(half_num_steps), device)
        offset = -constant_tensor_factory(math.ceil(half_num_steps), device)
    else:
        delta = (encoding_max - encoding_min) / num_steps_tensor
        if use_symmetric_encodings:
            # unsigned symmetric
            offset = encoding_min / delta
        else:
            # asymmetric
            zero_tensor = constant_tensor_factory(0., device)
            b_zero = torch.round(-encoding_min / delta)
            b_zero = torch.min(num_steps_tensor, torch.max(zero_tensor, b_zero))
            offset = -b_zero.clone().detach()

    return delta, offset, num_steps_tensor


def _compute_variables_for_range_learning(tensor: torch.Tensor,
                                          bitwidth: int,
                                          encoding_min: torch.nn.Parameter,
                                          encoding_max: torch.nn.Parameter,
                                          channel_axis: int,
                                          use_symmetric_encodings: bool,
                                          use_strict_symmetric: bool,
                                          is_unsigned_symmetric: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate required variables for range learning
    :param tensor: torch Tensor
    :param bitwidth: Bitwidth for quantization
    :param encoding_min: Encoding min
    :param encoding_max: Encoding max
    :param channel_axis: Channel axis to use for per-channel quant
    :param use_symmetric_encodings: True if symmetric encoding is used. False otherwise
    :param use_strict_symmetric: True if strict symmetric encoding is used. False otherwise
    :param is_unsigned_symmetric: Whether to use signed/unsigned in symmetric case
    """
    delta, offset, num_steps = get_computed_encodings(bitwidth, encoding_min, encoding_max,
                                                      use_symmetric_encodings, use_strict_symmetric, is_unsigned_symmetric)
    # broadcasting
    if len(encoding_min) > 1:
        delta = broadcast_to_tensor(tensor, delta, channel_axis)
        offset = broadcast_to_tensor(tensor, offset, channel_axis)

    return delta, offset, num_steps


# pylint:disable=too-many-locals
def calculate_forward_pass(tensor: torch.Tensor,
                           tensor_quantizer: "LearnedGridTensorQuantizer",
                           encoding_min: torch.nn.Parameter,
                           encoding_max: torch.nn.Parameter) -> Tuple[torch.Tensor, IntermediateResult]:
    """
    Calculate forward pass logic of range learning
    :param tensor: Target tensor to compute
    :param tensor_quantizer: LearnedGridTensorQuantizer corresponding to target tensor
    :param encoding_min: Encoding min
    :param encoding_max: Encoding max
    :return: QuantizeDequantize out and intermediate result tuple
    """
    use_symmetric_encodings = tensor_quantizer.use_symmetric_encodings
    is_unsigned_symmetric = tensor_quantizer.is_unsigned_symmetric
    delta, offset, num_steps = _compute_variables_for_range_learning(tensor,
                                                                     tensor_quantizer.bitwidth,
                                                                     encoding_min,
                                                                     encoding_max,
                                                                     tensor_quantizer.channel_axis,
                                                                     use_symmetric_encodings,
                                                                     tensor_quantizer.use_strict_symmetric,
                                                                     is_unsigned_symmetric)

    zero = torch.zeros_like(num_steps)

    x_round = torch.round(tensor / delta) - offset
    x_quant = x_round.clamp(zero, num_steps)
    x_dequant = (x_quant + offset) * delta

    mask_tensor = x_round.ge(zero) * x_round.le(num_steps)

    # Downcast x_quant if bitwidth is less than or equal to 8 to reduce memory consumption
    if tensor_quantizer.bitwidth <= 8:
        x_quant = x_quant.to(dtype=torch.uint8)

    intermediate_result = IntermediateResult(x_quant,
                                             encoding_min, encoding_max,
                                             delta, offset, mask_tensor, num_steps,
                                             use_symmetric_encodings, is_unsigned_symmetric)
    return x_dequant, intermediate_result


# pylint:disable=too-many-locals
def asymmetric_gradients(tensor: torch.Tensor,
                         grad: torch.Tensor,
                         intermediate_result: IntermediateResult,
                         channel_axis: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate asymmetric gradients with respect to tensor, gradients of encoding min and max
    :param tensor: Given tensor
    :param grad: Gradient using which other gradients will be calculated
    :param intermediate_result: Intermediate result from forward pass
    :param channel_axis: Channel axis
    :return: Gradients with respect to tensor, gradients of encoding min and max
    """
    delta = intermediate_result.delta
    offset = intermediate_result.offset
    x_quant = intermediate_result.x_quant
    mask_tensor = intermediate_result.mask_tensor

    grad_xq = delta * grad
    mask_tensor = Variable(mask_tensor.type_as(grad_xq.data))

    grad_tensor = grad * mask_tensor
    grad_scale = (x_quant + offset - tensor * mask_tensor / delta) * grad
    grad_offset = grad_xq * (1 - mask_tensor)

    dim = list(range(len(tensor.shape)))
    if len(delta) > 1 and len(tensor.shape) > 1:
        dim.pop(channel_axis)

    num_steps = intermediate_result.num_steps
    encoding_min = intermediate_result.encoding_min
    encoding_max = intermediate_result.encoding_max

    if len(delta) > 1 and len(tensor.shape) == 1:
        # NOTE: Handle when applying per-channel quant to 1-D Tensor case such as bias tensor in Conv or beta/gamma in BatchNorm
        intermediate_term1 = grad_scale / num_steps
        intermediate_term2 = num_steps / (encoding_max - encoding_min) ** 2 * grad_offset
    else:
        # Per-channel quant to k-D Tensor (k >= 2) or per-tensor case
        intermediate_term1 = grad_scale.sum(dim=dim) / num_steps
        intermediate_term2 = num_steps / (encoding_max - encoding_min) ** 2 * grad_offset.sum(dim=dim)

    grad_encoding_min = -intermediate_term1 + encoding_max * intermediate_term2
    grad_encoding_max = intermediate_term1 - encoding_min * intermediate_term2

    return grad_tensor, grad_encoding_min, grad_encoding_max


# pylint:disable=too-many-locals
def unsigned_symmetric_gradients(tensor: torch.Tensor,
                                 grad: torch.Tensor,
                                 intermediate_result: IntermediateResult,
                                 channel_axis: int) -> Tuple[torch.Tensor, None, torch.Tensor]:
    """
    Calculate unsigned symmetric gradients with respect to tensor, gradients of encoding min and max
    :param tensor: Given tensor
    :param grad: Gradient using which other gradients will be calculated
    :param intermediate_result: Intermediate result from forward pass
    :param channel_axis: Channel axis
    :return: Gradients with respect to tensor, gradients of encoding min and max
    """
    delta = intermediate_result.delta
    x_quant = intermediate_result.x_quant
    mask_tensor = intermediate_result.mask_tensor

    mask_tensor = Variable(mask_tensor.type_as(grad.data))

    dim = list(range(len(tensor.shape)))
    if len(delta) > 1 and len(tensor.shape) > 1:
        dim.pop(channel_axis)

    num_steps = intermediate_result.num_steps
    grad_tensor = mask_tensor * grad

    if len(delta) > 1 and len(tensor.shape) == 1:
        # NOTE: Handle when applying per-channel quant to 1-D Tensor case such as bias tensor in Conv or beta/gamma in BatchNorm
        grad_encoding_max = (x_quant * grad) - (mask_tensor * (tensor / delta) * grad)
    else:
        # Per-channel quant to k-D Tensor (k >= 2) or per-tensor case
        grad_encoding_max = (x_quant * grad).sum(dim=dim) - (mask_tensor * (tensor / delta) * grad).sum(dim=dim)

    grad_encoding_max = grad_encoding_max / num_steps

    return grad_tensor, None, grad_encoding_max


# pylint:disable=too-many-locals
def symmetric_gradients(tensor: torch.Tensor,
                        grad: torch.Tensor,
                        intermediate_result: IntermediateResult,
                        channel_axis: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate signed symmetric gradients with respect to tensor, gradients of encoding min and max
    :param tensor: Given tensor
    :param grad: Gradient using which other gradients will be calculated
    :param intermediate_result: Intermediate result from forward pass
    :param channel_axis: Channel axis
    :return: Gradients with respect to tensor, gradients of encoding min and max
    """
    delta = intermediate_result.delta
    offset = intermediate_result.offset
    x_quant = intermediate_result.x_quant
    mask_tensor = intermediate_result.mask_tensor
    mask_tensor = Variable(mask_tensor.type_as(grad.data))

    dim = list(range(len(tensor.shape)))
    if len(delta) > 1 and len(tensor.shape) > 1:
        dim.pop(channel_axis)

    num_steps = intermediate_result.num_steps
    grad_tensor = mask_tensor * grad

    if len(delta) > 1 and len(tensor.shape) == 1:
        # NOTE: Handle when applying per-channel quant to 1-D Tensor case such as bias tensor in Conv or beta/gamma in BatchNorm
        grad_encoding_max = ((x_quant + offset) * grad) - (mask_tensor * (tensor / delta) * grad)
    else:
        # Per-channel quant to k-D Tensor (k >= 2) or per-tensor case
        grad_encoding_max = ((x_quant + offset) * grad).sum(dim=dim) - (mask_tensor * (tensor / delta) * grad).sum(dim=dim)

    grad_encoding_max = grad_encoding_max / torch.div(num_steps, 2, rounding_mode="floor")

    return grad_tensor, -grad_encoding_max, grad_encoding_max


# pylint:disable=too-many-locals
def calculate_gradients(tensor: torch.Tensor,
                        grad: torch.Tensor,
                        intermediate_result: IntermediateResult,
                        channel_axis: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate gradients with respect to tensor, gradients of encoding min and max
    :param tensor: Given tensor
    :param grad: Gradient using which other gradients will be calculated
    :param intermediate_result: Intermediate result from forward pass
    :param channel_axis: Channel axis
    :return: Gradients with respect to tensor, gradients of encoding min and max
    """
    is_symmetric = intermediate_result.is_symmetric
    is_unsigned = intermediate_result.is_unsigned

    if not is_symmetric:
        return asymmetric_gradients(tensor, grad, intermediate_result, channel_axis)

    return unsigned_symmetric_gradients(tensor, grad, intermediate_result, channel_axis) if is_unsigned else \
        symmetric_gradients(tensor, grad, intermediate_result, channel_axis)


class RoundStraightThrough(torch.autograd.Function):
    """
    Defining gradient of rounding function as pass-through since round is a non-linearity
    """

    @staticmethod
    # pylint: disable=arguments-differ
    def forward(ctx, *x):
        return torch.round(*x)

    @staticmethod
    def backward(ctx, *output_grad):
        return output_grad
