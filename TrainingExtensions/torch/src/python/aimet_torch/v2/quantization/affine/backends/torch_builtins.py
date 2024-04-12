# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Default quantization backend for quantizing weights and activations """
import torch

from aimet_torch.v2.utils import _is_expandable


def _is_value_representable(dtype: torch.dtype, value):
    """
    Return whether a value can be represented with the given dtype
    """
    finfo = torch.finfo(dtype)
    return finfo.min < value < finfo.max


def _is_range_representable(dtype: torch.dtype, qmin: int, qmax: int):
    """
    Return whether a range can be represented with the given dtype
    """
    return _is_value_representable(dtype, qmax) and \
            _is_value_representable(dtype, qmin) and \
            _is_value_representable(dtype, qmax - qmin)


def _is_numerically_stable(dtype: torch.dtype, qmin: int, qmax: int):
    """
    Return whether a range can be **stably** represented with the given dtype
    """
    if not _is_range_representable(dtype, qmin, qmax):
        return False

    # NOTE: This is a heuristic criteria. It doesn't perfectly guarantee numerical stability
    #       This criteria allows 8-bit quantization of float16, but it needs more discussion
    if torch.finfo(dtype).tiny > 1e-1 / (qmax - qmin):
        return False

    return True


def _validate_arguments(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
                        qmin: int = None, qmax: int = None):
    if not tensor.dtype == scale.dtype == offset.dtype:
        raise RuntimeError("Data type of tensor, scale, and offset are should be the same")
    if not _is_expandable(scale.shape, tensor.shape):
        raise RuntimeError(f"Scale of shape {scale.shape} cannot be expanded like input tensor of shape {tensor.shape}")

    if qmin is not None and qmax is not None:
        if qmin >= qmax:
            raise RuntimeError(f"qmin ({qmin}) must be smaller than qmax ({qmax})")


def quantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
             qmin: int, qmax: int) -> torch.Tensor:
    """
    Performs differentiable quantization given scale, offset, and quantization range.

    :param tensor: Tensor to quantize
    :param scale: Scale factor for quantization
    :param offset: Offset value for quantization
    :param qmin: Minimum value of the quantization range
    :param qmax: Maximum value of the quantization range
    """
    _validate_arguments(tensor, scale, offset, qmin, qmax)

    if not _is_range_representable(tensor.dtype, qmin, qmax):
        msg = f"{tensor.dtype} is unable to represent quantized output of range [{qmin}, {qmax}]."
        raise RuntimeError(msg)

    return QuantizeFunc.apply(tensor, scale, offset, qmin, qmax)

def quantize_dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
                        qmin: int, qmax: int) -> torch.Tensor:
    """
    Performs differentiable quantize-dequantize given scale, offset, and quantization range.

    :param tensor: Tensor to quantize
    :param scale: Scale factor for quantization
    :param offset: Offset value for quantization
    :param qmin: Minimum value of the quantization range
    :param qmax: Maximum value of the quantization range
    """
    _validate_arguments(tensor, scale, offset, qmin, qmax)

    output_dtype = internal_dtype = tensor.dtype

    if not _is_numerically_stable(internal_dtype, qmin, qmax):
        internal_dtype = torch.float32

    if not _is_range_representable(internal_dtype, qmin, qmax):
        msg = f"{internal_dtype} is unable to represent quantized output of range [{qmin}, {qmax}]."
        raise RuntimeError(msg)

    return QuantDequantFunc.apply(tensor.to(internal_dtype),
                                  scale.to(internal_dtype),
                                  offset.to(internal_dtype),
                                  qmin, qmax).to(output_dtype)

def dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """
    Performs differentiable dequantize operation given scale and offset.

    :param tensor: Tensor to quantize
    :param scale: Scale factor for quantization
    :param offset: Offset value for quantization
    :return: Resulting tensor
    """
    _validate_arguments(tensor, scale, offset)
    return DequantizeFunc.apply(tensor, scale, offset)


# pylint: disable=abstract-method
class QuantizeFunc(torch.autograd.Function):
    """
    Custom gradient function for quantization
    """
    # pylint: disable=arguments-differ
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, qmin: int, qmax: int):
        x_round = torch.round(tensor / scale) - offset
        if tensor.requires_grad or scale.requires_grad or offset.requires_grad:
            mask = (x_round >= qmin) * (x_round <= qmax)
        else:
            mask = None
        ctx.tensor_requires_grad = tensor.requires_grad
        ctx.scale_requires_grad = scale.requires_grad
        ctx.offset_requires_grad = offset.requires_grad
        ctx.save_for_backward(tensor, scale, mask)
        return torch.clamp(x_round, qmin, qmax)

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad):
        tensor, scale, mask = ctx.saved_tensors
        if ctx.tensor_requires_grad or ctx.scale_requires_grad or ctx.offset_requires_grad:
            masked_grad = grad * mask
        tensor_grad = masked_grad / scale if ctx.tensor_requires_grad else None
        scale_grad = -masked_grad * tensor / scale / scale if ctx.scale_requires_grad else None
        offset_grad = -masked_grad if ctx.offset_requires_grad else None
        return tensor_grad, scale_grad, offset_grad, None, None


# pylint: disable=abstract-method
class DequantizeFunc(torch.autograd.Function):
    """
    Custom gradient function for dequantization
    """
    # pylint: disable=arguments-differ
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor):
        x_dequant = (tensor + offset) * scale
        ctx.tensor_requires_grad = tensor.requires_grad
        ctx.scale_requires_grad = scale.requires_grad
        ctx.offset_requires_grad = offset.requires_grad
        ctx.save_for_backward(tensor, scale, offset)
        return x_dequant

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad):
        tensor, scale, offset = ctx.saved_tensors
        if ctx.tensor_requires_grad or ctx.offset_requires_grad:
            tensor_and_offset_grad = grad * scale
        tensor_grad = tensor_and_offset_grad if ctx.tensor_requires_grad else None
        scale_grad = grad * (tensor + offset) if ctx.scale_requires_grad else None
        offset_grad = tensor_and_offset_grad if ctx.offset_requires_grad else None
        return tensor_grad, scale_grad, offset_grad


# pylint: disable=abstract-method
class QuantDequantFunc(torch.autograd.Function):
    """
    Custom gradient function for quant-dequant
    """
    # pylint: disable=arguments-differ, misplaced-comparison-constant
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, qmin: int, qmax: int):
        x_round = torch.round(tensor / scale) - offset
        x_quant = torch.clamp(x_round, qmin, qmax)
        if tensor.requires_grad or scale.requires_grad or offset.requires_grad:
            mask = (x_round >= qmin) * (x_round <= qmax)
        else:
            mask = None
        x_dequant = (x_quant + offset) * scale

        # Downcast x_quant if bitwidth is less than or equal to 8 to reduce memory consumption
        if 0 <= qmin and qmax <= 255:
            x_quant = x_quant.to(dtype=torch.uint8)
        elif -128 <= qmin and qmax <= 127:
            x_quant = x_quant.to(torch.int8)

        ctx.tensor_requires_grad = tensor.requires_grad
        ctx.scale_requires_grad = scale.requires_grad
        ctx.offset_requires_grad = offset.requires_grad
        ctx.save_for_backward(tensor, scale, offset, mask, x_quant)
        return x_dequant


    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad):
        tensor, scale, offset, mask, x_quant = ctx.saved_tensors
        tensor_grad = grad * mask if ctx.tensor_requires_grad else None
        scale_grad = grad * (x_quant + offset - mask * tensor / scale) \
            if ctx.scale_requires_grad else None
        offset_grad = -grad * (mask * scale - scale) if ctx.offset_requires_grad else None
        return tensor_grad, scale_grad, offset_grad, None, None
