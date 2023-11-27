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
from typing import Union
import torch

from aimet_torch.experimental.v2.utils import _is_expandable


def _validate_arguments(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, bitwidth: Union[torch.Tensor, int] = None):
    if not tensor.dtype == scale.dtype == offset.dtype:
        raise RuntimeError("Data type of tensor, scale, and offset are should be the same")
    if bitwidth and torch.finfo(tensor.dtype).bits <= bitwidth:
        raise RuntimeError(f"Dtype {tensor.dtype} has insufficient bitwidth to perform {bitwidth} quantization")
    if not _is_expandable(scale.shape, tensor.shape):
        raise RuntimeError(f"Scale of shape {scale.shape} cannot be expanded like input tensor of shape {tensor.shape}")

def quantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, bitwidth: Union[torch.Tensor, int]) -> torch.Tensor:
    """
    Performs differentiable quantization on tensor using scale, offset, and bitwidth parameters.

    :param tensor: Tensor to quantize
    :param scale: Scale factor for quantization
    :param offset: Offset value for quantization
    :param bitwidth: Output bitwidth of quantized tensor
    :return: Resulting tensor
    """
    _validate_arguments(tensor, scale, offset, bitwidth)
    return QuantizeFunc.apply(tensor, scale, offset, bitwidth)

def quantize_dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, bitwidth: Union[torch.Tensor, int]) -> torch.Tensor:
    """
    Performs differentiable quantize-dequantize operation on tensor using scale, offset, and bitwidth parameters.

    :param tensor: Tensor to quantize-dequantize
    :param scale: Scale factor for quantization
    :param offset: Offset value for quantization
    :param bitwidth: simulated quantization bitwidth
    :return: Resulting tensor
    """
    _validate_arguments(tensor, scale, offset, bitwidth)
    return QuantDequantFunc.apply(tensor, scale, offset, bitwidth)

def dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """
    Performs differentiable dequantize operation on tensor using scale and offset parameters.

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
    def forward(ctx, tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, bitwidth: Union[torch.Tensor, int]):
        x_round = torch.round(tensor / scale) - offset
        if tensor.requires_grad or scale.requires_grad or offset.requires_grad:
            mask = (x_round >= 0) * (x_round <= (2 ** bitwidth - 1))
        else:
            mask = None
        ctx.tensor_requires_grad = tensor.requires_grad
        ctx.scale_requires_grad = scale.requires_grad
        ctx.offset_requires_grad = offset.requires_grad
        ctx.save_for_backward(tensor, scale, mask)
        return torch.clamp(x_round, 0, 2 ** bitwidth - 1)

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad):
        tensor, scale, mask = ctx.saved_tensors
        if ctx.tensor_requires_grad or ctx.scale_requires_grad or ctx.offset_requires_grad:
            masked_grad = grad * mask
        tensor_grad = masked_grad / scale if ctx.tensor_requires_grad else None
        scale_grad = -masked_grad * tensor / scale / scale if ctx.scale_requires_grad else None
        offset_grad = -masked_grad if ctx.offset_requires_grad else None
        return tensor_grad, scale_grad, offset_grad, None


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
    # pylint: disable=arguments-differ
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, bitwidth: Union[torch.Tensor, int]):
        x_round = torch.round(tensor / scale) - offset
        x_quant = torch.clamp(x_round, 0, 2 ** bitwidth - 1)
        if tensor.requires_grad or scale.requires_grad or offset.requires_grad:
            mask = (x_round >= 0) * (x_round <= (2 ** bitwidth - 1))
        else:
            mask = None
        x_dequant = (x_quant + offset) * scale

        # Downcast x_quant if bitwidth is less than or equal to 8 to reduce memory consumption
        if bitwidth <= 8:
            x_quant = x_quant.to(dtype=torch.uint8)

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
        return tensor_grad, scale_grad, offset_grad, None
