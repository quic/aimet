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

class DefaultOpImpl:
    """
    Default quantization backend class. Supports quantization, dequantization, and
    quant-dequant using pytorch operations.
    """
    @staticmethod
    def _is_a_broadcasted_to_b(a: torch.Tensor, b: torch.Tensor) -> bool:
        """
        Checks any dimensions of tensor a is broadcasted to the corresponding dimension of b
        """
        for dim_a, dim_b in zip(a.shape[::-1], b.shape[::-1]):
            if dim_a == 1 and dim_a < dim_b:
                return True
        return False

    @staticmethod
    def _validate_arguments(tensor: torch.Tensor, delta: torch.Tensor, offset: torch.Tensor, bitwidth: Union[torch.Tensor, int] = None):
        if not tensor.dtype == delta.dtype == offset.dtype:
            raise RuntimeError("Data type of tensor, delta, and offset are should be the same")
        if bitwidth and torch.finfo(tensor.dtype).bits < bitwidth:
            raise RuntimeError("Quantization bitwidth should be smaller than the number of bits of input dtype")
        if len(tensor.shape) < len(delta.shape) or DefaultOpImpl._is_a_broadcasted_to_b(tensor, delta):
            raise RuntimeError("Input tensor should not be broadcasted to encoding shape")

    @staticmethod
    def quantize(tensor: torch.Tensor, delta: torch.Tensor, offset: torch.Tensor, bitwidth: Union[torch.Tensor, int]) -> torch.Tensor:
        """
        Quantize the tensor, using given parameters

        :param tensor: Tensor to quantize
        :param delta: Delta value for quantization
        :param offset: Offset value for quantization
        :param bitwidth: Quantization bitwidth
        :return: Resulting tensor
        """
        DefaultOpImpl._validate_arguments(tensor, delta, offset, bitwidth)
        return QuantizeFunc.apply(tensor, delta, offset, bitwidth)

    @staticmethod
    def dequantize(tensor: torch.Tensor, delta: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """
        Dequantize the tensor, using given parameters

        :param tensor: Tensor to dequantize
        :param delta: Delta value for dequantization
        :param offset: Offset value for dequantization
        :return: Resulting tensor
        """
        DefaultOpImpl._validate_arguments(tensor, delta, offset)
        return DequantizeFunc.apply(tensor, delta, offset)

    @staticmethod
    def quantize_dequantize(tensor: torch.Tensor, delta: torch.Tensor, offset: torch.Tensor, bitwidth: Union[torch.Tensor, int]) -> torch.Tensor:
        """
        Quantize-dequantize the tensor, using given parameters

        :param tensor: Tensor to quantize-dequantize
        :param delta: Delta value for quantize-dequantize
        :param offset: Offset value for quantize-dequantize
        :param bitwidth: Quantize-dequantize bitwidth
        :return: Resulting tensor
        """
        DefaultOpImpl._validate_arguments(tensor, delta, offset, bitwidth)
        return QuantDequantFunc.apply(tensor, delta, offset, bitwidth)


# pylint: disable=abstract-method
class QuantizeFunc(torch.autograd.Function):
    """
    Custom gradient function for quantization
    """
    # pylint: disable=arguments-differ
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, delta: torch.Tensor, offset: torch.Tensor, bitwidth: Union[torch.Tensor, int]):
        x_round = torch.round(tensor / delta) - torch.round(offset)
        if tensor.requires_grad or delta.requires_grad or offset.requires_grad:
            mask = (x_round >= 0) * (x_round <= (2 ** bitwidth - 1))
        else:
            mask = None
        ctx.save_for_backward(tensor, delta, offset, mask)
        return torch.clamp(x_round, 0, 2 ** bitwidth - 1)

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad):
        tensor, delta, offset, mask = ctx.saved_tensors
        tensor_grad = grad * mask / delta if tensor.requires_grad else None
        delta_grad = -grad * mask * tensor / (delta ** 2) if delta.requires_grad else None
        offset_grad = -grad * mask if offset.requires_grad else None
        return tensor_grad, delta_grad, offset_grad, None


# pylint: disable=abstract-method
class DequantizeFunc(torch.autograd.Function):
    """
    Custom gradient function for dequantization
    """
    # pylint: disable=arguments-differ
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, delta: torch.Tensor, offset: torch.Tensor):
        rounded_offset = torch.round(offset)
        x_dequant = (tensor + rounded_offset) * delta
        ctx.save_for_backward(tensor, delta, rounded_offset)
        return x_dequant

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad):
        tensor, delta, rounded_offset = ctx.saved_tensors
        tensor_grad = grad * delta
        delta_grad = grad * (tensor + rounded_offset)
        offset_grad = grad * delta
        return tensor_grad, delta_grad, offset_grad


# pylint: disable=abstract-method
class QuantDequantFunc(torch.autograd.Function):
    """
    Custom gradient function for quant-dequant
    """
    # pylint: disable=arguments-differ
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, delta: torch.Tensor, offset: torch.Tensor, bitwidth: Union[torch.Tensor, int]):
        rounded_offset = torch.round(offset)
        x_round = torch.round(tensor / delta) - rounded_offset
        x_quant = torch.clamp(x_round, 0, 2 ** bitwidth - 1)
        if tensor.requires_grad or delta.requires_grad or offset.requires_grad:
            mask = (x_round >= 0) * (x_round <= (2 ** bitwidth - 1))
        else:
            mask = None
        ctx.save_for_backward(tensor, delta, offset, rounded_offset, mask, x_quant)
        return (x_quant + rounded_offset) * delta

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad):
        tensor, delta, offset, rounded_offset, mask, x_quant = ctx.saved_tensors
        tensor_grad = grad * mask if tensor.requires_grad else None
        delta_grad = -grad * (mask * tensor / delta - x_quant - rounded_offset) \
            if delta.requires_grad else None
        offset_grad = -grad * (mask * delta - delta) if offset.requires_grad else None
        return tensor_grad, delta_grad, offset_grad, None
