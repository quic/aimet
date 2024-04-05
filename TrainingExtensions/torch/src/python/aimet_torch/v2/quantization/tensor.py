# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Quantized tensor class implementation """

import abc
import copy

import torch
from torch.utils._pytree import tree_map, tree_flatten

from aimet_torch.v2.quantization.base import EncodingBase


__all__ = ['QuantizedTensorBase', 'QuantizedTensor', 'DequantizedTensor', 'EncodingError']


class QuantizedTensorBase(torch.Tensor):
    """
    Represents a quantized or dequantized tensor as a subclass of torch.Tensor which also holds the quantization encodings.
    This is used to safely pass encoding information between layers of a torch model and into operator libraries.
    """

    encoding: EncodingBase

    _cast_ops = [
        torch.Tensor.half,
        torch.Tensor.float,
        torch.Tensor.double,
        torch.Tensor.char,
        torch.Tensor.short,
        torch.Tensor.int,
        torch.Tensor.long,
        torch.Tensor.cuda,
        torch.Tensor.cpu,
        torch.Tensor.to,
    ]

    @abc.abstractmethod
    def quantize(self) -> "QuantizedTensorBase":
        """
        Quantize tensor with the associated encoding

        NOTE: This method must be an IDEMPOTENT function.
              The result of calling this method multiple times should be equal to calling it only once.
              In other words, calling this method multiple times should not result in duplicate quantization.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def dequantize(self) -> "QuantizedTensorBase":
        """
        Dequantize tensor with the associated encoding

        NOTE: This method must be an IDEMPOTENT function.
              The result of calling this method multiple times should be equal to calling it only once.
              In other words, calling this method multiple times should not result in duplicate dequantization.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def quantized_repr(self) -> torch.Tensor:
        """
        Return the quantized representation of the tensor as a torch.Tensor with data type self.encoding.dtype
        """
        raise NotImplementedError

    @classmethod
    def __new__(cls, *args, **kwargs):
        encoding = kwargs.pop('encoding', None)
        ret = super().__new__(*args, **kwargs)
        if not ret.is_floating_point():
            raise RuntimeError(f"Non-floating point dtype `{ret.dtype}` is not allowed for quantized tensors.")
        ret.encoding = encoding
        return ret

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        ret = super().__torch_function__(func, types, args, kwargs)

        flattened_args, _ = tree_flatten((args, kwargs))
        if any(ret is arg for arg in flattened_args):
            # Return value is the same object as one of the arguments.
            # This implies that func is likely (but not necessarily) an in-place operator.
            return ret

        if func in cls._cast_ops:
            if not ret.dtype.is_floating_point:
                raise RuntimeError(
                    f"Type casting to non-floating point dtype `{ret.dtype}` is not allowed for quantized tensors. "
                    "To cast quantized tensors to integer, use `qtensor.quantzed_repr()`."
                )

            # Outputs of cast ops can inherit the same encoding as its parents
            self, *_ = args
            ret.encoding = copy.copy(self.encoding) # shallow copy

        def set_encoding(qtensor):
            if not hasattr(qtensor, 'encoding'):
                qtensor.encoding = None

            if qtensor.encoding is None:
                # If encoding does not exist, return a plain torch.Tensor
                return qtensor.as_subclass(torch.Tensor)

            # Change device of encoding
            # NOTE: We don't change the dtypes of encoding because scale/offset
            #       are sensitive to dtype
            qtensor.encoding = qtensor.encoding.to(device=qtensor.device)

            return qtensor

        return tree_map(lambda t: set_encoding(t) if isinstance(t, cls) else t, ret)


class QuantizedTensor(QuantizedTensorBase):
    """
    Represents quantized tensors
    """
    def quantize(self) -> "QuantizedTensor":
        if self.encoding is None:
            raise EncodingError("Encoding does not exist")
        return self

    def dequantize(self) -> "DequantizedTensor":
        if self.encoding is None:
            raise EncodingError("Encoding does not exist")

        qtensor = self.encoding.dequantize(self.as_subclass(torch.Tensor))
        qtensor = qtensor.as_subclass(DequantizedTensor)
        qtensor.encoding = copy.copy(self.encoding)
        return qtensor

    def quantized_repr(self) -> torch.Tensor:
        # FIXME(kyunggeu): This only works for affine encodings.
        #                  Needs to be generalized for any kind of encodings
        return self.quantize().as_subclass(torch.Tensor).to(self.encoding.dtype)


class DequantizedTensor(QuantizedTensorBase):
    """
    Represents dequantized tensors
    """
    def quantize(self) -> QuantizedTensor:
        if self.encoding is None:
            raise EncodingError("Encoding does not exist")

        qtensor = self.encoding.quantize(self.as_subclass(torch.Tensor))
        qtensor = qtensor.as_subclass(QuantizedTensor)
        qtensor.encoding = copy.copy(self.encoding)
        return qtensor

    def dequantize(self) -> "DequantizedTensor":
        if self.encoding is None:
            raise EncodingError("Encoding does not exist")
        return self

    def quantized_repr(self) -> torch.Tensor:
        # FIXME(kyunggeu): This only works for affine encodings.
        #                  Needs to be generalized for any kind of encodings
        return self.quantize().as_subclass(torch.Tensor).to(self.encoding.dtype)


class EncodingError(RuntimeError):
    """Error that indicates an encoding is missing or invalid"""
