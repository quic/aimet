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

from typing import Callable

import torch
from torch._C._nn import _parse_to
from torch.utils._pytree import tree_map_only

from aimet_torch.experimental.v2.quantization.backends import get_backend
from aimet_torch.experimental.v2.quantization.encodings import EncodingBase, AffineEncoding

# Operations that encodings can propagate through without change (to be populated)
PASSTHROUGH_OPS = {}


class QuantizedTensor(torch.Tensor):
    """
    Represents a quantized tensor as a subclass of torch.Tensor which also holds the quantization encodings. This is
    used to safely pass encoding information between layers of a torch model and into operator libraries.

    If a floating point operation is called on a QuantizedTensor, it will dequantize itself to a floating point
    representation before calling the operation.
    """

    # pylint:disable = unused-argument
    def __new__(cls, data, encoding, *args, **kwargs):

        # data.as_subclass will return a QuantizedTensor with the same data pointer as data
        # At this point, the data is still in floating point, but with quantized values
        return data.as_subclass(cls)

    # pylint:disable = unused-argument
    def __init__(self,
                 data: torch.Tensor,
                 encoding: EncodingBase,
                 dequant_fn: Callable[["QuantizedTensor"], torch.Tensor],
                 *args, **kwargs) -> "QuantizedTensor":
        """
        Creates a QuantizedTensor object which contains both quantized tensor data and quantization encodings.

        :param data: torch tensor containing quantized values in a floating point representation
        :param encoding: Encoding object storing quantization parameters
        :param dequant_fn: Mapping function from the quantized representation back to floating point representation
        """
        super().__init__()
        self._encoding = encoding
        self._dequant_fn = dequant_fn

    @property
    def encoding(self) -> EncodingBase:
        """
        Returns the QuantizedTensor's encoding
        """
        return self._encoding

    def attach_encoding(self, encoding, dequant_fn):
        """
        Attach a new encoding and dequantization function to the given tensor

        :param encoding: Encoding object holding quantization parameters
        :param dequant_fn: Function used to dequantize to a floating point tensor
        """
        self._encoding = encoding
        self._dequant_fn = dequant_fn
        return self

    def dequantize(self) -> torch.Tensor:
        """
        Dequantize to a floating point torch.Tensor object
        """
        return self._dequant_fn(self)

    def quantized_repr(self) -> torch.Tensor:
        """
        Return the quantized representation of the tensor as a torch.Tensor with data type self.encoding.dtype
        """
        return torch.Tensor(self).to(self.encoding.dtype)

    def __str__(self):
        return f"QuantizedTensor({self.quantized_repr()}, encoding: {self.encoding})"

    def to(self, *args, **kwargs):
        """
        Must behave similar to torch.Tensor.to, i.e., return a copy of self with the specified device/dtype
        without altering any attributes of self
        """
        device, dtype, non_blocking, mem_format = _parse_to(*args, **kwargs)
        dtype = dtype if dtype else self.dtype
        device = device if device else self.device
        if not dtype.is_floating_point:
            raise RuntimeError("Cannot send QuantizedTensor to a non-floating point data type")
        if dtype is self.dtype and device is self.device:
            return self
        data = super().to(dtype=dtype, device=device, non_blocking=non_blocking, memory_format=mem_format)
        enc = self.encoding.to(dtype=dtype, device=device, non_blocking=non_blocking)
        return type(self)(data, enc, self._dequant_fn)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        This function will be called anytime a torch operation is called with a QuantizedTensor as one of the
        arguments. Here, we dequantize all QuantizedTensors before calling into any torch function which has not been
        designated as passthrough (aka "math invariant")
        """
        if kwargs is None:
            kwargs = {}
        if func in PASSTHROUGH_OPS:
            for arg in args:
                if isinstance(arg, QuantizedTensor):
                    # pylint:disable = protected-access
                    encoding, dequant_fn = arg.encoding, arg._dequant_fn
                    break
            output = super().__torch_function__(func, types, args, kwargs)
            return tree_map_only(QuantizedTensor, lambda qt: qt.attach_encoding(encoding, dequant_fn), output)
        args, kwargs = tree_map_only(QuantizedTensor, lambda qt: qt.dequantize(), (args, kwargs))
        output = super().__torch_function__(func, types, args, kwargs)
        return tree_map_only(QuantizedTensor, torch.Tensor, output)


def affine_quantize(tensor: torch.Tensor,
                    scale: torch.Tensor,
                    offset: torch.Tensor, bitwidth: int,
                    signed: bool = False,
                    strict: bool = False) -> QuantizedTensor:
    """
    Quantizes the input tensor into a QuantizedTensor using the quantization parameters
    """
    tensor_q = get_backend().quantize(tensor, scale, offset, bitwidth)
    encoding = AffineEncoding(scale, offset, bitwidth, signed, strict)
    dequant = get_backend().dequantize
    dequant_fn = lambda t: dequant(torch.Tensor(t), t.encoding.scale, t.encoding.offset)
    qtensor = QuantizedTensor(tensor_q, encoding, dequant_fn=dequant_fn)
    return qtensor
