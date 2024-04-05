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
# pylint: disable=redefined-builtin
""" Affine encoding definition """

import torch
from torch._C._nn import _parse_to as parse_to_args

from aimet_torch.v2.quantization.base import EncodingBase
from aimet_torch.v2.quantization.affine.backends import get_backend


__all__ = ["AffineEncoding"]


class AffineEncoding(EncodingBase):
    """
    Encoding object for affine quantization
    """
    def __init__(self, scale: torch.Tensor, offset: torch.Tensor, bitwidth: int, signed=False, symmetry=False):
        self._scale = scale
        self._offset = offset
        self._symmetry = symmetry
        self._bitwidth = bitwidth
        self._signed = signed

    @property
    def mapping(self) -> str:
        """
        Returns the mapping method for this encoding
        """
        return "affine"

    @property
    def granularity(self) -> str:
        """
        Returns the granularity of the quantizer encoding
        """
        if self.scale.shape in (torch.Size([]), torch.Size([1])):
            return "pertensor"
        if any(dim > 1 for dim in self.scale.shape):
            return "perchannel"
        return "unknown"

    @property
    def scale(self) -> torch.Tensor:
        """
        Returns the scale of the quantizer encoding
        """
        return self._scale

    @property
    def offset(self) -> torch.Tensor:
        """
        Returns the offset of the quantizer encoding
        """
        return self._offset

    @property
    def num_steps(self) -> int:
        """
        Returns the number of steps of the quantizer encoding
        """
        return 2 ** self.bitwidth - 1

    @property
    def num_negative_steps(self):
        """
        Returns the number of negative steps of the quantizer encoding
        """
        return self.num_steps - self.num_positive_steps

    @property
    def num_positive_steps(self):
        """
        Returns the number of positive steps of the quantizer encoding
        """
        if self._signed:
            return 2 ** (self.bitwidth - 1) - 1
        return self.num_steps

    @property
    def min(self) -> torch.Tensor:
        """
        Returns the min value of the quantizer encoding
        """
        return (self.offset - self.num_negative_steps) * self.scale

    @property
    def max(self) -> torch.Tensor:
        """
        Returns the max value of the quantizer encoding
        """
        return (self._offset + self.num_positive_steps) * self.scale

    @property
    def symmetry(self) -> bool:
        """
        Returns the symmetry mode of the quantizer encoding
        """
        return self._symmetry

    @property
    def signed(self) -> bool:
        """
        Returns whether the encoding uses signed integer representation
        """
        return self._signed

    @property
    def bitwidth(self) -> int:
        """
        Returns the bitwidth of the quantizer encoding
        """
        return self._bitwidth

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the quantizer encoding
        """
        if not self._signed:
            if self.bitwidth <= 8:
                return torch.uint8
            # No torch.uint16
            return torch.int32
        if self.bitwidth <= 8:
            return torch.int8
        if self.bitwidth <= 16:
            return torch.int16
        return torch.int32

    def to(self, *args, **kwargs):
        """
        Changes dtype of data in quantizer encoding or device where the data is.
        Behaves similar to torch.Tensor.to
        """
        to_args = parse_to_args(*args, **kwargs)
        device, dtype, _, _ = to_args
        dtype = dtype if dtype else self._scale.dtype
        device = device if device else self._scale.device

        if dtype is self._scale.dtype and device is self._scale.device:
            return self

        if not dtype.is_floating_point:
            raise RuntimeError(f"Cannot change encoding data dtype to {dtype}, "
                               "only floating point data types are supported")

        scale = self._scale.to(dtype=dtype, device=device)
        offset = self._offset.to(dtype=dtype, device=device)
        return type(self)(scale, offset, self._bitwidth)

    def quantize(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        offset = self.offset
        bitwidth = self.bitwidth
        signed = self.signed

        # Use dtype with more precision
        if torch.finfo(input.dtype).bits >= torch.finfo(scale.dtype).bits:
            dtype = input.dtype
        else:
            dtype = scale.dtype

        return get_backend().quantize(input.to(dtype),
                                      scale.to(dtype),
                                      offset.to(dtype),
                                      bitwidth,
                                      signed).to(input.dtype)

    def dequantize(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        offset = self.offset

        # Use dtype with more precision
        if torch.finfo(input.dtype).bits >= torch.finfo(scale.dtype).bits:
            dtype = input.dtype
        else:
            dtype = scale.dtype

        return get_backend().dequantize(input.to(dtype),
                                        scale.to(dtype),
                                        offset.to(dtype)).to(input.dtype)
