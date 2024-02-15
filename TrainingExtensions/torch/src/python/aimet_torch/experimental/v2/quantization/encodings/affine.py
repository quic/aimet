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
""" Affine encoding definition """

import torch
from torch._C._nn import _parse_to as parse_to_args

from aimet_torch.experimental.v2.quantization.encodings.base import EncodingBase


__all__ = ["AffineEncoding"]


def get_symmetry_mode(symmetric=False, signed=False, strict=False) -> str:
    """
    Returns matched symmetry mode string from encoding property flags
    """
    if not symmetric:
        return "asymmetric"
    if (signed, strict) == (True, True):
        return "strict_symmetric"
    if (signed, strict) == (True, False):
        return "signed_symmetric"
    if (signed, strict) == (False, False):
        return "unsigned_symmetric"
    raise RuntimeError("No matching symmetry exists for symmetric={}, signed={}, strict={}".format(symmetric, signed, strict))

class AffineEncoding(EncodingBase):
    """
    Encoding object for affine quantization
    """
    def __init__(self, scale: torch.Tensor, offset: torch.Tensor, bitwidth: int,
                 signed: bool = False, strict: bool = False):
        self._scale = scale
        self._offset = offset
        symmetric = (signed and torch.all(offset == 0)) or (not signed and torch.all(offset == - 2 ** (bitwidth - 1)))
        self._symmetry = get_symmetry_mode(symmetric, signed, strict)
        self._bitwidth = bitwidth

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
        return "perchannel"

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
        if self.symmetry in ("strict_symmetric", "signed_symmetric"):
            return torch.zeros_like(self._offset)
        return self._offset

    @property
    def num_steps(self) -> int:
        """
        Returns the number of steps of the quantizer encoding
        """
        if self.symmetry == "strict_symmetric":
            return 2 ** self.bitwidth - 2
        return 2 ** self.bitwidth - 1

    @property
    def min(self) -> torch.Tensor:
        """
        Returns the min value of the quantizer encoding
        """
        if self.symmetry == "strict_symmetric":
            return - (2 ** (self.bitwidth - 1) - 1) / self.scale
        if self.symmetry == "signed_symmetric":
            return - (2 ** (self.bitwidth - 1)) / self.scale
        return self.offset / self.scale

    @property
    def max(self) -> torch.Tensor:
        """
        Returns the max value of the quantizer encoding
        """
        if self.symmetry in ("signed_symmetric", "strict_symmetric"):
            return (2 ** (self.bitwidth - 1) - 1) / self.scale
        return (self._offset + self.num_steps) / self.scale

    @property
    def symmetry(self) -> str:
        """
        Returns the symmetry mode of the quantizer encoding
        """
        return self._symmetry

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
        if self.symmetry in ("unsigned_symmetric", "asymmetric"):
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
        Changes dtype of data in quantizer encoding or device where the data is
        """
        to_args = parse_to_args(*args, **kwargs)
        device, dtype_, _, _ = to_args
        if dtype_ in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            self._scale = self._scale.to(dtype_)
            self._offset = self._offset.to(dtype_)
        elif dtype_:
            raise RuntimeError(f"Cannot change encoding data dtype to {dtype_}, "
                               "only floating point data types are supported")
        self._scale = self._scale.to(device)
        self._offset = self._offset.to(device)
        return self
