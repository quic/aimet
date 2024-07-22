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

from typing import List, Optional, Dict, Any
import torch
from torch._C._nn import _parse_to as parse_to_args

from aimet_torch.v2.quantization.base import EncodingBase
from aimet_torch.v2.quantization.affine.backends import quantize, dequantize


__all__ = ["AffineEncoding", "VectorEncoding"]


class AffineEncoding(EncodingBase):
    """
    Encoding object for affine quantization
    """
    def __init__(self, scale: torch.Tensor, offset: torch.Tensor, bitwidth: int, signed=False, symmetry=False,
                 block_size: Optional[List] = None):
        self._scale = scale
        self._offset = offset
        self._symmetry = symmetry
        self._bitwidth = bitwidth
        self._signed = signed
        self._block_size = block_size

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
        if self.scale.dim() == 0:
            return "pertensor"
        if self.block_size is not None:
            return "blockwise"
        non_singleton_dims = tuple(dim for dim in self.scale.shape if dim > 1)
        if len(non_singleton_dims) <= 1:
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

    @property
    def block_size(self) -> Optional[List]:
        """
        Returns the block sizes of the quantizer encoding
        """
        return self._block_size

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
        properties = self._get_additional_properties()
        return type(self)(scale, offset, self._bitwidth, self._signed, self._symmetry, **properties)

    def quantize(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        offset = self.offset
        bitwidth = self.bitwidth
        signed = self.signed
        block_size = self.block_size

        # Subclasses of torch.Tensor with custom __torch_function__ (in our case, QuantizedTensorBase)
        # is known to introduce substantial CPU overhead.
        # Cast types of the inputs to plain torch.Tensor for faster execution.
        return quantize(input.as_subclass(torch.Tensor),
                        scale.to(input.dtype).as_subclass(torch.Tensor),
                        offset.to(input.dtype).as_subclass(torch.Tensor),
                        bitwidth, signed, block_size=block_size)

    def dequantize(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        offset = self.offset
        block_size = self.block_size

        # Subclasses of torch.Tensor with custom __torch_function__ (in our case, QuantizedTensorBase)
        # is known to introduce substantial CPU overhead.
        # Cast types of the inputs to plain torch.Tensor for faster execution.
        return dequantize(input.as_subclass(torch.Tensor),
                          scale.to(input.dtype).as_subclass(torch.Tensor),
                          offset.to(input.dtype).as_subclass(torch.Tensor),
                          block_size=block_size)

    def _to_legacy_format(self):
        min = self.min.flatten()
        max = self.max.flatten()
        scale = self.scale.flatten()

        if self._signed: # Legacy behavior is to use offset = 2 ** (bitwidth - 1) for signed symmetric
            offset = self.offset.flatten() - 2 ** (self.bitwidth - 1)
        else:
            offset = self.offset.flatten()

        return [
            {'min': float(min_), 'max': float(max_),
             'scale': float(scale_), 'offset': int(offset_),
             'bitwidth': self.bitwidth, 'dtype': 'int', 'is_symmetric': str(self.symmetry)}
            for min_, max_, scale_, offset_ in zip(min, max, scale, offset)
        ]

    # pylint: disable=no-self-use
    def _get_additional_properties(self) -> Dict[str, Any]:
        return {}

class VectorEncoding(AffineEncoding):
    """
    Encoding object for vector quantization
    """
    def __init__(
        self,
        scale: torch.Tensor,
        offset: torch.Tensor,
        bitwidth: int,
        signed=False,
        symmetry=False,
        block_size: Optional[List] = None,
        **kwargs,
    ):
        super().__init__(scale, offset, bitwidth, signed, symmetry, block_size)
        self.rows_per_block = kwargs["rows_per_block"]
        self.cols_per_block = kwargs["cols_per_block"]
        self.vector_dim = kwargs["vector_dim"]
        self.vector_stride = kwargs["vector_stride"]
        self.index_bw = kwargs["index_bw"]

    def _to_legacy_format(self):
        encoding = super()._to_legacy_format()
        for i, _ in enumerate(encoding):
            encoding[i].update(
                rows_per_block=self.rows_per_block,
                cols_per_block=self.cols_per_block,
                vector_dim=self.vector_dim,
                vector_stride=self.vector_stride,
                index_bw=self.index_bw,
            )
        return encoding

    def _get_additional_properties(self) -> Dict[str, Any]:
        return {
            "rows_per_block": self.rows_per_block,
            "cols_per_block": self.cols_per_block,
            "vector_dim": self.vector_dim,
            "vector_stride": self.vector_stride,
            "index_bw": self.index_bw,
        }
