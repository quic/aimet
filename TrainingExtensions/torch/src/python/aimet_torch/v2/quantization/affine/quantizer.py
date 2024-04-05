# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Affine quantizers """

import abc
from typing import Optional, List, Dict
import contextlib
import functools

import torch
from torch import nn

from aimet_torch.v2.utils import patch_attr, _is_expandable, StatisticsNotFoundError
from aimet_torch.v2.quantization.encoding_analyzer import EncodingAnalyzer, MinMaxEncodingAnalyzer
from aimet_torch.v2.quantization.affine import AffineEncoding
from aimet_torch.v2.quantization.tensor import QuantizedTensor, DequantizedTensor
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine.backends import get_backend
from aimet_torch.v2.utils import ste_round


__all__ = ['AffineQuantizerBase', 'MinMaxQuantizer', 'Quantize', 'QuantizeDequantize', 'Dequantize']


class AffineQuantizerBase(QuantizerBase):
    """
    Base class for linear quantization modules.

    :param shape: Shape of the quantization parameters.
    :param bitwidth: Quantization bitwidth.
    :param symmetric: If True, performs symmetric quantization;
                      otherwise, performs asymmetric quantization.
    :param encoding_analyzer: Encoding analyzer for calibrating quantization encodings.
                              (default: absolute min-max encoding analyzer)
    """
    def __init__(self, shape, bitwidth: int, symmetric: bool, encoding_analyzer: EncodingAnalyzer = None):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.bitwidth = bitwidth
        self._symmetric = symmetric
        # We support two quantization modes: (unsigned) asymmetric and signed-symmetric
        self._signed = symmetric
        self.encoding_analyzer = encoding_analyzer or MinMaxEncodingAnalyzer(shape)

        if not _is_expandable(self.encoding_analyzer.observer.shape, self.shape):
            raise RuntimeError(f'Encoding analyzer of shape {self.encoding_analyzer.observer.shape} '
                               f'is incompatible with quantizer of shape {self.shape}.')

    @abc.abstractmethod
    def get_min(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization min to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        :param dtype: dtype of the computed min
        :return: Quantization min
        """

    @abc.abstractmethod
    def get_max(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization max to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        :param dtype: dtype of the computed max
        :return: Quantization max
        """

    @abc.abstractmethod
    def get_scale(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization scale to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        :param dtype: dtype of the computed scale
        :return: Quantization scale
        """

    @abc.abstractmethod
    def get_offset(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization offset to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        :param dtype: dtype of the computed offset
        :return: Quantization offset
        """

    @abc.abstractmethod
    def set_range(self, min: torch.Tensor, max: torch.Tensor):
        """
        Set quantization parameters to the given min-max range
        """

    def get_encoding(self) -> Optional[AffineEncoding]:
        """
        Return the quantizer's encodings as an AffineEncoding object
        """
        if self.is_initialized():
            return AffineEncoding(self.get_scale(dtype=torch.float32),
                                  self.get_offset(dtype=torch.float32),
                                  self.bitwidth, self._signed, self._symmetric)
        return None

    @torch.no_grad()
    def get_legacy_encodings(self) -> Optional[List[Dict]]:
        """
        Returns a list of encodings, each represented as a List of Dicts
        """
        # pylint: disable=redefined-builtin

        if not self.is_initialized():
            return None

        min = self.get_min(dtype=torch.float32).flatten()
        max = self.get_max(dtype=torch.float32).flatten()
        scale = self.get_scale(dtype=torch.float32).flatten()
        offset = self.get_offset(dtype=torch.float32).flatten()
        if self._signed: # Legacy behavior is to use offset = 2 ** (bitwidth - 1) for signed symmetric
            offset -= 2 ** (self.bitwidth - 1)
        bitwidth = self.bitwidth
        dtype = "int"
        is_symmetric = self.symmetric

        return [
            {'min': float(min_), 'max': float(max_),
             'scale': float(scale_), 'offset': int(offset_),
             'bitwidth': bitwidth, 'dtype': dtype, 'is_symmetric': str(is_symmetric)}
            for min_, max_, scale_, offset_ in zip(min, max, scale, offset)
        ]

    @torch.no_grad()
    def set_legacy_encodings(self, encodings: List[Dict]):
        """
        Set encodings represented in the same format as the output of get_legacy_encodings as below:

        [
            {'min': float, 'max': float, 'scale': float, 'offset': float,
                     'bitwidth': int, 'dtype': str, 'is_symmetric': str},
            {'min': float, 'max': float, 'scale': float, 'offset': float,
                     'bitwidth': int, 'dtype': str, 'is_symmetric': str},
            ...
        ]
        """
        def str_to_bool(s: str):
            s = s.lower()
            if s == "false":
                return False
            if s == "true":
                return True
            raise ValueError

        self.bitwidth = encodings[0]['bitwidth']
        self.symmetric = str_to_bool(encodings[0]['is_symmetric'])
        min_ = torch.tensor([e['min'] for e in encodings])
        max_ = torch.tensor([e['max'] for e in encodings])
        self.set_range(min_, max_)

    def extra_repr(self) -> str:
        return f'shape={self.shape}, bitwidth={self.bitwidth}, symmetric={self.symmetric}'

    @property
    def symmetric(self) -> bool:
        """
        Indicates whether this quantizer uses symmetric quantization
        """
        return self._symmetric

    @symmetric.setter
    def symmetric(self, symmetric: bool):
        """
        Set the quantizer symmetry

        :param symmetric: If True, use symmetric encodings. Else, use asymmetric encodings
        """
        self._symmetric = symmetric

    @property
    def signed(self)-> bool:
        """
        Indicates whether this quantizer uses signed quantization
        """
        return self._signed

    @signed.setter
    def signed(self, signed: bool):
        """
        Set the quantizer to use signed or unsigned quantization

        :param signed: If True, use signed encodings, else use unsigned encodings
        """
        self._signed = signed


class MinMaxQuantizer(AffineQuantizerBase): # pylint: disable=abstract-method
    """
    Affine quantizer with min-max as trainable parameters
    """

    min: torch.nn.Parameter
    max: torch.nn.Parameter

    def __init__(self, shape, bitwidth: int, symmetric: bool, encoding_analyzer: EncodingAnalyzer = None):
        super().__init__(shape, bitwidth, symmetric, encoding_analyzer)

        self.register_quantization_parameter('min', nn.Parameter(-torch.ones(self.shape)))
        self.register_quantization_parameter('max', nn.Parameter(torch.ones(self.shape)))

    @contextlib.contextmanager
    def compute_encodings(self):
        """
        Observe inputs and update quantization parameters based on the input statistics.
        During ``compute_encodings`` is enabled, the quantizer forward pass performs
        dynamic quantization using the batch statistics.
        """
        if not self.encoding_analyzer:
            yield
            return

        original_forward = self.forward

        @functools.wraps(original_forward)
        def forward_wrapper(input):
            batch_statistics = self.encoding_analyzer.update_stats(input)
            dynamic_min, dynamic_max =\
                    self.encoding_analyzer.compute_encodings_from_stats(batch_statistics,
                                                                        self.bitwidth,
                                                                        self.symmetric)
            dynamic_min = dynamic_min.to(dtype=self.min.dtype,
                                         device=self.min.device).expand_as(self.min)
            dynamic_max = dynamic_max.to(dtype=self.max.dtype,
                                         device=self.max.device).expand_as(self.max)

            with patch_attr(self, 'min', dynamic_min),\
                    patch_attr(self, 'max', dynamic_max):
                return original_forward(input)

        try:
            with patch_attr(self, 'forward', forward_wrapper):
                yield
        except: # pylint: disable=try-except-raise
            raise
        else:
            try:
                min, max = self.encoding_analyzer.compute_encodings(self.bitwidth, self.symmetric)
            except StatisticsNotFoundError:
                return

            if min is None or max is None:
                return

            self.set_range(min, max)

        finally:
            self.encoding_analyzer.reset_stats()

    def get_min(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization min to be used for forward pass.

        NOTE: self.min may not be equal to self.get_min().
              self.get_min() returns slightly recalibrated version of self.min.

        :param dtype: dtype of the computed min. Use of self.min.dtype by default.
        :return: Quantization min
        """
        if not self.is_initialized():
            return None
        num_negative_steps = 2 ** (self.bitwidth - 1) if self._signed else 0
        return self.get_scale(dtype) * (self.get_offset(dtype) - num_negative_steps)

    def get_max(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization max to be used for forward pass.

        NOTE: self.max may not be equal to self.get_max()
              self.get_max() returns slightly recalibrated version of self.max.

        :param dtype: dtype of the computed max. Use of self.min.dtype by default.
        :return: Quantization max
        """
        if not self.is_initialized():
            return None
        num_positive_steps = 2 ** (self.bitwidth - 1) - 1 if self._signed else 2 ** self.bitwidth - 1
        return self.get_scale(dtype) * (self.get_offset(dtype) + num_positive_steps)

    def get_scale(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization scale to be used for forward pass.

        :param dtype: dtype of the computed scale. Use of self.min.dtype by default.
        :return: Quantization scale
        """
        if not self.is_initialized():
            return None

        dtype = dtype or self.min.dtype
        num_bins = 2 ** self.bitwidth - 1

        scale = (self.max.to(dtype) - self.min.to(dtype)) / num_bins
        return scale.to(dtype)

    def get_offset(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization offset to be used for forward pass.

        :param dtype: dtype of the computed offset. Use of self.min.dtype by default.
        :return: Quantization offset
        """
        if not self.is_initialized():
            return None

        dtype = dtype or self.min.dtype

        if self.symmetric:
            offset = torch.zeros_like(self.min, requires_grad=False, dtype=dtype)
        else:
            offset = ste_round(self.min.to(dtype) / self.get_scale(dtype))

            if self._signed:
                offset += 2 ** (self.bitwidth - 1)

        return offset.to(dtype)

    def set_range(self, min: torch.Tensor, max: torch.Tensor):
        """
        Set quantization parameters to the given min-max range
        """
        with torch.no_grad():
            self.min.copy_(min)
            self.max.copy_(max)


class Quantize(MinMaxQuantizer):
    """
    Applies quantization to the input
    """
    def forward(self, input: torch.Tensor) -> QuantizedTensor:
        """
        :param input: Input to quantize
        :return: Quantized output and scale/offset associated with it
        """
        if not self.is_initialized():
            raise RuntimeError(
                'Failed to run Quantize since quantization parameters are not initialized.'
                ' Please initialize the quantization parameters using `compute_encodings()`.'
            )

        dtype = input.dtype

        if torch.finfo(dtype).max < 2 ** self.bitwidth - 1:
            msg = f"{dtype} is unable to represent quantized output "\
                  f"of range [0, 2**(bitwidth={self.bitwidth})-1]. "\
                  "Please consider lowering the quantization bitwidth."
            raise RuntimeError(msg)

        encoding = self.get_encoding()
        output = get_backend().quantize(input.to(dtype),
                                        encoding.scale.to(dtype),
                                        encoding.offset.to(dtype),
                                        encoding.bitwidth,
                                        encoding.signed)
        output = output.as_subclass(QuantizedTensor)
        output.encoding = encoding
        return output.to(dtype)


class QuantizeDequantize(MinMaxQuantizer):
    """
    Applies quantization followed by dequantization to the input
    """
    def forward(self, input: torch.Tensor) -> DequantizedTensor:
        """
        :param input: Input to quantize and dequantize
        :return: Quantize-dequantized output
        """
        if not self.is_initialized():
            raise RuntimeError(
                'Failed to run QuantizeDequantize since quantization parameters are not initialized.'
                ' Please initialize the quantization parameters using `compute_encodings()`.'
            )

        output_dtype = internal_dtype = input.dtype

        if torch.finfo(internal_dtype).max < 2 ** self.bitwidth - 1:
            internal_dtype = torch.float32
            if torch.finfo(internal_dtype).max < 2 ** self.bitwidth - 1:
                msg = f"{internal_dtype} is unable to represent quantized output "\
                      f"of range [0, 2**(bitwidth={self.bitwidth})-1]. "\
                      "Please consider lowering the quantization bitwidth."
                raise RuntimeError(msg)

        encoding = self.get_encoding()
        output = get_backend().quantize_dequantize(input.to(internal_dtype),
                                                   encoding.scale.to(internal_dtype),
                                                   encoding.offset.to(internal_dtype),
                                                   encoding.bitwidth,
                                                   encoding.signed)
        output = output.as_subclass(DequantizedTensor)
        output.encoding = encoding
        return output.to(output_dtype)


class Dequantize(torch.nn.Module):
    """
    Applies dequantization to the input
    """
    def forward(self, input: QuantizedTensor) -> DequantizedTensor:
        # pylint: disable=no-self-use
        """
        :param input: Input to dequantize
        :return: Dequantized output
        """
        return input.dequantize()
