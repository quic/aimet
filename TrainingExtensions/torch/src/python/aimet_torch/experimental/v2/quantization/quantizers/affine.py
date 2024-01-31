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
from typing import Optional, Tuple, List, Dict
import contextlib
import functools

import torch
from torch import nn

from aimet_torch.experimental.v2.utils import patch_attr, _is_expandable, StatisticsNotFoundError
from aimet_torch.experimental.v2.quantization.encoding_analyzer import EncodingAnalyzer, MinMaxEncodingAnalyzer
from aimet_torch.experimental.v2.quantization.quantizers.base import QuantizerBase
from aimet_torch.experimental.v2.quantization.backends import get_backend
from aimet_torch.experimental.v2.utils import ste_round


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
        self.symmetric = symmetric
        self.encoding_analyzer = encoding_analyzer or MinMaxEncodingAnalyzer(shape)

        if not _is_expandable(self.encoding_analyzer.observer.shape, self.shape):
            raise RuntimeError(f'Encoding analyzer of shape {self.encoding_analyzer.observer.shape} '
                               f'is incompatible with quantizer of shape {self.shape}.')

    @abc.abstractmethod
    def get_min(self) -> torch.Tensor:
        """
        Compute quantization min to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        :return: Quantization min
        """

    @abc.abstractmethod
    def get_max(self) -> torch.Tensor:
        """
        Compute quantization max to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        :return: Quantization max
        """

    @abc.abstractmethod
    def get_scale(self) -> torch.Tensor:
        """
        Compute quantization scale to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        :return: Quantization scale
        """

    @abc.abstractmethod
    def get_offset(self) -> torch.Tensor:
        """
        Compute quantization offset to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        :return: Quantization offset
        """

    @abc.abstractmethod
    def set_range(self, min: torch.Tensor, max: torch.Tensor):
        """
        Set quantization parameters to the given min-max range
        """

    @torch.no_grad()
    def get_encodings(self) -> Optional[List[Dict]]:
        """
        Returns a list of encodings, each represented as a List of Dicts
        """
        # pylint: disable=redefined-builtin

        if not self.is_initialized():
            return None

        min = self.get_min().flatten()
        max = self.get_max().flatten()
        scale = self.get_scale().flatten()
        offset = self.get_offset().flatten()
        bitwidth = self.bitwidth
        dtype = "int"
        is_symmetric = self.symmetric

        return [
            {'min': float(min_), 'max': float(max_),
             'scale': float(scale_), 'offset': float(offset_),
             'bitwidth': bitwidth, 'dtype': dtype, 'is_symmetric': str(is_symmetric)}
            for min_, max_, scale_, offset_ in zip(min, max, scale, offset)
        ]

    def extra_repr(self) -> str:
        return f'shape={self.shape}, bitwidth={self.bitwidth}, symmetric={self.symmetric}'


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
        original_forward = self.forward

        @functools.wraps(original_forward)
        def forward_wrapper(input):
            batch_statistics = self.encoding_analyzer.update_stats(input)
            dynamic_min, dynamic_max =\
                    self.encoding_analyzer.compute_encodings_from_stats(batch_statistics,
                                                                        self.bitwidth,
                                                                        self.symmetric)
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

    def get_min(self) -> Optional[torch.Tensor]:
        """
        Compute quantization min to be used for forward pass.

        NOTE: self.min may not be equal to self.get_min().
              self.get_min() returns slightly recalibrated version of self.min.

        :return: Quantization min
        """
        if not self.is_initialized():
            return None
        return self.get_scale() * self.get_offset()

    def get_max(self) -> Optional[torch.Tensor]:
        """
        Compute quantization max to be used for forward pass.

        NOTE: self.max may not be equal to self.get_max()
              self.get_max() returns slightly recalibrated version of self.max.

        :return: Quantization max
        """
        if not self.is_initialized():
            return None
        return self.get_scale() * (self.get_offset() + 2 ** self.bitwidth - 1)

    def get_scale(self) -> Optional[torch.Tensor]:
        """
        Compute quantization scale to be used for forward pass.

        :return: Quantization scale
        """
        if not self.is_initialized():
            return None

        num_bins = 2 ** self.bitwidth - 1

        if self.symmetric:
            positive_bins = num_bins // 2
            negative_bins = positive_bins + 1
            scale = torch.maximum(-self.min / negative_bins, self.max / positive_bins)
        else:
            scale = (self.max - self.min) / num_bins

        return scale

    def get_offset(self) -> Optional[torch.Tensor]:
        """
        Compute quantization offset to be used for forward pass.

        :return: Quantization offset
        """
        if not self.is_initialized():
            return None

        if self.symmetric:
            with torch.no_grad():
                offset = -torch.ones_like(self.min) * 2 ** (self.bitwidth - 1)
        else:
            offset = ste_round(self.min / self.get_scale())

        return offset

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
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param input: Input to quantize
        :return: Quantized output and scale/offset associated with it
        """
        if not self.is_initialized():
            raise RuntimeError(
                'Failed to run Quantize since quantization parameters are not initialized.'
                ' Please initialize the quantization parameters using `compute_encodings()`.'
            )

        scale = self.get_scale()
        offset = self.get_offset()
        input_q = get_backend().quantize(input, scale, offset, self.bitwidth)
        return input_q, scale, offset


class QuantizeDequantize(MinMaxQuantizer):
    """
    Applies quantization followed by dequantization to the input
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        :param input: Input to quantize and dequantize
        :return: Quantize-dequantized output
        """
        if not self.is_initialized():
            raise RuntimeError(
                'Failed to run QuantizeDequantize since quantization parameters are not initialized.'
                ' Please initialize the quantization parameters using `compute_encodings()`.'
            )

        scale = self.get_scale()
        offset = self.get_offset()
        return get_backend().quantize_dequantize(input, scale, offset, self.bitwidth)


class Dequantize(torch.nn.Module):
    """
    Applies dequantization to the input
    """
    def forward(self,
                input: torch.Tensor,
                scale: torch.Tensor,
                offset: torch.Tensor) -> torch.Tensor:
        # pylint: disable=no-self-use
        """
        :param input: Input to dequantize
        :param scale: Quantization scale
        :param offset: Quantization offset
        :return: Dequantized output
        """
        return get_backend().dequantize(input, scale, offset)
