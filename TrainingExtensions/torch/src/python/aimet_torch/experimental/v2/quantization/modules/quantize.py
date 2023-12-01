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
# pylint: disable=redefined-builtin
""" nn.Modules for quantization operators """

from typing import Optional
import contextlib
import functools

import torch

from aimet_torch.experimental.v2.utils import patch_attr, patch_param
from aimet_torch.experimental.v2.quantization.encoding_analyzer import get_encoding_analyzer_cls
from aimet_torch.experimental.v2.quantization.backends import get_backend
from aimet_torch.experimental.v2.utils import ste_round


__all__ = ['Quantize', 'QuantizeDequantize', 'Dequantize']


class _QuantizerBase(torch.nn.Module): # pylint: disable=abstract-method
    """
    Base class for quantization modules.

    :param shape: Shape of the quantization parameters.
    :param bitwidth: Quantization bitwidth.
    :param symmetric: If True, performs symmetric quantization;
                      otherwise, performs asymmetric quantization.
    :param qscheme: Quantization scheme
    """

    min: torch.nn.Parameter
    max: torch.nn.Parameter

    def __init__(self, shape, bitwidth: int, symmetric: bool, qscheme):
        super().__init__()
        self.shape = shape
        self.bitwidth = bitwidth
        self.symmetric = symmetric
        self.qscheme = qscheme
        self.encoding_analyzer = get_encoding_analyzer_cls(qscheme)(shape)

        # Raw quantization parameters
        self.register_parameter("min", None)
        self.register_parameter("max", None)

    def is_initialized(self) -> bool:
        """
        Returns true if the quantization parameters are initialized.
        """
        return self.min is not None or self.max is not None

    def get_min(self) -> Optional[torch.Tensor]:
        """
        Compute quantization min to be used for forward pass based on raw parameters.

        :return: Quantization min
        """
        if not self.is_initialized():
            return None
        return self.get_scale() * self.get_offset()

    def get_max(self) -> Optional[torch.Tensor]:
        """
        Compute quantization max to be used for forward pass based on raw parameters.

        :return: Quantization max
        """
        if not self.is_initialized():
            return None
        return self.get_scale() * (self.get_offset() + 2 ** self.bitwidth - 1)

    def get_scale(self) -> Optional[torch.Tensor]:
        """
        Compute quantization scale to be used for forward pass based on raw parameters.

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
        Compute quantization offset to be used for forward pass based on raw parameters.

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
                                                                        self.symmetric,
                                                                        self.bitwidth)
            with patch_param(self, 'min', dynamic_min),\
                    patch_param(self, 'max', dynamic_max):
                return original_forward(input)

        try:
            with patch_attr(self, 'forward', forward_wrapper):
                yield
        except: # pylint: disable=try-except-raise
            raise
        else:
            min, max = self.encoding_analyzer.compute_encodings(self.symmetric, self.bitwidth)

            if min is None or max is None:
                return

            if not self.is_initialized():
                self.min = torch.nn.Parameter(torch.empty(self.shape))
                self.max = torch.nn.Parameter(torch.empty(self.shape))

            with torch.no_grad():
                self.min.copy_(min)
                self.max.copy_(max)
        finally:
            self.encoding_analyzer.reset_stats()


class Quantize(_QuantizerBase):
    """
    Applies quantization to the input
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        :param input: Input to quantize
        :return: Quantized output
        """
        if not self.is_initialized():
            raise RuntimeError(
                'Failed to run Quantize since quantization parameters are not initialized.'
                ' Please initialize the quantization parameters using `compute_encodings()`.'
            )

        scale = self.get_scale()
        offset = self.get_offset()
        return get_backend().quantize(input, scale, offset, self.bitwidth)


class QuantizeDequantize(_QuantizerBase):
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
