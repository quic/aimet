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

import copy
from typing import Optional, Tuple
import contextlib
from collections import OrderedDict
import functools
import weakref

import torch
from torch import nn

from aimet_torch.experimental.v2.utils import patch_attr, patch_param, StatisticsNotFoundError
from aimet_torch.experimental.v2.quantization.encoding_analyzer import EncodingAnalyzer, MinMaxEncodingAnalyzer
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
    :param encoding_analyzer: Encoding analyzer for calibrating quantization encodings.
                              (default: absolute min-max encoding analyzer)
    """

    min: torch.nn.Parameter
    max: torch.nn.Parameter

    def __init__(self, shape, bitwidth: int, symmetric: bool, encoding_analyzer: EncodingAnalyzer = None):
        super().__init__()
        self.shape = shape
        self.bitwidth = bitwidth
        self.symmetric = symmetric
        self.encoding_analyzer = encoding_analyzer or MinMaxEncodingAnalyzer(shape)

        # param_name -> (weakref of initial parameter, version info of the initial parameter)
        # This info will be used for judging whether the current parameter has ever been
        # initialized after it was instantiated.
        self._initial_parameters = OrderedDict()

        # Raw quantization parameters
        self.register_quantization_parameter('min', nn.Parameter(-torch.ones(self.shape)))
        self.register_quantization_parameter('max', nn.Parameter(torch.ones(self.shape)))

    @torch.no_grad()
    def __deepcopy__(self, memo):
        self_copy = self.__new__(type(self))
        self_copy.__dict__ = copy.deepcopy(self.__dict__, memo)

        for name, param in self_copy.named_parameters():
            # Register parameters to the copied quantizer
            self_copy.register_quantization_parameter(name, param)

            # If the parameter has been already initialized,
            # artificially increment the parameter version to mark as initialized
            if self._is_initialized(name):
                param.mul_(1.)

        return self_copy

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_initial_parameters')
        state['initialized_parameters'] = [param_name for param_name, _ in self.named_parameters()
                                           if self._is_initialized(param_name)]
        return state

    @torch.no_grad()
    def __setstate__(self, state):
        initialized_parameters = state.pop('initialized_parameters')
        self.__dict__.update(state)

        self._initial_parameters = OrderedDict()
        for param_name, param in self.named_parameters():
            # Register parameters to the loaded quantizer
            self.register_quantization_parameter(param_name, param)

            # If the parameter has been already initialized,
            # artificially increment the parameter version to mark as initialized
            if param_name in initialized_parameters:
                param.mul_(1.)

    def register_quantization_parameter(self, name: str, param: nn.Parameter):
        """
        Register quantization parameter.
        """
        # pylint: disable=protected-access

        self.register_parameter(name, param)
        param = getattr(self, name)
        self._initial_parameters[name] = (weakref.ref(param), param._version)

    def _is_initialized(self, param_name) -> bool:
        # pylint: disable=protected-access

        initial_param_weakref, initial_param_version = self._initial_parameters[param_name]
        initial_param = initial_param_weakref()

        if initial_param is None:
            # The initial parameter object doesn't exist in memory space anymore.
            return True

        current_param = getattr(self, param_name)

        if current_param is initial_param and current_param._version == initial_param_version:
            # 1. Current parameter is the identical object as the initial parameter
            # 2. The version nubmer of the current parameter never changed
            return False

        return True

    def is_initialized(self) -> bool:
        """
        Returns true if the quantization parameters are initialized.
        """
        for param_name, _ in self.named_parameters():
            if not self._is_initialized(param_name):
                return False
        return True

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
                                                                        self.bitwidth,
                                                                        self.symmetric)
            with patch_param(self, 'min', dynamic_min),\
                    patch_param(self, 'max', dynamic_max):
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

            with torch.no_grad():
                self.min.copy_(min)
                self.max.copy_(max)
        finally:
            self.encoding_analyzer.reset_stats()

    def extra_repr(self) -> str:
        return f'shape={self.shape}, bitwidth={self.bitwidth}, symmetric={self.symmetric}'


class Quantize(_QuantizerBase):
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
