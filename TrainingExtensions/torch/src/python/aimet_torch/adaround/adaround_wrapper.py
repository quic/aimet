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

""" Custom Wrapper for quantizing weights using Adaround """

import abc
import contextlib
from typing import Tuple
import torch
import torch.nn

# Import AIMET specific modules
import aimet_common.AimetTensorQuantizer as AimetTensorQuantizer
from aimet_common.defs import AdaroundConstants, MAP_QUANT_SCHEME_TO_PYMO
from aimet_torch.tensor_quantizer import StaticGridPerChannelQuantizer
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch.quantsim_straight_through_grad import broadcast_to_tensor
from aimet_torch.v2.utils import patch_attr


class AdaroundWrapperBase(abc.ABC, torch.nn.Module):
    """
    Adaround base class
    """
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Apply adaround and run forward function of the wrapped module
        """

    @abc.abstractmethod
    def get_original_module(self) -> torch.nn.Module:
        """
        Returns original module so that we can check its
        module type or access its weight
        """

    @abc.abstractmethod
    def apply_adaround(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply adaround to the input tensor
        """

    @property
    def weight(self) -> torch.Tensor:
        """
        Returns the weight of the original model
        """
        return getattr(self.get_original_module(), self.weight_name)

    @property
    def weight_name(self) -> str:
        """
        Returns the name of the weight to apply adaround
        """
        return 'weight'

class AdaroundWrapper(AdaroundWrapperBase):
    """
    Basic adaround wrapper class for AIMET v1
    """
    def __init__(self, module: QcQuantizeWrapper):
        super().__init__()
        assert self.weight_name in module.param_quantizers
        self.module_to_wrap = module
        self._init_param()

    def forward(self, *args, **kwargs):
        """
        Temporarily replace weight of the wrapped module by adarounded weight
        and run forward function of wrapped module
        """
        origianl_module = self.get_original_module()
        weight = self.weight
        if self._is_weight_quantizer_enabled():
            weight = self.apply_adaround(weight)

        with self._disable_weight_quantizer(), \
            patch_attr(origianl_module, self.weight_name, weight):
            return self.module_to_wrap.forward(*args, **kwargs)

    def get_original_module(self) -> torch.nn.Module:
        """
        Returns original module so that we can check its
        module type or access its weight
        """
        # pylint: disable=protected-access
        return self.module_to_wrap._module_to_wrap

    def apply_adaround(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply adaround to the input tensor
        """
        input_dtype = tensor.dtype
        alpha = self.alpha.to(device=tensor.device, dtype=tensor.dtype)

        # Scale the tensor
        tensor = torch.floor(tensor / self.broadcasted_delta)

        # Soft rounding maps alpha parameter between zero and one using
        # rectified sigmoid function and hard rounding maps it to exactly zero or one
        if self.use_soft_rounding:
            h_alpha = torch.clamp(torch.sigmoid(alpha) * (AdaroundConstants.ZETA - AdaroundConstants.GAMMA) +
                                  AdaroundConstants.GAMMA, 0, 1)
        else:
            h_alpha = (alpha >= 0).to(tensor.dtype)

        # Adaround the tensor
        tensor = tensor + h_alpha

        # Quantize and de-quantize the tensor
        tensor_quant = torch.clamp(tensor - self.broadcasted_offset, self.clip_min, self.clip_max)
        tensor_dequant = (tensor_quant + self.broadcasted_offset) * self.broadcasted_delta

        return tensor_dequant.to(input_dtype)

    @contextlib.contextmanager
    def _disable_weight_quantizer(self):
        """
        Temporarily disable weight quantizer
        """
        weight_quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        is_enabled = weight_quantizer.enabled
        weight_quantizer.enabled = False
        yield
        weight_quantizer.enabled = is_enabled

    def _is_weight_quantizer_enabled(self) -> bool:
        """
        Returns true if the weight quantizer is enabled
        """
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        return quantizer.enabled

    def _get_weight_quantizer_channel_axis(self) -> int:
        """
        Returns channel axis of the current weight quantizer
        """
        # pylint: disable = protected-access
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        if isinstance(quantizer, StaticGridPerChannelQuantizer):
            return quantizer._ch_axis
        return 0

    def _get_weight_quantizer_delta_and_offset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns delta and offset of the weight quantizer
        """
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        if isinstance(quantizer.encoding, list):
            # pylint: disable = protected-access
            cpp_op = AimetTensorQuantizer.AimetTensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[quantizer.quant_scheme])
            delta, offset = cpp_op.makeDeltaOffsetTensor(self.weight.device, quantizer.encoding)
        else:
            delta, offset = quantizer.encoding.delta, quantizer.encoding.offset

        ch_axis = self._get_weight_quantizer_channel_axis()
        return broadcast_to_tensor(self.weight, delta, ch_axis), broadcast_to_tensor(self.weight, offset, ch_axis)

    def _get_weight_quantizer_bitwidth(self) -> int:
        """
        Returns bitwidth of the weight quantizer
        """
        quantizer = self.module_to_wrap.param_quantizers[self.weight_name]
        return quantizer.bitwidth

    def _init_param(self):
        """
        Initialize adaround parameter using the original module
        """
        self.broadcasted_delta, self.broadcasted_offset = self._get_weight_quantizer_delta_and_offset()
        self.alpha = self._generate_alpha_parameter(self.weight, self.broadcasted_delta)
        self.bitwidth = self._get_weight_quantizer_bitwidth()
        self.use_soft_rounding = True
        self.clip_max = 2 ** self.bitwidth - 1
        self.clip_min = 0

    @staticmethod
    def _generate_alpha_parameter(tensor: torch.Tensor, delta: torch.Tensor) -> torch.nn.Parameter:
        """
        Initializes alpha parameter, same shape as the weight tensor
        :param tensor: The weight tensor to be ada rounded
        """
        tensor_floor = torch.floor(tensor / delta)
        tensor = (tensor / delta) - tensor_floor
        alpha = - torch.log((AdaroundConstants.ZETA - AdaroundConstants.GAMMA) / (tensor - AdaroundConstants.GAMMA) - 1)

        # Even if the input is float16, alpha has to be kept in float32
        # in order to be updated by the optimizer
        return torch.nn.Parameter(alpha.float(), requires_grad=True)
