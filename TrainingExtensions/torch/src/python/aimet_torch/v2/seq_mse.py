# /usr/bin/env python
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
""" Sequential MSE implementation """

from typing import List, Optional, Tuple
import contextlib
import torch
from torch import nn

from aimet_torch.seq_mse import SequentialMse as V1SequentialMse
from aimet_torch.seq_mse import SeqMseParams as V1SeqMseParams
from aimet_torch.seq_mse import SUPPORTED_MODULES
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine import AffineQuantizerBase
from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
from aimet_torch.v2.nn.base import BaseQuantizationMixin
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.utils import reduce, _is_reducible


SeqMseParams = V1SeqMseParams


def _observe(x_min: torch.Tensor,
             x_max: torch.Tensor,
             num_steps: int,
             symmetric: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    encoding_analyzer = MinMaxEncodingAnalyzer(x_min.shape)
    min, max = encoding_analyzer.compute_dynamic_encodings(torch.stack([x_min, x_max]),
                                                           num_steps=num_steps,
                                                           is_symmetric=symmetric)
    return min, max


class SequentialMse(V1SequentialMse):
    """
    Sequentially minimizing activation MSE loss in layer-wise way to decide optimal param quantization encodings.
    """
    @staticmethod
    def compute_all_param_encodings(sim: QuantizationSimModel):
        """
        Compute encodings for all parameters, needed for initializing Sequential MSE

        :param sim: Quant sim
        """
        for _, qmodule in sim.named_qmodules():
            qmodule._compute_param_encodings(overwrite=True) # pylint: disable=protected-access

    @staticmethod
    @contextlib.contextmanager
    def temporarily_disable_quantizers(
            model: torch.nn.Module,
            sim: QuantizationSimModel,
            modules_to_exclude: Optional[List[torch.nn.Module]],
    ):
        """
        For given quantsim model, disable quantizers needed to be diabled before applying sequential MSE.

        :param model: Original fp32 model
        :param sim: QuantizationSimModel object
        :param modules_to_exclude: List of supported modules to exclude when applying Sequential MSE
        :return: List of quantizers to be disabled.
        """
        # pylint: disable=protected-access
        name_to_fp32_module_dict = {}
        for name, fp32_module in model.named_modules():
            name_to_fp32_module_dict[name] = fp32_module

        original_input_quantizers = {}
        original_output_quantizers = {}
        original_param_quantizers = {}
        for name, qmodule in sim.named_qmodules():
            original_input_quantizers[name] = qmodule.input_quantizers
            original_output_quantizers[name] = qmodule.output_quantizers
            qmodule.input_quantizers = nn.ModuleList([None for _ in qmodule.input_quantizers])
            qmodule.output_quantizers = nn.ModuleList([None for _ in qmodule.output_quantizers])

            if not isinstance(qmodule, SUPPORTED_MODULES):
                original_param_quantizers[name] = qmodule.param_quantizers
                qmodule.param_quantizers = nn.ModuleDict({key: None for key in qmodule.param_quantizers.keys()})

            # disable param quantizers from exclusion list
            if modules_to_exclude:
                with contextlib.suppress(KeyError):
                    fp32_module = name_to_fp32_module_dict[name]
                    if fp32_module in modules_to_exclude:
                        original_param_quantizers[name] = qmodule.param_quantizers
                        qmodule.param_quantizers = nn.ModuleDict({key: None for key in qmodule.param_quantizers.keys()})

        yield

        for name, qmodule in sim.named_qmodules():
            qmodule.input_quantizers = original_input_quantizers[name]
            qmodule.output_quantizers = original_output_quantizers[name]

            if name in original_param_quantizers:
                qmodule.param_quantizers = original_param_quantizers[name]

    @staticmethod
    def compute_param_encodings(quantizer: QuantizerBase,
                                x_min: torch.Tensor,
                                x_max: torch.Tensor):
        """
        Compute encodings for parameter quantizer using given x_min and x_max values.

        :param quantizer: Tensor quantizer
        :param x_min: min values
        :param x_max: max values
        """
        # Unsqueeze x_min/x_max until they become reducible to quantizer.min/max
        while x_min.dim() < quantizer.min.dim():
            x_min = x_min[..., None]
        while x_max.dim() < quantizer.max.dim():
            x_max = x_max[..., None]
        assert _is_reducible(x_min.shape, quantizer.min.shape)
        assert _is_reducible(x_max.shape, quantizer.max.shape)

        x_min = reduce(x_min, quantizer.shape, torch.min).values
        x_max = reduce(x_max, quantizer.shape, torch.max).values

        num_steps = 2 ** quantizer.bitwidth - 1
        symmetric = quantizer.symmetric

        # The values of x_min and x_max don't necessarily satisfy the symmetry constraints.
        # Therefore, we need to adjust their values to ensure min and max are in symmetric grids.
        min, max = _observe(x_min, x_max, num_steps=num_steps, symmetric=symmetric)

        with torch.no_grad():
            quantizer.min.copy_(min)
            quantizer.max.copy_(max)

    @staticmethod
    def _is_symmetric_quantizer(quantizer: AffineQuantizerBase):
        # pylint: disable=protected-access
        return quantizer._symmetric

    @staticmethod
    def _freeze_quantizer_encoding(quantizer: QuantizerBase):
        # pylint: disable=protected-access
        quantizer.requires_grad_(False)
        quantizer.allow_overwrite(False)

    @staticmethod
    def _get_quantized_weight(quant_module: BaseQuantizationMixin):
        w = quant_module.weight
        return quant_module.param_quantizers['weight'](w)

    @staticmethod
    def _get_original_module(quant_module: BaseQuantizationMixin):
        return quant_module


# Global variables for compatibility
apply_seq_mse = SequentialMse.apply_seq_mse
get_candidates = SequentialMse.get_candidates
optimize_module = SequentialMse.optimize_module
