# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Weight padding API"""
# pylint: disable=protected-access
from typing import Dict
import torch
import aimet_common.libpymo as libpymo
from aimet_common.defs import MAP_ROUND_MODE_TO_PYMO
from aimet_common.quantsim import recompute_grid_params
from aimet_torch.qc_quantize_op import StaticGridQuantWrapper
from aimet_torch.quantsim import QuantizationSimModel


class WeightPaddingParams:
    """
    Handles bitwidth parameters for weight padding
    """

    def __init__(self, simulated_bw, target_kernel_bw):
        """
        :param simulated_bw: Simulated bitwidth
        :param target_kernel_bw: Target kernel bitwith
        """
        self.simulated_bw = simulated_bw
        self.target_kernel_bw = target_kernel_bw


def weight_pad(sim: QuantizationSimModel, layer_bw_dict: Dict[str, WeightPaddingParams]):
    """
    This method simulates low bit-width quantization on higher bit-width hardware to pad weights
    with consecutive zeros. This weight padding helps reduce power consumption of neural networks.

    :param sim: QuantSim model whose weights will be padded
    :param layer_bw_dict: Dictionary with layer names as keys ands WeightPaddingParams as values
    :return None
    """
    # iterate through quant wrappers in model
    for layer_name, layer in sim.model.named_modules():
        if isinstance(layer, StaticGridQuantWrapper):
            # access bitwidth params per layer
            bw_values = layer_bw_dict[layer_name]

            param_quant_dict = layer.param_quantizers
            # checks if weights and proper bitwidth params present
            if 'weight' in param_quant_dict and bw_values.target_kernel_bw > bw_values.simulated_bw:
                # access weights associated with param quantizer per layer
                param_weight_quant = param_quant_dict['weight']
                layer_weights = layer._module_to_wrap.weight

                # compute encodings with lower simulated bitwidth
                param_weight_quant.encoding = recompute_grid_params(param_weight_quant.encoding,
                                                                    bw_values.simulated_bw,
                                                                    use_symmetric_encoding=False)
                # perform quant dequant on weights
                quant_dequant = param_weight_quant.quantize_dequantize(layer_weights.clone(),
                                                                       MAP_ROUND_MODE_TO_PYMO['nearest'])
                # update weights
                layer._module_to_wrap.weight = torch.nn.Parameter(quant_dequant.clone())

                # recompute encodings with updated weights and target kernel bitwidth
                param_weight_quant.update_encoding_stats(quant_dequant.clone())
                param_weight_quant.encoding = recompute_encodings(param_weight_quant.encoding, bw_values)


def recompute_encodings(encoding: libpymo.TfEncoding(), bw_params: WeightPaddingParams):
    """
    Recomputes encodings to account for adjusted quantization scale

    :param encoding: Initial encoding
    :param bw_params: Bitwidth parameters (simulated bitwidth, kernel bitwidth)
    :return: Updated encoding
    :raises AssertionError if simulated bitwidth is not less than kernel bitwidth
    """
    updated_encoding = libpymo.TfEncoding()
    updated_encoding.bw = bw_params.target_kernel_bw
    updated_encoding.delta = recompute_scale(encoding.delta, bw_params)
    updated_encoding.offset = round(encoding.min / updated_encoding.delta)
    updated_encoding.min = updated_encoding.delta * updated_encoding.offset
    updated_encoding.max = updated_encoding.min + updated_encoding.delta * ((2 ** updated_encoding.bw) - 1)
    return updated_encoding


def recompute_scale(initial_scale: float, bw_params: WeightPaddingParams):
    """
    Adjusts quantization scale to account for shifted weights.

    :param initial_scale: Initial quantization scale
    :param bw_params: Bitwidth parameters (simulated bitwidth, kernel bitwidth)
    :return: Updated scale
    :raises AssertionError if simulated bitwidth is not less than kernel bitwidth
   """
    bitwidth_diff = bw_params.target_kernel_bw - bw_params.simulated_bw
    assert bitwidth_diff > 0

    initial_scale /= 2 ** bitwidth_diff
    return initial_scale
