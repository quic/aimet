# /usr/bin/env python3.8
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
# =============================================================================
"""Quantizer utility"""
from typing import List

from aimet_tensorflow.keras.quant_sim.tensor_quantizer import TensorQuantizer
from aimet_tensorflow.keras.quantsim import QuantizationSimModel


def get_enabled_param_quantizers(sim: QuantizationSimModel) -> List[TensorQuantizer]:
    """
    For given quantsim model, get all enabled param quantizers.
    :param sim: Quantsim model.
    :return: List of enabled param quantizers.
    """
    enabled_param_quantizers = []
    for quant_wrapper in sim.quant_wrappers():
        for quantizer in quant_wrapper.param_quantizers:
            if quantizer.is_enabled():
                enabled_param_quantizers.append(quantizer)

    return enabled_param_quantizers


def get_enabled_activation_quantizers(sim: QuantizationSimModel) -> List[TensorQuantizer]:
    """
    For given quantsim model, get all enabled activation quantizers.
    :param sim: Quantsim model.
    :return: List of enabled activation quantizers.
    """
    enabled_activation_quantizers = []
    for quant_wrapper in sim.quant_wrappers():
        for quantizer in quant_wrapper.input_quantizers:
            if quantizer.is_enabled():
                enabled_activation_quantizers.append(quantizer)

        for quantizer in quant_wrapper.output_quantizers:
            if quantizer.is_enabled():
                enabled_activation_quantizers.append(quantizer)

    return enabled_activation_quantizers


def enable_disable_quantizers(quantizers: List[TensorQuantizer],
                              enabled: bool):
    """
    For given list of quantizers, set (enable/disable) quantizer's enabled.
    :param quantizers: List of quantizers.
    :param enabled: Enabled flag.
    """
    if enabled:
        for quantizer in quantizers:
            quantizer.enable()
    else:
        for quantizer in quantizers:
            quantizer.disable()
