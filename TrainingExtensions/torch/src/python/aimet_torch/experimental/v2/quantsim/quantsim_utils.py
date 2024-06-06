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
""" Experimental quantsim utilities """

import torch
from aimet_common.utils import AimetLogger
from aimet_torch.v2.quantization.affine.quantizer import QuantizeDequantize

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

def clip_weights_to_7f7f(sim: 'QuantizationSimModel'):
    """
    Clip sim model weights which are 16 bit symmetric to have a max of 0x7f7f when quantized.

    :param sim: Quantsim model to clip weights for
    """
    affected_layers = []
    for name, quant_layer in sim.quant_wrappers():
        # pylint: disable=too-many-boolean-expressions
        if 'weight' in quant_layer.param_quantizers and \
                quant_layer.param_quantizers['weight'] is not None and \
                quant_layer.param_quantizers['weight'].bitwidth == 16 and \
                isinstance(quant_layer.param_quantizers['weight'], QuantizeDequantize) and \
                quant_layer.param_quantizers['weight'].symmetric and \
                quant_layer.param_quantizers['weight'].is_initialized():
            clipped_weight = torch.minimum(quant_layer.weight,
                                           quant_layer.param_quantizers['weight'].get_scale() * 32639)
            with torch.no_grad():
                quant_layer.weight.copy_(clipped_weight)

            affected_layers.append(name)
    logger_str = f'Clipping weights of the following layers to 0x7f7f max quantized value: {affected_layers}'
    logger.debug(logger_str)
