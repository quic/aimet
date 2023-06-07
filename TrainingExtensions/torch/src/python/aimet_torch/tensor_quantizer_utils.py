# /usr/bin/env python3.5
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

""" Utilities that are used for different quantization simulation features """

import math
from typing import Tuple, Union
import torch

from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def calc_params_for_native_torch_quantizer(quantizer, per_channel_enabled: bool, device: torch.device) \
        -> Tuple[Union[torch.Tensor, float], Union[torch.Tensor, int], int, int]:
    """
    This function merely transforms previously computed quant encodings to the expected pytorch function format
    :param quantizer: input or output quantizer
    :param per_channel_enabled: bool to enable/disable per-channel quantization
    :param device: device on which model is
    :return: tuple of quantization parameters used to set up native torch quantizer
    """

    numSteps = pow(2, quantizer.bitwidth) - 1
    encodings = quantizer.encoding
    if quantizer.use_strict_symmetric:
        error_msg = ('Strict symmetric is not supported by native torch quantizer')
        logger.error(error_msg)
        raise ValueError(error_msg)

    if per_channel_enabled:
        scale = torch.Tensor([encoding.delta for encoding in encodings]).to(device)
        zero_point = torch.Tensor([int(-encoding.offset) for encoding in encodings]).long().to(device)
        if quantizer.use_symmetric_encodings and (all([encoding.min < 0 for encoding in encodings])
                                                  or (not quantizer.use_unsigned_symmetric)):
            # Symmetric quantization
            q_max = math.floor(numSteps / 2)
            q_min = -math.ceil(numSteps / 2)
            zero_point = torch.zeros_like(zero_point)
        else:
            # Unsigned symmetric
            q_min, q_max = 0, numSteps

    else:
        scale = float(encodings.delta)
        zero_point = int(-encodings.offset)
        if quantizer.use_symmetric_encodings and (encodings.min < 0
                                                  or (not quantizer.use_unsigned_symmetric)):
            # Symmetric quantization
            q_max = math.floor(numSteps / 2)
            q_min = -math.ceil(numSteps / 2)
            zero_point = 0
        else:
            # Unsigned symmetric
            q_min, q_max = 0, numSteps

    return scale, zero_point, q_max, q_min
