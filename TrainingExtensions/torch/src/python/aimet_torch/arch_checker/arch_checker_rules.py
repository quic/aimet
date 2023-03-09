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

""" Utility for rules to check architecture. """

from typing import Dict
import torch

from aimet_common.utils import AimetLogger
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

def get_check_dict()-> Dict:
    """
    Get dictionary for arch check.
    :return check_dicts: {check target type: list of checks}.
    """
    check_dicts = {torch.nn.modules.conv.Conv2d: [_check_conv_channel_32_base,
                                                  _check_conv_channel_larger_than_32]}
    return check_dicts

def _check_conv_channel_32_base(node: torch.nn.Module)-> bool:
    """
    Channels should be mutilple of 32 for better performance.
    :param node: torch node to be checked.
    :return True if both model's input and output channel depth is 32's multiple.
    """
    if node.in_channels % 32 == 0 and node.out_channels % 32 == 0:
        return True
    return False

def _check_conv_channel_larger_than_32(node: torch.nn.Module)-> bool:
    """
    Channels should be at least 32 for better performance.
    :param node: torch node to be checked.
    :return model's input/output channel depth is larger than 32 or not.
    """
    if node.in_channels >= 32 and node.out_channels >= 32:
        return True
    return False
