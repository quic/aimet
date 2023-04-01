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

"""
Utility for rules to check architecture.
Node checks should follow :param node: :return bool:.
Pattern checks should follow :param connected_graph: :return list[ops]:
"""

from typing import Dict, List
import torch

from aimet_common.utils import AimetLogger

from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.batch_norm_fold import find_standalone_batchnorm_ops

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

def get_node_check_dict()-> Dict:
    """
    Get dictionary for node checks.
    :return check_dicts: {check target type: list of checks}.
    """
    check_dicts = {torch.nn.modules.conv.Conv2d: [_check_conv_channel_32_base,
                                                  _check_conv_channel_larger_than_32],
                   TorchActivations: [_activation_checks],}
    return check_dicts

def get_pattern_check_list()-> List:
    """
    Get a list of pattern checks.
    :return: List of pattern checks.
    """
    return [_check_batch_norm_fold]

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

def _activation_checks(node: torch.nn.Module)-> bool:
    """
    Common checkes for all torch activations.
    Prelu and swish (SiLU) degenerates the quantization performance.
    :param node: torch node to be checked.
    :return True if not a activation with bad bad quantization performance activations function.
    """
    _degenerating_activation_tuple = (torch.nn.modules.activation.SiLU,
                                      torch.nn.modules.activation.PReLU)

    if isinstance(node, _degenerating_activation_tuple):
        return False
    return True

def _check_batch_norm_fold(connected_graph: ConnectedGraph) -> List:
    """
    Pattern checker: return all standalone batchnorms.
    :param connected_graph: Connected_graph object.
    :return: List of stand alone (not foldable) batch norms in connected_graph.
    """
    stand_alone_bn_ops = find_standalone_batchnorm_ops(connected_graph)

    return list(stand_alone_bn_ops)

class CheckType(type):
    """ Metaclass to overwrite __instancecheck__. """
    def __instancecheck__(cls, obj):
        return cls._test(obj)

# A type class to put all torch activations together.
# pylint: disable=too-few-public-methods
class TorchActivations(metaclass=CheckType):
    """ Type class for all torch activations. """
    @classmethod
    def _test(cls, module):
        if module.__module__ == 'torch.nn.modules.activation':
            return True
        return False
