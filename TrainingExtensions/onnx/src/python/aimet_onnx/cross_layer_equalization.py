# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Cross Layer Equalization

Some terminology for this code.
CLS set: Set of layers (2 or 3) that can be used for cross-layer scaling
Layer groups: Groups of layers that are immediately connected and can be decomposed further into CLS sets
"""

from typing import Tuple, List, Union
import numpy as np

from aimet_common.utils import AimetLogger
from aimet_common.connected_graph.connectedgraph import get_ordered_ops

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

ClsSet = Union[Tuple['Conv', 'Conv'],
               Tuple['Conv', 'Conv', 'Conv']]
ScaleFactor = Union[np.ndarray, Tuple[np.ndarray]]
cls_supported_layer_types = ['Conv', 'ConvTranspose']
cls_supported_activation_types = ['Relu', 'PRelu']


def get_ordered_list_of_conv_modules(list_of_starting_ops: List) -> List:
    """
    Finds order of nodes in graph
    :param list_of_starting_ops: list of starting ops for the model
    :return: List of names in graph in order
    """
    module_list = get_ordered_ops(list_of_starting_ops)
    module_list = [[module.dotted_name, module] for module in module_list if module.type in cls_supported_layer_types]
    return module_list
