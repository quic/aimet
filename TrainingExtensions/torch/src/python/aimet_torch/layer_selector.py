# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018-19, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Selects layers for compression based on different criteria """

from typing import List

import torch
import torch.nn

from aimet_common.defs import LayerCompRatioPair
from aimet_common.layer_database import LayerDatabase
from aimet_common.layer_selector import LayerSelector


class ConvFcLayerSelector(LayerSelector):
    """
    Selects conv and fc layers
    """

    def select(self, layer_db: LayerDatabase, modules_to_ignore: List[torch.nn.Module]):
        selected_layers = []
        for layer in layer_db:
            if layer.module in modules_to_ignore:
                continue

            if isinstance(layer.module, torch.nn.Linear):
                selected_layers.append(layer)

            elif isinstance(layer.module, torch.nn.Conv2d) and (layer.module.groups == 1):
                selected_layers.append(layer)

        layer_db.mark_picked_layers(selected_layers)


class ConvNoDepthwiseLayerSelector(LayerSelector):
    """
    Selects conv layers (non-depthwise) for compression
    """

    def select(self, layer_db: LayerDatabase, modules_to_ignore: List[torch.nn.Module]):
        selected_layers = []
        for layer in layer_db:
            if layer.module in modules_to_ignore:
                continue

            if isinstance(layer.module, torch.nn.Conv2d) and (layer.module.groups == 1):
                selected_layers.append(layer)

        layer_db.mark_picked_layers(selected_layers)


class ManualLayerSelector(LayerSelector):
    """
    Marks layers which were manually selected and passed in by the user
    """

    def __init__(self, layer_comp_ratio_pairs: List[LayerCompRatioPair]):
        self._layer_comp_ratio_pairs = layer_comp_ratio_pairs

    def select(self, layer_db: LayerDatabase, _modules_to_ignore):

        selected_layers = [pair.layer for pair in self._layer_comp_ratio_pairs]
        layer_db.mark_picked_layers(selected_layers)
