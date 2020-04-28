# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

import unittest
from unittest.mock import MagicMock

import torch.nn as nn
from torch.nn import Conv2d, Linear

from aimet_torch.layer_database import Layer
from aimet_torch.layer_selector import ConvFcLayerSelector, ConvNoDepthwiseLayerSelector


class TestLayerSelector(unittest.TestCase):

    def test_select_all_conv_layers(self):

        mock_output_shape = (1, 1, 1, 1)

        # Two regular conv layers
        layer1 = Layer(Conv2d(10, 20, 5), '', mock_output_shape)
        layer2 = Layer(Conv2d(10, 20, 5), '', mock_output_shape)
        layer3 = Layer(Conv2d(10, 10, 5, groups=10), '', mock_output_shape)

        layer_db = MagicMock()
        layer_db.__iter__.return_value = [layer1, layer2, layer3]

        layer_selector = ConvNoDepthwiseLayerSelector()
        layer_selector.select(layer_db, [])
        layer_db.mark_picked_layers.assert_called_once_with([layer1, layer2])

        # One conv and one linear layer
        layer1 = Layer(Conv2d(10, 20, 5), '', mock_output_shape)
        layer2 = Layer(Linear(10, 20), '', mock_output_shape)

        layer_db = MagicMock()
        layer_db.__iter__.return_value = [layer1, layer2]

        layer_selector.select(layer_db, [])
        layer_db.mark_picked_layers.assert_called_once_with([layer1])

        # Two regular conv layers - one in ignore list
        layer1 = Layer(Conv2d(10, 20, 5), '', mock_output_shape)
        layer2 = Layer(Conv2d(10, 20, 5), '', mock_output_shape)

        layer_db = MagicMock()
        layer_db.__iter__.return_value = [layer1, layer2]

        layer_selector.select(layer_db, [layer2.module])
        layer_db.mark_picked_layers.assert_called_once_with([layer1])

    def test_select_all_conv_and_fc_layers(self):

        mock_output_shape = (1, 1, 1, 1)

        # one regular conv layer, one depth wise conv layer and one FC layer
        layer1 = Layer(Conv2d(10, 10, 5, groups=10), '', mock_output_shape)
        layer2 = Layer(Linear(10, 20), '', mock_output_shape)
        layer3 = Layer(Conv2d(20, 40, 5), '', mock_output_shape)

        layer_db = MagicMock()
        layer_db.__iter__.return_value = [layer1, layer2, layer3]

        layer_selector = ConvFcLayerSelector()
        layer_selector.select(layer_db, [])
        layer_db.mark_picked_layers.assert_called_once_with([layer2, layer3])

        # Two regular conv layers and one FC layer - one in ignore list
        layer1 = Layer(Conv2d(10, 20, 5), '', mock_output_shape)
        layer2 = Layer(Linear(10, 20), '', mock_output_shape)
        layer3 = Layer(Conv2d(20, 40, 5), '', mock_output_shape)

        layer_db = MagicMock()
        layer_db.__iter__.return_value = [layer1, layer2, layer3]

        layer_selector.select(layer_db, [layer2.module])
        layer_db.mark_picked_layers.assert_called_once_with([layer1, layer3])
