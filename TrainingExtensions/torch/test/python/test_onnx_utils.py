# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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

import unittest.mock
import logging

import torch
from torchvision import models

import onnx
from aimet_common.utils import AimetLogger
from aimet_torch import onnx_utils
import onnx
from torchvision.models import resnet18

class TestOnnxUtils(unittest.TestCase):

    def test_add_pytorch_node_names_to_onnx(self):

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        model = models.resnet18(pretrained=False)
        input_shape = (1, 3, 224, 224)
        torch.onnx.export(model, torch.rand(*input_shape), './data/resnet18.onnx')

        onnx_utils.OnnxSaver.set_node_names('./data/resnet18.onnx', model, input_shape)

        onnx_model = onnx.load('./data/resnet18.onnx')
        for node in onnx_model.graph.node:
            if node.op_type in ('Conv', 'Gemm', 'MaxPool'):
                self.assertTrue(node.name)

        self.assertEqual('conv1', onnx_model.graph.node[0].name)
        self.assertEqual('bn1', onnx_model.graph.node[1].name)
        self.assertEqual('relu', onnx_model.graph.node[2].name)
        self.assertEqual('maxpool', onnx_model.graph.node[3].name)

        # last op in the model is expected to be fully-connected layer
        self.assertEqual('fc', onnx_model.graph.node[-1].name)

    def test_onnx_node_name_to_input_output_names_util(self):
        """ test onxx based utility to find mapping between onnx node names and io tensors"""
        model = models.resnet18(pretrained=False)
        input_shape = (1, 3, 224, 224)
        torch.onnx.export(model, torch.rand(*input_shape), './data/resnet18.onnx')
        onnx_utils.OnnxSaver.set_node_names('./data/resnet18.onnx', model, input_shape)
        onnx_model = onnx.load('./data/resnet18.onnx')

        # Set onnx node names to corresponding pytorch layer names
        # This is necessary before the 'map_onnx_node_name_to_input_output_names' utility can be used below
        # onnx_utils.OnnxSaver.set_node_names(f.name, resnet18, input_shape)

        # Get Dict mapping node name to the input and output names
        node_to_io_dict = onnx_utils.OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)

        node_0 = onnx_model.graph.node[0]
        self.assertEqual(node_0.input, node_to_io_dict[node_0.name].inputs)
        self.assertEqual(node_0.output, node_to_io_dict[node_0.name].outputs)
