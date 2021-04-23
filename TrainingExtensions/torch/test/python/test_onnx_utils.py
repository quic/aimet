# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

from aimet_common.utils import AimetLogger
from aimet_torch import onnx_utils
import onnx


class OutOfOrderModel(torch.nn.Module):

    def __init__(self):
        super(OutOfOrderModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(32, 32, 3)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        y1 = self.conv2(x)
        y1 = self.bn2(y1)
        y1 = self.relu2(y1)

        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = self.relu3(x1)

        return x1 + y1


class TestOnnxUtils:

    def test_add_pytorch_node_names_to_onnx_resnet(self):

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        model_name = 'resnet18'
        model = models.resnet18(pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)

        torch.onnx.export(model, dummy_input, './data/' + model_name + '.onnx')
        onnx_utils.OnnxSaver.set_node_names('./data/' + model_name + '.onnx', model, dummy_input)

        onnx_model = onnx.load('./data/' + model_name + '.onnx')
        for node in onnx_model.graph.node:
            if node.op_type in ('Conv', 'Gemm', 'MaxPool'):
                assert node.name

            for in_tensor in node.input:
                if in_tensor.endswith('weight'):
                    print("Checking " + in_tensor)
                    assert node.name == in_tensor[:-7]

    def test_add_pytorch_node_names_to_onnx_ooo(self):

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        model_name = 'out_of_order'
        model = OutOfOrderModel()
        dummy_input = torch.randn(1, 16, 20, 20)

        onnx_utils.OnnxSaver.set_node_names('./data/' + model_name + '.onnx', model, dummy_input)

        onnx_model = onnx.load('./data/' + model_name + '.onnx')
        for node in onnx_model.graph.node:
            if node.op_type in ('Conv', 'Gemm', 'MaxPool'):
                assert node.name

            for in_tensor in node.input:
                if in_tensor.endswith('weight'):
                    print("Checking " + in_tensor)
                    assert node.name == in_tensor[:-7]

    def test_onnx_node_name_to_input_output_names_util(self):
        """ test onxx based utility to find mapping between onnx node names and io tensors"""
        model = models.resnet18(pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model, dummy_input, './data/resnet18.onnx')
        onnx_utils.OnnxSaver.set_node_names('./data/resnet18.onnx', model, dummy_input)
        onnx_model = onnx.load('./data/resnet18.onnx')

        # Get Dict mapping node name to the input and output names
        node_to_io_dict,_ = onnx_utils.OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)

        node_0 = onnx_model.graph.node[0]
        assert node_0.input == node_to_io_dict[node_0.name].inputs
        assert node_0.output == node_to_io_dict[node_0.name].outputs

    def test_single_pytorch_module_mapping_to_many_onnx_nodes(self):
        """ test onxx based utility to find mapping between onnx node names and io tensors
        when more than one onnx node maps to the same torch module
        """

        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        class TwoLayerLstmModel(torch.nn.Module):
            """
            Model using torch.nn.LSTM module
            """
            def __init__(self):
                super(TwoLayerLstmModel, self).__init__()
                self.lstm = torch.nn.LSTM(input_size=3, hidden_size=5, num_layers=3)

            def forward(self, x, hx=None):
                return self.lstm(x, hx)

        model_name = 'multilayer_lstm'
        model = TwoLayerLstmModel()
        dummy_input = torch.randn(10, 1, 3)

        torch.onnx.export(model, dummy_input, './data/' + model_name + '.onnx')
        onnx_utils.OnnxSaver.set_node_names('./data/' + model_name + '.onnx', model, dummy_input)
        onnx_model = onnx.load('./data/' + model_name + '.onnx')

        lstm_nodes = [node for node in onnx_model.graph.node if node.op_type == 'LSTM']
        assert 3 == len(lstm_nodes)

        node_to_io_dict, _ = onnx_utils.OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)
        assert isinstance(node_to_io_dict['lstm'], list)
        assert 3 == len(node_to_io_dict['lstm'])

    def test_onnx_export(self):

        from aimet_torch.elementwise_ops import Add

        class ResidualLayer(torch.nn.Module):
            def __init__(self):
                super(ResidualLayer, self).__init__()
                self.conv1 = torch.nn.Conv2d(20, 20, 3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(20, 20, 3)
                self.relu2 = torch.nn.ReLU()

                self.conv3 = torch.nn.Conv2d(20, 20, 3)
                self.relu3 = torch.nn.ReLU()
                self.conv4 = torch.nn.Conv2d(20, 20, 3)
                self.relu4 = torch.nn.ReLU()

            def forward(self, x):
                y1 = self.conv1(x)
                y1 = self.relu1(y1)
                y2 = self.conv2(x)
                y2 = self.relu2(y2)

                y1 = self.conv3(y1)
                y1 = self.relu3(y1)
                y2 = self.conv4(y2)
                y2 = self.relu4(y2)

                return y1 + y2

        class TwoLevelLayer(torch.nn.Module):
            def __init__(self):
                super(TwoLevelLayer, self).__init__()
                self.conv1 = torch.nn.Conv2d(20, 20, 3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(20, 20, 3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.conv2(x)
                return x

        class CustomLayer(torch.nn.Module):
            def __init__(self):
                super(CustomLayer, self).__init__()

            def forward(self, x):
                x = x * 10
                x = torch.nn.functional.relu(x)
                return x

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv0 = torch.nn.Conv2d(10, 20, 3)
                self.conv2 = torch.nn.Conv2d(20, 20, 3)
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.custom = CustomLayer()
                self.block1 = TwoLevelLayer()
                self.block2 = ResidualLayer()
                self.add = Add()
                self.relu1 = torch.nn.ReLU()

            def forward(self, x):
                z = self.conv0(x)
                x = self.conv1(x)
                x = torch.nn.functional.relu(x)
                x = self.custom(x)
                x = self.conv2(x)
                x = self.block1(x)
                x = self.block2(x)
                x = self.add(x, x)
                x = x[:, :, 0, 0]
                x = x.reshape(x.shape[0], x.shape[1], 1, 1)
                x = self.relu1(x)
                return x

        model = MyModel()

        onnx_utils.OnnxSaver.set_node_names('./data/MyModel.onnx', model, dummy_input=torch.rand(1, 10, 24, 24))
        onnx_model = onnx.load('./data/MyModel.onnx')
        expected_conv_names = ['conv1', 'conv2', 'block1.conv1', 'block1.conv2', 'block2.conv1', 'block2.conv2',
                               'block2.conv3', 'block2.conv4']
        expected_other_node_names = ['relu1', 'add', 'block1.relu1', 'block2.relu1', 'block2.relu2', 'block2.relu3',
                                     'block2.relu4']
        not_expected_names = ['conv0']

        for node in onnx_model.graph.node:
            if node.op_type == 'Conv':
                assert node.name in expected_conv_names

        actual_node_names = [node.name for node in onnx_model.graph.node]
        for name in expected_other_node_names:
            assert name in actual_node_names
        for name in not_expected_names:
            assert name not in actual_node_names

