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
""" This file contains unit tests for testing batch norm folding in ONNX """

from onnx import load_model
from onnxruntime.quantization.onnx_quantizer import ONNXModel

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from aimet_onnx.batch_norm_fold import _find_conv_bn_pairs, find_all_batch_norms_to_fold
from aimet_onnx.meta.connectedgraph import ConnectedGraph


def _convert_to_onnx_no_fold(model: torch.nn.Module, dummy_input, filename='./temp_model.onnx'):
    torch.onnx.export(model.eval(),
                      dummy_input,
                      filename,
                      training=torch.onnx.TrainingMode.TRAINING,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=False,
                      input_names=['input'],
                      output_names=['output'])
    model = ONNXModel(load_model(filename))
    return model


class TwoInputs(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(TwoInputs, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.ada = torch.nn.AdaptiveAvgPool2d(16)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc = torch.nn.Linear(1296, num_classes)

    def forward(self, *inputs):
        x1 = self.conv1(inputs[0])
        x1 = self.bn1(x1)
        x2 = self.conv2(inputs[1])
        x2 = self.bn2(x2)
        x2 = self.conv3(x2)
        x2 = self.ada(x2)
        x = x1 + x2
        x = self.relu1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BNBeforeConv(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1):
        super(BNBeforeConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 3, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.conv2 = torch.nn.Conv2d(20, 20, 3, bias=False, padding=padding, stride=stride,
                                     dilation=dilation, groups=groups)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)

        return x


class BNAfterConv(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1):
        super(BNAfterConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 3, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(20, 20, 3, bias=False, padding=padding, stride=stride,
                                     dilation=dilation, groups=groups)
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)

        return x


class BNAfterLinear(torch.nn.Module):
    def __init__(self):
        super(BNAfterLinear, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.bn1 = torch.nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)

        return x


class BNBeforeLinear(torch.nn.Module):
    def __init__(self):
        super(BNBeforeLinear, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(20)
        self.fc2 = torch.nn.Linear(20, 20)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.fc2(x)

        return x


class BNBeforeFlattenLinear(torch.nn.Module):
    def __init__(self):
        super(BNBeforeFlattenLinear, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 3, padding=1, stride=2)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.flatten = torch.nn.Flatten()
        self.fc2 = torch.nn.Linear(20 * 12 * 12, 20)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.flatten(x)
        x = self.fc2(x)

        return x


class BNAfterConvTranspose(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, output_padding=0):
        super(BNAfterConvTranspose, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3, padding=padding, stride=stride, dilation=dilation,
                                              groups=groups, output_padding=output_padding)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x


class BNBeforeConvTranspose(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, output_padding=0):
        super(BNBeforeConvTranspose, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 10, 3, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.ConvTranspose2d(10, 10, 3, padding=padding, stride=stride, dilation=dilation,
                                              groups=groups, output_padding=output_padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)

        return x


class BNAfterConv1d(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1):
        super(BNAfterConv1d, self).__init__()
        self.conv1 = torch.nn.Conv1d(10, 10, 3, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(10, 20, 3)
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(20, 15, 3)
        self.relu2 = torch.nn.ReLU()

        self.bn2 = torch.nn.BatchNorm2d(15)
        self.conv3 = torch.nn.Conv2d(15, 20, 3)

        self.conv4 = torch.nn.Conv2d(20, 20, 3)
        self.bn3 = torch.nn.BatchNorm2d(20)
        self.bn4 = torch.nn.BatchNorm2d(20)

        self.fc1 = torch.nn.Linear(5120, 10)

    def forward(self, x):
        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Non-linearity between conv and bn, not a candidate for fold
        x = self.conv2(x)
        x = self.relu2(x)

        # Case where BN can fold into an immediate downstream conv
        x = self.bn2(x)
        x = self.conv3(x)

        # No fold if there is a split between conv and BN
        x = self.conv4(x)
        bn1_out = self.bn3(x)
        bn2_out = self.bn4(x)

        x = bn1_out + bn2_out

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def _initialize_bn_params(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, _BatchNorm) and module.affine:
            with torch.no_grad():
                module.weight.copy_(torch.randn_like(module.weight))
                module.bias.copy_(torch.randn_like(module.bias))
                module.running_mean.copy_(torch.randn_like(module.bias))
                module.running_var.add_(torch.randn_like(module.bias).abs())


class TestBatchNormFold:
    """ Test methods for BatchNormFold"""

    def test_find_batch_norms_to_fold(self):
        model = MyModel().eval()
        _initialize_bn_params(model)

        input_shape = (2, 10, 24, 24)
        x = torch.randn(*input_shape, requires_grad=True)

        # Export the model
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          "./model_single_residual.onnx",
                          # where to save the model (can be a file or file-like object),
                          training=torch.onnx.TrainingMode.TRAINING,
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])
        model = ONNXModel(load_model('./model_single_residual.onnx'))

        connected_graph = ConnectedGraph(model)
        bn_info = _find_conv_bn_pairs(connected_graph)
        conv1 = connected_graph.get_op_from_module_name('Conv_0')
        conv3 = connected_graph.get_op_from_module_name('Conv_6')
        assert len(bn_info.keys()) == 2
        assert connected_graph.get_op_from_module_name('BatchNormalization_1') == bn_info[conv1].output_bn
        assert connected_graph.get_op_from_module_name('BatchNormalization_5') == bn_info[conv3].input_bn

    def test_find_bn_before_linear(self):
        x = torch.randn((32, 10))
        model = BNBeforeLinear()
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        bn_info = _find_conv_bn_pairs(conn_graph)
        assert len(bn_info.keys()) == 1
        assert 'Gemm' in list(bn_info.keys())[0].name

    def test_find_bn_before_flatten(self):
        x = torch.randn((2, 10, 24, 24))
        model = BNBeforeFlattenLinear()
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        bn_info = _find_conv_bn_pairs(conn_graph)
        linear_layer = conn_graph.get_op_from_module_name('Gemm_4')
        assert len(bn_info.keys()) == 1
        assert linear_layer in bn_info.keys()
        assert bn_info[linear_layer].input_bn == conn_graph.get_op_from_module_name('BatchNormalization_2')

    def test_find_bn_after_linear(self):
        x = torch.randn((32, 10))
        model = BNAfterLinear()
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        bn_info = _find_conv_bn_pairs(conn_graph)
        linear_layer = conn_graph.get_op_from_module_name('Gemm_0')
        assert len(bn_info.keys()) == 1
        assert linear_layer in bn_info.keys()
        assert bn_info[linear_layer].output_bn == conn_graph.get_op_from_module_name('BatchNormalization_1')

    def test_find_bn_after_convtranspose(self):
        x = torch.randn((2, 10, 24, 24))
        model = BNAfterConvTranspose()
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        bn_info = _find_conv_bn_pairs(conn_graph)
        conv_layer = conn_graph.get_op_from_module_name('ConvTranspose_0')
        assert len(bn_info.keys()) == 1
        assert conv_layer in bn_info.keys()
        assert bn_info[conv_layer].output_bn == conn_graph.get_op_from_module_name('BatchNormalization_1')

    def test_find_bn_after_conv1d(self):
        x = torch.randn((2, 10, 24))
        model = BNAfterConv1d()
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        bn_info = _find_conv_bn_pairs(conn_graph)
        conv_layer = conn_graph.get_op_from_module_name('Conv_0')
        assert len(bn_info.keys()) == 1
        assert conv_layer in bn_info.keys()
        assert bn_info[conv_layer].output_bn == conn_graph.get_op_from_module_name('BatchNormalization_1')

    def test_filter_bn_before_conv_transpose(self):
        x = torch.randn((2, 10, 24, 24))
        model = BNBeforeConvTranspose()
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        conv_bn, bn_conv = find_all_batch_norms_to_fold(conn_graph)
        assert not conv_bn
        assert not bn_conv


    def test_filter_bn_after_conv_transpose(self):
        x = torch.randn((2, 10, 24, 24))
        model = BNAfterConvTranspose()
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        conv_bn, bn_conv = find_all_batch_norms_to_fold(conn_graph)
        assert len(conv_bn) == 1
        assert not bn_conv
        model = BNAfterConvTranspose(groups=2)
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        conv_bn, bn_conv = find_all_batch_norms_to_fold(conn_graph)
        assert not conv_bn
        assert not bn_conv
        model = BNAfterConvTranspose(groups=10)
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        conv_bn, bn_conv = find_all_batch_norms_to_fold(conn_graph)
        assert len(conv_bn) == 1
        assert not bn_conv

    def test_filter_bn_before_conv(self):
        x = torch.randn((2, 10, 24, 24))
        model = BNBeforeConv(padding=1)
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        conv_bn, bn_conv = find_all_batch_norms_to_fold(conn_graph)
        # should not fold if there is zero padding
        assert not conv_bn
        assert not bn_conv
        model = BNBeforeConv(padding=0)
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        conv_bn, bn_conv = find_all_batch_norms_to_fold(conn_graph)
        assert not conv_bn
        assert len(bn_conv) == 1
        model = BNBeforeConv(groups=20)
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        conv_bn, bn_conv = find_all_batch_norms_to_fold(conn_graph)
        assert not conv_bn
        assert not bn_conv

    def test_filter_bn_after_conv(self):
        x = torch.randn((2, 10, 24, 24))
        model = BNAfterConv(padding=1)
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        conv_bn, bn_conv = find_all_batch_norms_to_fold(conn_graph)
        assert len(conv_bn) == 1
        assert not bn_conv
        model = BNAfterConv(padding=0)
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        conv_bn, bn_conv = find_all_batch_norms_to_fold(conn_graph)
        assert len(conv_bn) == 1
        assert not bn_conv
        model = BNAfterConv(groups=20)
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        conv_bn, bn_conv = find_all_batch_norms_to_fold(conn_graph)
        assert len(conv_bn) == 1
        assert not bn_conv

    def test_filter_bn_before_flatten(self):
        x = torch.randn((2, 10, 24, 24))
        model = BNBeforeFlattenLinear()
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        conv_bn, bn_conv = find_all_batch_norms_to_fold(conn_graph)
        linear_layer = conn_graph.get_op_from_module_name('Gemm_4')
        assert len(bn_conv) == 1
        assert linear_layer == bn_conv[0][1]
        assert not conv_bn


