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
"""Dummy models for testing"""
import numpy as np
import torch
from onnx import helper, numpy_helper, OperatorSetIdProto, TensorProto, load_model
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from aimet_torch.examples.test_models import SingleResidualWithAvgPool, ModelWithTwoInputs, TransposedConvModel, \
    ConcatModel, HierarchicalModel, TransposedConvModelWithoutBN
from aimet_torch.examples.mobilenet import MockMobileNetV1

# pylint: disable=no-member
def build_dummy_model():
    """BUild dummy ONNX model for testing"""
    op = OperatorSetIdProto()
    op.version = 13
    input_info = helper.make_tensor_value_info(name='input', elem_type=TensorProto.FLOAT,
                                               shape=[1, 3, 32, 32])

    output_info = helper.make_tensor_value_info(name='output', elem_type=TensorProto.FLOAT,
                                                shape=[1, 10])
    conv_node = helper.make_node('Conv',
                                 ['input', 'conv_w', 'conv_b'],
                                 ['3'],
                                 'conv',
                                 kernel_shape=[3, 3],
                                 pads=[1, 1, 1, 1],)
    relu_node = helper.make_node('Relu',
                                 ['3'],
                                 ['4'],
                                 'relu')
    maxpool_node = helper.make_node('MaxPool',
                                    ['4'],
                                    ['5'],
                                    'maxpool',
                                    kernel_shape=[3, 3],
                                    pads=[1, 1, 1, 1],
                                    strides=[2, 2],)

    flatten_node = helper.make_node('Flatten',
                                    ['5'],
                                    ['6'],
                                    'flatten')
    fc_node = helper.make_node('Gemm',
                               ['6', 'fc_w', 'fc_b'],
                               ['output'],
                               'fc')

    conv_w_init = numpy_helper.from_array(np.random.rand(1, 3, 3, 3).astype(np.float32), 'conv_w')
    conv_b_init = numpy_helper.from_array(np.random.rand(1).astype(np.float32), 'conv_b')
    fc_w_init = numpy_helper.from_array(np.random.rand(256, 10).astype(np.float32), 'fc_w')
    fc_b_init = numpy_helper.from_array(np.random.rand(10).astype(np.float32), 'fc_b')

    onnx_graph = helper.make_graph([conv_node, relu_node, maxpool_node, flatten_node, fc_node],
                                   'dummy_graph', [input_info], [output_info],
                                   [conv_w_init, conv_b_init, fc_w_init, fc_b_init])

    model = helper.make_model(onnx_graph, opset_imports=[op])

    return model

def single_residual_model():
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    model = SingleResidualWithAvgPool()

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_single_residual.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model = ONNXModel(load_model('./model_single_residual.onnx'))
    return model

def multi_input_model():
    x = (torch.rand(32, 1, 28, 28, requires_grad=True), torch.rand(32, 1, 28, 28, requires_grad=True))
    model = ModelWithTwoInputs()

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_multi_input.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model = ONNXModel(load_model('./model_multi_input.onnx'))
    return model

def transposed_conv_model():
    x = torch.randn(10, 10, 4, 4, requires_grad=True)
    model = TransposedConvModel()

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_transposed_conv.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model = ONNXModel(load_model('./model_transposed_conv.onnx'))
    return model

def transposed_conv_model_without_bn():
    x = torch.randn(10, 10, 4, 4, requires_grad=True)
    model = TransposedConvModelWithoutBN()

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_transposed_conv_without_bn.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model = ONNXModel(load_model('./model_transposed_conv_without_bn.onnx'))
    return model

def depthwise_conv_model():
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model = MockMobileNetV1()

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_mock_mobilenet.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model = ONNXModel(load_model('./model_mock_mobilenet.onnx'))
    return model

def concat_model():
    x = (torch.rand(1, 3, 8, 8, requires_grad=True), torch.rand(1, 3, 8, 8, requires_grad=True),
         torch.rand(1, 3, 8, 8, requires_grad=True))
    model = ConcatModel()

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./concat_model.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model = ONNXModel(load_model('./concat_model.onnx'))
    return model

def hierarchical_model():
    conv_shape = (1, 64, 32, 32)
    inp_shape = (1, 3, 32, 32)
    seq_shape = (1, 3, 8, 8)
    [conv_shape, inp_shape, conv_shape, inp_shape, seq_shape]
    x = (torch.rand(1, 64, 32, 32, requires_grad=True), torch.rand(1, 3, 32, 32, requires_grad=True),
         torch.rand(1, 64, 32, 32, requires_grad=True), torch.rand(1, 3, 32, 32, requires_grad=True),
         torch.rand(1, 3, 8, 8, requires_grad=True))
    model = HierarchicalModel()

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./hierarchical_model.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    model = ONNXModel(load_model('./hierarchical_model.onnx'))
    return model

class BNBeforeConv(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(BNBeforeConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 3, bias=bias)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.conv2 = torch.nn.Conv2d(20, 20, 3, bias=bias, padding=padding, stride=stride,
                                     dilation=dilation, groups=groups)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)

        return x


class BNAfterConv(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(BNAfterConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 3, bias=bias)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(20, 20, 3, bias=bias, padding=padding, stride=stride,
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
    def __init__(self, bias=False):
        super(BNAfterLinear, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20, bias=bias)
        self.bn1 = torch.nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)

        return x


class BNBeforeLinear(torch.nn.Module):
    def __init__(self, bias=False):
        super(BNBeforeLinear, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20, bias=bias)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(20)
        self.fc2 = torch.nn.Linear(20, 20, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.fc2(x)

        return x


class BNBeforeFlattenLinear(torch.nn.Module):
    def __init__(self, bias=False):
        super(BNBeforeFlattenLinear, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 3, padding=1, stride=2)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.flatten = torch.nn.Flatten()
        self.fc2 = torch.nn.Linear(20 * 12 * 12, 20, bias=bias)

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
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(BNAfterConv1d, self).__init__()
        self.conv1 = torch.nn.Conv1d(10, 10, 3, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x

class BNBeforeConv1d(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):

        super(BNBeforeConv1d, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.conv1d = torch.nn.Conv1d(10, 20, 3, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1d(x)

        return x

class BNAfterDynamicMatMul(torch.nn.Module):
    def __init__(self, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(BNAfterDynamicMatMul, self).__init__()
        self.conv1d = torch.nn.Conv1d(10, 20, 3, padding=padding, stride=stride, dilation=dilation, groups=groups,
                                      bias=bias)
        self.fc1 = torch.nn.Linear(24*10, 20)
        self.flatten = torch.nn.Flatten()
        self.conv1d = torch.nn.Conv1d(10, 20, 3, padding=padding, stride=stride, dilation=dilation, groups=groups,
                                      bias=bias)
        self.bn1 = torch.nn.BatchNorm1d(1)

    def forward(self, x):
        x1 = self.conv1d(x)
        x2 = self.fc1(self.flatten(x)).unsqueeze(1)
        x = torch.matmul(x2, x1)
        x = self.bn1(x)
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

def _convert_to_onnx_no_fold(model: torch.nn.Module, dummy_input, filename='./temp_model.onnx'):
    torch.onnx.export(model.eval(),
                      dummy_input,
                      filename,
                      training=torch.onnx.TrainingMode.PRESERVE,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=False,
                      input_names=['input'],
                      output_names=['output'])
    model = ONNXModel(load_model(filename))
    return model


def _convert_to_onnx(model: torch.nn.Module, dummy_input, filename='./temp_model.onnx'):
    torch.onnx.export(model.eval(),
                      dummy_input,
                      filename,
                      training=torch.onnx.TrainingMode.EVAL,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])
    model = ONNXModel(load_model(filename))
    return model

