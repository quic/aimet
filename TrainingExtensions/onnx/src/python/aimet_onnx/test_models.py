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
from aimet_torch.examples.test_models import SingleResidualWithAvgPool, ModelWithTwoInputs

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