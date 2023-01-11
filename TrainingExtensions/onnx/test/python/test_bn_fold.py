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

import onnx
from onnx import load_model
from onnxruntime.quantization.onnx_quantizer import ONNXModel
import onnxruntime as rt
import numpy as np
import torchvision
import pytest

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from aimet_onnx.batch_norm_fold import _find_conv_bn_pairs, find_all_batch_norms_to_fold, fold_all_batch_norms_to_weight
from aimet_onnx.meta.connectedgraph import ConnectedGraph

from test_models import BNAfterConv, BNBeforeConv, BNAfterDynamicMatMul, BNAfterConvTranspose, BNAfterConv1d, \
                        BNAfterLinear, BNBeforeLinear, BNBeforeFlattenLinear, BNBeforeConv1d, BNBeforeConvTranspose, \
                        MyModel, _convert_to_onnx_no_fold, _convert_to_onnx


def get_outputs_after_fold(model, test_data):
    onnx.checker.check_model(model.model)
    filename = './onnx_test_model.onnx'
    onnx.save(model.model, filename)
    pairs = fold_all_batch_norms_to_weight(model.model)

    onnx.checker.check_model(model.model)
    folded_filename = './onnx_test_model_folded.onnx'
    onnx.save(model.model, folded_filename)

    sess = rt.InferenceSession(filename)
    fold_sess = rt.InferenceSession(folded_filename)

    input_name = sess.get_inputs()[0].name
    baseline_output = sess.run(None, {input_name: test_data})
    input_name = fold_sess.get_inputs()[0].name
    folded_output = fold_sess.run(None, {input_name: test_data})
    return baseline_output, folded_output, pairs


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
                          do_constant_folding=False,  # whether to execute constant folding for optimization
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
        assert 'MatMul' in list(bn_info.keys())[0].name

    def test_find_bn_before_flatten(self):
        x = torch.randn((2, 10, 24, 24))
        model = BNBeforeFlattenLinear()
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        bn_info = _find_conv_bn_pairs(conn_graph)
        linear_layer = conn_graph.get_op_from_module_name('MatMul_5')
        assert len(bn_info.keys()) == 1
        assert linear_layer in bn_info.keys()
        assert bn_info[linear_layer].input_bn == conn_graph.get_op_from_module_name('BatchNormalization_2')

    def test_find_bn_after_linear(self):
        x = torch.randn((32, 10))
        model = BNAfterLinear(bias=True)
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
        model = BNBeforeFlattenLinear(bias=True)
        model = _convert_to_onnx_no_fold(model, x)
        conn_graph = ConnectedGraph(model)
        conv_bn, bn_conv = find_all_batch_norms_to_fold(conn_graph)
        linear_layer = conn_graph.get_op_from_module_name('Gemm_4')
        assert len(bn_conv) == 1
        assert linear_layer.get_module() == bn_conv[0][1]
        assert not conv_bn

    def test_fold_bn_before_flatten_no_bias(self):
        torch.manual_seed(10)
        torch_model = BNBeforeFlattenLinear()
        torch_model.eval()
        _initialize_bn_params(torch_model)

        input_shape = (2, 10, 24, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert pairs[0][0].name == "Gemm_4"
        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    def test_fold_bn_before_flatten_no_bias_with_transpose(self):
        torch.manual_seed(10)
        torch_model = BNBeforeFlattenLinear()
        torch_model.eval()
        _initialize_bn_params(torch_model)

        input_shape = (2, 10, 24, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert pairs[0][0].name == "Gemm_5"
        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    def test_fold_resnet18(self):
        torch.manual_seed(10)
        torch_model = torchvision.models.resnet18()
        num_batchnorm = len([m for m in torch_model.modules() if isinstance(m, torch.nn.BatchNorm2d)])
        _initialize_bn_params(torch_model)

        input_shape = (2, 3, 224, 224)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)
        assert len(pairs) == num_batchnorm
        assert len(model.graph().node) == layers_orig - num_batchnorm
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    @pytest.mark.parametrize("bias", [True, False])
    def test_fold_bn_before_conv(self, bias):
        torch.manual_seed(10)
        torch_model = BNBeforeConv(bias=bias)
        torch_model.eval()
        _initialize_bn_params(torch_model)

        input_shape = (2, 10, 24, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert pairs[0][0].name == "Conv_3"
        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    def test_fold_bn_before_conv_depthwise(self):
        torch.manual_seed(10)
        torch_model = BNBeforeConv(bias=True, groups=20)
        torch_model.eval()
        _initialize_bn_params(torch_model)

        input_shape = (2, 10, 24, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert len(pairs) == 0
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.parametrize("groups", [1, 5, 20])
    def test_fold_bn_after_conv_no_bias(self, bias, padding, groups):
        torch.manual_seed(10)
        torch_model = BNAfterConv(bias=bias, padding=padding, groups=groups)
        torch_model.eval()

        input_shape = (2, 10, 24, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert pairs[0][0].name == "Conv_2"
        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    def test_fold_bn_after_transposed_conv_depthwise(self):
        torch.manual_seed(10)
        torch_model = BNAfterConvTranspose(groups=10)
        torch_model.eval()
        _initialize_bn_params(torch_model)

        input_shape = (2, 10, 24, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert pairs[0][0].name == "ConvTranspose_0"
        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    def test_fold_bn_before_linear_layer_no_bias(self):
        torch.manual_seed(10)
        torch_model = BNBeforeLinear(bias=False)
        torch_model.eval()
        _initialize_bn_params(torch_model)

        input_shape = (32, 10)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert pairs[0][0].name == "Gemm_5"
        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

        model = _convert_to_onnx(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert pairs[0][0].name == "Gemm_3"
        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    def test_fold_bn_before_linear_layer_with_bias(self):
        torch.manual_seed(10)
        torch_model = BNBeforeLinear(bias=True)
        torch_model.eval()
        _initialize_bn_params(torch_model)

        input_shape = (32, 10)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert pairs[0][0].name == "Gemm_3"
        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    def test_fold_bn_after_linear_layer_with_bias(self):
        torch.manual_seed(10)
        torch_model = BNAfterLinear(bias=True)
        torch_model.eval()
        _initialize_bn_params(torch_model)

        input_shape = (32, 10)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert pairs[0][0].name == "Gemm_0"
        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    def test_fold_bn_after_linear_layer_no_bias(self):
        torch.manual_seed(10)
        torch_model = BNAfterLinear(bias=False)
        torch_model.eval()
        _initialize_bn_params(torch_model)

        input_shape = (32, 10)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert pairs[0][0].name == "Gemm_1"
        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

        model = _convert_to_onnx(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert pairs[0][0].name == "Gemm_0"
        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    @pytest.mark.parametrize("bias", [True, False])
    def test_fold_bn_before_conv1d(self, bias):
        torch.manual_seed(10)
        torch_model = BNBeforeConv1d(bias=bias)
        torch_model.eval()
        _initialize_bn_params(torch_model)

        input_shape = (2, 10, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    @pytest.mark.parametrize("bias", [True, False])
    def test_fold_bn_after_conv1d(self, bias):
        torch.manual_seed(10)
        torch_model = BNAfterConv1d(bias=bias)
        torch_model.eval()
        _initialize_bn_params(torch_model)

        input_shape = (2, 10, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))
        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert len(model.graph().node) == layers_orig - 1
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)

    @pytest.mark.parametrize("bias", [True, False])
    def test_fold_bn_after_dynamic_matmul(self, bias):
        torch.manual_seed(10)
        torch_model = BNAfterDynamicMatMul(bias=bias, padding=1)
        torch_model.eval()
        _initialize_bn_params(torch_model)

        input_shape = (32, 10, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        model = _convert_to_onnx(torch_model, torch.randn(input_shape))

        layers_orig = len(model.graph().node)
        baseline_output, folded_output, pairs = get_outputs_after_fold(model, test_data)

        assert len(model.graph().node) == layers_orig
        assert np.allclose(baseline_output[0], folded_output[0], rtol=1e-2, atol=1e-6)
