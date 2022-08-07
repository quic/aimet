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

import pytest
import json
import os
import torch
from torchvision import models

import numpy as np

from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.batch_norm_fold import (
    fold_given_batch_norms,
    fold_all_batch_norms,
    fold_all_batch_norms_to_scale,
    _find_all_batch_norms_to_fold,
)
from aimet_torch.examples.test_models import TransposedConvModel
from aimet_torch.utils import create_rand_tensors_given_shapes
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme

from torch.nn.modules.batchnorm import _BatchNorm


torch.manual_seed(1228)


def _initialize_bn_params(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, _BatchNorm) and module.affine:
            with torch.no_grad():
                module.weight.copy_(torch.randn_like(module.weight))
                module.bias.copy_(torch.randn_like(module.bias))


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


class TwoInputs(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(TwoInputs, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=2)
        self.ada = torch.nn.AdaptiveAvgPool2d(18)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc = torch.nn.Linear(1600, num_classes)

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


class TestTrainingExtensionBnFold:

    def test_fold_two_conv_layers(self):
        torch.manual_seed(10)
        model = models.resnet18()
        _initialize_bn_params(model)

        model = model.eval()
        random_input = torch.rand(1, 3, 224, 224)

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.layer2[0].conv1, model.layer2[0].bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.layer2[0].bn1, torch.nn.BatchNorm2d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_before_conv_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 2, bias=False)
                self.reul1 = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.conv2 = torch.nn.Conv2d(20, 40, 2, bias=False)

            def forward(self, x):
                x = self.conv1(x)
                x = self.reul1(x)
                x = self.bn1(x)
                x = self.conv2(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()
        random_input = torch.rand(20, 10, 4, 4)

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((20, 10, 4, 4)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.bn1, model.conv2)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert model.conv2.weight.requires_grad == model.conv2.bias.requires_grad
        assert model.conv2.weight.device == model.conv2.bias.device
        assert model.conv2.weight.dtype == model.conv2.bias.dtype

    def test_fold_bn_before_conv_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.reul1 = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.conv2 = torch.nn.Conv2d(20, 30, 3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.reul1(x)
                x = self.bn1(x)
                x = self.conv2(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()
        random_input = torch.rand(2, 10, 24, 24)

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 24, 24)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.bn1, model.conv2)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-1)

    def test_fold_bn_after_conv_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.reul1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.reul1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()
        random_input = torch.rand(2, 10, 24, 24)

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 24, 24)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.conv1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert model.conv1.weight.requires_grad == model.conv1.bias.requires_grad
        assert model.conv1.weight.device == model.conv1.bias.device
        assert model.conv1.weight.dtype == model.conv1.bias.dtype

    def test_fold_bn_after_conv_depthwise(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 10, 3, groups=10)
                self.bn1 = torch.nn.BatchNorm2d(10)
                self.reul1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.reul1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()
        random_input = torch.rand(2, 10, 24, 24)

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 24, 24)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        fold_all_batch_norms(model, (2, 10, 24, 24))

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_after_transposed_conv_depthwise(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3, groups=10)
                self.bn1 = torch.nn.BatchNorm2d(10)
                self.reul1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.reul1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()
        random_input = torch.rand(2, 10, 24, 24)

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 24, 24)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        fold_all_batch_norms(model, (2, 10, 24, 24))

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_after_conv_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.reul1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.reul1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()
        random_input = torch.rand(2, 10, 24, 24)

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 24, 24)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.conv1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_before_linear_layer_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.fc1 = torch.nn.Linear(10, 20, bias=False)

            def forward(self, x):
                x = self.bn1(x)
                x = self.fc1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        random_input = torch.randn((32, 10))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((32, 10)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.bn1, model.fc1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert model.fc1.weight.requires_grad == model.fc1.bias.requires_grad
        assert model.fc1.weight.device == model.fc1.bias.device
        assert model.fc1.weight.dtype == model.fc1.bias.dtype

    def test_fold_bn_before_linear_layer_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.fc1 = torch.nn.Linear(10, 20)

            def forward(self, x):
                x = self.bn1(x)
                x = self.fc1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        random_input = torch.randn((32, 10))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((32, 10)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.bn1, model.fc1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_after_linear_layer_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.fc1 = torch.nn.Linear(10, 20, bias=False)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        random_input = torch.randn((32, 10))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((32, 10)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.fc1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert model.fc1.weight.requires_grad == model.fc1.bias.requires_grad
        assert model.fc1.weight.device == model.fc1.bias.device
        assert model.fc1.weight.dtype == model.fc1.bias.dtype

    def test_fold_bn_after_linear_layer_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        random_input = torch.randn((32, 10))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((32, 10)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.fc1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_find_batch_norms_to_fold(self):

        model = MyModel()
        _initialize_bn_params(model)
        model.eval()

        input_shape = (2, 10, 24, 24)
        connected_graph = ConnectedGraph(model,
                                         create_rand_tensors_given_shapes(input_shape))
        conv_bn_pairs, bn_conv_pairs = _find_all_batch_norms_to_fold(model, input_shape, connected_graph)
        assert len(conv_bn_pairs) == len(bn_conv_pairs) == 1
        assert (model.conv1, model.bn1) in conv_bn_pairs
        assert (model.bn2, model.conv3) in bn_conv_pairs

    def test_bn_fold_auto_mode_transposed_conv2d(self):
        torch.manual_seed(10)
        model = TransposedConvModel()
        _initialize_bn_params(model)
        model = model.eval()

        random_input = torch.rand((10, 10, 4, 4))

        baseline_output = model(random_input).detach().numpy()

        folded_pairs = fold_all_batch_norms(model, (10, 10, 4, 4))

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)

        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert len(folded_pairs) == 2

    def test_find_batch_norms_to_fold_multi_input(self):

        model = TwoInputs()
        _initialize_bn_params(model)
        model.eval()
        inp_shapes = [(1, 3, 32, 32), (1, 3, 20, 20)]

        connected_graph = ConnectedGraph(model,
                                         create_rand_tensors_given_shapes(inp_shapes))
        conv_bn_pairs, bn_conv_pairs = _find_all_batch_norms_to_fold(model, inp_shapes, connected_graph)
        assert len(conv_bn_pairs) == 2
        assert not bn_conv_pairs
        assert (model.conv1, model.bn1) in conv_bn_pairs
        assert (model.conv2, model.bn2) in conv_bn_pairs

    def test_bn_fold_auto_mode(self):
        torch.manual_seed(10)

        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()
        random_input = torch.rand(2, 10, 24, 24)

        baseline_output = model(random_input).detach().numpy()

        folded_pairs = fold_all_batch_norms(model, (2, 10, 24, 24))

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1, torch.nn.BatchNorm2d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert len(folded_pairs) == 2

    def test_fold_auto_mode_with_bn_after_Conv1d_layer(self):
        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.conv1d(x)
                x = self.bn1(x)

                return x

        model = MyModel()
        _initialize_bn_params(model)
        model.eval()

        random_input = torch.randn((2, 10, 32))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 32)))
        model.eval()

        baseline_output = model(random_input)
        orig_bn = model.bn1

        bn_pairs = fold_all_batch_norms(model, (2, 10, 32))
        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

        assert 1 == len(bn_pairs)
        assert (model.conv1d, orig_bn) in bn_pairs

    def test_fold_manual_with_bn_after_Conv1d_layer_no_bias(self):
        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2, bias=False)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.conv1d(x)
                x = self.bn1(x)

                return x

        model = MyModel()
        _initialize_bn_params(model)
        model.eval()

        random_input = torch.randn((2, 10, 32))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 32)))
        model.eval()

        baseline_output = model(random_input)

        layer_list = [(model.conv1d, model.bn1)]
        fold_given_batch_norms(model, layer_list)
        output_after_fold = (model.conv1d(random_input))

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert model.conv1d.weight.requires_grad == model.conv1d.bias.requires_grad
        assert model.conv1d.weight.device == model.conv1d.bias.device
        assert model.conv1d.weight.dtype == model.conv1d.bias.dtype

    @pytest.mark.cuda
    def test_multi_gpu(self):

        torch.manual_seed(10)
        model = MyModel()
        model.eval()
        model = torch.nn.DataParallel(model)
        model.to(device='cuda:0')
        random_input = torch.rand(2, 10, 24, 24).to(device='cuda:0')
        output_before = model(random_input)

        # BN fold
        fold_all_batch_norms(model, (2, 10, 24, 24))

        output_after = model(random_input)
        assert torch.allclose(output_before, output_after)

    def test_fold_bn_before_Conv1d_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2)

            def forward(self, x):
                x = self.bn1(x)
                x = self.conv1d(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        random_input = torch.randn((2, 10, 32))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 32)))
        model.eval()

        baseline_output = model(random_input)
        orig_bn = model.bn1
        bn_pairs = fold_all_batch_norms(model, (2, 10, 32))

        output_after_fold = model(random_input)

        assert 1 == len(bn_pairs)
        assert (model.conv1d, orig_bn) in bn_pairs
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_before_Conv1d_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(4)
                self.conv1d = torch.nn.Conv1d(4, 4, kernel_size=2, bias=False)

            def forward(self, x):
                x = self.bn1(x)
                x = self.conv1d(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        random_input = torch.randn((2, 4, 4))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 4, 4)))
        model.eval()

        baseline_output = model(random_input)

        layer_list = [(model.bn1, model.conv1d)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input)

        assert not isinstance(model.bn1, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert model.conv1d.weight.requires_grad == model.conv1d.bias.requires_grad
        assert model.conv1d.weight.device == model.conv1d.bias.device
        assert model.conv1d.weight.dtype == model.conv1d.bias.dtype


quantsim_config = {
    "defaults": {
        "ops": { "is_output_quantized": "True" },
        "params": { "is_quantized": "True" },
        "strict_symmetric": "False",
        "unsigned_symmetric": "True",
        "per_channel_quantization": "True"
    },
    "params": {
        "bias": { "is_quantized": "False" }
    },
    "op_type": {},
    "supergroups": [
        { "op_list": ["Conv", "Relu"] },
        { "op_list": ["Conv", "Clip"] },
        { "op_list": ["Add", "Relu"] },
        { "op_list": ["Gemm", "Relu"] },
        { "op_list": ["Conv", "BatchNormalization"] },
        { "op_list": ["Gemm", "BatchNormalization"] },
        { "op_list": ["ConvTranspose", "BatchNormalization"] }
    ],
    "model_input": { "is_input_quantized": "True" },
    "model_output": {}
}

config_file_path = "/tmp/quantsim_config.json"

def quantsim(model, input_shape):
    try:
        with open(config_file_path, 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(model,
                                   torch.randn(input_shape),
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   config_file=config_file_path)

        def forward_pass_callback(model, _):
            model(torch.randn(input_shape))

        sim.compute_encodings(forward_pass_callback, None)
        return sim

    finally:
        try:
            os.remove(config_file_path)
        except FileNotFoundError:
            pass


class TestTrainingExtensionBnFoldToScale:

    def test_fold_two_conv_layers(self):
        torch.manual_seed(10)
        model = models.resnet18()
        _initialize_bn_params(model)
        input_shape = (1, 3, 224, 224)

        model = model.eval()
        random_input = torch.rand(1, 3, 224, 224)

        sim = quantsim(model, input_shape)
        model = sim.model

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.layer2[0].conv1, model.layer2[0].bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.layer2[0].bn1._module_to_wrap, torch.nn.BatchNorm2d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_before_conv_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 2, bias=False)
                self.reul1 = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.conv2 = torch.nn.Conv2d(20, 40, 2, bias=False)

            def forward(self, x):
                x = self.conv1(x)
                x = self.reul1(x)
                x = self.bn1(x)
                x = self.conv2(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((20, 10, 4, 4)))
        model.eval()

        sim = quantsim(model, (20, 10, 4, 4))
        model = sim.model

        layer_list = [(model.bn1, model.conv2)]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    def test_fold_bn_before_conv_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.reul1 = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.conv2 = torch.nn.Conv2d(20, 30, 3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.reul1(x)
                x = self.bn1(x)
                x = self.conv2(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 24, 24)))
        model.eval()

        sim = quantsim(model, (2, 10, 24, 24))
        model = sim.model

        layer_list = [(model.bn1, model.conv2)]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    def test_fold_bn_after_conv_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.reul1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.reul1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()
        random_input = torch.rand(2, 10, 24, 24)

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 24, 24)))
        model.eval()

        sim = quantsim(model, (2, 10, 24,  24))
        model = sim.model

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.conv1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1._module_to_wrap, torch.nn.BatchNorm2d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)

        conv1 = model.conv1._module_to_wrap
        assert conv1.weight.requires_grad == conv1.bias.requires_grad
        assert conv1.weight.device == conv1.bias.device
        assert conv1.weight.dtype == conv1.bias.dtype

    def test_fold_bn_after_conv_depthwise(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 10, 3, groups=10)
                self.bn1 = torch.nn.BatchNorm2d(10)
                self.reul1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.reul1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()
        random_input = torch.rand(2, 10, 24, 24)

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 24, 24)))
        model.eval()

        sim = quantsim(model, (2, 10, 24,  24))
        model = sim.model

        baseline_output = model(random_input).detach().numpy()

        fold_all_batch_norms_to_scale(sim, (2, 10, 24, 24))

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1._module_to_wrap, torch.nn.BatchNorm2d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_after_transposed_conv_depthwise(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.ConvTranspose2d(10, 10, 3, groups=10)
                self.bn1 = torch.nn.BatchNorm2d(10)
                self.reul1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.reul1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 24, 24)))
        model.eval()

        sim = quantsim(model, (2, 10, 24,  24))
        model = sim.model

        fold_all_batch_norms_to_scale(sim, (2, 10, 24, 24))
        # Folding BatchNorm to transposed depthwise convolution is not supported
        assert isinstance(model.bn1._module_to_wrap, torch.nn.BatchNorm2d)

    def test_fold_bn_after_conv_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3)
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.reul1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.reul1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()
        random_input = torch.rand(2, 10, 24, 24)

        sim = quantsim(model, (2, 10, 24, 24))
        model = sim.model

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 24, 24)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.conv1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1._module_to_wrap, torch.nn.BatchNorm2d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_fold_bn_before_linear_layer_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.fc1 = torch.nn.Linear(10, 20, bias=False)

            def forward(self, x):
                x = self.bn1(x)
                x = self.fc1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((32, 10)))
        model.eval()

        sim = quantsim(model, (32, 10))
        model = sim.model

        layer_list = [(model.bn1, model.fc1)]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    def test_fold_bn_before_linear_layer_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.fc1 = torch.nn.Linear(10, 20)

            def forward(self, x):
                x = self.bn1(x)
                x = self.fc1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((32, 10)))
        model.eval()

        sim = quantsim(model, (32, 10))
        model = sim.model

        layer_list = [(model.bn1, model.fc1)]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)

    def test_fold_bn_after_linear_layer_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.fc1 = torch.nn.Linear(10, 20, bias=False)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        random_input = torch.randn((32, 10))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((32, 10)))
        model.eval()

        sim = quantsim(model, (32, 10))
        model = sim.model

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.fc1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1._module_to_wrap, torch.nn.BatchNorm1d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)

        fc1 = model.fc1._module_to_wrap
        assert fc1.weight.requires_grad == fc1.bias.requires_grad
        assert fc1.weight.device == fc1.bias.device
        assert fc1.weight.dtype == fc1.bias.dtype

    def test_fold_bn_after_linear_layer_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        random_input = torch.randn((32, 10))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((32, 10)))
        model.eval()

        sim = quantsim(model, (32, 10))
        model = sim.model

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.fc1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1._module_to_wrap, torch.nn.BatchNorm1d)
        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)

    def test_bn_fold_auto_mode_transposed_conv2d(self):
        torch.manual_seed(10)
        model = TransposedConvModel()
        _initialize_bn_params(model)
        model = model.eval()

        sim = quantsim(model, (10, 10 ,4, 4))
        model = sim.model

        random_input = torch.rand((10, 10, 4, 4))
        baseline_output = model(random_input).detach().numpy()
        folded_pairs = fold_all_batch_norms_to_scale(sim, (10, 10, 4, 4))
        output_after_fold = model(random_input).detach().numpy()

        assert not isinstance(model.bn1._module_to_wrap, torch.nn.BatchNorm2d)

        assert np.allclose(baseline_output, output_after_fold, rtol=1.e-2)
        assert len(folded_pairs) == 2

    def test_bn_fold_auto_mode(self):
        torch.manual_seed(10)

        model = MyModel()
        _initialize_bn_params(model)

        model = model.eval()
        sim = quantsim(model, (2, 10, 24, 24))

        with pytest.raises(RuntimeError):
            fold_all_batch_norms_to_scale(sim, (2, 10, 24, 24))

    def test_fold_auto_mode_with_bn_after_Conv1d_layer(self):
        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.conv1d(x)
                x = self.bn1(x)

                return x

        model = MyModel()
        _initialize_bn_params(model)
        model.eval()

        random_input = torch.randn((2, 10, 32))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 32)))
        model.eval()

        sim = quantsim(model, (2, 10, 32))
        model = sim.model

        baseline_output = model(random_input)
        orig_bn = model.bn1

        bn_pairs = fold_all_batch_norms_to_scale(sim, (2, 10, 32))
        output_after_fold = model(random_input)

        assert not isinstance(model.bn1._module_to_wrap, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

        assert 1 == len(bn_pairs)
        assert (model.conv1d, orig_bn) in bn_pairs

    def test_fold_manual_with_bn_after_Conv1d_layer_no_bias(self):
        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2, bias=False)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.conv1d(x)
                x = self.bn1(x)

                return x

        model = MyModel()
        _initialize_bn_params(model)
        model.eval()

        random_input = torch.randn((2, 10, 32))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 32)))
        model.eval()

        sim = quantsim(model, (2, 10, 32))
        model = sim.model

        baseline_output = model(random_input)

        layer_list = [(model.conv1d, model.bn1)]
        fold_given_batch_norms(model, layer_list)
        output_after_fold = model(random_input)

        assert not isinstance(model.bn1._module_to_wrap, torch.nn.BatchNorm1d)
        assert torch.allclose(baseline_output, output_after_fold, rtol=1.e-2)

        conv1d = model.conv1d._module_to_wrap
        assert conv1d.weight.requires_grad == conv1d.bias.requires_grad
        assert conv1d.weight.device == conv1d.bias.device
        assert conv1d.weight.dtype == conv1d.bias.dtype

    def test_fold_bn_before_Conv1d_with_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.conv1d = torch.nn.Conv1d(10, 20, kernel_size=2)

            def forward(self, x):
                x = self.bn1(x)
                x = self.conv1d(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 10, 32)))
        model.eval()

        sim = quantsim(model, (2, 10, 32))

        with pytest.raises(RuntimeError):
            fold_all_batch_norms_to_scale(sim, (2, 10, 32))

    def test_fold_bn_before_Conv1d_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(4)
                self.conv1d = torch.nn.Conv1d(4, 4, kernel_size=2, bias=False)

            def forward(self, x):
                x = self.bn1(x)
                x = self.conv1d(x)

                return x

        torch.manual_seed(10)
        model = MyModel()
        _initialize_bn_params(model)

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((2, 4, 4)))
        model.eval()

        sim = quantsim(model, (2, 4, 4))
        model = sim.model

        layer_list = [(model.bn1, model.conv1d)]

        with pytest.raises(RuntimeError):
            fold_given_batch_norms(model, layer_list)
