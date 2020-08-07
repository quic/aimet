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
import torch
import torch.nn as nn
from torchvision import models

import numpy as np

from aimet_torch.batch_norm_fold import fold_given_batch_norms, fold_all_batch_norms, find_all_batch_norms_to_fold
from aimet_torch.examples.test_models import TransposedConvModel


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

        self.conv5 = torch.nn.Conv2d(20, 20, 3)

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


class TestTrainingExtensionBnFold(unittest.TestCase):

    def test_fold_two_conv_layers(self):
        torch.manual_seed(10)
        model = models.resnet18()

        model = model.eval()
        random_input = torch.rand(1, 3, 224, 224)

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.layer2[0].conv1, model.layer2[0].bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        self.assertFalse(isinstance(model.layer2[0].bn1, torch.nn.BatchNorm2d))
        self.assertTrue(np.allclose(baseline_output, output_after_fold, rtol=1.e-2))

    def test_fold_bn_before_conv_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 2, bias=None)
                self.reul1 = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.conv2 = torch.nn.Conv2d(20, 40, 2, bias=None)

            def forward(self, x):
                x = self.conv1(x)
                x = self.reul1(x)
                x = self.bn1(x)
                x = self.conv2(x)

                return x

        torch.manual_seed(10)
        model = MyModel()

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

        self.assertFalse(isinstance(model.bn1, torch.nn.BatchNorm2d))
        self.assertTrue(np.allclose(baseline_output, output_after_fold, rtol=1.e-2))

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

        self.assertFalse(isinstance(model.bn1, torch.nn.BatchNorm2d))
        self.assertTrue(np.allclose(baseline_output, output_after_fold, rtol=1.e-1))

    def test_fold_bn_after_conv_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(10, 20, 3, bias=None)
                self.bn1 = torch.nn.BatchNorm2d(20)
                self.reul1 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.reul1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()

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

        self.assertFalse(isinstance(model.bn1, torch.nn.BatchNorm2d))
        self.assertTrue(np.allclose(baseline_output, output_after_fold, rtol=1.e-2))

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

        self.assertFalse(isinstance(model.bn1, torch.nn.BatchNorm2d))
        self.assertTrue(np.allclose(baseline_output, output_after_fold, rtol=1.e-2))

    def test_fold_bn_before_linear_layer_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.bn1 = torch.nn.BatchNorm1d(10)
                self.fc1 = torch.nn.Linear(10, 20, bias=None)

            def forward(self, x):
                x = self.bn1(x)
                x = self.fc1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()

        random_input = torch.randn((32, 10))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((32, 10)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.bn1, model.fc1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        self.assertFalse(isinstance(model.bn1, torch.nn.BatchNorm1d))
        self.assertTrue(np.allclose(baseline_output, output_after_fold, rtol=1.e-2))

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

        random_input = torch.randn((32, 10))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((32, 10)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.bn1, model.fc1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        self.assertFalse(isinstance(model.bn1, torch.nn.BatchNorm1d))
        self.assertTrue(np.allclose(baseline_output, output_after_fold, rtol=1.e-2))

    def test_fold_bn_after_linear_layer_no_bias(self):

        class MyModel(torch.nn.Module):
            def __init__(self):

                super(MyModel, self).__init__()
                self.fc1 = torch.nn.Linear(10, 20, bias=None)
                self.bn1 = torch.nn.BatchNorm1d(20)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)

                return x

        torch.manual_seed(10)
        model = MyModel()

        random_input = torch.randn((32, 10))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((32, 10)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.fc1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        self.assertFalse(isinstance(model.bn1, torch.nn.BatchNorm1d))
        self.assertTrue(np.allclose(baseline_output, output_after_fold, rtol=1.e-2))

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

        random_input = torch.randn((32, 10))

        # Set the batch norm params to something non-zero with a random batch
        model.train()
        model(torch.randn((32, 10)))
        model.eval()

        baseline_output = model(random_input).detach().numpy()

        layer_list = [(model.fc1, model.bn1)]

        fold_given_batch_norms(model, layer_list)

        output_after_fold = model(random_input).detach().numpy()

        self.assertFalse(isinstance(model.bn1, torch.nn.BatchNorm1d))
        self.assertTrue(np.allclose(baseline_output, output_after_fold, rtol=1.e-2))

    def test_find_batch_norms_to_fold(self):

        model = MyModel()
        model.eval()

        bn_pairs = find_all_batch_norms_to_fold(model, (2, 10, 24, 24))
        self.assertEqual(2, len(bn_pairs))
        self.assertTrue((model.conv1, model.bn1) in bn_pairs)
        self.assertTrue((model.bn2, model.conv3) in bn_pairs)

    def test_bn_fold_auto_mode_transposed_conv2d(self):
        torch.manual_seed(0)
        model = TransposedConvModel()
        model = model.eval()

        random_input = torch.rand((10, 10, 4, 4))

        baseline_output = model(random_input).detach().numpy()

        folded_pairs = fold_all_batch_norms(model, (10, 10, 4, 4))

        output_after_fold = model(random_input).detach().numpy()

        self.assertFalse(isinstance(model.bn1, torch.nn.BatchNorm2d))
        self.assertTrue(np.allclose(baseline_output, output_after_fold, rtol=1.e-2))
        self.assertEqual(len(folded_pairs), 2)

    def test_find_batch_norms_to_fold_multi_input(self):

        model = TwoInputs()
        model.eval()
        inp_shapes = [(1, 3, 32, 32), (1, 3, 20, 20)]

        bn_pairs = find_all_batch_norms_to_fold(model, inp_shapes)
        self.assertEqual(2, len(bn_pairs))
        self.assertTrue((model.conv1, model.bn1) in bn_pairs)
        self.assertTrue((model.conv2, model.bn2) in bn_pairs)

    def test_bn_fold_auto_mode(self):
        torch.manual_seed(10)

        model = MyModel()

        model = model.eval()
        random_input = torch.rand(2, 10, 24, 24)

        baseline_output = model(random_input).detach().numpy()

        folded_pairs = fold_all_batch_norms(model, (2, 10, 24, 24))

        output_after_fold = model(random_input).detach().numpy()

        self.assertFalse(isinstance(model.bn1, torch.nn.BatchNorm2d))
        self.assertTrue(np.allclose(baseline_output, output_after_fold, rtol=1.e-2))
        self.assertEqual(len(folded_pairs), 2)
