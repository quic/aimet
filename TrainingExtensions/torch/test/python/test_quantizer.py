# /usr/bin/env python2.7
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
import numpy as np
import torch
import torch.nn as nn
import json as json


from torchvision import models
from aimet_common.defs import QuantScheme

from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.quantsim_straight_through_grad import compute_dloss_by_dx
from aimet_torch.defs import PassThroughOp

from aimet_torch.qc_quantize_op import QcQuantizeWrapper, QcQuantizeStandalone, MAP_ROUND_MODE_TO_PYMO, \
    MAP_QUANT_SCHEME_TO_PYMO, QcPostTrainingWrapper
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


def dummy_forward_pass(model, args):
    model.eval()
    with torch.no_grad():
        output = model(torch.randn((32, 1, 28, 28)))
    return output


class SmallMnistNoDropoutWithPassThrough(nn.Module):
    def __init__(self):
        super(SmallMnistNoDropoutWithPassThrough, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pt1 = PassThroughOp()
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pt2 = PassThroughOp()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.pt1(self.conv1(x)))
        x = self.conv2(x)
        x = self.relu2(self.pt2(x))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.log_softmax(x)


class SmallMnist(nn.Module):
    def __init__(self):
        super(SmallMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.relu2(self.conv2_drop(x))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.log_softmax(x)


class SmallMnistNoDropout(nn.Module):
    def __init__(self):
        super(SmallMnistNoDropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return self.log_softmax(x)


class ModelWithStandaloneOps(nn.Module):
    def __init__(self):
        super(ModelWithStandaloneOps, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.myquant = QcQuantizeStandalone(activation_bw=8, round_mode=MAP_ROUND_MODE_TO_PYMO['nearest'],
                                            quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf_enhanced],
                                            is_symmetric=False)
        self.conv2_drop = nn.Dropout2d()
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout2d()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = self.relu1(self.maxpool1(self.conv1(inputs[0])))
        x = self.conv2(x)
        x = self.myquant(x)
        x = self.relu2(self.maxpool2(self.conv2_drop(x)))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.log_softmax(x)


class ModelWithTwoInputs(nn.Module):

    def __init__(self):
        super(ModelWithTwoInputs, self).__init__()
        self.conv1_a = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_a = nn.MaxPool2d(2)
        self.relu1_a = nn.ReLU()

        self.conv1_b = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1_b = nn.MaxPool2d(2)
        self.relu1_b = nn.ReLU()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.relu1_a(self.maxpool1_a(self.conv1_a(x1)))
        x2 = self.relu1_b(self.maxpool1_b(self.conv1_b(x2)))
        x = x1 + x2
        x = self.relu2(self.maxpool2(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


class TestQuantizationSim(unittest.TestCase):
    def test_is_leaf_module_positive(self):
        """With an actual leaf module"""
        conv1 = nn.Conv2d(1, 10, 5)
        self.assertTrue(QuantizationSimModel._is_leaf_module(conv1))

    # -------------------------------------------
    def test_is_leaf_module_negative(self):
        """With a non-leaf module"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        net = Net()
        model = net.to(torch.device('cpu'))

        self.assertFalse(QuantizationSimModel._is_leaf_module(model))

    # -------------------------------------------------------------
    def test_is_quantizable_module_positive(self):
        """With a quantizable module"""
        conv1 = nn.Conv2d(1, 10, 5)
        self.assertTrue(QuantizationSimModel._is_quantizable_module(conv1))

    # -------------------------------------------------------------
    def test_is_quantizable_module_negative(self):
        """With a non-quantizable module"""
        conv1 = QcPostTrainingWrapper(nn.Conv2d(1, 10, 5), weight_bw=8, activation_bw=8, round_mode='nearest',
                                      quant_scheme=QuantScheme.post_training_tf_enhanced)
        self.assertFalse(QuantizationSimModel._is_quantizable_module(conv1))

    # ------------------------------------------------------------
    def verify_quantization_wrappers(self, original_model, quantized_model, quant_scheme=QuantScheme.post_training_tf_enhanced):
        """Test utility to determine if quantization wrappers were added correctly"""

        # All leaf modules in the original model
        orig_modules = [(name, module) for name, module in original_model.named_modules()
                        if len(list(module.modules())) == 1]

        # All QcQuantized modules in the quantized model
        quant_modules = [(name, module) for name, module in quantized_model.named_modules()
                         if isinstance(module, QcQuantizeWrapper)]

        for i, orig_mod_tuple in enumerate(orig_modules):
            quant_mod_tuple = quant_modules[i]

            # If the original model has any QcQuantizeWrapper node, skip the checks
            if '_module_to_wrap' in orig_mod_tuple[0]:
                continue

            # Checks --------
            # Modules should be in the same order as before
            self.assertEqual(orig_mod_tuple[0], quant_mod_tuple[0], "Quantized model has a incorrectly named module")

            if quant_scheme in [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced]:
                # For every leaf module in the first list, there is a corresponding QcQuantized model in the second list
                self.assertEqual(str(type(quant_mod_tuple[1]).__name__), 'QcPostTrainingWrapper')

            # Each QcQuantized model has 1 child, that is the same type as the corresponding module in the original list
            self.assertEqual(len(list(quant_mod_tuple[1].modules())), 2)
            child = list(quant_mod_tuple[1].modules())[1]
            logger.debug("{} -> {}".format(type(child), type(orig_mod_tuple[1])))
            self.assertEqual(type(child), type(orig_mod_tuple[1]))

    # --------------------------------------------------------
    def test_add_quantization_wrappers_one_deep(self):
        """With a one-deep model"""
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                x = self.conv2_drop(x)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        model = Net()
        sim = QuantizationSimModel(model, input_shapes=(1, 1, 12, 12))

        self.verify_quantization_wrappers(model, sim.model)

    # ------------------------------------------------------
    def test_add_quantization_wrappers_with_preexisting_quantization_layers(self):
        """With a one-deep model"""
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = QcPostTrainingWrapper(nn.Conv2d(1, 10, 5), weight_bw=8, activation_bw=8,
                                                   round_mode='stochastic', quant_scheme=QuantScheme.post_training_tf_enhanced)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                x = self.conv2_drop(x)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        net = Net()
        model = net.to(torch.device('cpu'))

        sim = QuantizationSimModel(model, input_shapes=(1, 1, 12, 12))

        # Add wrappers again, expect to be a nop
        sim._add_quantization_wrappers(model)

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_two_deep(self):
        """With a one-deep model"""
        class SubNet(nn.Module):
            def __init__(self):
                super(SubNet, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = SubNet()
                self.fc1 = nn.Linear(320, 50)
                self.SubNet2 = SubNet()
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                return self.conv1(inputs[0])

        net = Net()
        model = net.to(torch.device('cpu'))

        sim = QuantizationSimModel(model, input_shapes=(1, 1, 12, 12))

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_sequentials(self):
        """With a one-deep model"""
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc1 = nn.Linear(320, 50)
                self.subnet2 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                return self.conv1(inputs[0])

        net = Net()
        model = net.to(torch.device('cpu'))
        print(model)
        sim = QuantizationSimModel(model, input_shapes=(1, 1, 12, 12))
        print(sim.model)
        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_sequential_two_deep(self):
        """With a one-deep model"""
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Sequential(
                        nn.Conv2d(1, 10, 5),
                        nn.ReLU(),
                        nn.Conv2d(1, 10, 5)
                    ),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                return self.conv1(inputs[0])

        net = Net()
        model = net.to(torch.device('cpu'))
        sim = QuantizationSimModel(model, input_shapes=(1, 1, 12, 12))

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_modulelist(self):
        """With a one-deep model using ModuleList"""
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layers = nn.ModuleList([nn.Linear(1, 32), nn.Linear(32, 64), nn.Conv2d(1, 32, 5),
                                             QcPostTrainingWrapper(nn.Conv2d(1, 10, 5), weight_bw=8, activation_bw=8,
                                                                   round_mode='nearest',
                                                                   quant_scheme=QuantScheme.post_training_tf_enhanced)])

            def forward(self, *inputs):
                return self.layers[2](inputs[0])

        model = Net()
        sim = QuantizationSimModel(model, input_shapes=(1, 1, 12, 12))

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_modulelist_two_deep(self):
        """With a two-deep model using ModuleList"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layers = nn.ModuleList([nn.Linear(1, 32), nn.Linear(32, 64), nn.Conv2d(3, 32, kernel_size=3)])
                self.layers_deep = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(10), nn.ReLU()]),
                                                  nn.Linear(3, 32), nn.Linear(32, 64), nn.Conv2d(1, 32, 5),
                                                  QcPostTrainingWrapper(nn.Conv2d(1, 10, 5), weight_bw=8, activation_bw=8,
                                                                    round_mode='nearest', quant_scheme=QuantScheme.post_training_tf_enhanced)])

            def forward(self, *inputs):
                return self.layers[2](inputs[0])

        model = Net()
        print(model)
        sim = QuantizationSimModel(model, input_shapes=(1, 3, 12, 12))
        print(sim.model)

        self.verify_quantization_wrappers(model, sim.model)

    # -------------------------------------------
    def test_add_quantization_wrappers_with_modulelist_with_layers_to_ignore(self):
        """With a two-deep model using ModuleList and layers_to_ignore"""

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layers = nn.ModuleList([nn.Linear(1, 32), nn.Linear(32, 64), nn.Conv2d(3, 32, kernel_size=3)])
                self.layers_deep = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(10), nn.ReLU()]),
                                                  nn.Linear(3, 32), nn.Linear(32, 64), nn.Conv2d(1, 32, 5),
                                                  QcPostTrainingWrapper(nn.Conv2d(1, 10, 5), weight_bw=8, activation_bw=8,
                                                                    round_mode='nearest',
                                                                    quant_scheme=QuantScheme.post_training_tf_enhanced)])

            def forward(self, *inputs):
                return self.layers[2](inputs[0])

        model = Net()

        sim = QuantizationSimModel(model, input_shapes=(1, 3, 12, 12))
        layers_to_exclude = [sim.model.layers_deep[1], sim.model.layers_deep[3]]
        sim.exclude_layers_from_quantization(layers_to_exclude)
        print(sim.model)

        self.assertTrue(isinstance(sim.model.layers[0]._module_to_wrap, nn.Linear))
        self.assertTrue(isinstance(sim.model.layers[1]._module_to_wrap, nn.Linear))
        self.assertTrue(isinstance(sim.model.layers[2]._module_to_wrap, nn.Conv2d))

        self.assertTrue(isinstance(sim.model.layers_deep[0][0]._module_to_wrap, nn.BatchNorm2d))
        self.assertTrue(isinstance(sim.model.layers_deep[0][1]._module_to_wrap, nn.ReLU))

        # layer ignored, so no QcQuantizeWrapper wrapper
        self.assertTrue(isinstance(sim.model.layers_deep[1], nn.Linear))
        self.assertTrue(isinstance(sim.model.layers_deep[2]._module_to_wrap, nn.Linear))

        # layer ignored, so no QcQuantizeWrapper wrapper
        self.assertTrue(isinstance(sim.model.layers_deep[3], nn.Conv2d))

    # -------------------------------------------
    def test_model_with_two_inputs(self):
        """Model with more than 1 input"""

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(torch.randn((32, 1, 28, 28)), torch.randn(32, 1, 28, 28))

        model = ModelWithTwoInputs()

        sim = QuantizationSimModel(model, input_shapes=[(32, 1, 28, 28), (32, 1, 28, 28)])

        # Quantize
        sim.compute_encodings(forward_pass, None)

        # save encodings
        sim.export('./data/', 'two_input_model', input_shape=[(1, 1, 28, 28), (1, 1, 28, 28)])

    # -------------------------------------------

    def test_export_unified_encoding_format(self):
        """ test export functionality on ResNet18 """

        resnet18 = models.resnet18()
        resnet18.eval()
        input_shapes = (1, 3, 224, 224)

        # Get Dict mapping node name to the input and output names
        sim = QuantizationSimModel(resnet18, input_shapes=(1, 3, 224, 224))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(torch.randn(1, 3, 224, 224))

        # Quantize
        sim.compute_encodings(forward_pass, None)

        sim.export('./data/', 'resnet18', input_shape=(1, 3, 224, 224))
        with open('./data/resnet18.encodings') as json_file:
            encoding_data = json.load(json_file)
            print(encoding_data)

        activation_keys = list(encoding_data["activation_encodings"].keys())
        self.assertTrue(activation_keys[0] == "123")
        self.assertTrue(isinstance(encoding_data["activation_encodings"]["123"], list))

        param_keys = list(encoding_data["param_encodings"].keys())
        self.assertTrue(param_keys[1] == "conv1.weight")
        self.assertTrue(isinstance(encoding_data["param_encodings"]["conv1.weight"], list))

    # -------------------------------------------

    def test_export(self):
        """Exporting encodings and model"""

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(torch.randn((32, 1, 28, 28)), torch.randn(32, 1, 28, 28))

        model = ModelWithTwoInputs()
        sim = QuantizationSimModel(model, input_shapes=[(32, 1, 28, 28), (32, 1, 28, 28)])

        # Quantize
        sim.compute_encodings(forward_pass, None)

        sim.model.conv1_a.param_quantizers['weight'].encoding.max = 10
        sim.model.conv1_a.output_quantizer.encoding.max = 30

        # save encodings
        sim.export('./data/', 'two_input_model', input_shape=[(1, 1, 28, 28), (1, 1, 28, 28)])

        # check the encodings
        with open('./data/two_input_model.encodings', 'r') as fp:
            encodings = json.load(fp)

            activation_encodings = encodings['activation_encodings']
            param_encodings = encodings['param_encodings']
            self.assertEqual(13, len(activation_encodings))
            self.assertNotIn('conv1_a.bias', param_encodings)
            self.assertEqual(5, len(param_encodings['conv1_a.weight'][0]))
            self.assertEqual(10, param_encodings['conv1_a.weight'][0]['max'])

        # check the exported model
        loaded_model = torch.load('./data/two_input_model.pth')
        loaded_model(torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28))

    # -------------------------------------------
    def test_no_fine_tuning_tf_enhanced(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced, input_shapes=(1, 1, 12, 12))
        self.assertTrue(isinstance(sim.model.conv1, QcQuantizeWrapper))

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        dummy_forward_pass(sim.model, None)

    # -------------------------------------------
    def test_input_quantization(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 1, 12, 12))
        for module in sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                module.output_quantizer.enabled = False
                module.input_quantizer.enabled = True

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        output = dummy_forward_pass(sim.model, None)

        self.assertFalse(sim.model.conv1.output_quantizer.encoding)
        self.assertTrue(sim.model.conv1.input_quantizer.encoding)

        print(sim.model.conv1.input_quantizer)
        print(sim.model.conv1.output_quantizer)

    # -------------------------------------------
    def test_input_and_output_quantization(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 1, 12, 12))
        for module in sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                module.output_quantizer.enabled = True
                module.input_quantizer.enabled = True

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        output = dummy_forward_pass(sim.model, None)

        self.assertTrue(sim.model.conv1.output_quantizer.encoding)
        self.assertTrue(sim.model.conv1.input_quantizer.encoding)

        print(sim.model.conv1.input_quantizer)
        print(sim.model.conv1.output_quantizer)

    def test_super_node_creation(self):
        """"""
        model = SmallMnistNoDropout()
        print(model)

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 1, 28, 28))
        sim.create_super_nodes_of_layers_and_activation_functions((1, 1, 28, 28))

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        output = dummy_forward_pass(sim.model, None)

        print(sim)

        # Checks
        for module in sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                if isinstance(module._module_to_wrap, torch.nn.Conv2d):
                    self.assertFalse(module.output_quantizer.enabled)
                    self.assertFalse(module.input_quantizer.enabled)
                    self.assertTrue(module.param_quantizers['weight'].enabled)

                if isinstance(module._module_to_wrap, torch.nn.ReLU):
                    self.assertTrue(module.output_quantizer.enabled)
                    self.assertFalse(module.input_quantizer.enabled)

    def test_super_node_creation_with_passthrough(self):
        """"""
        model = SmallMnistNoDropoutWithPassThrough().eval()
        print(model)

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 1, 28, 28))
        sim.create_super_nodes_of_layers_and_activation_functions((1, 1, 28, 28))

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        output = dummy_forward_pass(sim.model, None)

        print(sim)

        # Checks
        for module in sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                if isinstance(module._module_to_wrap, torch.nn.Conv2d):
                    self.assertFalse(module.output_quantizer.enabled)
                    self.assertFalse(module.input_quantizer.enabled)
                    self.assertTrue(module.param_quantizers['weight'].enabled)

                if isinstance(module._module_to_wrap, torch.nn.ReLU):
                    self.assertTrue(module.output_quantizer.enabled)
                    self.assertFalse(module.input_quantizer.enabled)

    def test_super_node_creation_with_dropout(self):
        """"""
        model = SmallMnist().eval()
        print(model)

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 1, 28, 28))
        sim.create_super_nodes_of_layers_and_activation_functions((1, 1, 28, 28))

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        output = dummy_forward_pass(sim.model, None)

        print(sim)

        # Checks
        for module in sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                # sim.model.conv2 is followed by a dropout
                if isinstance(module._module_to_wrap, torch.nn.Conv2d) and module != sim.model.conv2:
                    self.assertFalse(module.output_quantizer.enabled)
                    self.assertFalse(module.input_quantizer.enabled)
                    self.assertTrue(module.param_quantizers['weight'].enabled)

                if isinstance(module._module_to_wrap, torch.nn.ReLU):
                    self.assertTrue(module.output_quantizer.enabled)
                    self.assertFalse(module.input_quantizer.enabled)

    def test_quantizing_models_with_add_ops(self):
        """
        Testing models with add functional ops
        :return:
        """
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya + yb
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)
                x = ya + yb
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 3, 28, 28))

        self.assertTrue(sim.model.conv3.input_quantizer.enabled)
        self.assertTrue(sim.model.conv5.input_quantizer.enabled)

        print(sim)

    def test_quantizing_models_with_add_followed_by_split(self):
        """
        Testing models with add functional ops followed by a split
        :return:
        """
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya + yb

                ya = self.conv4a(x)
                yb = self.conv4b(x)
                x = ya + yb

                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 3, 28, 28))

        self.assertTrue(sim.model.conv4a.input_quantizer.enabled)
        self.assertTrue(sim.model.conv4b.input_quantizer.enabled)
        self.assertTrue(sim.model.conv5.input_quantizer.enabled)

        print(sim)

    def test_quantizing_models_with_add_followed_by_add(self):
        """
        Testing models with add functional ops followed by a split and then another add.
        This is similar to the resnet architecture where there are no ops on the residual connection
        :return:
        """
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5, padding=2)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya + yb

                ya = self.conv4a(x)
                yb = x
                x = ya + yb

                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 3, 28, 28))

        self.assertTrue(sim.model.conv4a.input_quantizer.enabled)
        self.assertTrue(sim.model.conv5.input_quantizer.enabled)

        print(sim)

    def test_quantizing_model_with_input_leading_to_add(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=5)
                self.softmax = nn.LogSoftmax(dim=1)

            def forward(self, x1, x2):
                x = self.conv1(x1)
                x = x + x2
                return self.softmax(x)
        model = Net().eval()
        _ = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=[(1, 3, 28, 28), (1, 3, 24, 24)])

    def test_quantizing_models_with_mul_ops(self):
        """
        Testing models with elementwise multiply functional ops
        :return:
        """
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya * yb
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)
                x = ya * yb
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 3, 28, 28))

        self.assertTrue(sim.model.conv3.input_quantizer.enabled)
        self.assertTrue(sim.model.conv5.input_quantizer.enabled)

        print(sim)

    def test_quantizing_models_with_div_ops(self):
        """
        Testing models with elementwise division functional ops
        :return:
        """
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 20, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 20, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = ya / yb
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)
                x = ya / yb
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 3, 28, 28))

        self.assertTrue(sim.model.conv3.input_quantizer.enabled)
        self.assertTrue(sim.model.conv5.input_quantizer.enabled)

        print(sim)

    def test_quantizing_models_with_concat_ops(self):
        """
        Testing models with concat functional ops
        :return:
        """
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
                self.conv2a = nn.Conv2d(10, 10, kernel_size=5)
                self.conv2b = nn.Conv2d(10, 10, kernel_size=5)
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv4a = nn.Conv2d(20, 10, kernel_size=5)
                self.conv4b = nn.Conv2d(20, 10, kernel_size=5)
                self.conv5 = nn.Conv2d(20, 20, kernel_size=5)

            def forward(self, input):
                x = self.conv1(input)

                ya = self.conv2a(x)
                yb = self.conv2b(x)

                x = torch.cat((ya, yb), 1)
                x = self.conv3(x)

                ya = self.conv4a(x)
                yb = self.conv4b(x)

                x = torch.cat((ya, yb), 1)
                x = self.conv5(x)

                return x

        model = Net()
        model(torch.rand(1, 3, 28, 28))
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 3, 28, 28))

        self.assertTrue(sim.model.conv3.input_quantizer.enabled)
        self.assertTrue(sim.model.conv5.input_quantizer.enabled)

        print(sim)

    def test_no_finetuning_tf(self):
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 1, 28, 28))
        self.assertTrue(isinstance(sim.model.conv1, QcQuantizeWrapper))

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)
        dummy_forward_pass(sim.model, None)

    def test_per_layer_bitwidths(self):
        """"""
        model = SmallMnist()

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, input_shapes=(1, 1, 28, 28))
        self.assertTrue(isinstance(sim.model.conv1, QcQuantizeWrapper))
        sim.model.conv1.set_output_bw(16)

        # Find encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # try one forward pass
        dummy_forward_pass(model, None)

        self.assertEqual(16, sim.model.conv1.output_quantizer.bitwidth)
        self.assertEqual(8, sim.model.conv2.output_quantizer.bitwidth)

    # -------------------------------------------
    def test_with_standalone_ops(self):

        model = ModelWithStandaloneOps()
        sim = QuantizationSimModel(model=model, input_shapes=(1, 1, 28, 28))

        # Quantize
        sim.compute_encodings(dummy_forward_pass, None)
        dummy_forward_pass(sim.model, None)

        # Save encodings
        sim.export("./data/", "encodings_with_standalone_ops", input_shape=(1, 1, 28, 28))
        with open('./data/encodings_with_standalone_ops.encodings') as json_file:
            encoding_data = json.load(json_file)
        # in onnx definition tensor 16 is output of Reshape, to be ignored
        self.assertTrue("16" not in encoding_data["activation_encodings"].keys())

    # -------------------------------------------------------------------------------
    def test_layers_to_ignore(self):
        """ Test the  capability to skip quantizing the layers specified by the user"""

        model = SmallMnist()

        sim = QuantizationSimModel(model, input_shapes=(1, 1, 28, 28))
        layers_to_ignore = [sim.model.conv1, sim.model.fc2]
        sim.exclude_layers_from_quantization(layers_to_ignore)

        # Compute encodings
        sim.compute_encodings(dummy_forward_pass, None)

        # Check
        self.assertTrue(isinstance(sim.model.conv1, nn.Conv2d))
        self.assertFalse(isinstance(sim.model.conv2, nn.Conv2d))
        self.assertTrue(isinstance(sim.model.fc2, nn.Linear))

    def check_quant_params(self, model_layer, loaded_model_layer, check_weights):
        output_encoding1 = model_layer.output_quantizer.encoding
        output_encoding2 = loaded_model_layer.output_quantizer.encoding

        self.assertEqual(model_layer.output_quantizer.bitwidth, loaded_model_layer.output_quantizer.bitwidth)
        self.assertEqual(output_encoding1.max, output_encoding2.max)
        self.assertEqual(output_encoding1.min, output_encoding2.min)
        self.assertEqual(output_encoding1.delta, output_encoding2.delta)
        self.assertEqual(output_encoding1.offset, output_encoding2.offset)

        if model_layer.param_quantizers:
            self.assertEqual(next(iter(model_layer.param_quantizers.values())).bitwidth,
                             next(iter(loaded_model_layer.param_quantizers.values())).bitwidth)

        self.assertEqual(model_layer.output_quantizer.round_mode, loaded_model_layer.output_quantizer.round_mode)
        self.assertEqual(model_layer.output_quantizer.quant_scheme, loaded_model_layer.output_quantizer.quant_scheme)

        if check_weights:
            self.assertTrue(np.allclose(model_layer._module_to_wrap.weight.detach().numpy(),
                                        loaded_model_layer._module_to_wrap.weight.detach().numpy()))

    def test_save_and_load(self):

        model = ModelWithStandaloneOps()

        sim = QuantizationSimModel(model, input_shapes=(32, 1, 28, 28))

        # Quantize
        sim.compute_encodings(dummy_forward_pass, None)

        # Run some inferences - mimic using a forward pass
        sim.model.eval()
        dummy_input = torch.randn((32, 1, 28, 28))
        output_before_save = sim.model(dummy_input)

        # Save quantized model
        torch.save(sim.model, './data/xx')

        loaded_model = torch.load('./data/xx')
        loaded_model.eval()
        output_after_load = loaded_model(dummy_input)

        self.check_quant_params(sim.model.conv1, loaded_model.conv1, True)
        self.check_quant_params(sim.model.conv2, loaded_model.conv2, True)
        self.check_quant_params(sim.model.conv2_drop, loaded_model.conv2_drop, False)
        self.check_quant_params(sim.model.fc2, loaded_model.fc2, True)

        self.assertTrue(np.allclose(output_before_save.detach().numpy(),
                                    output_after_load.detach().numpy()))

    def test_ste_gradient_math(self):
        """
        Unit test to validate custom gradient computation with auto grad computation.
        :return: None
        """

        c_enc_min = torch.Tensor([-0.25])
        c_enc_max = torch.Tensor([1.0])
        grad = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])

        # input > max
        custom_input_1 = torch.Tensor([[1.0, 1.5], [0.125, -0.12]])
        expected_grad_1 = torch.Tensor([[1.0, 0.0], [1.0, 1.0]])
        grad_out_1 = compute_dloss_by_dx(custom_input_1, grad, c_enc_min, c_enc_max)

        # input < min
        custom_input_2 = torch.Tensor([[1.0, 0.5], [0.125, -0.30]])
        expected_grad_2 = torch.Tensor([[1.0, 1.0], [1.0, 0.0]])
        grad_out_2 = compute_dloss_by_dx(custom_input_2, grad, c_enc_min, c_enc_max)

        # valid input range
        custom_input_3 = torch.Tensor([[1.0, 0.5], [0.125, -0.25]])
        expected_grad_3 = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])
        grad_out_3 = compute_dloss_by_dx(custom_input_3, grad, c_enc_min, c_enc_max)

        self.assertTrue(np.allclose(expected_grad_1, grad_out_1))
        self.assertTrue(np.allclose(expected_grad_2, grad_out_2))
        self.assertTrue(np.allclose(expected_grad_3, grad_out_3))

