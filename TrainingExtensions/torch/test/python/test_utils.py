# /usr/bin/env python3.8
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

import pytest
import unittest.mock
import numpy as np
import shutil
import math
import torch
import torchvision
import torch.nn.functional as F

import aimet_torch.model_validator.validation_checks
import aimet_torch.utils
from aimet_common.utils import round_up_to_multiplicity, round_down_to_multiplicity
from aimet_torch import utils, elementwise_ops

from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.examples.test_models import TinyModel, MultiInput, ModelWithReusedNodes, SingleResidual


class TestTrainingExtensionsUtils(unittest.TestCase):

    def test_round_up_to_higher_multiplicity(self):
        self.assertEqual(round_up_to_multiplicity(8, 3, 32), 8)
        self.assertEqual(round_up_to_multiplicity(8, 13, 32), 16)
        self.assertEqual(round_up_to_multiplicity(8, 17, 32), 24)
        self.assertEqual(round_up_to_multiplicity(8, 29, 32), 32)

    def test_round_down_to_lower_multiplicity(self):
        self.assertEqual(round_down_to_multiplicity(8, 3), 3)
        self.assertEqual(round_down_to_multiplicity(8, 13), 8)
        self.assertEqual(round_down_to_multiplicity(8, 17), 16)
        self.assertEqual(round_down_to_multiplicity(8, 29), 24)
        self.assertEqual(round_down_to_multiplicity(8, 16), 8)
        self.assertEqual(round_down_to_multiplicity(32, 64), 32)

    def test_replace_relu_with_relu6(self):
        model = torchvision.models.resnet18()
        model.eval()

        utils.replace_modules_of_type1_with_type2(model, torch.nn.ReLU, torch.nn.ReLU6)

        # check - no ReLU modules left in the model anymore
        for module in model.modules():
            self.assertTrue(not isinstance(module, torch.nn.ReLU))

        # sanity-check: forward pass continues to work
        with torch.no_grad():
            x = torch.rand(1, 3, 224, 224)
            output = model(x)

    def test_replace_some_bns_with_passthrough(self):
        model = torchvision.models.resnet18()
        model.eval()

        utils.replace_modules_with_instances_of_new_type(model, [model.layer1[0].bn1, model.layer1[1].bn1],
                                                         torch.nn.Identity)

        # check - given modules have been replaced
        self.assertTrue(isinstance(model.layer1[0].bn1, torch.nn.Identity))
        self.assertTrue(isinstance(model.layer1[1].bn1, torch.nn.Identity))

        # check - other bn layers have not been modified
        self.assertFalse(isinstance(model.layer1[0].bn2, torch.nn.Identity))
        self.assertFalse(isinstance(model.layer1[1].bn2, torch.nn.Identity))

        # sanity-check: forward pass continues to work
        with torch.no_grad():
            x = torch.rand(1, 3, 224, 224)
            output = model(x)

    def test_get_ordered_ops(self):
        model = torchvision.models.resnet18(pretrained=False)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        all_ops = utils.get_ordered_list_of_modules(model, dummy_input)

        self.assertEqual(60, len(all_ops))

    def test_get_reused_modules(self):
        """ Test get_reused_modules utility """
        model = ModelWithReusedNodes()
        model_input = torch.randn((1, 3, 32, 32))
        reused_modules = aimet_torch.utils.get_reused_modules(model, model_input)
        self.assertEqual(1, len(reused_modules))
        self.assertEqual(reused_modules[0][1], model.relu1)

    @pytest.mark.cuda
    def test_create_rand_tensors_given_shapes(self):
        shape_1 = (1, 32)
        shape_2 = (3, 3)
        rand_tensors = utils.create_rand_tensors_given_shapes([shape_1, shape_2], device=torch.device('cpu'))
        self.assertEqual(2, len(rand_tensors))
        self.assertEqual(shape_1, rand_tensors[0].shape)
        self.assertEqual(shape_2, rand_tensors[1].shape)
        self.assertEqual(torch.device('cpu'), rand_tensors[0].device)

        rand_tensors = utils.create_rand_tensors_given_shapes([shape_1, shape_2], device=torch.device('cuda:0'))
        self.assertEqual(torch.device('cuda:0'), rand_tensors[0].device)

    @pytest.mark.cuda
    def test_change_tensor_device(self):

        # 1) test only tensor on CPU and GPU

        random_tensor = torch.rand(2, 2)
        random_tensor_new = utils.change_tensor_device_placement(random_tensor, device=torch.device('cuda:0'))

        self.assertEqual(random_tensor.device, torch.device('cpu'))
        self.assertEqual(random_tensor_new.device, torch.device('cuda:0'))

        random_tensor = torch.rand(2, 2).to(device='cuda:0')
        random_tensor_new = utils.change_tensor_device_placement(random_tensor, device=torch.device('cpu'))

        self.assertEqual(random_tensor.device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new.device, torch.device('cpu'))

        # 2) list of tensors

        random_tensor = [
            torch.rand(2, 2),
            torch.rand(2, 2),
            torch.rand(2, 2)
        ]

        random_tensor_new = utils.change_tensor_device_placement(random_tensor, device=torch.device('cuda:0'))

        for item in random_tensor_new:
            self.assertEqual(item.device, torch.device('cuda:0'))

        self.assertEqual(len(random_tensor), len(random_tensor_new))

        random_tensor = [
            torch.rand(2, 2).to(device='cuda:0'),
            torch.rand(2, 2).to(device='cuda:0'),
            torch.rand(2, 2).to(device='cuda:0')
        ]

        random_tensor_new = utils.change_tensor_device_placement(random_tensor, device=torch.device('cpu'))

        for item in random_tensor_new:
            self.assertEqual(item.device, torch.device('cpu'))

        self.assertEqual(len(random_tensor), len(random_tensor_new))

        # 3) list of list of tenors

        random_tensor = [
            [torch.rand(1, 1), torch.rand(1, 1)],
            [torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2)],
            torch.rand(2, 2)
        ]

        random_tensor_new = utils.change_tensor_device_placement(random_tensor, device=torch.device('cuda:0'))

        self.assertEqual(random_tensor_new[0][0].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[0][1].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[1][0].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[1][1].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[1][2].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[1][3].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[2].device, torch.device('cuda:0'))

        self.assertEqual(len(random_tensor), len(random_tensor_new))

        # 4) tuple of tensors
        random_tensor = (
            [torch.rand(1, 1), torch.rand(1, 1)],
            [torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2)],
            (torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2)),
        )
        random_tensor_new = utils.change_tensor_device_placement(random_tensor, device=torch.device('cuda:0'))

        self.assertEqual(random_tensor_new[0][0].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[0][1].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[1][0].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[1][1].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[1][2].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[1][3].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[2][0].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[2][1].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[2][2].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[2][3].device, torch.device('cuda:0'))

        self.assertEqual(len(random_tensor), len(random_tensor_new))
        self.assertTrue(isinstance(random_tensor_new, tuple))
        self.assertTrue(isinstance(random_tensor_new[0], list))
        self.assertTrue(isinstance(random_tensor_new[1], list))
        self.assertTrue(isinstance(random_tensor_new[2], tuple))

        # 4) tuple of tuple of tenors

        random_tensor = (
            (torch.rand(1, 1), torch.rand(1, 1)),
            torch.rand(2, 2),
            torch.rand(2, 2)
        )

        random_tensor_new = utils.change_tensor_device_placement(random_tensor, device=torch.device('cuda:0'))

        self.assertEqual(random_tensor_new[0][0].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[0][1].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[1].device, torch.device('cuda:0'))
        self.assertEqual(random_tensor_new[2].device, torch.device('cuda:0'))

        self.assertEqual(len(random_tensor), len(random_tensor_new))
        self.assertTrue(isinstance(random_tensor_new, tuple))
        self.assertTrue(isinstance(random_tensor_new[0], tuple))

    def _collect_inp_out_data(self, device):
        model = TinyModel().to(device=device)
        model.eval()
        model_input = torch.randn(1, 3, 32, 32).to(device=device)

        module_data = utils.ModuleData(model, model.conv1)
        inp, out = module_data.collect_inp_out_data(model_input, collect_input=False, collect_output=False)
        self.assertEqual(inp, None)
        self.assertEqual(out, None)

        module_data = utils.ModuleData(model, model.conv1)
        inp, out = module_data.collect_inp_out_data(model_input, collect_input=True, collect_output=False)
        self.assertTrue(np.array_equal(utils.to_numpy(inp), utils.to_numpy(model_input)))
        self.assertEqual(out, None)

        module_data = utils.ModuleData(model, model.conv1)
        inp, out = module_data.collect_inp_out_data(model_input, collect_input=False, collect_output=True)
        conv1_out = model.conv1(model_input)
        self.assertTrue(np.array_equal(utils.to_numpy(out), utils.to_numpy(conv1_out)))
        self.assertEqual(inp, None)

        module_data = utils.ModuleData(model, model.conv1)
        inp, out = module_data.collect_inp_out_data(model_input, collect_input=True, collect_output=True)
        conv1_out = model.conv1(model_input)
        self.assertTrue(np.array_equal(utils.to_numpy(out), utils.to_numpy(conv1_out)))
        self.assertTrue(np.array_equal(utils.to_numpy(inp), utils.to_numpy(model_input)))

        module_data = utils.ModuleData(model, model.fc)
        inp, out = module_data.collect_inp_out_data(model_input, collect_input=False, collect_output=True)
        fc_out = model(model_input)
        self.assertTrue(np.array_equal(utils.to_numpy(out), utils.to_numpy(fc_out)))
        self.assertEqual(inp, None)

    def test_collect_inp_out_data_cpu(self):
        """ test collect input output data from module """

        self._collect_inp_out_data(torch.device('cpu'))

    @pytest.mark.cuda
    def test_collect_inp_out_data_gpu(self):
        """ test collect input output data from module """

        self._collect_inp_out_data(torch.device('cuda:0'))

    def _collect_inp_out_data_multi_input(self, device):
        model = MultiInput().to(device=device)
        model.eval()
        inp_shape_1 = (1, 3, 32, 32)
        inp_shape_2 = (1, 3, 20, 20)
        model_input = utils.create_rand_tensors_given_shapes([inp_shape_1, inp_shape_2], device)
        def forward_fn(model, inputs):
            model(*inputs)

        module_data = utils.ModuleData(model, model.conv1, forward_fn)
        inp, out = module_data.collect_inp_out_data(model_input, collect_input=True, collect_output=False)
        self.assertTrue(np.array_equal(utils.to_numpy(inp), utils.to_numpy(model_input[0])))
        self.assertEqual(out, None)

        module_data = utils.ModuleData(model, model.conv1, forward_fn)
        inp, out = module_data.collect_inp_out_data(model_input, collect_input=False, collect_output=True)
        conv1_out = model.conv1(model_input[0])
        self.assertTrue(np.array_equal(utils.to_numpy(out), utils.to_numpy(conv1_out)))
        self.assertEqual(inp, None)

        module_data = utils.ModuleData(model, model.conv3, forward_fn)
        inp, out = module_data.collect_inp_out_data(model_input, collect_input=True, collect_output=True)
        conv3_out = model.conv3(model_input[1])
        self.assertTrue(np.array_equal(utils.to_numpy(out), utils.to_numpy(conv3_out)))
        self.assertTrue(np.array_equal(utils.to_numpy(inp), utils.to_numpy(model_input[1])))

        module_data = utils.ModuleData(model, model.fc, forward_fn)
        inp, out = module_data.collect_inp_out_data(model_input, collect_input=False, collect_output=True)
        fc_out = model(*model_input)
        self.assertTrue(np.array_equal(utils.to_numpy(out), utils.to_numpy(fc_out)))
        self.assertEqual(inp, None)

    def test_collect_inp_out_data_multi_input_cpu(self):
        """ test collect input output data from module using multi input """

        self._collect_inp_out_data_multi_input(torch.device('cpu'))

    @pytest.mark.cuda
    def test_collect_inp_out_data_multi_input_gpu(self):
        """ test collect input output data from module using multi input """

        self._collect_inp_out_data_multi_input(torch.device('cuda:0'))

    def test_collect_inp_out_data_quantsim_model_cpu(self):
        """ test collect input output data from module """

        device_list = [torch.device('cpu')]

        for device in device_list:
            model = TinyModel().to(device=device)
            model_input = torch.randn(1, 3, 32, 32).to(device=device)
            sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 3, 32, 32))

            module_data = utils.ModuleData(model, model.fc)
            inp, out = module_data.collect_inp_out_data(model_input, collect_input=False, collect_output=True)
            fc_out = sim.model(model_input)
            self.assertFalse(np.array_equal(utils.to_numpy(out), utils.to_numpy(fc_out)))

            module_data = utils.ModuleData(model, model.conv1)
            inp, out = module_data.collect_inp_out_data(model_input, collect_input=True, collect_output=False)
            self.assertTrue(np.array_equal(utils.to_numpy(inp), utils.to_numpy(model_input)))

    @pytest.mark.cuda
    def test_collect_inp_out_data_quantsim_model_gpu(self):
        """ test collect input output data from module """

        device_list = [torch.device('cuda:0')]

        for device in device_list:
            model = TinyModel().to(device=device)
            model_input = torch.randn(1, 3, 32, 32).to(device=device)
            sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 3, 32, 32).to(device=device))

            module_data = utils.ModuleData(model, model.fc)
            inp, out = module_data.collect_inp_out_data(model_input, collect_input=False, collect_output=True)
            fc_out = sim.model(model_input)
            self.assertFalse(np.array_equal(utils.to_numpy(out), utils.to_numpy(fc_out)))

            module_data = utils.ModuleData(model, model.conv1)
            inp, out = module_data.collect_inp_out_data(model_input, collect_input=True, collect_output=False)
            self.assertTrue(np.array_equal(utils.to_numpy(inp), utils.to_numpy(model_input)))

    def test_cached_dataset(self):
        """ Test cache data loader splitting into train and validation """
        dataset_size = 256
        batch_size = 16

        # create fake data loader with image size (1, 2, 2)
        data_loader = utils.create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size,
                                                    image_size=(1, 2, 2))
        num_batches = 6
        path = '/tmp/test_cached_dataset/'
        cached_dataset = utils.CachedDataset(data_loader, num_batches, path)
        self.assertEqual(len(cached_dataset), 6)

        # Try creating cached data loader by more than possible batches from data loader and expect ValueError
        possible_batches = math.ceil(dataset_size / batch_size)
        with pytest.raises(ValueError):
            utils.CachedDataset(data_loader, possible_batches + 1, path)

        shutil.rmtree('/tmp/test_cached_dataset/')

    def test_find_num_inout_map(self):
        """
        Test functionality to find cardinality of the inputs, outputs for each leaf module
        """
        model = SingleResidual()
        inout_map = utils.find_num_inout_tensors_per_module(model, [torch.rand(1, 3, 32, 32)])

        inout_counts_check = [num_outputs == (1, 1) for num_outputs in inout_map.values()]
        self.assertTrue(all(inout_counts_check))

        # Create a model with a layer with multi-outputs
        class MyLayer(torch.nn.Module):
            def __init__(self):
                super(MyLayer, self).__init__()

            def forward(self, inputs):
                return inputs * 100, inputs + 100

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, 3)
                self.relu1 = torch.nn.ReLU()
                self.layer1 = MyLayer()
                self.conv2 = torch.nn.Conv2d(32, 32, 3)
                self.conv3 = torch.nn.Conv2d(32, 32, 3)
                self.add = elementwise_ops.Add()

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x1, x2 = self.layer1(x)
                x1 = self.conv2(x1)
                x2 = self.conv2(x2)
                x = self.add(x1, x2)
                return x

        model = MyModel()
        inout_map = utils.find_num_inout_tensors_per_module(model, [torch.rand(1, 3, 32, 32)])
        inout_counts_check = [num_outputs == (1, 1) for num_outputs in inout_map.values()]

        self.assertFalse(all(inout_counts_check))
        self.assertEqual(2, inout_counts_check.count(False))
        self.assertEqual((1, 2), inout_map[model.layer1])
        self.assertEqual((2, 1), inout_map[model.add])

    def test_model_in_eval_mode(self):
        """
        Test in_eval_mode functionality for given model
        """
        model = TinyModel().eval()
        model_input = torch.randn(1, 3, 32, 32)
        #1 model in eval mode in the beginning
        model.eval()
        with utils.in_eval_mode(model):
            model(model_input)
            _assert_mode_recursive(model, training=False)
        _assert_mode_recursive(model, training=False)

        #2 model in train mode in the beginning
        model.train()
        with utils.in_eval_mode(model):
            model(model_input)
            _assert_mode_recursive(model, training=False)
        _assert_mode_recursive(model, training=True)

        #3 model in train mode in the beginning with exception safety check
        model.train()
        try:
            with utils.in_eval_mode(model):
                model(model_input)
                _assert_mode_recursive(model, training=False)
                raise AssertionError   # raise an exception
        except:
            pass
        _assert_mode_recursive(model, training=True)

    def test_model_in_train_mode(self):
        """
        Test in_train_mode functionality for given model
        """
        model = TinyModel().eval()
        model_input = torch.randn(1, 3, 32, 32)
        #1 model in eval mode in the beginning
        model.eval()
        with utils.in_train_mode(model):
            model(model_input)
            _assert_mode_recursive(model, training=True)
        _assert_mode_recursive(model, training=False)

        #2 model in train mode in the beginning
        model.train()
        with utils.in_train_mode(model):
            model(model_input)
            _assert_mode_recursive(model, training=True)
        _assert_mode_recursive(model, training=True)

        #3 model in eval mode in the beginning with exception safety check
        model.eval()
        try:
            with utils.in_train_mode(model):
                model(model_input)
                _assert_mode_recursive(model, training=True)
                raise AssertionError   # raise an exception
        except:
            pass
        _assert_mode_recursive(model, training=False)

    def test_is_torch_module(self):
        """ test _is_torch_nn_module() utility """
        assert utils.is_torch_nn_module(torch.nn.Conv2d(3, 3, 2))
        assert utils.is_torch_nn_module(torch.nn.Linear(3, 10))
        assert utils.is_torch_nn_module(torch.nn.BatchNorm2d(3))
        assert utils.is_torch_nn_module(torch.nn.RNN(input_size=3, hidden_size=5, num_layers=1))
        assert utils.is_torch_nn_module(torch.nn.LSTM(input_size=3, hidden_size=5, num_layers=1, bidirectional=True))
        assert utils.is_torch_nn_module(torch.nn.Sequential(torch.nn.Conv2d(3, 16, 2), torch.nn.BatchNorm2d(16)))
        assert utils.is_torch_nn_module(torch.nn.ModuleList([torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                                                             torch.nn.ReLU(inplace=True),
                                                             torch.nn.Conv2d(16, 8, kernel_size=2)]))
        assert not utils.is_torch_nn_module(elementwise_ops.Add())
        assert not utils.is_torch_nn_module(elementwise_ops.Multiply())
        assert not utils.is_torch_nn_module(elementwise_ops.Concat())

        class CustomModule(torch.nn.Module):
            @staticmethod
            def forward(x):
                return x * F.softplus(x).sigmoid()

        assert not utils.is_torch_nn_module(CustomModule())


def _assert_mode_recursive(root: torch.nn.Module, training: bool):
    for module in root.modules():
        assert module.training == training
