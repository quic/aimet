# /usr/bin/env python3.5
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
import torch
import torchvision

from aimet_common.utils import round_up_to_multiplicity, round_down_to_multiplicity
from aimet_torch import utils

from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.defs import PassThroughOp
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
                                                         PassThroughOp)

        # check - given modules have been replaced
        self.assertTrue(isinstance(model.layer1[0].bn1, PassThroughOp))
        self.assertTrue(isinstance(model.layer1[1].bn1, PassThroughOp))

        # check - other bn layers have not been modified
        self.assertFalse(isinstance(model.layer1[0].bn2, PassThroughOp))
        self.assertFalse(isinstance(model.layer1[1].bn2, PassThroughOp))

        # sanity-check: forward pass continues to work
        with torch.no_grad():
            x = torch.rand(1, 3, 224, 224)
            output = model(x)

    def test_get_ordered_ops(self):
        model = torchvision.models.resnet18(pretrained=False)
        model.eval()

        all_ops = utils.get_ordered_list_of_modules(model, (1, 3, 224, 224))
        conv_ops = utils.get_ordered_list_of_conv_modules(model, (1, 3, 224, 224))

        self.assertEqual(60, len(all_ops))
        self.assertEqual(20, len(conv_ops))
        for _, module in conv_ops:
            self.assertTrue(isinstance(module, torch.nn.Conv2d))

    def test_get_reused_modules(self):
        """ Test get_reused_modules utility """
        model = ModelWithReusedNodes()
        inp_shape = (1, 3, 32, 32)
        reused_modules = utils.get_reused_modules(model, inp_shape)
        self.assertEqual(1, len(reused_modules))
        self.assertEqual(reused_modules[0][1], model.relu1)

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
        inp_shape_1 = (1, 3, 32, 32)
        inp_shape_2 = (1, 3, 20, 20)
        model_input = utils.create_rand_tensors_given_shapes([inp_shape_1, inp_shape_2])

        module_data = utils.ModuleData(model, model.conv1)
        inp, out = module_data.collect_inp_out_data(model_input, collect_input=True, collect_output=False)
        self.assertTrue(np.array_equal(utils.to_numpy(inp), utils.to_numpy(model_input[0])))
        self.assertEqual(out, None)

        module_data = utils.ModuleData(model, model.conv1)
        inp, out = module_data.collect_inp_out_data(model_input, collect_input=False, collect_output=True)
        conv1_out = model.conv1(model_input[0])
        self.assertTrue(np.array_equal(utils.to_numpy(out), utils.to_numpy(conv1_out)))
        self.assertEqual(inp, None)

        module_data = utils.ModuleData(model, model.conv3)
        inp, out = module_data.collect_inp_out_data(model_input, collect_input=True, collect_output=True)
        conv3_out = model.conv3(model_input[1])
        self.assertTrue(np.array_equal(utils.to_numpy(out), utils.to_numpy(conv3_out)))
        self.assertTrue(np.array_equal(utils.to_numpy(inp), utils.to_numpy(model_input[1])))

        module_data = utils.ModuleData(model, model.fc)
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
            sim = QuantizationSimModel(model, input_shapes=(1, 3, 32, 32))

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
            sim = QuantizationSimModel(model, input_shapes=(1, 3, 32, 32))

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
        possible_batches = int(dataset_size / batch_size)
        with pytest.raises(ValueError):
            utils.CachedDataset(data_loader, possible_batches + 1, path)

        shutil.rmtree('/tmp/test_cached_dataset/')

    def test_find_num_outputs_map(self):
        """
        Test functionality to find cardinality of the outputs for each module
        """
        model = SingleResidual()
        outputs_map = utils.find_num_output_tensors_per_module(model, [torch.rand(1, 3, 32, 32)])
        outputs_counts_check = [num_outputs == 1 for num_outputs in outputs_map.values()]

        self.assertTrue(all(outputs_counts_check))

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

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x1, x2 = self.layer1(x)
                x1 = self.conv2(x1)
                x2 = self.conv2(x2)
                return x1 + x2

        model = MyModel()
        outputs_map = utils.find_num_output_tensors_per_module(model, [torch.rand(1, 3, 32, 32)])
        outputs_counts_check = [num_outputs == 1 for num_outputs in outputs_map.values()]

        self.assertFalse(all(outputs_counts_check))
        self.assertEqual(1, outputs_counts_check.count(False))
        self.assertEqual(2, outputs_map[model.layer1])
