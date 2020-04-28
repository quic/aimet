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
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

import aimet_torch.examples.mnist_torch_model as mnist_model
from aimet_torch import bias_correction
from aimet_torch.utils import to_numpy, create_fake_data_loader, get_ordered_list_of_conv_modules
from aimet_torch.bias_correction import find_all_conv_bn_with_activation
from aimet_torch import quantsim as qsim
from aimet_torch.examples.mobilenet import MockMobileNetV11 as MockMobileNetV1


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.fc1 = nn.Linear(160, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = functional.relu(functional.max_pool2d(self.conv1(x), 2))
        x = functional.relu(functional.max_pool2d(self.conv2(x), 2))
        x = x.view(x.view(0), -1)
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return functional.log_softmax(x, dim=1)


class TestTrainingExtensionBnFold(unittest.TestCase):
    def test_get_output_of_layer(self):
        model = TestNet().cuda()
        dataset_size = 2
        batch_size = 2
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size)
        for images_in_one_batch, _ in data_loader:
            conv2_output_data = bias_correction.get_output_data(model.conv2, model, images_in_one_batch)

        # max out number of batches
        number_of_batches = 1
        iterator = data_loader.__iter__()

        for batch in range(number_of_batches):

            images_in_one_batch, _ = iterator.__next__()
            conv1_output = model.conv1(images_in_one_batch.cuda())
            conv2_input = conv1_output
            conv2_output = model.conv2(functional.relu(functional.max_pool2d(conv2_input, 2)))
            # compare the output from conv2 layer
            self.assertTrue(np.allclose(to_numpy(conv2_output),
                                        np.asarray(conv2_output_data)[batch * batch_size: (batch + 1) *
                                                                                          batch_size, :, :, :]))

    def test_get_ordering_of_nodes_in_model(self):
        model = mnist_model.ExtendedNet()
        list_modules = get_ordered_list_of_conv_modules(model, input_shapes=(1, 1, 28, 28))
        self.assertEqual(list_modules[0][0], 'conv1')
        self.assertEqual(list_modules[1][0], 'conv2')

    def test_get_quantized_weight(self):
        model = mnist_model.Net()

        params = qsim.QuantParams(weight_bw=4, act_bw=4, round_mode="nearest",
                                       quant_scheme='tf'
                                 )
        use_cuda = False
        dataset_size = 2
        batch_size = 1
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size)
        def pass_data_through_model(model, early_stopping_iterations=None, use_cuda=False):
        # forward pass for given number of batches for model
            for _, (images_in_one_batch, _) in enumerate(data_loader):
                model(images_in_one_batch)

        quantsim = qsim.QuantizationSimModel(model=model, quant_scheme=params.quant_scheme,
                                             rounding_mode=params.round_mode,
                                             default_output_bw=params.act_bw,
                                             default_param_bw=params.weight_bw,
                                             in_place=False)
        quantsim.compute_encodings(pass_data_through_model, None)
        layer = quantsim.model.conv2
        quant_dequant_weights = bias_correction.get_quantized_dequantized_weight(layer, use_cuda)
        self.assertEqual(quant_dequant_weights.shape, torch.Size([64, 32, 5, 5]))

    def test_bias_correction_analytical_and_empirical(self):
        torch.manual_seed(10)
        model = MockMobileNetV1()
        model = model.eval()
        dataset_size = 2
        batch_size = 1

        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size, image_size=(3, 224, 224))

        params = qsim.QuantParams(weight_bw=8, act_bw=8, round_mode="nearest",
                                  quant_scheme='tf'
                                 )
        conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape=(1, 3, 224, 224))

        with unittest.mock.patch('aimet_torch.bias_correction.call_analytical_mo_correct_bias') as analytical_mock:
            with unittest.mock.patch('aimet_torch.bias_correction.call_empirical_mo_correct_bias') as empirical_mock:
                bias_correction.correct_bias(model, params, 2, data_loader, 2, conv_bn_dict, perform_only_empirical_bias_corr=False)
        self.assertEqual(analytical_mock.call_count, 9)
        self.assertEqual(empirical_mock.call_count, 9)
        self.assertTrue(model.model[1][0].bias.detach().cpu().numpy() is not None)

    def test_bias_correction_empirical(self):
        # Using a dummy extension of MNIST
        torch.manual_seed(10)
        model = mnist_model.Net()

        model = model.eval()

        model_copy = copy.deepcopy(model)
        dataset_size = 2
        batch_size = 1

        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size, image_size=(1, 28, 28))

        params = qsim.QuantParams(weight_bw=4, act_bw=4, round_mode="nearest",
                                  quant_scheme='tf'
                                  )
        with unittest.mock.patch('aimet_torch.bias_correction.call_empirical_mo_correct_bias') as empirical_mock:
            bias_correction.correct_bias(model, params, 2, data_loader, 2)

        self.assertEqual(empirical_mock.call_count, 4)
        self.assertTrue(np.allclose(model.conv1.bias.detach().cpu().numpy(),
                                    model_copy.conv1.bias.detach().cpu().numpy()))

        self.assertTrue(model.conv2.bias.detach().cpu().numpy() is not None)
        self.assertTrue(model.fc1.bias.detach().cpu().numpy() is not None)

    def test_layer_selection_bn_based_bc_no_residual(self):
        model = MockMobileNetV1()
        model = model.eval()
        conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape=(1, 3, 224, 224))
        conv_2 = model.model[1][0]
        self.assertEqual(conv_bn_dict[conv_2].output_bn, None)
        self.assertEqual(18, len(conv_bn_dict))

    def test_bias_correction_with_updated_quantsim(self):
        '''
        tests the updated quantizer
        :return:
        '''
        def _hook_to_collect_output_data(module, _, out_data):
            """
            hook to collect output data
            """
            from aimet_torch import utils
            out_data = utils.to_numpy(out_data)
            orig_layer_out_data.append(out_data)

        torch.manual_seed(10)
        model = MockMobileNetV1()
        model.eval()
        dataset_size = 2
        batch_size = 1

        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size, image_size=(3, 224, 224))
        params = qsim.QuantParams(weight_bw=8, act_bw=8, round_mode="nearest",
                                  quant_scheme='tf'
                                 )
        conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape=(1, 3, 224, 224))

        bias_correction.correct_bias(model, params, 2, data_loader, 2, conv_bn_dict, perform_only_empirical_bias_corr = True)


        # For layer3 we perform empirical BC that uses quantizer.
        # check the bias and compare
        new_model_1_2_bias  = model.model[1][2].bias.detach().cpu().numpy().reshape(-1)
        new_model_1_2_weights  = model.model[1][2].weight.detach().cpu().numpy().reshape(-1)

        # in case we want to check layer output
        hook_handles = list()
        orig_layer_out_data = list()
        hook_handles.append(model.model[1][2].register_forward_hook(_hook_to_collect_output_data))

        np.random.seed(0)
        inp_data = torch.rand(1, 3, 224, 224)

        with torch.no_grad():
            _ = model(inp_data)

        _ = orig_layer_out_data[0].reshape(-1)

        for hook_handle in hook_handles:
            hook_handle.remove()

        # we recorded the bias we observed when old quantizer was used.
        ref_model_1_2_bias = [0.13416477, 0.16463749, 0.006915508, -0.11823632, -0.042272247, -0.1231364, -0.14524677, 0.11495129, 0.13525873,
                    -0.12211427, 0.13239658, -0.026878614, 0.032136727, -0.049814742, 0.1496965, 0.023653626, 0.004951938, -0.07484536,
                    0.15355694, 0.05401889, 0.004869867, 0.13777985, 0.052989908, -0.006976959, 0.040764436, -0.006203071, 0.13691252,
                    -0.14307816, -0.13579868, -0.13622427, 0.10511946, 0.00884756, 0.075533904, -0.036618445, 0.07738414, -0.14853345,
                    0.091384344, -0.16644184, -0.11411467, -0.0064487234, -0.0132480515, 0.010812189, -0.099912055, 0.15872453, 0.17395662,
                    -0.1206288, 0.17381206, 0.16031563, -0.009573824, 0.11802861, -0.15066624, -0.05074876, -0.14428341, -0.17168832,
                    0.045424946, -0.11126628, -0.09583544, 0.03932162, -0.123743564, 0.06084253, 0.09650699, -0.02348134, 0.14565137, -0.017025044]

        # validation : compare bias with new quantisim implementation with ref_model_1_2_bias
        self.assertTrue(np.allclose(ref_model_1_2_bias, new_model_1_2_bias, rtol=1e-1))

    def test_bias_correction_analytical_and_empirical_ignore_layer(self):
        '''
        Test bias correction with ignore layers specified.
        :return:
        '''
        torch.manual_seed(10)
        model = MockMobileNetV1()
        model = model.eval()
        dataset_size = 2
        batch_size = 1

        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size, image_size=(3, 224, 224))
        params = qsim.QuantParams(weight_bw=8, act_bw=8, round_mode="nearest",
                                  quant_scheme='tf'
                                  )
        conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape=(1, 3, 224, 224))

        layer = model.model[0][0]
        layers_to_ignore = [layer]

        with unittest.mock.patch('aimet_torch.bias_correction.call_analytical_mo_correct_bias') as analytical_mock:
            with unittest.mock.patch('aimet_torch.bias_correction.call_empirical_mo_correct_bias') as empirical_mock:
                bias_correction.correct_bias(model, params, 2, data_loader, 2, conv_bn_dict,
                                             perform_only_empirical_bias_corr=False,
                                             layers_to_ignore = layers_to_ignore)

        self.assertEqual(analytical_mock.call_count, 8) # one layer ignored
        self.assertEqual(empirical_mock.call_count, 9)
        self.assertTrue(model.model[1][0].bias.detach().cpu().numpy() is not None)
