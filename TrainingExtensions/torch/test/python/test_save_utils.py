# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
import os

import torch, torch.nn as nn
import numpy as np
import torch.nn.functional as F

from aimet_torch import save_utils as su
from aimet_torch import quantizer as q
import aimet_torch.examples.mnist_torch_model as mnist_model
from aimet_torch.qc_quantize_op import QcQuantizeWrapper


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.l1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.l2 = BasicConv2d(32, 32, kernel_size=3)

    def forward(self, x):
        x = self.l1(x)
        # 149 x 149 x 32
        x = self.l2(x)
        return x


class CommonCode:
    def models(self):
        net = mnist_model.Net()
        model = net.to(torch.device('cpu'))
        quantizer = q.Quantizer(model=model, use_cuda=False)

        # Quantize
        quantizer.quantize_net(run_model=mnist_model.evaluate, bw_params=8, bw_acts=8, iterations=10)
        # Run some inferences
        mnist_model.evaluate(model, 10)

        # train the model again
        # mnist_model.train(model, 1, num_batches=1, batch_callback=check_if_layer_weights_are_updating)
        net = mnist_model.Net()
        original_model = net.to(torch.device('cpu'))
        return model, original_model


class TestSaveUtils(unittest.TestCase):

    @unittest.skip
    def test_save_encodings_to_files(self):
        """
        Test for old deprecated code. Please delete
        """
        save_encodings = su.SaveUtils()
        common = CommonCode()
        quantized_model, _ = common.models()
        filename_prefix = 'test_encoding'
        path = './data/'
        if not os.path.exists(path):
            os.makedirs(path)

        save_encodings.save_encodings_to_files(model=quantized_model, path=path, filename_prefix=filename_prefix,
                                              input_shape=(1, 1, 28, 28))


    def test_save_weights_as_onnx(self):
        sw = su.SaveUtils
        common = CommonCode()
        quantized_model, model = common.models()
        path = './data/'
        if not os.path.exists(path):
            os.makedirs(path)
        filename = 'mnist'
        save_as_onnx = True
        sw.save_weights(quantized_model, model, path, filename, save_as_onnx, 'module_to_wrap',
                        torch.randn(1, 1, 28, 28, requires_grad=True))

    def test_save_weights_as_pth(self):
        sw = su.SaveUtils
        common = CommonCode()
        quantized_model, model = common.models()
        path = './data/'
        if not os.path.exists(path):
            os.makedirs(path)
        filename = 'mnist'
        save_as_onnx = False
        sw.save_weights(quantized_model, model, path, filename, save_as_onnx, 'module_to_wrap',
                        torch.randn(1, 1, 28, 28, requires_grad=True))

    def test_pytorch_to_onnx_dict(self):
        model = DummyModel()
        input_shape = [1, 3, 299, 299]
        x_input = torch.rand(input_shape)
        pytorch_onnx_names_dict = su.SaveUtils().get_name_of_op_from_graph(model, x_input)
        self.assertTrue(pytorch_onnx_names_dict['l1.bn'],
                        'DummyModel/BasicConv2d[l1]/BatchNorm2d[bn]/BatchNormalization_12')

    def test_pytorch_to_onnx_dict(self):
        common = CommonCode()
        quantized_model, _ = common.models()
        su.SaveUtils.remove_quantization_wrappers(quantized_model)

        for module_name, module in quantized_model.named_children():
            self.assertFalse(isinstance(module, QcQuantizeWrapper))



def check_if_layer_weights_are_updating(model, batch_idx):

    # Creating an alias for easier reference
    f = check_if_layer_weights_are_updating

    # get the initial values of some layers
    conv1_w_value = model.conv1._module_to_wrap.weight

    if batch_idx != 0:
        assert not np.allclose(conv1_w_value.detach().numpy(), f.conv1_w_value_old.detach().numpy())
    else:
        f.conv1_w_value_old = conv1_w_value.clone()
