# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

import json
import unittest.mock
import torch
import torch.nn as nn
import numpy as np
from aimet_torch.elementwise_ops import Add, Subtract, Multiply, Divide, Concat
from aimet_torch.quantsim import QuantizationSimModel
import libpymo


class Model(nn.Module):
    def __init__(self, op):
        super(Model, self).__init__()
        self.op1 = op

    def forward(self, input1, input2):
        x = self.op1(input1, input2)
        return x


class Model2(nn.Module):
    def __init__(self, op):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv2d(10, 10, 5, padding=2)
        self.op1 = op

    def forward(self, input):
        x = self.conv1(input)
        x = self.op1(x, input)
        return x


def forward_pass(model, iterations):
    return torch.rand(1)

class TestTrainingExtensionElementwiseOps(unittest.TestCase):
    def test_add_op(self):
        torch.manual_seed(10)
        model = Model(Add())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = input1 + input2
        self.assertTrue(np.allclose(out, out1))

    def test_quantsim_export(self):
        torch.manual_seed(10)
        model = Model2(Add())
        dummy_input = torch.randn(5, 10, 10, 20)
        sim = QuantizationSimModel(model, dummy_input)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5
        encodings.min = -5
        encodings.delta = 1
        encodings.offset = 0.2
        sim.model.op1.output_quantizer.encoding = encodings
        sim.model.conv1.output_quantizer.encoding = encodings
        sim.model.conv1.param_quantizers['weight'].encoding = encodings
        sim.export(path='./data', filename_prefix='quant_model', dummy_input=dummy_input)

        with open('./data/quant_model.encodings') as f:
            data = json.load(f)

        self.assertTrue(len(data['activation_encodings']) == 3)
        self.assertTrue(len(data['param_encodings']) == 2)


    def test_subtract_op(self):
        torch.manual_seed(10)
        model = Model(Subtract())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = input1 - input2
        self.assertTrue(np.allclose(out, out1))

    def test_multiply_op(self):
        torch.manual_seed(10)
        model = Model(Multiply())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = input1 * input2
        self.assertTrue(np.allclose(out, out1))

    def test_divide_op(self):
        torch.manual_seed(10)
        model = Model(Divide())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = torch.div(input1, input2)
        self.assertTrue(np.allclose(out, out1))

    def test_concat_op(self):
        torch.manual_seed(10)
        model = Model(Concat(axis=0))
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = torch.cat((input1, input2), 0)
        self.assertTrue(np.allclose(out, out1))
