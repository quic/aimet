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
import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme
from aimet_torch.elementwise_ops import Add, Subtract, Multiply, Divide, Concat, MatMul
from aimet_torch.quantsim import QuantizationSimModel


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


class Model3(nn.Module):
    def __init__(self, op):
        super(Model3, self).__init__()
        self.op1 = op

    def forward(self, *x):
        x = self.op1(*x)
        return x


def dummy_forward_pass(model, args):
    model.eval()
    with torch.no_grad():
        output = model(torch.randn((5, 10, 10, 20)))
    return output


def evaluate(model: torch.nn.Module, dummy_input: torch.Tensor):
    """
    Helper function to evaluate model given dummy input
    :param model: torch model
    :param dummy_input: dummy input to model
    """
    model.eval()
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = [dummy_input]
    with torch.no_grad():
        model(*dummy_input)


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
        sim.model.op1.output_quantizers[0].encoding = encodings
        sim.model.conv1.output_quantizers[0].encoding = encodings
        sim.model.conv1.param_quantizers['weight'].encoding = encodings
        sim.export(path='./data', filename_prefix='quant_model', dummy_input=dummy_input)

        with open('./data/quant_model.encodings') as f:
            data = json.load(f)

        self.assertTrue(len(data['activation_encodings']) == 3)
        self.assertTrue(len(data['param_encodings']) == 1)

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

    def test_concat_op_two_input_tensors(self):
        torch.manual_seed(10)
        model = Model3(Concat())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = torch.cat((input1, input2), 0)
        self.assertTrue(np.allclose(out, out1))

    def test_concat_op_four_input_tensors(self):
        torch.manual_seed(10)
        model = Model3(Concat())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        input3 = torch.rand((5, 10, 10, 20))
        input4 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2, input3, input4)
        out1 = torch.cat((input1, input2, input3, input4), 0)
        self.assertTrue(np.allclose(out, out1))

    def test_concat_compute_encodings(self):
        torch.manual_seed(10)
        model = Model3(Concat())
        dummy_input = torch.randn(5, 10, 10, 20)
        sim = QuantizationSimModel(model, dummy_input)
        sim.compute_encodings(dummy_forward_pass, None)
        print(sim)
        sim.export(path='./data', filename_prefix='concat_model', dummy_input=dummy_input)

    def test_matmul_op(self):
        torch.manual_seed(10)
        model = Model(MatMul())
        tensor1 = torch.randn(10, 3, 4)
        tensor2 = torch.randn(10, 4, 5)
        out = model(tensor1, tensor2)
        out1 = torch.matmul(tensor1, tensor2)
        self.assertTrue(np.allclose(out, out1))

    def test_concat_op_with_qat(self):
        """
        Test torch.cat op for both QAT and QAT-range learning
        """
        class ModelWithConCatWrapper(nn.Module):
            """ A model with concat wrapper. Use this model for unit testing purposes.
                Expected inputs: 3 inputs, of size (1, 3, 8, 8) """

            def __init__(self):
                super(ModelWithConCatWrapper, self).__init__()
                self.conv1 = nn.Conv2d(10, 10, 5, bias=False)
                self.conv2 = nn.Conv2d(10, 10, 5)
                self.conv3 = nn.Conv2d(10, 10, 5)
                self.conv4 = nn.Conv2d(10, 10, 1)
                self.cat = Concat(1)
                self.relu1 = nn.ReLU(inplace=True)
                self.relu2 = nn.ReLU(inplace=True)

            def forward(self, *inputs):
                c1 = self.conv1(inputs[0])
                c1 = self.relu1(c1)
                c2 = self.conv2(inputs[1])
                c3 = self.conv3(inputs[2])
                c3 = self.conv4(c3)
                cat_inputs = [c1, c2, c3]
                x = self.cat(*cat_inputs)
                x = self.relu2(x)
                return x

        # Test with torch.cat dim=1 with both QAT and QAT-range learning.
        input_shape = (1, 10, 10, 20)
        dummy_input = [torch.randn(*input_shape), torch.randn(*input_shape), torch.randn(*input_shape)]
        model = ModelWithConCatWrapper().eval()
        quant_schemes = [QuantScheme.post_training_tf_enhanced,
                         QuantScheme.training_range_learning_with_tf_init,
                         QuantScheme.post_training_tf_enhanced,
                         QuantScheme.training_range_learning_with_tf_enhanced_init]
        for quant_scheme in quant_schemes:
            quant_sim = QuantizationSimModel(model, dummy_input, quant_scheme=quant_scheme)
            quant_sim.compute_encodings(evaluate, dummy_input)
            quant_sim.model.train()
            out = quant_sim.model(*dummy_input)
            loss = out.flatten().sum()
            loss.backward()
            assert quant_sim.model.cat.output_quantizers[0] is not None

    def test_dtypes_to_ignore_for_quantization(self):
        """
        test dtypes to be ignored for quantization when inputs to elementwise ops are scalar numbers.
        We just skip quantization for scalars.
        """
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(10, 20, 5)
                self.add = Add()
                self.mul = Multiply()

            def forward(self, input1):
                x = self.conv(input1)
                x = self.add(x, 2)
                x = self.mul(x, 1)
                return x

        model = Model().eval()
        dummy_input = torch.randn(1, 10, 10, 20)
        model(dummy_input)
        sim = QuantizationSimModel(model, dummy_input)
        sim.compute_encodings(dummy_forward_pass, None)
        sim.model(dummy_input)

        assert not sim.model.add.input_quantizers[0].encoding
        assert not sim.model.add.input_quantizers[1].encoding
        assert sim.model.add.output_quantizers[0].encoding

        assert not sim.model.mul.input_quantizers[0].encoding
        assert not sim.model.mul.input_quantizers[1].encoding
        assert sim.model.mul.output_quantizers[0].encoding

    def test_dtypes_to_ignore_for_quantization_quant_scheme_range_learning(self):
        """
        test dtypes to be ignored for quantization when inputs to elementwise ops are scalar numbers.
        We just skip quantization for scalars.
        """
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(10, 20, 5)
                self.add = Add()
                self.mul = Multiply()

            def forward(self, input1):
                x = self.conv(input1)
                x = self.add(x, 2)
                x = self.mul(x, 1)
                return x

        model = Model().eval()
        dummy_input = torch.randn(1, 10, 10, 20)
        model(dummy_input)
        sim = QuantizationSimModel(model, dummy_input,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_enhanced_init)
        sim.compute_encodings(dummy_forward_pass, None)
        sim.model.train()
        out = sim.model(dummy_input)
        loss = out.flatten().sum()
        loss.backward()

        assert not sim.model.add.input_quantizers[0].encoding
        assert not sim.model.add.input_quantizers[1].encoding
        assert sim.model.add.output_quantizers[0].encoding

        assert not sim.model.mul.input_quantizers[0].encoding
        assert not sim.model.mul.input_quantizers[1].encoding
        assert sim.model.mul.output_quantizers[0].encoding
