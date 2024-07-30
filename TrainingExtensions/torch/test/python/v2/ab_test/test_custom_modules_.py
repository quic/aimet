# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import tempfile
import torch
import torch.nn as nn
from aimet_common.defs import QuantScheme
import aimet_torch.v2.nn.modules.custom as custom
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch import utils as v1_utils


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


def dummy_forward(model: torch.nn.Module, input: torch.Tensor):
    if isinstance(input, torch.Tensor):
        input = [input]
    with torch.no_grad(), v1_utils.in_eval_mode(model):
        model(*input)


# From https://github.com/quic/aimet/blob/8ed479b24010834bfea09885cf6879b9bd916e8a/TrainingExtensions/torch/test/python/test_elementwise_ops.py#L101
class TestTrainingExtensionElementwiseOps:
    def test_quantsim_export(self):
        model = Model2(custom.Add())
        dummy_input = torch.randn(5, 10, 10, 20)
        sim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf)

        sim.model.op1.output_quantizers[0].bitwidth = 8
        sim.model.op1.output_quantizers[0].min.copy_(-5)
        sim.model.op1.output_quantizers[0].max.copy_(5)

        sim.model.op1.input_quantizers[1] = None

        sim.model.conv1.output_quantizers[0].bitwidth = 8
        sim.model.conv1.output_quantizers[0].min.copy_(-5)
        sim.model.conv1.output_quantizers[0].max.copy_(5)

        sim.model.conv1.param_quantizers['weight'].bitwidth = 8
        sim.model.conv1.param_quantizers['weight'].min.copy_(-5)
        sim.model.conv1.param_quantizers['weight'].max.copy_(5)

        with tempfile.TemporaryDirectory() as path:
            sim.export(path=path, filename_prefix='quant_model', dummy_input=dummy_input)

            with open(f'{path}/quant_model.encodings') as f:
                data = json.load(f)

            assert len(data['activation_encodings']) == 2
            assert len(data['param_encodings']) == 1

    def test_concat_compute_encodings(self):
        torch.manual_seed(10)
        model = Model3(custom.Concat())
        dummy_input = torch.randn(5, 10, 10, 20), torch.randn(5, 10, 10, 20)
        sim = QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf)
        sim.compute_encodings(dummy_forward, dummy_input)
        with tempfile.TemporaryDirectory() as path:
            sim.export(path=path, filename_prefix='concat_model', dummy_input=dummy_input)

    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init])
    def test_concat_op_with_qat(self, quant_scheme):
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
                self.cat = custom.Concat(1)
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
        quant_sim = QuantizationSimModel(model, dummy_input, quant_scheme=quant_scheme)
        quant_sim.compute_encodings(dummy_forward, dummy_input)
        quant_sim.model.train()
        out = quant_sim.model(*dummy_input)
        loss = out.flatten().sum()
        loss.backward()

        if quant_scheme == QuantScheme.post_training_tf:
            assert quant_sim.model.cat.output_quantizers[0].min.grad is None
            assert quant_sim.model.cat.output_quantizers[0].max.grad is None
        elif quant_scheme == QuantScheme.training_range_learning_with_tf_init:
            assert quant_sim.model.cat.output_quantizers[0].min.grad is not None
            assert quant_sim.model.cat.output_quantizers[0].max.grad is not None
        else:
            raise

    @pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                              QuantScheme.training_range_learning_with_tf_init])
    def test_dtypes_to_ignore_for_quantization(self, quant_scheme):
        """
        test dtypes to be ignored for quantization when inputs to elementwise ops are scalar numbers.
        We just skip quantization for scalars.
        """
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(10, 20, 5)
                self.add = custom.Add()
                self.mul = custom.Multiply()

            def forward(self, x, y, z):
                x = self.conv(x)
                x = self.add(x, y)
                x = self.mul(x, z)
                return x

        model = Model().eval()
        dummy_input = (torch.randn(1, 10, 10, 20),
                       torch.tensor([2.0]),
                       torch.tensor([1.0]))
        sim = QuantizationSimModel(model, dummy_input, quant_scheme=quant_scheme)
        """
        When: Pass integers as input during compute_encodings
        Then: The input quantizers that took the integers as input should not be initialized
        """
        sim.compute_encodings(dummy_forward, (dummy_input[0], 2, 1))

        assert sim.model.add.input_quantizers[0] is None
        assert not sim.model.add.input_quantizers[1].is_initialized()
        assert sim.model.add.output_quantizers[0].is_initialized()

        assert sim.model.mul.input_quantizers[0] is None
        assert not sim.model.mul.input_quantizers[1].is_initialized()
        assert sim.model.mul.output_quantizers[0].is_initialized()
