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
""" Tests for blockwise quant tensor split utility """

import json
import pytest
import torch
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch.blockwise_quant_tensor_split import (BlockwiseLinear, replace_linears_for_blockwise_quant,
                                                      tie_blockwise_linear_quantizers)

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 3)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(3, 2, bias=False)
        self.softmax = torch.nn.Softmax()

    def forward(self, inp):
        x = self.linear1(inp)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

class InnerModel(torch.nn.Module):
    def __init__(self):
        super(InnerModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 3)
        self.seq = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 5), torch.nn.ReLU())
        self.seq2 = torch.nn.Sequential(torch.nn.Linear(5, 2))

    def forward(self, inp):
        x = self.linear1(inp)
        x = self.seq(x)
        x = self.seq2(x)
        return x

class NestedModel(torch.nn.Module):
    def __init__(self):
        super(NestedModel, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.inner_linears = InnerModel()

    def forward(self, inp):
        x = self.inner_linears(inp)
        x = self.sigmoid(x)
        return x

@pytest.fixture(scope='module')
def create_per_channel_config():
    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "False"
            },
            "params": {
                "is_quantized": "False",
                "is_symmetric": "True"
            },
            "per_channel_quantization": "True",
        },
        "params": {},
        "op_type": {
            'Split': {
                'is_output_quantized': False
            }
        },
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }
    with open('./data/quantsim_config.json', 'w') as f:
        json.dump(quantsim_config, f)

@pytest.mark.parametrize('model, dummy_input, block_size', [(torch.nn.Linear(8, 3), torch.randn(1, 8), 3),
                                                             (torch.nn.Linear(8, 3, bias=False), torch.randn(1, 8), 3),
                                                             (torch.nn.Linear(3, 2), torch.randn(1, 3), 3),
                                                             (torch.nn.Linear(3, 2), torch.randn(1, 3), 4)])
def test_blockwise_linear(model, dummy_input, block_size):
    blockwise_linear = BlockwiseLinear(model, block_size=block_size)
    orig_out = model(dummy_input)
    new_out = blockwise_linear(dummy_input)
    assert torch.allclose(orig_out, new_out, atol=1e-6)

def test_replace_linears_for_blockwise_quant():
    dummy_input = torch.randn(1, 8)
    model = LinearModel()
    linear1 = model.linear1
    orig_out = model(dummy_input)
    replace_linears_for_blockwise_quant(model, 3)

    assert len(model.linear1.linears) == 3
    assert torch.equal(model.linear1.linears[0].bias, linear1.bias)
    for linear in model.linear1.linears[1:]:
        assert linear.bias is None

    assert len(model.linear1.elementwise_adds) == 2
    assert len(model.linear2.linears) == 1
    assert model.linear2.elementwise_adds is None
    new_out = model(dummy_input)
    assert torch.allclose(orig_out, new_out, atol=1e-6)

def test_quantize_blockwise_linear(create_per_channel_config):
    dummy_input = torch.randn(1, 8)
    model = BlockwiseLinear(torch.nn.Linear(8, 3), 3)
    qsim = QuantizationSimModel(model, dummy_input=dummy_input)

    assert isinstance(qsim.model.split, QcQuantizeWrapper)
    assert isinstance(qsim.model.linears[0], QcQuantizeWrapper)
    assert isinstance(qsim.model.linears[1], QcQuantizeWrapper)
    assert isinstance(qsim.model.linears[2], QcQuantizeWrapper)
    assert isinstance(qsim.model.elementwise_adds[0], QcQuantizeWrapper)
    assert isinstance(qsim.model.elementwise_adds[1], QcQuantizeWrapper)

    # Temporary hack to disable split op output quantizers while handling for CG split op is reworked
    for output_quantizer in qsim.model.split.output_quantizers:
        output_quantizer.enabled = False

    # Temporary hack to enable model input split op input quantizer while handling for CG split op is reworked
    qsim.model.split.input_quantizers[0].enabled = True

    qsim.compute_encodings(lambda m, _: m(dummy_input), None)
    tie_blockwise_linear_quantizers(qsim)
    _ = qsim.model(dummy_input)

    assert (qsim.model.linears[0].output_quantizers[0].encoding.max ==
            qsim.model.linears[1].output_quantizers[0].encoding.max)
    assert (qsim.model.linears[0].output_quantizers[0].encoding.max ==
            qsim.model.linears[2].output_quantizers[0].encoding.max)
    assert (qsim.model.linears[0].output_quantizers[0].encoding.max ==
            qsim.model.elementwise_adds[0].output_quantizers[0].encoding.max)
    assert (qsim.model.linears[0].output_quantizers[0].encoding.max ==
            qsim.model.elementwise_adds[1].output_quantizers[0].encoding.max)

def test_blockwise_quant_with_small_linear(create_per_channel_config):
    dummy_input = torch.randn(1, 3)
    model = BlockwiseLinear(torch.nn.Linear(3, 2), 3)
    qsim = QuantizationSimModel(model, dummy_input=dummy_input)
    # Temporary hack to disable split op output quantizers while handling for CG split op is reworked
    for output_quantizer in qsim.model.split.output_quantizers:
        output_quantizer.enabled = False

    # Temporary hack to enable model input split op input quantizer while handling for CG split op is reworked
    qsim.model.split.input_quantizers[0].enabled = True

    tie_blockwise_linear_quantizers(qsim)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)
    assert len(qsim.model.linears) == 1
    assert qsim.model.elementwise_adds is None

    _ = qsim.model(dummy_input)
    assert len(qsim.connected_graph.get_all_ops()) == 1

def test_nested_sequential_linears():
    model = NestedModel()
    replace_linears_for_blockwise_quant(model, block_size=2)
    num_blockwise_linears = 0
    for _, module in model.named_modules():
        if isinstance(module, BlockwiseLinear):
            num_blockwise_linears += 1
    assert num_blockwise_linears == 4
