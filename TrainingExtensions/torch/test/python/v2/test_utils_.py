# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
import os
import pytest
import tempfile
import torch

from .models_.test_models import ModelWithMatMul2
from aimet_common.defs import QuantScheme
from aimet_torch.v2.experimental import set_matmul_second_input_producer_to_8bit_symmetric
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.utils import allow_recompute, enable_recompute, reduce, patch_attr

@pytest.mark.parametrize('reduce_dim, target_shape', [
    # | reduce dim   | target shape |
    # | -------------|--------------|
    (   [0,1,2,3],     []          ),

    (   [0,1,2],       [6]         ),
    (   [0,1,2],       [1,6]       ),
    (   [0,1,2],       [1,1,6]     ),
    (   [0,1,2],       [1,1,1,6]   ),
    (   [0,1,3],       [5,1]       ),
    (   [0,1,3],       [1,5,1]     ),
    (   [0,1,3],       [1,1,5,1]   ),
    (   [0,2,3],       [4,1,1]     ),
    (   [0,2,3],       [1,4,1,1]   ),
    (   [1,2,3],       [3,1,1,1]   ),

    (   [0,1],         [5,6]       ),
    (   [0,1],         [1,5,6]     ),
    (   [0,1],         [1,1,5,6]   ),
    (   [0,2],         [4,1,6]     ),
    (   [0,2],         [1,4,1,6]   ),
    (   [1,2],         [3,1,1,6]   ),
    (   [0,3],         [4,5,1]     ),
    (   [0,3],         [1,4,5,1]   ),
    (   [1,3],         [3,1,5,1]   ),
    (   [2,3],         [3,4,1,1]   ),

    (   [0],           [4,5,6]     ),
    (   [0],           [1,4,5,6]   ),
    (   [1],           [3,1,5,6]   ),
    (   [2],           [3,4,1,6]   ),
    (   [3],           [3,4,5,1]   ),
])
def test_reduce(reduce_dim, target_shape):
    x = torch.arange(start=0, end=3*4*5*6).view(3,4,5,6)
    out = reduce(x, target_shape, torch.sum)
    expected = torch.sum(x, dim=reduce_dim, keepdim=True)
    assert list(out.shape) == list(target_shape)
    assert torch.allclose(out, expected)


def test_patch_attr():
    conv = torch.nn.Conv2d(3, 3, 3)
    old_forward = conv.forward
    old_dict = conv.__dict__.copy()

    with patch_attr(conv, 'forward', lambda x: x):
        pass

    assert conv.forward == old_forward
    assert old_dict == conv.__dict__

    replica = conv._replicate_for_data_parallel()
    assert replica.forward.__self__ is replica

    with patch_attr(conv, 'no_exist_attribute', 1):
        assert conv.no_exist_attribute == 1

    assert not hasattr(conv, 'no_exist_attribute')


@pytest.fixture
def use_deterministic_algorithms():
    orig_flag = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(orig_flag)


@pytest.mark.cuda
def test_allow_recompute(use_deterministic_algorithms):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.relu1 = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(3, 3, 3)
            self.relu2 = torch.nn.ReLU()

        @allow_recompute
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            return x

    model = Model().cuda()
    x = torch.randn((100, 3, 224, 224), device="cuda:0")

    torch.cuda.empty_cache()
    with enable_recompute():
        out = model(x)
    torch.cuda.synchronize()
    mem_with_recompute = torch.cuda.memory_allocated()

    out.backward(torch.ones_like(out))
    conv1_grad_with_recompute = model.conv1.weight.grad.clone().detach().cpu()
    conv2_grad_with_recompute = model.conv2.weight.grad.clone().detach().cpu()

    del out
    model.conv1.weight.grad = None
    model.conv2.weight.grad = None

    torch.cuda.empty_cache()
    out = model(x)
    torch.cuda.synchronize()
    mem_without_recompute = torch.cuda.memory_allocated()

    out.backward(torch.ones_like(out))
    conv1_grad_without_recompute = model.conv1.weight.grad.clone().detach().cpu()
    conv2_grad_without_recompute = model.conv2.weight.grad.clone().detach().cpu()

    # Expected memory saving:
    #   - relu1 & 2 saves a mask (1 byte per elem) of shape [100 * 3 * 224 * 224]
    #   - conv2 saves a float32 input of shape [100 * 3 * 224 * 224]
    expected_memory_saving = x.numel() * (4 * 1 * 1)
    actual_memory_saving = mem_without_recompute - mem_with_recompute

    # Considering noise factors, actual memory saving should be no less than
    # 90% of the expected memory saving
    assert expected_memory_saving * 0.9 <= actual_memory_saving


    assert torch.equal(conv1_grad_with_recompute, conv1_grad_without_recompute)
    assert torch.equal(conv2_grad_with_recompute, conv2_grad_without_recompute)

def test_matmul_bit_override():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = ModelWithMatMul2().to(device)
    dummy_input = (
        torch.randn(10, 3, 4, device=device),
        torch.randn(10, 5, 4, device=device),
    )

    quantsim_config = {
        "defaults": {
            "hw_version": 'V79',
            "ops": {"is_output_quantized": "True"},
            "params": {},
        },
        "params": {},
        "op_type": {
            "Relu": {"is_output_quantized": "False"},
        },
        "supergroups": [],
        "model_input": {},
        "model_output": {},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "quantsim_config.json")

        with open(config_path, "w") as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(
            model,
            dummy_input,
            quant_scheme=QuantScheme.post_training_tf,
            config_file=config_path,
            default_output_bw = 16,
            default_param_bw=4,
        )

    sim.compute_encodings(
        lambda sim_model, _: sim_model(*dummy_input), forward_pass_callback_args=None
    )
    set_matmul_second_input_producer_to_8bit_symmetric(sim)

    closest_output_quantizer_of_second_input = sim.model.act3.output_quantizers[0]
    assert closest_output_quantizer_of_second_input.bitwidth == 8
    assert closest_output_quantizer_of_second_input.symmetric
