# /usr/bin/env python
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

import random
import os

import pytest
import numpy as np
import torch
import torch.nn.functional as functional
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, RandomSampler
import deepspeed as ds


from aimet_common.defs import QuantScheme

from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn.true_quant import QuantizedLinear
from aimet_torch.v2.deepspeed_utils import transfer_quant_params
from .models_.mnist_torch_model import Net

from transformers.utils import ContextManagers
from transformers.deepspeed import HfDeepSpeedConfig


class CustomMPU:
    def __init__(self, group):
        self.group = group

    def get_model_parallel_group(self):
        return self.group

    def get_data_parallel_group(self):
        return dist.group.WORLD

    def get_model_parallel_rank(self):
        return dist.get_rank(self.group)

    def get_model_parallel_world_size(self):
        return dist.get_world_size(self.group)

    def get_data_parallel_world_size(self):
        return dist.get_world_size(dist.group.WORLD)


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(999)
    torch.manual_seed(0)
    np.random.seed(0)

@pytest.fixture
def init_process_group(scope='function'):

    # https://stackoverflow.com/a/63851681/9201239
    def get_all_subclasses(cls):
        subclass_list = []

        def recurse(cl):
            for subclass in cl.__subclasses__():
                subclass_list.append(subclass)
                recurse(subclass)

        recurse(cls)

        return set(subclass_list)

    LOCAL_RANK = os.getenv('LOCAL_RANK', None)
    # Deepspeed can't unpatch below hooks
    linear_bk = functional.linear
    # LinearFunctionForZeroStage3 will make F.linear-->torch.addmm or torch.matmul, it's a workaround to resolve it
    QuantizedLinear._builtin_torch_fn = torch.addmm
    subclass_lst = get_all_subclasses(torch.nn.modules.module.Module)
    for subclass in subclass_lst:
        subclass.old_init = subclass.__init__
        subclass.old_apply_hook = subclass._apply

    try:
        # Create process group of size 2
        dist.init_process_group(backend='nccl',
                                store=dist.HashStore(),
                                world_size=1,
                                rank=0)
        os.environ['LOCAL_RANK'] = '0'
        yield dist.new_group(ranks=[0])
    finally:
        # Restore init function to bypass DeepSpeed bug
        for subclass in subclass_lst:
            subclass.__init__ = subclass.old_init
            subclass._apply = subclass.old_apply_hook
        torch.nn.functional.linear = linear_bk
        QuantizedLinear._builtin_torch_fn = linear_bk
        if dist.is_initialized():
            dist.destroy_process_group()
        if LOCAL_RANK is not None:
            os.environ['LOCAL_RANK'] = LOCAL_RANK

@pytest.fixture(scope="session")
def dummy_input():
    return torch.randn((1, 1, 28, 28))

@pytest.fixture(scope="session")
def unlabeled_data_loader(dummy_input):
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    dataset = MyDataset([dummy_input[0, :] for _ in range(32)])
    # TODO: (huzh) Change RandomSampler to DistributedSampler for testing with multiple GPUs
    sampler = RandomSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=1)

def calibrate(model, inputs):

    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    model.eval()
    with torch.no_grad():
        model(*inputs)

@pytest.fixture
def deepspeed_zero3_offload_config():
    return {
        "zero_optimization": 
        {
            "stage": 3,
            "offload_param": {"device": "cpu"},
            "offload_optimizer": {"device": "cpu"}
        },
        "train_batch_size": 1,
        "optimizer":
        {
            "type": "AdamW",
            "params": {
                "lr": 1e-6,
                "betas": [0.9, 0.999],
                "eps": 1e-7,
                "weight_decay": 1e-5
            }
        }
    }

@pytest.mark.cuda
@pytest.mark.parametrize("qscheme", [QuantScheme.post_training_tf])
def test_is_initialized_with_deepspeed_zero3_offload(unlabeled_data_loader, init_process_group, deepspeed_zero3_offload_config, qscheme):

    """
    When: Offload parameters into CPU using Zero-offload
    Then: quantizer.is_initialized() flag should be preserved after pertitioning
    """
    # Initialize DeepSpeed config
    ds_config = HfDeepSpeedConfig(deepspeed_zero3_offload_config)
    # Instantiate model
    init_contexts = []

    init_contexts = [ds.zero.Init(config_dict_or_path=deepspeed_zero3_offload_config)] + init_contexts

    torch.manual_seed(0)
    # TODO: (huzh) Replace with AIMET internal context manager
    with ContextManagers(init_contexts):
        # Set model to train mode during training, as it is difficult to change after QSim initialization
        model = Net().train().cuda()
    # Ensure parameters are offloaded to CPU
    assert model.fc1.weight.numel() == 0
    assert model.fc2.weight.numel() == 0
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    engine, ds_optimizer, *_ = ds.initialize(model=model, model_parameters=model_parameters, config=deepspeed_zero3_offload_config, mpu=CustomMPU(init_process_group))

    dummy_input = torch.randn(1, 1, 28, 28).cuda()
    sim = QuantizationSimModel(model, dummy_input, default_param_bw=4, quant_scheme=qscheme, in_place=True)
    sim.model.requires_grad_(True)
    # Compute encodings
    sim.compute_encodings(calibrate, dummy_input)

    # Ensure AIMET quantization parameters have not been initialized by Zero-offload
    assert sim.model.fc1.param_quantizers['weight'].min.numel() != 0
    assert sim.model.fc1.param_quantizers['weight'].max.numel() != 0
    assert sim.model.fc1.param_quantizers['weight'].min.requires_grad
    assert sim.model.fc1.param_quantizers['weight'].max.requires_grad
    assert sim.model.fc1.output_quantizers[0].is_initialized()
    assert sim.model.fc2.output_quantizers[0].is_initialized()

    # Train model and ensure it works
    enc_before = sim.model.fc1.param_quantizers['weight'].get_encoding()

    device = next(model.parameters()).device

    target = torch.ones((1,10)).float().to(device)
    model.train()
    with transfer_quant_params(sim.model, requires_grad=True) as model_parameters:
        optimizer = torch.optim.AdamW(model_parameters, lr=0.01)

        for _, data in enumerate(unlabeled_data_loader):
            data = data.to(device)
            output = model(data)
            loss = functional.mse_loss(output, target)
            engine.backward(loss)
            ds_optimizer.step()
            optimizer.step()
            optimizer.zero_grad()

    enc_after = sim.model.fc1.param_quantizers['weight'].get_encoding()
    assert enc_before.min != enc_after.min
