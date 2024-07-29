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
import itertools

import pytest
import numpy as np
import torch
import torch.nn.functional as functional
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, RandomSampler
import deepspeed as ds
import tempfile
import json


from aimet_common import quantsim_config
from aimet_common.defs import QuantScheme
import aimet_torch.v2 as aimet
from aimet_torch.v2.quantization.affine.quantizer import QuantizeDequantize
from aimet_torch.v2.quantization.base.quantizer import QuantizerBase
from aimet_torch.v2.quantization import DequantizedTensor

from aimet_torch.v2.quantsim import QuantizationSimModel
from .models_.mnist_torch_model import Net
from .models_.test_models import TransposedConvModel


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


def get_all_subclasses(cls):
    def _get_all_subclasses(cls):
        for subcls in cls.__subclasses__():
            yield from _get_all_subclasses(subcls)
        yield cls
    return set(_get_all_subclasses(cls))


@pytest.fixture(autouse=True)
def teardown():
    """
    Custom teardown logic to bypass bugs in DeepSpeed.
    Deepspeed temporarily monkey-patches some attributes of PyTorch classes and subpackages
    such as <ModuleType>.__init__ or torch.nn.functional.<function> during ``with ds.zero.Init(...):``
    but fails to restore the original class definitions when exiting the context.

    This is an obvious bug that has to be fixed in deepspeed source code directly.
    For now, we add our custom teardown logic so this bug in deepspeed doesn't leave
    permanent side effect to the whole test session, leading other unrelated test cases to fail
    """
    orig_cls_defs = {
        subcls: subcls.__dict__.copy()
        for subcls in get_all_subclasses(torch.nn.Module)
    }

    orig_pkg_defs = {
        subpkg: subpkg.__dict__.copy()
        for subpkg in (torch, torch.nn, torch.nn.functional)
    }

    try:
        yield
    finally:
        # Restore original class/package definitions to bypass bugs in deepspeed
        for cls_or_pkg, orig_attrs in itertools.chain(orig_cls_defs.items(), orig_pkg_defs.items()):
            for attr_name in tuple(cls_or_pkg.__dict__.keys()):
                if attr_name == '__dict__':
                    continue
                if attr_name in orig_attrs:
                    setattr(cls_or_pkg, attr_name, orig_attrs[attr_name])
                else:
                    delattr(cls_or_pkg, attr_name)


@pytest.fixture
def init_process_group():
    LOCAL_RANK = os.getenv('LOCAL_RANK', None)

    try:
        # Create process group of size 2
        dist.init_process_group(backend='nccl',
                                store=dist.HashStore(),
                                world_size=1,
                                rank=0)
        os.environ['LOCAL_RANK'] = '0'
        yield dist.new_group(ranks=[0])
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if LOCAL_RANK is None:
            del os.environ['LOCAL_RANK']
        else:
            os.environ['LOCAL_RANK'] = LOCAL_RANK


@pytest.fixture(scope="session")
def unlabeled_data_loader():
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    dataset = MyDataset([torch.randn(1, 28, 28) for _ in range(10)])
    # TODO: (huzh) Change RandomSampler to DistributedSampler for testing with multiple GPUs
    sampler = RandomSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=1)


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
                "lr": 1e-2,
                "betas": [0.9, 0.999],
                "eps": 1e-7,
                "weight_decay": 1e-5
            }
        }
    }


@pytest.fixture
def per_channel_quantsim_config():
    with open(os.path.join(quantsim_config.__path__[0], 'default_config_per_channel.json')) as f:
        config = json.load(f)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, 'config_file.json'), 'w') as f:
            json.dump(config, f)
        yield os.path.join(tmp_dir, 'config_file.json')


@pytest.mark.cuda
def test_deepspeed_zero3_offload(unlabeled_data_loader,
                                 per_channel_quantsim_config,
                                 init_process_group,
                                 deepspeed_zero3_offload_config):
    """
    Given: Model pre-partitioned with deepspeed zero3 offload
    """
    with ds.zero.Init(config_dict_or_path=deepspeed_zero3_offload_config):
        # ds.zero.Init context pre-partitoins the pytorch models at instantiation time.
        # PyTorch modules instantiated under this context will only hold a partition
        # of their parameters
        model = Net().cuda()
        assert all(param.numel() == 0 for param in model.parameters())         # sanity check
        assert all(hasattr(param, 'ds_shape') for param in model.parameters()) # sanity check

    """
    When: Create quantsim with the model pre-partitioned model
    Then: Quantizers should be instantiated with correct shape
    """
    sim = QuantizationSimModel(model,
                               torch.randn(1, 1, 28, 28).cuda(),
                               default_param_bw=4,
                               config_file=per_channel_quantsim_config,
                               quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                               in_place=True)

    assert isinstance(sim.model.conv1.input_quantizers[0], QuantizeDequantize)
    assert isinstance(sim.model.conv1.param_quantizers['weight'], QuantizeDequantize)
    assert isinstance(sim.model.conv1.output_quantizers[0], QuantizeDequantize)
    assert isinstance(sim.model.maxpool1.output_quantizers[0], QuantizeDequantize)
    assert isinstance(sim.model.relu1.output_quantizers[0], QuantizeDequantize)
    assert isinstance(sim.model.conv2.param_quantizers['weight'], QuantizeDequantize)
    assert isinstance(sim.model.conv2.output_quantizers[0], QuantizeDequantize)
    assert isinstance(sim.model.maxpool2.output_quantizers[0], QuantizeDequantize)
    assert isinstance(sim.model.relu2.output_quantizers[0], QuantizeDequantize)
    assert isinstance(sim.model.fc1.param_quantizers['weight'], QuantizeDequantize)
    assert sim.model.fc1.output_quantizers[0] is None
    assert isinstance(sim.model.relu3.output_quantizers[0], QuantizeDequantize)
    assert isinstance(sim.model.fc2.param_quantizers['weight'], QuantizeDequantize)
    assert isinstance(sim.model.fc2.output_quantizers[0], QuantizeDequantize)
    assert isinstance(sim.model.log_softmax.output_quantizers[0], QuantizeDequantize)

    assert sim.model.conv1.param_quantizers['weight'].shape == (32, 1, 1, 1)
    assert sim.model.conv2.param_quantizers['weight'].shape == (64, 1, 1, 1)

    # NOTE: default per-channel quantsim config doesn't apply per-channel qtzn to nn.Linear
    assert sim.model.fc1.param_quantizers['weight'].shape == ()
    assert sim.model.fc2.param_quantizers['weight'].shape == ()

    assert sim.model.conv1.input_quantizers[0].shape ==\
           sim.model.conv1.output_quantizers[0].shape ==\
           sim.model.maxpool1.output_quantizers[0].shape ==\
           sim.model.relu1.output_quantizers[0].shape ==\
           sim.model.conv2.output_quantizers[0].shape ==\
           sim.model.maxpool2.output_quantizers[0].shape ==\
           sim.model.relu2.output_quantizers[0].shape ==\
           sim.model.relu3.output_quantizers[0].shape ==\
           sim.model.fc2.output_quantizers[0].shape ==\
           sim.model.log_softmax.output_quantizers[0].shape == ()

    """
    When: Initialize quantsim model with deepspeed zero3 offload
    Then:
      1) All parameters must be initialized with deepspeed zero3 parameter partitioning mechanism
      2) No quantizer encoding is not initialized yet
      3) get_{encoding, scale, offset, min, max} doesn't throw error but returns None
    """
    engine, ds_optimizer, *_ = ds.initialize(model=sim.model,
                                             model_parameters=sim.model.parameters(),
                                             config=deepspeed_zero3_offload_config,
                                             mpu=CustomMPU(init_process_group))
    assert all(hasattr(param, 'ds_shape') for param in model.parameters())

    for qtzr in sim.model.modules():
        if isinstance(qtzr, QuantizerBase):
            assert not qtzr.is_initialized()
            assert qtzr.get_encoding() is None
            assert qtzr.get_scale() is None
            assert qtzr.get_offset() is None
            assert qtzr.get_min() is None
            assert qtzr.get_max() is None

    """
    When: Compute encodings after deepspeed initialization
    Then:
      1) All quantizer encodings must be inititalized
      2) get_{encoding, scale, offset, min, max} doesn't throw error but returns empty tensor
    """
    with aimet.nn.compute_encodings(model):
        for data in itertools.islice(unlabeled_data_loader, 3):
            _ = sim.model(data.cuda())

    for qtzr in sim.model.modules():
        if isinstance(qtzr, QuantizerBase):
            assert qtzr.is_initialized()
            encoding = qtzr.get_encoding()
            assert encoding.scale.numel() == 0
            assert encoding.offset.numel() == 0
            assert encoding.min.numel() == 0
            assert encoding.max.numel() == 0
            assert qtzr.get_scale().numel() == 0
            assert qtzr.get_offset().numel() == 0
            assert qtzr.get_min().numel() == 0
            assert qtzr.get_max().numel() == 0

    """
    When: Run training loop
    Then: All trainable parameters must be udpated by training
    """
    with ds.runtime.zero.GatheredParameters(sim.model.parameters()):
        params_before = {
            name: param.clone().detach() for name, param in sim.model.named_parameters()
        }

    target = torch.ones((1, 10)).float().cuda()

    for _, data in enumerate(unlabeled_data_loader):
        output = sim.model(data.cuda())
        assert isinstance(output, DequantizedTensor)
        assert output.encoding.scale.numel() == 1
        assert output.encoding.offset.numel() == 1

        loss = functional.mse_loss(output, target)
        engine.backward(loss)
        ds_optimizer.step()

    with ds.runtime.zero.GatheredParameters(sim.model.parameters()):
        params_after = {
            name: param.clone().detach() for name, param in sim.model.named_parameters()
        }

    assert params_before.keys() == params_after.keys()
    for param_name in params_before:
        before = params_before[param_name]
        after = params_after[param_name]
        assert not torch.equal(before, after)


@pytest.mark.cuda
def test_conv_transpose(per_channel_quantsim_config,
                        init_process_group,
                        deepspeed_zero3_offload_config):
    """
    Given: Model containing ConvTransposeNd, pre-partitioned with deepspeed zero3 offload
    """
    with ds.zero.Init(config_dict_or_path=deepspeed_zero3_offload_config):
        # ds.zero.Init context pre-partitoins the pytorch models at instantiation time.
        # PyTorch modules instantiated under this context will only hold a partition
        # of their parameters
        model = TransposedConvModel().cuda()
        assert all(param.numel() == 0 for param in model.parameters())         # sanity check
        assert all(hasattr(param, 'ds_shape') for param in model.parameters()) # sanity check

    """
    When: Create quantsim with the model pre-partitioned model
    Then: Quantizers should be instantiated with correct shape
    """
    sim = QuantizationSimModel(model,
                               torch.randn(1, 10, 28, 28).cuda(),
                               default_param_bw=4,
                               config_file=per_channel_quantsim_config,
                               quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                               in_place=True)

    assert sim.model.conv1.param_quantizers['weight'].shape == (1, 10, 1, 1)
    assert sim.model.conv2.param_quantizers['weight'].shape == (1, 10, 1, 1)
