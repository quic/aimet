# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme
from aimet_torch.bn_reestimation import reestimate_bn_stats, _get_active_bn_modules

import torch
from torch.utils.data import DataLoader, Dataset


torch.manual_seed(1350)

class Model(torch.nn.Module):
    """
    Model
    """

    def __init__(self):
        super(Model, self).__init__()
        self._bn = torch.nn.BatchNorm2d(3)

    def forward(self, x: torch.Tensor):
        return self._bn(x)


@pytest.fixture
def fp32_model():
    return Model().cpu()


@pytest.fixture
def quantsim_model(fp32_model, dummy_input):
    sim = QuantizationSimModel(fp32_model,
                               dummy_input,
                               quant_scheme=QuantScheme.training_range_learning_with_tf_init)
    sim.compute_encodings(lambda model, _: model(dummy_input), None)
    return sim.model


@pytest.fixture(scope="session")
def dummy_input():
    return torch.randn((1, 3, 8, 8))


@pytest.fixture(scope="session")
def data_loader(dummy_input):
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    dataset = MyDataset([torch.randn_like(dummy_input[0]) for _ in range(1)])
    return DataLoader(dataset)


def test_reestimation_with_fp32_model(fp32_model, data_loader):
    expected_mean = [torch.mean(data, dim=(0,2,3)) for data in data_loader]
    expected_mean = sum(expected_mean) / len(data_loader)
    expected_var = [torch.var(data, dim=(0,2,3)) for data in data_loader]
    expected_var = sum(expected_var) / len(data_loader)
    _test_reestimation(fp32_model, data_loader, expected_mean, expected_var)


def test_reestimation_with_quantsim_model(quantsim_model, data_loader):
    def quantize_input(data):
        input_quantizer = quantsim_model._bn.input_quantizer
        encoding = input_quantizer.encoding
        encoding_min = torch.tensor([encoding.min])
        encoding_max = torch.tensor([encoding.max])
        return input_quantizer.quantize_dequantize(data, encoding_min, encoding_max)

    expected_mean = [torch.mean(quantize_input(data), dim=(0,2,3)) for data in data_loader]
    expected_mean = sum(expected_mean) / len(data_loader)
    expected_var = [torch.var(quantize_input(data), dim=(0,2,3)) for data in data_loader]
    expected_var = sum(expected_var) / len(data_loader)
    _test_reestimation(quantsim_model, data_loader, expected_mean, expected_var)


def _test_reestimation(model, data_loader, expected_mean, expected_var):
    old_params = list(model.named_parameters())

    with torch.no_grad():
        for data in data_loader:
            model(data)

    mean_orig, var_orig = [
        ( bn.running_mean.clone().detach(), bn.running_var.clone().detach() )
        for bn in _get_active_bn_modules(model)
    ][0]

    with reestimate_bn_stats(model, data_loader):
        for bn in _get_active_bn_modules(model):
            assert bn.momentum != 1.0

        mean_reestimated, var_reestimated = [
            ( bn.running_mean.clone().detach(), bn.running_var.clone().detach() )
            for bn in _get_active_bn_modules(model)
        ][0]

        assert torch.equal(mean_reestimated, expected_mean)
        assert torch.equal(var_reestimated, expected_var)

    new_params = list(model.named_parameters())

    # All the model parameters should remain the same
    assert old_params == new_params

    mean_restored, var_restored = [
        ( bn.running_mean.clone().detach(), bn.running_var.clone().detach() )
        for bn in _get_active_bn_modules(model)
    ][0]

    assert not torch.equal(mean_orig, mean_reestimated)
    assert torch.equal(mean_orig, mean_restored)

    assert not torch.equal(var_orig, var_reestimated)
    assert torch.equal(var_orig, var_restored)
