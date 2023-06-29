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
import torch
from peft.tuners.lora.layer import LoraLayer
from peft import LoraConfig, get_peft_model
from aimet_torch.peft import replace_lora_layers_with_quantizable_layers, track_lora_meta_data, QcLoraLayer

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        return x


def one_adapter_model():
    model = DummyModel()
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=4,
        bias="none",
        target_modules=["linear"],
    )

    model = get_peft_model(model, lora_config)
    return model

def two_adapter_model():
    model = DummyModel()
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=4,
        bias="none",
        target_modules=["linear"],
    )

    model = get_peft_model(model, lora_config)
    model.add_adapter("default_new", lora_config)
    return model

class TestLoraAdapterPeft:
    def test_replace_adapter(self):
        model = one_adapter_model()
        count_lora_layer = 0
        for _, module in model.named_modules():
            if isinstance(module, LoraLayer):
                count_lora_layer += 1
        replace_lora_layers_with_quantizable_layers(model)
        count_qc_lora_layer = 0
        new_count_lora_layer = 0
        for _, module in model.named_modules():
            if isinstance(module, QcLoraLayer):
                count_qc_lora_layer += 1
            if isinstance(module, LoraLayer):
                new_count_lora_layer += 1
        assert new_count_lora_layer == 0
        assert count_qc_lora_layer == count_lora_layer

    def test_track_adapter_meta_data(self):
        model = two_adapter_model()
        meta_data = track_lora_meta_data(model)
        assert len(meta_data) == 2
        assert 'default' in meta_data
        assert 'default_new' in meta_data
        assert meta_data['default'].alpha == 16
        assert meta_data['default_new'].lora_b == ['base_model.model.linear.lora_B.default_new']


