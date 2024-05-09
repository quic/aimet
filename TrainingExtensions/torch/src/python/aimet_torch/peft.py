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

""" Implementation for handling LoRA adapters added using PEFT """
from collections import defaultdict
import torch.nn as nn
import torch

# pylint: disable=import-error
# pylint: disable=no-name-in-module
from peft.tuners.lora.layer import LoraLayer

from aimet_torch.utils import replace_modules_of_type1_using_constructor
from aimet_torch.elementwise_ops import Add


class QcLoraLayer(torch.nn.Module):
    """
    Quantizable lora layer
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, lora_layer: LoraLayer):
        """
        :param lora_layer: Lora layer we want to replace
        """
        super().__init__()
        self.base_layer = lora_layer.base_layer
        self.r = lora_layer.r
        self.lora_alpha = lora_layer.lora_alpha
        self.scaling = lora_layer.scaling
        self.lora_dropout = nn.ModuleList({})
        self.adapter_name_to_index = {}
        self.lora_A = nn.ModuleList([])
        self.lora_B = nn.ModuleList([])
        self.active_adapters = {}
        self._swap_module_dict_with_list(lora_layer)
        self.in_features = lora_layer.in_features
        self.out_features = lora_layer.out_features
        self.add_lora_to_res = Add()

    def _swap_module_dict_with_list(self, lora_layer):
        for index, adapter_name in enumerate(lora_layer.lora_A):
            self.lora_A.append(lora_layer.lora_A[adapter_name])
            self.lora_B.append(lora_layer.lora_B[adapter_name])
            self.lora_dropout.append(lora_layer.lora_dropout[adapter_name])
            self.adapter_name_to_index[adapter_name] = index
            if adapter_name in lora_layer.active_adapter:
                self.active_adapters[adapter_name] = True
            else:
                self.active_adapters[adapter_name] = False

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """ Forward pass for replaced layer"""
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.adapter_name_to_index:
                continue
            lora_A = self.lora_A[self.adapter_name_to_index[active_adapter]]
            lora_B = self.lora_B[self.adapter_name_to_index[active_adapter]]
            dropout = self.lora_dropout[self.adapter_name_to_index[active_adapter]]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)

            result = self.add_lora_to_res(result, lora_B(lora_A(dropout(x)) * scaling))

        result = result.to(torch_result_dtype)
        return result


def replace_lora_layers_with_quantizable_layers(model: torch.nn.Module):
    """
    Utility to replace lora layers with Quantizable Lora layers

    :param model: PEFT model
    """
    replace_modules_of_type1_using_constructor(model, LoraLayer, QcLoraLayer)


class AdapterMetaData:
    """
    Tracks meta data for lora layers. Tracks names of lora_a & b as well as alpha values
    """
    def __init__(self):
        self.lora_a = []
        self.lora_b = []
        self.alpha = None


def track_lora_meta_data(model: torch.nn.Module):
    """
    Utility to track adapter names and corresponding metadata

    :param model: PEFT model
    """
    for name, module in model.named_modules():
        module.name = name
    adapter_name_to_meta_data = defaultdict(AdapterMetaData)
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            for lora_adapter_name in module.lora_A:
                adapter_name_to_meta_data[lora_adapter_name].lora_a.append(module.lora_A[lora_adapter_name].name)
            for lora_adapter_name in module.lora_B:
                adapter_name_to_meta_data[lora_adapter_name].lora_b.append(module.lora_B[lora_adapter_name].name)
            for lora_adapter_name in module.lora_alpha:
                adapter_name_to_meta_data[lora_adapter_name].alpha = module.lora_alpha[lora_adapter_name]
    return adapter_name_to_meta_data
