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
from typing import Dict
import os
from collections import defaultdict
import torch.nn as nn
import torch
from safetensors.torch import save_file
from safetensors import safe_open

# pylint: disable=import-error
# pylint: disable=no-name-in-module
from peft.tuners.lora.layer import LoraLayer as PeftLoraLayer

from aimet_torch.utils import replace_modules_of_type1_using_constructor
from aimet_torch.elementwise_ops import Add
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import BaseQuantizationMixin


class LoraLayer(torch.nn.Module):
    """
    Quantizable lora layer
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, lora_layer: PeftLoraLayer):
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
        self.index_to_adapter_name = {}
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
        for adapter_name in self.adapter_name_to_index:
            self.index_to_adapter_name[self.adapter_name_to_index[adapter_name]] = adapter_name

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
    replace_modules_of_type1_using_constructor(model, PeftLoraLayer, LoraLayer)


class AdapterMetaData:
    """
    Tracks meta data for lora layers. Tracks names of lora_a & b as well as alpha values
    """
    def __init__(self):
        self.lora_A = []
        self.lora_B = []
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
            for index, lora_layer in enumerate(module.lora_A):
                adapter_name_to_meta_data[module.index_to_adapter_name[index]].lora_A.append(lora_layer.name)
            for index, lora_layer in enumerate(module.lora_B):
                adapter_name_to_meta_data[module.index_to_adapter_name[index]].lora_B.append(lora_layer.name)
            for lora_adapter_name in module.lora_alpha:
                adapter_name_to_meta_data[lora_adapter_name].alpha = module.lora_alpha[lora_adapter_name]
    return adapter_name_to_meta_data


class PeftQuantUtils:
    """
    Utilities for quantizing peft model
    """
    def __init__(self, model, adapater_name_to_meta_data: Dict, prepared_model: bool = False):
        """
        Init for Peft utilities for quantization

        :param model: Torch model
        :param adapater_name_to_meta_data: Dict showing adapter name to meta data
        :param prepared_model: Bool. If true, then user has passed a prepared model as the model
        """
        self.adapter_name_to_meta_data = adapater_name_to_meta_data
        self.lora_layers = self._get_lora_layers()
        self.pt_name_to_onnx_name, self.onnx_name_to_pt_name = None, None
        self.pt_to_lora_name = dict.fromkeys(self.lora_layers, '')
        if prepared_model:
            self.pt_name_to_onnx_name, self.onnx_name_to_pt_name = self._get_pytorch_name_to_onnx_name(model)
            self.lora_to_pt_name, self.pt_to_lora_name = self._get_lora_name_to_pytorch_name()

    @staticmethod
    def _get_pytorch_name_to_onnx_name(model: torch.nn.Module):
        """
        Gets onnx names to pytorch names mapping and vice versa

        :param model: PT model
        """
        pt_name_to_onnx_name = {}
        onnx_name_to_pt_name = {}
        for name, module in model.named_modules():
            for pytorch_name in model.name_to_module_dict:
                pytorch_module = model.name_to_module_dict[pytorch_name][0]
                if pytorch_module == module:
                    pt_name_to_onnx_name[pytorch_name] = name
                    onnx_name_to_pt_name[name] = pytorch_name
        return pt_name_to_onnx_name, onnx_name_to_pt_name

    def _get_lora_name_to_pytorch_name(self):
        """
        Gets most similar pytorch name for every lora name
        """
        lora_to_pytorch_name = {}
        pytorch_to_lora_name = {}
        for pt_name in self.pt_name_to_onnx_name:
            for lora_name in self.lora_layers:
                if pt_name in lora_name:
                    lora_to_pytorch_name[lora_name] = pt_name
                    pytorch_to_lora_name[pt_name] = lora_name
        return lora_to_pytorch_name, pytorch_to_lora_name

    def _get_lora_layers(self) -> set:
        """
        Gets all lora layers
        """
        lora_layers = set()
        for adapter_name in self.adapter_name_to_meta_data:
            for lora_module in self.adapter_name_to_meta_data[adapter_name].lora_A:
                lora_layers.add(lora_module)
            for lora_module in self.adapter_name_to_meta_data[adapter_name].lora_B:
                lora_layers.add(lora_module)
        return lora_layers

    @staticmethod
    def _freeze_quantizer(quantizer):
        """
        Disables compute encodings and gradient update for a quantizer

        :param quantizer: Param, output or Input quantizer
        """
        # pylint:disable = protected-access
        quantizer._allow_overwrite = False
        quantizer.requires_grad_(False)

    def freeze_base_model_param_quantizers(self, sim: QuantizationSimModel):
        """
        Freeze parameter quantizers of base model

        :param sim: QuantSim model
        """
        for module_name, module in sim.model.named_modules():
            if self.onnx_name_to_pt_name and module_name in self.onnx_name_to_pt_name:
                module_name = self.onnx_name_to_pt_name[module_name]
            if isinstance(module, BaseQuantizationMixin) and module_name not in self.pt_to_lora_name:
                for _, param_quantizer in module.param_quantizers.items():
                    if param_quantizer:
                        self._freeze_quantizer(param_quantizer)

    def freeze_base_model_activation_quantizers(self, sim: QuantizationSimModel):
        """
        Freeze activation quantizers of base model

        :param sim: QuantSim model
        """
        for module_name, module in sim.model.named_modules():
            if self.onnx_name_to_pt_name and module_name in self.onnx_name_to_pt_name:
                module_name = self.onnx_name_to_pt_name[module_name]
            if isinstance(module, BaseQuantizationMixin) and module_name not in self.pt_to_lora_name:
                for input_quantizer, output_quantizer in zip(module.input_quantizers, module.output_quantizers):
                    if input_quantizer:
                        self._freeze_quantizer(input_quantizer)
                    if output_quantizer:
                        self._freeze_quantizer(output_quantizer)

    def freeze_base_model(self, sim: QuantizationSimModel):
        """
        Freeze entire base model

        :param sim: QuantSim model
        """
        self.freeze_base_model_activation_quantizers(sim)
        self.freeze_base_model_param_quantizers(sim)

    def set_bitwidth_for_lora_adapters(self, sim: QuantizationSimModel,
                                       output_bw: int, param_bw: int):
        """
        Sets output and param bitwidth for all Lora adapters added to the model

        :param sim: QuantSim model
        :param output_bw: Output BW
        :param param_bw: Parameter BW
        """
        for module_name, module in sim.model.named_modules():
            if self.onnx_name_to_pt_name and module_name in self.onnx_name_to_pt_name:
                module_name = self.onnx_name_to_pt_name[module_name]
            if isinstance(module, BaseQuantizationMixin) and module_name in self.pt_to_lora_name:
                self._set_bitwidth_for_module(module, output_bw, param_bw)

    def set_bitwidth_for_given_lora_adapters(self, sim: QuantizationSimModel,
                                             adapter_name: str, output_bw: int, param_bw: int):
        """
        Sets output and param bitwidth for specific adapters. The specific adapter is specified using adapter name

        :param sim: QuantSim model
        :param adapter_name: Name of the adapter for which the Bitwidth need to be set
        :param output_bw: Output BW
        :param param_bw: Parameter BW
        """
        for _, module in sim.model.named_modules():
            if isinstance(module, LoraLayer):
                if adapter_name in module.adapter_name_to_index:
                    index = module.adapter_name_to_index[adapter_name]
                    lora_a = module.lora_A[index]
                    lora_b = module.lora_B[index]
                    self._set_bitwidth_for_module(lora_a, output_bw, param_bw)
                    self._set_bitwidth_for_module(lora_b, output_bw, param_bw)

    @staticmethod
    def _set_bitwidth_for_module(module: BaseQuantizationMixin, output_bw: int, param_bw: int):
        """
        Sets bitwidth for a QcQuantizeWrapper module

        :param module: QcQuantize wrapper module
        :param output_bw: Output BW
        :param param_bw: Parameter BW
        """
        for output_quantizer in module.output_quantizers:
            output_quantizer.bitwidth = output_bw
        for _, param_quantizer in module.param_quantizers.items():
            param_quantizer.bitwidth = param_bw

    def export_adapter_weights(self, sim: QuantizationSimModel, path: str, filename_prefix: str):
        """
        Exports adapter weights to safetensor format

        :param sim: QuantSim model
        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        """
        tensors = {}

        for module_name, module in sim.model.named_modules():
            org_name = module_name
            if self.onnx_name_to_pt_name and module_name in self.onnx_name_to_pt_name:
                module_name = 'base_model.' + self.onnx_name_to_pt_name[module_name]
            if module_name in self.lora_layers:
                for param_name, param in module.named_parameters():
                    if param_name in ['weight', 'bias']:
                        tensor_name = org_name + '.' + param_name
                        tensors[tensor_name] = param
        filename_prefix = filename_prefix + '.safetensor'
        model_params_path = os.path.join(path, filename_prefix)
        save_file(tensors, model_params_path)

    def enable_adapter_and_load_weights(self, sim: QuantizationSimModel, adapter_weights_path):
        """
        Enables adapter effect on base model by loading weights to model

        :param sim: QuantSim model
        :param adapter_weights_path: Path to adapter weights
        """
        tensors = {}
        with safe_open(adapter_weights_path, framework="pt", device=0) as f:
            for key in f.keys():
                tensor_name = key
                if self.onnx_name_to_pt_name:
                    temp_key = key[0:key.find('.weight')]
                    tensor_name = self.pt_name_to_onnx_name[self.lora_to_pt_name[temp_key]] + '.weight'
                tensors[tensor_name] = f.get_tensor(key)

        sim.model.load_state_dict(tensors, strict=False)
