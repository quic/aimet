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
import os
import pytest

import tempfile
import torch
from safetensors.torch import save_file
from safetensors import safe_open
import functools
import copy

from peft.tuners.lora.layer import LoraLayer as PeftLoraLayer
from peft import LoraConfig, get_peft_model
from aimet_torch.peft import replace_lora_layers_with_quantizable_layers, track_lora_meta_data, LoraLayer, \
    PeftQuantUtils
from aimet_torch.v2.quantsim import QuantizationSimModel

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def replace_linears_with_convs(model: torch.nn.Module,
                               linear_types= torch.nn.Linear) -> torch.nn.Module:
    model_conv = copy.deepcopy(model)

    for name, module in model_conv.named_modules():
        if isinstance(module, linear_types):
            conv_layer = ConvInplaceLinear(module)
            rsetattr(model_conv, name, conv_layer)

    return model_conv

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


class ConvInplaceLinear(torch.nn.Module):
    """ Convolution module that replaces a Linear layer inplace"""
    def __init__(self, linear):
        super(ConvInplaceLinear, self).__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.conv2d = torch.nn.Conv2d(linear.in_features, linear.out_features, 1, bias=True if linear.bias is not None else False)
        self.conv2d.weight.data.copy_(linear.weight.data[:, :, None, None])
        if linear.bias is not None:
            self.conv2d.bias.data.copy_(linear.bias.data)
        self.conv2d.to(linear.weight.data.device)

    def __getattr__(self, attr):
        conv2d = self._modules['conv2d']
        if attr == 'conv2d':
            return conv2d
        return getattr(conv2d, attr)

    def forward(self, x: torch.Tensor, scale: float = 1.0):
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0).unsqueeze(-1).permute(0, 2, 3, 1) # (emb_dim, C) -> (1, C, 1, emb_dim)
        elif ndim == 3:
            x = x.unsqueeze(-1).permute(0, 2, 3, 1) # (B, emb_dim, C) -> (B, C, 1, emb_dim)
        elif ndim == 4:
            x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        else:
            raise NotImplementedError(f"ConvInplaceLinear could not handle input with shape {x.shape}")

        x = self.conv2d(x)

        if ndim == 2:
            return x.permute(0, 3, 1, 2).squeeze(-1).squeeze(0) # (1, C, 1, emb_dim) -> # (emb_dim, C)
        elif ndim == 3:
             return x.permute(0, 3, 1, 2).squeeze(-1) # (1, C, 1, emb_dim) -> # (B, emb_dim, C)
        elif ndim == 4:
            x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        return x


class TestLoraAdapterPeft:
    def test_replace_adapter(self):
        model = one_adapter_model()
        count_lora_layer = 0
        for _, module in model.named_modules():
            if isinstance(module, PeftLoraLayer):
                count_lora_layer += 1
        replace_lora_layers_with_quantizable_layers(model)

        count_qc_lora_layer = 0
        new_count_lora_layer = 0
        for _, module in model.named_modules():
            if isinstance(module, LoraLayer):
                count_qc_lora_layer += 1
            if isinstance(module, PeftLoraLayer):
                new_count_lora_layer += 1
        assert new_count_lora_layer == 0
        assert count_qc_lora_layer == count_lora_layer

    def test_add_quantizer_and_set_range_for_scale(self):
        model = one_adapter_model()
        replace_lora_layers_with_quantizable_layers(model)
        dummy_inputs = torch.randn(10, 10)

        meta_path = './tmp'
        if not os.path.exists(meta_path):
            os.mkdir(meta_path)

        meta_data = track_lora_meta_data(model, './tmp', 'meta_data')

        peft_utils = PeftQuantUtils(meta_data)

        sim = QuantizationSimModel(model, dummy_input=dummy_inputs)

        peft_utils.quantize_lora_scale_with_fixed_range(sim, 0.0, 1.0)
        def forward_pass(model, forward_pass_callback=None):
            return model(dummy_inputs)
        sim.compute_encodings(forward_pass, None)
        for name, module in sim.model.named_modules():
            if 'mul_scale' in name and hasattr(module, 'input_quantizers'):
                assert module.input_quantizers[1].min == 0.0
                assert module.input_quantizers[1].max == 1.0

    def test_track_adapter_meta_data(self):
        model = two_adapter_model()
        replace_lora_layers_with_quantizable_layers(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_data = track_lora_meta_data(model, tmpdir, 'meta_data')
        assert len(meta_data) == 2
        assert 'default' in meta_data
        assert 'default_new' in meta_data
        assert meta_data['default'].alpha == 16
        assert meta_data['default_new'].lora_B == ['base_model.model.linear.lora_B.1']

    def test_freeze_base_model_params_and_activations(self):
        model = two_adapter_model()
        replace_lora_layers_with_quantizable_layers(model)

        dummy_inputs = torch.randn(10, 10)

        def forward_pass(model, forward_pass_callback=None):
            return model(dummy_inputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_data = track_lora_meta_data(model, tmpdir, 'meta_data')

        peft_utils = PeftQuantUtils(meta_data)

        sim = QuantizationSimModel(model, dummy_input= dummy_inputs)
        sim.compute_encodings(forward_pass, forward_pass_callback_args=None)

        qc_lora = sim.model.base_model.model.linear

        assert not _is_frozen(qc_lora.base_layer.param_quantizers['weight'])

        peft_utils.freeze_base_model_param_quantizers(sim)

        assert _is_frozen(qc_lora.base_layer.param_quantizers['weight'])
        assert not _is_frozen(qc_lora.lora_A[0].param_quantizers['weight'])
        assert not _is_frozen(qc_lora.lora_A[1].param_quantizers['weight'])
        assert not _is_frozen(qc_lora.lora_B[0].param_quantizers['weight'])
        assert not _is_frozen(qc_lora.lora_B[1].param_quantizers['weight'])

        assert not _is_frozen(qc_lora.base_layer.output_quantizers[0])

        peft_utils.freeze_base_model_activation_quantizers(sim)

        assert _is_frozen(qc_lora.base_layer.output_quantizers[0])
        assert not _is_frozen(qc_lora.lora_A[0].output_quantizers[0])
        assert not _is_frozen(qc_lora.lora_B[1].output_quantizers[0])

    def test_set_bitwidth_for_lora_adapters(self):
        model = two_adapter_model()
        replace_lora_layers_with_quantizable_layers(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_data = track_lora_meta_data(model, tmpdir, 'meta_data')
        dummy_inputs = torch.randn(10, 10)

        peft_utils = PeftQuantUtils(meta_data)

        sim = QuantizationSimModel(model, dummy_input=dummy_inputs)

        qc_lora = sim.model.base_model.model.linear

        peft_utils.set_bitwidth_for_lora_adapters(sim, output_bw=4, param_bw=4)
        assert qc_lora.base_layer.output_quantizers[0].bitwidth == 8
        assert qc_lora.lora_A[0].param_quantizers['weight'].bitwidth == 4
        assert qc_lora.lora_A[1].param_quantizers['weight'].bitwidth == 4
        assert qc_lora.lora_B[1].output_quantizers[0].bitwidth == 4

    @pytest.mark.cuda
    def test_enable_and_load_weights_adapter(self):
        model = one_adapter_model()
        replace_lora_layers_with_quantizable_layers(model)
        meta_data = track_lora_meta_data(model, './', 'meta_data')
        dummy_inputs = torch.randn(10, 10)

        peft_utils = PeftQuantUtils(meta_data)

        sim = QuantizationSimModel(model, dummy_input=dummy_inputs)
        qc_lora = sim.model.base_model.model.linear
        assert torch.all(qc_lora.lora_B[0].weight == torch.zeros((10, 4)))

        with tempfile.TemporaryDirectory() as tmpdir:
            tensors = {'base_model.model.linear.lora_A.0.weight': torch.randn((4, 10)),
                       'base_model.model.linear.lora_B.0.weight': torch.randn((10, 4))}
            path = os.path.join(tmpdir, 'weight.safetensor')
            save_file(tensors, path)
            peft_utils.enable_adapter_and_load_weights(sim, path)
            assert torch.all(qc_lora.lora_B[0].weight == tensors['base_model.model.linear.lora_B.0.weight'])

    def test_lora_layer_generator(self):
        model = two_adapter_model()
        replace_lora_layers_with_quantizable_layers(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_data = track_lora_meta_data(model, tmpdir, 'meta_data')
        dummy_inputs = torch.randn(10, 10)

        peft_utils = PeftQuantUtils(meta_data)

        sim = QuantizationSimModel(model, dummy_input=dummy_inputs)

        count = 0
        for name, layer in peft_utils.get_quantized_lora_layer(sim):
            count += 1
            assert 'lora' in name

        assert count == 4

    def test_export_encodings_multiple_times(self):
        model = one_adapter_model()

        replace_lora_layers_with_quantizable_layers(model)
        dummy_inputs = torch.randn(10, 10)

        meta_data = track_lora_meta_data(model, './', 'meta_data')

        peft_utils = PeftQuantUtils(meta_data)
        sim = QuantizationSimModel(model, dummy_input=dummy_inputs)

        dummy_inputs = torch.randn(10, 10)

        def forward_pass(model, forward_pass_callback=None):
            return model(dummy_inputs)

        sim.compute_encodings(forward_pass, forward_pass_callback_args=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            sim.export(tmpdir, 'model', dummy_input=dummy_inputs, export_model=True, filename_prefix_encodings='encodings')
            sim.export(tmpdir, 'model', dummy_input=dummy_inputs, export_model=False, filename_prefix_encodings='encodings_2')
            model_path = os.path.join(tmpdir, 'model.onnx')
            peft_utils.export_adapter_weights(sim, tmpdir, 'weight', model_path)
            assert os.path.exists(os.path.join(tmpdir, 'encodings.encodings'))
            assert os.path.exists(os.path.join(tmpdir, 'encodings_2.encodings'))

            tensor_name = []
            with safe_open(os.path.join(tmpdir, 'weight.safetensor'), framework="pt", device=0) as f:
                for key in f.keys():
                    tensor_name.append(key)
            assert len(tensor_name) == 2
            tensors = ['base_model.model.linear.lora_A.0.weight',
                       'base_model.model.linear.lora_B.0.weight']
            assert sorted(tensor_name) == sorted(tensors)

def _is_frozen(quantizer):
    return quantizer._allow_overwrite == False and\
           quantizer.min.requires_grad == False and\
           quantizer.max.requires_grad == False
