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
import itertools

import torch
import tempfile
import os
import json
import pytest
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.quantization.encoding_analyzer import PercentileEncodingAnalyzer
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine import AffineQuantizerBase
from aimet_torch.v2.nn import BaseQuantizationMixin
from ..models_ import test_models

def encodings_are_close(quantizer_1: AffineQuantizerBase, quantizer_2: AffineQuantizerBase):
    min_1, max_1 = quantizer_1.get_min(), quantizer_1.get_max()
    min_2, max_2 = quantizer_2.get_min(), quantizer_2.get_max()
    return torch.allclose(min_1, min_2) \
           and torch.allclose(max_1, max_2) \
           and quantizer_1.bitwidth == quantizer_2.bitwidth \
           and quantizer_1.symmetric == quantizer_2.symmetric


class TestPercentileScheme:
    """ Test Percentile quantization scheme """

    def test_set_percentile_value(self):
        """ Test pecentile scheme by setting different percentile values """

        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input, quant_scheme="percentile")
        weight_quantizer = sim.model.conv.param_quantizers["weight"]
        assert isinstance(weight_quantizer.encoding_analyzer, PercentileEncodingAnalyzer)

        sim.set_percentile_value(99.9)
        assert weight_quantizer.encoding_analyzer.percentile == 99.9

        sim.compute_encodings(forward_pass, None)
        weight_max_99p9 = weight_quantizer.get_max()

        sim.set_percentile_value(90.0)
        assert weight_quantizer.encoding_analyzer.percentile == 90.0
        sim.compute_encodings(forward_pass, None)
        weight_max_90p0 = weight_quantizer.get_max()

        assert torch.all(weight_max_99p9.gt(weight_max_90p0))

    @pytest.mark.parametrize("config_file", (None, get_path_for_per_channel_config()))
    def test_set_and_freeze_param_encodings(self, config_file):
        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)
        sim = QuantizationSimModel(model, dummy_input, config_file=config_file)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = "test_model"
            sim.export(temp_dir, fname, dummy_input)
            file_path = os.path.join(temp_dir, fname + '.encodings')

            sim_2 = QuantizationSimModel(model, dummy_input, config_file=config_file)

            """
            Given: call set_and_freeze_param_encodigns
            """
            sim_2.set_and_freeze_param_encodings(file_path)

        """
        When: Compare sim_2 param encodings to sim_1 param encodings
        Then: Encodings should matchn
        """
        assert encodings_are_close(sim.model.conv.param_quantizers["weight"], sim_2.model.conv.param_quantizers["weight"])

        """
        When: Inspect param quantizers
        Then: param_quantizer._is_encoding_frozen() == True
        """
        assert sim_2.model.conv.param_quantizers["weight"]._is_encoding_frozen()
        assert not sim_2.model.conv.output_quantizers[0]._is_encoding_frozen()

    @pytest.mark.parametrize("config_file", (None, get_path_for_per_channel_config()))
    def test_load_and_freeze_encodings(self, config_file):
        model = test_models.TinyModel()
        dummy_input = torch.rand(1, 3, 32, 32)
        sim = QuantizationSimModel(model, dummy_input, config_file=config_file)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = "test_model"
            sim.export(temp_dir, fname, dummy_input)
            file_path = os.path.join(temp_dir, fname + '_torch.encodings')
            sim_2 = QuantizationSimModel(test_models.TinyModel(), dummy_input, config_file=config_file)
            sim_2.load_and_freeze_encodings(file_path)

        for module in sim_2.model.modules():
            if isinstance(module, QuantizerBase):
                assert module._is_encoding_frozen()

        for name, child in sim.model.named_children():
            child_2 = getattr(sim_2.model, name)

            for quantizer_1, quantizer_2 in zip(itertools.chain(child.output_quantizers, child.input_quantizers, child.param_quantizers.values()),
                                                itertools.chain(child_2.output_quantizers, child_2.input_quantizers, child_2.param_quantizers.values())):
                if quantizer_1 is None:
                    assert quantizer_2 is None
                    continue
                if quantizer_2 is None:
                    assert quantizer_1 is None
                    continue
                assert encodings_are_close(quantizer_1, quantizer_2)

    def test_load_and_freeze_with_partial_encodings(self):
        """ Test load_and_freeze encoding API with partial_encodings """
        model = test_models.TinyModelWithNoMathInvariantOps()
        dummy_input = torch.randn([1, 3, 24, 24])

        sample_encoding = {"min": -4, "max": 4, "scale": 0.03, "offset": 8,
                           "bitwidth": 8, "is_symmetric": "False", "dtype": "int"}

        sample_partial_encodings_list = [{"activation_encodings": {"conv1": {"input": {"0": sample_encoding}},
                                                                   "mul1": {"output": {"0": sample_encoding}}},
                                          "param_encodings": {}},
                                         {"activation_encodings": {"add1": {"output": {"0": sample_encoding}}},
                                          "param_encodings": {}},
                                         {"activation_encodings": {"mul1": {"output": {"0": sample_encoding}}},
                                          "param_encodings": {"conv1.weight": [sample_encoding]}}]

        for partial_encodings in sample_partial_encodings_list:
            with open('./data/partial_encoding.json', 'w') as f:
                json.dump(partial_encodings, f)

            sim = QuantizationSimModel(model, dummy_input)

            sim.load_and_freeze_encodings('./data/partial_encoding.json', ignore_when_quantizer_disabled=True)

            # all output quantizers and model input quantizer need to enabled (no op specific config)
            assert sim.model.mul1.output_quantizers[0] is not None
            assert sim.model.mul2.output_quantizers[0] is not None
            assert sim.model.add1.output_quantizers[0] is not None
            assert sim.model.conv1.output_quantizers[0] is not None
            assert sim.model.conv1.input_quantizers[0] is not None

            # conv param quantizer needs to be enabled
            assert sim.model.conv1.param_quantizers['weight'] is not None



class TestQuantsimUtilities:

    def test_populate_marker_map(self):
        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)
        sim = QuantizationSimModel(model, dummy_input)
        conv_layer = sim.model.conv
        for name, module in sim.model.named_modules():
            if module is conv_layer:
                conv_name = name
                break
        assert conv_name not in sim._module_marker_map.keys()
        sim.run_modules_for_traced_custom_marker([conv_layer], dummy_input)
        assert conv_name in sim._module_marker_map.keys()
        assert torch.equal(sim._module_marker_map[conv_name](dummy_input), conv_layer.get_original_module()(dummy_input))

    def test_get_qc_quantized_modules(self):
        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)
        sim = QuantizationSimModel(model, dummy_input)
        conv_layer = sim.model.conv
        assert ("conv", conv_layer) in sim._get_qc_quantized_layers(sim.model)

    def test_get_leaf_module_to_name_map(self):
        model = test_models.NestedConditional()
        dummy_input = torch.rand(1, 3), torch.tensor([True])
        sim = QuantizationSimModel(model, dummy_input)
        leaf_modules = sim._get_leaf_module_to_name_map()
        for name, module in sim.model.named_modules():
            if isinstance(module, BaseQuantizationMixin):
                assert module in leaf_modules.keys()
                assert leaf_modules[module] == name
