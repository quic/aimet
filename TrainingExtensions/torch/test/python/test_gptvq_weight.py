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
"""Test GPTVQ weight"""

import json
import tempfile

import pytest
import torch

from aimet_torch.gptvq.defs import GPTVQSupportedModules
from aimet_torch.gptvq.gptvq_weight import GPTVQ, GPTVQParameters
from aimet_torch.v2.nn import BaseQuantizationMixin
from models import test_models


QUANTSIM_CONFIG = {
    "defaults": {
        "ops": {"is_output_quantized": "True"},
        "params": {"is_quantized": "True", "is_symmetric": "True"},
        "strict_symmetric": "False",
        "per_channel_quantization": "True",
    },
    "params": {"bias": {"is_quantized": "False"}},
    "op_type": {
        "Squeeze": {"is_output_quantized": "False"},
        "Pad": {"is_output_quantized": "False"},
        "Mean": {"is_output_quantized": "False"},
        # Enable per-channel quantization for Gemm to validate GPTVQ algorithm
        "Gemm": {"per_channel_quantization": "True"},
        "LayerNorm": {"per_channel_quantization": "False"},
        "Gather": {"is_output_quantized": "False"},
    },
    "supergroups": [
        {"op_list": ["Conv", "Relu"]},
        {"op_list": ["Conv", "Clip"]},
        {"op_list": ["ConvTranspose", "Relu"]},
        {"op_list": ["Add", "Relu"]},
        {"op_list": ["Gemm", "Relu"]},
    ],
    "model_input": {"is_input_quantized": "True"},
    "model_output": {},
}


class TestGPTVQWeight:
    @pytest.mark.parametrize("vector_bw", [4, 8, 16])
    @pytest.mark.parametrize("rows_per_block", [32, 64])
    def test_quant_sim_initialization_in_gptvq(self, vector_bw, rows_per_block):
        model = test_models.ModelWithThreeLinears()
        gptvq_parameters = GPTVQParameters(
            vector_bw=vector_bw, rows_per_block=rows_per_block
        )
        dummy_input = torch.randn(1, 768)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/quantsim_config.json"
            with open(config_path, "w") as f:
                json.dump(QUANTSIM_CONFIG, f)

            quant_sim = GPTVQ._get_quantsim(
                model, dummy_input, gptvq_parameters, config_file_path=config_path
            )

        for module in quant_sim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                # Input/Output quantizers should be disabled
                assert all((x is None for x in module.input_quantizers))
                assert all((x is None for x in module.output_quantizers))

                if isinstance(module.get_original_module(), GPTVQSupportedModules):
                    weight_shape = module.weight.shape
                    weight_quantizer = module.param_quantizers["weight"]
                    # Bitwidth, shape and block_size should be matched with GPTVQ parameters
                    assert weight_quantizer.bitwidth == gptvq_parameters.vector_bw
                    assert weight_quantizer.shape == (weight_shape[0] // gptvq_parameters.rows_per_block, 1)
                    assert weight_quantizer.block_size == (gptvq_parameters.rows_per_block, weight_shape[1])
                    assert weight_quantizer.is_initialized()

    def test_exported_param_encodings_after_gptvq(self):
        model = test_models.ModelWithThreeLinears()
        gptvq_parameters = GPTVQParameters()
        dummy_input = torch.randn(1, 768)
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/quantsim_config.json"
            with open(config_path, "w") as f:
                json.dump(QUANTSIM_CONFIG, f)

            _ = GPTVQ.apply_gptvq(
                model,
                dummy_input,
                gptvq_parameters,
                param_encoding_path=temp_dir,
                config_file_path=config_path,
            )

            with open(f"{temp_dir}/gptvq.encodings") as f:
                encodings = json.load(f)
            param_encodings = encodings["param_encodings"]

        for name, module in model.named_modules():
            if isinstance(module, GPTVQSupportedModules):
                num_of_channels = module.weight.shape[0]
                weight_encodings = param_encodings[f"{name}.weight"]
                # The number of encodings should be same with the number of channels
                assert num_of_channels == len(weight_encodings)
                # Encodings in same block should have same encodings parameters
                # e.g., 0 to 31 channel (First block) should have same min/max encodings
                for i in range(0, num_of_channels, gptvq_parameters.rows_per_block):
                    assert len({x["min"] for x in weight_encodings[i : i + gptvq_parameters.rows_per_block]}) == 1
                    assert len({x["max"] for x in weight_encodings[i : i + gptvq_parameters.rows_per_block]}) == 1
