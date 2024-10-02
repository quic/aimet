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
""" Module for checking consistenty between old quantsim and quantsim v1.5 """

import os
import json
import tempfile
import pytest
import torch
import random
import numpy as np

from ..models_ import models_to_test

from aimet_common.defs import QuantScheme

from aimet_torch.v1.quantsim import QuantizationSimModel as V1QuantizationSimModel
from aimet_torch.v2.quantsim import QuantizationSimModel as V2QuantizationSimModel


CONFIG_DEFAULT = {
    "defaults": {
        "ops": {
            "is_output_quantized": "True",
            "is_symmetric": "False"
        },
        "params": {
            "is_quantized": "False",
            "is_symmetric": "True"
        },
        "per_channel_quantization": "True",
    },
    "params": {},
    "op_type": {},
    "supergroups": [],
    "model_input": {},
    "model_output": {}
}

CONFIG_PARAM_QUANT = {
    "defaults": {
        "ops": {
            "is_output_quantized": "True",
            "is_symmetric": "False"
        },
        "params": {
            "is_quantized": "False",
            "is_symmetric": "True"
        }
    },
    "params": {
        "weight": {
            "is_quantized": "True",
            "is_symmetric": "False"
        }
    },
    "op_type": {},
    "supergroups": [],
    "model_input": {},
    "model_output": {}
}

CONFIG_OP_SPECIFIC_QUANT = {
    "defaults": {
        "ops": {
            "is_output_quantized": "True",
            "is_symmetric": "False"
        },
        "params": {
            "is_quantized": "False",
            "is_symmetric": "True"
        }
    },
    "params": {},
    "op_type": {
        "Conv": {
            "is_input_quantized": "True",
            "is_symmetric": "False",
            "params": {
                "bias": {
                    "is_quantized": "True",
                    "is_symmetric": "False"
                }
            },
        }
    },
    "supergroups": [],
    "model_input": {},
    "model_output": {}
}

CONFIG_OP_SPECIFIC_QUANT_PER_CHANNEL = {
    "defaults": {
        "ops": {
            "is_output_quantized": "True",
            "is_symmetric": "False"
        },
        "params": {
            "is_quantized": "True",
            "is_symmetric": "True"
        },
        "per_channel_quantization": "True",
    },
    "params": {
        "bias": {
            "is_quantized": "False"
        },
    },
    "op_type": {
        "Conv": {
            "per_channel_quantization": "True"
        },
    },
    "supergroups": [],
    "model_input": {},
    "model_output": {}
}

CONFIG_SUPERGROUP = {
    "defaults": {
        "ops": {
            "is_output_quantized": "True",
            "is_symmetric": "False"
        },
        "params": {
            "is_quantized": "False",
            "is_symmetric": "False"
        }
    },
    "params": {},
    "op_type": {},
    "supergroups": [
        {
            "op_list": ["Conv", "Relu"]
        },
        {
            "op_list": ["Relu", "MaxPool"]
        },
        {
            "op_list": ["Conv", "Relu", "AveragePool"]
        },
        {
            "op_list": ["Conv", "Clip"]
        },
    ],
    "model_input": {},
    "model_output": {}
}

@pytest.fixture
def config_path(request):
    config_json = request.param
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config_path = os.path.join(temp_dir, "quantsim_config.json")
        with open(temp_config_path, 'w') as temp_config_file:
            json.dump(config_json, temp_config_file)
        yield temp_config_path


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _compute_sqnr(orig_tensor: torch.Tensor,
                  noisy_tensor: torch.Tensor,
                  in_place: bool = False) -> float:
    """
    Compute SQNR between two tensors.

    IMPORTANT: If in_place is True, the input tensors will be used as in-place buffer
               and hence will have been corrupted when this function returns.

    :param orig_tensor: Original tensor
    :param noisy_tensor: Noisy tensor
    :param in_place: If True, use the input tensors as in-place buffer
    :return: SQNR
    """
    assert orig_tensor.shape == noisy_tensor.shape
    assert orig_tensor.dtype == noisy_tensor.dtype

    # SQNR := E[signal**2] / E[noise**2]
    if in_place:
        signal = orig_tensor
        noise = noisy_tensor.negative_().add_(orig_tensor)
    else:
        signal = orig_tensor.detach().clone()
        noise = orig_tensor - noisy_tensor

    sqnr = signal.square_().mean() / (noise.square_().mean() + 0.0001)
    return float(sqnr)


@pytest.mark.parametrize('quant_scheme', [QuantScheme.post_training_tf,
                                          QuantScheme.training_range_learning_with_tf_init,
                                          QuantScheme.post_training_percentile,
                                          QuantScheme.post_training_tf_enhanced,
                                         ])
@pytest.mark.parametrize('seed', range(3))
class TestQuantsimLogits:
    @staticmethod
    @torch.no_grad()
    def check_qsim_logit_consistency(config, quant_scheme, model, dummy_input):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "quantsim_config.json")
            with open(config_path, 'w') as temp_config_file:
                json.dump(config, temp_config_file)

            v1_sim = V1QuantizationSimModel(model, dummy_input, quant_scheme,
                                            default_param_bw=8,
                                            default_output_bw=16,
                                            config_file=config_path)
            v1_sim.set_percentile_value(99)

            v2_sim = V2QuantizationSimModel(model, dummy_input, quant_scheme,
                                            default_param_bw=8,
                                            default_output_bw=16,
                                            config_file=config_path)
            v2_sim.set_percentile_value(99)

            if isinstance(dummy_input, torch.Tensor):
                dummy_input = (dummy_input,)

            v1_sim.compute_encodings(lambda sim_model, _: sim_model(*dummy_input),
                                    forward_pass_callback_args=None)

            v2_sim.compute_encodings(lambda sim_model, _: sim_model(*dummy_input),
                                    forward_pass_callback_args=None)

            v1_logits = v1_sim.model(*dummy_input)
            v2_logits = v2_sim.model(*dummy_input)
            fp_logits = model(*dummy_input)

            if not isinstance(v1_logits, list):
                v1_logits = [v1_logits]
                v2_logits = [v2_logits]
                fp_logits = [fp_logits]

            assert len(v1_logits) == len(v2_logits)
            for v1_logit, v2_logit, fp_logit in zip(v1_logits, v2_logits, fp_logits):
                # off-by-one test is not realistic for histogram-based encoding analyzers
                # due to some hidden behaviors of V1 histogram
                v1_sqnr = _compute_sqnr(fp_logit, v1_logit)
                v2_sqnr = _compute_sqnr(fp_logit, v2_logit)
                assert v1_sqnr * 0.95 < v2_sqnr
                    

    @pytest.mark.parametrize('model_cls,input_shape', [(models_to_test.SingleResidual, (10, 3, 32, 32)),
                                                       (models_to_test.SoftMaxAvgPoolModel, (10, 4, 256, 512)),
                                                       (models_to_test.QuantSimTinyModel, (10, 3, 32, 32))])
    def test_default_config(self, model_cls, input_shape, quant_scheme, seed):
        set_seed(seed)
        model = model_cls()
        dummy_input = torch.randn(input_shape)
        self.check_qsim_logit_consistency(CONFIG_DEFAULT, quant_scheme, model, dummy_input)

    @pytest.mark.parametrize('model_cls,input_shape', [(models_to_test.SingleResidual, (10, 3, 32, 32)),
                                                       (models_to_test.QuantSimTinyModel, (10, 3, 32, 32))])
    def test_param_quant(self, model_cls, input_shape, quant_scheme, seed):
        set_seed(seed)
        model = model_cls()
        dummy_input = torch.randn(input_shape)
        self.check_qsim_logit_consistency(CONFIG_PARAM_QUANT, quant_scheme, model, dummy_input)

    @pytest.mark.parametrize('model_cls,input_shape', [(models_to_test.SingleResidual, (10, 3, 32, 32)),
                                                       (models_to_test.QuantSimTinyModel, (10, 3, 32, 32))])
    def test_op_specific_quant(self, model_cls, input_shape, quant_scheme, seed):
        set_seed(seed)
        model = model_cls()
        dummy_input = torch.randn(input_shape)
        # Check per-tensor quantization for conv op
        self.check_qsim_logit_consistency(CONFIG_OP_SPECIFIC_QUANT, quant_scheme, model, dummy_input)

        # Check per-channel quantization for conv op
        self.check_qsim_logit_consistency(CONFIG_OP_SPECIFIC_QUANT_PER_CHANNEL, quant_scheme, model, dummy_input)

    def test_supergroup(self, quant_scheme, seed):
        set_seed(seed)
        model = models_to_test.QuantSimTinyModel()
        dummy_input = torch.randn(10, 3, 32, 32)
        self.check_qsim_logit_consistency(CONFIG_SUPERGROUP, quant_scheme, model, dummy_input)

    def test_multi_input(self, quant_scheme, seed):
        set_seed(seed)
        model = models_to_test.MultiInput()
        dummy_input = (torch.rand(10, 3, 32, 32), torch.rand(10, 3, 20, 20))
        self.check_qsim_logit_consistency(CONFIG_DEFAULT, quant_scheme, model, dummy_input)

    def test_multi_output(self, quant_scheme, seed):
        set_seed(seed)
        model = models_to_test.ModelWith5Output()
        dummy_input = torch.randn(1, 3, 224, 224)
        self.check_qsim_logit_consistency(CONFIG_DEFAULT, quant_scheme, model, dummy_input)
