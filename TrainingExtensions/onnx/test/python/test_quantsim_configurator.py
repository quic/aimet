# /usr/bin/env python3.8
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
import json
import os
from aimet_onnx.quantsim import QuantizationSimModel
import test_models


class TestQuantSimConfig:
    """Tests for applying config to QuantizationSimModel"""
    def test_qs_config_dummy_model(self):
        model = test_models.build_dummy_model()
        sim = QuantizationSimModel(model)
        assert sim.qc_quantize_op_dict['conv_w'].enabled == True
        assert sim.qc_quantize_op_dict['conv_b'].enabled == False
        assert sim.qc_quantize_op_dict['fc_w'].enabled == True
        assert sim.qc_quantize_op_dict['fc_b'].enabled == False
        assert sim.qc_quantize_op_dict['input'].enabled == True
        assert sim.qc_quantize_op_dict['3'].enabled == False
        assert sim.qc_quantize_op_dict['4'].enabled == True
        assert sim.qc_quantize_op_dict['5'].enabled == True
        assert sim.qc_quantize_op_dict['6'].enabled == True
        assert sim.qc_quantize_op_dict['output'].enabled == True

    def test_default_config(self):
        model = test_models.build_dummy_model()

        quantsim_config = {
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
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json')
        for name in ['3', '4', '5', '6', 'output']:
            assert sim.qc_quantize_op_dict[name].enabled == True
            assert sim.qc_quantize_op_dict[name].use_symmetric_encodings == False

    def test_param_config(self):
        model = test_models.build_dummy_model()

        quantsim_config = {
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
            "params":
                {
                    "weight": {
                        "is_quantized": "True",
                        "is_symmetric": "True"
                    },
                },
            "op_type": {
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json')
        for name in ['conv_w', 'fc_w']:
            assert sim.qc_quantize_op_dict[name].enabled == True
            assert sim.qc_quantize_op_dict[name].use_symmetric_encodings == True

        for name in ['conv_b', 'fc_b']:
            assert sim.qc_quantize_op_dict[name].enabled == False
            assert sim.qc_quantize_op_dict[name].use_symmetric_encodings == True

    def test_op_level_config_and_model_output(self):
        model = test_models.build_dummy_model()

        quantsim_config = {
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
                        "weight": {
                            "is_quantized": "True",
                            "is_symmetric": "False"
                        }
                    },
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True",
            }
        }
        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json')

        assert sim.qc_quantize_op_dict['conv_w'].enabled == True
        assert sim.qc_quantize_op_dict['conv_w'].use_symmetric_encodings == False
        assert sim.qc_quantize_op_dict['input'].enabled == True
        assert sim.qc_quantize_op_dict['input'].use_symmetric_encodings == False
        assert sim.qc_quantize_op_dict['output'].enabled == True

    def test_config_for_model_input(self):
        model = test_models.build_dummy_model()

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }

        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json')
        assert sim.qc_quantize_op_dict['input'].enabled == True

    def test_parse_config_file_supergroups(self):
        """ Test that supergroup quantization parameters are set correctly when using json config file """
        model = test_models.build_dummy_model()

        quantsim_config = {
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
            ],
            "model_input": {},
            "model_output": {}
        }

        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json')

        # 3 in conv output, 4 is relu output (even though it was not touched with Conv, relu pattern, it was disabled for
        # relu maxpool pattern
        for name in ['3', '4',]:
            assert sim.qc_quantize_op_dict[name].enabled == False

        assert sim.qc_quantize_op_dict['5'].enabled == True

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_parse_config_file_symmetric_modes(self):
        """ Test that model output quantization parameters are set correctly when using json config file """
        model = test_models.build_dummy_model()

        quantsim_config = {
            "defaults":
            {
                "ops": {},
                "params":
                {
                    "is_symmetric": "True"
                },
                "strict_symmetric": "True",
                "unsigned_symmetric": "False"
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
            }
        }
        if not os.path.exists('./data'):
            os.makedirs('./data')
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, config_file='./data/quantsim_config.json')

        for quantizer in sim.qc_quantize_op_dict.values():
            assert quantizer.use_strict_symmetric == True
            assert quantizer.use_unsigned_symmetric == False
