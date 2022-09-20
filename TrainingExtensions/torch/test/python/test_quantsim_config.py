# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Module for testing quantsim config feature """
import pytest
import json
import os
import torch
import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme, QuantizationDataType, QuantDtypeBwInfo
from aimet_torch.examples.test_models import SingleResidual, QuantSimTinyModel, MultiInput, SingleResidualWithModuleAdd
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.quantsim_config import quantsim_config as qsim_config
from aimet_torch.quantsim_config.quantsim_config import get_all_ops_in_neighborhood
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch import utils
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.tensor_quantizer import StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer

class ModelWithBertCustomLayerNormGelu(torch.nn.Module):
    """ Model with PyTorch LayerNorm and gelu """

    def __init__(self):
        super(ModelWithBertCustomLayerNormGelu, self).__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        # default attribute -
        # eps = 1e-05 and elementwise_affine = True
        # parameters : weight and bias
        self.customln1 = torch.nn.LayerNorm(4)
        self.gelu1 = torch.nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.customln1(x)
        x = self.gelu1(x)
        return x


# pylint: disable=protected-access
class TestQuantsimConfig:
    """ Class containing unit tests for quantsim config feature """

    def test_parse_config_file_defaults(self):
        """ Test that default quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
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
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32), in_place=True)
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # Output of add op is input quantized
                if name == 'relu3':
                    assert module.input_quantizer.enabled
                else:
                    assert not module.input_quantizer.enabled
                assert module.output_quantizers[0].enabled
                assert not module.input_quantizer.use_symmetric_encodings
                assert not module.output_quantizers[0].use_symmetric_encodings
                if module.param_quantizers:
                    for _, param_quantizer in module.param_quantizers.items():
                        assert not param_quantizer.enabled
                        assert param_quantizer.use_symmetric_encodings
                        assert len(param_quantizer._cppOp) > 1

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_parse_config_file_params(self):
        """ Test that param specific quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()

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
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32))
        for _, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if module.param_quantizers:
                    for param_name, param_quantizer in module.param_quantizers.items():
                        if param_name == 'weight':
                            assert param_quantizer.enabled
                            assert not param_quantizer.use_symmetric_encodings
                        else:
                            assert not param_quantizer.enabled
                            assert param_quantizer.use_symmetric_encodings
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_parse_config_file_default_supported_kernels(self):
        """
        Test that the supported_kernels in the defaults section is parsed correctly and its values are added
        in the dict _supported_kernels
        """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                },
                "supported_kernels":[
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 8,
                            "dtype": "int"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "float"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "float"
                        }
                    }
                ]
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

        expected_supported_kernels = [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": QuantizationDataType.int
                        },
                        "param": {
                            "bitwidth": 8,
                            "dtype": QuantizationDataType.int
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": QuantizationDataType.float
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": QuantizationDataType.float
                        }
                    }
                ]

        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32))

        supported_kernels_in_defaults = sim.get_supported_kernels()["defaults"]
        assert len(supported_kernels_in_defaults) == 2
        assert supported_kernels_in_defaults == expected_supported_kernels

    def test_parse_config_file_op_type(self):
        """ Test that op specific quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()

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
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32))
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if isinstance(module._module_to_wrap, torch.nn.Conv2d):
                    assert module.input_quantizer.enabled
                    assert not module.input_quantizer.use_symmetric_encodings
                    assert not module.output_quantizers[0].use_symmetric_encodings
                else:
                    # Output of add op is input quantized
                    if name == 'relu3':
                        assert module.input_quantizer.enabled
                    else:
                        assert not module.input_quantizer.enabled
                    assert module.output_quantizers[0].enabled
                    assert not module.input_quantizer.use_symmetric_encodings
                    assert not module.output_quantizers[0].use_symmetric_encodings
                if module.param_quantizers:
                    for param_name, param_quantizer in module.param_quantizers.items():
                        if isinstance(module._module_to_wrap, torch.nn.Conv2d) and param_name == 'bias':
                            assert param_quantizer.enabled
                            assert not param_quantizer.use_symmetric_encodings
                        else:
                            assert not param_quantizer.enabled
                            assert param_quantizer.use_symmetric_encodings
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def _test_parse_config_file_op_type_per_channel_helper(self, per_channel_fields):
        """ helper function to test out per_channel_quantization"""
        for k in ['defaults', 'Conv', 'Gemm']:
            assert k in per_channel_fields.keys()

        model = MultiInput()
        model.eval()
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                },
                "per_channel_quantization": per_channel_fields["defaults"],
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                },
            },
            "op_type": {
                "Conv": {
                    "per_channel_quantization": per_channel_fields["Conv"]
                },
                "Gemm": {
                    "per_channel_quantization": per_channel_fields["Gemm"]
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20)))

        return sim

    def test_parse_config_file_op_type_per_channel(self):
        """ Test that op specific quantization parameters are set correctly when using json config file """

        # test 1: expect all to be StaticGridPerChannelQuantizer
        per_channel_fields = {"defaults": "True", "Conv": "True", "Gemm": "True"}
        sim = self._test_parse_config_file_op_type_per_channel_helper(per_channel_fields)

        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if 'weight' in module.param_quantizers:
                    assert isinstance(module.param_quantizers['weight'], StaticGridPerChannelQuantizer)


        # test 2: expect all but Conv to be StaticGridPerChannelQuantizer
        per_channel_fields = {"defaults": "True", "Conv": "False", "Gemm": "True"}
        sim = self._test_parse_config_file_op_type_per_channel_helper(per_channel_fields)

        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if 'weight' in module.param_quantizers:
                    if isinstance(module._module_to_wrap, torch.nn.Conv2d):
                        assert isinstance(module.param_quantizers['weight'], StaticGridPerTensorQuantizer)
                    else:
                        assert isinstance(module.param_quantizers['weight'], StaticGridPerChannelQuantizer)


        # test 3: expect all but Conv and Gemm to be StaticGridPerChannelQuantizer
        per_channel_fields = {"defaults": "True", "Conv": "False", "Gemm": "False"}
        sim = self._test_parse_config_file_op_type_per_channel_helper(per_channel_fields)

        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if 'weight' in module.param_quantizers:
                    if isinstance(module._module_to_wrap, torch.nn.Conv2d) or isinstance(module._module_to_wrap,
                                                                                         torch.nn.Linear):
                        assert isinstance(module.param_quantizers['weight'], StaticGridPerTensorQuantizer)
                    else:
                        assert isinstance(module.param_quantizers['weight'], StaticGridPerChannelQuantizer)


        # test 4: expect all in StaticGridPerTensorQuantizer except Conv which will be in StaticGridPerChannelQuantizer
        per_channel_fields = {"defaults": "False", "Conv": "True", "Gemm": "False"}
        sim = self._test_parse_config_file_op_type_per_channel_helper(per_channel_fields)

        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if 'weight' in module.param_quantizers:
                    if isinstance(module._module_to_wrap, torch.nn.Conv2d):
                        assert isinstance(module.param_quantizers['weight'], StaticGridPerChannelQuantizer)
                    else:
                        assert isinstance(module.param_quantizers['weight'], StaticGridPerTensorQuantizer)

        random_input = (torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20))

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*random_input)


        # test 5: test to make sure only Conv has param encodings of length greater than 1(per-channel),
        # others have only one entry
        sim.compute_encodings(forward_pass, None)
        sim.export('./data/', 'test_parse_config_file_op_type_per_channel',
                   dummy_input=(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20)))

        with open("./data/test_parse_config_file_op_type_per_channel.encodings", "r") as encodings_file:
            encodings = json.load(encodings_file)
            assert len(encodings["param_encodings"]["bn1.weight"]) == 1
            assert len(encodings["param_encodings"]["fc.weight"]) == 1
            assert len(encodings["param_encodings"]["conv1.weight"]) == 16
            assert len(encodings["param_encodings"]["conv2.weight"]) == 8
            assert len(encodings["param_encodings"]["conv3.weight"]) == 8

    def test_parse_config_file_op_type_supported_kernels(self):
        """
        Test that the supported_kernels in the op_type section is parsed correctly and its values are added
        in the dict _supported_kernels
        """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "False",
                    "is_symmetric": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "weight": {
                    "is_quantized": "True",
                    "is_symmetric": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "supported_kernels": [
                        {
                            "activation": {
                                "bitwidth": 16,
                                "dtype": "int"
                            },
                            "param": {
                                "bitwidth": 8,
                                "dtype": "int"
                            }
                        }
                    ]
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

        expected_supported_kernels = [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": QuantizationDataType.int
                        },
                        "param": {
                            "bitwidth": 8,
                            "dtype": QuantizationDataType.int
                        }
                    }
                ]

        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32))

        supported_kernels_in_defaults = sim.get_supported_kernels()["Conv"]
        assert len(supported_kernels_in_defaults) == 1
        assert supported_kernels_in_defaults == expected_supported_kernels

    def test_parse_config_file_supergroups(self):
        """ Test that supergroup quantization parameters are set correctly when using json config file """
        model = QuantSimTinyModel()
        model.eval()

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
                    "op_list": ["Conv", "BatchNormalization"]
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
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        # Use in_place=True here for easy access to modules through model instance variables
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   in_place=True, dummy_input=torch.rand(1, 3, 32, 32))
        for _, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # Check configs for starts of supergroups
                if module in [model.conv1, model.relu1, model.conv2, model.conv3]:
                    assert not module.output_quantizers[0].enabled
                # Check configs for middle ops in supergroups
                elif module == model.relu3:
                    assert not module.input_quantizer.enabled
                    assert not module.output_quantizers[0].enabled
                # Check configs for ends of supergroups
                elif module in [model.bn1, model.maxpool, model.bn2, model.avgpool, model.relu2]:
                    assert not module.input_quantizer.enabled
                    assert module.output_quantizers[0].enabled
                else:
                    assert not module.input_quantizer.enabled
                    assert module.output_quantizers[0].enabled

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_parse_config_file_elementwise_ops(self):
        """ Test that elementwise op quantizers are set as expected """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {}
            },
            "params": {},
            "op_type": {
                "Add": {
                    "is_input_quantized": "True"
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32))
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if name in ['conv3', 'ada']:
                    # model.conv3 and model.ada are inputs to add
                    assert module.output_quantizers[0].enabled
                else:
                    assert not module.output_quantizers[0].enabled
                assert not module.input_quantizer.enabled
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_parse_config_file_model_inputs(self):
        """ Test that model input quantization parameters are set correctly when using json config file """
        model = MultiInput()
        model.eval()

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
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced, config_file='./data/quantsim_config.json',
                                   dummy_input=(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 20, 20)), in_place=True)
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # Output of add op is input quantized
                if name in ('conv1', 'conv3'):
                    assert module.input_quantizer.enabled
                else:
                    assert not module.input_quantizer.enabled
                assert not module.output_quantizers[0].enabled
                assert not module.input_quantizer.use_symmetric_encodings
                assert not module.output_quantizers[0].use_symmetric_encodings

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_parse_config_file_model_outputs(self):
        """ Test that model output quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced, config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32))
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if name == 'fc':
                    # model.conv3 and model.ada are inputs to add
                    assert module.output_quantizers[0].enabled
                else:
                    assert not module.output_quantizers[0].enabled
                assert not module.input_quantizer.enabled
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_supergroups_with_functional_add(self):
        """ Test supergroup with functional add """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [
                {
                    "op_list": ["Add", "Relu"]
                }
            ],
            "model_input": {},
            "model_output": {}
        }
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        # Use in_place=True here for easy access to modules through model instance variables
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   in_place=True, dummy_input=torch.rand(1, 3, 32, 32))
        for _, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # Check configs for starts of supergroups
                if module == model.relu3:
                    # If add were not part of the supergroup, relu's input quantizer would be enabled
                    assert not module.input_quantizer.enabled

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_supergroups_with_module_add(self):
        """ Test supergroup with add module """
        model = SingleResidualWithModuleAdd()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [
                {
                    "op_list": ["Add", "Relu"]
                }
            ],
            "model_input": {},
            "model_output": {}
        }
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        # Use in_place=True here for easy access to modules through model instance variables
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   in_place=True, dummy_input=torch.rand(1, 3, 32, 32))
        for _, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # Check configs for starts of supergroups
                if module == model.add:
                    # If add were not part of the supergroup, relu's input quantizer would be enabled
                    assert not module.output_quantizer.enabled
                else:
                    assert module.output_quantizer.enabled
                assert not module.input_quantizer.enabled

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_parse_config_file_symmetric_modes(self):
        """ Test that model output quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {},
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
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32))
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                for q in module.input_quantizers:
                    assert q.use_strict_symmetric
                    assert not q.use_unsigned_symmetric
                for q in module.output_quantizers:
                    assert q.use_strict_symmetric
                    assert not q.use_unsigned_symmetric
                for q in module.param_quantizers.values():
                    assert q.use_strict_symmetric
                    assert not q.use_unsigned_symmetric
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_get_all_ops_in_neighborhood(self):
        """ Test that default quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()
        input_shapes = (1, 3, 32, 32)

        random_inputs = utils.create_rand_tensors_given_shapes(input_shapes)
        conn_graph = ConnectedGraph(model, random_inputs)
        starting_op = conn_graph.get_op_from_module_name('SingleResidual.conv3')
        add_op = [op for op in conn_graph.get_all_ops().values() if op.type == 'Add'][0]
        neighborhood = get_all_ops_in_neighborhood(starting_op, 'output')
        assert len(neighborhood) == 2
        assert starting_op in neighborhood
        assert add_op in neighborhood

    @pytest.mark.cuda
    def test_parse_config_file_defaults_gpu(self):
        """ Test that default quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()
        model.cuda()

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
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced, config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32).cuda(), in_place=True)
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # Output of add op is input quantized
                if name == 'relu3':
                    assert module.input_quantizer.enabled
                else:
                    assert not module.input_quantizer.enabled
                assert module.output_quantizers[0].enabled
                assert not module.input_quantizer.use_symmetric_encodings
                assert not module.output_quantizers[0].use_symmetric_encodings
                if module.param_quantizers:
                    for _, param_quantizer in module.param_quantizers.items():
                        assert not param_quantizer.enabled
                        assert param_quantizer.use_symmetric_encodings

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_gelu_layernorm_quantsim_config(self):
        """
        Create a network with LayerNorm and GELU
        Override quantization config and check config is applied
        This is a validation for actual entry in the map_torch_types_to_onnx in onnx_utils
        This is used by connected graph to apply op level specific quantsim config.
        :return:
        """

        import json
        import aimet_common.libpymo as libpymo
        from aimet_common.defs import QuantScheme
        from aimet_torch.quantsim import QuantizationSimModel

        class ModelWithGeluLayerNorm(torch.nn.Module):
            def __init__(self):
                super(ModelWithGeluLayerNorm, self).__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                # default attribute -
                # eps = 1e-05 and elementwise_affine = True
                # parameters : weight and bias
                self.ln1 = torch.nn.LayerNorm(4)
                self.gelu1 = torch.nn.GELU()

            def forward(self, x):
                x = self.linear1(x)
                x = self.ln1(x)
                x = self.gelu1(x)
                return x

            # create custom config to override LayerNorm and GELU
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
                    "LayerNorm": {
                        "is_input_quantized": "True",
                        "params": {
                            "bias": {
                                "is_quantized": "True"
                            },
                        },
                    },
                    "GELU": {
                        "is_input_quantized": "True",
                    }
                },
                "supergroups": [],
                "model_input": {},
                "model_output": {}
            }

            with open('./data/quantsim_config.json', 'w') as f:
                json.dump(quantsim_config, f)

        model = ModelWithGeluLayerNorm()
        model.eval()
        random_input = torch.rand(1, 4, 4)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*random_input)

        # QuantSim for model
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=random_input)

        sim.compute_encodings(forward_pass, None)

        #  check quantizer added to parameters of LayerNorm
        from aimet_torch.qc_quantize_op import StaticGridPerTensorQuantizer
        assert(isinstance(sim.model.ln1.param_quantizers['weight'], StaticGridPerTensorQuantizer))
        assert(isinstance(sim.model.ln1.param_quantizers['bias'], StaticGridPerTensorQuantizer))


        # LayerNorm input quantization is disabled by default
        # override with custom config file, this needs appropriate entry in onnx node name mapping
        assert(isinstance(sim.model.ln1.input_quantizer, StaticGridPerTensorQuantizer))
        assert(sim.model.ln1.input_quantizer.encoding)
        in_quantizer = sim.model.ln1.input_quantizer
        assert(in_quantizer.enabled)  # disabled by default, override with config file
        assert(in_quantizer.round_mode == libpymo.RoundingMode.ROUND_NEAREST)
        assert(in_quantizer.quant_scheme == QuantScheme.post_training_tf)
        assert(in_quantizer.bitwidth == 8)


        # GELU input quantization is disabled by default
        # override with custom config file, this needs appropriate entry in onnx node name mapping
        assert(isinstance(sim.model.gelu1.input_quantizer, StaticGridPerTensorQuantizer))
        assert(sim.model.gelu1.input_quantizer.encoding)
        in_quantizer = sim.model.gelu1.input_quantizer
        assert(in_quantizer.enabled)  # disabled by default, override with config file
        assert(in_quantizer.round_mode == libpymo.RoundingMode.ROUND_NEAREST)
        assert(in_quantizer.quant_scheme == QuantScheme.post_training_tf)
        assert(in_quantizer.bitwidth == 8)

        # remove test config created
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_default_quantsim_config_not_in_default_config_file_enforce_false(self):
        """
        Tests application of override config rule for default bitwidth and dtype for params and act.
        In this test, default supported kernel list (fp 16) in the config file DOES NOT SUPPORT
        default quantsim config (int 8) + ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG is also set to True.
        Tests application of default config rule for op level bitwidth and dtype for params
        :return:
        """
        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False
        model = SingleResidual()
        model.eval()

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "float"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "float"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    }
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }

        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        INPUT_SHAPE = (1, 3, 32, 32)
        def forward_fn(model, _):
            torch.manual_seed(10)
            model.eval()
            with torch.no_grad():
                _ = model(torch.randn(INPUT_SHAPE))

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32), in_place=True,
                                   default_param_bw=8, default_output_bw=8, default_data_type=QuantizationDataType.int)
        sim.compute_encodings(forward_fn, forward_pass_callback_args=None)

        # all quantizers should be quantsim default quantsim dtype and bw  (int 8)
        assert(sim.model.conv1.param_quantizers['weight'].enabled == True)
        assert(sim.model.conv1.param_quantizers['weight'].bitwidth == 8)
        assert(sim.model.conv1.param_quantizers['weight'].data_type == QuantizationDataType.int)

        assert(sim.model.conv1.output_quantizers[0].bitwidth == 8)
        assert(sim.model.conv1.output_quantizers[0].data_type == QuantizationDataType.int)

        # all quantizers should be quantsim default quantsim dtype and bw  (int 8)
        # that is  QUANTSIM DEFAULT bw / dtype (int 8).
        assert(sim.model.fc.param_quantizers['weight'].enabled)
        assert(sim.model.fc.param_quantizers['bias'].enabled == False)
        assert(sim.model.fc.param_quantizers['weight'].bitwidth == 8)
        assert(sim.model.fc.param_quantizers['weight'].data_type == QuantizationDataType.int)
        assert(sim.model.fc.param_quantizers['bias'].bitwidth == 8)
        assert(sim.model.fc.param_quantizers['bias'].data_type == QuantizationDataType.int)
        assert(sim.model.fc.output_quantizers[0].bitwidth == 8)
        assert(sim.model.fc.output_quantizers[0].data_type == QuantizationDataType.int)
        assert(sim.model.relu1.output_quantizers[0].bitwidth == 8)
        assert(sim.model.relu1.output_quantizers[0].data_type == QuantizationDataType.int)

        # remove test config created
        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_default_quantsim_config_in_default_config_file_enforce_true(self):
        """
        Tests application of override config rule for default bitwidth and dtype for params and act.
        In this test, default supported kernel list (int 8, fp 16) in the config file CONTAINS
        default quantsim config (int 8) + ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG is also set to True.
        Tests application of default config rule for op level bitwidth and dtype for params
        :return:
        """

        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True
        model = SingleResidual()
        model.eval()
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "float"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "float"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 8,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 8,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                            {
                                "activation": {
                                    "bitwidth": 8,
                                    "dtype": "int"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "int"
                                }
                            },
                        ],
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    }
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }

        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        INPUT_SHAPE = (1, 3, 32, 32)
        def forward_fn(model, _):
            torch.manual_seed(10)
            model.eval()
            with torch.no_grad():
                _ = model(torch.randn(INPUT_SHAPE))

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32), in_place=True,
                                   default_data_type=QuantizationDataType.int, default_output_bw=8, default_param_bw=8)
        sim.compute_encodings(forward_fn, forward_pass_callback_args=None)

        # enforce is true, however default quantsim bw / dtype (fp16) is not the config file supported kernels override at index 0.
        # apply override 0 # activation : bw = 16, float # param : bw = 16, float
        assert(sim.model.conv1.param_quantizers['weight'].enabled == True)
        assert(sim.model.conv1.param_quantizers['weight'].bitwidth == 16)
        assert(sim.model.conv1.param_quantizers['weight'].data_type == QuantizationDataType.float)

        assert(sim.model.conv1.output_quantizers[0].bitwidth == 16)
        assert(sim.model.conv1.output_quantizers[0].data_type == QuantizationDataType.float)

        assert(sim.model.fc.param_quantizers['weight'].enabled)
        assert(sim.model.fc.param_quantizers['bias'].enabled == False)
        assert(sim.model.fc.param_quantizers['weight'].bitwidth == 16)
        assert(sim.model.fc.param_quantizers['weight'].data_type == QuantizationDataType.float)
        assert(sim.model.fc.param_quantizers['bias'].bitwidth == 16)
        assert(sim.model.fc.param_quantizers['bias'].data_type == QuantizationDataType.float)
        assert(sim.model.fc.output_quantizers[0].bitwidth == 16)
        assert(sim.model.fc.output_quantizers[0].data_type == QuantizationDataType.float)
        assert(sim.model.relu1.output_quantizers[0].bitwidth == 16)
        assert(sim.model.relu1.output_quantizers[0].data_type == QuantizationDataType.float)

        # remove test config created
        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_default_quantsim_config_not_in_default_config_file_enforce_true(self):
        """
        Tests application of override config rule for default bitwidth and dtype for params and act.
        In this test, default supported kernel list (fp 16) in the config file DOES NOT SUPPORT
        default quantsim config (int 8) + ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG is also set to True.
        Tests application of default config rule for op level bitwidth and dtype for params
        :return:
        """

        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True
        model = SingleResidual()
        model.eval()
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "float"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "float"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    }
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }

        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        INPUT_SHAPE = (1, 3, 32, 32)
        def forward_fn(model, _):
            torch.manual_seed(10)
            model.eval()
            with torch.no_grad():
                _ = model(torch.randn(INPUT_SHAPE))

        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32), in_place=True,
                                   default_data_type=QuantizationDataType.int, default_output_bw=8, default_param_bw=8)
        sim.compute_encodings(forward_fn, forward_pass_callback_args=None)

        # enforce is true, however default quantsim bw / dtype (int 8) is NOT IN the config file supported kernels
        # should be configured with config file default supported kernel [0]
        # activation : bw = 16 , float
        # param : bw = 16, float
        assert(sim.model.conv1.param_quantizers['weight'].enabled == True)
        assert(sim.model.conv1.param_quantizers['weight'].bitwidth == 16)
        assert(sim.model.conv1.param_quantizers['weight'].data_type == QuantizationDataType.float)

        assert(sim.model.conv1.output_quantizers[0].bitwidth == 16)
        assert(sim.model.conv1.output_quantizers[0].data_type == QuantizationDataType.float)

        assert(sim.model.fc.param_quantizers['weight'].enabled)
        assert(sim.model.fc.param_quantizers['bias'].enabled == False)
        assert(sim.model.fc.param_quantizers['weight'].bitwidth == 16)
        assert(sim.model.fc.param_quantizers['weight'].data_type == QuantizationDataType.float)
        assert(sim.model.fc.param_quantizers['bias'].bitwidth == 16)
        assert(sim.model.fc.param_quantizers['bias'].data_type == QuantizationDataType.float)
        assert(sim.model.fc.output_quantizers[0].bitwidth == 16)
        assert(sim.model.fc.output_quantizers[0].data_type == QuantizationDataType.float)
        assert(sim.model.relu1.output_quantizers[0].bitwidth == 16)
        assert(sim.model.relu1.output_quantizers[0].data_type == QuantizationDataType.float)

        # remove test config created
        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_check_correctness_of_dtype_bw_rules_valid_case(self):
        """
        Test to check api check_correctness_of_dtype_bw_rules, valid config case
        :return:
        """

        model = SingleResidual()
        model.eval()
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 8,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                        ],
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    }
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }

        config_file = './data/quantsim_config.json'
        with open(config_file, 'w') as f:
            json.dump(quantsim_config, f)

        INPUT_SHAPE = (1, 3, 32, 32)
        def forward_fn(model, _):
            torch.manual_seed(10)
            model.eval()
            with torch.no_grad():
                _ = model(torch.randn(INPUT_SHAPE))
        from aimet_torch.quantsim_config.quantsim_config import QuantSimConfigurator
        dummy_input = torch.randn(INPUT_SHAPE)
        connected_graph = ConnectedGraph(model, dummy_input)

        qsim_config = QuantSimConfigurator(model, connected_graph, config_file,
                                           quantsim_output_bw=8, quantsim_param_bw=8,
                                           quantsim_data_type=QuantizationDataType.int)

        qsim_dtype_bw = QuantDtypeBwInfo(data_type=QuantizationDataType.int, act_bw=8 , param_bw=8)

        assert qsim_config.check_correctness_of_dtype_bw_rules(qsim_dtype_bw)

        # remove test config created
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_check_correctness_of_dtype_bw_rules_default_supported_kernels_exception_case(self):
        """
        Test to check api check_correctness_of_dtype_bw_rules, invalid default supported_kernels case
        :return:
        """

        model = SingleResidual()
        model.eval()
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 4,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 8,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                        ],
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    }
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }

        config_file = './data/quantsim_config.json'
        with open(config_file, 'w') as f:
            json.dump(quantsim_config, f)

        INPUT_SHAPE = (1, 3, 32, 32)
        from aimet_torch.quantsim_config.quantsim_config import QuantSimConfigurator
        dummy_input = torch.randn(INPUT_SHAPE)
        connected_graph = ConnectedGraph(model, dummy_input)
        qsim_config = QuantSimConfigurator(model, connected_graph, config_file,
                                           quantsim_output_bw=8, quantsim_param_bw=8,
                                           quantsim_data_type=QuantizationDataType.int)

        qsim_dtype_bw = QuantDtypeBwInfo(data_type=QuantizationDataType.int, act_bw=8 , param_bw=8)
        exception_raised = False
        try:
            qsim_config.check_correctness_of_dtype_bw_rules(qsim_dtype_bw)
        except NotImplementedError as exc:
            print(" Test raised exception as expected ", exc)
            exception_raised = True

        assert exception_raised

        # remove test config created
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_check_correctness_of_dtype_bw_rules_op_level_supported_kernels_exception_case(self):
        """
        Test to check api check_correctness_of_dtype_bw_rules, invalid op level supported_kernels case
        :return:
        """

        model = SingleResidual()
        model.eval()
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 8,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 8,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 8,
                                    "dtype": "int"
                                },
                                "param": {
                                    "bitwidth": 4,
                                    "dtype": "int"
                                }
                            },
                        ],
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    }
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }

        config_file = './data/quantsim_config.json'
        with open(config_file, 'w') as f:
            json.dump(quantsim_config, f)

        INPUT_SHAPE = (1, 3, 32, 32)
        def forward_fn(model, _):
            torch.manual_seed(10)
            model.eval()
            with torch.no_grad():
                _ = model(torch.randn(INPUT_SHAPE))
        from aimet_torch.quantsim_config.quantsim_config import QuantSimConfigurator
        dummy_input = torch.randn(INPUT_SHAPE)
        connected_graph = ConnectedGraph(model, dummy_input)

        qsim_config = QuantSimConfigurator(model, connected_graph, config_file,
                                           quantsim_output_bw=8, quantsim_param_bw=8,
                                           quantsim_data_type=QuantizationDataType.int)

        qsim_dtype_bw = QuantDtypeBwInfo(data_type=QuantizationDataType.int, act_bw=8 , param_bw=8)
        exception_raised = False
        try:
            qsim_config.check_correctness_of_dtype_bw_rules(qsim_dtype_bw)
        except NotImplementedError as exc:
            print(" Test raised exception as expected ", exc)
            exception_raised = True
        assert exception_raised

        # remove test config created
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_target_rule_enforced_apply_default_and_op_level_overrides_valid_case(self):
        """
        validates Config overrides provided are valid combination and application of both default level as well as
        op level kernel overrides for dtype and bitiwdth.
        Quantsim created with (int4, int4) defaults
        Default supported kernels override (at index 0 ) is (int 8, int 8) --> applied.
        Default at op level override at index 0 of supported_kernels for Conv type is
        (fp 16, fp16) --> applied to weight param.
        :return:
        """

        model = SingleResidual()
        model.eval()

        # quantsim config has default kernel overrides as well as op level kernel override
        # we begin with quantsim default config (int 4, int 4), during isntantiation of quantsim object.
        # Then, using config file we apply two levels of overrides.
        # 1) default supported_kernels at index 0 , is used to override default act/param bw dtype with int8 / int8
        # 2) After this, at op level, specifically for Conv types, there is a override provided as fp16/ fp16
        # So, param quantizers of conv shall be updated to FP16 as a override, while retaining output at int 8 as
        # configured by default level supported_kernels.

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 8,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 8,
                            "dtype": "int"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 4,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 4,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    },
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                        ]
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }

        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        INPUT_SHAPE = (1, 3, 32, 32)
        def forward_fn(model, _):
            torch.manual_seed(10)
            model.eval()
            with torch.no_grad():
                _ = model(torch.randn(INPUT_SHAPE))

        # set enforce to true for this test
        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                   config_file='./data/quantsim_config.json',
                                   dummy_input=torch.rand(1, 3, 32, 32), in_place=True,
                                   default_data_type=QuantizationDataType.int, default_output_bw=4, default_param_bw=4)
        sim.compute_encodings(forward_fn, forward_pass_callback_args=None)

        # enforce is set to true
        # default supported kernels at index DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX (=0 in this case)
        # is not same as default quantsim bw and dtype(int 4/int4), apply default overrides (int8/ int8).

        assert(sim.model.fc.param_quantizers['weight'].enabled)
        assert(sim.model.fc.param_quantizers['bias'].enabled == False)
        assert(sim.model.fc.param_quantizers['weight'].bitwidth == 8)
        assert(sim.model.fc.param_quantizers['weight'].data_type == QuantizationDataType.int)
        assert(sim.model.fc.param_quantizers['bias'].bitwidth == 8)
        assert(sim.model.fc.param_quantizers['bias'].data_type == QuantizationDataType.int)
        assert(sim.model.fc.output_quantizers[0].bitwidth == 8)
        assert(sim.model.fc.output_quantizers[0].data_type == QuantizationDataType.int)
        assert(sim.model.relu1.output_quantizers[0].bitwidth == 8)
        assert(sim.model.relu1.output_quantizers[0].data_type == QuantizationDataType.int)

        # at op level (for Conv) check param quantizers are updated to fp16 while output is still retained at int8
        assert(sim.model.conv1.param_quantizers['weight'].enabled == True)
        assert(sim.model.conv1.param_quantizers['weight'].bitwidth == 16)
        assert(sim.model.conv1.param_quantizers['weight'].data_type == QuantizationDataType.float)

        assert(sim.model.conv1.output_quantizers[0].bitwidth == 8)
        assert(sim.model.conv1.output_quantizers[0].data_type == QuantizationDataType.int)

        # remove test config created
        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_target_rule_enforced_apply_default_and_op_level_overrides_invalid_case(self):
        """
        Tests application of override config rule that is not valid.
        :return:
        """
        model = SingleResidual()
        model.eval()

        # quantsim config has default kernel overrides as well as op lebel kernel override
        # we begin with quantsim default config (int 4, int 4), during isntantiation of quantsim object.
        # Then, using config file we apply two levels of overrides.
        # 1) default supported_kernels at index 0 , is used to override default act/param bw dtype with int8 / int8
        # 2) After this, at op level, specifically for Conv types, there is a override provided as fp16/ fp16
        # So, param quantizers of conv shall be updated to FP16 as a override, while retaining output at int 8 as
        # configured by default level supported_kernels.

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "float"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "float"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 4,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 4,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    },
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 8,
                                    "dtype": "int"
                                },
                                "param": {
                                    "bitwidth": 8,
                                    "dtype": "int"
                                }
                            },
                        ]
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }

        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        INPUT_SHAPE = (1, 3, 32, 32)
        def forward_fn(model, _):
            torch.manual_seed(10)
            model.eval()
            with torch.no_grad():
                _ = model(torch.randn(INPUT_SHAPE))

        # set enforce to true for this test
        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True

        # enforce is set to true
        # default supported kernels at index DEFAULT_OVERRIDE_SUPPORTED_KERNEL_INDEX (=0 in this case)
        # is not same as default quantsim bw and dtype(int4/int4), apply default overrides (fp16/ fp16).
        # so, default qsim is created with fp16/fp16 as per default level supported_kernel at override index/
        # But, op level has a kernel that is lower precision (int 8,int8) as compared to this.
        # so, rule checker should flag and cause exception in this case.
        exception_raised = False
        try:
            sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf_enhanced,
                                       config_file='./data/quantsim_config.json',
                                       dummy_input=torch.rand(1, 3, 32, 32), in_place=True,
                                       default_data_type=QuantizationDataType.int, default_output_bw=4, default_param_bw=4)
        except NotImplementedError as exc:
            exception_raised = True
            print(" Test raised exception as expected ", exc)

        assert exception_raised

        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_target_rule_enforced_apply_op_level_overrides_fp16(self):
        """
        validates Config overrides provided are valid combination and application of aic100 specific rules.
        No dfeualt supported kernel override and op level FP16 support for LayerNorm and GeLU.
        Quantsim created with (int8, int8) defaults
        a) Default supported kernels override not provided.
        b) op level override at index 0 of supported_kernels for LayerNorm/GeLU type is
        (fp 16, fp16) --> applied to params.
        For GeLu, nothing is applied, as it has not params. output is retained at int 8.
        :return:
        """

        # aic100, no default supported kernels, op level for layernorm and gelu
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                }
            },
            "params": {},
            "op_type": {
                "LayerNorm": {
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                        ]
                },
                "GELU": {
                    "is_output_quantized": "True",
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                        ]
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }


        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        torch.manual_seed(10)
        model = ModelWithBertCustomLayerNormGelu()
        model.eval()

        random_input = torch.rand(1, 4, 4)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(*random_input)

        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True
        sim = QuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf,
                                   dummy_input=random_input, default_data_type=QuantizationDataType.int,
                                   default_output_bw=8, default_param_bw=8,
                                   config_file='./data/quantsim_config.json')


        # enforce is set to true
        # LayerNorm params should be set to FP 16, while output is maintained at quantsim defaults (int8)
        assert(sim.model.customln1.output_quantizer.data_type == QuantizationDataType.int)
        assert(sim.model.customln1.output_quantizer.bitwidth == 8)

        # override this with custom config (matches aic100_config.json)
        assert(sim.model.customln1.param_quantizers['weight'].data_type == QuantizationDataType.float)
        assert(sim.model.customln1.param_quantizers['weight'].bitwidth == 16)

        # gelu output should be retained at quantsim defaults (int8) although it has supported_kernels = FP16
        # as this op doesn't have params
        assert(sim.model.gelu1.output_quantizer.data_type == QuantizationDataType.int)
        assert(sim.model.gelu1.output_quantizer.bitwidth == 8)

        # remove test config created
        qsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')
