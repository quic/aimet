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
import libpymo
from aimet_common.defs import QuantScheme
from aimet_torch.examples.test_models import SingleResidual, QuantSimTinyModel, MultiInput, SingleResidualWithModuleAdd
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.quantsim_config.quantsim_config import _get_all_ops_in_neighborhood
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch import utils
from aimet_torch.meta.connectedgraph import ConnectedGraph


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
                    }
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
        starting_op = conn_graph.get_all_ops()['Conv_7']
        add_10_op = conn_graph.get_all_ops()['Add_10']
        adaptive_avg_pool2d_9_op = conn_graph.get_all_ops()['GlobalAveragePool_9']
        neighborhood = _get_all_ops_in_neighborhood(starting_op, 'output')
        assert len(neighborhood) == 3
        assert starting_op in neighborhood
        assert add_10_op in neighborhood
        assert adaptive_avg_pool2d_9_op in neighborhood

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
        from aimet_common.defs import QuantScheme
        from aimet_torch.quantsim import QuantizationSimModel
        import libpymo

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
                            }
                        }
                    },
                    "GELU": {
                        "is_input_quantized": "True"
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

