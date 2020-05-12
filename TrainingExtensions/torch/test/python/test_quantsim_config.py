# /usr/bin/env python2.7
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
import unittest
import json
import os
import torch

from aimet_torch.examples.test_models import SingleResidual, TinyModel, MultiInput
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.quantsim_config.quantsim_config import _get_all_ops_in_neighborhood
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch import utils
from aimet_torch.meta.connectedgraph import ConnectedGraph


# pylint: disable=protected-access
class TestQuantsimConfig(unittest.TestCase):
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

        sim = QuantizationSimModel(model, quant_scheme='tf_enhanced', config_file='./data/quantsim_config.json',
                                   input_shapes=(1, 3, 32, 32), in_place=True)
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # Output of add op is input quantized
                if name == 'relu3':
                    self.assertTrue(module.input_quantizer.enabled)
                else:
                    self.assertTrue(not module.input_quantizer.enabled)
                self.assertTrue(module.output_quantizer.enabled)
                self.assertTrue(not module.input_quantizer.use_symmetric_encodings)
                self.assertTrue(not module.output_quantizer.use_symmetric_encodings)
                if module.param_quantizers:
                    for _, param_quantizer in module.param_quantizers.items():
                        self.assertTrue(not param_quantizer.enabled)
                        self.assertTrue(param_quantizer.use_symmetric_encodings)

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
        sim = QuantizationSimModel(model, quant_scheme='tf_enhanced', config_file='./data/quantsim_config.json',
                                   input_shapes=(1, 3, 32, 32))
        for _, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if module.param_quantizers:
                    for param_name, param_quantizer in module.param_quantizers.items():
                        if param_name == 'weight':
                            self.assertTrue(param_quantizer.enabled)
                            self.assertTrue(not param_quantizer.use_symmetric_encodings)
                        else:
                            self.assertTrue(not param_quantizer.enabled)
                            self.assertTrue(param_quantizer.use_symmetric_encodings)
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
        sim = QuantizationSimModel(model, quant_scheme='tf_enhanced', config_file='./data/quantsim_config.json',
                                   input_shapes=(1, 3, 32, 32))
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if isinstance(module._module_to_wrap, torch.nn.Conv2d):
                    self.assertTrue(module.input_quantizer.enabled)
                    self.assertTrue(not module.input_quantizer.use_symmetric_encodings)
                    self.assertTrue(not module.output_quantizer.use_symmetric_encodings)
                else:
                    # Output of add op is input quantized
                    if name == 'relu3':
                        self.assertTrue(module.input_quantizer.enabled)
                    else:
                        self.assertTrue(not module.input_quantizer.enabled)
                    self.assertTrue(module.output_quantizer.enabled)
                    self.assertTrue(not module.input_quantizer.use_symmetric_encodings)
                    self.assertTrue(not module.output_quantizer.use_symmetric_encodings)
                if module.param_quantizers:
                    for param_name, param_quantizer in module.param_quantizers.items():
                        if isinstance(module._module_to_wrap, torch.nn.Conv2d) and param_name == 'bias':
                            self.assertTrue(param_quantizer.enabled)
                            self.assertTrue(not param_quantizer.use_symmetric_encodings)
                        else:
                            self.assertTrue(not param_quantizer.enabled)
                            self.assertTrue(param_quantizer.use_symmetric_encodings)
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_parse_config_file_supergroups(self):
        """ Test that supergroup quantization parameters are set correctly when using json config file """
        model = TinyModel()
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
                }
            ],
            "model_input": {},
            "model_output": {}
        }
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        # Use in_place=True here for easy access to modules through model instance variables
        sim = QuantizationSimModel(model, quant_scheme='tf_enhanced', config_file='./data/quantsim_config.json',
                                   in_place=True, input_shapes=(1, 3, 32, 32))
        for _, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # Check configs for starts of supergroups
                if module in [model.conv1, model.relu1, model.conv2, model.conv3]:
                    self.assertTrue(not module.output_quantizer.enabled)
                # Check configs for middle ops in supergroups
                elif module == model.relu3:
                    self.assertTrue(not module.input_quantizer.enabled)
                    self.assertTrue(not module.output_quantizer.enabled)
                # Check configs for ends of supergroups
                elif module in [model.bn1, model.maxpool, model.bn2, model.avgpool]:
                    self.assertTrue(not module.input_quantizer.enabled)
                    self.assertTrue(module.output_quantizer.enabled)
                else:
                    self.assertTrue(not module.input_quantizer.enabled)
                    self.assertTrue(module.output_quantizer.enabled)

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
        sim = QuantizationSimModel(model, quant_scheme='tf_enhanced', config_file='./data/quantsim_config.json',
                                   input_shapes=(1, 3, 32, 32))
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if name in ['conv3', 'ada']:
                    # model.conv3 and model.ada are inputs to add
                    self.assertTrue(module.output_quantizer.enabled)
                else:
                    self.assertTrue(not module.output_quantizer.enabled)
                self.assertTrue(not module.input_quantizer.enabled)
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

        sim = QuantizationSimModel(model, quant_scheme='tf_enhanced', config_file='./data/quantsim_config.json',
                                   input_shapes=[(1, 3, 32, 32), (1, 3, 20, 20)], in_place=True)
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # Output of add op is input quantized
                if name in ('conv1', 'conv3'):
                    self.assertTrue(module.input_quantizer.enabled)
                else:
                    self.assertTrue(not module.input_quantizer.enabled)
                self.assertTrue(not module.output_quantizer.enabled)
                self.assertTrue(not module.input_quantizer.use_symmetric_encodings)
                self.assertTrue(not module.output_quantizer.use_symmetric_encodings)

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
        sim = QuantizationSimModel(model, quant_scheme='tf_enhanced', config_file='./data/quantsim_config.json',
                                   input_shapes=(1, 3, 32, 32))
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                if name == 'fc':
                    # model.conv3 and model.ada are inputs to add
                    self.assertTrue(module.output_quantizer.enabled)
                else:
                    self.assertTrue(not module.output_quantizer.enabled)
                self.assertTrue(not module.input_quantizer.enabled)
        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_supergroups_with_elementwise_add(self):
        """ Test that supergroup quantization parameters are set correctly when using json config file """
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
        sim = QuantizationSimModel(model, quant_scheme='tf_enhanced', config_file='./data/quantsim_config.json',
                                   in_place=True, input_shapes=(1, 3, 32, 32))
        for _, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # Check configs for starts of supergroups
                if module == model.relu3:
                    # If add were not part of the supergroup, relu's input quantizer would be enabled
                    self.assertTrue(not module.input_quantizer.enabled)

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_get_all_ops_in_neighborhood(self):
        """ Test that default quantization parameters are set correctly when using json config file """
        model = SingleResidual()
        model.eval()
        input_shapes = (1, 3, 32, 32)

        random_inputs = utils.create_rand_tensors_given_shapes(input_shapes)
        conn_graph = ConnectedGraph(model, random_inputs)
        starting_op = conn_graph.get_all_ops()['convolution_7']
        add_10_op = conn_graph.get_all_ops()['add_10']
        adaptive_avg_pool2d_9_op = conn_graph.get_all_ops()['adaptive_avg_pool2d_9']
        neighborhood = _get_all_ops_in_neighborhood(starting_op, 'output')
        self.assertEqual(3, len(neighborhood))
        self.assertTrue(starting_op in neighborhood)
        self.assertTrue(add_10_op in neighborhood)
        self.assertTrue(adaptive_avg_pool2d_9_op in neighborhood)


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

        sim = QuantizationSimModel(model, quant_scheme='tf_enhanced', config_file='./data/quantsim_config.json',
                                   input_shapes=(1, 3, 32, 32), in_place=True)
        for name, module in sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # Output of add op is input quantized
                if name == 'relu3':
                    self.assertTrue(module.input_quantizer.enabled)
                else:
                    self.assertTrue(not module.input_quantizer.enabled)
                self.assertTrue(module.output_quantizer.enabled)
                self.assertTrue(not module.input_quantizer.use_symmetric_encodings)
                self.assertTrue(not module.output_quantizer.use_symmetric_encodings)
                if module.param_quantizers:
                    for _, param_quantizer in module.param_quantizers.items():
                        self.assertTrue(not param_quantizer.enabled)
                        self.assertTrue(param_quantizer.use_symmetric_encodings)

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')
