# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

import pytest
import tensorflow as tf
from tensorflow.keras.layers import InputLayer

from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_tensorflow.keras.connectedgraph import ConnectedGraph
from aimet_tensorflow.keras.quant_sim.tensor_quantizer import ParamPerChannelQuantizer, ParamPerTensorQuantizer
from aimet_tensorflow.keras.quantsim_config.quantsim_config import QuantSimConfigurator, INPUT_QUANTIZERS, \
    OUTPUT_QUANTIZERS, PARAM_QUANTIZERS
from test_models_keras import single_residual, concat_functional, tiny_conv_net


class TestQuantSimConfig:
    """
    Class containing unit tests for quantsim config feature
    """

    def test_mapping_layer_to_affected_quantizers_for_multi_input(self):
        """
        Test if layer and its affected quantizers are correctly mapped
        """
        model = concat_functional()
        connected_graph = ConnectedGraph(model)
        quant_sim_configurator = QuantSimConfigurator(connected_graph, QuantScheme.post_training_tf_enhanced,
                                                      "nearest", 8, 8)

        # layers excluding InputLayer
        layers = [x for x in model.layers if not isinstance(x, InputLayer)]
        dense1, dense2, dense3, concat1, dense4, dense5 = layers

        # Note:
        # 0 (Affected quantizers when enabling input quantizer of this layer)
        # 1 (Affected quantizers when enabling output quantizer of this layer)
        # 2 (Affected quantizers when disabling input quantizer of this layer)
        # 3 (Affected quantizers when disabling output quantizer of this layer)
        layer_to_affected_tensor_quantizers_dict = quant_sim_configurator._layer_to_affected_quantizer_info_dict

        # Input layer 1
        assert len(layer_to_affected_tensor_quantizers_dict[dense1][0]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[dense1][1]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[dense1][2]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[dense1][3]) == 2
        assert (dense1, "output") in layer_to_affected_tensor_quantizers_dict[dense1][3]
        assert (dense3, "input") in layer_to_affected_tensor_quantizers_dict[dense1][3]

        # Input layer 2
        assert len(layer_to_affected_tensor_quantizers_dict[dense2][0]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[dense2][1]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[dense2][2]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[dense2][3]) == 2
        assert (dense2, "output") in layer_to_affected_tensor_quantizers_dict[dense2][3]
        assert (concat1, "input") in layer_to_affected_tensor_quantizers_dict[dense2][3]

        # Layer having multiple producers (Concat layer)
        assert len(layer_to_affected_tensor_quantizers_dict[concat1][0]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[concat1][1]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[concat1][2]) == 3
        assert (concat1, "input") in layer_to_affected_tensor_quantizers_dict[concat1][2]
        assert (dense2, "output") in layer_to_affected_tensor_quantizers_dict[concat1][2]
        assert (dense3, "output") in layer_to_affected_tensor_quantizers_dict[concat1][2]
        assert len(layer_to_affected_tensor_quantizers_dict[concat1][3]) == 2

        # Last layer
        assert len(layer_to_affected_tensor_quantizers_dict[dense5][0]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[dense5][1]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[dense5][2]) == 2
        assert len(layer_to_affected_tensor_quantizers_dict[dense5][3]) == 1
        assert (dense5, "output") in layer_to_affected_tensor_quantizers_dict[dense5][3]

    def test_mapping_layer_to_affected_quantizers_for_single_residual(self):
        """
        Test if layer and its affected quantizers are correctly mapped
        """
        model = single_residual()
        connected_graph = ConnectedGraph(model)
        quant_sim_configurator = QuantSimConfigurator(connected_graph, QuantScheme.post_training_tf_enhanced,
                                                      "nearest", 8, 8, "")

        # layers excluding InputLayer
        layers = model.layers[1:]
        conv1, bn1, max_pool1, conv2, conv4 = layers[0], layers[1], layers[3], layers[4], layers[7]
        conv3, avg_pool1, add1, dense1 = layers[8], layers[9], layers[10], layers[14]

        # Note:
        # 0 (Affected quantizers when enabling input quantizer of this layer)
        # 1 (Affected quantizers when enabling output quantizer of this layer)
        # 2 (Affected quantizers when disabling input quantizer of this layer)
        # 3 (Affected quantizers when disabling output quantizer of this layer)
        layer_to_affected_tensor_quantizers_dict = quant_sim_configurator._layer_to_affected_quantizer_info_dict

        # First layer
        assert len(layer_to_affected_tensor_quantizers_dict[conv1][0]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[conv1][1]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[conv1][2]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[conv1][3]) == 2
        assert (conv1, "output") in layer_to_affected_tensor_quantizers_dict[conv1][3]
        assert (bn1, "input") in layer_to_affected_tensor_quantizers_dict[conv1][3]

        # Layer having multiple consumers (MaxPool layer)
        assert len(layer_to_affected_tensor_quantizers_dict[max_pool1][0]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[max_pool1][1]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[max_pool1][2]) == 2
        assert len(layer_to_affected_tensor_quantizers_dict[max_pool1][3]) == 3
        assert (max_pool1, "output") in layer_to_affected_tensor_quantizers_dict[max_pool1][3]
        assert (conv2, "input") in layer_to_affected_tensor_quantizers_dict[max_pool1][3]
        assert (conv4, "input") in layer_to_affected_tensor_quantizers_dict[max_pool1][3]

        # Layer having multiple producers (Add layer)
        assert len(layer_to_affected_tensor_quantizers_dict[add1][0]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[add1][1]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[add1][2]) == 3
        assert (add1, "input") in layer_to_affected_tensor_quantizers_dict[add1][2]
        assert (conv3, "output") in layer_to_affected_tensor_quantizers_dict[add1][2]
        assert (avg_pool1, "output") in layer_to_affected_tensor_quantizers_dict[add1][2]
        assert len(layer_to_affected_tensor_quantizers_dict[add1][3]) == 2

        # Last layer
        assert len(layer_to_affected_tensor_quantizers_dict[dense1][0]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[dense1][1]) == 1
        assert len(layer_to_affected_tensor_quantizers_dict[dense1][2]) == 2
        assert len(layer_to_affected_tensor_quantizers_dict[dense1][3]) == 1

    def test_parse_config_file_defaults(self):
        """
        Test that default quantization parameters are set correctly when using json config file
        """
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

        model = single_residual()
        connected_graph = ConnectedGraph(model)
        quant_sim_configurator = QuantSimConfigurator(connected_graph, QuantScheme.post_training_tf_enhanced,
                                                      "nearest", 8, 8, QuantizationDataType.int, "./data/quantsim_config.json")

        layer_to_quantizers_dict = quant_sim_configurator._layer_to_quantizers_dict
        for op in connected_graph.ordered_ops:
            layer = op.get_module()

            for q in layer_to_quantizers_dict[layer][INPUT_QUANTIZERS]:
                assert not q.is_enabled()
                assert not q.is_symmetric
                assert not q.use_unsigned_symmetric
                assert not q.use_strict_symmetric

            for q in layer_to_quantizers_dict[layer][OUTPUT_QUANTIZERS]:
                assert q.is_enabled()
                assert not q.is_symmetric
                assert not q.use_unsigned_symmetric
                assert not q.use_strict_symmetric

            for q in layer_to_quantizers_dict[layer][PARAM_QUANTIZERS]:
                assert not q.is_enabled()
                assert q.is_symmetric
                if isinstance(q._tensor_quantizer, list):
                    for tensor_quantizer in q._tensor_quantizer:
                        assert not tensor_quantizer.getUnsignedSymmetric()
                        assert not tensor_quantizer.getStrictSymmetric()
                else:
                    assert not q._tensor_quantizer.getUnsignedSymmetric()
                    assert not q._tensor_quantizer.getStrictSymmetric()

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_parse_config_file_symmetric_modes(self):
        """
        Test that model output quantization parameters are set correctly when using json config file
        """
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

        model = single_residual()
        connected_graph = ConnectedGraph(model)
        quant_sim_configurator = QuantSimConfigurator(connected_graph, QuantScheme.post_training_tf_enhanced,
                                                      "nearest", 8, 8, QuantizationDataType.int, "./data/quantsim_config.json")

        layer_to_quantizers_dict = quant_sim_configurator._layer_to_quantizers_dict
        for op in connected_graph.ordered_ops:
            layer = op.get_module()

            for q in layer_to_quantizers_dict[layer][INPUT_QUANTIZERS]:
                assert q.use_strict_symmetric
                assert not q.use_unsigned_symmetric

            for q in layer_to_quantizers_dict[layer][OUTPUT_QUANTIZERS]:
                assert q.use_strict_symmetric
                assert not q.use_unsigned_symmetric

            for q in layer_to_quantizers_dict[layer][PARAM_QUANTIZERS]:
                assert q.use_strict_symmetric
                assert not q.use_unsigned_symmetric

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_parse_config_file_params(self):
        """
        Test that param specific quantization parameters are set correctly when using json config file
        """
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
            "model_output": {
            }
        }
        with open("./data/quantsim_config.json", "w") as f:
            json.dump(quantsim_config, f)

        model = single_residual()
        connected_graph = ConnectedGraph(model)
        quant_sim_configurator = QuantSimConfigurator(connected_graph, QuantScheme.post_training_tf_enhanced,
                                                      "nearest", 8, 8, QuantizationDataType.int, "./data/quantsim_config.json")

        layer_to_quantizers_dict = quant_sim_configurator._layer_to_quantizers_dict
        for op in connected_graph.ordered_ops:
            layer = op.get_module()

            for q in layer_to_quantizers_dict[layer][PARAM_QUANTIZERS]:
                if "bias" in q.name:
                    assert not q.is_enabled()
                    assert q.is_symmetric
                else:
                    assert q.is_enabled()
                    assert not q.is_symmetric

        if os.path.exists("./data/quantsim_config.json"):
            os.remove("./data/quantsim_config.json")

    def test_parse_config_file_op_type(self):
        """
        Test that op specific quantization parameters are set correctly when using json config file
        """

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
            "model_output": {
            }
        }
        with open("./data/quantsim_config.json", "w") as f:
            json.dump(quantsim_config, f)

        model = single_residual()
        connected_graph = ConnectedGraph(model)
        quant_sim_configurator = QuantSimConfigurator(connected_graph, QuantScheme.post_training_tf_enhanced,
                                                      "nearest", 8, 8, QuantizationDataType.int, "./data/quantsim_config.json")

        layer_to_quantizers_dict = quant_sim_configurator._layer_to_quantizers_dict
        for op in connected_graph.ordered_ops:
            layer = op.get_module()

            for q in layer_to_quantizers_dict[layer][INPUT_QUANTIZERS]:
                if op.type == "Conv":
                    assert q.is_enabled()
                    assert not q.is_symmetric
                else:
                    assert not q.is_enabled()
                    assert not q.is_symmetric

            for q in layer_to_quantizers_dict[layer][OUTPUT_QUANTIZERS]:
                assert q.is_enabled()
                assert not q.is_symmetric

            for q in layer_to_quantizers_dict[layer][PARAM_QUANTIZERS]:
                if op.type == "Conv" and "bias" in q.name:
                    assert q.is_enabled()
                    assert not q.is_symmetric
                else:
                    assert not q.is_enabled()
                    assert q.is_symmetric

        if os.path.exists("./data/quantsim_config.json"):
            os.remove("./data/quantsim_config.json")

    def test_parse_config_file_op_type_conflict_case(self):
        """
        Test that op specific quantization parameters are set correctly when using json config file
        """

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {}
            },
            "params": {},
            "op_type": {
                "Conv": {
                    "is_input_quantized": "True",
                    "is_output_quantized": "False"
                },
                "BatchNormalization": {
                    "is_input_quantized": "True"
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        model = single_residual()
        connected_graph = ConnectedGraph(model)

        with pytest.raises(RuntimeError):
            _ = QuantSimConfigurator(connected_graph, QuantScheme.post_training_tf_enhanced, "nearest",
                                     8, 8, QuantizationDataType.int, "./data/quantsim_config.json")

        if os.path.exists("./data/quantsim_config.json"):
            os.remove("./data/quantsim_config.json")

    def test_parse_config_file_model_inputs(self):
        """
        Test that model input quantization parameters are set correctly when using json config file
        """
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
        with open("./data/quantsim_config.json", "w") as f:
            json.dump(quantsim_config, f)

        model = single_residual()
        connected_graph = ConnectedGraph(model)

        quant_sim_configurator = QuantSimConfigurator(connected_graph, QuantScheme.post_training_tf_enhanced,
                                                      "nearest", 8, 8, QuantizationDataType.int, "./data/quantsim_config.json")

        layer_to_quantizers_dict = quant_sim_configurator._layer_to_quantizers_dict
        conv1 = model.layers[1]
        for op in connected_graph.ordered_ops:
            layer = op.get_module()

            for q in layer_to_quantizers_dict[layer][INPUT_QUANTIZERS]:
                if layer == conv1:
                    assert q.is_enabled()
                else:
                    assert not q.is_enabled()
                assert not q.is_symmetric

            for q in layer_to_quantizers_dict[layer][OUTPUT_QUANTIZERS]:
                assert not q.is_enabled()
                assert not q.is_symmetric

            for q in layer_to_quantizers_dict[layer][PARAM_QUANTIZERS]:
                assert not q.is_enabled()
                assert not q.is_symmetric

        if os.path.exists("./data/quantsim_config.json"):
            os.remove("./data/quantsim_config.json")

    def test_parse_config_file_supergroups(self):
        """
        Test that supergroup quantization parameters are set correctly when using json config file
        """
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
        with open("./data/quantsim_config.json", "w") as f:
            json.dump(quantsim_config, f)

        model = tiny_conv_net()
        connected_graph = ConnectedGraph(model)

        quant_sim_configurator = QuantSimConfigurator(connected_graph, QuantScheme.post_training_tf_enhanced,
                                                      "nearest", 8, 8, QuantizationDataType.int, "./data/quantsim_config.json")

        layer_to_quantizers_dict = quant_sim_configurator._layer_to_quantizers_dict
        conv1, relu1, conv2, conv3 = model.layers[1], model.layers[3], model.layers[5], model.layers[8]
        relu3 = model.layers[9]
        bn1, maxpool, bn2, avgpool = model.layers[2], model.layers[4], model.layers[6], model.layers[10]
        for op in connected_graph.ordered_ops:
            layer = op.get_module()

            # Check configs for starts of supergroups
            if layer in [conv1, relu1, conv2, conv3]:
                for q in layer_to_quantizers_dict[layer][OUTPUT_QUANTIZERS]:
                    assert not q.is_enabled()
            # Check configs for middle ops in supergroups
            elif layer == relu3:
                for q in layer_to_quantizers_dict[layer][INPUT_QUANTIZERS]:
                    assert not q.is_enabled()
                for q in layer_to_quantizers_dict[layer][OUTPUT_QUANTIZERS]:
                    assert not q.is_enabled()
            # Check configs for ends of supergroups
            elif layer in [bn1, maxpool, bn2, avgpool]:
                for q in layer_to_quantizers_dict[layer][INPUT_QUANTIZERS]:
                    assert not q.is_enabled()
                for q in layer_to_quantizers_dict[layer][OUTPUT_QUANTIZERS]:
                    assert q.is_enabled()
            else:
                for q in layer_to_quantizers_dict[layer][INPUT_QUANTIZERS]:
                    assert not q.is_enabled()
                for q in layer_to_quantizers_dict[layer][OUTPUT_QUANTIZERS]:
                    assert q.is_enabled()

        if os.path.exists("./data/quantsim_config.json"):
            os.remove("./data/quantsim_config.json")

    def test_parse_config_file_model_outputs(self):
        """
        Test that model output quantization parameters are set correctly when using json config file
        """
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
        with open("./data/quantsim_config.json", "w") as f:
            json.dump(quantsim_config, f)

        model = single_residual()
        connected_graph = ConnectedGraph(model)

        quant_sim_configurator = QuantSimConfigurator(connected_graph, QuantScheme.post_training_tf_enhanced,
                                                      "nearest", 8, 8, QuantizationDataType.int, "./data/quantsim_config.json")

        layer_to_quantizers_dict = quant_sim_configurator._layer_to_quantizers_dict
        fc = model.layers[-1]
        for op in connected_graph.ordered_ops:
            layer = op.get_module()

            for q in layer_to_quantizers_dict[layer][INPUT_QUANTIZERS]:
                assert not q.is_enabled()
                assert not q.is_symmetric

            for q in layer_to_quantizers_dict[layer][OUTPUT_QUANTIZERS]:
                if layer == fc:
                    assert q.is_enabled()
                else:
                    assert not q.is_enabled()
                assert not q.is_symmetric

            for q in layer_to_quantizers_dict[layer][PARAM_QUANTIZERS]:
                assert not q.is_enabled()
                assert not q.is_symmetric

        if os.path.exists("./data/quantsim_config.json"):
            os.remove("./data/quantsim_config.json")

    def test_parse_op_type_per_channel_quantization(self) -> None:
        """
        Test when specific op type has per_channel_quantization property in json config file
        """
        quantsim_config = {
            "defaults": {
                "ops": { "is_output_quantized": "True" },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                },
                "strict_symmetric": "False",
                "unsigned_symmetric": "True",
                "per_channel_quantization": "True"
            },
            "params": {
                "bias": { "is_quantized": "False" }
            },
            "op_type": {
                "Gemm": { "per_channel_quantization": "False" }
            },
            "supergroups": [],
            "model_input": { "is_input_quantized": "True" },
            "model_output": {}
        }
        with open("./data/quantsim_config.json", "w") as f:
            json.dump(quantsim_config, f)

        model = tiny_conv_net()
        connected_graph = ConnectedGraph(model)
        quant_sim_configurator = QuantSimConfigurator(connected_graph, QuantScheme.post_training_tf_enhanced,
                                                      "nearest", 8, 8, QuantizationDataType.int, "./data/quantsim_config.json")

        layer_to_quantizers_dict = quant_sim_configurator._layer_to_quantizers_dict
        for layer, quantizers in layer_to_quantizers_dict.items():
            if isinstance(layer, tf.keras.layers.Conv2D):
                for q in layer_to_quantizers_dict[layer][PARAM_QUANTIZERS]:
                    assert isinstance(q, ParamPerChannelQuantizer)

            if isinstance(layer, tf.keras.layers.Dense):
                for q in layer_to_quantizers_dict[layer][PARAM_QUANTIZERS]:
                    assert isinstance(q, ParamPerTensorQuantizer)

        if os.path.exists("./data/quantsim_config.json"):
            os.remove("./data/quantsim_config.json")
