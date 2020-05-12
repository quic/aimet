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

import json
import os
import unittest
import tensorflow as tf
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops
from aimet_tensorflow.examples.test_models import single_residual
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.common.connectedgraph import ConnectedGraph

import libpymo as pymo


# pylint: disable=protected-access
# pylint: disable=too-many-locals
class TestQuantsimConfig(unittest.TestCase):
    """ Class containing unit tests for quantsim config feature """

    def test_empty_config_file(self):
        """ Check that with an empty config file, all op modes and use symmetric encoding settings are set to
        passThrough and False respectively. """
        tf.reset_default_graph()
        sess = tf.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.global_variables_initializer()
            sess.run(init)

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with open('./quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(sess, ['input_1'], ['single_residual/Softmax'],
                                   config_file='./quantsim_config.json')
        all_quantize_ops = [op for op in sim.session.graph.get_operations() if op.type == 'QcQuantize']
        self.assertTrue(all_quantize_ops is not None)
        for op in all_quantize_ops:
            is_symmetric_tensor = sim.session.graph.get_tensor_by_name(op.name + '_use_symmetric_encoding:0')
            op_mode_tensor = sim.session.graph.get_tensor_by_name(op.name + '_op_mode:0')
            self.assertEqual(sim.session.run(is_symmetric_tensor), False)
            self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.passThrough))
        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.reset_default_graph()

    def test_parse_config_file_defaults(self):
        """ Test that default quantization parameters are set correctly when using json config file """
        tf.reset_default_graph()
        sess = tf.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.global_variables_initializer()
            sess.run(init)

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                }
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with open('./quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(sess, ['input_1'], ['single_residual/Softmax'],
                                   config_file='./quantsim_config.json')
        for op, (param_quantizer_dict, output_quantizer) in sim._op_to_quant_ops_dict.items():
            for param_quantizer_op_set in param_quantizer_dict.values():
                for op in param_quantizer_op_set:
                    is_symmetric_tensor = sim.session.graph.get_tensor_by_name(op.name + '_use_symmetric_encoding:0')
                    op_mode_tensor = sim.session.graph.get_tensor_by_name(op.name + '_op_mode:0')
                    self.assertEqual(sim.session.run(is_symmetric_tensor), True)
                    self.assertEqual(sim.session.run(op_mode_tensor),
                                     int(pymo.TensorQuantizerOpMode.oneShotQuantizeDequantize))
            op_mode_tensor = sim.session.graph.get_tensor_by_name(output_quantizer.name + '_op_mode:0')
            if op.name == 'input_1':
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.passThrough))
            else:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))
        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.reset_default_graph()

    def test_parse_config_file_params(self):
        """ Test that param specific quantization parameters are set correctly when using json config file """
        tf.reset_default_graph()
        sess = tf.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.global_variables_initializer()
            sess.run(init)

        quantsim_config = {
            "defaults": {
                "ops": {},
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
        with open('./quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(sess, ['input_1'], ['single_residual/Softmax'],
                                   config_file='./quantsim_config.json')

        saw_weight_op = False
        for param_quantizer_dict, _ in sim._op_to_quant_ops_dict.values():
            for param_name, op_set in param_quantizer_dict.items():
                for op in op_set:
                    is_symmetric_tensor = sim.session.graph.get_tensor_by_name(op.name + '_use_symmetric_encoding:0')
                    op_mode_tensor = sim.session.graph.get_tensor_by_name(op.name + '_op_mode:0')
                    if param_name == 'weight':
                        self.assertEqual(sim.session.run(op_mode_tensor),
                                         int(pymo.TensorQuantizerOpMode.oneShotQuantizeDequantize))
                        self.assertEqual(sim.session.run(is_symmetric_tensor), False)
                        saw_weight_op = True
                    else:
                        self.assertEqual(sim.session.run(op_mode_tensor),
                                         int(pymo.TensorQuantizerOpMode.passThrough))
                        self.assertEqual(sim.session.run(is_symmetric_tensor), True)
        # Make sure we found at least one weight op
        self.assertTrue(saw_weight_op)
        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.reset_default_graph()

    def test_parse_config_file_op_type(self):
        """ Test that op specific quantization parameters are set correctly when using json config file """
        tf.reset_default_graph()
        sess = tf.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.global_variables_initializer()
            sess.run(init)

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {}
            },
            "params": {},
            "op_type": {
                "Conv": {
                    "is_input_quantized": "True",
                    "params": {
                        "bias": {
                            "is_quantized": "True",
                            "is_symmetric": "True"
                        }
                    }
                },
                "Gemm": {
                    "is_input_quantized": "True",
                    "params": {
                        "bias": {
                            "is_quantized": "True",
                            "is_symmetric": "True"
                        }
                    }
                },
                "BatchNormalization": {
                    "is_input_quantized": "True"
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with open('./quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(sess, ['input_1'], ['single_residual/Softmax'],
                                   config_file='./quantsim_config.json')

        saw_conv_op = False
        saw_fused_bn_op = False
        saw_dense_op = False
        for op, (param_quantizer_dict, _) in sim._op_to_quant_ops_dict.items():
            if op.type in ['Conv2D', 'Dense']:
                bias_quantize_op_set = param_quantizer_dict['bias']
                for bias_quantize_op in bias_quantize_op_set:
                    bias_is_symmetric_tensor = sim.session.graph.get_tensor_by_name(bias_quantize_op.name +
                                                                                    '_use_symmetric_encoding:0')
                    bias_op_mode_tensor = sim.session.graph.get_tensor_by_name(bias_quantize_op.name + '_op_mode:0')
                    self.assertEqual(sim.session.run(bias_op_mode_tensor),
                                     int(pymo.TensorQuantizerOpMode.oneShotQuantizeDequantize))
                    self.assertEqual(sim.session.run(bias_is_symmetric_tensor), True)

                parent_op = op.inputs[0].producer
                while parent_op not in sim._op_to_quant_ops_dict:
                    parent_op = parent_op.inputs[0].producer
                _, parent_quantize_op = sim._op_to_quant_ops_dict[parent_op]

                op_mode_tensor = sim.session.graph.get_tensor_by_name(parent_quantize_op.name + '_op_mode:0')
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))

                if op.type == 'Conv2D':
                    saw_conv_op = True
                else:
                    saw_dense_op = True
            if op.type == 'FusedBatchNormV3':
                parent_op = op.inputs[0].producer
                while parent_op not in sim._op_to_quant_ops_dict:
                    parent_op = parent_op.inputs[0].producer
                _, parent_quantize_op = sim._op_to_quant_ops_dict[parent_op]

                op_mode_tensor = sim.session.graph.get_tensor_by_name(parent_quantize_op.name + '_op_mode:0')
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))
                saw_fused_bn_op = True
        self.assertTrue(saw_fused_bn_op and saw_dense_op and saw_conv_op)

        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.reset_default_graph()

    def test_parse_config_file_supergroups(self):
        """ Test that supergroup quantization parameters are set correctly when using json config file """
        tf.reset_default_graph()
        sess = tf.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.global_variables_initializer()
            sess.run(init)

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
                    "op_list": ["Conv", "AveragePool"]
                },
                {
                    "op_list": ["Add", "Relu"]
                },
                {
                    "op_list": ["Conv", "BatchNormalization"]
                }
            ],
            "model_input": {},
            "model_output": {}
        }
        with open('./quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        conn_graph = ConnectedGraph(sess.graph, ['input_1'], ['single_residual/Softmax'])
        sim = QuantizationSimModel(sess, ['input_1'], ['single_residual/Softmax'],
                                   config_file='./quantsim_config.json')
        ops_with_deactivated_output_quantizers = set()
        num_deactivated_quantizers = 0
        for op in conn_graph.get_all_ops().values():
            if op.type == 'Conv2D' and op.output.consumers[0].type == 'FusedBatchNormV3':
                ops_with_deactivated_output_quantizers.add(op)
                num_deactivated_quantizers += 1
            elif op.type == 'Add' and op.output.consumers[0].type == 'Relu':
                ops_with_deactivated_output_quantizers.add(op)
                num_deactivated_quantizers += 1
            elif op.type == 'Conv2D' and op.output.consumers[0].type == 'AvgPool':
                ops_with_deactivated_output_quantizers.add(op)
                num_deactivated_quantizers += 1
            elif op in get_all_input_ops(conn_graph):
                ops_with_deactivated_output_quantizers.add(op)
                num_deactivated_quantizers += 1
        for op, (_, output_quantizer) in sim._op_to_quant_ops_dict.items():
            op_mode_tensor = sim.session.graph.get_tensor_by_name(output_quantizer.name + '_op_mode:0')
            ops_with_deactivated_output_quantizers_names = [op.name for op in ops_with_deactivated_output_quantizers]
            if op.name in ops_with_deactivated_output_quantizers_names:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.passThrough))
            else:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))
        self.assertEqual(5, num_deactivated_quantizers)

        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.reset_default_graph()

    def test_parse_config_file_model_inputs(self):
        """ Test that model input quantization parameters are set correctly when using json config file """
        tf.reset_default_graph()
        sess = tf.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.global_variables_initializer()
            sess.run(init)

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
        with open('./quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(sess, ['input_1'], ['single_residual/Softmax'],
                                   config_file='./quantsim_config.json')

        for op in sim._op_to_quant_ops_dict.keys():
            if op.name == 'input_1':
                _, input_1_quantize_op = sim._op_to_quant_ops_dict[op]
        op_mode_tensor = sim.session.graph.get_tensor_by_name(input_1_quantize_op.name + '_op_mode:0')
        self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))

        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.reset_default_graph()

    def test_parse_config_file_model_outputs(self):
        """ Test that model output quantization parameters are set correctly when using json config file """
        tf.reset_default_graph()
        sess = tf.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.global_variables_initializer()
            sess.run(init)

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
        with open('./quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(sess, ['input_1'], ['single_residual/Softmax'],
                                   config_file='./quantsim_config.json')

        _, softmax_quantize_op = [(sim._op_to_quant_ops_dict[op]) for op in sim._op_to_quant_ops_dict.keys()
             if op.name == 'single_residual/Softmax'][0]
        op_mode_tensor = sim.session.graph.get_tensor_by_name(softmax_quantize_op.name + '_op_mode:0')
        self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))

        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.reset_default_graph()
