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
import pytest
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import unittest
import tensorflow as tf
from aimet_tensorflow.examples.test_models import single_residual, single_residual_for_tf2
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils.constants import QuantizeOpIndices
from aimet_tensorflow.quantsim_config.quantsim_config import QuantSimConfigurator
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_common.defs import QuantizationDataType, QuantDtypeBwInfo
import libpymo as pymo

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()

# pylint: disable=protected-access
# pylint: disable=too-many-locals
class TestQuantsimConfig(unittest.TestCase):
    """ Class containing unit tests for quantsim config feature """

    def test_empty_config_file(self):
        """ Check that with an empty config file, all op modes and use symmetric encoding settings are set to
        passThrough and False respectively. """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.compat.v1.global_variables_initializer()
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
            is_symmetric_tensor = op.inputs[QuantizeOpIndices.use_symmetric_encoding]
            op_mode_tensor = op.inputs[QuantizeOpIndices.op_mode]
            self.assertEqual(sim.session.run(is_symmetric_tensor), False)
            self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.passThrough))
        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.compat.v1.reset_default_graph()

    @pytest.mark.tf1
    def test_parse_config_file_defaults(self):
        """ Test that default quantization parameters are set correctly when using json config file """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.compat.v1.global_variables_initializer()
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

        activation_quantizers = [
            'conv2d/BiasAdd_quantized',
            'conv2d_1/BiasAdd_quantized',
            'conv2d_2/BiasAdd_quantized',
            'conv2d_3/BiasAdd_quantized',
            'conv2d_4/BiasAdd_quantized',
            'input_1_quantized',
            'batch_normalization/cond/Merge_quantized',
            'Relu_quantized',
            'max_pooling2d/MaxPool_quantized',
            'batch_normalization_1/cond/Merge_quantized',
            'Add_quantized',
            'Relu_2_quantized',
            'average_pooling2d/AvgPool_quantized',
            'single_residual/Softmax_quantized',
            'Relu_1_quantized'
        ]

        weight_quantizers = [
            'conv2d/Conv2D/ReadVariableOp_quantized',
            'conv2d_1/Conv2D/ReadVariableOp_quantized',
            'conv2d_2/Conv2D/ReadVariableOp_quantized',
            'conv2d_3/Conv2D/ReadVariableOp_quantized',
            'conv2d_4/Conv2D/ReadVariableOp_quantized',
            'single_residual/MatMul/ReadVariableOp_quantized',
            'conv2d/BiasAdd/ReadVariableOp_quantized',
            'conv2d_1/BiasAdd/ReadVariableOp_quantized',
            'conv2d_2/BiasAdd/ReadVariableOp_quantized',
            'conv2d_3/BiasAdd/ReadVariableOp_quantized',
            'conv2d_4/BiasAdd/ReadVariableOp_quantized',
            'single_residual/BiasAdd/ReadVariableOp_quantized'
        ]

        for op_name in weight_quantizers:
            op = sim.session.graph.get_operation_by_name(op_name)
            is_symmetric_tensor = op.inputs[QuantizeOpIndices.use_symmetric_encoding]
            op_mode_tensor = op.inputs[QuantizeOpIndices.op_mode]
            self.assertEqual(sim.session.run(is_symmetric_tensor), True)
            self.assertEqual(sim.session.run(op_mode_tensor),
                             int(pymo.TensorQuantizerOpMode.oneShotQuantizeDequantize))

        for op_name in activation_quantizers:
            op = sim.session.graph.get_operation_by_name(op_name)
            is_symmetric_tensor = op.inputs[QuantizeOpIndices.use_symmetric_encoding]
            op_mode_tensor = op.inputs[QuantizeOpIndices.op_mode]
            if 'input_1' in op_name:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.passThrough))
            else:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))
            self.assertEqual(sim.session.run(is_symmetric_tensor), False)

        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.compat.v1.reset_default_graph()

    @pytest.mark.tf2
    def test_parse_config_file_defaults_tf2(self):
        """ Test that default quantization parameters are set correctly when using json config file """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            _ = single_residual_for_tf2()
            init = tf.compat.v1.global_variables_initializer()
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

        activation_quantizers = [
            'conv2d/BiasAdd_quantized',
            'conv2d_1/BiasAdd_quantized',
            'conv2d_2/BiasAdd_quantized',
            'conv2d_3/BiasAdd_quantized',
            'conv2d_4/BiasAdd_quantized',
            'input_1_quantized',
            'batch_normalization/FusedBatchNormV3_quantized',
            'Relu_quantized',
            'max_pooling2d/MaxPool_quantized',
            'batch_normalization_1/FusedBatchNormV3_quantized',
            'Add_quantized',
            'Relu_2_quantized',
            'average_pooling2d/AvgPool_quantized',
            'single_residual/Softmax_quantized',
            'Relu_1_quantized'
        ]

        weight_quantizers = [
            'conv2d/Conv2D/ReadVariableOp_quantized',
            'conv2d_1/Conv2D/ReadVariableOp_quantized',
            'conv2d_2/Conv2D/ReadVariableOp_quantized',
            'conv2d_3/Conv2D/ReadVariableOp_quantized',
            'conv2d_4/Conv2D/ReadVariableOp_quantized',
            'single_residual/MatMul/ReadVariableOp_quantized',
            'conv2d/BiasAdd/ReadVariableOp_quantized',
            'conv2d_1/BiasAdd/ReadVariableOp_quantized',
            'conv2d_2/BiasAdd/ReadVariableOp_quantized',
            'conv2d_3/BiasAdd/ReadVariableOp_quantized',
            'conv2d_4/BiasAdd/ReadVariableOp_quantized',
            'single_residual/BiasAdd/ReadVariableOp_quantized'
        ]

        for op_name in weight_quantizers:
            op = sim.session.graph.get_operation_by_name(op_name)
            is_symmetric_tensor = op.inputs[QuantizeOpIndices.use_symmetric_encoding]
            op_mode_tensor = op.inputs[QuantizeOpIndices.op_mode]
            self.assertEqual(sim.session.run(is_symmetric_tensor), True)
            self.assertEqual(sim.session.run(op_mode_tensor),
                             int(pymo.TensorQuantizerOpMode.oneShotQuantizeDequantize))

        for op_name in activation_quantizers:
            op = sim.session.graph.get_operation_by_name(op_name)
            is_symmetric_tensor = op.inputs[QuantizeOpIndices.use_symmetric_encoding]
            op_mode_tensor = op.inputs[QuantizeOpIndices.op_mode]
            if 'input_1' in op_name:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.passThrough))
            else:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))
            self.assertEqual(sim.session.run(is_symmetric_tensor), False)

        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.compat.v1.reset_default_graph()

    def test_parse_config_file_params(self):
        """ Test that param specific quantization parameters are set correctly when using json config file """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.compat.v1.global_variables_initializer()
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

        weight_quantizers = [
            'conv2d/Conv2D/ReadVariableOp_quantized',
            'conv2d_1/Conv2D/ReadVariableOp_quantized',
            'conv2d_2/Conv2D/ReadVariableOp_quantized',
            'conv2d_3/Conv2D/ReadVariableOp_quantized',
            'conv2d_4/Conv2D/ReadVariableOp_quantized',
            'single_residual/MatMul/ReadVariableOp_quantized'
        ]

        bias_quantizers = [
            'conv2d/BiasAdd/ReadVariableOp_quantized',
            'conv2d_1/BiasAdd/ReadVariableOp_quantized',
            'conv2d_2/BiasAdd/ReadVariableOp_quantized',
            'conv2d_3/BiasAdd/ReadVariableOp_quantized',
            'conv2d_4/BiasAdd/ReadVariableOp_quantized',
            'single_residual/BiasAdd/ReadVariableOp_quantized'
        ]

        for param_quantizer in weight_quantizers:
            op = sim.session.graph.get_operation_by_name(param_quantizer)
            is_symmetric_tensor = op.inputs[QuantizeOpIndices.use_symmetric_encoding]
            op_mode_tensor = op.inputs[QuantizeOpIndices.op_mode]
            self.assertEqual(sim.session.run(op_mode_tensor),
                             int(pymo.TensorQuantizerOpMode.oneShotQuantizeDequantize))
            self.assertEqual(sim.session.run(is_symmetric_tensor), False)
        for param_quantizer in bias_quantizers:
            op = sim.session.graph.get_operation_by_name(param_quantizer)
            is_symmetric_tensor = op.inputs[QuantizeOpIndices.use_symmetric_encoding]
            op_mode_tensor = op.inputs[QuantizeOpIndices.op_mode]
            self.assertEqual(sim.session.run(op_mode_tensor),
                             int(pymo.TensorQuantizerOpMode.passThrough))
            self.assertEqual(sim.session.run(is_symmetric_tensor), True)

        sess.close()
        sim.session.close()
        tf.compat.v1.reset_default_graph()

    @pytest.mark.tf1
    def test_parse_config_file_op_type(self):
        """ Test that op specific quantization parameters are set correctly when using json config file """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {},
                "strict_symmetric": "True",
                "unsigned_symmetric": "True"
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
        for q_config in sim._param_quantizers.values():
            self.assertTrue(q_config.use_strict_symmetric)
            self.assertTrue(q_config.use_unsigned_symmetric)

        for q_config in sim._activation_quantizers.values():
            self.assertTrue(q_config.use_strict_symmetric)
            self.assertTrue(q_config.use_unsigned_symmetric)

        activation_quantizers = [
            'conv2d/BiasAdd_quantized',
            'conv2d_1/BiasAdd_quantized',
            'conv2d_2/BiasAdd_quantized',
            'conv2d_3/BiasAdd_quantized',
            'conv2d_4/BiasAdd_quantized',
            'input_1_quantized',
            'batch_normalization/cond/Merge_quantized',
            'Relu_quantized',
            'max_pooling2d/MaxPool_quantized',
            'batch_normalization_1/cond/Merge_quantized',
            'Add_quantized',
            'Relu_2_quantized',
            'average_pooling2d/AvgPool_quantized',
            'single_residual/Softmax_quantized',
            'Relu_1_quantized'
        ]

        weight_quantizers = [
            'conv2d/Conv2D/ReadVariableOp_quantized',
            'conv2d_1/Conv2D/ReadVariableOp_quantized',
            'conv2d_2/Conv2D/ReadVariableOp_quantized',
            'conv2d_3/Conv2D/ReadVariableOp_quantized',
            'conv2d_4/Conv2D/ReadVariableOp_quantized',
            'single_residual/MatMul/ReadVariableOp_quantized',
            'conv2d/BiasAdd/ReadVariableOp_quantized',
            'conv2d_1/BiasAdd/ReadVariableOp_quantized',
            'conv2d_2/BiasAdd/ReadVariableOp_quantized',
            'conv2d_3/BiasAdd/ReadVariableOp_quantized',
            'conv2d_4/BiasAdd/ReadVariableOp_quantized',
            'single_residual/BiasAdd/ReadVariableOp_quantized'
        ]

        for activation_quantizer in activation_quantizers:
            op_mode_tensor = sim.session.graph.get_tensor_by_name(activation_quantizer + '_op_mode:0')
            if activation_quantizer in ['input_1_quantized',
                                        'conv2d/BiasAdd_quantized',
                                        'max_pooling2d/MaxPool_quantized',
                                        'conv2d_2/BiasAdd_quantized',
                                        'conv2d_3/BiasAdd_quantized',
                                        'Relu_2_quantized',
                                        'average_pooling2d/AvgPool_quantized']:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))
            else:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.passThrough))
        for weight_quantizer in weight_quantizers:
            is_symmetric_tensor = sim.session.graph.get_tensor_by_name(weight_quantizer +
                                                                            '_use_symmetric_encoding:0')
            op_mode_tensor = sim.session.graph.get_tensor_by_name(weight_quantizer + '_op_mode:0')
            if weight_quantizer in ['conv2d/BiasAdd/ReadVariableOp_quantized',
                                    'conv2d_1/BiasAdd/ReadVariableOp_quantized',
                                    'conv2d_2/BiasAdd/ReadVariableOp_quantized',
                                    'conv2d_3/BiasAdd/ReadVariableOp_quantized',
                                    'conv2d_4/BiasAdd/ReadVariableOp_quantized',
                                    'single_residual/BiasAdd/ReadVariableOp_quantized']:
                self.assertEqual(sim.session.run(op_mode_tensor),
                                 int(pymo.TensorQuantizerOpMode.oneShotQuantizeDequantize))
                self.assertEqual(sim.session.run(is_symmetric_tensor), True)
            else:

                self.assertEqual(sim.session.run(op_mode_tensor),
                                 int(pymo.TensorQuantizerOpMode.passThrough))
                self.assertEqual(sim.session.run(is_symmetric_tensor), False)

        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.compat.v1.reset_default_graph()

    @pytest.mark.tf2
    def test_parse_config_file_op_type_with_tf2(self):
        """ Test that op specific quantization parameters are set correctly when using json config file """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()

        with sess.graph.as_default():
            _ = single_residual_for_tf2()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {},
                "strict_symmetric": "True",
                "unsigned_symmetric": "True"
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
        for q_config in sim._param_quantizers.values():
            self.assertTrue(q_config.use_strict_symmetric)
            self.assertTrue(q_config.use_unsigned_symmetric)

        for q_config in sim._activation_quantizers.values():
            self.assertTrue(q_config.use_strict_symmetric)
            self.assertTrue(q_config.use_unsigned_symmetric)

        activation_quantizers = [
            'conv2d/BiasAdd_quantized',
            'conv2d_1/BiasAdd_quantized',
            'conv2d_2/BiasAdd_quantized',
            'conv2d_3/BiasAdd_quantized',
            'conv2d_4/BiasAdd_quantized',
            'input_1_quantized',
            'batch_normalization/FusedBatchNormV3_quantized',
            'Relu_quantized',
            'max_pooling2d/MaxPool_quantized',
            'batch_normalization_1/FusedBatchNormV3_quantized',
            'Add_quantized',
            'Relu_2_quantized',
            'average_pooling2d/AvgPool_quantized',
            'single_residual/Softmax_quantized',
            'Relu_1_quantized'
        ]

        weight_quantizers = [
            'conv2d/Conv2D/ReadVariableOp_quantized',
            'conv2d_1/Conv2D/ReadVariableOp_quantized',
            'conv2d_2/Conv2D/ReadVariableOp_quantized',
            'conv2d_3/Conv2D/ReadVariableOp_quantized',
            'conv2d_4/Conv2D/ReadVariableOp_quantized',
            'single_residual/MatMul/ReadVariableOp_quantized',
            'conv2d/BiasAdd/ReadVariableOp_quantized',
            'conv2d_1/BiasAdd/ReadVariableOp_quantized',
            'conv2d_2/BiasAdd/ReadVariableOp_quantized',
            'conv2d_3/BiasAdd/ReadVariableOp_quantized',
            'conv2d_4/BiasAdd/ReadVariableOp_quantized',
            'single_residual/BiasAdd/ReadVariableOp_quantized'
        ]

        for activation_quantizer in activation_quantizers:
            op = sim.session.graph.get_operation_by_name(activation_quantizer)
            op_mode_tensor = op.inputs[QuantizeOpIndices.op_mode]
            if activation_quantizer in ['input_1_quantized',
                                        'conv2d/BiasAdd_quantized',
                                        'max_pooling2d/MaxPool_quantized',
                                        'conv2d_2/BiasAdd_quantized',
                                        'conv2d_3/BiasAdd_quantized',
                                        'Relu_2_quantized',
                                        'average_pooling2d/AvgPool_quantized']:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))
            else:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.passThrough))
        for weight_quantizer in weight_quantizers:
            op = sim.session.graph.get_operation_by_name(weight_quantizer)
            is_symmetric_tensor = op.inputs[QuantizeOpIndices.use_symmetric_encoding]
            op_mode_tensor = op.inputs[QuantizeOpIndices.op_mode]
            if weight_quantizer in ['conv2d/BiasAdd/ReadVariableOp_quantized',
                                    'conv2d_1/BiasAdd/ReadVariableOp_quantized',
                                    'conv2d_2/BiasAdd/ReadVariableOp_quantized',
                                    'conv2d_3/BiasAdd/ReadVariableOp_quantized',
                                    'conv2d_4/BiasAdd/ReadVariableOp_quantized',
                                    'single_residual/BiasAdd/ReadVariableOp_quantized']:
                self.assertEqual(sim.session.run(op_mode_tensor),
                                 int(pymo.TensorQuantizerOpMode.oneShotQuantizeDequantize))
                self.assertEqual(sim.session.run(is_symmetric_tensor), True)
            else:

                self.assertEqual(sim.session.run(op_mode_tensor),
                                 int(pymo.TensorQuantizerOpMode.passThrough))
                self.assertEqual(sim.session.run(is_symmetric_tensor), False)

        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.compat.v1.reset_default_graph()

    @pytest.mark.tf1
    def test_parse_config_file_supergroups(self):
        """ Test that supergroup quantization parameters are set correctly when using json config file,
         corresponding tf2 test is test_parse_config_file_supergroups_with_tf2"""
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.compat.v1.global_variables_initializer()
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
                },
                {
                    "op_list": ["Conv", "Clip"]
                },
            ],
            "model_input": {},
            "model_output": {}
        }
        with open('./quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(sess, ['input_1'], ['single_residual/Softmax'],
                                   config_file='./quantsim_config.json')

        activation_quantizers = [
            'conv2d/BiasAdd_quantized',
            'conv2d_1/BiasAdd_quantized',
            'conv2d_2/BiasAdd_quantized',
            'conv2d_3/BiasAdd_quantized',
            'conv2d_4/BiasAdd_quantized',
            'input_1_quantized',
            'batch_normalization/cond/Merge_quantized',
            'Relu_quantized',
            'max_pooling2d/MaxPool_quantized',
            'batch_normalization_1/cond/Merge_quantized',
            'Add_quantized',
            'Relu_2_quantized',
            'average_pooling2d/AvgPool_quantized',
            'single_residual/Softmax_quantized',
            'Relu_1_quantized'
        ]

        for activation_quantizer in activation_quantizers:
            op_mode_tensor = sim.session.graph.get_tensor_by_name(activation_quantizer + '_op_mode:0')
            if activation_quantizer in ['input_1_quantized',
                                        'conv2d/BiasAdd_quantized',
                                        'conv2d_3/BiasAdd_quantized',
                                        'Add_quantized',
                                        'conv2d_4/BiasAdd_quantized'
                                        ]:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.passThrough))
            else:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))

        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.compat.v1.reset_default_graph()

    @pytest.mark.tf2
    def test_parse_config_file_supergroups_with_tf2(self):
        """ Test that supergroup quantization parameters are set correctly when using json config file with tf2 """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()

        with sess.graph.as_default():
            _ = single_residual_for_tf2()
            init = tf.compat.v1.global_variables_initializer()
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
                },
                {
                    "op_list": ["Conv", "Clip"]
                },
            ],
            "model_input": {},
            "model_output": {}
        }
        with open('./quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)
        sim = QuantizationSimModel(sess, ['input_1'], ['single_residual/Softmax'],
                                   config_file='./quantsim_config.json')

        activation_quantizers = [
            'conv2d/BiasAdd_quantized',
            'conv2d_1/BiasAdd_quantized',
            'conv2d_2/BiasAdd_quantized',
            'conv2d_3/BiasAdd_quantized',
            'conv2d_4/BiasAdd_quantized',
            'input_1_quantized',
            'batch_normalization/FusedBatchNormV3_quantized',
            'Relu_quantized',
            'max_pooling2d/MaxPool_quantized',
            'batch_normalization_1/FusedBatchNormV3_quantized',
            'Add_quantized',
            'Relu_2_quantized',
            'average_pooling2d/AvgPool_quantized',
            'single_residual/Softmax_quantized',
            'Relu_1_quantized'
        ]
        for op_name in activation_quantizers:
            op = sim.session.graph.get_operation_by_name(op_name)
            op_mode_tensor = op.inputs[QuantizeOpIndices.op_mode]
            if op.name in ['input_1_quantized',
                           'conv2d/BiasAdd_quantized',
                           'conv2d_3/BiasAdd_quantized',
                           'Add_quantized',
                           'conv2d_4/BiasAdd_quantized'
                           ]:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.passThrough))
            else:
                self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))

        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.compat.v1.reset_default_graph()

    def test_parse_config_file_model_inputs(self):
        """ Test that model input quantization parameters are set correctly when using json config file """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.compat.v1.global_variables_initializer()
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

        quantize_op = sim.session.graph.get_operation_by_name('input_1_quantized')
        op_mode_tensor = quantize_op.inputs[QuantizeOpIndices.op_mode]
        self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))

        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.compat.v1.reset_default_graph()

    def test_parse_config_file_per_channel_quantization(self):
        """ Test if per channel quantization property gets set correctly"""
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "True"
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
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
        with open('./quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(sess, ['input_1'], ['single_residual/Softmax'],
                                   config_file='./quantsim_config.json')
        self.assertEqual(sim.per_channel_quantization_enabled, True)
        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.compat.v1.reset_default_graph()

    def test_parse_config_file_model_outputs(self):
        """ Test that model output quantization parameters are set correctly when using json config file """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.compat.v1.global_variables_initializer()
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

        quantize_op = sim.session.graph.get_operation_by_name('single_residual/Softmax_quantized')
        op_mode_tensor = quantize_op.inputs[QuantizeOpIndices.op_mode]
        self.assertEqual(sim.session.run(op_mode_tensor), int(pymo.TensorQuantizerOpMode.updateStats))

        if os.path.exists('./quantsim_config.json'):
            os.remove('./quantsim_config.json')
        sess.close()
        sim.session.close()
        tf.compat.v1.reset_default_graph()

    def test_check_correct_of_dtype_bw_rulese(self):
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

        config_file = '/tmp/quantsim_config.json'
        with open(config_file, 'w') as f:
            json.dump(quantsim_config, f)

        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()

        starting_op_names = [input.op.name for input in model.inputs]
        output_op_names = [output.op.name for output in model.outputs]
        connected_graph = ConnectedGraph(sess.graph, starting_op_names, output_op_names)

        supported_kernels = {}
        qsim_config = QuantSimConfigurator(sess.graph, connected_graph, config_file, supported_kernels,
                                           quantsim_output_bw=8, quantsim_param_bw=8,
                                           quantsim_data_type=QuantizationDataType.int)

        qsim_dtype_bw = QuantDtypeBwInfo(data_type=QuantizationDataType.int, act_bw=8, param_bw=8)
        exception_raised = False
        try:
            qsim_config.check_correctness_of_dtype_bw_rules(qsim_dtype_bw)
        except NotImplementedError as exc:
            print(" Test raised exception as expected ", exc)
            exception_raised = True

        assert exception_raised

        # remove test config created
        if os.path.exists(config_file):
            os.remove(config_file)
