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

import shutil
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import numpy as np
import json
import pytest
import unittest.mock
import tensorflow as tf
from packaging import version

import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_common.quantsim import encoding_version
from aimet_tensorflow import graph_editor
from aimet_tensorflow.quantsim import QuantizationSimModel, check_accumulator_overflow
from aimet_tensorflow.quantsim_straight_through_grad import _get_n_and_p, _compute_dloss_by_dmax, \
    compute_intermediate_result_for_learned_grid, LearnedGridParams
from aimet_tensorflow.utils.graph_saver import load_model_from_meta
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.defs import ParameterInfo
from aimet_tensorflow.examples.test_models import model_with_dtype_int, keras_model
from aimet_tensorflow.quantsim import save_checkpoint, load_checkpoint
from aimet_tensorflow.utils.constants import QuantizeOpIndices
from aimet_tensorflow.utils import transformer_utils
from aimet_tensorflow.examples.test_models import transposed_conv2d_model


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()


class TestQuantSim(unittest.TestCase):

    def test_construction_cpu_model(self):
        """
        Create QuantSim for a CPU model and check that quantizers have been added to the graph
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=False)

        # One run through the model to check if the ops got added correctly
        model_output = sess.graph.get_tensor_by_name('conv2d_1/BiasAdd_quantized:0')
        model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
        dummy_input = np.random.randn(20, 28, 28, 3)
        sess.run(model_output, feed_dict={model_input: dummy_input})

        # Check that quantized ops got added for all params
        quant_ops = [op for op in sess.graph.get_operations() if op.type == 'QcQuantize']
        for op in quant_ops:
            print(op.name)
        self.assertEqual(10, len(quant_ops))

        # Check that the quant ops are correctly connected in the graph
        self.assertEqual('Conv2D', quant_ops[0].outputs[0].consumers()[0].type)
        self.assertEqual('BiasAdd', quant_ops[1].outputs[0].consumers()[0].type)
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.passThrough), sess.run(quant_ops[1].inputs[1]))

        # Check that op-mode is set correctly
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                         sess.run(quant_ops[0].inputs[1]))

        sess.close()
        sim.session.close()
        del sim

    @pytest.mark.cuda
    def test_construction_gpu_model(self):
        """
        Create QuantSim for a GPU model and check that quantizers have been added to the graph
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/gpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=True)

        # One run through the model to check if the ops got added correctly
        model_output = sess.graph.get_tensor_by_name('conv2d_1/BiasAdd_quantized:0')
        model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
        dummy_input = np.random.randn(20, 28, 28, 3)
        sess.run(model_output, feed_dict={model_input: dummy_input})

        # Check that quantized ops got added for all params
        quant_ops = [op for op in sess.graph.get_operations() if op.type == 'QcQuantize']
        for op in quant_ops:
            print(op.name)
        self.assertEqual(10, len(quant_ops))

        # Check that the quant ops are correctly connected in the graph
        self.assertEqual('Conv2D', quant_ops[0].outputs[0].consumers()[0].type)
        self.assertEqual('BiasAdd', quant_ops[1].outputs[0].consumers()[0].type)

        # Check that op-mode is set correctly
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                         sess.run(quant_ops[0].inputs[1]))

        sess.close()
        sim.session.close()
        del sim

    def test_compute_encodings_cpu_model(self):
        """
        Create QuantSim for a CPU model and test that activation encodings are computed
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=False)

        # Check that op-mode is set correctly
        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
        conv2d_output_quant_op = sim.session.graph.get_operation_by_name('conv2d/Relu_quantized')
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                         sim.session.run(conv2d_weight_quant_op.inputs[1]))
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.updateStats),
                         sim.session.run(conv2d_output_quant_op.inputs[1]))

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('conv2d_1/Relu_quantized:0')
            model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
            dummy_input = np.random.randn(20, 28, 28, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        # Check if encodings have been calculated
        deactivated_quantizers = [
            'conv2d_input_quantized',
            'conv2d/BiasAdd_quantized',
            'conv2d_1/BiasAdd_quantized'
        ]
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            op_mode_name_suffix = '_op_mode/Read/ReadVariableOp:0'
        else:
            op_mode_name_suffix = '_op_mode/read:0'

        for name, quantizer in sim._activation_quantizers.items():
            if name in deactivated_quantizers:
                self.assertTrue(int(libpymo.TensorQuantizerOpMode.passThrough),
                                sim.session.run(name + op_mode_name_suffix))
            else:
                self.assertTrue(quantizer.tensor_quantizer.isEncodingValid,
                                "quantizer: {} does not have a valid encoding".format(name))

        # Check that op-mode is set correctly
        # Check that quantized ops got added for all params
        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
        conv2d_output_quant_op = sim.session.graph.get_operation_by_name('conv2d/Relu_quantized')

        self.assertEqual(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                         sim.session.run(conv2d_weight_quant_op.inputs[1]))
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.quantizeDequantize),
                         sim.session.run(conv2d_output_quant_op.inputs[1]))

    @pytest.mark.cuda
    def test_compute_encodings_transposed_conv_model(self):
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with tf.device('/gpu:0'):
            _ = transposed_conv2d_model()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

        sim = QuantizationSimModel(sess, ['input_1'], ['conv2d_transpose/BiasAdd'], use_cuda=True,
                                   quant_scheme='tf')

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('conv2d_transpose/BiasAdd:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(1, 7, 7, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d_transpose/conv2d_transpose/ReadVariableOp_quantized')
        assert int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize) == sim.session.run(conv2d_weight_quant_op.inputs[1])

        sim.export('/tmp', 'quant_sim_model')
        with open('/tmp/quant_sim_model.encodings') as json_file:
            encoding_data = json.load(json_file)

        param_keys = list(encoding_data["param_encodings"].keys())
        assert param_keys[0] == "conv2d_transpose/conv2d_transpose/ReadVariableOp:0"
        sess.close()

    def _test_compute_encodings_fp16(self, device: str):
        tf.compat.v1.reset_default_graph()
        with tf.device(device):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=False, default_output_bw=16,
                                   default_param_bw=16, default_data_type=QuantizationDataType.float)

        # Check that op-mode is set correctly
        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
        conv2d_output_quant_op = sim.session.graph.get_operation_by_name('conv2d/Relu_quantized')
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                         sim.session.run(conv2d_weight_quant_op.inputs[1]))
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.updateStats),
                         sim.session.run(conv2d_output_quant_op.inputs[1]))

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('conv2d_1/Relu_quantized:0')
            model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
            dummy_input = np.random.randn(20, 28, 28, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        # Check if encodings have been calculated
        deactivated_quantizers = [
            'conv2d_input_quantized',
            'conv2d/BiasAdd_quantized',
            'conv2d_1/BiasAdd_quantized'
        ]
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            op_mode_name_suffix = '_op_mode/Read/ReadVariableOp:0'
        else:
            op_mode_name_suffix = '_op_mode/read:0'

        for name, quantizer in sim._activation_quantizers.items():
            if name in deactivated_quantizers:
                self.assertTrue(int(libpymo.TensorQuantizerOpMode.passThrough),
                                sim.session.run(name + op_mode_name_suffix))
            else:
                self.assertTrue(int(libpymo.TensorQuantizerOpMode.quantizeDequantize),
                                sim.session.run(name + op_mode_name_suffix))

        sim.export('/tmp', 'quant_sim_model_fp16')

        with open('/tmp/quant_sim_model_fp16.encodings') as json_file:
            encoding_data = json.load(json_file)

        generated_encoding_version = encoding_data["version"]
        self.assertEqual(encoding_version, generated_encoding_version)

        # This is a fp16 sim model. Make sure all the layers contain fp16 encodings
        activation_keys = list(encoding_data["activation_encodings"].keys())
        for key in activation_keys:
            act_encoding_keys = encoding_data["activation_encodings"][key][0]
            self.assertTrue("bitwidth" in act_encoding_keys)
            self.assertEqual(act_encoding_keys['bitwidth'], 16)
            self.assertTrue("dtype" in act_encoding_keys)
            self.assertEqual(act_encoding_keys['dtype'], 'float')

        param_keys = list(encoding_data["param_encodings"].keys())
        for key in param_keys:
            param_encoding_keys = encoding_data["param_encodings"][key][0]
            self.assertTrue("bitwidth" in param_encoding_keys)
            self.assertEqual(param_encoding_keys['bitwidth'], 16)
            self.assertTrue("dtype" in param_encoding_keys)
            self.assertEqual(param_encoding_keys['dtype'], 'float')

        sess.close()
        sim.session.close()
        del sim

    def test_compute_encodings_cpu_model_fp16(self):
        """
        Create QuantSim for a CPU model and test that activation encodings are computed
        """
        self._test_compute_encodings_fp16('/cpu:0')

    @pytest.mark.cuda
    def test_compute_encodings_gpu_model_fp16(self):
        """
        Create QuantSim for a CPU model and test that activation encodings are computed
        """
        self._test_compute_encodings_fp16('/gpu:0')

    def _save_to_keras_common_test_code(self, use_cuda):
        tf.compat.v1.reset_default_graph()
        if not use_cuda:
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()
        else:
            with tf.device('/cpu:0'):
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
                model.add(tf.keras.layers.MaxPooling2D((2, 2)))
                model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
                model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=use_cuda)

        # Check that op-mode is set correctly
        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
        conv2d_output_quant_op = sim.session.graph.get_operation_by_name('conv2d/Relu_quantized')
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                         sim.session.run(conv2d_weight_quant_op.inputs[1]))
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.updateStats),
                         sim.session.run(conv2d_output_quant_op.inputs[1]))

        def dummy_forward_pass(sess, eval_tensor_name):
            model_output = sess.graph.get_tensor_by_name(eval_tensor_name)
            model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
            dummy_input = np.random.randn(20, 28, 28, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, 'conv2d_1/Relu_quantized:0')
        mod_sess = sim.save_to_keras()

        # Check 1: The new graph is well formed. Try forward pass through the graph.
        dummy_forward_pass(mod_sess, 'conv2d_1/Relu_quantized_static:0')

        # Check 2: All the QcQuantizeOp nodes have no output - meaning are disconnected from the main graph
        op_count = 0
        for op in mod_sess.graph.get_operations():
            if op.type == "QcQuantize":
                op_count += 1
                self.assertFalse(op.outputs[0].consumers())

        # Check 3: One QcQuantizeStatic for each QcQuantize op
        static_op_count = 0
        for op in mod_sess.graph.get_operations():
            if op.type == "QcQuantizeStatic":
                static_op_count += 1
        self.assertEqual(op_count, static_op_count)

        # Check 4: Make sure the attributes are set correctly
        op = mod_sess.graph.get_operation_by_name("conv2d/Conv2D/ReadVariableOp_quantized_static")
        self.assertEqual(8, op.get_attr("bitwidth"))
        self.assertEqual(1, op.get_attr("quant_scheme"))  # TF-Enhanced
        self.assertEqual(1, op.get_attr("op_mode"))  # oneShotQuantizeDequantize

        op = mod_sess.graph.get_operation_by_name("conv2d/BiasAdd_quantized_static")
        self.assertEqual(3, op.get_attr("op_mode"))  # passThrough

        op = mod_sess.graph.get_operation_by_name("conv2d/Relu_quantized_static")
        self.assertEqual(8, op.get_attr("bitwidth"))
        self.assertEqual(1, op.get_attr("quant_scheme"))  # TF-Enhanced
        self.assertEqual(2, op.get_attr("op_mode"))  # quantizeDequantize

        sess.close()
        sim.session.close()
        del sim

    def test_parse_config_file_default_supported_kernels(self):
        """
        Test that the supported_kernels in the defaults section is parsed correctly and its values are added
        in the dict _supported_kernels
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

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

        if not os.path.exists("data"):
            os.mkdir("data")

        with open('data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=False, default_output_bw=16,
                                   default_param_bw=16, default_data_type=QuantizationDataType.float,
                                   config_file='data/quantsim_config.json')

        supported_kernels_in_defaults = sim.get_supported_kernels()["defaults"]
        assert len(supported_kernels_in_defaults) == 2
        assert supported_kernels_in_defaults == expected_supported_kernels

    def test_parse_config_file_op_type_supported_kernels(self):
        """
        Test that the supported_kernels in the op_type section is parsed correctly and its values are added
        in the dict _supported_kernels
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

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
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=False, default_output_bw=16,
                                   default_param_bw=16, default_data_type=QuantizationDataType.float,
                                   config_file='data/quantsim_config.json')

        supported_kernels_in_defaults = sim.get_supported_kernels()["Conv"]
        assert len(supported_kernels_in_defaults) == 1
        assert supported_kernels_in_defaults == expected_supported_kernels


    def test_save_to_keras_cpu_model(self):
        """
        Create sim model for a keras pipeline
        """
        self._save_to_keras_common_test_code(False)

    def test_save_to_keras_gpu_model(self):
        """
        Create sim model for a keras pipeline
        """
        self._save_to_keras_common_test_code(True)

    @pytest.mark.cuda
    def test_compute_encodings_gpu_model(self):
        """
        Create QuantSim for a CPU model and test that activation encodings are computed
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/gpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=True)

        # Check that op-mode is set correctly
        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
        conv2d_output_quant_op = sim.session.graph.get_operation_by_name('conv2d/Relu_quantized')
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                         sim.session.run(conv2d_weight_quant_op.inputs[1]))
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.updateStats),
                         sim.session.run(conv2d_output_quant_op.inputs[1]))

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('conv2d_1/Relu_quantized:0')
            model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
            dummy_input = np.random.randn(20, 28, 28, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        # Check if encodings have been calculated
        deactivated_quantizers = [
            'conv2d_input_quantized',
            'conv2d/BiasAdd_quantized',
            'conv2d_1/BiasAdd_quantized'
        ]
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            op_mode_name_suffix = '_op_mode/Read/ReadVariableOp:0'
        else:
            op_mode_name_suffix = '_op_mode/read:0'

        for name, quantizer in sim._activation_quantizers.items():
            if name in deactivated_quantizers:
                self.assertTrue(int(libpymo.TensorQuantizerOpMode.passThrough),
                                sim.session.run(name + op_mode_name_suffix))
            else:
                self.assertTrue(quantizer.tensor_quantizer.isEncodingValid,
                                "quantizer: {} does not have a valid encoding".format(name))

        # Check that op-mode is set correctly
        # Check that quantized ops got added for all params
        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
        conv2d_output_quant_op = sim.session.graph.get_operation_by_name('conv2d/Relu_quantized')

        self.assertEqual(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                         sim.session.run(conv2d_weight_quant_op.inputs[1]))
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.quantizeDequantize),
                         sim.session.run(conv2d_output_quant_op.inputs[1]))

        sess.close()
        sim.session.close()
        del sim

    @pytest.mark.cuda
    def test_compute_encodings_quant_scheme_update(self):
        """
        Create QuantSim model and update quantScheme using property interface
        """
        tf.compat.v1.reset_default_graph()
        np.random.seed(0)
        tf.compat.v1.set_random_seed(0)

        with tf.device('/gpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=True)

        # Check that op-mode is set correctly
        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')

        self.assertEqual(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                         sim.session.run(conv2d_weight_quant_op.inputs[1]))

        def dummy_forward_pass(sess, args):
            np.random.seed(0)
            tf.compat.v1.set_random_seed(0)
            model_output = sess.graph.get_tensor_by_name('conv2d_1/Relu_quantized:0')
            model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
            dummy_input = np.random.randn(20, 28, 28, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        p_quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')
        old_p_encoding_min = p_quantizer.get_variable_from_op(QuantizeOpIndices.encoding_min)
        old_p_encoding_max = p_quantizer.get_variable_from_op(QuantizeOpIndices.encoding_max)

        self.assertEqual(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED, p_quantizer.quant_scheme)
        p_quantizer.quant_scheme = QuantScheme.post_training_tf
        self.assertEqual(libpymo.QuantizationMode.QUANTIZATION_TF, p_quantizer.quant_scheme)

        # invoke compute encoding after quantScheme update
        sim.compute_encodings(dummy_forward_pass, None)
        new_p_encoding_min = p_quantizer.get_variable_from_op(QuantizeOpIndices.encoding_min)
        new_p_encoding_max = p_quantizer.get_variable_from_op(QuantizeOpIndices.encoding_max)

        # validate
        self.assertNotEqual(old_p_encoding_min, new_p_encoding_min)
        self.assertNotEqual(old_p_encoding_max, new_p_encoding_max)

        sess.close()
        sim.session.close()
        del sim

    def test_export_quantizer_args(self):
        """
        Create QuantSim for a CPU model, compute encodings and export out a resulting model
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, [model.input.op.name], [model.output.op.name], use_cuda=False,
                                   quant_scheme=QuantScheme.post_training_tf_enhanced, default_output_bw=16,
                                   default_param_bw=4)

        sim.export('/tmp', 'quant_sim_with_quantizer_args')
        with open('/tmp/quant_sim_with_quantizer_args.encodings') as json_file:
             encoding_data = json.load(json_file)

        assert "quantizer_args" in encoding_data
        quantizer_args = encoding_data["quantizer_args"]
        assert quantizer_args["activation_bitwidth"] == 16
        assert quantizer_args["param_bitwidth"] == 4
        assert quantizer_args["per_channel_quantization"] == False
        assert quantizer_args["quant_scheme"] == QuantScheme.post_training_tf_enhanced.name
        assert quantizer_args["is_symmetric"] == True
        assert quantizer_args["dtype"] == "int"

    def test_export_cpu_model(self):
        """
        Create QuantSim for a CPU model, compute encodings and export out a resulting model
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, [model.input.op.name], [model.output.op.name], use_cuda=False)

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name(model.output.name)
            model_output = model_output.consumers()[0].outputs[0]
            model_input = sess.graph.get_tensor_by_name(model.input.name)
            dummy_input = np.random.randn(20, 28, 28, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        # Make some changes to model parameters to see if they are part of the exported model
        with sim.session.graph.as_default():
            first_bias_tensor = sim.session.graph.get_tensor_by_name('conv2d/BiasAdd/ReadVariableOp:0')
            first_bias_tensor_val = sim.session.run(first_bias_tensor)
            self.assertTrue(np.any(first_bias_tensor_val == 0))
            first_bias_tensor_var = [var for var in tf.compat.v1.global_variables() if var.name == 'conv2d/bias:0'][0]
            first_bias_tensor_var.load(np.ones(32), sim.session)

        all_op_types = [op.type for op in sim.session.graph.get_operations()]
        self.assertIn('QcQuantize', all_op_types)

        sim.export('/tmp', 'quant_sim_model')

        with open('/tmp/quant_sim_model.encodings') as json_file:
            encoding_data = json.load(json_file)
        activation_keys = list(encoding_data["activation_encodings"].keys())
        self.assertTrue(activation_keys[0] == "conv2d/Relu:0")
        self.assertTrue(isinstance(encoding_data["activation_encodings"]["conv2d/Relu:0"], list))
        act_encoding_keys = encoding_data["activation_encodings"]["conv2d/Relu:0"][0].keys()
        self.assertTrue("bitwidth" in act_encoding_keys)
        self.assertTrue("is_symmetric" in act_encoding_keys)
        self.assertTrue("max" in act_encoding_keys)
        self.assertTrue("min" in act_encoding_keys)
        self.assertTrue("offset" in act_encoding_keys)
        self.assertTrue("scale" in act_encoding_keys)

        param_keys = list(encoding_data["param_encodings"].keys())
        self.assertTrue(param_keys[0] == "conv2d/Conv2D/ReadVariableOp:0")
        self.assertTrue(isinstance(encoding_data["param_encodings"]["conv2d/Conv2D/ReadVariableOp:0"], list))
        param_encoding_keys = encoding_data["param_encodings"]["conv2d/Conv2D/ReadVariableOp:0"][0].keys()
        self.assertTrue("bitwidth" in param_encoding_keys)
        self.assertTrue("is_symmetric" in param_encoding_keys)
        self.assertTrue("max" in param_encoding_keys)
        self.assertTrue("min" in param_encoding_keys)
        self.assertTrue("offset" in param_encoding_keys)
        self.assertTrue("scale" in param_encoding_keys)

        new_sess = load_model_from_meta('/tmp/quant_sim_model.meta')
        first_bias_tensor = new_sess.graph.get_tensor_by_name('conv2d/BiasAdd/ReadVariableOp:0')
        first_bias_tensor_val = new_sess.run(first_bias_tensor)
        self.assertTrue(np.any(first_bias_tensor_val == 1))

        all_op_types = [op.type for op in new_sess.graph.get_operations()]
        self.assertNotIn('QcQuantize', all_op_types)
        sess.close()
        sim.session.close()
        del sim

    def test_save_load_ckpt_cpu_model(self):
        """
        Create QuantSim for a CPU model, test save and load on a quantsim model.
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, [model.input.op.name], [model.output.op.name], use_cuda=False)

        # save quantsim model
        save_checkpoint(sim, './test_3', 'orig_quantsim_model')

        new_quantsim = load_checkpoint('./test_3', 'orig_quantsim_model')

        # validations
        assert(sim is not new_quantsim)
        self.assertTrue(new_quantsim.session is not None)
        self.assertTrue(new_quantsim._quant_scheme == sim._quant_scheme)
        self.assertTrue(new_quantsim._rounding_mode == sim._rounding_mode)
        self.assertTrue(new_quantsim._use_cuda == sim._use_cuda)
        self.assertTrue(len(new_quantsim._param_quantizers) == len(sim._param_quantizers))
        self.assertTrue(len(new_quantsim._activation_quantizers) == len(sim._activation_quantizers))

        for quantize_op in new_quantsim._param_quantizers:
            self.assertFalse(sim._param_quantizers[quantize_op].session ==
                             new_quantsim._param_quantizers[quantize_op].session)
            self.assertTrue(sim._param_quantizers[quantize_op].tensor_quantizer.getQuantScheme() ==
                            new_quantsim._param_quantizers[quantize_op].tensor_quantizer.getQuantScheme())
            self.assertTrue(sim._param_quantizers[quantize_op].tensor_quantizer.roundingMode ==
                            new_quantsim._param_quantizers[quantize_op].tensor_quantizer.roundingMode)
            self.assertFalse(sim._param_quantizers[quantize_op].tensor_quantizer.isEncodingValid)
            self.assertFalse(new_quantsim._param_quantizers[quantize_op].tensor_quantizer.isEncodingValid)

        for quantize_op in new_quantsim._activation_quantizers:
            self.assertFalse(sim._activation_quantizers[quantize_op].session ==
                             new_quantsim._activation_quantizers[quantize_op].session)
            self.assertTrue(sim._activation_quantizers[quantize_op].tensor_quantizer.getQuantScheme() ==
                            new_quantsim._activation_quantizers[quantize_op].tensor_quantizer.getQuantScheme())
            self.assertTrue(sim._activation_quantizers[quantize_op].tensor_quantizer.roundingMode ==
                            new_quantsim._activation_quantizers[quantize_op].tensor_quantizer.roundingMode)
            self.assertFalse(sim._activation_quantizers[quantize_op].tensor_quantizer.isEncodingValid)
            self.assertFalse(new_quantsim._activation_quantizers[quantize_op].tensor_quantizer.isEncodingValid)

        # remove the old quant sim reference and session
        # to test that everything is loaded correctly on new quantsim including tensor quantizer references
        sim.session.close()
        del sim

        # delete temp folder created and close sessions
        shutil.rmtree('./test_3')
        sess.close()
        new_quantsim.session.close()
        del new_quantsim

    def test_save_load_ckpt_after_compute_encoding_on_orig_object(self):
        """
        Create QuantSim for a CPU model, test save and load on a quantsim model
        when encodings have been computed on original quantsim object
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, [model.input.op.name], [model.output.op.name], use_cuda=False)

        def dummy_forward_pass(n_sess, args):
            model_output = n_sess.graph.get_tensor_by_name(model.output.name)
            model_output = model_output.consumers()[0].outputs[0]
            model_input = n_sess.graph.get_tensor_by_name(model.input.name)
            dummy_input = np.random.randn(20, 28, 28, 3)
            n_sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        # save quantsim model
        save_checkpoint(sim, './test_3', 'orig_quantsim_model')

        new_quantsim = load_checkpoint('./test_3', 'orig_quantsim_model')

        # validations
        assert(sim is not new_quantsim)

        # as we have performed computeEncodings() on saved quantsim object, these must be set to True/False
        # in loaded quantsim object as on orig model
        for quantize_op in new_quantsim._param_quantizers:
            self.assertTrue(new_quantsim._param_quantizers[quantize_op].tensor_quantizer.isEncodingValid ==
                            sim._param_quantizers[quantize_op].tensor_quantizer.isEncodingValid)
            self.assertTrue(new_quantsim._param_quantizers[quantize_op].
                            get_variable_from_op(QuantizeOpIndices.encoding_min) ==
                            sim._param_quantizers[quantize_op].
                            get_variable_from_op(QuantizeOpIndices.encoding_min))
            self.assertTrue(new_quantsim._param_quantizers[quantize_op].
                            get_variable_from_op(QuantizeOpIndices.encoding_max) ==
                            sim._param_quantizers[quantize_op].
                            get_variable_from_op(QuantizeOpIndices.encoding_max))

        for quantize_op in new_quantsim._activation_quantizers:
            self.assertTrue(new_quantsim._activation_quantizers[quantize_op].tensor_quantizer.isEncodingValid ==
                            sim._activation_quantizers[quantize_op].tensor_quantizer.isEncodingValid)
            self.assertTrue(new_quantsim._activation_quantizers[quantize_op].
                            get_variable_from_op(QuantizeOpIndices.encoding_min) ==
                            sim._activation_quantizers[quantize_op].
                            get_variable_from_op(QuantizeOpIndices.encoding_min))
            self.assertTrue(new_quantsim._activation_quantizers[quantize_op].
                            get_variable_from_op(QuantizeOpIndices.encoding_max) ==
                            sim._activation_quantizers[quantize_op].
                            get_variable_from_op(QuantizeOpIndices.encoding_max))

        # delete temp folder created and close sessions
        shutil.rmtree('./test_3')
        sess.close()
        sim.session.close()
        new_quantsim.session.close()
        del sim
        del new_quantsim

    def test_set_get_quantizer_params_using_properties(self):
        """
        Create QuantSim for a CPU model, test param read and write using properties
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, [model.input.op.name], [model.output.op.name], use_cuda=False)

        p_quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')
        o_quantizer = sim.quantizer_config('conv2d/Relu_quantized')
        bias_quantizer = sim.quantizer_config('conv2d/BiasAdd/ReadVariableOp_quantized')

        # check if __str__ can print the object info
        print(p_quantizer)
        bitwidth = p_quantizer.bitwidth
        self.assertEqual(8, bitwidth)
        p_quantizer.bitwidth = 6
        bitwidth = p_quantizer.bitwidth
        self.assertEqual(6, bitwidth)

        bitwidth = o_quantizer.bitwidth
        self.assertEqual(8, bitwidth)
        o_quantizer.bitwidth = 6
        bitwidth = o_quantizer.bitwidth
        self.assertEqual(6, bitwidth)

        sym_encoding = bias_quantizer.use_symmetric_encoding
        self.assertTrue(sym_encoding)
        bias_quantizer.use_symmetric_encoding = False
        sym_encoding = bias_quantizer.use_symmetric_encoding
        self.assertFalse(sym_encoding)

        rounding_mode = o_quantizer.rounding_mode
        self.assertEqual(libpymo.RoundingMode.ROUND_NEAREST, rounding_mode)
        o_quantizer.rounding_mode = libpymo.RoundingMode.ROUND_STOCHASTIC
        rounding_mode = o_quantizer.rounding_mode
        self.assertEqual(libpymo.RoundingMode.ROUND_STOCHASTIC, rounding_mode)

        quant_scheme = o_quantizer.quant_scheme
        self.assertEqual(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED, quant_scheme)
        o_quantizer.quant_scheme = QuantScheme.post_training_tf
        quant_scheme = o_quantizer.quant_scheme
        self.assertEqual(libpymo.QuantizationMode.QUANTIZATION_TF, quant_scheme)
        self.assertFalse(o_quantizer.tensor_quantizer.isEncodingValid)

        is_enabled = p_quantizer.enabled
        self.assertTrue(is_enabled)
        p_quantizer.enabled = False
        is_enabled = p_quantizer.enabled
        self.assertFalse(is_enabled)

        # use strict symmetric and unsigned symmetric
        use_strict_symmetric = p_quantizer.use_strict_symmetric
        self.assertFalse(use_strict_symmetric)
        p_quantizer.use_strict_symmetric = True
        use_strict_symmetric = p_quantizer.use_strict_symmetric
        self.assertTrue(use_strict_symmetric)

        use_unsigned_symmetric = p_quantizer.use_unsigned_symmetric
        self.assertFalse(use_unsigned_symmetric)
        p_quantizer.use_unsigned_symmetric = True
        use_unsigned_symmetric = p_quantizer.use_unsigned_symmetric
        self.assertTrue(use_unsigned_symmetric)

        sim.session.close()
        del sim

    def test_manual_quantize(self):
        """ Test quantizing a model by manually specifying ops to quantize """
        def get_manual_activations(_graph, _conn_graph):
            """
            Overriding function for getting a list of ops to insert activation quantizers for
            :param _graph: Unused argument
            :param _conn_graph: Unused argument
            :return: List of ops to insert activation quantizers for
            """
            return ['conv2d/Relu']

        def get_manual_params(_graph, _conn_graph, _starting_ops, _ending_ops):
            """
            Overriding function for getting a list of ops to insert param quantizers for
            :param _graph: Unused argument
            :param _conn_graph: Unused argument
            :param _starting_ops: Unused argument
            :param _ending_ops: Unused argument
            :return: List of ops to insert param quantizers for, and list of param indices for these ops
            """
            return {'conv2d_1/Conv2D/ReadVariableOp': ParameterInfo('weight', ['conv2d_1/Conv2D'])}

        def configure_quantization_ops(self, _conn_graph, _ops_with_param_names, _indices,
                                       _params_to_quantize, _activation_op_names):
            """
            Overriding function for configuring quantization ops inserted by QuantizationSimModel
            :param self: Self refers to QuantizationSimModel object
            :param _conn_graph: Unused argument
            :param _ops_with_param_names: Unused argument
            :param _indices: Unused argument
            :param _params_to_quantize: Unused argument
            :param _activation_op_names: Unused argument
            """
            conv2d_relu_quant_info = self._activation_quantizers['conv2d/Relu_quantized']
            conv2d_relu_quant_info.enabled = False
            conv2d_relu_quant_info.enabled = True
            conv2d_1_weight_quant_info = self._param_quantizers['conv2d_1/Conv2D/ReadVariableOp_quantized']
            conv2d_1_weight_quant_info.enabled = False
            conv2d_1_weight_quant_info.enabled = True

        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        orig_get_ops_to_quantize_activations_for = QuantizationSimModel._get_ops_to_quantize_activations_for
        orig_get_ops_to_quantize_weights_for = QuantizationSimModel._get_ops_to_quantize_params_for
        orig_configure_quantization_ops = QuantizationSimModel.configure_quantization_ops
        QuantizationSimModel._get_ops_to_quantize_activations_for = get_manual_activations
        QuantizationSimModel._get_ops_to_quantize_params_for = get_manual_params
        QuantizationSimModel.configure_quantization_ops = configure_quantization_ops
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=False)
        self.assertEqual(1, len(sim._activation_quantizers))
        self.assertEqual(1, len(sim._param_quantizers))
        sess.close()
        sim.session.close()
        QuantizationSimModel._get_ops_to_quantize_activations_for = orig_get_ops_to_quantize_activations_for
        QuantizationSimModel._get_ops_to_quantize_params_for = orig_get_ops_to_quantize_weights_for
        QuantizationSimModel.configure_quantization_ops = orig_configure_quantization_ops

        sim.session.close()
        del sim

    def test_skip_quantizing_dtype_int(self):
        """ Test that op with dtype int32 is skipped during quantization """
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            _ = model_with_dtype_int()
            initialize_uninitialized_vars(sess)
            sim = QuantizationSimModel(sess, ['input_1', 'input_2'], ['model_with_dtype_int/Softmax'], use_cuda=False)
            self.assertEqual(6, len(sim._activation_quantizers))
            self.assertTrue('input_1_quantized' not in sim._activation_quantizers)
            self.assertTrue('input_2_quantized' in sim._activation_quantizers)
            sim.session.close()
            del sim

    def test_compute_encodings(self):
        """ Test that ops not evaluated during compute encodings are set to passThrough mode. """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        test_inp = np.ndarray((1, 32, 32, 3))

        def keras_model_functional():
            """ Function for returning basic keras model defined functionally """
            inputs = tf.keras.Input(shape=(32, 32, 3,))
            x = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
            x = tf.keras.layers.BatchNormalization(momentum=.3, epsilon=.65)(x, training=True)
            with tf.compat.v1.variable_scope("scope_1"):
                x = tf.keras.layers.Conv2D(16, (2, 2), activation=tf.nn.tanh)(x)
                x = tf.keras.layers.BatchNormalization(momentum=.4, epsilon=.25)(x, training=False)
                x = tf.keras.layers.Conv2D(8, (2, 2), activation=tf.nn.tanh)(x)
                x = tf.keras.layers.BatchNormalization(momentum=.5, epsilon=.35)(x, training=False)
                x = tf.keras.layers.Conv2D(4, (2, 2), activation=tf.nn.relu6)(x)
            x = tf.keras.layers.Flatten()(x)
            outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="keras_model_functional")(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        def dummy_forward_func(sess, _):
            input_tensor = sess.graph.get_tensor_by_name('input_1:0')
            output_tensor = sess.graph.get_tensor_by_name('flatten/Reshape:0')
            sess.run(output_tensor, feed_dict={input_tensor: test_inp})

        with sess.as_default():
            _ = keras_model_functional()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            sim = QuantizationSimModel(sess, ['input_1'], ['keras_model_functional/Softmax'])
            sim.compute_encodings(dummy_forward_func, None)

            for name, quant_info in sim._activation_quantizers.items():
                if name in ['keras_model_functional/Softmax_quantized', 'keras_model_functional/BiasAdd_quantized']:
                    # Check that quantizers after op evaluated in compute_encodings are in passThrough (3) mode
                    self.assertEqual(quant_info.get_op_mode(), 3)
                    self.assertFalse(quant_info.tensor_quantizer.isEncodingValid)
                elif name in ['scope_1/conv2d_3/BiasAdd_quantized']:
                    # Check that passThrough quantizers remain as passThrough (3)
                    self.assertEqual(quant_info.get_op_mode(), 3)
                    self.assertFalse(quant_info.tensor_quantizer.isEncodingValid)
                else:
                    # Check that all other quantizers are in quantizeDequantize (2) mode
                    self.assertEqual(quant_info.get_op_mode(), 2)
                    self.assertTrue(quant_info.tensor_quantizer.isEncodingValid)

            input_tensor = sim.session.graph.get_tensor_by_name('input_1:0')
            output_tensor = sim.session.graph.get_tensor_by_name('keras_model_functional/Softmax:0')
            sim.session.run(output_tensor, feed_dict={input_tensor: test_inp})
            sim.session.close()
            del sim

    def test_set_and_freeze_param_encodings(self):
        """ Test set and freeze parameter encodings functionality """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            _ = keras_model()
            init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session()
        session.run(init)

        sim = QuantizationSimModel(session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
        param_encodings = {'conv2d/Conv2D/ReadVariableOp:0': [{'bitwidth': 4, 'is_symmetric': False,
                                                               'max': 0.14584073424339294,
                                                               'min': -0.12761062383651733,
                                                               'offset': -7.0, 'scale': 0.01823008991777897}]}
        # export encodings to JSON file
        encoding_file_path = os.path.join('./', 'dummy.encodings')
        with open(encoding_file_path, 'w') as encoding_fp:
            json.dump(param_encodings, encoding_fp, sort_keys=True, indent=4)

        sim.set_and_freeze_param_encodings(encoding_path='./dummy.encodings')

        quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')
        encoding = param_encodings['conv2d/Conv2D/ReadVariableOp:0'][0]

        encoding_max = quantizer.get_variable_from_op(QuantizeOpIndices.encoding_max)
        encoding_min = quantizer.get_variable_from_op(QuantizeOpIndices.encoding_min)

        self.assertEqual(encoding_min, encoding.get('min'))
        self.assertEqual(encoding_max, encoding.get('max'))
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.quantizeDequantize), quantizer.get_op_mode())
        self.assertEqual(quantizer.is_encoding_valid(), True)

        session.close()

        # Delete encodings JSON file
        if os.path.exists("./dummy.encodings"):
            os.remove("./dummy.encodings")

    @pytest.mark.tf2
    def test_transformer_mask_override(self):
        tf.compat.v1.reset_default_graph()

        mask_add_name = 'dummy_attention/mask_add'
        np.random.seed(0)
        dummy_input_0 = np.random.uniform(low=-10.0, high=10.0, size=(28, 28, 3))
        dummy_input_1 = np.random.uniform(low=-10000.0, high=10.0, size=(28, 28, 3))
        with tf.device('/cpu:0'):
            input_0 = tf.Variable(initial_value=dummy_input_0, shape=(28, 28, 3), dtype=tf.float32)
            input_1 = tf.Variable(initial_value=dummy_input_1, shape=(28, 28, 3), dtype=tf.float32)
            mask_add = tf.math.add(input_0, input_1, name=mask_add_name)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        transformer_utils.register_attention_mask_override(mask_add_name)
        sim = QuantizationSimModel(sess, [input_0.op.name, input_1.op.name], [mask_add.op.name], use_cuda=False)

        def dummy_forward_pass(sess, args):
            mask_add_tensor = sess.graph.get_operation_by_name(mask_add.op.name + '_quantized').outputs[0]
            sess.run(mask_add_tensor)

        sim.compute_encodings(dummy_forward_pass, None)
        mask_quantizer = sim.quantizer_config(mask_add_name + '_quantized')
        self.assertTrue(mask_quantizer._is_encoding_frozen)
        encoding_min = mask_quantizer.get_variable_from_op(QuantizeOpIndices.encoding_min)

        self.assertAlmostEqual(encoding_min, transformer_utils.MASK_OVERRIDE_VALUE, places=1)

        del sim
        sess.close()

    @pytest.mark.tf2
    def test_transformer_mask_override_f16_skip(self):
        tf.compat.v1.reset_default_graph()

        mask_add_name = 'dummy_attention/mask_add'
        np.random.seed(0)
        dummy_input_0 = np.random.uniform(low=-10.0, high=10.0, size=(28, 28, 3))
        dummy_input_1 = np.random.uniform(low=-10000.0, high=10.0, size=(28, 28, 3))
        with tf.device('/cpu:0'):
            input_0 = tf.Variable(initial_value=dummy_input_0, shape=(28, 28, 3), dtype=tf.float32)
            input_1 = tf.Variable(initial_value=dummy_input_1, shape=(28, 28, 3), dtype=tf.float32)
            mask_add = tf.math.add(input_0, input_1, name=mask_add_name)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        transformer_utils.register_attention_mask_override(mask_add_name)
        sim = QuantizationSimModel(sess, [input_0.op.name, input_1.op.name], [mask_add.op.name], use_cuda=False,
                                   default_data_type=QuantizationDataType.float, default_output_bw=16, default_param_bw=16)

        def dummy_forward_pass(sess, args):
            mask_add_tensor = sess.graph.get_operation_by_name(mask_add.op.name + '_quantized').outputs[0]
            sess.run(mask_add_tensor)

        sim.compute_encodings(dummy_forward_pass, None)
        mask_quantizer = sim.quantizer_config(mask_add_name + '_quantized')
        self.assertFalse(mask_quantizer._is_encoding_frozen)
        is_int_data_type = mask_quantizer.get_variable_from_op(QuantizeOpIndices.is_int_data_type)
        self.assertFalse(is_int_data_type)
        # No encoding is computed for float mode
        encoding_min = mask_quantizer.get_variable_from_op(QuantizeOpIndices.encoding_min)
        encoding_max = mask_quantizer.get_variable_from_op(QuantizeOpIndices.encoding_max)
        self.assertEqual(encoding_min, 0)
        self.assertEqual(encoding_max, 0)

        del sim
        sess.close()

    def test_save_model_with_embedded_quantization_nodes(self):
        """
        Create QuantSim for a CPU model, compute encodings, replace quantization nodes and export out a resulting model
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        # Make some changes to model parameters to see if they are part of the exported model
        with sess.graph.as_default():
            first_conv_tensor_var = [var for var in tf.compat.v1.global_variables() if var.name == 'conv2d/kernel:0'][0]
            first_conv_tensor_var.load(np.ones([3,3,3,32]), sess)
            saver = tf.compat.v1.train.Saver()
        saver.save(sess, save_path='/tmp/quantsim/'+'orig_model_before_quantsim')
        sim = QuantizationSimModel(sess, [model.input.op.name], [model.output.op.name], use_cuda=False)

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name(model.output.name)
            model_output = model_output.consumers()[0].outputs[0]
            model_input = sess.graph.get_tensor_by_name(model.input.name)
            dummy_input = np.random.randn(20, 28, 28, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)
        encoding = libpymo.TfEncoding()
        encoding.bw = 8
        encoding.max = 1.0
        sim._param_quantizers['conv2d/Conv2D/ReadVariableOp_quantized'].set_encoding(encoding)

        all_op_types = [op.type for op in sim.session.graph.get_operations()]
        self.assertIn('QcQuantize', all_op_types)
        self.assertNotIn('FakeQuantWithMinMaxVars', all_op_types)

        # Save model without encodings file
        sim.save_model_with_embedded_quantization_nodes(os.path.join('/tmp', 'tf_fakequant_model'))

        new_sess = load_model_from_meta('/tmp/tf_fakequant_model_embedded_quant_nodes.meta')
        first_conv_tensor = new_sess.graph.get_tensor_by_name('conv2d/Conv2D/ReadVariableOp:0')
        first_conv_tensor_val = new_sess.run(first_conv_tensor)
        self.assertTrue(np.any(first_conv_tensor_val == 1))
        first_conv_tensor_fakequant_max_tensor = new_sess.graph.get_tensor_by_name('conv2d/Conv2D/ReadVariableOp_quantized/max:0')
        first_conv_tensor_fakequant_max_val = new_sess.run(first_conv_tensor_fakequant_max_tensor)
        self.assertTrue(first_conv_tensor_fakequant_max_val == 1)

        all_op_types = [op.type for op in new_sess.graph.get_operations()]
        self.assertNotIn('QcQuantize', all_op_types)
        self.assertIn('FakeQuantWithMinMaxVars', all_op_types)

        # Save model with encodings file
        sim._export_encodings('/tmp/tf_fakequant_model.encodings')
        sim.save_model_with_embedded_quantization_nodes(os.path.join('/tmp', 'tf_fakequant_model'), '/tmp/tf_fakequant_model.encodings')

        new_sess = load_model_from_meta('/tmp/tf_fakequant_model_embedded_quant_nodes.meta')
        first_conv_tensor = new_sess.graph.get_tensor_by_name('conv2d/Conv2D/ReadVariableOp:0')
        first_conv_tensor_val = new_sess.run(first_conv_tensor)
        self.assertTrue(np.any(first_conv_tensor_val == 1))
        first_conv_tensor_fakequant_max_tensor = new_sess.graph.get_tensor_by_name('conv2d/Conv2D/ReadVariableOp_quantized/max:0')
        first_conv_tensor_fakequant_max_val = new_sess.run(first_conv_tensor_fakequant_max_tensor)
        self.assertTrue(first_conv_tensor_fakequant_max_val == 1)

        all_op_types = [op.type for op in new_sess.graph.get_operations()]
        self.assertNotIn('QcQuantize', all_op_types)
        self.assertIn('FakeQuantWithMinMaxVars', all_op_types)
        sess.close()
        sim.session.close()
        new_sess.close()
        del sim

    def test_save_model_with_embedded_quantization_nodes_fp16(self):
        """
        Create QuantSim for a CPU model, compute encodings, replace quantization nodes to cast
        and compare the results of the generated session is same as quant sim model
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=False, default_output_bw=16,
                                    default_param_bw=16, default_data_type=QuantizationDataType.float)

        def dummy_forward_pass_quant_sim(sess, dummy_input):
            model_output = sess.graph.get_tensor_by_name('conv2d_1/Relu_quantized:0')
            model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
            return sess.run(model_output, feed_dict={model_input: dummy_input})

        def dummy_forward_pass_quant_embedded_native_nodes(sess, dummy_input):
            model_output = sess.graph.get_tensor_by_name('Cast_11:0')
            model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
            return sess.run(model_output, feed_dict={model_input: dummy_input})

        dummy_input = np.random.randn(20, 28, 28, 3)
        sim.compute_encodings(dummy_forward_pass_quant_sim, dummy_input)

        output_aimet_fk_fp16 = dummy_forward_pass_quant_sim(sim.session, dummy_input)
        orig_sess = sim.save_model_with_embedded_quantization_nodes(os.path.join('data', 'quant_sim_model_fp16'))
        output_embedded_fk_fp16 = dummy_forward_pass_quant_embedded_native_nodes(orig_sess, dummy_input)
        self.assertTrue(np.all(output_embedded_fk_fp16 == output_aimet_fk_fp16))

    def test_model_with_const_param(self):
        """
        Create Quantsim model with a const weight
        """
        tf.compat.v1.reset_default_graph()

        input = tf.keras.Input([28, 28, 3], dtype=tf.float32, name='input')
        const_kernel = tf.constant(np.random.randn(3, 3, 3, 32), dtype=tf.float32)
        conv2d = tf.nn.conv2d(input, const_kernel, strides=[1, 1, 2, 1], padding='VALID', name='conv2d')
        maxpool = tf.keras.layers.MaxPooling2D((2, 2))(conv2d)
        output = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')(maxpool)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, [input.op.name], [output.op.name], use_cuda=False)

        const_kernel_quantizer = 'Const_quantized'
        self.assertTrue(const_kernel_quantizer in sim._param_quantizers)

        del sim
        sess.close()

    def test_model_with_multiple_param_outputs(self):
        """
        Create Quantsim model with param having multiple consumers
        """
        tf.compat.v1.reset_default_graph()

        tf.compat.v1.set_random_seed(0)
        with tf.device('/cpu:0'):
            inputs = tf.keras.Input(shape=(32, 32, 1,))
            conv_op = tf.keras.layers.Conv2D(1, (2, 2))(inputs)
            relu_op = tf.nn.relu(conv_op)
            reshape = tf.keras.layers.Flatten()(relu_op)
            output = tf.keras.layers.Dense(10)(reshape)

        var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        labels_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 10], name='labels')
        loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels_placeholder, logits=output)

        with tf.name_scope('custom_gradient_name'):
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)
            gradients = optimizer.compute_gradients(loss, var_list)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, [inputs.op.name], [output.op.name], use_cuda=True)

        conv2d_weight = sim.session.graph.get_tensor_by_name('conv2d/Conv2D/ReadVariableOp:0')
        conv2d_weight_quantizer = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
        consumers = conv2d_weight.consumers()
        self.assertGreater(len(consumers), 1)
        self.assertTrue(conv2d_weight_quantizer in consumers)

        dense_weight = sim.session.graph.get_tensor_by_name('dense/MatMul/ReadVariableOp:0')
        dense_weight_quantizer = sim.session.graph.get_operation_by_name('dense/MatMul/ReadVariableOp_quantized')
        consumers = dense_weight.consumers()
        self.assertGreater(len(consumers), 1)
        self.assertTrue(dense_weight_quantizer in consumers)

    def test_weight_sharing_quantization(self):
        """
        Test that quantizers are inserted correctly for a model with weight sharing.
        """
        rand_inp = np.random.randn(1, 16, 16, 3)
        w = np.random.randn(2, 2, 3, 8)

        inp = tf.compat.v1.placeholder(tf.float32, shape=(1, 16, 16, 3))
        w_var = tf.Variable(w, dtype=tf.float32)
        c1 = tf.nn.conv2d(inp, w_var, strides=[1, 1, 1, 1], padding='VALID')
        c2 = tf.nn.conv2d(inp, w_var, strides=[1, 1, 1, 1], padding='VALID')
        relu = tf.nn.relu(w_var)
        _ = tf.add(c1, c2)
        session = tf.compat.v1.Session()

        # Even though only one variable was used, TF creates three read variable ops between the variable and the two
        # convs and relu ops. Rewire the inputs of one of the convs and the relu op to be the same as that of the first
        # conv to simulate weight sharing.
        graph_editor.reroute_ts([c1.op.inputs[1]], [c2.op.inputs[1]])
        graph_editor.reroute_ts([c1.op.inputs[1]], [relu.op.inputs[0]])

        # Check that weight was rewired correctly.
        assert c1.op.inputs[1] == c2.op.inputs[1]
        assert c1.op.inputs[1] == relu.op.inputs[0]

        init = tf.compat.v1.global_variables_initializer()
        session.run(init)
        _ = session.run(session.graph.get_tensor_by_name('Add:0'),
                          feed_dict={session.graph.get_tensor_by_name('Placeholder:0'): rand_inp})

        qsim = QuantizationSimModel(session, starting_op_names=['Placeholder'], output_op_names=['Add', 'Relu'])
        new_c1 = qsim.session.graph.get_operation_by_name('Conv2D')
        new_c2 = qsim.session.graph.get_operation_by_name('Conv2D_1')
        new_relu = qsim.session.graph.get_operation_by_name('Relu')
        orig_param_tensor = qsim.session.graph.get_tensor_by_name('Conv2D/ReadVariableOp:0')
        quantized_param_tensor = qsim.session.graph.get_tensor_by_name('Conv2D/ReadVariableOp_quantized:0')

        # Check quantize op was inserted correctly. Quantize op should only be connected to the two conv ops, and not
        # the relu op, since it is not a parameter of the relu.
        assert new_c1.inputs[1] == quantized_param_tensor
        assert new_c2.inputs[1] == quantized_param_tensor
        assert new_relu.inputs[0] == orig_param_tensor
        assert quantized_param_tensor.op.inputs[0] == orig_param_tensor

class TestQuantSimRangeLearning:
    """ Test methods for Quantization Simulation """

    def test_cpu_model_quantize_op_input_params_update(self):

        """
        Create QuantSim for a CPU model, check variable for encoding_min/max, bit_width,
        and use_symmetric_encoding flag are set.
        """

        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=False,
                                   default_output_bw=6,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init)

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('conv2d_1/Relu_quantized:0')

            model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
            np.random.seed(0)
            dummy_input = 7 * np.random.randn(20, 28, 28, 3) + 8
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        # Check if encodings have been calculated
        for name, quantizer in sim._activation_quantizers.items():
            if name not in ['conv2d_input_quantized', 'conv2d/BiasAdd_quantized', 'conv2d_1/BiasAdd_quantized']:
                assert quantizer.tensor_quantizer.isEncodingValid

        # check encoding min and max got updated
        with sim.session.graph.as_default():
            conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
            conv2d_weight_encoding_min = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_min])
            conv2d_weight_encoding_max = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_max])

            conv2d_output_quant_op = sim.session.graph.get_operation_by_name('conv2d/Relu_quantized')
            conv2d_output_encoding_min = sim.session.run(conv2d_output_quant_op.inputs[QuantizeOpIndices.encoding_min])
            conv2d_output_encoding_max = sim.session.run(conv2d_output_quant_op.inputs[QuantizeOpIndices.encoding_max])

        # check correct bit-widths are set for param and output quantize ops
        # inputs[3] : min, inputs[4] : max, inputs[5] bit_width, inputs[6] use_symmetric_encoding flag
        assert 8 == sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.bit_width])
        assert 6 == sim.session.run(conv2d_output_quant_op.inputs[QuantizeOpIndices.bit_width])
        assert sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.use_symmetric_encoding])
        assert not sim.session.run(conv2d_output_quant_op.inputs[QuantizeOpIndices.use_symmetric_encoding])

        # check encodings are being set
        assert conv2d_weight_encoding_min != 0.0
        assert conv2d_weight_encoding_max != 0.0
        assert conv2d_output_encoding_min == 0.0
        assert conv2d_output_encoding_max != 0.0

        sess.close()
        sim.session.close()

    @pytest.mark.cuda
    def test_gpu_model_quantize_op_input_params_update(self):
        """
        Create QuantSim for a GPU model and test that encoding_min/max,
        bit_width and use_symmetric_encoding params are getting set.
        """

        tf.compat.v1.reset_default_graph()
        with tf.device('/gpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=True,
                                   default_output_bw=4, default_param_bw=6,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   )

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('conv2d_1/Relu_quantized:0')

            model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
            np.random.seed(0)
            dummy_input = 7 * np.random.randn(20, 28, 28, 3) + 8
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        # Check if encodings have been calculated
        for name, quantizer in sim._activation_quantizers.items():
            if name not in ['conv2d_input_quantized', 'conv2d/BiasAdd_quantized', 'conv2d_1/BiasAdd_quantized']:
                assert quantizer.tensor_quantizer.isEncodingValid

        # check encoding min and max got updated
        # inputs[3] : min, inputs[4] : max, inputs[5] bit_width, inputs[6] use_symmetric_encoding flag
        with sim.session.graph.as_default():
            conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
            conv2d_weight_encoding_min = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_min])
            conv2d_weight_encoding_max = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_max])

            conv2d_output_quant_op = sim.session.graph.get_operation_by_name('conv2d/Relu_quantized')
            conv2d_output_encoding_min = sim.session.run(conv2d_output_quant_op.inputs[QuantizeOpIndices.encoding_min])
            conv2d_output_encoding_max = sim.session.run(conv2d_output_quant_op.inputs[QuantizeOpIndices.encoding_max])

        # check correct bit-width/ use_symmetric_encoding flag are set for param and output quantize ops
        assert 6 == sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.bit_width])
        assert 4 == sim.session.run(conv2d_output_quant_op.inputs[QuantizeOpIndices.bit_width])
        assert sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.use_symmetric_encoding])
        assert not sim.session.run(conv2d_output_quant_op.inputs[QuantizeOpIndices.use_symmetric_encoding])

        # check encodings are being set
        assert conv2d_weight_encoding_min != 0.0
        assert conv2d_weight_encoding_max != 0.0
        assert conv2d_output_encoding_min == 0.0
        assert conv2d_output_encoding_max != 0.0

        sess.close()
        sim.session.close()

    @staticmethod
    def _compute_gradients_numpy(x, encoding_min, encoding_max, bitwidth, use_symmetric_endcoding):
        """

        :return:
        """
        # x = x.astype(np.float32)
        # compute steps, scaling and offset
        steps = (np.power(2, bitwidth) - 1).astype(np.float32)
        scaling = ((encoding_max - encoding_min) / steps).astype(np.float32)
        offset = (np.round(encoding_min / scaling)).astype(np.float32)

        # R(x/s) + R(o)
        r_x_by_s_plus_round_o = (np.round(x / scaling) + np.round(offset)).astype(np.float32)
        r_x_by_s_minus_x_by_s = (np.round(x / scaling) - x / scaling).astype(np.float32)

        # find n and p
        n = 0.0
        if use_symmetric_endcoding:
            n = -1 * (np.power(2, bitwidth) + 1).astype(np.float32)
        p = (np.power(2, bitwidth) - 1).astype(np.float32)

        # compute dq_by_dmax
        dq_by_dmax = []
        dq_by_dx = [0.0, 0.0, 0.0, 0.0]
        r_x_by_s_plus_round_o_flat = np.ndarray.flatten(r_x_by_s_plus_round_o)
        r_x_by_s_minus_x_by_s_flat = np.ndarray.flatten(r_x_by_s_minus_x_by_s)

        # instead of this, we could also use np.where here
        for i, each_elem in enumerate(r_x_by_s_plus_round_o_flat):
            if n <= each_elem <= p:
                dq_by_dmax.append(r_x_by_s_minus_x_by_s_flat[i] * 1.0 / steps)
                dq_by_dx[i] = 1.0
            elif each_elem < n:
                dq_by_dmax.append(n * 1.0 / steps)
            else:
                dq_by_dmax.append(p * 1.0 / steps)

        dq_by_dmax_reduced = np.sum(dq_by_dmax).astype(np.float32)

        # return gradients w.r.t input, min and max as scalars
        return dq_by_dx, -(dq_by_dmax_reduced.astype(np.float32)), dq_by_dmax_reduced.astype(np.float32)

    @pytest.mark.tf1
    def test_qc_custom_gradient_backward_pass(self):
        """
        test to validate custom gradient computed against numpy computations for
        quant sim model with quant scheme set to range learning.
        """

        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        with tf.device('/cpu:0'):
            inputs = tf.keras.Input(shape=(2, 2, 1,))
            conv_op = tf.keras.layers.Conv2D(1, (2, 2),
                                             kernel_initializer=tf.random_uniform_initializer(-1, 2),
                                             bias_initializer='random_uniform')(inputs)
            _ = tf.nn.relu(conv_op)

        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['input_1'], ['Relu'], use_cuda=False,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init)
        np.random.seed(0)

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('Relu_quantized:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            shape = model_input.shape
            dummy_input = np.random.randn(1, shape[1], shape[2], shape[3])
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        with sim.session.graph.as_default():
            conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
            conv2d_weight_quant_op_output = sim.session.graph.get_tensor_by_name(
                'conv2d/Conv2D/ReadVariableOp_quantized:0')

        inp_tensor = conv2d_weight_quant_op.inputs[0]
        np.random.seed(0)
        w_shape = inp_tensor.shape
        inp_data = np.random.rand(2, w_shape[1], w_shape[2], w_shape[3])

        grads = tf.gradients(conv2d_weight_quant_op_output, [inp_tensor,
                                                             conv2d_weight_quant_op.inputs[
                                                                 QuantizeOpIndices.encoding_min],
                                                             conv2d_weight_quant_op.inputs[
                                                                 QuantizeOpIndices.encoding_max]])

        assert len(grads) == 3
        dlossbydx, dlossbydmin, dlossbydmax = grads
        assert (dlossbydx is not None)
        assert (dlossbydmin is not None)
        assert (dlossbydmax is not None)

        with sim.session.graph.as_default():
            enc_min = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_min])
            enc_max = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_max])
            bw = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.bit_width])
            use_symmetric_encoding = sim.session.run(
                conv2d_weight_quant_op.inputs[QuantizeOpIndices.use_symmetric_encoding])

            min_gradient = sim.session.run([dlossbydmin], feed_dict={inp_tensor: inp_data})[0]
            max_gradient = sim.session.run([dlossbydmax], feed_dict={inp_tensor: inp_data})[0]
            input_gradient = sim.session.run([dlossbydx], feed_dict={inp_tensor: inp_data})[0]

        numpy_dq_by_dx, numpy_min_grad, numpy_max_grad = \
            TestQuantSimRangeLearning._compute_gradients_numpy(inp_data.reshape(-1), enc_min, enc_max, bw,
                                                               use_symmetric_encoding)

        # check against numpy computed values
        assert np.isclose(numpy_min_grad.astype(float), min_gradient, atol=1e-06)
        assert np.isclose(numpy_max_grad.astype(float), max_gradient, atol=1e-06)
        assert np.allclose(numpy_dq_by_dx, input_gradient)

        sess.close()
        sim.session.close()

    def test_n_p_computation(self):
        """
        validate n and p values computed for symmetric and asymmetric case.
        :return:
        """

        tf.compat.v1.reset_default_graph()
        bitwidth = 8

        sym_n, sym_p = _get_n_and_p(bitwidth, tf.cast(True, tf.bool))
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        # for 8 bit , -127 to +127
        expected_sym_n = (-2 ** (bitwidth - 1)) + 1
        expected_sym_p = (2 ** (bitwidth - 1)) - 1

        comp_symmetric_n = sess.run(sym_n)
        comp_symmetric_p = sess.run(sym_p)

        asym_n, asym_p = _get_n_and_p(bitwidth, tf.cast(False, tf.bool))
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        # for 8 bit , 0 to 255
        expected_asym_n = 0
        expected_asym_p = (2 ** bitwidth) - 1
        comp_asymmetric_n = sess.run(asym_n)
        comp_asymmetric_p = sess.run(asym_p)

        assert expected_asym_n == comp_asymmetric_n
        assert expected_asym_p == comp_asymmetric_p

        assert expected_sym_n == comp_symmetric_n
        assert expected_sym_p == comp_symmetric_p

        sess.close()

    def test_compute_dloss_by_dmax_shape(self):
        """ Test compute_dloss_by_dmax returns tensor with correct shape """
        tf.compat.v1.set_random_seed(0)

        # Per tensor case
        tf.compat.v1.reset_default_graph()
        graph = tf.Graph()
        sess = tf.compat.v1.Session(graph=graph)
        with graph.as_default():
            inputs = tf.random.uniform(shape=[3, 3, 4, 2], dtype=tf.float32)
            grad = tf.random.uniform(shape=[3, 3, 4, 2], dtype=tf.float32)
            scaling = tf.random.uniform(shape=[], dtype=tf.float32)
            offset = tf.random.uniform(shape=[], dtype=tf.float32)
            bitwidth = tf.constant(8.0, dtype=tf.float32)
            is_symmetric = tf.constant(False)

            intermediate_result = compute_intermediate_result_for_learned_grid(inputs, scaling, offset)
            n, p = _get_n_and_p(bitwidth, is_symmetric)
            grid_params = LearnedGridParams(scaling, offset, n, p)
            dloss_by_dmax = _compute_dloss_by_dmax(inputs, grad, intermediate_result, grid_params)
            assert sess.run(dloss_by_dmax).shape == ()

        # Per channel case with weights
        tf.compat.v1.reset_default_graph()
        graph = tf.Graph()
        sess = tf.compat.v1.Session(graph=graph)
        with graph.as_default():
            inputs = tf.random.uniform(shape=[3, 3, 4, 2], dtype=tf.float32)
            grad = tf.random.uniform(shape=[3, 3, 4, 2], dtype=tf.float32)
            scaling = tf.random.uniform(shape=[2,], dtype=tf.float32)
            offset = tf.random.uniform(shape=[2,], dtype=tf.float32)
            bitwidth = tf.constant(8.0, dtype=tf.float32)
            is_symmetric = tf.constant(False)

            intermediate_result = compute_intermediate_result_for_learned_grid(inputs, scaling, offset)
            n, p = _get_n_and_p(bitwidth, is_symmetric)
            grid_params = LearnedGridParams(scaling, offset, n, p)
            dloss_by_dmax = _compute_dloss_by_dmax(inputs, grad, intermediate_result, grid_params)
            assert sess.run(dloss_by_dmax).shape == (2,)

        # Per channel case with bias
        tf.compat.v1.reset_default_graph()
        graph = tf.Graph()
        sess = tf.compat.v1.Session(graph=graph)
        with graph.as_default():
            inputs = tf.random.uniform(shape=[10,], dtype=tf.float32)
            grad = tf.random.uniform(shape=[10,], dtype=tf.float32)
            scaling = tf.random.uniform(shape=[10,], dtype=tf.float32)
            offset = tf.random.uniform(shape=[10,], dtype=tf.float32)
            bitwidth = tf.constant(8.0, dtype=tf.float32)
            is_symmetric = tf.constant(False)

            intermediate_result = compute_intermediate_result_for_learned_grid(inputs, scaling, offset)
            n, p = _get_n_and_p(bitwidth, is_symmetric)
            grid_params = LearnedGridParams(scaling, offset, n, p)
            dloss_by_dmax = _compute_dloss_by_dmax(inputs, grad, intermediate_result, grid_params)
            assert sess.run(dloss_by_dmax).shape == (10,)

    def test_qat_fp16(self, iterations=5):
        """
        test qat fp16
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        np.random.seed(0)
        with tf.device('/cpu:0'):
            inputs = tf.keras.Input(shape=(32, 32, 4,))
            conv_op = tf.keras.layers.Conv2D(2, (3, 3),
                                             kernel_initializer=tf.random_uniform_initializer(-1, 2),
                                             bias_initializer='random_uniform',
                                             padding='SAME')(inputs)
            relu_op = tf.nn.relu(conv_op)
            reshape = tf.keras.layers.Flatten()(relu_op)
            _ = tf.keras.layers.Dense(10, bias_initializer='random_uniform')(reshape)

        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        # create quantsim model without config file
        sim = QuantizationSimModel(sess, ['input_1'], ['dense/BiasAdd'], use_cuda=False,
                                   quant_scheme=QuantScheme.post_training_tf, default_output_bw=16, default_param_bw=16,
                                   default_data_type=QuantizationDataType.float)

        def dummy_forward_pass(sess, _):
            model_output = sess.graph.get_tensor_by_name('dense/BiasAdd_quantized:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            shape = model_input.shape
            dummy_input = np.random.randn(1, shape[1], shape[2], shape[3])
            sess.run(model_output, feed_dict={model_input: dummy_input})

        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')

        # enable input
        sim.compute_encodings(dummy_forward_pass, None)

        inp_tensor = sim.session.graph.get_tensor_by_name('input_1:0')
        w_shape = inp_tensor.shape
        batches = 32
        inp_data = np.random.rand(batches, w_shape[1], w_shape[2], w_shape[3])
        logits = sim.session.graph.get_tensor_by_name('dense/BiasAdd_quantized:0')

        labels = np.random.randint(10, size=batches)
        one_hot_labels = np.eye(10)[labels]

        with sim.session.graph.as_default():
            var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            labels_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 10], name='labels')
            loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels_placeholder, logits=logits)

            update_ops = []
            global_step = tf.compat.v1.train.create_global_step()
            initialize_uninitialized_vars(sim.session)

            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)
            gradients = optimizer.compute_gradients(loss, var_list)

            grad_updates = optimizer.apply_gradients(gradients, global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)

            with tf.control_dependencies([update_op]):
                train_op = tf.identity(loss, name='train_op')

            # start training
            for _ in range(iterations):
                weights_before_train = sim.session.run(conv2d_weight_quant_op.inputs[0])
                _ = sim.session.run(train_op, feed_dict={inp_tensor: inp_data, labels_placeholder: one_hot_labels})
                weights_after_train = sim.session.run(conv2d_weight_quant_op.inputs[0])
                assert np.allclose(weights_before_train, weights_after_train, atol=1e-2)
                assert not np.allclose(weights_before_train, weights_after_train, atol=1e-3)


    def test_qc_custom_gradient_training_loop_range_learning(self, iterations=1):
        """
        test to get average time spent in range learning grad
        """

        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        with tf.device('/cpu:0'):
            inputs = tf.keras.Input(shape=(32, 32, 1,))
            conv_op = tf.keras.layers.Conv2D(1, (2, 2),
                                             kernel_initializer=tf.random_uniform_initializer(-1, 2),
                                             bias_initializer='random_uniform')(inputs)
            relu_op = tf.nn.relu(conv_op)
            reshape = tf.keras.layers.Flatten()(relu_op)
            _ = tf.keras.layers.Dense(10)(reshape)

        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        sim = QuantizationSimModel(sess, ['input_1'], ['dense/BiasAdd'], use_cuda=False,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init)

        np.random.seed(0)

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('dense/MatMul:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            shape = model_input.shape
            dummy_input = np.random.randn(1, shape[1], shape[2], shape[3])
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        inp_tensor = sim.session.graph.get_tensor_by_name('input_1:0')
        np.random.seed(0)
        w_shape = inp_tensor.shape
        batches = 32
        inp_data = np.random.rand(batches, w_shape[1], w_shape[2], w_shape[3])
        logits = sim.session.graph.get_tensor_by_name('dense/MatMul:0')

        labels = np.random.randint(10, size=batches)
        one_hot_labels = np.eye(10)[labels]

        with sim.session.graph.as_default():
            var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            labels_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 10], name='labels')
            loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels_placeholder, logits=logits)

            update_ops = []
            global_step = tf.compat.v1.train.create_global_step()
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)
            gradients = optimizer.compute_gradients(loss, var_list)

            init_global = tf.compat.v1.global_variables_initializer()
            init_local = tf.compat.v1.local_variables_initializer()
            init = tf.group(init_global, init_local)
            sim.session.run(init)

            grad_updates = optimizer.apply_gradients(gradients, global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)

            with tf.control_dependencies([update_op]):
                train_op = tf.identity(loss, name='train_op')

            # start training
            time_taken_by_default_grad = 0
            for i in range(iterations):
                start_time = time.perf_counter()
                _ = sim.session.run(train_op, feed_dict={inp_tensor: inp_data, labels_placeholder: one_hot_labels})
                exec_time = time.perf_counter() - start_time
            time_taken_by_default_grad = time_taken_by_default_grad + exec_time

        default_grad_avg_time = time_taken_by_default_grad / iterations
        print('Avg time taken by custom grad', default_grad_avg_time)
        sess.close()
        sim.session.close()
        return default_grad_avg_time

    def test_qc_custom_gradient_training_loop_pass_through(self, iterations=1):
        """
        test to get average time spent with pass through grad
        """

        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        with tf.device('/cpu:0'):
            inputs = tf.keras.Input(shape=(32, 32, 1,))
            conv_op = tf.keras.layers.Conv2D(1, (2, 2),
                                             kernel_initializer=tf.random_uniform_initializer(-1, 2),
                                             bias_initializer='random_uniform')(inputs)
            relu_op = tf.nn.relu(conv_op)
            reshape = tf.keras.layers.Flatten()(relu_op)
            _ = tf.keras.layers.Dense(10)(reshape)

        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        sim = QuantizationSimModel(sess, ['input_1'], ['dense/BiasAdd'], use_cuda=False,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_enhanced_init)

        np.random.seed(0)

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('dense/MatMul:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            shape = model_input.shape
            dummy_input = np.random.randn(1, shape[1], shape[2], shape[3])
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        inp_tensor = sim.session.graph.get_tensor_by_name('input_1:0')
        np.random.seed(0)
        w_shape = inp_tensor.shape
        batches = 32
        inp_data = np.random.rand(batches, w_shape[1], w_shape[2], w_shape[3])
        logits = sim.session.graph.get_tensor_by_name('dense/MatMul:0')

        labels = np.random.randint(10, size=batches)
        one_hot_labels = np.eye(10)[labels]

        with sim.session.graph.as_default():
            var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            labels_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 10], name='labels')
            loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels_placeholder, logits=logits)

            update_ops = []
            global_step = tf.compat.v1.train.create_global_step()
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)
            gradients = optimizer.compute_gradients(loss, var_list)

            init_global = tf.compat.v1.global_variables_initializer()
            init_local = tf.compat.v1.local_variables_initializer()
            init = tf.group(init_global, init_local)
            sim.session.run(init)

            grad_updates = optimizer.apply_gradients(gradients, global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)

            with tf.control_dependencies([update_op]):
                train_op = tf.identity(loss, name='train_op')

            # start training
            time_taken_by_default_grad = 0
            for i in range(iterations):
                start_time = time.perf_counter()
                _ = sim.session.run(train_op, feed_dict={inp_tensor: inp_data, labels_placeholder: one_hot_labels})
                exec_time = time.perf_counter() - start_time
                time_taken_by_default_grad = time_taken_by_default_grad + exec_time

        default_grad_avg_time = time_taken_by_default_grad / iterations
        print('Avg time taken by custom grad', default_grad_avg_time)

        sess.close()
        sim.session.close()
        return default_grad_avg_time

    def test_accumulator_overflow(self):
        """ Test check accumulator overflow utility """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(8, 8, 3,))
            _ = tf.keras.layers.Conv2D(10, (2, 2), kernel_initializer=tf.constant_initializer(255))(inputs)
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

        layer, range_used = check_accumulator_overflow(sess, 8, 27)
        assert 'conv2d/Conv2D' == layer
        assert 100 * range_used == pytest.approx(0.5836, 0.001)
        sess.close()

    def _compare_range_learning_with_default_grad(self):
        """
        Test to compare time taken by range learning grad and default grad.
        There is no validation criterion for this test. It is only for study.
        :return:
        """

        iterations = 10
        pass_through_grad_avg_time = self.test_qc_custom_gradient_training_loop_pass_through(iterations)
        range_learning_avg_time = self.test_qc_custom_gradient_training_loop_range_learning(iterations)
        print('% increase ', ((range_learning_avg_time - pass_through_grad_avg_time)
                              / pass_through_grad_avg_time) * 100)

    def test_qc_custom_gradient_training_loop_param_learning(self):
        """
        Test to validate gradient is updating for param quantizer ops
        with bias add skipped
        """

        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        np.random.seed(0)
        with tf.device('/cpu:0'):
            inputs = tf.keras.Input(shape=(32, 32, 1,))
            conv_op = tf.keras.layers.Conv2D(1, (2, 2),
                                             kernel_initializer=tf.random_uniform_initializer(-1, 2),
                                             bias_initializer='random_uniform',
                                             padding='SAME')(inputs)
            relu_op = tf.nn.relu(conv_op)
            reshape = tf.keras.layers.Flatten()(relu_op)
            _ = tf.keras.layers.Dense(10)(reshape)

        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        # create quantsim model without config file
        sim = QuantizationSimModel(sess, ['input_1'], ['dense/BiasAdd'], use_cuda=False,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init)

        for quant_op_name in sim._param_quantizers.keys():
            print(sim._param_quantizers[quant_op_name])

        for quant_op_name in sim._activation_quantizers.keys():
            print(sim._activation_quantizers[quant_op_name])

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('dense/MatMul:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            shape = model_input.shape
            dummy_input = np.random.randn(1, shape[1], shape[2], shape[3])
            sess.run(model_output, feed_dict={model_input: dummy_input})

        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
        relu_output_quant_op = sim.session.graph.get_operation_by_name('Relu_quantized')

        # enable input
        sim.compute_encodings(dummy_forward_pass, None)

        inp_tensor = sim.session.graph.get_tensor_by_name('input_1:0')
        np.random.seed(0)
        w_shape = inp_tensor.shape
        batches = 32
        inp_data = np.random.rand(batches, w_shape[1], w_shape[2], w_shape[3])
        logits = sim.session.graph.get_tensor_by_name('dense/MatMul:0')

        labels = np.random.randint(10, size=batches)
        one_hot_labels = np.eye(10)[labels]

        with sim.session.graph.as_default():
            var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            labels_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 10], name='labels')
            loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels_placeholder, logits=logits)

            update_ops = []
            global_step = tf.compat.v1.train.create_global_step()
            initialize_uninitialized_vars(sim.session)

            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)
            gradients = optimizer.compute_gradients(loss, var_list)

            sim.compute_encodings(dummy_forward_pass, None)
            grad_updates = optimizer.apply_gradients(gradients, global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)

            conv_inp_tensor = conv2d_weight_quant_op.inputs[0]
            grads = tf.gradients(loss, [conv_inp_tensor,
                                        conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_min],
                                        conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_max]])
            dqbydx, dqbydmin, dqbydmax = grads
            input_gradient = sim.session.run([dqbydx], feed_dict={inp_tensor: inp_data,
                                                                  labels_placeholder: one_hot_labels})[0]
            min_gradient = sim.session.run([dqbydmin], feed_dict={inp_tensor: inp_data,
                                                                  labels_placeholder: one_hot_labels})[0]
            max_gradient = sim.session.run([dqbydmax], feed_dict={inp_tensor: inp_data,
                                                                  labels_placeholder: one_hot_labels})[0]

            weights_before_train = sim.session.run(conv2d_weight_quant_op.inputs[0])
            encoding_min_before_train = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_min])
            encoding_max_before_train = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_max])
            relu_output_encoding_min_before_train = sim.session.run(relu_output_quant_op.inputs[
                                                                        QuantizeOpIndices.encoding_min])
            relu_output_encoding_max_before_train = sim.session.run(relu_output_quant_op.inputs[
                                                                        QuantizeOpIndices.encoding_max])
            with tf.control_dependencies([update_op]):
                train_op = tf.identity(loss, name='train_op')

            for quant_op_name in sim._param_quantizers.keys():
                print(quant_op_name + '_min_before_train = ' + str(sim.session.run(
                    sim.session.graph.get_operation_by_name(quant_op_name).inputs[QuantizeOpIndices.encoding_min])))
                print(quant_op_name + '_max_before_train = ' + str(sim.session.run(
                    sim.session.graph.get_operation_by_name(quant_op_name).inputs[QuantizeOpIndices.encoding_max])))

            # start training
            _ = sim.session.run(train_op, feed_dict={inp_tensor: inp_data, labels_placeholder: one_hot_labels})

            for quant_op_name in sim._param_quantizers.keys():
                print(quant_op_name + '_min = ' + str(sim.session.run(sim.session.graph.get_operation_by_name
                                                                      (quant_op_name).inputs[
                                                                          QuantizeOpIndices.encoding_min])))
                print(quant_op_name + '_max = ' + str(sim.session.run(sim.session.graph.get_operation_by_name
                                                                      (quant_op_name).inputs[
                                                                          QuantizeOpIndices.encoding_max])))

            weights_after_train = sim.session.run(conv2d_weight_quant_op.inputs[0])
            relu_output_encoding_min_after_train = sim.session.run(relu_output_quant_op.inputs[
                                                                       QuantizeOpIndices.encoding_min])
            relu_output_encoding_max_after_train = sim.session.run(relu_output_quant_op.inputs[
                                                                       QuantizeOpIndices.encoding_max])
            encoding_min_after_train = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_min])
            encoding_max_after_train = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_max])

            assert not np.allclose(weights_before_train, weights_after_train, atol=1e-6)
            assert encoding_min_before_train != encoding_min_after_train
            assert encoding_max_before_train != encoding_max_after_train
            assert relu_output_encoding_min_before_train != relu_output_encoding_min_after_train
            assert relu_output_encoding_max_before_train != relu_output_encoding_max_after_train

        sess.close()
        sim.session.close()
