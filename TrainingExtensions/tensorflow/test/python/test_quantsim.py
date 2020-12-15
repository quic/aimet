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

import pytest
import unittest
import shutil

import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.logging.WARN)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import json
import libpymo
import time
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils.graph_saver import load_model_from_meta
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.examples.test_models import model_with_dtype_int, keras_model_functional
from aimet_common.defs import QuantScheme
from aimet_tensorflow.quantsim import save_checkpoint, load_checkpoint
from aimet_tensorflow.utils.constants import QuantizeOpIndices


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
        for name, quantizer in sim._activation_quantizers.items():
            if name in deactivated_quantizers:
                self.assertTrue(int(libpymo.TensorQuantizerOpMode.passThrough),
                                sim.session.run(name + '_op_mode/read:0'))
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
        for name, quantizer in sim._activation_quantizers.items():
            if name in deactivated_quantizers:
                self.assertTrue(int(libpymo.TensorQuantizerOpMode.passThrough),
                                sim.session.run(name + '_op_mode/read:0'))
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
        self.assertFalse(sym_encoding)
        bias_quantizer.use_symmetric_encoding = True
        sym_encoding = bias_quantizer.use_symmetric_encoding
        self.assertTrue(sym_encoding)

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

    def test_manual_quantize(self):
        """ Test quantizing a model by manually specifying ops to quantize """
        def get_manual_activations(_graph, _starting_ops, _ending_ops):
            """
            Overriding function for getting a list of ops to insert activation quantizers for
            :param _graph: Unused argument
            :param _starting_ops: Unused argument
            :param _ending_ops: Unused argument
            :return: List of ops to insert activation quantizers for, None for placeholder
            """
            return ['conv2d/Relu'], None

        def get_manual_params(_graph, _starting_ops, _ending_ops):
            """
            Overriding function for getting a list of ops to insert param quantizers for
            :param _graph: Unused argument
            :param _starting_ops: Unused argument
            :param _ending_ops: Unused argument
            :return: List of ops to insert param quantizers for, and list of param indices for these ops
            """
            return ['conv2d_1/Conv2D'], [1]

        def configure_quantization_ops(self, _conn_graph, _ops_with_param_names, _indices, _activation_op_names,
                                       _config_file):
            """
            Overriding function for configuring quantization ops inserted by QuantizationSimModel
            :param self: Self refers to QuantizationSimModel object
            :param _conn_graph: Unused argument
            :param _ops_with_param_names: Unused argument
            :param _indices: Unused argument
            :param _activation_op_names: Unused argument
            :param _config_file: Unused argument
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

    def test_insert_quant_op_recurrent(self):

        """ test insertion of quant ops to recurrent layer with conditional blocks """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            # Add an RNN layer
            x = tf.keras.layers.SimpleRNN(12)(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="simplernn_model")(x)

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        ops = sess.graph.get_operations()
        quant_op_inside_while_block_name = "simple_rnn/while/MatMul/ReadVariableOp_quantized"
        self.assertFalse(quant_op_inside_while_block_name in [op.name for op in ops])

        # construct a quantization sim model
        sim = QuantizationSimModel(sess, ['input_1'], ['simplernn_model/Softmax'], use_cuda=False)

        while_matmul_weight_read_tensor_name = "simple_rnn/while/MatMul/ReadVariableOp:0"
        while_matmul_weight_read_tensor = sim.session.graph.get_tensor_by_name(while_matmul_weight_read_tensor_name)
        op_mode = libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize

        # get ops and make sure we have a quantized op added to the conditional block
        ops = sim.session.graph.get_operations()
        self.assertTrue(quant_op_inside_while_block_name in [op.name for op in ops])

    def test_compute_encodings(self):
        """ Test that ops not evaluated during compute encodings are set to passThrough mode. """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        test_inp = np.ndarray((1, 32, 32, 3))

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
                elif name in ['input_1_quantized', 'scope_1/conv2d_3/BiasAdd_quantized']:
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

    def test_matmul_param_selection_lstm(self):
        """ Test apis to select input params to MatMuls within LSTM for quantization """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            x = tf.keras.layers.LSTM(12, name='lstm0')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="matmul0")(x)

            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            # _ = tf.compat.v1.summary.FileWriter('./lstm', sess.graph)

            matmul_with_split_inside_lstm = "lstm0/while/MatMul"
            tf_split_op_in = sess.graph.get_operation_by_name("lstm0/while/split")
            tf_matmul_with_split_inside_lstm = sess.graph.get_operation_by_name(matmul_with_split_inside_lstm)
            param_in_through_split = sess.graph.get_tensor_by_name("lstm0/while/split/ReadVariableOp:0")

            can_modify_op, param_in = QuantizationSimModel._get_op_to_modify_with_param_in(
                tf_matmul_with_split_inside_lstm, 1)

            self.assertEqual(can_modify_op, tf_split_op_in)
            self.assertEqual(param_in, param_in_through_split)

            matmul_with_slice_inside_lstm = "lstm0/while/MatMul_5"
            tf_strided_slice_op_in = sess.graph.get_operation_by_name("lstm0/while/strided_slice_1")
            tf_matmul_with_slice_inside_lstm = sess.graph.get_operation_by_name(matmul_with_slice_inside_lstm)
            param_in_through_strided_slice = sess.graph.get_tensor_by_name("lstm0/while/ReadVariableOp_1:0")

            can_modify_op, param_in = QuantizationSimModel._get_op_to_modify_with_param_in(
                tf_matmul_with_slice_inside_lstm, 1)

            self.assertEqual(can_modify_op, tf_strided_slice_op_in)
            self.assertEqual(param_in, param_in_through_strided_slice)

    def validate_simple_rnn_auto_insertion_and_forward_pass(self, sess):
        """
        common api to validate auto quant node insertion and forward pass for simple rnn layer
        :param sess: TensorFlow session
        :return:
        """

        np.random.seed(0)
        tf.set_random_seed(0)

        ops = sess.graph.get_operations()
        matmul_param_quant_op_inside_while_block_name = "simple_rnn/while/MatMul/ReadVariableOp_quantized"
        self.assertFalse(matmul_param_quant_op_inside_while_block_name in [op.name for op in ops])
        # _ = tf.summary.FileWriter('./test_simple_rnn_keras', sess.graph)
        # construct a quantization sim model
        sim = QuantizationSimModel(sess, ['input_1'], ['simplernn_model/Softmax'], use_cuda=False)

        # params that must have quantizers
        matmul_2_param_quant_op_inside_while_block_name = "simple_rnn/while/MatMul_1/ReadVariableOp_quantized"
        # check biasadd param quantizers are disabled
        param_quantizers = sim._param_quantizers
        for p_quantizer in param_quantizers.keys():
            if 'BiasAdd' in p_quantizer:
                p_quant_config = sim.quantizer_config(p_quantizer)
                self.assertFalse(p_quant_config.enabled)

        # activations with quantizers
        activation_bias_add_op_inside_while_block_name = "simple_rnn/while/BiasAdd_quantized"
        add_op_inside_while_block_name = "simple_rnn/while/add_quantized"

        # these should not have activation quantizers
        activation_matmul_op_inside_while_block_name = "simple_rnn/while/MatMul_quantized"
        activation_matmul_2_op_inside_while_block_name = "simple_rnn/while/MatMul_1_quantized"

        # get ops and make sure we have a quantized op added to the conditional block
        quantized_graph_op_names = self._get_quant_ops_from_tf_graph(sim.session.graph)

        # while block ops
        # bias and kernel quantizers
        self.assertTrue(matmul_param_quant_op_inside_while_block_name in quantized_graph_op_names)
        self.assertTrue(matmul_2_param_quant_op_inside_while_block_name in quantized_graph_op_names)

        # output quantizers
        self.assertFalse(activation_bias_add_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(add_op_inside_while_block_name in quantized_graph_op_names)

        self.assertFalse(activation_matmul_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_2_op_inside_while_block_name in quantized_graph_op_names)

        # check for input quantizers
        input_matmul_op_inside_while_block_name = "simple_rnn/while/TensorArrayReadV3_quantized"
        input_matmul_2_op_inside_while_block_name = "simple_rnn/while/Identity_2_quantized"
        self.assertTrue(input_matmul_op_inside_while_block_name in quantized_graph_op_names)
        self.assertTrue(input_matmul_2_op_inside_while_block_name in quantized_graph_op_names)

        # validate encodings
        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('simplernn_model/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(16, 3, 100)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        # check encoding min and max got updated
        with sim.session.graph.as_default():
            matmul_weight_quant_op = sim.session.graph.get_operation_by_name('simple_rnn/while/MatMul/ReadVariableOp_quantized')
            matmul_1_weight_quant_op = sim.session.graph.get_operation_by_name('simple_rnn/while/MatMul_1/ReadVariableOp_quantized')

            matmul_weight_quant_op_encoding_min = sim.session.run(matmul_weight_quant_op.inputs[
                                                                      QuantizeOpIndices.encoding_min])
            matmul_weight_quant_op_encoding_max = sim.session.run(matmul_weight_quant_op.inputs[
                                                                      QuantizeOpIndices.encoding_max])

            matmul_1_weight_quant_op_encoding_min = sim.session.run(matmul_1_weight_quant_op.inputs[
                                                                        QuantizeOpIndices.encoding_min])
            matmul_1_weight_quant_op_encoding_max = sim.session.run(matmul_1_weight_quant_op.inputs[
                                                                        QuantizeOpIndices.encoding_max])

        # check weight encodings are set correctly (we have fixed the TF seed above)
        # self.assertEqual(matmul_weight_quant_op_encoding_min, -0.22920194268226624)
        # self.assertEqual(matmul_weight_quant_op_encoding_max, 0.2310066819190979)
        # self.assertEqual(matmul_1_weight_quant_op_encoding_min, -0.73015958070755)
        # self.assertEqual(matmul_1_weight_quant_op_encoding_max, 0.7359088063240051)

        # close tf sessions
        sess.close()
        sim.session.close()
        sim.session.close()

    def test_insert_quant_op_forward_pass_simple_rnn(self):

        """ test insertion of quant ops to recurrent layer with conditional blocks """

        tf.reset_default_graph()
        np.random.seed(0)
        tf.set_random_seed(0)
        sess = tf.Session()

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer
            x = tf.keras.layers.SimpleRNN(12,
                                          kernel_initializer='glorot_uniform',
                                          recurrent_initializer='orthogonal')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="simplernn_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)

        self.validate_simple_rnn_auto_insertion_and_forward_pass(sess)
        sess.close()

    def test_insert_quant_op_forward_pass_simple_rnn_with_relu(self):

        """ test insertion of quant ops to simple rnn with relu """

        tf.reset_default_graph()
        np.random.seed(0)
        tf.set_random_seed(0)
        sess = tf.Session()

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer
            x = tf.keras.layers.SimpleRNN(12, activation='relu',
                                          kernel_initializer='glorot_uniform',
                                          recurrent_initializer='orthogonal')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="simplernn_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)

        self.validate_simple_rnn_auto_insertion_and_forward_pass(sess)
        sess.close()

    def test_insert_quant_op_forward_pass_simple_rnn_multiple_layers(self):

        """ test insertion of quant ops to simple rnn with multiple layes """

        tf.reset_default_graph()
        np.random.seed(0)
        tf.set_random_seed(0)
        sess = tf.Session()

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer
            x = tf.keras.layers.SimpleRNN(12, activation='tanh',
                                          kernel_initializer='glorot_uniform',
                                          recurrent_initializer='orthogonal',
                                          return_sequences=True)(inputs)
            x = tf.keras.layers.SimpleRNN(12, name='rnn1', activation='relu', return_sequences=True)(x)
            x = tf.keras.layers.SimpleRNN(12, name='rnn2', activation='tanh')(x)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="simplernn_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)

        self.validate_simple_rnn_auto_insertion_and_forward_pass(sess)
        # note - we will need to disable quantizers on identity nodes in this case

        sess.close()

    def test_backward_pass_time_taken_simple_rnn(self, is_quantized=True, iterations=10, time_steps=1):
        """ perform backward pass with quantized simple RNN block"""

        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        batches = 16

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(1, 100))

            # Add an RNN layer with 12 internal units.
            x = tf.keras.layers.SimpleRNN(12, kernel_initializer='glorot_uniform',
                                          recurrent_initializer='orthogonal')(inputs)

            _ = tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                      name="simplernn_model")(x)

            init = tf.global_variables_initializer()
            sess.run(init)
        curr_sess = sess
        var = tf.compat.v1.global_variables()
        ops = sess.graph.get_operations()

        if is_quantized:
            sim = QuantizationSimModel(sess, ['input_1'], ['simplernn_model/Softmax'], use_cuda=False)
            # sim = QuantizationSimModel(sess, ['input_1'], ['matmul0/Softmax'], use_cuda=False)
            #
            def dummy_forward_pass(sess, args):
                model_output = sess.graph.get_tensor_by_name('simplernn_model/Softmax:0')
                # model_output = sess.graph.get_tensor_by_name('matmul0/Softmax:0')
                model_input = sess.graph.get_tensor_by_name('input_1:0')
                dummy_input = np.random.randn(batches, 1, 100)
                sess.run(model_output, feed_dict={model_input: dummy_input})

            sim.compute_encodings(dummy_forward_pass, None)

            curr_sess = sim.session

        inp_tensor = curr_sess.graph.get_tensor_by_name('input_1:0')
        np.random.seed(0)
        w_shape = inp_tensor.shape

        inp_data = np.random.rand(batches, w_shape[1], w_shape[2])
        logits = curr_sess.graph.get_tensor_by_name('simplernn_model/MatMul:0')
        # logits = curr_sess.graph.get_tensor_by_name('matmul0/MatMul:0')

        labels = np.random.randint(10, size=batches)
        one_hot_labels = np.eye(10)[labels]

        with curr_sess.graph.as_default():
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            labels_placeholder = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_placeholder, logits=logits)

            update_ops = []
            global_step = tf.train.create_global_step()
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
            gradients = optimizer.compute_gradients(loss, var_list)

            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            init = tf.group(init_global, init_local)
            curr_sess.run(init)

            grad_updates = optimizer.apply_gradients(gradients, global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)

            with tf.control_dependencies([update_op]):
                train_op = tf.identity(loss, name='train_op')

            # start training
            time_taken_by_default_grad = 0
            for i in range(iterations):
                start_time = time.perf_counter()
                _ = curr_sess.run(train_op, feed_dict={inp_tensor: inp_data, labels_placeholder: one_hot_labels})
                exec_time = time.perf_counter() - start_time
                time_taken_by_default_grad = time_taken_by_default_grad + exec_time

            default_grad_avg_time = time_taken_by_default_grad / iterations

        # close session
        sess.close()
        if is_quantized:
            sim.session.close()

        return default_grad_avg_time

    # Keeping this disabled, this is for study purpose
    def _test_compare_simple_rnn_training_processing_time_increase(self):
        """
        Test to compare time taken by simple rnn node quantized versus no quantization
        There is no validation criterion for this test. It is only for study.
        :return:
        """

        # compare with and without quantize nodes
        itr = 1
        no_quant_simple_rnn_train_avg_time = self.test_backward_pass_time_taken_simple_rnn(is_quantized=False,
                                                                                           iterations=itr)
        quantized_simple_rnn_train_avg_time = self.test_backward_pass_time_taken_simple_rnn(is_quantized=True,
                                                                                            iterations=itr)

        print('\nquantized_simple_rnn_train_avg_time = ', quantized_simple_rnn_train_avg_time)
        print('\nno_quant_simple_rnn_train_avg_time = ', no_quant_simple_rnn_train_avg_time)
        print(' There is a ', ((quantized_simple_rnn_train_avg_time - no_quant_simple_rnn_train_avg_time)
                               / no_quant_simple_rnn_train_avg_time),
              'x increase in processing time with quant nodes in rnn block')

    def validate_internal_lstm_quantisim_nodes(self, quantized_graph_op_names, block_name='lstm', is_stacked=False, is_time_major=False):
        """
        Given a list of quantized_graph_op_names, this is utility function to validate
        the quantisim nodes are properly inserted
        :return:
        """
        # params that must have quantizers
        bias_param_quant_op_inside_while_block_name = block_name + "/while/split_1/ReadVariableOp_quantized"
        kernel_param_quant_op_inside_while_block_name = block_name + "/while/split/ReadVariableOp_quantized"
        recurrent_kenel_param_quant_op_inside_while_block_name_cp1 = block_name + "/while/ReadVariableOp_quantized"
        recurrent_kenel_param_quant_op_inside_while_block_name_cp2 = block_name + "/while/ReadVariableOp_1_quantized"
        recurrent_kenel_param_quant_op_inside_while_block_name_cp3 = block_name + "/while/ReadVariableOp_2_quantized"
        recurrent_kenel_param_quant_op_inside_while_block_name_cp4 = block_name + "/while/ReadVariableOp_3_quantized"

        # activations with quantizers
        activation_tanh1_op_inside_while_block_name = block_name + "/while/Tanh_quantized"
        activation_tanh2_op_inside_while_block_name = block_name + "/while/Tanh_1_quantized"

        # these should not have activation quantizers
        activation_matmul_op_inside_while_block_name = block_name + "/while/MatMul_quantized"
        activation_matmul_1_op_inside_while_block_name = block_name + "/while/MatMul_1_quantized"
        activation_matmul_2_op_inside_while_block_name = block_name + "/while/MatMul_2_quantized"
        activation_matmul_3_op_inside_while_block_name = block_name + "/while/MatMul_3_quantized"

        activation_matmul_4_op_inside_while_block_name = block_name + "/while/MatMul_4_quantized"
        activation_matmul_5_op_inside_while_block_name = block_name + "/while/MatMul_5_quantized"
        activation_matmul_6_op_inside_while_block_name = block_name + "/while/MatMul_6_quantized"
        activation_matmul_7_op_inside_while_block_name = block_name + "/while/MatMul_7_quantized"

        activation_bias_add_op_inside_while_block_name = block_name + "/while/BiasAdd_quantized"
        activation_bias_add_1_op_inside_while_block_name = block_name + "/while/BiasAdd_1_quantized"
        activation_bias_add_2_op_inside_while_block_name = block_name + "/while/BiasAdd_2_quantized"
        activation_bias_add_3_op_inside_while_block_name = block_name + "/while/BiasAdd_3_quantized"

        activation_add_op_inside_while_block_name = block_name + "/while/add_quantized"
        activation_add_2_op_inside_while_block_name = block_name + "/while/add_2_quantized"
        activation_add_4_op_inside_while_block_name = block_name + "/while/add_4_quantized"
        activation_add_6_op_inside_while_block_name = block_name + "/while/add_6_quantized"

        activation_mul_op_inside_while_block_name = block_name + "/while/Mul_quantized"
        activation_mul_1_op_inside_while_block_name = block_name + "/while/Mul_1_quantized"
        activation_mul_4_op_inside_while_block_name = block_name + "/while/Mul_4_quantized"

        activation_add_1_op_inside_while_block_name = block_name + "/while/Add_1_quantized"
        activation_add_3_op_inside_while_block_name = block_name + "/while/Add_3_quantized"
        activation_add_7_op_inside_while_block_name = block_name + "/while/Add_7_quantized"

        activation_mul_2_op_inside_while_block_name = block_name + "/while/mul_2_quantized"
        activation_mul_3_op_inside_while_block_name = block_name + "/while/mul_3_quantized"
        activation_mul_5_op_inside_while_block_name = block_name + "/while/mul_5_quantized"

        activation_add_5_op_inside_while_block_name = block_name + "/while/add_5_quantized"

        # while block ops
        # bias and kernel quantizers
        self.assertTrue(bias_param_quant_op_inside_while_block_name in quantized_graph_op_names)
        self.assertTrue(kernel_param_quant_op_inside_while_block_name in quantized_graph_op_names)
        self.assertTrue(recurrent_kenel_param_quant_op_inside_while_block_name_cp1 in quantized_graph_op_names)
        self.assertTrue(recurrent_kenel_param_quant_op_inside_while_block_name_cp2 in quantized_graph_op_names)
        self.assertTrue(recurrent_kenel_param_quant_op_inside_while_block_name_cp3 in quantized_graph_op_names)
        self.assertTrue(recurrent_kenel_param_quant_op_inside_while_block_name_cp4 in quantized_graph_op_names)

        # output quantizers: no activation quantizer is added for eAI
        #self.assertTrue(activation_tanh1_op_inside_while_block_name in quantized_graph_op_names)
        #self.assertTrue(activation_tanh2_op_inside_while_block_name in quantized_graph_op_names)

        # activations that are not quantized
        self.assertFalse(activation_matmul_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_1_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_2_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_3_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_4_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_5_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_6_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_7_op_inside_while_block_name in quantized_graph_op_names)

        self.assertFalse(activation_bias_add_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_bias_add_1_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_bias_add_2_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_bias_add_3_op_inside_while_block_name in quantized_graph_op_names)

        self.assertFalse(activation_add_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_1_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_2_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_3_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_4_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_5_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_6_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_7_op_inside_while_block_name in quantized_graph_op_names)

        self.assertFalse(activation_mul_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_mul_1_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_mul_2_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_mul_3_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_mul_4_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_mul_5_op_inside_while_block_name in quantized_graph_op_names)

        # check for input quantizers
        input_x_op_inside_while_block_name = block_name + "/while/TensorArrayReadV3_quantized"
        input_h_op_inside_while_block_name = block_name + "/while/Identity_2_quantized"
        self.assertTrue(input_x_op_inside_while_block_name in quantized_graph_op_names)
        self.assertTrue(input_h_op_inside_while_block_name in quantized_graph_op_names)

        # check for input quantizer in stacked mode
        if is_stacked:
            if is_time_major:
                input_h_op_pass_to_last_lstm_name = block_name + "/TensorArrayStack/TensorArrayGatherV3_quantized"
            else:
                input_h_op_pass_to_last_lstm_name = block_name + "/transpose_1_quantized"
            self.assertTrue(input_h_op_pass_to_last_lstm_name in quantized_graph_op_names)

    def validate_general_lstm_forward_pass_and_encoding(self, sim, block_name='lstm', num_activation_quantizer=6, num_param_quantizer=8, is_stacked_last_lstm=False, is_time_major=False):
        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('lstm_model/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(16, 3, 100)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)
        activation_quantizers = sim._activation_quantizers
        param_quantizers = sim._param_quantizers

        # check the number of quantizers
        self.assertEqual(len(activation_quantizers), num_activation_quantizer)

        # kernel, recurrent kernelx4, bias
        # one bias and kernel of dense layer MatMul
        self.assertEqual(len(param_quantizers), num_param_quantizer)

        # Check if encodings have been calculated
        for name, quantizer in activation_quantizers.items():
            if quantizer.enabled:
                self.assertTrue(quantizer.tensor_quantizer.isEncodingValid,
                                "enabled quantizer: {} does not have a valid encoding set ".format(name))

        # check encoding min and max got updated
        with sim.session.graph.as_default():
            kernel_param_quant_op = sim.session.graph.get_operation_by_name(block_name + '/while/split/ReadVariableOp_quantized')
            recurrent_kenel_param_quant_op_cp1 = sim.session.graph.get_operation_by_name(block_name + '/while/ReadVariableOp_quantized')
            recurrent_kenel_param_quant_op_cp2 = sim.session.graph.get_operation_by_name(block_name + '/while/ReadVariableOp_1_quantized')
            recurrent_kenel_param_quant_op_cp3 = sim.session.graph.get_operation_by_name(block_name + '/while/ReadVariableOp_2_quantized')
            recurrent_kenel_param_quant_op_cp4 = sim.session.graph.get_operation_by_name(block_name + '/while/ReadVariableOp_3_quantized')

            kernel_param_quant_op_encoding_min = sim.session.run(kernel_param_quant_op.inputs[
                                                                     QuantizeOpIndices.encoding_min])
            kernel_param_quant_op_encoding_max = sim.session.run(kernel_param_quant_op.inputs[
                                                                     QuantizeOpIndices.encoding_max])

            recurrent_kenel_param_quant_op_cp1_encoding_min = sim.session.run(recurrent_kenel_param_quant_op_cp1.inputs[
                                                                                  QuantizeOpIndices.encoding_min])
            recurrent_kenel_param_quant_op_cp1_encoding_max = sim.session.run(recurrent_kenel_param_quant_op_cp1.inputs[
                                                                                  QuantizeOpIndices.encoding_max])
            recurrent_kenel_param_quant_op_cp2_encoding_min = sim.session.run(recurrent_kenel_param_quant_op_cp2.inputs[
                                                                                  QuantizeOpIndices.encoding_min])
            recurrent_kenel_param_quant_op_cp2_encoding_max = sim.session.run(recurrent_kenel_param_quant_op_cp2.inputs[
                                                                                  QuantizeOpIndices.encoding_max])
            recurrent_kenel_param_quant_op_cp3_encoding_min = sim.session.run(recurrent_kenel_param_quant_op_cp3.inputs[
                                                                                  QuantizeOpIndices.encoding_min])
            recurrent_kenel_param_quant_op_cp3_encoding_max = sim.session.run(recurrent_kenel_param_quant_op_cp3.inputs[
                                                                                  QuantizeOpIndices.encoding_max])
            recurrent_kenel_param_quant_op_cp4_encoding_min = sim.session.run(recurrent_kenel_param_quant_op_cp4.inputs[
                                                                                  QuantizeOpIndices.encoding_min])
            recurrent_kenel_param_quant_op_cp4_encoding_max = sim.session.run(recurrent_kenel_param_quant_op_cp4.inputs[
                                                                                  QuantizeOpIndices.encoding_max])

        # check weight encodings are set correctly (we have fixed the TF seed above)
        # if is_stacked_last_lstm:
        #     if is_time_major:
        #         self.assertEqual(kernel_param_quant_op_encoding_min, -0.3130139708518982)
        #         self.assertEqual(kernel_param_quant_op_encoding_max, 0.31547868251800537)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp1_encoding_min, -0.39643698930740356)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp1_encoding_max, 0.39955854415893555)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp2_encoding_min, -0.39643698930740356)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp2_encoding_max, 0.39955854415893555)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp3_encoding_min, -0.39643698930740356)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp3_encoding_max, 0.39955854415893555)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp4_encoding_min, -0.39643698930740356)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp4_encoding_max, 0.39955854415893555)
        #     else:
        #         self.assertEqual(kernel_param_quant_op_encoding_min, -0.3133370876312256)
        #         self.assertEqual(kernel_param_quant_op_encoding_max, 0.31580430269241333)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp1_encoding_min, -0.4347202777862549)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp1_encoding_max, 0.4381433129310608)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp2_encoding_min, -0.4347202777862549)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp2_encoding_max, 0.4381433129310608)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp3_encoding_min, -0.4347202777862549)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp3_encoding_max, 0.4381433129310608)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp4_encoding_min, -0.4347202777862549)
        #         self.assertEqual(recurrent_kenel_param_quant_op_cp4_encoding_max, 0.4381433129310608)
        # else:
        #     self.assertEqual(kernel_param_quant_op_encoding_min, -0.19972173869609833)
        #     self.assertEqual(kernel_param_quant_op_encoding_max, 0.20129434764385223)
        #     self.assertEqual(recurrent_kenel_param_quant_op_cp1_encoding_min, -0.43212029337882996)
        #     self.assertEqual(recurrent_kenel_param_quant_op_cp1_encoding_max, 0.43552282452583313)
        #     self.assertEqual(recurrent_kenel_param_quant_op_cp2_encoding_min, -0.43212029337882996)
        #     self.assertEqual(recurrent_kenel_param_quant_op_cp2_encoding_max, 0.43552282452583313)
        #     self.assertEqual(recurrent_kenel_param_quant_op_cp3_encoding_min, -0.43212029337882996)
        #     self.assertEqual(recurrent_kenel_param_quant_op_cp3_encoding_max, 0.43552282452583313)
        #     self.assertEqual(recurrent_kenel_param_quant_op_cp4_encoding_min, -0.43212029337882996)
        #     self.assertEqual(recurrent_kenel_param_quant_op_cp4_encoding_max, 0.43552282452583313)

    def test_quantize_lstm_default_quantsim_and_forward_pass(self):
        """ Test connected graph construction on a model with lstm op """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add a LSTM layer with 12 internal units.
            x = tf.keras.layers.LSTM(12)(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="lstm_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)
        # _ = tf.summary.FileWriter('./lstm', sess.graph)

        sim = QuantizationSimModel(sess, ['input_1'], ['lstm_model/Softmax'],
                                   use_cuda=False)

        # validate quantsim
        # get ops and make sure we have a quantized op added to the conditional block
        quantized_graph_op_names = self._get_quant_ops_from_tf_graph(sim.session.graph)

        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names)

        # validate forward pass
        self.validate_general_lstm_forward_pass_and_encoding(sim)

        # close tf sessions
        sess.close()
        sim.session.close()

    def test_quantize_simple_rnn_export(self):
        """ Test model export for recurrent models """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            x = tf.keras.layers.SimpleRNN(10, name='rnn1', return_sequences=True)(inputs)
            x = tf.keras.layers.SimpleRNN(10, name='rnn2')(x)

            _ = tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                      name="fc")(x)

            init = tf.global_variables_initializer()
            sess.run(init)

        sim = QuantizationSimModel(sess, ['input_1'], ['fc/Softmax'],
                                   use_cuda=False)

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('fc/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(1, 3, 100)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)
        sim.export('./data', 'rnn_quantsim')

        new_sess = load_model_from_meta('./data/rnn_quantsim.meta')

        dummy_forward_pass(new_sess, None)

        all_op_types = [op.type for op in new_sess.graph.get_operations()]
        self.assertNotIn('QcQuantize', all_op_types)

        # Load the encodings file to check if the encodings were exported correctly
        with open("./data/rnn_quantsim.encodings", "r") as encodings_file:
            encodings = json.load(encodings_file)
            self.assertEqual(7, len(encodings['activation_encodings']))
            self.assertEqual(5, len(encodings['param_encodings']))

        # close tf sessions
        sess.close()
        sim.session.close()

    def _get_quant_ops_from_tf_graph(self, gr: tf.Graph):
        """
        utility to get quant op names in given graph
        :param graph: tf.Graph
        :return:
        """
        ops = gr.get_operations()
        quantized_graph_op_names = [op.name for op in ops if op.type in ["QcQuantize", "QcQuantizeRecurrentParam"]]

        return quantized_graph_op_names

    def test_quantize_simple_rnn_save_and_load_checkpoint(self):
        """ Test model export for recurrent models """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            x = tf.keras.layers.SimpleRNN(10, name='rnn1', return_sequences=True)(inputs)
            x = tf.keras.layers.SimpleRNN(10, name='rnn2')(x)

            _ = tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                      name="fc")(x)

            init = tf.global_variables_initializer()
            sess.run(init)

        sim = QuantizationSimModel(sess, ['input_1'], ['fc/Softmax'],
                                   use_cuda=False)

        def eval(sess, input_tensor):
            model_output = sess.graph.get_tensor_by_name('fc/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            out = sess.run(model_output, feed_dict={model_input: input_tensor})
            return out

        def dummy_forward_pass(sess, args):
            dummy_input = np.random.randn(1, 3, 100)
            eval(sess, dummy_input)

        sim.compute_encodings(dummy_forward_pass, None)
        random_tensor = np.random.randn(1, 3, 100)
        old_out = eval(sim.session, random_tensor)

        save_checkpoint(sim, './data/', 'simple_rnn_save')
        new_sim = load_checkpoint('./data', 'simple_rnn_save')

        # Check to make sure that inference through the new sim produces exactly the same output as the old sim
        # This checks that quantization parameters have been restored correctly
        # Also checks that we are able to invoke quantize-dequantize ops in the new session (so pymo objects were
        # restored correctly etc.)
        new_out  = eval(new_sim.session, random_tensor)
        self.assertTrue(np.allclose(old_out, new_out))
        print(new_sim)

    def test_quantize_lstm_sigmoid_quantsim_and_forward_pass(self):
        """ Test connected graph construction on a model with lstm op """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add a LSTM layer with 12 internal units.
            x = tf.keras.layers.LSTM(12, recurrent_activation='sigmoid')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="lstm_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)
        # _ = tf.summary.FileWriter('./lstm', sess.graph)

        sim = QuantizationSimModel(sess, ['input_1'], ['lstm_model/Softmax'],
                                   use_cuda=False)

        # validate quantsim
        # get ops and make sure we have a quantized op added to the conditional block
        quantized_graph_op_names = self._get_quant_ops_from_tf_graph(sim.session.graph)

        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names)

        # validate forward pass
        self.validate_general_lstm_forward_pass_and_encoding(sim)

        # close tf sessions
        sess.close()
        sim.session.close()

    def test_quantize_lstm_time_major_true_quantsim_and_forward_pass(self):
        """ Test connected graph construction on a model with lstm op """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add a LSTM layer with 12 internal units.
            x = tf.keras.layers.LSTM(12, time_major=True, name='lstm_tm')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="lstm_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)
        # _ = tf.summary.FileWriter('./lstm', sess.graph)

        sim = QuantizationSimModel(sess, ['input_1'], ['lstm_model/Softmax'],
                                   use_cuda=False)

        # validate quantsim
        # get ops and make sure we have a quantized op added to the conditional blocks
        quantized_graph_op_names = self._get_quant_ops_from_tf_graph(sim.session.graph)

        batches = 32
        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('lstm_model/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(batches, 3, 100)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names, 'lstm_tm')

        # validate forward pass
        self.validate_general_lstm_forward_pass_and_encoding(sim, 'lstm_tm')

        # close tf sessions
        sess.close()
        sim.session.close()

    def test_quantize_lstm_deepspeech_time_major_true_quantsim_and_forward_pass(self):
        """ Test connected graph construction on a model with lstm op """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add a LSTM layer with 12 internal units.
            x = tf.keras.layers.LSTM(12,
                                     unit_forget_bias=False,
                                     time_major=True,
                                     return_sequences=True,
                                     name='lstm_stacked')(inputs)
            x2 = tf.keras.layers.LSTM(12, name='last_lstm')(x)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="lstm_model")(x2)

        init = tf.global_variables_initializer()
        sess.run(init)
        # _ = tf.summary.FileWriter('./lstm', sess.graph)

        sim = QuantizationSimModel(sess, ['input_1'], ['lstm_model/Softmax'],
                                   use_cuda=False)

        # validate quantsim
        # get ops and make sure we have a quantized op added to the conditional block
        quantized_graph_op_names = self._get_quant_ops_from_tf_graph(sim.session.graph)

        # _ = tf.summary.FileWriter('./lstm_tm', sess.graph)
        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names, 'lstm_stacked', True, True)
        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names, 'last_lstm')

        # validate forward pass
        self.validate_general_lstm_forward_pass_and_encoding(sim, 'lstm_stacked', 9, 14, False, True)
        self.validate_general_lstm_forward_pass_and_encoding(sim, 'last_lstm', 9, 14, True, True)

        # close tf sessions
        sess.close()
        sim.session.close()

    def test_quantize_lstm_deepspeech_time_major_false_quantsim_and_forward_pass(self):
        """ Test connected graph construction on a model with lstm op """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add a LSTM layer with 12 internal units.
            x, state_h, state_c = tf.keras.layers.LSTM(12, return_state=True,
                                                       return_sequences=True,
                                                       name='lstm_stacked')(inputs)
            x2 = tf.keras.layers.LSTM(12, name='last_lstm')(x)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="lstm_model")(x2)

        init = tf.global_variables_initializer()
        sess.run(init)

        sim = QuantizationSimModel(sess, ['input_1'], ['lstm_model/Softmax'],
                                   use_cuda=False)

        # validate quantsim
        # get ops and make sure we have a quantized op added to the conditional block
        quantized_graph_op_names = self._get_quant_ops_from_tf_graph(sim.session.graph)

        # _ = tf.summary.FileWriter('./lstm_tm', sess.graph)
        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names, 'lstm_stacked', True, False)
        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names, 'last_lstm')

        # validate forward pass
        self.validate_general_lstm_forward_pass_and_encoding(sim, 'lstm_stacked', 9, 14, False, False)
        self.validate_general_lstm_forward_pass_and_encoding(sim, 'last_lstm', 9, 14, True, False)

        # close tf sessions
        sess.close()
        sim.session.close()

    def test_backward_pass_time_taken_lstm(self, is_quantized=True, iterations=1):
        """ perform backward pass with quantized lstm block"""

        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)
        timesteps = 5
        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(timesteps, 100))

            # Add a lstm layer with 12 internal units.
            x = tf.keras.layers.LSTM(12)(inputs)

            _ = tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                      name="lstm_model")(x)

            init = tf.global_variables_initializer()
            sess.run(init)
        curr_sess = sess
        if is_quantized:
            sim = QuantizationSimModel(sess, ['input_1'], ['lstm_model/Softmax'], use_cuda=False)

            def dummy_forward_pass(sess, args):
                model_output = sess.graph.get_tensor_by_name('lstm_model/Softmax:0')
                model_input = sess.graph.get_tensor_by_name('input_1:0')
                dummy_input = np.random.randn(32, 5, 100)  # time_steps = 5
                sess.run(model_output, feed_dict={model_input: dummy_input})

            sim.compute_encodings(dummy_forward_pass, None)

            curr_sess = sim.session

        inp_tensor = curr_sess.graph.get_tensor_by_name('input_1:0')
        np.random.seed(0)
        w_shape = inp_tensor.shape
        batches = 32
        inp_data = np.random.rand(batches, w_shape[1], w_shape[2])
        logits = curr_sess.graph.get_tensor_by_name('lstm_model/MatMul:0')

        labels = np.random.randint(10, size=batches)
        one_hot_labels = np.eye(10)[labels]

        with curr_sess.graph.as_default():
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            labels_placeholder = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_placeholder, logits=logits)

            update_ops = []
            global_step = tf.train.create_global_step()
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
            gradients = optimizer.compute_gradients(loss, var_list)

            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            init = tf.group(init_global, init_local)
            curr_sess.run(init)

            grad_updates = optimizer.apply_gradients(gradients, global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)

            with tf.control_dependencies([update_op]):
                train_op = tf.identity(loss, name='train_op')

            # start training
            time_taken_by_default_grad = 0
            for i in range(iterations):
                start_time = time.perf_counter()
                _ = curr_sess.run(train_op, feed_dict={inp_tensor: inp_data, labels_placeholder: one_hot_labels})
                exec_time = time.perf_counter() - start_time
                time_taken_by_default_grad = time_taken_by_default_grad + exec_time

            default_grad_avg_time = time_taken_by_default_grad / iterations

        # close session
        sess.close()
        if is_quantized:
            sim.session.close()

        return default_grad_avg_time

    # Keeping this disabled, this is for study purpose
    def _test_compare_lstm_training_processing_time_increase(self):
        """
        Test to compare time taken by lstm node quantized versus no quantization
        There is no validation criterion for this test. It is only for study.
        :return:
        """

        # compare with and without quantize nodes
        itr = 10000
        no_quant_lstm_train_avg_time = self.test_backward_pass_time_taken_lstm(is_quantized=False,
                                                                               iterations=itr)
        quantized_lstm_train_avg_time = self.test_backward_pass_time_taken_lstm(is_quantized=True,
                                                                                iterations=itr)

        print('\nquantized_lstm_train_avg_time = ', quantized_lstm_train_avg_time)
        print('\nno_quant_lstm_train_avg_time = ', no_quant_lstm_train_avg_time)
        print(' There is a ', ((quantized_lstm_train_avg_time - no_quant_lstm_train_avg_time)
                               / no_quant_lstm_train_avg_time),
              'x increase in processing time with quant nodes in lstm block')
