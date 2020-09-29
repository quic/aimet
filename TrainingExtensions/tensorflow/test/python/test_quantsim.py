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

import unittest
import shutil
import tensorflow as tf
import numpy as np
import json
import libpymo
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

        tf.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.Session()
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

    def test_construction_gpu_model(self):
        """
        Create QuantSim for a GPU model and check that quantizers have been added to the graph
        """
        tf.reset_default_graph()
        with tf.device('/gpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.Session()
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

        tf.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.Session()
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
        tf.reset_default_graph()
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

        sess = tf.Session()
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

    def test_compute_encodings_gpu_model(self):
        """
        Create QuantSim for a CPU model and test that activation encodings are computed
        """

        tf.reset_default_graph()
        with tf.device('/gpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.Session()
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

    def test_compute_encodings_quant_scheme_update(self):
        """
        Create QuantSim model and update quantScheme using property interface
        """


        tf.reset_default_graph()
        np.random.seed(0)
        tf.set_random_seed(0)

        with tf.device('/gpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=True)

        # Check that op-mode is set correctly
        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')

        self.assertEqual(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                         sim.session.run(conv2d_weight_quant_op.inputs[1]))

        def dummy_forward_pass(sess, args):
            np.random.seed(0)
            tf.set_random_seed(0)
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
        tf.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.Session()
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
            first_bias_tensor_var = [var for var in tf.global_variables() if var.name == 'conv2d/bias:0'][0]
            first_bias_tensor_var.load(np.ones(32), sim.session)

        all_op_types = [op.type for op in sim.session.graph.get_operations()]
        self.assertIn('QcQuantize', all_op_types)

        sim.export('/tmp', 'quant_sim_model')

        with open('/tmp/quant_sim_model.encodings') as json_file:
            encoding_data = json.load(json_file)
        activation_keys = list(encoding_data["activation_encodings"].keys())
        self.assertTrue(activation_keys[0] == "conv2d/Relu:0")
        self.assertTrue(isinstance(encoding_data["activation_encodings"]["conv2d/Relu:0"], list))

        param_keys = list(encoding_data["param_encodings"].keys())
        self.assertTrue(param_keys[0] == "conv2d/Conv2D/ReadVariableOp:0")
        self.assertTrue(isinstance(encoding_data["param_encodings"]["conv2d/Conv2D/ReadVariableOp:0"], list))

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
        tf.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.Session()
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
        tf.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.Session()
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

        tf.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.Session()
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

        tf.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.Session()
        initialize_uninitialized_vars(sess)

        orig_get_ops_to_quantize_activations_for = QuantizationSimModel.get_ops_to_quantize_activations_for
        orig_get_ops_to_quantize_weights_for = QuantizationSimModel.get_ops_to_quantize_params_for
        orig_configure_quantization_ops = QuantizationSimModel.configure_quantization_ops
        QuantizationSimModel.get_ops_to_quantize_activations_for = get_manual_activations
        QuantizationSimModel.get_ops_to_quantize_params_for = get_manual_params
        QuantizationSimModel.configure_quantization_ops = configure_quantization_ops
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=False)
        self.assertEqual(1, len(sim._activation_quantizers))
        self.assertEqual(1, len(sim._param_quantizers))
        sess.close()
        sim.session.close()
        QuantizationSimModel.get_ops_to_quantize_activations_for = orig_get_ops_to_quantize_activations_for
        QuantizationSimModel.get_ops_to_quantize_params_for = orig_get_ops_to_quantize_weights_for
        QuantizationSimModel.configure_quantization_ops = orig_configure_quantization_ops

    def test_skip_quantizing_dtype_int(self):
        """ Test that op with dtype int32 is skipped during quantization """
        tf.reset_default_graph()
        with tf.Session() as sess:
            _ = model_with_dtype_int()
            initialize_uninitialized_vars(sess)
            sim = QuantizationSimModel(sess, ['input_1', 'input_2'], ['model_with_dtype_int/Softmax'], use_cuda=False)
            self.assertEqual(6, len(sim._activation_quantizers))
            self.assertTrue('input_1_quantized' not in sim._activation_quantizers)
            self.assertTrue('input_2_quantized' in sim._activation_quantizers)

    def test_insert_quant_op_recurrent(self):

        """ test insertion of quant ops to recurrent layer with conditional blocks """

        tf.reset_default_graph()
        sess = tf.Session()

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            # Add an RNN layer
            x = tf.keras.layers.SimpleRNN(12)(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="simplernn_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)
        ops = sess.graph.get_operations()
        quant_op_inside_while_block_name = "simple_rnn/while/MatMul/ReadVariableOp_quantized"
        self.assertFalse(quant_op_inside_while_block_name in [op.name for op in ops])

        # construct a quantization sim model
        sim = QuantizationSimModel(sess, ['input_1'], ['simplernn_model/Softmax'], use_cuda=False)

        while_matmul_weight_read_tensor_name = "simple_rnn/while/MatMul/ReadVariableOp:0"
        while_matmul_weight_read_tensor = sim.session.graph.get_tensor_by_name(while_matmul_weight_read_tensor_name)
        op_mode = libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize

        # while block ops won't be updated by sim before invoking new api below
        ops = sim.session.graph.get_operations()
        self.assertFalse(quant_op_inside_while_block_name in [op.name for op in ops])

        # test the new api to insert quant ops to conditional blocks
        sim._insert_post_training_quant_op_recurrent(while_matmul_weight_read_tensor,
                                                     quant_op_inside_while_block_name,
                                                     op_mode, sim._param_quantizers, 0, 8)
        # make sure graph is updated
        sim._save_and_load_sim_model()

        # get ops and make sure we have a quantized op added to the conditional block
        ops = sim.session.graph.get_operations()
        self.assertTrue(quant_op_inside_while_block_name in [op.name for op in ops])

    def test_compute_encodings(self):
        """ Test that ops not evaluated during compute encodings are set to passThrough mode. """
        tf.reset_default_graph()
        sess = tf.Session()
        test_inp = np.ndarray((1, 32, 32, 3))

        def dummy_forward_func(sess, _):
            input_tensor = sess.graph.get_tensor_by_name('input_1:0')
            output_tensor = sess.graph.get_tensor_by_name('flatten/Reshape:0')
            sess.run(output_tensor, feed_dict={input_tensor: test_inp})

        with sess.as_default():
            _ = keras_model_functional()
            init = tf.global_variables_initializer()
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
