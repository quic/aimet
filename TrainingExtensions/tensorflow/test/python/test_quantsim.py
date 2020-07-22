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

import libpymo
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils.graph_saver import load_model_from_meta
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
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
