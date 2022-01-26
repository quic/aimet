# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
import random
import numpy as np
import time
import os
import json

import tensorflow as tf

from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.examples.test_models import depthwise_conv2d_model
from aimet_common.quantsim import calculate_delta_offset

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import libpymo

class TestTrainingExtensionsQcQuantizeOpPerChannel(unittest.TestCase):

    def test_qc_quantize_op_cpu_conv(self):
        """
        test custom op with CPU
        """
        np.random.seed(0)
        zero_out_module = tf.load_op_library('libaimet_tf_ops.so')
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = True

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                num_output_channels = 3
                inp = tf.compat.v1.placeholder(tf.float32, shape=[1, 1, 2, num_output_channels], name='input')
                # Assuming 3 output channels
                tensor_quantizer_int64 = [None] * num_output_channels
                tensor_quantizers = [None] * num_output_channels
                # Create a tensor_quantizer per channel
                for i in range(num_output_channels):
                    tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                               libpymo.RoundingMode.ROUND_NEAREST)

                    tensor_quantizers[i] = tensor_quantizer
                    val = libpymo.PtrToInt64(tensor_quantizer)
                    tensor_quantizer_int64[i] = val

                tensor_quant_ref = tf.Variable(tensor_quantizer_int64, trainable=False, dtype=tf.int64)

                en_min = (np.zeros(num_output_channels)).tolist()
                en_max = [1.0, 2.0, 2.5]
                encoding_min = tf.Variable(en_min,
                                           trainable=True, dtype=tf.double)
                encoding_max = tf.Variable(en_max,
                                           trainable=True, dtype=tf.double)

                bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
                use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)

                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.updateStats),
                                       trainable=False, dtype=tf.int32)
                axis = tf.Variable(initial_value=3, trainable=False, dtype=tf.int32)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          axis.initializer])
                # Giving axis = 3
                pass_through_op_output = zero_out_module.qc_quantize_per_channel(name='quant_op', in_tensor=inp,
                                                                                 op_mode=mode_var,
                                                                                 tensor_quantizer_reference=tensor_quant_ref,
                                                                                 encoding_min=encoding_min,
                                                                                 encoding_max=encoding_max,
                                                                                 bit_width=bit_width,
                                                                                 use_symmetric_encoding=use_symmetric_encoding,
                                                                                 axis=axis)

            inp_tensor = sess.graph.get_tensor_by_name('input:0')
            inp_data = np.ones((1, 1, 2, num_output_channels))

            inp_data[:, :, :, 1] *= 2
            inp_data[:, :, :, 2] *= 3
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
            for i in range(num_output_channels):
                encoding = tensor_quantizers[i].computeEncoding(bitwidth, use_symm_encoding, False, False)
            mode_var.load(int(libpymo.TensorQuantizerOpMode.quantizeDequantize), sess)
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
            # compare qc_quantize op's output with input
            expected_output = np.ones((1, 1, 2, num_output_channels))
            expected_output[:, :, :, 1] *= 2
            expected_output[:, :, :, 2] *= 2.5
            self.assertTrue(np.allclose(out_data, expected_output))
            sess.close()

    @pytest.mark.cuda
    def test_qc_quantize_op_gpu_conv(self):
        """
        test custom op with GPU

        """
        np.random.seed(0)
        zero_out_module = tf.load_op_library('libaimet_tf_ops.so')
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = True

        with graph.as_default():
            # place holder for the input

            num_output_channels = 3
            inp = tf.compat.v1.placeholder(tf.float32, shape=[1, 1, 2, num_output_channels], name='input')
            # Assuming 3 output channels
            tensor_quantizer_int64 = [None] * num_output_channels
            tensor_quantizers = [None] * num_output_channels
            # Create a tensor_quantizer per channel
            for i in range(num_output_channels):
                tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                           libpymo.RoundingMode.ROUND_NEAREST)

                tensor_quantizers[i] = tensor_quantizer
                val = libpymo.PtrToInt64(tensor_quantizer)
                tensor_quantizer_int64[i] = val

            tensor_quant_ref = tf.Variable(tensor_quantizer_int64, trainable=False, dtype=tf.int64)

            en_min = (np.zeros(num_output_channels)).tolist()
            en_max = [1.0, 2.0, 2.5]
            encoding_min = tf.Variable(en_min,
                                       trainable=True, dtype=tf.double)
            encoding_max = tf.Variable(en_max,
                                       trainable=True, dtype=tf.double)

            bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
            use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)

            mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.updateStats),
                                   trainable=False, dtype=tf.int32)
            axis = tf.Variable(initial_value=3, trainable=False, dtype=tf.int32)

            sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                      encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                      axis.initializer])
            with tf.device("/device:GPU:0"):
                # Giving axis = 3
                pass_through_op_output = zero_out_module.qc_quantize_per_channel(name='quant_op', in_tensor=inp,
                                                                                 op_mode=mode_var,
                                                                                 tensor_quantizer_reference=tensor_quant_ref,
                                                                                 encoding_min=encoding_min,
                                                                                 encoding_max=encoding_max,
                                                                                 bit_width=bit_width,
                                                                                 use_symmetric_encoding=use_symmetric_encoding,
                                                                                 axis=3)

            inp_tensor = sess.graph.get_tensor_by_name('input:0')
            inp_data = np.ones((1, 1, 2, num_output_channels))

            inp_data[:, :, :, 1] *= 2
            inp_data[:, :, :, 2] *= 3
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
            for i in range(num_output_channels):
                encoding = tensor_quantizers[i].computeEncoding(bitwidth, use_symm_encoding, False, False)
            mode_var.load(int(libpymo.TensorQuantizerOpMode.quantizeDequantize), sess)
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
            # compare qc_quantize op's output with input
            expected_output = np.ones((1, 1, 2, num_output_channels))
            expected_output[:, :, :, 1] *= 2
            expected_output[:, :, :, 2] *= 2.5
            self.assertTrue(np.allclose(out_data, expected_output))
            sess.close()

    def test_qc_quantize_op_cpu_linear(self):
        """
        test custom op with CPU
        """
        np.random.seed(0)
        zero_out_module = tf.load_op_library('libaimet_tf_ops.so')
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = True

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[2, 3], name='input')
                # Assuming 3 output channels
                tensor_quantizer_int64 = [None] * 3
                tensor_quantizers = [None] * 3
                # Create a tensor_quantizer per channel
                for i in range(3):
                    tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                               libpymo.RoundingMode.ROUND_NEAREST)

                    tensor_quantizers[i] = tensor_quantizer
                    val = libpymo.PtrToInt64(tensor_quantizer)
                    tensor_quantizer_int64[i] = val

                tensor_quant_ref = tf.Variable(tensor_quantizer_int64, trainable=False, dtype=tf.int64)

                encoding_min = tf.Variable([1.0, 2.0, 3.0],
                                           trainable=True, dtype=tf.double)
                encoding_max = tf.Variable([2.0, 3.0, 4.0],
                                           trainable=True, dtype=tf.double)

                bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
                use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)

                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.updateStats),
                                       trainable=False, dtype=tf.int32)
                axis = tf.Variable(initial_value=1, trainable=False, dtype=tf.int32)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          axis.initializer])
                # Giving axis = 1
                pass_through_op_output = zero_out_module.qc_quantize_per_channel(name='quant_op', in_tensor=inp,
                                                                                 op_mode=mode_var,
                                                                                 tensor_quantizer_reference=tensor_quant_ref,
                                                                                 encoding_min=encoding_min,
                                                                                 encoding_max=encoding_max,
                                                                                 bit_width=bit_width,
                                                                                 use_symmetric_encoding=use_symmetric_encoding,
                                                                                 axis=axis)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.ones((2, 3))
        inp_data[:, 1] *= 2
        inp_data[:, 2] *= 3

        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})

        # compare qc_quantize op's output with input
        self.assertTrue(np.allclose(out_data, inp_data))
        sess.close()

    def test_qc_quantize_op_cpu_bias(self):
        """
        test custom op with CPU
        """
        np.random.seed(0)
        zero_out_module = tf.load_op_library('libaimet_tf_ops.so')
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = True

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[3], name='input')
                # Assuming 3 output channels
                tensor_quantizer_int64 = [None] * 3
                tensor_quantizers = [None] * 3
                # Create a tensor_quantizer per channel
                for i in range(3):
                    tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                               libpymo.RoundingMode.ROUND_NEAREST)

                    tensor_quantizers[i] = tensor_quantizer
                    val = libpymo.PtrToInt64(tensor_quantizer)
                    tensor_quantizer_int64[i] = val

                tensor_quant_ref = tf.Variable(tensor_quantizer_int64, trainable=False, dtype=tf.int64)

                encoding_min = tf.Variable([1.0, 2.0, 3.0],
                                           trainable=True, dtype=tf.double)
                encoding_max = tf.Variable([2.0, 3.0, 4.0],
                                           trainable=True, dtype=tf.double)

                bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
                use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)

                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.updateStats),
                                       trainable=False, dtype=tf.int32)
                axis = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          axis.initializer])
                # Giving axis = 3
                pass_through_op_output = zero_out_module.qc_quantize_per_channel(name='quant_op', in_tensor=inp,
                                                                                 op_mode=mode_var,
                                                                                 tensor_quantizer_reference=tensor_quant_ref,
                                                                                 encoding_min=encoding_min,
                                                                                 encoding_max=encoding_max,
                                                                                 bit_width=bit_width,
                                                                                 use_symmetric_encoding=use_symmetric_encoding,
                                                                                 axis=axis)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.ones(3)
        inp_data[1] *= 2
        inp_data[2] *= 3

        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})

        # compare qc_quantize op's output with input
        self.assertTrue(np.allclose(out_data, inp_data))
        sess.close()

    def test_compute_encodings_cpu(self):
        save_config_file_for_per_channel_quantization()

        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(2, kernel_size=3, input_shape=(5, 5, 3)))
            model.summary()

        with tf.device("/device:CPU:0"):

            sess = tf.compat.v1.Session()
            initialize_uninitialized_vars(sess)
            sim = QuantizationSimModel(sess, [model.input.op.name], [model.output.op.name], use_cuda=False,
                                       config_file='./quantsim_config.json')

            for quant_op_name, quantizer_info in sim._activation_quantizers.items():
                assert isinstance(quantizer_info.tensor_quantizer, libpymo.TensorQuantizer)

            for quant_op_name, quantizer_info in sim._param_quantizers.items():
                op = sim.session.graph.get_operation_by_name(quant_op_name)
                assert op.type == 'QcQuantizePerChannel'
                shape = op.inputs[0].shape.as_list()
                assert  len(quantizer_info.tensor_quantizer) == shape[-1]


            def dummy_forward_pass(sess, args):
                model_output = sess.graph.get_tensor_by_name('conv2d_input:0')

                model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
                np.random.seed(0)
                dummy_input = np.random.randn(20, 5, 5, 3)
                sess.run(model_output, feed_dict={model_input: dummy_input})

            sim.compute_encodings(dummy_forward_pass, None)
            for quant_op_name, quantizer_info in sim._param_quantizers.items():
                encoding = quantizer_info.get_encoding()
                assert isinstance(encoding, list)

    def test_export_encodings(self):
        save_config_file_for_per_channel_quantization()

        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(5, 5, 3), activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, [model.input.op.name], [model.output.op.name], use_cuda=False,
                                   config_file='./quantsim_config.json')

        def create_encoding():
            _encoding = libpymo.TfEncoding()
            _encoding.min = random.uniform(0, 1)
            _encoding.max = random.uniform(1, 3)
            _encoding.bw = 8
            _encoding.delta, _encoding.offset = calculate_delta_offset(_encoding.min, _encoding.max,
                                                                       8)
            return _encoding

        # Set the encodings for activation quantizers
        for quant_op_name, quantizer_info in sim._activation_quantizers.items():
            _encoding = create_encoding()
            quantizer_info.set_encoding(_encoding)

        # Set encodings for parameter quantizers
        for quant_op_name, quantizer_info in sim._param_quantizers.items():
            encoding = []
            for i in range(len(quantizer_info.tensor_quantizer)):
                _encoding = create_encoding()
                encoding.append(_encoding)
            quantizer_info.set_encoding(encoding)

        sim.export('/tmp', 'quant_sim_model')

        with open('/tmp/quant_sim_model.encodings') as json_file:
            encoding_data = json.load(json_file)

        param_keys = list(encoding_data["param_encodings"].keys())
        self.assertTrue(param_keys[1] == "conv2d/Conv2D/ReadVariableOp:0")
        self.assertTrue(isinstance(encoding_data["param_encodings"]["conv2d/Conv2D/ReadVariableOp:0"], list))
        self.assertTrue(isinstance(encoding_data["param_encodings"]["conv2d/Conv2D/ReadVariableOp:0"][0]['max'], list))
        self.assertTrue(isinstance(encoding_data["param_encodings"]["conv2d/Conv2D/ReadVariableOp:0"][0]['min'], list))

    @pytest.mark.cuda
    def test_compute_encodings_gpu_model(self):
        """
        Create QuantSim for a CPU model and test that activation encodings are computed
        """
        save_config_file_for_per_channel_quantization()

        tf.compat.v1.reset_default_graph()
        with tf.device('/gpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.add(tf.keras.layers.Dense(2, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['dense/Relu'], use_cuda=True,
                                   config_file='./quantsim_config.json')

        # Check that op-mode is set correctly
        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
        conv2d_output_quant_op = sim.session.graph.get_operation_by_name('dense/Relu_quantized')
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                         sim.session.run(conv2d_weight_quant_op.inputs[1]))
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.updateStats),
                         sim.session.run(conv2d_output_quant_op.inputs[1]))

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('dense/Relu_quantized:0')
            model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
            dummy_input = np.random.randn(20, 28, 28, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)
        for quant_op_name, quantizer_info in sim._param_quantizers.items():
            encoding = quantizer_info.get_encoding()
            assert isinstance(encoding, list)

    @pytest.mark.cuda
    def test_to_compare_time_per_channel_and_per_tensor_quantization(self):
        save_config_file_for_per_channel_quantization()

        tf.compat.v1.reset_default_graph()

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('conv2d_1/Relu_quantized:0')
            model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
            dummy_input = np.random.randn(20, 28, 28, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        with tf.device('/gpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d_1/Relu'], use_cuda=True,
                                   config_file='./quantsim_config.json')


        start_time = time.time()
        sim.compute_encodings(dummy_forward_pass, None)
        per_channel_quantization_time = time.time() - start_time
        print("--- %s seconds ---" % per_channel_quantization_time)

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

        start_time = time.time()
        sim.compute_encodings(dummy_forward_pass, None)
        per_tensor_quantization_time = time.time() - start_time
        print("--- %s seconds ---" % per_tensor_quantization_time)

    @pytest.mark.cuda
    def test_compute_encodings_gpu_model_depthwise_model(self):
        """
        Create QuantSim for a CPU model and test that activation encodings are computed
        """
        save_config_file_for_per_channel_quantization()

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with tf.device('/gpu:0'):
            _ = depthwise_conv2d_model()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

        sim = QuantizationSimModel(sess, ['input_1'], ['depthwise_conv2d_model/Softmax'], use_cuda=True,
                                   config_file='./quantsim_config.json')

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('depthwise_conv2d_model/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(1, 10, 10, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)
        for quant_op_name, quantizer_info in sim._param_quantizers.items():
            encoding = quantizer_info.get_encoding()
            assert isinstance(encoding, list)


def save_config_file_for_per_channel_quantization():
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
            "per_channel_quantization": "True",
        },
        "params": {},
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }

    with open('./quantsim_config.json', 'w') as f:
        json.dump(quantsim_config, f)
