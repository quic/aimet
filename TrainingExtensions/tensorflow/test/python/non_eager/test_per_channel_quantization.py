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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import tensorflow as tf

from aimet_common.defs import QuantScheme
from aimet_common.quantsim import calculate_delta_offset
import aimet_common.libpymo as libpymo
import aimet_common.libaimet_tf_ops as zero_out_module
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.quantsim import QuantizationSimModel, AxisHandling
from aimet_tensorflow.examples.test_models import depthwise_conv2d_model, transposed_conv2d_model
from aimet_tensorflow.utils.constants import QuantizeOpIndices
from aimet_tensorflow.utils.op.conv import WeightTensorUtils
from aimet_tensorflow.utils.graph_saver import load_model_from_meta

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)


def nn_depthwise_conv2d_model():
    """ Returns a model with depthwise conv2d """

    num_input_channels = 3
    channel_multiplier = 2
    num_output_channels = num_input_channels * channel_multiplier
    # tf nn depthwise2d conv weight shape is H, W, I, channel multiplier. Number of output channels = I * channel
    # multiplier
    depthwise_kernel = tf.Variable(initial_value=tf.random.uniform(shape=[2, 2, num_input_channels, channel_multiplier],
                                                                   dtype=tf.float32),
                                   name='depthwise_kernel', trainable=True, dtype=tf.float32)

    inputs = tf.keras.Input(shape=(10, 10, 3,))
    x = tf.nn.depthwise_conv2d(inputs, depthwise_kernel, strides=[1, 1, 1, 1], padding='VALID')
    x = tf.nn.softmax(x)
    return x

class TestTrainingExtensionsQcQuantizeOpPerChannel(unittest.TestCase):

    def test_qc_quantize_op_cpu_conv(self):
        """
        test custom op with CPU
        """
        np.random.seed(0)
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = True

        with graph.as_default():
            # placeholder for the input
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
                is_int_data_type = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)

                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                                       trainable=False, dtype=tf.int32)
                # axis handling for getting number of channels.
                axis_handling = tf.Variable(initial_value=AxisHandling.LAST_AXIS.value, trainable=False, dtype=tf.int32)
                is_training = tf.keras.backend.learning_phase()

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          is_int_data_type.initializer, axis_handling.initializer])
                pass_through_op_output = zero_out_module.qc_quantize_per_channel(name='quant_op', in_tensor=inp,
                                                                                 op_mode=mode_var,
                                                                                 tensor_quantizer_reference=tensor_quant_ref,
                                                                                 encoding_min=encoding_min,
                                                                                 encoding_max=encoding_max,
                                                                                 bit_width=bit_width,
                                                                                 use_symmetric_encoding=use_symmetric_encoding,
                                                                                 is_int_data_type=is_int_data_type,
                                                                                 axis_handling=axis_handling,
                                                                                 is_training=is_training)

            inp_tensor = sess.graph.get_tensor_by_name('input:0')
            inp_data = np.ones((1, 1, 2, num_output_channels))

            inp_data[:, :, :, 1] *= 2
            inp_data[:, :, :, 2] *= 3
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
            for i in range(num_output_channels):
                encoding = tensor_quantizers[i].computeEncoding(bitwidth, use_symm_encoding)
            mode_var.load(int(libpymo.TensorQuantizerOpMode.quantizeDequantize), sess)
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
            # compare qc_quantize op's output with input
            expected_output = np.ones((1, 1, 2, num_output_channels))
            expected_output[:, :, :, 1] *= 2
            expected_output[:, :, :, 2] *= 2.5
            self.assertTrue(np.allclose(out_data, expected_output, rtol=0.01))
            sess.close()

    def _test_qc_quantize_fp16_helper(self, device: str):
        """
        test custom op which is operating in fp16 mode
        """
        np.random.seed(0)
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 16
        use_symm_encoding = True

        with graph.as_default():
            # placeholder for the input
            num_output_channels = 3
            inp = tf.compat.v1.placeholder(tf.float32, shape=[10, num_output_channels], name='input')
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
            is_int_data_type = tf.Variable(initial_value=False, trainable=False, dtype=tf.bool)

            mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                                   trainable=False, dtype=tf.int32)
            # axis handling for getting number of channels.
            axis_handling = tf.Variable(initial_value=AxisHandling.LAST_AXIS.value, trainable=False, dtype=tf.int32)
            is_training = tf.keras.backend.learning_phase()

            sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                      encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                      is_int_data_type.initializer, axis_handling.initializer])

            with tf.device(device):
                pass_through_op_output = zero_out_module.qc_quantize_per_channel(name='quant_op', in_tensor=inp,
                                                                                 op_mode=mode_var,
                                                                                 tensor_quantizer_reference=tensor_quant_ref,
                                                                                 encoding_min=encoding_min,
                                                                                 encoding_max=encoding_max,
                                                                                 bit_width=bit_width,
                                                                                 use_symmetric_encoding=use_symmetric_encoding,
                                                                                 is_int_data_type=is_int_data_type,
                                                                                 axis_handling=axis_handling,
                                                                                 is_training=is_training)

                inp_tensor = sess.graph.get_tensor_by_name('input:0')
                inp_data = np.array([[0.78027296, 0.44164285, 0.6942797],
                                     [0.69774085, 0.55863863, 0.29553035],
                                     [0.219199, 0.09483732, 0.55075675],
                                     [0.6348504, 0.78027296, 0.44164285],
                                     [0.6942797, 0.69774085, 0.55863863],
                                     [0.29553035, 0.219199, 0.09483732],
                                     [0.55075675, 0.6348504, 0.78027296],
                                     [0.44164285, 0.6942797, 0.69774085],
                                     [0.55863863, 0.29553035, 0.219199],
                                     [0.09483732, 0.55075675, 0.6348504]], dtype=np.float32)

                out_exp_fp16 = np.array([[0.78027344, 0.4416504, 0.69433594],
                                    [0.6977539, 0.55859375, 0.29541016],
                                    [0.21923828, 0.09484863, 0.55078125],
                                    [0.6347656, 0.78027344, 0.4416504],
                                    [0.69433594, 0.6977539, 0.55859375],
                                    [0.29541016, 0.21923828, 0.09484863],
                                    [0.55078125, 0.6347656, 0.78027344],
                                    [0.4416504, 0.69433594, 0.6977539],
                                    [0.55859375, 0.29541016, 0.21923828],
                                    [0.09484863, 0.55078125, 0.6347656]], dtype=np.float32)


                # 1. verify that the output from the above op is as expected (mode: oneShotQuantizeDequantize)
                out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
                self.assertTrue(np.allclose(out_data, out_exp_fp16))

                # 2. change the mode now to updateStats and verify that the output is the same as input
                mode_var.load(int(libpymo.TensorQuantizerOpMode.updateStats), sess)
                out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
                self.assertTrue(np.allclose(out_data, inp_data))

                # 3. change the mode now to quantizeDequantize and verify that the output is equal to out_exp_fp16
                mode_var.load(int(libpymo.TensorQuantizerOpMode.quantizeDequantize), sess)
                out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
                self.assertTrue(np.allclose(out_data, out_exp_fp16))

                # 4. change the mode now to passThrough and verify that the output is equal to input
                mode_var.load(int(libpymo.TensorQuantizerOpMode.passThrough), sess)
                out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
                self.assertTrue(np.allclose(out_data, inp_data))

    def test_qc_quantize_fp16_cpu(self):
        """
        test per-channel op in fp16 mode (cpu)
        """
        self._test_qc_quantize_fp16_helper(device="/device:CPU:0")

    @pytest.mark.cuda
    def test_qc_quantize_fp16_gpu(self):
        """
        test per-channel op in fp16 mode (gpu)
        """
        self._test_qc_quantize_fp16_helper(device="/device:GPU:0")

    @pytest.mark.cuda
    def test_qc_quantize_op_gpu_conv(self):
        """
        test custom op with GPU

        """
        np.random.seed(0)
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
            is_int_data_type = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)

            mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                                   trainable=False, dtype=tf.int32)
            axis_handling = tf.Variable(initial_value=AxisHandling.LAST_AXIS.value, trainable=False, dtype=tf.int32)
            is_training = tf.keras.backend.learning_phase()

            sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                      encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                      is_int_data_type.initializer, axis_handling.initializer])
            with tf.device("/device:GPU:0"):
                pass_through_op_output = zero_out_module.qc_quantize_per_channel(name='quant_op', in_tensor=inp,
                                                                                 op_mode=mode_var,
                                                                                 tensor_quantizer_reference=tensor_quant_ref,
                                                                                 encoding_min=encoding_min,
                                                                                 encoding_max=encoding_max,
                                                                                 bit_width=bit_width,
                                                                                 use_symmetric_encoding=use_symmetric_encoding,
                                                                                 is_int_data_type=is_int_data_type,
                                                                                 axis_handling=axis_handling,
                                                                                 is_training=is_training)

            inp_tensor = sess.graph.get_tensor_by_name('input:0')
            inp_data = np.ones((1, 1, 2, num_output_channels))

            inp_data[:, :, :, 1] *= 2
            inp_data[:, :, :, 2] *= 3
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})

            mode_var.load(int(libpymo.TensorQuantizerOpMode.quantizeDequantize), sess)
            out_data_8_bits = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
            bit_width.load(16, sess)
            out_data_16_bits = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
            # compare qc_quantize op's output with input
            expected_output = np.ones((1, 1, 2, num_output_channels))
            expected_output[:, :, :, 1] *= 2
            expected_output[:, :, :, 2] *= 2.5
            self.assertTrue(np.allclose(out_data_8_bits, expected_output))
            self.assertTrue(np.allclose(out_data_16_bits, expected_output))
            sess.close()

    @pytest.mark.cuda
    def test_qc_quantize_per_channel_op_gpu_last_two_axes(self):
        """
        test per channel op with axis handling = last two axes on GPU
        """
        np.random.seed(0)
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = True

        with graph.as_default():
            # place holder for the input

            num_input_channels = 3
            channel_multiplier = 2
            num_output_channels = num_input_channels * channel_multiplier
            # tf nn depthwise2d conv weight shape is H,W,I,channel multiplier. Number of output channels = I * channel multiplier
            inp = tf.compat.v1.placeholder(tf.float32, shape=[1, 1, num_input_channels, channel_multiplier],
                                           name='depthwise_kernel')
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
            en_max = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            encoding_min = tf.Variable(en_min,
                                       trainable=True, dtype=tf.double)
            encoding_max = tf.Variable(en_max,
                                       trainable=True, dtype=tf.double)

            bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
            use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)
            is_int_data_type = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)

            mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                                   trainable=False, dtype=tf.int32)

            # Axis handling for depthwise
            axis_handling = tf.Variable(initial_value=AxisHandling.LAST_TWO_AXES.value, trainable=False, dtype=tf.int32)
            is_training = tf.keras.backend.learning_phase()

            sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                      encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                      is_int_data_type.initializer, axis_handling.initializer])
            with tf.device("/device:GPU:0"):
                pass_through_op_output = zero_out_module.qc_quantize_per_channel(name='quant_op', in_tensor=inp,
                                                                                 op_mode=mode_var,
                                                                                 tensor_quantizer_reference=tensor_quant_ref,
                                                                                 encoding_min=encoding_min,
                                                                                 encoding_max=encoding_max,
                                                                                 bit_width=bit_width,
                                                                                 use_symmetric_encoding=use_symmetric_encoding,
                                                                                 is_int_data_type=is_int_data_type,
                                                                                 axis_handling=axis_handling,
                                                                                 is_training=is_training)

            inp_tensor = sess.graph.get_tensor_by_name('depthwise_kernel:0')
            inp_data = np.zeros((1, 1, num_input_channels, channel_multiplier))

            inp_data[:, :, 0, 0] += 1.5
            inp_data[:, :, 0, 1] += 2.5
            inp_data[:, :, 1, 0] += 3.5
            inp_data[:, :, 1, 1] += 4.5
            inp_data[:, :, 2, 0] += 5.5
            inp_data[:, :, 2, 1] += 6.5
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})

            mode_var.load(int(libpymo.TensorQuantizerOpMode.quantizeDequantize), sess)
            out_data_8_bits = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
            bit_width.load(16, sess)
            out_data_16_bits = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
            # compare qc_quantize op's output with input
            expected_output = np.zeros((1, 1, num_input_channels, channel_multiplier))
            expected_output[:, :, 0, 0] += 1.0
            expected_output[:, :, 0, 1] += 2.0
            expected_output[:, :, 1, 0] += 3.0
            expected_output[:, :, 1, 1] += 4.0
            expected_output[:, :, 2, 0] += 5.0
            expected_output[:, :, 2, 1] += 6.0
            self.assertTrue(np.allclose(out_data_8_bits, expected_output))
            self.assertTrue(np.allclose(out_data_16_bits, expected_output))
            sess.close()

    def test_qc_quantize_op_cpu_linear(self):
        """
        test custom op with CPU
        """
        np.random.seed(0)
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

                encoding_min = tf.Variable([0.0, 0.0, 0.0],
                                           trainable=True, dtype=tf.double)
                encoding_max = tf.Variable([1.0, 2.0, 2.5],
                                           trainable=True, dtype=tf.double)

                bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
                use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)
                is_int_data_type = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)

                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                                       trainable=False, dtype=tf.int32)
                axis_handling = tf.Variable(initial_value=AxisHandling.LAST_AXIS.value, trainable=False, dtype=tf.int32)
                is_training = tf.keras.backend.learning_phase()

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          is_int_data_type.initializer, axis_handling.initializer])
                pass_through_op_output = zero_out_module.qc_quantize_per_channel(name='quant_op', in_tensor=inp,
                                                                                 op_mode=mode_var,
                                                                                 tensor_quantizer_reference=tensor_quant_ref,
                                                                                 encoding_min=encoding_min,
                                                                                 encoding_max=encoding_max,
                                                                                 bit_width=bit_width,
                                                                                 use_symmetric_encoding=use_symmetric_encoding,
                                                                                 is_int_data_type=is_int_data_type,
                                                                                 axis_handling=axis_handling,
                                                                                 is_training=is_training)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.ones((2, 3))
        inp_data[:, 1] *= 2
        inp_data[:, 2] *= 3

        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})

        mode_var.load(int(libpymo.TensorQuantizerOpMode.quantizeDequantize), sess)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        # compare qc_quantize op's output with input
        expected_output = np.ones((2, 3))
        expected_output[:, 1] *= 2
        expected_output[:, 2] *= 2.5
        self.assertTrue(np.allclose(out_data, expected_output, rtol=0.01))
        sess.close()


    def test_qc_quantize_op_cpu_bias(self):
        """
        test custom op with CPU
        """
        np.random.seed(0)
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
                is_int_data_type = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)

                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.updateStats),
                                       trainable=False, dtype=tf.int32)
                axis_handling = tf.Variable(initial_value=AxisHandling.LAST_AXIS.value, trainable=False, dtype=tf.int32)
                is_training = tf.keras.backend.learning_phase()

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          is_int_data_type.initializer, axis_handling.initializer])
                pass_through_op_output = zero_out_module.qc_quantize_per_channel(name='quant_op', in_tensor=inp,
                                                                                 op_mode=mode_var,
                                                                                 tensor_quantizer_reference=tensor_quant_ref,
                                                                                 encoding_min=encoding_min,
                                                                                 encoding_max=encoding_max,
                                                                                 bit_width=bit_width,
                                                                                 use_symmetric_encoding=use_symmetric_encoding,
                                                                                 is_int_data_type=is_int_data_type,
                                                                                 axis_handling=axis_handling,
                                                                                 is_training=is_training)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.ones(3)
        inp_data[1] *= 2
        inp_data[2] *= 3

        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})

        # compare qc_quantize op's output with input
        self.assertTrue(np.allclose(out_data, inp_data))
        sess.close()

    @pytest.mark.cuda
    def test_get_number_of_output_channels_and_quantization_axis(self):

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        with tf.device('/gpu:0'):
            inputs = tf.keras.Input(shape=(10, 10, 3,))
            conv2d = tf.keras.layers.Conv2D(16, (1, 1))(inputs)
            conv2d_transpose = tf.keras.layers.Conv2DTranspose(6, (2,2))(inputs)
            depthwise_conv2d = tf.keras.layers.DepthwiseConv2D(3)(inputs)
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)


        weight_shape = sess.run(sess.graph.get_tensor_by_name('conv2d/Conv2D/ReadVariableOp:0')).shape
        consumer_op_type = sess.graph.get_operation_by_name('conv2d/Conv2D').type
        num_output_channels, axis_handling = \
            QuantizationSimModel._get_number_of_output_channels_and_quantization_axis_handling(weight_shape,
                                                                                               consumer_op_type)
        assert num_output_channels == 16
        assert axis_handling == AxisHandling.LAST_AXIS

        weight_shape = sess.run(sess.graph.get_tensor_by_name('conv2d_transpose/conv2d_transpose/ReadVariableOp:0')).shape
        consumer_op_type = sess.graph.get_operation_by_name('conv2d_transpose/conv2d_transpose').type
        num_output_channels, axis_handling = \
            QuantizationSimModel._get_number_of_output_channels_and_quantization_axis_handling(weight_shape,
                                                                                               consumer_op_type)
        assert num_output_channels == 6
        assert axis_handling == AxisHandling.LAST_AXIS

        weight_shape = sess.run(sess.graph.get_tensor_by_name('depthwise_conv2d/depthwise/ReadVariableOp:0')).shape
        consumer_op_type = sess.graph.get_operation_by_name('depthwise_conv2d/depthwise').type
        num_output_channels, axis_handling = \
            QuantizationSimModel._get_number_of_output_channels_and_quantization_axis_handling(weight_shape,
                                                                                               consumer_op_type)
        assert num_output_channels == 3
        assert axis_handling == AxisHandling.LAST_TWO_AXES

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

            conv2d_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
            weight_val = np.random.randn(3, 3, 3, 2)
            WeightTensorUtils.update_tensor_for_op(sess, conv2d_op, weight_val)

            sim = QuantizationSimModel(sess, [model.input.op.name], [model.output.op.name], use_cuda=False,
                                       config_file='./quantsim_config.json')

            for quant_op_name, quantizer_info in sim._activation_quantizers.items():
                assert isinstance(quantizer_info.tensor_quantizer, libpymo.TensorQuantizer)

            for quant_op_name, quantizer_info in sim._param_quantizers.items():
                op = sim.session.graph.get_operation_by_name(quant_op_name)
                assert op.type == 'QcQuantizePerChannel'
                shape = op.inputs[0].shape.as_list()
                assert len(quantizer_info.tensor_quantizer) == shape[-1]

            def dummy_forward_pass(sess, args):
                model_output = sess.graph.get_tensor_by_name('conv2d/BiasAdd_quantized:0')

                model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
                np.random.seed(0)
                dummy_input = np.random.randn(20, 5, 5, 3)
                sess.run(model_output, feed_dict={model_input: dummy_input})

            sim.compute_encodings(dummy_forward_pass, None)

            encodings = []
            for quant_op_name, quantizer_info in sim._param_quantizers.items():
                if quantizer_info.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
                    encoding = quantizer_info.get_encoding()
                    assert isinstance(encoding, list)
                    lst = []
                    for enc in encoding:
                        lst.append((enc.min, enc.max))
                    encodings.append(lst)

            encoding_numpy = compute_tf_encodings_given_numpy_data(weight_val, axis=3)
            assert np.allclose(encoding_numpy, encodings, rtol=0.01)

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
            _encoding.delta, _encoding.offset = calculate_delta_offset(_encoding.min, _encoding.max, bitwidth=8,
                                                                       use_symmetric_encodings=False,
                                                                       use_strict_symmetric=False)
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
        self.assertEqual(len(encoding_data["param_encodings"]["conv2d/Conv2D/ReadVariableOp:0"]), 32)

    @pytest.mark.cuda
    def test_compute_encodings_gpu_model(self):
        """
        Create QuantSim for a GPU model and test that activation encodings are computed
        """
        save_config_file_for_per_channel_quantization()

        tf.compat.v1.reset_default_graph()
        with tf.device('/gpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(3, kernel_size=2, input_shape=(3, 3, 3)))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        conv2d_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        weight_val = np.random.randn(2, 2, 3, 3)
        WeightTensorUtils.update_tensor_for_op(sess, conv2d_op, weight_val)

        sim = QuantizationSimModel(sess, ['conv2d_input'], ['conv2d/BiasAdd'], use_cuda=True,
                                   config_file='./quantsim_config.json')

        def dummy_forward_pass(sess, args):
            # model_output = sess.graph.get_tensor_by_name('dense/Relu_quantized:0')
            model_output = sess.graph.get_tensor_by_name('conv2d/BiasAdd_quantized:0')
            model_input = sess.graph.get_tensor_by_name('conv2d_input:0')
            dummy_input = np.random.randn(1, 3, 3, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        encodings = []
        for quant_op_name, quantizer_info in sim._param_quantizers.items():
            if quantizer_info.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
                encoding = quantizer_info.get_encoding()
                lst = []
                for enc in encoding:
                    lst.append((enc.min, enc.max))
                encodings.append(lst)

        encoding_numpy = compute_tf_encodings_given_numpy_data(weight_val, axis=3)
        assert np.allclose(encoding_numpy, encodings, rtol=0.01)


    @pytest.mark.cuda
    def test_to_compare_time_per_channel_and_per_tensor_quantization(self):
        save_config_file_for_per_channel_quantization()

        def create_sim_run_compute_encodings(config_file):

            tf.compat.v1.reset_default_graph()

            def dummy_forward_pass(sess, args):
                np.random.seed(0)
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
                                       config_file=config_file)


            start_time = time.time()
            sim.compute_encodings(dummy_forward_pass, None)
            compute_encodings_time = time.time() - start_time
            return compute_encodings_time

        # Run one comment it, uncomment the other one and run again
        per_channel_time = create_sim_run_compute_encodings('./quantsim_config.json')
        print(per_channel_time)
        # per_tensor_time = create_sim_run_compute_encodings(None)
        # print(per_tensor_time)

    def test_compute_encodings_cpu_model_depthwise_model(self):
        """
        Create QuantSim for a CPU model and test that activation encodings are computed
        """
        save_config_file_for_per_channel_quantization()

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with tf.device('/cpu:0'):
            _ = depthwise_conv2d_model()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

        sim = QuantizationSimModel(sess, ['input_1'], ['depthwise_conv2d_model/Softmax'], use_cuda=False,
                                   config_file='./quantsim_config.json')

        def dummy_forward_pass(sess, _):
            model_output = sess.graph.get_tensor_by_name('depthwise_conv2d/BiasAdd:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(1, 10, 10, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)
        for quant_op_name, quantizer_info in sim._param_quantizers.items():
            if quantizer_info.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
                encoding = quantizer_info.get_encoding()
                assert isinstance(encoding, list)

        assert len(sim._param_quantizers['conv2d/Conv2D/ReadVariableOp_quantized'].tensor_quantizer) == 16
        assert len(sim._param_quantizers['conv2d/BiasAdd/ReadVariableOp_quantized'].tensor_quantizer) == 16
        assert len(sim._param_quantizers['separable_conv2d/separable_conv2d/ReadVariableOp_quantized'].tensor_quantizer) == 16
        assert len(sim._param_quantizers['separable_conv2d/separable_conv2d/ReadVariableOp_1_quantized'].tensor_quantizer) == 10
        assert len(sim._param_quantizers['separable_conv2d/BiasAdd/ReadVariableOp_quantized'].tensor_quantizer) == 10
        assert len(sim._param_quantizers['depthwise_conv2d/depthwise/ReadVariableOp_quantized'].tensor_quantizer) == 10
        assert len(sim._param_quantizers['depthwise_conv2d/BiasAdd/ReadVariableOp_quantized'].tensor_quantizer) == 10

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

        def dummy_forward_pass(sess, _):
            model_output = sess.graph.get_tensor_by_name('depthwise_conv2d/BiasAdd:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(1, 10, 10, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)
        for quant_op_name, quantizer_info in sim._param_quantizers.items():
            if quantizer_info.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
                encoding = quantizer_info.get_encoding()
                assert isinstance(encoding, list)

        assert len(sim._param_quantizers['conv2d/Conv2D/ReadVariableOp_quantized'].tensor_quantizer) == 16
        assert len(sim._param_quantizers['conv2d/BiasAdd/ReadVariableOp_quantized'].tensor_quantizer) == 16
        assert len(sim._param_quantizers['separable_conv2d/separable_conv2d/ReadVariableOp_quantized'].tensor_quantizer) == 16
        assert len(sim._param_quantizers['separable_conv2d/separable_conv2d/ReadVariableOp_1_quantized'].tensor_quantizer) == 10
        assert len(sim._param_quantizers['separable_conv2d/BiasAdd/ReadVariableOp_quantized'].tensor_quantizer) == 10
        assert len(sim._param_quantizers['depthwise_conv2d/depthwise/ReadVariableOp_quantized'].tensor_quantizer) == 10
        assert len(sim._param_quantizers['depthwise_conv2d/BiasAdd/ReadVariableOp_quantized'].tensor_quantizer) == 10


    @pytest.mark.cuda
    def test_compute_encodings_transposed_conv_model(self):
        save_config_file_for_per_channel_quantization()

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with tf.device('/gpu:0'):
            _ = transposed_conv2d_model()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
        op = sess.graph.get_operations()
        weight_val = sess.run(sess.graph.get_operation_by_name('conv2d_transpose/kernel/Read/ReadVariableOp').outputs[0])

        sim = QuantizationSimModel(sess, ['input_1'], ['conv2d_transpose/BiasAdd'], use_cuda=True,
                                   config_file='./quantsim_config.json', quant_scheme='tf')

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('conv2d_transpose/BiasAdd:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(1, 7, 7, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        encodings = []
        for quant_op_name, quantizer_info in sim._param_quantizers.items():
            if quantizer_info.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
                encoding = quantizer_info.get_encoding()
                lst = []
                for enc in encoding:
                    lst.append((enc.min, enc.max))
                encodings.append(lst)

        encoding_numpy = compute_tf_encodings_given_numpy_data(weight_val, axis=2)
        assert np.allclose(encoding_numpy, encodings, rtol=0.01)
        sim.export('/tmp', 'quant_sim_model')
        with open('/tmp/quant_sim_model.encodings') as json_file:
            encoding_data = json.load(json_file)

        param_keys = list(encoding_data["param_encodings"].keys())
        assert param_keys[0] == "conv2d_transpose/conv2d_transpose/ReadVariableOp:0"
        sess.close()

    @pytest.mark.cuda
    def test_compute_encodings_gpu_model_nn_depthwise_model(self):
        """
        Create QuantSim for a CPU model and test that activation encodings are computed
        """
        save_config_file_for_per_channel_quantization()

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with tf.device('/gpu:0'):
            _ = nn_depthwise_conv2d_model()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

        sim = QuantizationSimModel(sess, ['input_1'], ['Softmax'], use_cuda=True,
                                   config_file='./quantsim_config.json')

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(1, 10, 10, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)
        for quant_op_name, quantizer_info in sim._param_quantizers.items():
            if quantizer_info.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
                encoding = quantizer_info.get_encoding()
                assert isinstance(encoding, list)

        assert len(sim._param_quantizers['depthwise/ReadVariableOp_quantized'].tensor_quantizer) == 6

    # Mark below test as cuda until per channel on cpu is supported.
    @pytest.mark.cuda
    def test_per_channel_range_learning(self):
        """
        Test to validate per channel range learning
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

        save_config_file_bias_quantized_for_per_channel_quantization()

        # create quantsim model without config file
        sim = QuantizationSimModel(sess, ['input_1'], ['dense/BiasAdd'], use_cuda=True,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   config_file='./quantsim_config.json')

        def dummy_forward_pass(sess, _):
            model_output = sess.graph.get_tensor_by_name('dense/BiasAdd_quantized:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            shape = model_input.shape
            dummy_input = np.random.randn(1, shape[1], shape[2], shape[3])
            sess.run(model_output, feed_dict={model_input: dummy_input})

        conv2d_weight_quant_op = sim.session.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
        conv2d_output_quant_op = sim.session.graph.get_operation_by_name('conv2d/BiasAdd_quantized')
        dense_bias_quant_op = sim.session.graph.get_operation_by_name('dense/BiasAdd/ReadVariableOp_quantized')

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

            conv_inp_tensor = conv2d_weight_quant_op.inputs[0]
            grads = tf.gradients(loss, [conv_inp_tensor,
                                        conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_min],
                                        conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_max],
                                        dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_min],
                                        dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_max]])
            _, conv_dqbydmin, conv_dqbydmax, dense_dqbydmin, dense_dqbydmax = grads
            conv2d_weight_min_gradient = sim.session.run(conv_dqbydmin,
                                                         feed_dict={inp_tensor: inp_data,
                                                                    labels_placeholder: one_hot_labels})
            conv2d_weight_max_gradient = sim.session.run(conv_dqbydmax,
                                                         feed_dict={inp_tensor: inp_data,
                                                                    labels_placeholder: one_hot_labels})
            dense_bias_min_gradient = sim.session.run(dense_dqbydmin,
                                                      feed_dict={inp_tensor: inp_data,
                                                                 labels_placeholder: one_hot_labels})
            dense_bias_max_gradient = sim.session.run(dense_dqbydmax,
                                                      feed_dict={inp_tensor: inp_data,
                                                                 labels_placeholder: one_hot_labels})

            assert len(conv2d_weight_min_gradient) == 2
            assert len(conv2d_weight_max_gradient) == 2
            assert len(dense_bias_min_gradient) == 10
            assert len(dense_bias_max_gradient) == 10

            weights_before_train = sim.session.run(conv2d_weight_quant_op.inputs[0])
            encoding_min_before_train = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_min])
            encoding_max_before_train = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_max])
            conv2d_output_encoding_min_before_train = sim.session.run(conv2d_output_quant_op.inputs[
                                                                          QuantizeOpIndices.encoding_min])
            conv2d_output_encoding_max_before_train = sim.session.run(conv2d_output_quant_op.inputs[
                                                                          QuantizeOpIndices.encoding_max])
            dense_bias_encoding_min_before_train = \
                sim.session.run(dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_min])
            dense_bias_encoding_max_before_train = \
                sim.session.run(dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_max])
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
            conv2d_output_encoding_min_after_train = sim.session.run(conv2d_output_quant_op.inputs[
                                                                         QuantizeOpIndices.encoding_min])
            conv2d_output_encoding_max_after_train = sim.session.run(conv2d_output_quant_op.inputs[
                                                                         QuantizeOpIndices.encoding_max])
            encoding_min_after_train = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_min])
            encoding_max_after_train = sim.session.run(conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_max])
            dense_bias_encoding_min_after_train = \
                sim.session.run(dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_min])
            dense_bias_encoding_max_after_train = \
                sim.session.run(dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_max])

            assert not np.allclose(weights_before_train, weights_after_train, atol=1e-6)
            assert not np.array_equal(encoding_min_before_train, encoding_min_after_train)
            assert not np.array_equal(encoding_max_before_train, encoding_max_after_train)
            assert not np.array_equal(conv2d_output_encoding_min_before_train, conv2d_output_encoding_min_after_train)
            assert not np.array_equal(conv2d_output_encoding_max_before_train, conv2d_output_encoding_max_after_train)
            assert not np.array_equal(dense_bias_encoding_min_before_train, dense_bias_encoding_min_after_train)
            assert not np.array_equal(dense_bias_encoding_max_before_train, dense_bias_encoding_max_after_train)

        sess.close()
        sim.session.close()

    @pytest.mark.cuda
    def test_per_channel_range_learning_transposed_conv_model(self):
        """
        Test to validate per channel range learning transposed conv model
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        np.random.seed(0)
        with tf.device('/cpu:0'):
            x = transposed_conv2d_model()
            x = tf.keras.layers.Flatten()(x)
            _ = tf.keras.layers.Dense(10, bias_initializer='random_uniform')(x)

        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        save_config_file_bias_quantized_for_per_channel_quantization()

        # create quantsim model without config file
        sim = QuantizationSimModel(sess, ['input_1'], ['dense/BiasAdd'], use_cuda=True,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   config_file='./quantsim_config.json')

        def dummy_forward_pass(sess, _):
            model_output = sess.graph.get_tensor_by_name('dense/BiasAdd_quantized:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            shape = model_input.shape
            dummy_input = np.random.randn(1, shape[1], shape[2], shape[3])
            sess.run(model_output, feed_dict={model_input: dummy_input})

        conv2d_transpose_weight_quant_op = \
            sim.session.graph.get_operation_by_name('conv2d_transpose/conv2d_transpose/ReadVariableOp_quantized')
        conv2d_transpose_output_quant_op = sim.session.graph.get_operation_by_name('conv2d_transpose/BiasAdd_quantized')
        dense_bias_quant_op = sim.session.graph.get_operation_by_name('dense/BiasAdd/ReadVariableOp_quantized')

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

            conv_inp_tensor = conv2d_transpose_weight_quant_op.inputs[0]
            grads = tf.gradients(loss, [conv_inp_tensor,
                                        conv2d_transpose_weight_quant_op.inputs[QuantizeOpIndices.encoding_min],
                                        conv2d_transpose_weight_quant_op.inputs[QuantizeOpIndices.encoding_max],
                                        dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_min],
                                        dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_max]])
            _, conv_dqbydmin, conv_dqbydmax, dense_dqbydmin, dense_dqbydmax = grads
            conv2d_transpose_weight_min_gradient = sim.session.run(conv_dqbydmin,
                                                                   feed_dict={inp_tensor: inp_data,
                                                                              labels_placeholder: one_hot_labels})
            conv2d_transpose_weight_max_gradient = sim.session.run(conv_dqbydmax,
                                                                   feed_dict={inp_tensor: inp_data,
                                                                              labels_placeholder: one_hot_labels})
            dense_bias_min_gradient = sim.session.run(dense_dqbydmin,
                                                      feed_dict={inp_tensor: inp_data,
                                                                 labels_placeholder: one_hot_labels})
            dense_bias_max_gradient = sim.session.run(dense_dqbydmax,
                                                      feed_dict={inp_tensor: inp_data,
                                                                 labels_placeholder: one_hot_labels})

            assert len(conv2d_transpose_weight_min_gradient) == 3
            assert len(conv2d_transpose_weight_max_gradient) == 3
            assert len(dense_bias_min_gradient) == 10
            assert len(dense_bias_max_gradient) == 10

            weights_before_train = sim.session.run(conv2d_transpose_weight_quant_op.inputs[0])
            encoding_min_before_train = sim.session.run(
                conv2d_transpose_weight_quant_op.inputs[QuantizeOpIndices.encoding_min])
            encoding_max_before_train = sim.session.run(
                conv2d_transpose_weight_quant_op.inputs[QuantizeOpIndices.encoding_max])
            conv2d_transpose_output_encoding_min_before_train = sim.session.run(conv2d_transpose_output_quant_op.inputs[
                                                                                    QuantizeOpIndices.encoding_min])
            conv2d_transpose_output_encoding_max_before_train = sim.session.run(conv2d_transpose_output_quant_op.inputs[
                                                                                    QuantizeOpIndices.encoding_max])
            dense_bias_encoding_min_before_train = \
                sim.session.run(dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_min])
            dense_bias_encoding_max_before_train = \
                sim.session.run(dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_max])
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

            weights_after_train = sim.session.run(conv2d_transpose_weight_quant_op.inputs[0])
            conv2d_transpose_output_encoding_min_after_train = sim.session.run(conv2d_transpose_output_quant_op.inputs[
                                                                                   QuantizeOpIndices.encoding_min])
            conv2d_transpose_output_encoding_max_after_train = sim.session.run(conv2d_transpose_output_quant_op.inputs[
                                                                                   QuantizeOpIndices.encoding_max])
            encoding_min_after_train = sim.session.run(
                conv2d_transpose_weight_quant_op.inputs[QuantizeOpIndices.encoding_min])
            encoding_max_after_train = sim.session.run(
                conv2d_transpose_weight_quant_op.inputs[QuantizeOpIndices.encoding_max])
            dense_bias_encoding_min_after_train = \
                sim.session.run(dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_min])
            dense_bias_encoding_max_after_train = \
                sim.session.run(dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_max])

            assert not np.allclose(weights_before_train, weights_after_train, atol=1e-6)
            assert not np.array_equal(encoding_min_before_train, encoding_min_after_train)
            assert not np.array_equal(encoding_max_before_train, encoding_max_after_train)
            assert not np.array_equal(conv2d_transpose_output_encoding_min_before_train,
                                      conv2d_transpose_output_encoding_min_after_train)
            assert not np.array_equal(conv2d_transpose_output_encoding_max_before_train,
                                      conv2d_transpose_output_encoding_max_after_train)
            assert not np.array_equal(dense_bias_encoding_min_before_train, dense_bias_encoding_min_after_train)
            assert not np.array_equal(dense_bias_encoding_max_before_train, dense_bias_encoding_max_after_train)

        sess.close()
        sim.session.close()

    def test_per_channel_range_learning_depthwise_conv_model(self):
        """
        Test to validate per channel range learning depthwise conv model
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        np.random.seed(0)
        with tf.device('/cpu:0'):
            inputs = tf.keras.Input(shape=(32, 32, 4,))
            depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3)(inputs)
            relu_op = tf.nn.relu(depthwise_conv)
            reshape = tf.keras.layers.Flatten()(relu_op)
            _ = tf.keras.layers.Dense(10, bias_initializer='random_uniform')(reshape)

        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        initialize_uninitialized_vars(sess)

        save_config_file_bias_quantized_for_per_channel_quantization()

        sim = QuantizationSimModel(sess, ['input_1'], ['dense/BiasAdd'], use_cuda=True,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   config_file='./quantsim_config.json')

        def dummy_forward_pass(sess, _):
            model_output = sess.graph.get_tensor_by_name('dense/BiasAdd_quantized:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            shape = model_input.shape
            dummy_input = np.random.randn(1, shape[1], shape[2], shape[3])
            sess.run(model_output, feed_dict={model_input: dummy_input})

        depthwise_conv2d_weight_quant_op = \
            sim.session.graph.get_operation_by_name('depthwise_conv2d/depthwise/ReadVariableOp_quantized')
        depthwise_conv2d_output_quant_op = sim.session.graph.get_operation_by_name('depthwise_conv2d/BiasAdd_quantized')
        dense_bias_quant_op = sim.session.graph.get_operation_by_name('dense/BiasAdd/ReadVariableOp_quantized')

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

            conv_inp_tensor = depthwise_conv2d_weight_quant_op.inputs[0]
            grads = tf.gradients(loss, [conv_inp_tensor,
                                        depthwise_conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_min],
                                        depthwise_conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_max],
                                        dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_min],
                                        dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_max]])
            _, conv_dqbydmin, conv_dqbydmax, dense_dqbydmin, dense_dqbydmax = grads
            depthwise_conv2d_weight_min_gradient = sim.session.run(conv_dqbydmin,
                                                                   feed_dict={inp_tensor: inp_data,
                                                                              labels_placeholder: one_hot_labels})
            depthwise_conv2d_weight_max_gradient = sim.session.run(conv_dqbydmax,
                                                                   feed_dict={inp_tensor: inp_data,
                                                                              labels_placeholder: one_hot_labels})
            dense_bias_min_gradient = sim.session.run(dense_dqbydmin,
                                                      feed_dict={inp_tensor: inp_data,
                                                                 labels_placeholder: one_hot_labels})
            dense_bias_max_gradient = sim.session.run(dense_dqbydmax,
                                                      feed_dict={inp_tensor: inp_data,
                                                                 labels_placeholder: one_hot_labels})

            assert len(depthwise_conv2d_weight_min_gradient) == 4
            assert len(depthwise_conv2d_weight_max_gradient) == 4
            assert len(dense_bias_min_gradient) == 10
            assert len(dense_bias_max_gradient) == 10

            weights_before_train = sim.session.run(depthwise_conv2d_weight_quant_op.inputs[0])
            encoding_min_before_train = sim.session.run(
                depthwise_conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_min])
            encoding_max_before_train = sim.session.run(
                depthwise_conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_max])
            depthwise_conv2d_output_encoding_min_before_train = sim.session.run(depthwise_conv2d_output_quant_op.inputs[
                                                                                QuantizeOpIndices.encoding_min])
            depthwise_conv2d_output_encoding_max_before_train = sim.session.run(depthwise_conv2d_output_quant_op.inputs[
                                                                                QuantizeOpIndices.encoding_max])
            dense_bias_encoding_min_before_train = \
                sim.session.run(dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_min])
            dense_bias_encoding_max_before_train = \
                sim.session.run(dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_max])
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

            weights_after_train = sim.session.run(depthwise_conv2d_weight_quant_op.inputs[0])
            depthwise_conv2d_output_encoding_min_after_train = sim.session.run(depthwise_conv2d_output_quant_op.inputs[
                                                                                   QuantizeOpIndices.encoding_min])
            depthwise_conv2d_output_encoding_max_after_train = sim.session.run(depthwise_conv2d_output_quant_op.inputs[
                                                                                   QuantizeOpIndices.encoding_max])
            encoding_min_after_train = sim.session.run(
                depthwise_conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_min])
            encoding_max_after_train = sim.session.run(
                depthwise_conv2d_weight_quant_op.inputs[QuantizeOpIndices.encoding_max])
            dense_bias_encoding_min_after_train = \
                sim.session.run(dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_min])
            dense_bias_encoding_max_after_train = \
                sim.session.run(dense_bias_quant_op.inputs[QuantizeOpIndices.encoding_max])

            assert not np.allclose(weights_before_train, weights_after_train, atol=1e-6)
            assert not np.array_equal(encoding_min_before_train, encoding_min_after_train)
            assert not np.array_equal(encoding_max_before_train, encoding_max_after_train)
            assert not np.array_equal(depthwise_conv2d_output_encoding_min_before_train,
                                      depthwise_conv2d_output_encoding_min_after_train)
            assert not np.array_equal(depthwise_conv2d_output_encoding_max_before_train,
                                      depthwise_conv2d_output_encoding_max_after_train)
            assert not np.array_equal(dense_bias_encoding_min_before_train, dense_bias_encoding_min_after_train)
            assert not np.array_equal(dense_bias_encoding_max_before_train, dense_bias_encoding_max_after_train)

        sess.close()
        sim.session.close()

    def test_save_model_with_embedded_quantization_nodes_per_channel(self):
        save_config_file_for_per_channel_quantization()

        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(5, 5, 3), activation='relu'))
            model.summary()

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        with sess.graph.as_default():
            first_conv_tensor_var = [var for var in tf.compat.v1.global_variables() if var.name == 'conv2d/kernel:0'][0]
            first_conv_tensor_var.load(np.ones([3,3,3,32]), sess)
            saver = tf.compat.v1.train.Saver()
        saver.save(sess, save_path='/tmp/quantsim/'+'orig_model_before_quantsim')
        sim = QuantizationSimModel(sess, [model.input.op.name], [model.output.op.name], use_cuda=False,
                                   config_file='./quantsim_config.json')

        def create_encoding():
            _encoding = libpymo.TfEncoding()
            _encoding.min = random.uniform(0, 1)
            _encoding.max = random.uniform(1, 3)
            _encoding.bw = 8
            _encoding.delta, _encoding.offset = calculate_delta_offset(_encoding.min, _encoding.max, bitwidth=8,
                                                                       use_symmetric_encodings=False,
                                                                       use_strict_symmetric=False)
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

        # Make some changes to model parameters to see if they are part of the exported model
        quantizer_info = sim._param_quantizers['conv2d/Conv2D/ReadVariableOp_quantized']
        encoding = []
        for i in range(len(quantizer_info.tensor_quantizer)):
            _encoding = libpymo.TfEncoding()
            _encoding.min = 0
            _encoding.max = 1
            _encoding.bw = 8
            encoding.append(_encoding)
        quantizer_info.set_encoding(encoding)
        all_op_types = [op.type for op in sim.session.graph.get_operations()]
        self.assertIn('QcQuantize', all_op_types)
        self.assertNotIn('FakeQuantWithMinMaxVarsPerChannel', all_op_types)

        # Save model without encodings file
        sim.save_model_with_embedded_quantization_nodes(os.path.join('/tmp', 'tf_fakequant_model'))
        new_sess = load_model_from_meta('/tmp/tf_fakequant_model_embedded_quant_nodes.meta')
        all_op_types = [op.type for op in new_sess.graph.get_operations()]
        self.assertNotIn('QcQuantize', all_op_types)
        self.assertIn('FakeQuantWithMinMaxVarsPerChannel', all_op_types)
        first_conv_tensor = new_sess.graph.get_tensor_by_name('conv2d/Conv2D/ReadVariableOp:0')
        first_conv_tensor_val = new_sess.run(first_conv_tensor)
        self.assertTrue(np.any(first_conv_tensor_val == 1))
        first_conv_tensor_fakequant_max_tensor = new_sess.graph.get_tensor_by_name('conv2d/Conv2D/ReadVariableOp_quantized/max:0')
        first_conv_tensor_fakequant_max_val = new_sess.run(first_conv_tensor_fakequant_max_tensor)
        self.assertTrue(np.any(first_conv_tensor_fakequant_max_val == 1))
        self.assertTrue(isinstance(first_conv_tensor_fakequant_max_val, np.ndarray))

        # Save model with encodings file
        sim._export_encodings(os.path.join('/tmp', 'quant_sim_model.encodings'))
        sim.save_model_with_embedded_quantization_nodes(os.path.join('/tmp', 'tf_fakequant_model'), '/tmp/quant_sim_model.encodings')
        new_sess = load_model_from_meta('/tmp/tf_fakequant_model_embedded_quant_nodes.meta')
        all_op_types = [op.type for op in new_sess.graph.get_operations()]
        self.assertNotIn('QcQuantize', all_op_types)
        self.assertIn('FakeQuantWithMinMaxVarsPerChannel', all_op_types)
        first_conv_tensor = new_sess.graph.get_tensor_by_name('conv2d/Conv2D/ReadVariableOp:0')
        first_conv_tensor_val = new_sess.run(first_conv_tensor)
        self.assertTrue(np.any(first_conv_tensor_val == 1))
        first_conv_tensor_fakequant_max_tensor = new_sess.graph.get_tensor_by_name('conv2d/Conv2D/ReadVariableOp_quantized/max:0')
        first_conv_tensor_fakequant_max_val = new_sess.run(first_conv_tensor_fakequant_max_tensor)
        self.assertTrue(np.any(first_conv_tensor_fakequant_max_val == 1))
        self.assertTrue(isinstance(first_conv_tensor_fakequant_max_val, np.ndarray))

        sess.close()
        sim.session.close()
        new_sess.close()
        del sim


def save_config_file_for_per_channel_quantization():
    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "False"
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": "False"
            },
            "per_channel_quantization": "True",
        },
        "params": {"bias": {
            "is_quantized": "False"
        }},
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }

    with open('./quantsim_config.json', 'w') as f:
        json.dump(quantsim_config, f)


def save_config_file_bias_quantized_for_per_channel_quantization():
    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "False"
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": "False"
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


def compute_tf_encodings(sess, op, axis):

    data = WeightTensorUtils.get_tensor_as_numpy_data(sess, op)
    # print(data)
    data_size = np.size(data, axis)
    encodings = []
    for i in range(data_size):
        if axis == 1:
            data_flatten = data[:, i]
        else:
            data_flatten = data[:, :, :, i].flatten()
        encoding_min = min(0.0, np.min(data_flatten))
        encoding_max = max(0.0, np.max(data_flatten))
        encoding_max = max(encoding_max, encoding_max + 1e-05)
        encodings.append((encoding_min, encoding_max))

    return encodings

def compute_tf_encodings_given_numpy_data(data, axis):

    data_size = np.size(data, axis)
    encodings = []
    for i in range(data_size):
        if axis == 1:
            data_flatten = data[:, i]
        elif axis == 3:
            data_flatten = data[:, :, :, i].flatten()
        elif axis == 2:
            data_flatten = data[:, :, i, :].flatten()

        encoding_min = min(0, np.min(data_flatten))
        encoding_max = max(0, np.max(data_flatten))
        encodings.append((encoding_min, encoding_max))

    return encodings
