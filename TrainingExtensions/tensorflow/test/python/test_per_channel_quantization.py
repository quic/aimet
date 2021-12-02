# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

import numpy as np

import os
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.logging.WARN)
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
                axis = tf.Variable(initial_value=3, trainable=False, dtype=tf.int8)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          axis.initializer])
                # Giving axis = 3
                pass_through_op_output = zero_out_module.qc_quantize_per_channel_param(name='quant_op', in_tensor=inp,
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
                axis = tf.Variable(initial_value=1, trainable=False, dtype=tf.int8)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          axis.initializer])
                # Giving axis = 1
                pass_through_op_output = zero_out_module.qc_quantize_per_channel_param(name='quant_op', in_tensor=inp,
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
                axis = tf.Variable(initial_value=0, trainable=False, dtype=tf.int8)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          axis.initializer])
                # Giving axis = 3
                pass_through_op_output = zero_out_module.qc_quantize_per_channel_param(name='quant_op', in_tensor=inp,
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

