# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import aimet_common.libpymo as libpymo
import aimet_common.libaimet_tf_ops as zero_out_module
from aimet_tensorflow.utils.constants import QuantizeOpIndices
from aimet_tensorflow import quantsim_straight_through_grad

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()


class TestTrainingExtensionsQcQuantizeOp(unittest.TestCase):

    def test_qc_quantize_op_cpu(self):
        """
        test custom op with CPU
        """
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = True

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')
                tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                           libpymo.RoundingMode.ROUND_NEAREST)
                tensor_quantizer_val = libpymo.PtrToInt64(tensor_quantizer)
                tensor_quant_ref = tf.Variable(initial_value=tensor_quantizer_val, trainable=False, dtype=tf.int64)

                encoding_min = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
                encoding_max = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
                bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
                use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)
                is_int_data_type = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)

                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.updateStats),
                                       trainable=False, dtype=tf.int32)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                         encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          is_int_data_type.initializer])

                pass_through_op_output = zero_out_module.qc_quantize(name='quant_op', in_tensor=inp,
                                                                     op_mode=mode_var,
                                                                     tensor_quantizer_reference=tensor_quant_ref,
                                                                     encoding_min=encoding_min,
                                                                     encoding_max=encoding_max,
                                                                     bit_width=bit_width,
                                                                     use_symmetric_encoding=use_symmetric_encoding,
                                                                     is_int_data_type=is_int_data_type)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.random.rand(10)

        # get the output
        print("inp_data", inp_data)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print("out_data", out_data)

        # compare qc_quantize op's output with input
        self.assertTrue(np.allclose(out_data, inp_data))

        # compute encodings
        self.assertFalse(tensor_quantizer.isEncodingValid)
        encoding = tensor_quantizer.computeEncoding(bitwidth, use_symm_encoding)
        self.assertTrue(tensor_quantizer.isEncodingValid)
        print('min=', encoding.min, ', max=', encoding.max)

        # get the output
        inp_data = np.random.rand(10) * 2
        print(inp_data)
        mode_var.load(int(libpymo.TensorQuantizerOpMode.quantizeDequantize), sess)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print(out_data)

        # compare qc_quantize op's output with input
        self.assertFalse(np.allclose(out_data, inp_data))
        sess.close()

    def test_qc_quantize_op_cpu_fp16_quantize_dequantize(self):
        """
        test qc_quantize custom op with CPU with fp16 data type
        """
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = True

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')
                tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                           libpymo.RoundingMode.ROUND_NEAREST)
                tensor_quantizer_val = libpymo.PtrToInt64(tensor_quantizer)
                tensor_quant_ref = tf.Variable(initial_value=tensor_quantizer_val, trainable=False, dtype=tf.int64)

                encoding_min = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
                encoding_max = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
                bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
                use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)
                is_int_data_type = tf.Variable(initial_value=False, trainable=False, dtype=tf.bool)
                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.quantizeDequantize),
                                       trainable=False, dtype=tf.int32)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                         encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          is_int_data_type.initializer])

                pass_through_op_output = zero_out_module.qc_quantize(name='quant_op', in_tensor=inp,
                                                                     op_mode=mode_var,
                                                                     tensor_quantizer_reference=tensor_quant_ref,
                                                                     encoding_min=encoding_min,
                                                                     encoding_max=encoding_max,
                                                                     bit_width=bit_width,
                                                                     use_symmetric_encoding=use_symmetric_encoding,
                                                                     is_int_data_type=is_int_data_type)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')

        inp_data = np.array([0.78027299, 0.44164284, 0.6942797, 0.69774088, 0.55863863, 0.29553034, 0.219199,
                             0.09483732, 0.55075674, 0.6348504], dtype=np.float32)

        out_exp = np.array([0.78027344, 0.4416504, 0.69433594, 0.6977539, 0.55859375, 0.29541016, 0.21923828,
                            0.09484863, 0.55078125, 0.6347656], dtype=np.float32)

        # get the output
        print("inp_data", inp_data)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print("out_data", out_data)

        # compare qc_quantize op's output with expected output
        self.assertTrue(np.allclose(out_data, out_exp))

        sess.close()

    def test_qc_quantize_op_cpu_fp16_pass_through(self):
        """
        test qc_quantize custom op with CPU with fp16 data type in pass through mode
        """
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = True

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')
                tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                           libpymo.RoundingMode.ROUND_NEAREST)
                tensor_quantizer_val = libpymo.PtrToInt64(tensor_quantizer)
                tensor_quant_ref = tf.Variable(initial_value=tensor_quantizer_val, trainable=False, dtype=tf.int64)

                encoding_min = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
                encoding_max = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
                bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
                use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)
                is_int_data_type = tf.Variable(initial_value=False, trainable=False, dtype=tf.bool)
                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.passThrough),
                                       trainable=False, dtype=tf.int32)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                         encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          is_int_data_type.initializer])

                pass_through_op_output = zero_out_module.qc_quantize(name='quant_op', in_tensor=inp,
                                                                     op_mode=mode_var,
                                                                     tensor_quantizer_reference=tensor_quant_ref,
                                                                     encoding_min=encoding_min,
                                                                     encoding_max=encoding_max,
                                                                     bit_width=bit_width,
                                                                     use_symmetric_encoding=use_symmetric_encoding,
                                                                     is_int_data_type=is_int_data_type)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')

        inp_data = np.array([0.78027299, 0.44164284, 0.6942797, 0.69774088, 0.55863863, 0.29553034, 0.219199,
                             0.09483732, 0.55075674, 0.6348504], dtype=np.float32)

        # get the output
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})

        # compare qc_quantize op's output with expected output
        self.assertTrue(np.allclose(out_data, inp_data))

        sess.close()

    def test_qc_quantize_op_oneshot_cpu(self):
        """
        test custom op with CPU
        """
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = False
        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')
                tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                           libpymo.RoundingMode.ROUND_NEAREST)
                tensor_quantizer_val = libpymo.PtrToInt64(tensor_quantizer)
                tensor_quant_ref = tf.Variable(initial_value=tensor_quantizer_val, trainable=False, dtype=tf.int64)

                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                                       trainable=False, dtype=tf.int32)

                encoding_min = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
                encoding_max = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
                bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
                use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)
                is_int_data_type = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          is_int_data_type.initializer])

                pass_through_op_output = zero_out_module.qc_quantize(name='quant_op', in_tensor=inp,
                                                                     op_mode=mode_var,
                                                                     tensor_quantizer_reference=tensor_quant_ref,
                                                                     encoding_min=encoding_min,
                                                                     encoding_max=encoding_max,
                                                                     bit_width=bit_width,
                                                                     use_symmetric_encoding=use_symmetric_encoding,
                                                                     is_int_data_type=is_int_data_type)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.random.rand(10) * 256

        # get the output
        print(inp_data)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print(out_data)

        self.assertTrue(tensor_quantizer.isEncodingValid)
        encoding = tensor_quantizer.computeEncoding(bitwidth, use_symm_encoding)

        print('min=', encoding.min, ', max=', encoding.max)

        # compare qc_quantize op's output with input
        self.assertFalse(np.allclose(out_data, inp_data))

        sess.close()

    @pytest.mark.cuda
    def test_qc_quantize_op_gpu(self):
        """
        test custom op with GPU
        """
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = False
        with graph.as_default():

            inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')
            tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                       libpymo.RoundingMode.ROUND_NEAREST)
            tensor_quantizer_val = libpymo.PtrToInt64(tensor_quantizer)
            tensor_quant_ref = tf.Variable(initial_value=tensor_quantizer_val, trainable=False, dtype=tf.int64)

            mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.updateStats),
                                   trainable=False, dtype=tf.int32)

            encoding_min = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
            encoding_max = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
            bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
            use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)
            is_int_data_type = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)

            sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                      encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                      is_int_data_type.initializer])

            # place holder for the input
            with tf.device("/device:GPU:0"):

                pass_through_op_output = zero_out_module.qc_quantize(name='quant_op', in_tensor=inp,
                                                                     op_mode=mode_var,
                                                                     tensor_quantizer_reference=tensor_quant_ref,
                                                                     encoding_min=encoding_min,
                                                                     encoding_max=encoding_max,
                                                                     bit_width=bit_width,
                                                                     use_symmetric_encoding=use_symmetric_encoding,
                                                                     is_int_data_type=is_int_data_type)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.random.rand(10)

        # get the output

        print("inp_data", inp_data)
        with tf.device("/device:GPU:0"):
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print("out_data", out_data)

        # compare qc_quantize op's output with input
        self.assertTrue(np.allclose(out_data, inp_data))

        # compute encodings
        self.assertFalse(tensor_quantizer.isEncodingValid)
        encoding = tensor_quantizer.computeEncoding(bitwidth, use_symm_encoding)
        self.assertTrue(tensor_quantizer.isEncodingValid)
        print('min=', encoding.min, ', max=', encoding.max)

        # get the output
        inp_data = np.random.rand(10) * 2
        print("inp_data", inp_data)
        mode_var.load(int(libpymo.TensorQuantizerOpMode.quantizeDequantize), sess)
        with tf.device("/device:GPU:0"):
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print("out_data", out_data)

        # compare qc_quantize op's output with input
        self.assertFalse(np.allclose(out_data, inp_data))

        sess.close()

    @pytest.mark.cuda
    def test_qc_quantize_op_gpu_fp16(self):
        """
        test custom op with GPU
        """
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = False
        with graph.as_default():
            inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')
            tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                       libpymo.RoundingMode.ROUND_NEAREST)
            tensor_quantizer_val = libpymo.PtrToInt64(tensor_quantizer)
            tensor_quant_ref = tf.Variable(initial_value=tensor_quantizer_val, trainable=False, dtype=tf.int64)

            mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.quantizeDequantize),
                                   trainable=False, dtype=tf.int32)

            encoding_min = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
            encoding_max = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
            bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
            use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)
            is_int_data_type = tf.Variable(initial_value=False, trainable=False, dtype=tf.bool)

            sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                      encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                      is_int_data_type.initializer])

            # place holder for the input
            with tf.device("/device:GPU:0"):
                pass_through_op_output = zero_out_module.qc_quantize(name='quant_op', in_tensor=inp,
                                                                     op_mode=mode_var,
                                                                     tensor_quantizer_reference=tensor_quant_ref,
                                                                     encoding_min=encoding_min,
                                                                     encoding_max=encoding_max,
                                                                     bit_width=bit_width,
                                                                     use_symmetric_encoding=use_symmetric_encoding,
                                                                     is_int_data_type=is_int_data_type)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.array([0.78027299, 0.44164284, 0.6942797, 0.69774088, 0.55863863, 0.29553034, 0.219199,
                             0.09483732, 0.55075674, 0.6348504], dtype=np.float32)

        out_exp = np.array([0.78027344, 0.4416504, 0.69433594, 0.6977539, 0.55859375, 0.29541016, 0.21923828,
                            0.09484863, 0.55078125, 0.6347656], dtype=np.float32)

        print("inp_data", inp_data)
        with tf.device("/device:GPU:0"):
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print("out_data", out_data)
        self.assertTrue(np.allclose(out_data, out_exp))
        sess.close()

    def test_qc_quantize_static_op_cpu(self):
        """
        test custom static op with CPU
        """
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')

                pass_through_op_output = zero_out_module.qc_quantize_static(name='quant_op', in_tensor=inp,
                                                                            encoding_min=-1.0,
                                                                            encoding_max=1.0,
                                                                            bitwidth=8,
                                                                            quant_scheme=libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                                            op_mode=libpymo.TensorQuantizerOpMode.passThrough,
                                                                            is_symmetric=False)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.random.rand(10).astype(np.float32)

        # get the output
        print(inp_data)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print(out_data)

        # compare qc_quantize op's output with input
        self.assertTrue(np.allclose(out_data, inp_data, atol=1e-6))
        sess.close()

        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')

                pass_through_op_output = zero_out_module.qc_quantize_static(name='quant_op', in_tensor=inp,
                                                                            encoding_min=-1.0,
                                                                            encoding_max=0.5,
                                                                            bitwidth=8,
                                                                            quant_scheme=libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                                            op_mode=libpymo.TensorQuantizerOpMode.quantizeDequantize,
                                                                            is_symmetric=False)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.random.rand(10).astype(np.float32)

        # get the output
        print(inp_data)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print(out_data)

        # compare qc_quantize op's output with input
        self.assertFalse(np.allclose(out_data, inp_data, atol=1e-1))
        sess.close()

        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')

                pass_through_op_output = zero_out_module.qc_quantize_static(name='quant_op', in_tensor=inp,
                                                                            encoding_min=-1.0,
                                                                            encoding_max=1.0,
                                                                            bitwidth=8,
                                                                            quant_scheme=libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                                            op_mode=libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize,
                                                                            is_symmetric=False)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.random.rand(10).astype(np.float32)

        # get the output
        print(inp_data)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print(out_data)

        # compare qc_quantize op's output with input
        self.assertFalse(np.allclose(out_data, inp_data, atol=1e-3))

        sess.close()

    @pytest.mark.cuda
    def test_qc_quantize_static_op_gpu(self):
        """
        test custom static op with GPU
        """
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:GPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')

                pass_through_op_output = zero_out_module.qc_quantize_static(name='quant_op', in_tensor=inp,
                                                                            encoding_min=-1.0,
                                                                            encoding_max=1.0,
                                                                            bitwidth=8,
                                                                            quant_scheme=libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                                            op_mode=libpymo.TensorQuantizerOpMode.passThrough,
                                                                            is_symmetric=False)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.random.rand(10).astype(np.float32)

        # get the output
        print(inp_data)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print(out_data)

        # compare qc_quantize op's output with input
        self.assertTrue(np.allclose(out_data, inp_data, atol=1e-6))
        sess.close()

        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:GPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')

                pass_through_op_output = zero_out_module.qc_quantize_static(name='quant_op', in_tensor=inp,
                                                                            encoding_min=-1.0,
                                                                            encoding_max=0.5,
                                                                            bitwidth=8,
                                                                            quant_scheme=libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                                            op_mode=libpymo.TensorQuantizerOpMode.quantizeDequantize,
                                                                            is_symmetric=False)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.random.rand(10).astype(np.float32)

        # get the output
        print(inp_data)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print(out_data)

        # compare qc_quantize op's output with input
        self.assertFalse(np.allclose(out_data, inp_data, atol=1e-1))
        sess.close()

        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:GPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')

                pass_through_op_output = zero_out_module.qc_quantize_static(name='quant_op', in_tensor=inp,
                                                                            encoding_min=-1.0,
                                                                            encoding_max=1.0,
                                                                            bitwidth=8,
                                                                            quant_scheme=libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                                            op_mode=libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize,
                                                                            is_symmetric=False)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.random.rand(10).astype(np.float32)

        # get the output
        print(inp_data)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print(out_data)

        # compare qc_quantize op's output with input
        self.assertFalse(np.allclose(out_data, inp_data, atol=1e-3))

        sess.close()


    def test_qc_quantize_op_straight_through_gradient_computation(self):
        """
        test to validate tensorflow quantize op straight through estimator gradient computation
        """

        from aimet_tensorflow import quantsim_straight_through_grad

        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        with graph.as_default():
            inp = tf.compat.v1.placeholder(tf.float32, shape=[2, 2], name='input')
            tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                       libpymo.RoundingMode.ROUND_NEAREST)
            tensor_quantizer_val = libpymo.PtrToInt64(tensor_quantizer)
            tensor_quant_ref = tf.Variable(initial_value=tensor_quantizer_val, trainable=False, dtype=tf.int64)

            mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                                   trainable=False, dtype=tf.int32)

            # fix min max and bitwidth to be used
            encoding_min = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
            encoding_max = tf.Variable(initial_value=5.0, trainable=True, dtype=tf.double)
            bit_width = tf.Variable(initial_value=8, trainable=False, dtype=tf.int8)
            use_symmetric_encoding = tf.Variable(initial_value=False, trainable=False, dtype=tf.bool)
            is_int_data_type = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)

            sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                      encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                      is_int_data_type.initializer])

            # use default gradient
            pass_through_op_output = zero_out_module.qc_quantize(name='quant_op', in_tensor=inp,
                                                                 op_mode=mode_var,
                                                                 tensor_quantizer_reference=tensor_quant_ref,
                                                                 encoding_min=encoding_min,
                                                                 encoding_max=encoding_max,
                                                                 bit_width=bit_width,
                                                                 use_symmetric_encoding=use_symmetric_encoding,
                                                                 is_int_data_type=is_int_data_type)

            # pass_through_op = graph.get_operation_by_name('quant_op')

        inp_tensor = sess.graph.get_tensor_by_name('input:0')

        # set the encodings
        tensor_quantizer.isEncodingValid = True
        mode_var.load(int(libpymo.TensorQuantizerOpMode.quantizeDequantize), sess)

        # compute default gradient
        grads = tf.gradients(pass_through_op_output, [inp_tensor])
        dlossbydx = grads

        # send input, note the last value sent here is > 5.0 ,
        # we set encodings earlier to be min = 0.0 , max = 5.0
        # input has data > p
        inp_data = [[1.4581, 0.4829], [0.3125, 5.6150]]
        # check the gradient returned is a gated version, in this case should be [[1.0, 1.0],[1.0, 0.0]]
        with graph.as_default():
            input_gradient = sess.run([dlossbydx], feed_dict={inp_tensor: inp_data})[0]

        # validate valid clamping in gradient computation
        self.assertTrue(input_gradient[0][0][0] == 1.0)
        self.assertTrue(input_gradient[0][0][1] == 1.0)
        self.assertTrue(input_gradient[0][1][0] == 1.0)
        self.assertTrue(input_gradient[0][1][1] == 0.0)

        # pass input in correct range
        inp_data = [[1.4581, 0.4829], [0.3125, 1.6150]]
        # check the gradient returned is a gated version, in this case should be [[1.0, 1.0],[1.0, 0.0]]
        with graph.as_default():
            input_gradient = sess.run([dlossbydx], feed_dict={inp_tensor: inp_data})[0]

        # validate no clamping case in gradient computation
        self.assertTrue(input_gradient[0][0][0] == 1.0)
        self.assertTrue(input_gradient[0][0][1] == 1.0)
        self.assertTrue(input_gradient[0][1][0] == 1.0)
        self.assertTrue(input_gradient[0][1][1] == 1.0)

        # pass input with data < n , first value here is -0.5
        inp_data = [[-0.5, 0.4829], [0.3125, 1.6150]]
        # check the gradient returned is a gated version, in this case should be [[1.0, 1.0],[1.0, 0.0]]
        with graph.as_default():
            input_gradient = sess.run([dlossbydx], feed_dict={inp_tensor: inp_data})[0]

        # validate valid clamping case in gradient computation
        self.assertTrue(input_gradient[0][0][0] == 0.0)
        self.assertTrue(input_gradient[0][0][1] == 1.0)
        self.assertTrue(input_gradient[0][1][0] == 1.0)
        self.assertTrue(input_gradient[0][1][1] == 1.0)

    def test_qc_quantize_recurrent_param_op(self):
        """
        test custom recurrent param quantize op with CPU
        """
        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = True

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[10], name='input')
                tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF,
                                                           libpymo.RoundingMode.ROUND_NEAREST)
                tensor_quantizer_val = libpymo.PtrToInt64(tensor_quantizer)
                tensor_quant_ref = tf.Variable(initial_value=tensor_quantizer_val, trainable=False, dtype=tf.int64)

                time_step_tensor = tf.constant(1, dtype=tf.int32)

                encoding_min = tf.Variable(initial_value=-0.5, trainable=True, dtype=tf.double)
                encoding_max = tf.Variable(initial_value=0.5, trainable=True, dtype=tf.double)
                bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
                use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)

                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                                       trainable=False, dtype=tf.int32)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer])

                pass_through_op_output = zero_out_module.qc_quantize_recurrent_param(name='quant_op', in_tensor=inp,
                                                                                     op_mode=mode_var,
                                                                                     tensor_quantizer_reference=tensor_quant_ref,
                                                                                     encoding_min=encoding_min,
                                                                                     encoding_max=encoding_max,
                                                                                     bit_width=bit_width,
                                                                                     use_symmetric_encoding=use_symmetric_encoding,
                                                                                     time_steps=time_step_tensor)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        # inp_data = np.random.rand(10).astype(np.float32)
        np.random.seed(18)
        inp_data = np.random.randint(low=-1, high=2, size=10).astype(np.float32)

        # get the output
        print(inp_data)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print(out_data)

        # compare qc_quantize op's output with input
        # encodings being set to -0.5 and 0.5 should not have a bearing on this quantized output
        # we should not observe truncation if op's encoding min/max input values are used instead of cached values
        self.assertTrue(np.allclose(out_data, inp_data, atol=1e-6))
        sess.close()

    def test_qc_quantize_op_gradient_computation(self):
        """
        test to validate tensorflow custom gradient computation
        against golden test data (in this case : an equivalent Pytorch test with auto grad)
        """

        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        sess = tf.compat.v1.Session(graph=graph, config=config)

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.compat.v1.placeholder(tf.float32, shape=[2, 2], name='input')
                tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                           libpymo.RoundingMode.ROUND_NEAREST)
                tensor_quantizer_val = libpymo.PtrToInt64(tensor_quantizer)
                tensor_quant_ref = tf.Variable(initial_value=tensor_quantizer_val, trainable=False, dtype=tf.int64)

                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                                       trainable=False, dtype=tf.int32)

                # fix min max and bitwidth to be used
                encoding_min = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
                encoding_max = tf.Variable(initial_value=5.0, trainable=True, dtype=tf.double)
                bit_width = tf.Variable(initial_value=8, trainable=False, dtype=tf.int8)
                use_symmetric_encoding = tf.Variable(initial_value=False, trainable=False, dtype=tf.bool)
                is_int_data_type = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                          is_int_data_type.initializer])

                with graph.gradient_override_map(
                        {"QcQuantize": "QcQuantizeRangeLearningCustomGradient"}):

                    pass_through_op_output = zero_out_module.qc_quantize(name='quant_op', in_tensor=inp,
                                                                         op_mode=mode_var,
                                                                         tensor_quantizer_reference=tensor_quant_ref,
                                                                         encoding_min=encoding_min,
                                                                         encoding_max=encoding_max,
                                                                         bit_width=bit_width,
                                                                         use_symmetric_encoding=use_symmetric_encoding,
                                                                         is_int_data_type=is_int_data_type)

                pass_through_op = graph.get_operation_by_name('quant_op')

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        # fixed input data used
        inp_data = [[0.4581, 0.4829], [0.3125, 0.6150]]

        # get the output data @todo match these
        tensor_quantizer.isEncodingValid = True
        mode_var.load(int(libpymo.TensorQuantizerOpMode.quantizeDequantize), sess)

        # for matching with golden output, truncate to 4
        tf_output_data = np.around(sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data}), 4)
        exp_output = [[0.4510, 0.4902], [0.3137, 0.6078]]

        # dummy loss function to match with Pytorch
        def custom_loss(y_actual, y_pred):
            return tf.reduce_sum(tf.subtract(y_pred, y_actual-y_actual))

        with graph.as_default():
            var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            labels_placeholder = tf.compat.v1.placeholder(tf.float32, [2, 2], name='labels')

            # output tensor
            logits = sess.graph.get_tensor_by_name('quant_op:0')

            # dummy loss function is set to sum(output)
            current_loss = custom_loss(labels_placeholder, logits)
            labels = np.ones((2), dtype=int)  # np.random.randint(2, size=batches)
            one_hot_labels = np.eye(2)[labels]

            update_ops = []
            global_step = tf.compat.v1.train.create_global_step()
            # Stochastic GD in tf with momentum param
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=0.05, momentum=0.5)
            gradients = optimizer.compute_gradients(current_loss, var_list)

            grad_updates = optimizer.apply_gradients(gradients, global_step=global_step)
            init_global = tf.compat.v1.global_variables_initializer()
            init_local = tf.compat.v1.local_variables_initializer()
            init = tf.group(init_global, init_local)
            sess.run(init)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)

            with tf.control_dependencies([update_op]):
                train_op = tf.identity(current_loss, name='train_op')

            # enable this to check current loss value used
            _ = sess.run(current_loss, feed_dict={inp_tensor: inp_data, labels_placeholder: one_hot_labels})
            # start training
            _ = sess.run(train_op, feed_dict={inp_tensor: inp_data, labels_placeholder: one_hot_labels})
            tf_enc_min_after_train = sess.run(pass_through_op.inputs[QuantizeOpIndices.encoding_min])
            tf_enc_max_after_train = sess.run(pass_through_op.inputs[QuantizeOpIndices.encoding_max])

            # match outputs
            self.assertTrue(np.allclose(exp_output, tf_output_data))

            # compare min and max after update with expected values (Pytorch values)
            expected_enc_min_after_train = -5.7160621508955956e-05
            expected_enc_max_after_train = 5.000057220458984
            self.assertAlmostEqual(tf_enc_min_after_train, expected_enc_min_after_train, 6)
            self.assertAlmostEqual(tf_enc_max_after_train, expected_enc_max_after_train, 6)

        sess.close()
