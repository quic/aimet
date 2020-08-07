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

import unittest
import numpy as np
import tensorflow as tf

import libpymo


class TestTrainingExtensionsQcQuantizeOp(unittest.TestCase):

    def test_qc_quantize_op_cpu(self):
        """
        test custom op with CPU
        """
        zero_out_module = tf.load_op_library('libaimet_tf_ops.so')
        graph = tf.Graph()
        config = tf.ConfigProto(log_device_placement=False)
        sess = tf.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = True

        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.placeholder(tf.float32, shape=[10], name='input')
                tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                           libpymo.RoundingMode.ROUND_NEAREST)
                tensor_quantizer_val = libpymo.PtrToInt64(tensor_quantizer)
                tensor_quant_ref = tf.Variable(initial_value=tensor_quantizer_val, trainable=False, dtype=tf.int64)

                encoding_min = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
                encoding_max = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.double)
                bit_width = tf.Variable(initial_value=bitwidth, trainable=False, dtype=tf.int8)
                use_symmetric_encoding = tf.Variable(initial_value=use_symm_encoding, trainable=False, dtype=tf.bool)

                mode_var = tf.Variable(initial_value=int(libpymo.TensorQuantizerOpMode.updateStats),
                                       trainable=False, dtype=tf.int32)

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                         encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer])

                pass_through_op_output = zero_out_module.qc_quantize(name='quant_op', in_tensor=inp,
                                                                     op_mode=mode_var,
                                                                     tensor_quantizer_reference=tensor_quant_ref,
                                                                     encoding_min=encoding_min,
                                                                     encoding_max=encoding_max,
                                                                     bit_width=bit_width,
                                                                     use_symmetric_encoding=use_symmetric_encoding)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.random.rand(10)

        # get the output
        print(inp_data)
        out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print(out_data)

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

    def test_qc_quantize_op_oneshot_cpu(self):
        """
        test custom op with CPU
        """
        zero_out_module = tf.load_op_library('libaimet_tf_ops.so')
        graph = tf.Graph()
        config = tf.ConfigProto(log_device_placement=False)
        sess = tf.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = False
        with graph.as_default():
            # place holder for the input
            with tf.device("/device:CPU:0"):
                inp = tf.placeholder(tf.float32, shape=[10], name='input')
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

                sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                          encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer])

                pass_through_op_output = zero_out_module.qc_quantize(name='quant_op', in_tensor=inp,
                                                                     op_mode=mode_var,
                                                                     tensor_quantizer_reference=tensor_quant_ref,
                                                                     encoding_min=encoding_min,
                                                                     encoding_max=encoding_max,
                                                                     bit_width=bit_width,
                                                                     use_symmetric_encoding=use_symmetric_encoding)

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

    def test_qc_quantize_op_gpu(self):
        """
        test custom op with GPU
        """
        zero_out_module = tf.load_op_library('libaimet_tf_ops.so')
        graph = tf.Graph()
        config = tf.ConfigProto(log_device_placement=False)
        sess = tf.Session(graph=graph, config=config)
        bitwidth = 8
        use_symm_encoding = False
        with graph.as_default():

            inp = tf.placeholder(tf.float32, shape=[10], name='input')
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

            sess.run([mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                      encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer])

            # place holder for the input
            with tf.device("/device:GPU:0"):

                pass_through_op_output = zero_out_module.qc_quantize(name='quant_op', in_tensor=inp,
                                                                     op_mode=mode_var,
                                                                     tensor_quantizer_reference=tensor_quant_ref,
                                                                     encoding_min=encoding_min,
                                                                     encoding_max=encoding_max,
                                                                     bit_width=bit_width,
                                                                     use_symmetric_encoding=use_symmetric_encoding)

        inp_tensor = sess.graph.get_tensor_by_name('input:0')
        inp_data = np.random.rand(10)

        # get the output
        print(inp_data)
        with tf.device("/device:GPU:0"):
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print(out_data)

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
        with tf.device("/device:GPU:0"):
            out_data = sess.run(pass_through_op_output, feed_dict={inp_tensor: inp_data})
        print(out_data)

        # compare qc_quantize op's output with input
        self.assertFalse(np.allclose(out_data, inp_data))

        sess.close()

    def test_qc_quantize_static_op_cpu(self):
        """
        test custom static op with CPU
        """
        zero_out_module = tf.load_op_library('libaimet_tf_ops.so')
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

    def test_qc_quantize_static_op_gpu(self):
        """
        test custom static op with GPU
        """
        zero_out_module = tf.load_op_library('libaimet_tf_ops.so')
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
