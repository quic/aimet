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
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme
from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.quantizer_info import QuantizeOpIndices

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()

BUCKET_SIZE = 512


class TestQuantizerInfo(unittest.TestCase):
    """
    Quantizer Info unit tests
    """
    def test_set_and_freeze_encoding(self):
        """ Create QuantSim for a CPU model, test set and freeze encoding """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            _ = keras_model()
            init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session()
        session.run(init)

        sim = QuantizationSimModel(session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
        quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')

        encoding = quantizer.compute_encoding(8, False)
        print(encoding.max, encoding.min)
        # Set and freeze encoding
        quantizer.set_encoding(encoding)
        quantizer.freeze_encoding()

        old_encoding_min = quantizer.get_variable_from_op(QuantizeOpIndices.encoding_min)
        old_encoding_max = quantizer.get_variable_from_op(QuantizeOpIndices.encoding_max)

        self.assertEqual(encoding.min, old_encoding_min)
        self.assertEqual(encoding.max, old_encoding_max)
        self.assertEqual(quantizer.is_encoding_valid(), True)

        # Try updating encoding min and max with new values, but values can not be changed
        encoding.min = -0.4
        encoding.max = 0.6
        quantizer.set_encoding(encoding)

        self.assertEqual(old_encoding_min, quantizer.get_variable_from_op(QuantizeOpIndices.encoding_min))
        self.assertEqual(old_encoding_max, quantizer.get_variable_from_op(QuantizeOpIndices.encoding_max))

        session.close()

    def test_set_and_freeze_op_mode(self):
        """ Create QuantSim for a CPU model, test set and freeze op mode """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            _ = keras_model()
            init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session()
        session.run(init)

        sim = QuantizationSimModel(session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
        quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')

        op_mode = int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize)
        quantizer.set_op_mode(op_mode)
        quantizer.freeze_encoding()
        self.assertEqual(op_mode, quantizer.get_op_mode())

        new_op_mode = int(libpymo.TensorQuantizerOpMode.passThrough)
        quantizer.set_op_mode(new_op_mode)
        self.assertNotEqual(new_op_mode, quantizer.get_op_mode())
        self.assertEqual(op_mode, quantizer.get_op_mode())

        session.close()

    def test_compute_encoding(self):
        """ Create QuantSim for a CPU model, test compute encoding """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            _ = keras_model()
            init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session()
        session.run(init)

        sim = QuantizationSimModel(session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
        quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')

        # Freeze encoding before computing it
        quantizer.freeze_encoding()
        self.assertRaises(AssertionError, lambda: quantizer.compute_encoding(8, False))

        session.close()

    def test_get_encoding(self):
        """ Create QuantSim for a CPU model, test get encoding """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            _ = keras_model()
            init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session()
        session.run(init)

        sim = QuantizationSimModel(session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
        quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')

        self.assertRaises(AssertionError, lambda: quantizer.get_encoding())

        session.close()

    def test_get_stats_histogram(self):
        """ test get_stats_histogram() for per tensor """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            _ = keras_model()
            init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session()
        session.run(init)

        sim = QuantizationSimModel(session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
        quantizer_info = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')
        weight = quantizer_info.get_variable_from_op(0)

        # compute encoding.
        quantizer_info.tensor_quantizer.updateStats(weight, False)
        quantizer_info.compute_encoding(8, False)
        assert quantizer_info.is_encoding_valid()

        histograms = quantizer_info.get_stats_histogram()
        assert len(histograms) == 1
        for histogram in histograms:
            assert len(histogram) == BUCKET_SIZE

        session.close()

    def test_get_stats_histogram_with_invalid_quant_scheme(self):
        """
        test get_stats_histogram() with invalid inputs.
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            _ = keras_model()
            init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session()
        session.run(init)

        sim = QuantizationSimModel(session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False,
                                   quant_scheme=QuantScheme.post_training_tf)
        quantizer_info = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')

        with pytest.raises(RuntimeError):
            quantizer_info.get_stats_histogram()

        session.close()

    def test_get_stats_histogram_with_invalid_combination(self):
        """
        test get_stats_histogram() without computing encodings.
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            _ = keras_model()
            init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session()
        session.run(init)

        sim = QuantizationSimModel(session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
        quantizer_info = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')

        with pytest.raises(RuntimeError):
            quantizer_info.get_stats_histogram()

        session.close()
