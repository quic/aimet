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

from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.quantizer_info import QuantizeOpIndices
import libpymo

tf.compat.v1.logging.set_verbosity(tf.logging.WARN)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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

        quantizer.set_and_freeze_encoding(encoding)
        encoding_max = quantizer.get_variable_from_op(QuantizeOpIndices.encoding_max)
        encoding_min = quantizer.get_variable_from_op(QuantizeOpIndices.encoding_min)

        self.assertEqual(encoding_min, encoding.min)
        self.assertEqual(encoding_max, encoding.max)

        # Update encoding min and max value
        encoding.max = 0.3
        encoding.min = -0.2
        quantizer.set_encoding(encoding)

        new_encoding_max = quantizer.get_variable_from_op(QuantizeOpIndices.encoding_max)
        new_encoding_min = quantizer.get_variable_from_op(QuantizeOpIndices.encoding_min)
        self.assertEqual(encoding_min, new_encoding_min)
        self.assertEqual(encoding_max, new_encoding_max)

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
        quantizer.set_and_freeze_op_mode(op_mode)
        self.assertEqual(op_mode, quantizer.get_op_mode())

        new_op_mode = int(libpymo.TensorQuantizerOpMode.passThrough)
        quantizer.set_op_mode(new_op_mode)
        self.assertNotEqual(new_op_mode, quantizer.get_op_mode())
        self.assertEqual(op_mode, quantizer.get_op_mode())

        session.close()
