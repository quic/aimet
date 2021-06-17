# /usr/bin/env python3.6
# -*- mode: python -*-
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

""" Test Adaround wrapper """

import os
import pytest
import unittest.mock
import numpy as np
import tensorflow as tf
import libpymo

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_tensorflow.adaround.adaround_wrapper import AdaroundWrapper
from aimet_tensorflow.examples.test_models import depthwise_conv2d_model

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestAdaroundWrapper(unittest.TestCase):
    """ Test Adaround wrapper """

    def _initialize_alpha(self, device):
        """ test initialize alpha """
        tf.compat.v1.reset_default_graph()
        np.random.seed(0)
        weight = np.random.rand(32, 3, 12, 12)
        encoding = libpymo.TfEncoding()
        encoding.bw = 4
        encoding.offset = -127.0
        encoding.delta = 0.001551126479

        with tf.device(device):
            weight_tensor = tf.convert_to_tensor(weight, dtype=tf.float64)
            alpha = AdaroundWrapper._initialize_alpha(weight_tensor, encoding)

        session = tf.compat.v1.Session()
        session.run(tf.compat.v1.global_variables_initializer())

        self.assertAlmostEqual(float(alpha.eval(session=session)[0, 0, :1, :1]), 1.1715, places=4)
        session.close()

    @pytest.mark.cuda
    def test_initialize_alpha_gpu(self):
        """ test initialize alpha for GPU """
        device = '/gpu:0'
        self._initialize_alpha(device)

    def test_initialize_alpha(self):
        """ test initialize alpha for CPU """
        device = '/cpu:0'
        self._initialize_alpha(device)

    def _adaround_weights(self, device):
        """ test adaround weights """
        tf.compat.v1.reset_default_graph()
        with tf.device(device):

            _ = depthwise_conv2d_model()
            init = tf.compat.v1.global_variables_initializer()

            session = tf.compat.v1.Session()
            session.run(init)

            # 1) Conv2D
            conv = tf.compat.v1.get_default_graph().get_operation_by_name('conv2d/Conv2D')
            conv_wrapper = AdaroundWrapper(session, conv, 4, False, QuantScheme.post_training_tf)

            # 2) MatMul
            matmul = tf.compat.v1.get_default_graph().get_operation_by_name('depthwise_conv2d_model/MatMul')
            matmul_wrapper = AdaroundWrapper(session, matmul, 4, False, QuantScheme.post_training_tf)

            # 3) Depthwise Conv2D
            depthwise_conv = tf.compat.v1.get_default_graph().get_operation_by_name('depthwise_conv2d/depthwise')
            depthwise_conv_wrapper = AdaroundWrapper(session, depthwise_conv, 4, False, QuantScheme.post_training_tf)

            # Initialize alpha variable
            session.run(conv_wrapper.alpha.initializer)
            session.run(matmul_wrapper.alpha.initializer)
            session.run(depthwise_conv_wrapper.alpha.initializer)

            quantized_weight = session.run(matmul_wrapper.adaround_weights())
            orig_weight = session.run(matmul_wrapper._weight_tensor)
            self.assertEqual(orig_weight.shape, quantized_weight.shape)
            self.assertTrue(np.allclose(orig_weight, quantized_weight, atol=2 * matmul_wrapper.encoding.delta))

            quantized_weight = session.run(conv_wrapper.adaround_weights())
            orig_weight = session.run(conv_wrapper._weight_tensor)
            self.assertEqual(orig_weight.shape, quantized_weight.shape)
            self.assertTrue(np.allclose(orig_weight, quantized_weight, atol=2 * conv_wrapper.encoding.delta))

            quantized_weight = session.run(depthwise_conv_wrapper.adaround_weights())
            orig_weight = session.run(depthwise_conv_wrapper._weight_tensor)
            self.assertEqual(orig_weight.shape, quantized_weight.shape)
            self.assertTrue(np.allclose(orig_weight, quantized_weight, atol=2 * depthwise_conv_wrapper.encoding.delta))

            session.close()

    @pytest.mark.cuda
    def test_adaround_weights_gpu(self):
        """ test adaround weights for GPU """
        device = '/gpu:0'
        self._adaround_weights(device)

    def test_adaround_weights(self):
        """ adaround weights for CPU """
        device = '/cpu:0'
        self._adaround_weights(device)

    def test_adaround_weights_hard_rounding(self):
        """ test adaround weight with hard rounding """
        tf.compat.v1.reset_default_graph()
        _ = depthwise_conv2d_model()
        init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session()
        session.run(init)

        conv = tf.compat.v1.get_default_graph().get_operation_by_name('conv2d/Conv2D')
        conv_wrapper = AdaroundWrapper(session, conv, 4, False, QuantScheme.post_training_tf)

        # Initialize alpha variable
        session.run(conv_wrapper.alpha.initializer)

        soft_rounded_weight = session.run(conv_wrapper.adaround_weights())
        hard_rounded_weight = session.run(conv_wrapper.adaround_weights(),
                                          feed_dict={conv_wrapper.use_soft_rounding: False})

        orig_weight = session.run(conv_wrapper._weight_tensor)
        self.assertEqual(orig_weight.shape, soft_rounded_weight.shape)
        self.assertEqual(orig_weight.shape, hard_rounded_weight.shape)

        self.assertTrue(np.allclose(orig_weight, soft_rounded_weight, atol=2 * conv_wrapper.encoding.delta))
        self.assertTrue(np.allclose(orig_weight, hard_rounded_weight, atol=2 * conv_wrapper.encoding.delta))
        session.close()
