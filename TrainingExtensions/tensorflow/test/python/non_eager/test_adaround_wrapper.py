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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pytest
import unittest.mock
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2DTranspose, DepthwiseConv2D

import aimet_common.libpymo as libpymo
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_tensorflow.adaround.adaround_wrapper import AdaroundWrapper
from aimet_tensorflow.examples.test_models import depthwise_conv2d_model

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
tf.compat.v1.disable_eager_execution()


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
            alpha = AdaroundWrapper._initialize_alpha(weight_tensor, encoding, enable_per_channel=False,
                                                      ch_axis=len(list(weight_tensor.shape)) - 1)

        session = tf.compat.v1.Session()
        session.run(tf.compat.v1.global_variables_initializer())

        self.assertAlmostEqual(float(alpha.eval(session=session)[0, 0, :1, :1]), 1.1715, places=4)
        session.close()

    def _initialize_alpha_per_channel(self, device):
        """ test initialize alpha for per-channel"""
        tf.compat.v1.reset_default_graph()
        np.random.seed(0)
        weight = np.random.rand(8, 3, 12, 16)
        encoding_per_ch = libpymo.TfEncoding()
        encoding_per_ch.bw = 4
        encoding_per_ch.offset = -127.0
        encoding_per_ch.delta = 0.001551126479

        encoding = [encoding_per_ch for _ in range(8)]
        with tf.device(device):
            weight_tensor = tf.convert_to_tensor(weight, dtype=tf.float64)
            alpha = AdaroundWrapper._initialize_alpha(weight_tensor, encoding, enable_per_channel=True,
                                                      ch_axis=0)

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

    def test_initialize_alpha_per_channel(self):
        """ test initialize alpha for CPU """
        device = '/cpu:0'
        self._initialize_alpha_per_channel(device)

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
            conv_wrapper = AdaroundWrapper(session, conv, 4, QuantScheme.post_training_tf, False, False, True, False,
                                           output_height=None, output_width=None, output_channels=None)

            # 2) MatMul
            matmul = tf.compat.v1.get_default_graph().get_operation_by_name('depthwise_conv2d_model/MatMul')
            matmul_wrapper = AdaroundWrapper(session, matmul, 4, QuantScheme.post_training_tf, False, False, True,
                                             False, output_height=None, output_width=None, output_channels=None)

            # 3) Depthwise Conv2D
            depthwise_conv = tf.compat.v1.get_default_graph().get_operation_by_name('depthwise_conv2d/depthwise')
            depthwise_conv_wrapper = AdaroundWrapper(session, depthwise_conv, 4, QuantScheme.post_training_tf,
                                                     False, False, True, False, output_height=None, output_width=None,
                                                     output_channels=None)

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
        conv_wrapper = AdaroundWrapper(session, conv, 4, QuantScheme.post_training_tf, False, False, True, False,
                                       output_height=None, output_width=None, output_channels=None)

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

    def _test_broadcast_to_tensor(self, device, ch_axis: int, shape: tuple):
        """
        test the function _broadcast_to_tensor for different tensor shapes and channel axes

        - main flow does the following:
        weight: (A, B, C, D)
        ch_axis: 2
        encodings: (C,)
        delta/offset: slice of encodings which has other params as well (C,)
        after broadcast of encoding: (A, B, D, C) -> _get_broadcast_shape
        after transpose of encoding: (A, B, C, D)

        - This test tests broadcast from (C,) to (A, B, C, D)
        """
        session = tf.compat.v1.Session()
        np.random.seed(0)
        weight = np.random.rand(*shape)
        # encoding is of length shape[ch_axis]
        encoding = [np.random.rand() for _ in range(list(shape)[ch_axis])]

        with tf.device(device):
            weight_tensor = tf.convert_to_tensor(weight, dtype=tf.float64)
            broadcasted_delta = AdaroundWrapper._broadcast_to_tensor(weight_tensor, encoding, ch_axis=ch_axis)

        t = session.run(broadcasted_delta)

        # verify t is of same shape as shape
        self.assertEqual(t.shape, shape)

        # verify t has same encodings across all the axis
        res = np.all(t, axis=ch_axis)
        self.assertEqual(res.all(), True)
        session.close()

    def test_broadcast_to_tensor_cpu(self):
        """
        test _get_broadcast_shape for different combinations of ch_axis and weight shapes on CPU
        """
        device = '/cpu:0'
        for ch_axis in range(4):
            self._test_broadcast_to_tensor(device, ch_axis=ch_axis, shape=(2, 3, 4, 5))

        for ch_axis in range(3):
            self._test_broadcast_to_tensor(device, ch_axis=ch_axis, shape=(5, 6, 7))

    @pytest.mark.cuda
    def test_broadcast_to_tensor_gpu(self):
        """
        test _get_broadcast_shape for different combinations of ch_axis and weight shapes on GPU
        """
        device = '/gpu:0'
        for ch_axis in range(4):
            self._test_broadcast_to_tensor(device, ch_axis=ch_axis, shape=(2, 3, 4, 5))

        for ch_axis in range(3):
            self._test_broadcast_to_tensor(device, ch_axis=ch_axis, shape=(5, 6, 7))


    def test_generate_weight_transpose_perm(self):
        """
        test the function _generate_weight_transpose_perm
        """
        res = AdaroundWrapper._generate_weight_transpose_perm(shape=(2, 3, 4, 5), ch_axis=0)
        self.assertEqual(res, [0, 1, 2, 3])

        res = AdaroundWrapper._generate_weight_transpose_perm(shape=(2, 3, 4, 5), ch_axis=1)
        self.assertEqual(res, [1, 0, 2, 3])

        res = AdaroundWrapper._generate_weight_transpose_perm(shape=(2, 3, 4, 5), ch_axis=2)
        self.assertEqual(res, [2, 0, 1, 3])

        res = AdaroundWrapper._generate_weight_transpose_perm(shape=(2, 3, 4, 5), ch_axis=3)
        self.assertEqual(res, [3, 0, 1, 2])

    def test_transform_input_ndarray_for_depthwise_conv_2d(self):
        """
        test the function _transform_input_ndarray_for_depthwise_conv_2d
        """
        shape = (2, 2, 3, 4)
        a = np.random.rand(*shape)
        b = AdaroundWrapper.transform_input_ndarray_for_depthwise_conv_2d(a)
        assert b.shape == (2, 2, 12)
        assert (a - b.reshape(*shape)).all() == 0

    def test_conv_transpose_adaround_wrapper(self):
        """
        Test wrapper generation for conv transpose
        """
        tf.compat.v1.reset_default_graph()

        with tf.device('/cpu:0'):
            graph = tf.Graph()
            with graph.as_default():
                tf.compat.v1.set_random_seed(1)
                _ = Sequential([Conv2DTranspose(8, (2, 2), input_shape=(16, 16, 3,))])
                init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session(graph=graph)
        session.run(init)

        conv2d_transpose_op = session.graph.get_operation_by_name('conv2d_transpose/conv2d_transpose')

        graph = tf.Graph()
        with graph.as_default():
            wrapper = AdaroundWrapper(session, conv2d_transpose_op, param_bw=8,
                                      quant_scheme=QuantScheme.post_training_tf, is_symmetric=True,
                                      strict_symmetric=False, unsigned_symmetric=False, enable_per_channel=True,
                                      output_height=None, output_width=None, output_channels=None)
        assert len(wrapper.encoding) == 8
        assert wrapper.ch_axis == 2


    def test_depthwise_conv_adaround_wrapper(self):
        """
        Test wrapper generation for conv transpose
        """
        tf.compat.v1.reset_default_graph()

        with tf.device('/cpu:0'):
            graph = tf.Graph()
            with graph.as_default():
                tf.compat.v1.set_random_seed(1)
                _ = Sequential([DepthwiseConv2D(8, (2, 2), input_shape=(16, 16, 4,))])
                init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session(graph=graph)
        session.run(init)

        depthwise_conv2d_op = session.graph.get_operation_by_name('depthwise_conv2d/depthwise')

        graph = tf.Graph()
        with graph.as_default():
            wrapper = AdaroundWrapper(session, depthwise_conv2d_op, param_bw=8,
                                      quant_scheme=QuantScheme.post_training_tf, is_symmetric=True,
                                      strict_symmetric=False, unsigned_symmetric=False, enable_per_channel=True,
                                      output_height=None, output_width=None, output_channels=None)
        assert len(wrapper.encoding) == 4
        assert wrapper.ch_axis == 2
