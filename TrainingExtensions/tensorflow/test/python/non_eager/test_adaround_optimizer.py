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

""" Test AdaroundOptimizer """

import pytest
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import unittest.mock
import numpy as np
import tensorflow as tf

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_tensorflow.examples.test_models import depthwise_conv2d_model
from aimet_tensorflow.adaround.adaround_optimizer import AdaroundOptimizer
from aimet_tensorflow.adaround.adaround_wrapper import AdaroundWrapper
from aimet_tensorflow.adaround.adaround_loss import AdaroundLoss
from aimet_tensorflow.adaround.adaround_loss import AdaroundHyperParameters

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
tf.compat.v1.disable_eager_execution()


class TestAdaroundOptimizer(unittest.TestCase):
    """ Test AdaroundOptimizer """

    def test_optimize_rounding_conv2d(self):
        """ Test optimize rounding for Conv2d """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)

        # Create model in default graph
        _ = depthwise_conv2d_model()
        init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session()
        session.run(init)
        conv = tf.compat.v1.get_default_graph().get_operation_by_name('conv2d/Conv2D')
        orig_weight = session.run(conv.inputs[1])

        inp_data = np.random.rand(1, 5, 5, 3)
        out_data = np.random.rand(1, 5, 5, 16)
        opt_params = AdaroundHyperParameters(num_iterations=1, reg_param=0.01, beta_range=(20, 2), warm_start=0.2)

        # Adaround optimization in separate graph
        with tf.Graph().as_default():
            conv_wrapper = AdaroundWrapper(session, conv, 4, QuantScheme.post_training_tf, is_symmetric=False,
                                           strict_symmetric=False, unsigned_symmetric=True, enable_per_channel=False,
                                           output_height=None, output_width=None, output_channels=None)
            hard_rounded_weight, soft_rounded_weight = AdaroundOptimizer().optimize_rounding(conv_wrapper, tf.nn.relu,
                                                                                             inp_data, out_data,
                                                                                             opt_params)

        self.assertEqual(orig_weight.shape, hard_rounded_weight.shape)
        self.assertTrue(np.allclose(orig_weight, hard_rounded_weight, atol=2 * conv_wrapper.encoding.delta))

        self.assertEqual(orig_weight.shape, soft_rounded_weight.shape)
        self.assertTrue(np.allclose(orig_weight, soft_rounded_weight, atol=2 * conv_wrapper.encoding.delta))
        session.close()

    def test_optimize_rounding_matmul(self):
        """ Test optimize rounding for MatMul """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)

        # Create model in default graph
        _ = depthwise_conv2d_model()
        init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session()
        session.run(init)
        matmul = tf.compat.v1.get_default_graph().get_operation_by_name('depthwise_conv2d_model/MatMul')
        orig_weight = session.run(matmul.inputs[1])

        inp_data = np.random.rand(1, 392)
        out_data = np.random.rand(1, 10)
        opt_params = AdaroundHyperParameters(num_iterations=1, reg_param=0.01, beta_range=(20, 2), warm_start=0.2)

        # Adaround optimization in separate graph
        with tf.Graph().as_default():
            matmul_wrapper = AdaroundWrapper(session, matmul, 4, QuantScheme.post_training_tf, is_symmetric=False,
                                             strict_symmetric=False, unsigned_symmetric=True, enable_per_channel=False,
                                             output_height=None, output_width=None, output_channels=None)
            hard_rounded_weight, soft_rounded_weight = AdaroundOptimizer().optimize_rounding(matmul_wrapper, tf.nn.relu,
                                                                                             inp_data, out_data,
                                                                                             opt_params)

        self.assertEqual(orig_weight.shape, hard_rounded_weight.shape)
        self.assertTrue(np.allclose(orig_weight, hard_rounded_weight, atol=2 * matmul_wrapper.encoding.delta))

        self.assertEqual(orig_weight.shape, soft_rounded_weight.shape)
        self.assertTrue(np.allclose(orig_weight, soft_rounded_weight, atol=2 * matmul_wrapper.encoding.delta))
        session.close()

    def test_optimize_rounding_depthwise_conv2d(self):
        """ Test optimize rounding for Depthwise Conv2d """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)

        # Create model in default graph
        _ = depthwise_conv2d_model()
        init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session()
        session.run(init)
        depthwise_conv2d = tf.compat.v1.get_default_graph().get_operation_by_name('depthwise_conv2d/depthwise')
        orig_weight = session.run(depthwise_conv2d.inputs[1])

        inp_data = np.random.rand(1, 5, 5, 10)
        out_data = np.random.rand(1, 3, 3, 10)
        opt_params = AdaroundHyperParameters(num_iterations=1, reg_param=0.01, beta_range=(20, 2), warm_start=0.2)

        # Adaround optimization in separate graph
        with tf.Graph().as_default():
            depthwise_conv_wrapper = AdaroundWrapper(session, depthwise_conv2d, 4, QuantScheme.post_training_tf,
                                                     is_symmetric=False, strict_symmetric=False, unsigned_symmetric=True,
                                                     enable_per_channel=False, output_height=None, output_width=None,
                                                     output_channels=None)
            hard_rounded_weight, soft_rounded_weight = AdaroundOptimizer().optimize_rounding(depthwise_conv_wrapper,
                                                                                             tf.nn.relu, inp_data,
                                                                                             out_data, opt_params)

        self.assertEqual(orig_weight.shape, hard_rounded_weight.shape)
        self.assertTrue(np.allclose(orig_weight, hard_rounded_weight, atol=2 * depthwise_conv_wrapper.encoding.delta))

        self.assertEqual(orig_weight.shape, soft_rounded_weight.shape)
        self.assertTrue(np.allclose(orig_weight, soft_rounded_weight, atol=2 * depthwise_conv_wrapper.encoding.delta))
        session.close()

    @pytest.mark.cuda
    def test_compute_recons_metrics(self):
        """ Test compute reconstruction metrics function """
        np.random.seed(0)
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        quant_scheme = QuantScheme.post_training_tf_enhanced
        weight_bw = 8

        # Create weight data in common format then convert into tensorflow format
        weight_data = np.random.rand(4, 4, 1, 1).astype(dtype='float32')
        weight_data = np.transpose(weight_data, (2, 3, 1, 0))
        weight_tensor = tf.convert_to_tensor(weight_data, dtype=tf.float32)

        inp_data = np.random.rand(1, 4, 10, 10).astype(dtype='float32')
        inp_tensor = tf.convert_to_tensor(inp_data, dtype=tf.float32)
        out_data = np.random.rand(1, 4, 10, 10).astype(dtype='float32')
        out_tensor = tf.convert_to_tensor(out_data, dtype=tf.float32)

        _ = tf.nn.conv2d(inp_tensor, weight_tensor, strides=[1, 1, 1, 1], padding='SAME',
                         data_format="NCHW", name='Conv2D')

        init = tf.compat.v1.global_variables_initializer()
        session = tf.compat.v1.Session()

        conv = session.graph.get_operation_by_name('Conv2D')
        conv_wrapper = AdaroundWrapper(session, conv, weight_bw, quant_scheme, is_symmetric=False,
                                       strict_symmetric=False, unsigned_symmetric=True, enable_per_channel=False,
                                       output_height=None, output_width=None, output_channels=None)
        session.run(init)
        session.run(conv_wrapper.alpha.initializer)

        print(conv_wrapper.encoding.delta, conv_wrapper.encoding.max)
        self.assertAlmostEqual(conv_wrapper.encoding.delta, 0.003772232448682189, places=3)

        # Get reconstruction error tensor
        recons_error_tensor = AdaroundOptimizer._get_recons_err_tensor(conv_wrapper, None, inp_tensor, out_tensor)

        recons_err_soft = session.run(recons_error_tensor, feed_dict={conv_wrapper.use_soft_rounding: True})
        recons_err_hard = session.run(recons_error_tensor, feed_dict={conv_wrapper.use_soft_rounding: False})
        print(recons_err_hard, recons_err_soft)
        self.assertAlmostEqual(recons_err_hard, 0.610206663608551, places=4)
        self.assertAlmostEqual(recons_err_soft, 0.6107949018478394, places=4)
        session.close()

    @pytest.mark.cuda
    def test_compute_output_with_adarounded_weights(self):
        """ Test compute output with adarounded weights for Conv layer """
        np.random.seed(0)
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        quant_scheme = QuantScheme.post_training_tf_enhanced
        weight_bw = 8

        # Create weight data in common format then convert into tensorflow format
        weight_data = np.random.rand(4, 4, 1, 1).astype(dtype='float32')
        weight_data = np.transpose(weight_data, (2, 3, 1, 0))
        weight_tensor = tf.convert_to_tensor(weight_data, dtype=tf.float32)

        inp_data = np.random.rand(1, 4, 10, 10).astype(dtype='float32')
        inp_data_t = np.transpose(inp_data, (0, 2, 3, 1))
        inp_tensor = tf.convert_to_tensor(inp_data_t, dtype=tf.float32)
        out_data = np.random.rand(1, 4, 10, 10).astype(dtype='float32')
        out_data_t = np.transpose(out_data, (0, 2, 3, 1))
        out_tensor = tf.convert_to_tensor(out_data_t, dtype=tf.float32)

        _ = tf.nn.conv2d(inp_tensor, weight_tensor, strides=[1, 1, 1, 1], padding='SAME',
                         data_format="NHWC", name='Conv2D')

        init = tf.compat.v1.global_variables_initializer()
        session = tf.compat.v1.Session()

        conv = session.graph.get_operation_by_name('Conv2D')
        conv_wrapper = AdaroundWrapper(session, conv, weight_bw, quant_scheme, is_symmetric=False,
                                       strict_symmetric=False, unsigned_symmetric=True, enable_per_channel=False,
                                       output_height=None, output_width=None, output_channels=None)
        session.run(init)
        session.run(conv_wrapper.alpha.initializer)

        # Adaround forward pass
        adaround_out_tensor = conv_wrapper(inp_tensor)

        # Compute mse loss
        mse_loss_tensor = tf.reduce_mean(tf.math.squared_difference(adaround_out_tensor, out_tensor))
        mse_loss = session.run(mse_loss_tensor)
        print(mse_loss)
        self.assertAlmostEqual(mse_loss, 0.6107949, places=2)

        # Compute adaround reconstruction loss (squared Fro norm)
        channels_index = len(out_data.shape) - 1

        recon_loss_tensor = AdaroundLoss.compute_recon_loss(adaround_out_tensor, out_tensor, channels_index)
        recons_loss = session.run(recon_loss_tensor)
        print(recons_loss)
        self.assertAlmostEqual(recons_loss, 2.4431798, places=2)
        session.close()
