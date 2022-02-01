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

""" Test AdaroundLoss """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pytest
import unittest.mock
import numpy as np
import tensorflow as tf

from aimet_tensorflow.adaround.adaround_loss import AdaroundLoss

tf.compat.v1.disable_eager_execution()


class TestAdaroundLoss(unittest.TestCase):
    """ Test AdaroundLoss """

    def _compute_recon_loss(self, device):
        """ test compute reconstruction loss using dummy input and target tensors """
        tf.compat.v1.reset_default_graph()
        session = tf.compat.v1.Session()
        np.random.seed(0)
        inp = np.random.rand(32, 3, 12, 12)
        target = np.random.rand(32, 3, 12, 12)

        # transpose both the matrices - channels_last data format
        inp_t = np.transpose(inp, (0, 2, 3, 1))
        target_t = np.transpose(target, (0, 2, 3, 1))
        channels_index = len(target_t.shape) - 1

        with tf.device(device):
            inp_tensor = tf.convert_to_tensor(inp_t, dtype=tf.float32)
            target_tensor = tf.convert_to_tensor(target_t, dtype=tf.float32)
            recons_loss = AdaroundLoss.compute_recon_loss(inp_tensor, target_tensor, channels_index)

        self.assertAlmostEqual(session.run(recons_loss), 0.5023074755644538, places=4)

        # Matmul
        # pytorch and tensorflow both are channels_last, so transpose is not needed.
        inp = np.random.rand(32, 10)
        target = np.random.rand(32, 10)
        channels_index = len(target.shape) - 1

        with tf.device(device):
            inp_tensor = tf.convert_to_tensor(inp, dtype=tf.float32)
            target_tensor = tf.convert_to_tensor(target, dtype=tf.float32)
            recons_loss = AdaroundLoss.compute_recon_loss(inp_tensor, target_tensor, channels_index)

        self.assertAlmostEqual(session.run(recons_loss), 1.6614693981871231, places=4)

        session.close()

    @pytest.mark.cuda
    def test_compute_recon_loss_gpu(self):
        """ test compute reconstruction loss using dummy input and target tensors for GPU """
        device = '/gpu:0'
        self._compute_recon_loss(device)

    def test_compute_recon_loss(self):
        """ test compute reconstruction loss using dummy input and target tensors for CPU """
        device = '/cpu:0'
        self._compute_recon_loss(device)

    def _compute_round_loss(self, device):
        """ test compute rounding loss """
        tf.compat.v1.reset_default_graph()
        np.random.seed(0)
        alpha = np.random.rand(1, 3, 12, 12)
        reg_param = 0.01

        with tf.device(device):
            beta_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[])
            warm_start_tensor = tf.compat.v1.placeholder(dtype=tf.bool, shape=[])
            alpha_tensor = tf.convert_to_tensor(alpha, dtype=tf.float32)
            round_loss_tensor = AdaroundLoss.compute_round_loss(alpha_tensor, reg_param, warm_start_tensor, beta_tensor)
        session = tf.compat.v1.Session()

        beta = 17.388903112897612
        warm_start = True
        rounding_loss_1 = session.run(round_loss_tensor, feed_dict={beta_tensor: beta, warm_start_tensor: warm_start})
        # Since warm start is 0.2 (20%), cut iter < 2000 (20% of 10000 iterations) will have rounding loss = 0
        self.assertEqual(rounding_loss_1, 0)

        beta = 4.636038969321072
        warm_start = False
        rounding_loss_2 = session.run(round_loss_tensor, feed_dict={beta_tensor: beta, warm_start_tensor: warm_start})
        self.assertAlmostEqual(rounding_loss_2, 4.266156963161077, places=4)

        session.close()

    @pytest.mark.cuda
    def test_compute_round_loss_gpu(self):
        """ test compute rounding loss for GPU """
        device = '/gpu:0'
        self._compute_round_loss(device)

    def test_compute_round_loss(self):
        """ test compute rounding loss for CPU """
        device = '/cpu:0'
        self._compute_round_loss(device)

    def test_compute_beta(self):
        """ test compute beta """
        num_iterations = 10000
        cur_iter = 8000
        beta_range = (20, 2)
        warm_start = 0.2
        self.assertEqual(AdaroundLoss.compute_beta(num_iterations, cur_iter, beta_range, warm_start),
                         4.636038969321072)
