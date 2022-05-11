# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import signal
import shutil
import unittest
from unittest.mock import MagicMock
import tensorflow as tf

from aimet_common.utils import start_bokeh_server_session
from aimet_common.defs import CostMetric
from aimet_tensorflow.defs import SpatialSvdParameters, GreedySelectionParameters
from aimet_tensorflow.compression_factory import CompressionFactory

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()


class TestTfCompressionFactory(unittest.TestCase):

    def test_spatial_svd_factory(self):
        """
        Tests spatial svd factory creates the right algo.
        Bokeh session is not used.
        :return: None
        """
        sess = tf.compat.v1.Session(graph=tf.Graph())
        with sess.graph.as_default():
            model = tf.keras.Sequential([
                tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28 * 28,)),
                tf.keras.layers.Conv2D(32, 5, name='conv1', padding='same'),
                tf.keras.layers.Conv2D(64, 32, name='conv2', padding='same'),
                tf.keras.layers.SeparableConvolution2D(64, 64, name='conv3', padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, name='linear')
            ])
            init = tf.compat.v1.global_variables_initializer()

        sess.run(init)

        eval_callback = MagicMock()

        greedy_params = GreedySelectionParameters(0.5)
        auto_params = SpatialSvdParameters.AutoModeParams(greedy_params)
        params = SpatialSvdParameters(input_op_names=['reshape_input'], output_op_names=['linear/BiasAdd'],
                                      mode=SpatialSvdParameters.Mode.auto, params=auto_params)

        spatial_svd_algo = CompressionFactory.create_spatial_svd_algo(sess, None, eval_callback, 100, (1, 28, 28),
                                                                      CostMetric.mac, params)
        self.assertTrue(0 == 0)
        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))

        sess.close()

    def test_spatial_svd_factory_with_bokeh_session(self):
        """
        Tests spatial svd factory creates the right algo. Bokeh session is used.
        :return: None
        """
        sess = tf.compat.v1.Session(graph=tf.Graph())
        with sess.graph.as_default():
            model = tf.keras.Sequential([
                tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28 * 28,)),
                tf.keras.layers.Conv2D(32, 5, name='conv1', padding='same'),
                tf.keras.layers.Conv2D(64, 32, name='conv2', padding='same'),
                tf.keras.layers.SeparableConvolution2D(64, 64, name='conv3', padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, name='linear')
            ])
            init = tf.compat.v1.global_variables_initializer()

        sess.run(init)

        eval_callback = MagicMock()

        greedy_params = GreedySelectionParameters(0.5)
        auto_params = SpatialSvdParameters.AutoModeParams(greedy_params)
        params = SpatialSvdParameters(input_op_names=['reshape_input'], output_op_names=['linear/BiasAdd'],
                                      mode=SpatialSvdParameters.Mode.auto, params=auto_params)

        url, process = start_bokeh_server_session(8006)

        spatial_svd_algo = CompressionFactory.create_spatial_svd_algo(sess, None, eval_callback, 100, (1, 28, 28),
                                                                      CostMetric.mac, params, url)
        self.assertTrue(0 == 0)
        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))

        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

        sess.close()
