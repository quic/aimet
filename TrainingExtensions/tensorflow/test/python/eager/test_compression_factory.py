# /usr/bin/env python3.8
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

import aimet_common.cost_calculator
import aimet_tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import signal
import pytest
from unittest.mock import MagicMock
import tensorflow as tf

from aimet_common.utils import start_bokeh_server_session
from aimet_common.defs import CostMetric
from aimet_tensorflow.defs import SpatialSvdParameters, GreedySelectionParameters
from aimet_tensorflow.keras.compression_factory import CompressionFactory


def get_model():
    tf.keras.backend.clear_session()

    inp = tf.keras.Input((28, 28, 3))
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), name='conv1', padding='same')(inp)
    x = tf.keras.layers.Conv2D(64, 32, name='conv2', padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(10, name='linear')(x)

    return tf.keras.Model(inp, out)



class TestTfCompressionFactory():

    def test_spatial_svd_factory(self):
        """
        Tests spatial svd factory creates the right algo.
        Bokeh session is not used.
        :return: None
        """
        model = get_model()

        eval_callback = MagicMock()

        greedy_params = GreedySelectionParameters(0.5)
        auto_params = SpatialSvdParameters.AutoModeParams(greedy_params)
        params = SpatialSvdParameters(input_op_names=model.inputs, output_op_names=model.outputs,
                                      mode=SpatialSvdParameters.Mode.auto, params=auto_params)

        svd_algo = CompressionFactory.create_spatial_svd_algo(model, eval_callback, 100,
                                                                      CostMetric.mac, params)
        assert isinstance(svd_algo._pruner, aimet_tensorflow.keras.svd_pruner.SpatialSvdPruner)
        assert isinstance(svd_algo._cost_calculator, aimet_common.cost_calculator.SpatialSvdCostCalculator)

    def test_spatial_svd_factory_with_bokeh_session(self):
        """
        Tests spatial svd factory creates the right algo. Bokeh session is used.
        :return: None
        """

        model = get_model()

        eval_callback = MagicMock()

        greedy_params = GreedySelectionParameters(0.5)
        auto_params = SpatialSvdParameters.AutoModeParams(greedy_params)
        params = SpatialSvdParameters(input_op_names=model.inputs, output_op_names=model.outputs,
                                      mode=SpatialSvdParameters.Mode.auto, params=auto_params)

        try:
            url, process = start_bokeh_server_session(8016)

            svd_algo = CompressionFactory.create_spatial_svd_algo(model, eval_callback, 100,
                                                                          CostMetric.mac, params, url)
            assert isinstance(svd_algo._pruner, aimet_tensorflow.keras.svd_pruner.SpatialSvdPruner)
            assert isinstance(svd_algo._cost_calculator, aimet_common.cost_calculator.SpatialSvdCostCalculator)
            assert svd_algo._comp_ratio_select_algo.bokeh_session == url
        finally:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)