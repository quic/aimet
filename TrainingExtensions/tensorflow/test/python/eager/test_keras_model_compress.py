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

from aimet_tensorflow.keras.compress import ModelCompressor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from unittest.mock import MagicMock
import tensorflow as tf
import aimet_common.defs as aimet_common_defs
from aimet_tensorflow.defs import SpatialSvdParameters, GreedySelectionParameters


def get_model():
    tf.keras.backend.clear_session()

    inp = tf.keras.Input((28, 28, 3))
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), name='conv1', padding='same')(inp)
    x = tf.keras.layers.Conv2D(64, 32, name='conv2', padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(10, name='linear')(x)

    return tf.keras.Model(inp, out)


class TestSpatialSVD:
    def test_spatial_svd_compression(self):
        """
        Tests spatial svd compression utility.
        :return: None
        """
        model = get_model()

        eval_callback = MagicMock()
        eval_callback.side_effect = [0.4, 0.6, 0.6, 0.5, 0.4, 0.6, 0.6, 0.5, 0.4, 0.6]

        greedy_params = GreedySelectionParameters(0.5, 4)
        auto_params = SpatialSvdParameters.AutoModeParams(greedy_params)
        svd_params = SpatialSvdParameters(input_op_names=model.inputs, output_op_names=model.outputs,
                                          mode=SpatialSvdParameters.Mode.auto, params=auto_params)


        # Scheme is Spatial SVD:
        scheme = aimet_common_defs.CompressionScheme.spatial_svd

        # Cost metric is MAC, it can be MAC or Memory
        cost_metric = aimet_common_defs.CostMetric.mac

        compressed_model, stats = ModelCompressor.compress_model(model=model,
                                                 eval_callback=eval_callback,
                                                 eval_iterations=10,
                                                 compress_scheme=scheme,
                                                 cost_metric=cost_metric,
                                                 parameters=svd_params)

        # Check for evaluation result for each compression ratio based on the dummy result provided using Mock
        assert stats.compression_ratio_selection_stats.eval_scores_dictionary['conv1'][0.25] == 0.4
        assert stats.compression_ratio_selection_stats.eval_scores_dictionary['conv1'][0.5] == 0.6
        assert stats.compression_ratio_selection_stats.eval_scores_dictionary['conv1'][0.75] == 0.6

        assert stats.compression_ratio_selection_stats.eval_scores_dictionary['conv2'][0.25] == 0.5
        assert stats.compression_ratio_selection_stats.eval_scores_dictionary['conv2'][0.5] == 0.4
        assert stats.compression_ratio_selection_stats.eval_scores_dictionary['conv2'][0.75] == 0.6

