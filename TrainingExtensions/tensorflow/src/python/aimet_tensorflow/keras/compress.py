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

""" Top-level API for aimet compression library """

from typing import Union, Tuple, Callable
import tensorflow as tf

from aimet_common.defs import CostMetric, CompressionScheme, EvalFunction, CompressionStats
from aimet_common.bokeh_plots import BokehServerSession

from aimet_tensorflow.utils.graph_saver import keras_wrapper_func, keras_save_and_load_graph, keras_remove_hanging_nodes
from aimet_tensorflow.defs import SpatialSvdParameters
from aimet_tensorflow.keras.compression_factory import CompressionFactory


class ModelCompressor:
    """ aimet model compressor: Enables model compression using various schemes """

    # pylint: disable=too-many-arguments

    @staticmethod
    def compress_model(model: tf.keras.Model, eval_callback: EvalFunction, eval_iterations,
                       compress_scheme: CompressionScheme, cost_metric: CostMetric,
                       parameters: Union[SpatialSvdParameters],
                       trainer: Callable = None, visualization_url: str = None) -> Tuple[tf.keras.Model, CompressionStats]:
        """
        Compress a given model using the specified parameters

        :param model: Model, represented by a tf.keras.Model, to compress
        :param eval_callback:  Evaluation callback. Expected signature is evaluate(model, iterations, use_cuda).
                               Expected to return an accuracy metric.
        :param eval_iterations: Iterations to run evaluation for.
        :param compress_scheme: Compression scheme. See the enum for allowed values
        :param cost_metric: Cost metric to use for the compression-ratio (either mac or memory)
        :param parameters: Compression parameters specific to given compression scheme
        :param trainer: Training function
                        None: If per layer fine-tuning is not required while creating the final compressed model
        :param visualization_url: url the user will need to input where visualizations will appear
        :return: A tuple of the compressed model session, and compression statistics
        """

        # If no url is passed in, then do not create a bokeh server session
        if not visualization_url:
            bokeh_session = None
        else:
            # create a bokeh session to publish visualizations to the server document for compression
            bokeh_session = BokehServerSession(url=visualization_url, session_id="compression")

        if parameters.multiplicity < 1:
            raise ValueError('Rounding Multiplicity should be greater than 1')

        # wrapper_func saves and reloads the graph before evaluation
        # In Keras after making changes to the graph you must save and reload, then evaluate
        eval_callback = keras_wrapper_func(eval_callback)

        if compress_scheme == CompressionScheme.spatial_svd:
            algo = CompressionFactory.create_spatial_svd_algo(model, eval_callback, eval_iterations,
                                                              cost_metric, parameters, bokeh_session)
        elif compress_scheme == CompressionScheme.weight_svd:
            raise NotImplementedError("Not yet implemented for: {}".format(compress_scheme))
        elif compress_scheme == CompressionScheme.channel_pruning:
            raise NotImplementedError("Not yet implemented for: {}".format(compress_scheme))
        else:
            raise ValueError("Compression scheme not supported: {}".format(compress_scheme))

        compressed_layer_db, stats = algo.compress_model(cost_metric, trainer)

        # In keras after making changes to the model you must save and reload, then evaluate
        tmp_dir = './data/saved_model'
        updated_model = keras_save_and_load_graph(tmp_dir, compressed_layer_db.model)

        # Remove the hanging nodes
        updated_model = keras_remove_hanging_nodes(updated_model)

        return updated_model, stats
