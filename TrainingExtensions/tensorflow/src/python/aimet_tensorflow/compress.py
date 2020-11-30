# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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

from typing import Union, Tuple, List
import tensorflow as tf

from aimet_common.defs import CostMetric, CompressionScheme, EvalFunction, CompressionStats
from aimet_common.bokeh_plots import BokehServerSession

from aimet_tensorflow.utils.graph_saver import wrapper_func, save_and_load_graph
from aimet_tensorflow.defs import SpatialSvdParameters, ChannelPruningParameters
from aimet_tensorflow.compression_factory import CompressionFactory


class ModelCompressor:
    """ aimet model compressor: Enables model compression using various schemes """

    # pylint: disable=too-many-arguments

    @staticmethod
    def compress_model(sess: tf.compat.v1.Session, working_dir: str, eval_callback: EvalFunction, eval_iterations,
                       input_shape: Union[Tuple, List[Tuple]],
                       compress_scheme: CompressionScheme, cost_metric: CostMetric,
                       parameters: Union[SpatialSvdParameters,
                                         ChannelPruningParameters],
                       trainer=None, visualization_url=None) -> Tuple[tf.compat.v1.Session, CompressionStats]:
        """
        Compress a given model using the specified parameters

        :param sess: Model, represented by a tf.compat.v1.Session, to compress
        :param working_dir: File path to save compressed TensorFlow meta file
        :param eval_callback:  Evaluation callback. Expected signature is evaluate(model, iterations, use_cuda).
                               Expected to return an accuracy metric.
        :param eval_iterations: Iterations to run evaluation for
        :param trainer: Training Class: Contains a callable, train_model, which takes model, layer which is being fine
                        tuned and an optional parameter train_flag as a parameter
                        None: If per layer fine tuning is not required while creating the final compressed model
        :param input_shape: tuple or list of tuples of input shapes to the model (channels_last format)
        :param compress_scheme: Compression scheme. See the enum for allowed values
        :param cost_metric: Cost metric to use for the compression-ratio (either mac or memory)
        :param parameters: Compression parameters specific to given compression scheme
        :param trainer: Training function
                        None: If per layer fine tuning is not required while creating the final compressed model
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

        if compress_scheme == CompressionScheme.spatial_svd:
            # wrapper_func saves and reloads the graph before evaluation
            # In TF after making changes to the graph you must save and reload, then evaluate
            eval_callback = wrapper_func(eval_callback)

            algo = CompressionFactory.create_spatial_svd_algo(sess, working_dir, eval_callback, eval_iterations,
                                                              input_shape, cost_metric, parameters, bokeh_session)
        elif compress_scheme == CompressionScheme.channel_pruning:
            algo = CompressionFactory.create_channel_pruning_algo(sess, working_dir, eval_callback, input_shape,
                                                                  eval_iterations, cost_metric, parameters,
                                                                  bokeh_session)
        else:
            raise ValueError("Compression scheme not supported: {}".format(compress_scheme))

        compressed_layer_db, stats = algo.compress_model(cost_metric, trainer)

        # TODO: this is a temporary fix, needs to be resolved
        # In TF after making changes to the graph you must save and reload, then evaluate
        updated_model = save_and_load_graph('./saver', compressed_layer_db.model)
        compressed_layer_db.model.close()

        return updated_model, stats
