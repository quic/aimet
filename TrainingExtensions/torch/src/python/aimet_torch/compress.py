# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Top-level API to AIMET compression library """

from typing import Union, Tuple
import torch

from aimet_common.defs import CostMetric, CompressionScheme, EvalFunction, CompressionStats
from aimet_common.bokeh_plots import BokehServerSession

from aimet_torch.defs import SpatialSvdParameters, WeightSvdParameters, ChannelPruningParameters
from aimet_torch.compression_factory import CompressionFactory


class ModelCompressor:
    """ AIMET model compressor: Enables model compression using various schemes """
    # Too many arguments in this function, disabling pylint for now
    @staticmethod
    def compress_model(model: torch.nn.Module, eval_callback: EvalFunction, eval_iterations,
                       input_shape: Tuple,
                       compress_scheme: CompressionScheme, cost_metric: CostMetric,
                       parameters: Union[SpatialSvdParameters,
                                         WeightSvdParameters,
                                         ChannelPruningParameters],
                       trainer=None, visualization_url=None) -> Tuple[torch.nn.Module, CompressionStats]:

        """
        Compress a given model using the specified parameters

        :param model: Model to compress
        :param eval_callback:  Evaluation callback. Expected signature is evaluate(model, iterations, use_cuda).
                               Expected to return an accuracy metric.
        :param eval_iterations: Iterations to run evaluation for
        :param trainer: Training Class: Contains a callable, train_model, which takes model, layer which is being fine
                        tuned and an optional parameter train_flag as a parameter
                        None: If per layer fine tuning is not required while creating the final compressed model
        :param input_shape: Shape of the input tensor for model
        :param compress_scheme: Compression scheme. See the enum for allowed values
        :param cost_metric: Cost metric to use for the compression-ratio (either mac or memory)
        :param parameters: Compression parameters specific to given compression scheme
        :param visualization_url: url the user will need to input where visualizations will appear
        :return: A tuple of the compressed model, and compression statistics
        """
        # pylint:disable=too-many-arguments
        # If no url is passed in, then do not create a bokeh server session
        if not visualization_url:
            bokeh_session = None
        else:
            # create a bokeh session to publish visualizations to the server document for compression
            bokeh_session = BokehServerSession(url=visualization_url, session_id="compression")

        # put model in eval mode. This is important because otherwise running a forward pass can change module buffers
        # e.g. for batchnorm layers that can affect model evaluation results
        if trainer is not None:
            trainer.train_model(model, model, train_flag=False)

        model = model.eval()

        if parameters.multiplicity < 1:
            raise ValueError('Rounding Multiplicity should be greater than 1')

        if compress_scheme == CompressionScheme.spatial_svd:
            algo = CompressionFactory.create_spatial_svd_algo(model, eval_callback, eval_iterations,
                                                              input_shape, cost_metric, parameters, bokeh_session)

        elif compress_scheme == CompressionScheme.weight_svd:
            algo = CompressionFactory.create_weight_svd_algo(model, eval_callback, eval_iterations,
                                                             input_shape, cost_metric, parameters, bokeh_session)

        elif compress_scheme == CompressionScheme.channel_pruning:
            algo = CompressionFactory.create_channel_pruning_algo(model, eval_callback, eval_iterations,
                                                                  input_shape, cost_metric, parameters, bokeh_session)

        else:
            raise ValueError("Compression scheme not supported: {}".format(compress_scheme))

        compressed_layer_db, stats = algo.compress_model(cost_metric, trainer)
        return compressed_layer_db.model, stats
