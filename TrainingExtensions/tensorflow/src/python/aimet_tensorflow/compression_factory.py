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

""" Provides a factory to construct various aimet model compression classes based on a scheme """

from typing import Tuple, List, Union

import tensorflow as tf

from aimet_common.defs import CostMetric, EvalFunction, LayerCompRatioPair
from aimet_common.cost_calculator import SpatialSvdCostCalculator
from aimet_common.comp_ratio_select import GreedyCompRatioSelectAlgo, ManualCompRatioSelectAlgo
from aimet_common.comp_ratio_rounder import RankRounder, ChannelRounder
from aimet_common.compression_algo import CompressionAlgo
from aimet_common.bokeh_plots import BokehServerSession

from aimet_tensorflow.defs import SpatialSvdParameters, ModuleCompRatioPair, ChannelPruningParameters
from aimet_tensorflow.layer_selector import ConvNoDepthwiseLayerSelector, ManualLayerSelector
from aimet_tensorflow.layer_database import LayerDatabase
from aimet_tensorflow.svd_pruner import SpatialSvdPruner
from aimet_tensorflow.channel_pruning.channel_pruner import InputChannelPruner, ChannelPruningCostCalculator


class CompressionFactory:
    """ Factory to construct various aimet model compression classes based on a scheme """

    @classmethod
    def create_spatial_svd_algo(cls, sess: tf.compat.v1.Session, working_dir: str, eval_callback: EvalFunction, eval_iterations,
                                input_shape: Union[Tuple, List[Tuple]], cost_metric: CostMetric,
                                params: SpatialSvdParameters, bokeh_session=None) -> CompressionAlgo:
        """
        Factory method to construct SpatialSvdCompressionAlgo

        :param sess: Model, represented by a tf.compat.v1.Session, to compress
        :param working_dir: path to store temp meta and checkpoint files
        :param eval_callback: Evaluation callback for the model
        :param eval_iterations: Evaluation iterations
        :param input_shape: tuple or list of tuples of input shape to the model
        :param cost_metric: Cost metric (mac or memory)
        :param params: Spatial SVD compression parameters
        :param bokeh_session: The Bokeh Session to display plots
        :return: An instance of SpatialSvdCompressionAlgo
        """

        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        # Rationale: Factory functions unfortunately need to deal with a lot of parameters

        # Create a layer database
        layer_db = LayerDatabase(sess, input_shape, working_dir, starting_ops=params.input_op_names,
                                 ending_ops=params.output_op_names)
        use_cuda = False

        # Create a pruner
        pruner = SpatialSvdPruner()
        cost_calculator = SpatialSvdCostCalculator()
        comp_ratio_rounding_algo = RankRounder(params.multiplicity, cost_calculator)

        # Create a comp-ratio selection algorithm
        if params.mode == SpatialSvdParameters.Mode.auto:
            greedy_params = params.mode_params.greedy_params
            comp_ratio_select_algo = GreedyCompRatioSelectAlgo(layer_db, pruner, cost_calculator, eval_callback,
                                                               eval_iterations, cost_metric,
                                                               greedy_params.target_comp_ratio,
                                                               greedy_params.num_comp_ratio_candidates,
                                                               greedy_params.use_monotonic_fit,
                                                               greedy_params.saved_eval_scores_dict,
                                                               comp_ratio_rounding_algo, use_cuda,
                                                               bokeh_session=bokeh_session)
            layer_selector = ConvNoDepthwiseLayerSelector()
            modules_to_ignore = params.mode_params.modules_to_ignore

        else:
            # Convert (module,comp-ratio) pairs to (layer,comp-ratio) pairs
            layer_comp_ratio_pairs = cls._get_layer_pairs(layer_db, params.mode_params.list_of_module_comp_ratio_pairs)

            comp_ratio_select_algo = ManualCompRatioSelectAlgo(layer_db,
                                                               layer_comp_ratio_pairs,
                                                               comp_ratio_rounding_algo, cost_metric=cost_metric)

            layer_selector = ManualLayerSelector(layer_comp_ratio_pairs)
            modules_to_ignore = []

        # Create the overall Spatial SVD compression algorithm
        spatial_svd_algo = CompressionAlgo(layer_db, comp_ratio_select_algo, pruner, eval_callback,
                                           layer_selector, modules_to_ignore, cost_calculator, use_cuda)

        return spatial_svd_algo

    @classmethod
    def create_channel_pruning_algo(cls, sess: tf.compat.v1.Session, working_dir: str, eval_callback: EvalFunction,
                                    input_shape: Union[Tuple, List[Tuple]], eval_iterations, cost_metric: CostMetric,
                                    params: ChannelPruningParameters, bokeh_session: BokehServerSession) -> \
            CompressionAlgo:
        """
        Factory method to construct ChannelPruningCompressionAlgo
        :param sess: Model, represented by a tf.compat.v1.Session, to compress
        :param working_dir: path to store temp meta and checkpoint files
        :param eval_callback: Evaluation callback for the model
        :param eval_iterations: Evaluation iterations
        :param input_shape: tuple or list of tuples of input shapes to the model
        :param cost_metric: Cost metric (mac or memory)
        :param params: Channel Pruning compression parameters
        :param bokeh_session: The Bokeh session to display plots
        :return: An instance of ChannelPruningCompressionAlgo
        """

        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        # Rationale: Factory functions unfortunately need to deal with a lot of parameters

        # Create a layer database
        layer_db = LayerDatabase(sess, input_shape, working_dir, starting_ops=params.input_op_names,
                                 ending_ops=params.output_op_names)
        use_cuda = False

        # Create a pruner
        pruner = InputChannelPruner(input_op_names=params.input_op_names, output_op_names=params.output_op_names,
                                    data_set=params.data_set, batch_size=params.batch_size,
                                    num_reconstruction_samples=params.num_reconstruction_samples,
                                    allow_custom_downsample_ops=params.allow_custom_downsample_ops)

        comp_ratio_rounding_algo = ChannelRounder(params.multiplicity)

        # Create a comp-ratio selection algorithm
        cost_calculator = ChannelPruningCostCalculator(pruner)

        if params.mode == ChannelPruningParameters.Mode.auto:
            greedy_params = params.mode_params.greedy_params
            comp_ratio_select_algo = GreedyCompRatioSelectAlgo(layer_db, pruner, cost_calculator, eval_callback,
                                                               eval_iterations, cost_metric,
                                                               greedy_params.target_comp_ratio,
                                                               greedy_params.num_comp_ratio_candidates,
                                                               greedy_params.use_monotonic_fit,
                                                               greedy_params.saved_eval_scores_dict,
                                                               comp_ratio_rounding_algo, use_cuda,
                                                               bokeh_session=bokeh_session)
            layer_selector = ConvNoDepthwiseLayerSelector()
            modules_to_ignore = params.mode_params.modules_to_ignore

        else:
            # Convert (module,comp-ratio) pairs to (layer,comp-ratio) pairs
            layer_comp_ratio_pairs = cls._get_layer_pairs(layer_db, params.mode_params.list_of_module_comp_ratio_pairs)

            comp_ratio_select_algo = ManualCompRatioSelectAlgo(layer_db,
                                                               layer_comp_ratio_pairs,
                                                               comp_ratio_rounding_algo, cost_metric=cost_metric)

            layer_selector = ManualLayerSelector(layer_comp_ratio_pairs)
            modules_to_ignore = []

        # Create the overall Channel Pruning compression algorithm
        channel_pruning_algo = CompressionAlgo(layer_db, comp_ratio_select_algo, pruner, eval_callback,
                                               layer_selector, modules_to_ignore, cost_calculator, use_cuda)

        return channel_pruning_algo

    @staticmethod
    def _get_layer_pairs(layer_db: LayerDatabase, module_comp_ratio_pairs: List[ModuleCompRatioPair]):
        layer_comp_ratio_pairs = []

        for pair in module_comp_ratio_pairs:
            layer_comp_ratio_pair = LayerCompRatioPair(layer_db.find_layer_by_module(pair.module),
                                                       pair.comp_ratio)
            layer_comp_ratio_pairs.append(layer_comp_ratio_pair)

        return layer_comp_ratio_pairs
