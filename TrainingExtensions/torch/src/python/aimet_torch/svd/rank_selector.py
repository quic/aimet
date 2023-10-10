# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""Gives best rank for compression"""
from aimet_common.utils import AimetLogger
from aimet_common import statistics_util as stats_u, cost_calculator as cc
from aimet_torch.svd import model_stats_calculator as MS
from aimet_torch.svd import svd_pruner_deprecated

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class RankSelector:
    """
    Selects the best rank for a given layer
    """

    def __init__(self, svd_lib_ref):
        self._svd_lib_ref = svd_lib_ref

    def _select_candidate_ranks(self, num_rank_indices):
        return self._svd_lib_ref.SetCandidateRanks(num_rank_indices)

    def choose_best_rank(self, model, run_model, run_model_iterations, use_cuda, metric, error_margin, baseline_perf,
                         num_rank_indices, database):
        """
        :param model: Original model
        :param run_model: Method to run evaluation on model
        :param run_model_iterations: Number of iterations for run_model
        :param use_cuda: Model is on GPU or not
        :param metric: cost metric
        :param error_margin: permissible error allowed for rank selection
        :param baseline_perf: original model's accuracy
        :param num_rank_indices: number of rank indices
        :param database: reference to Layer Attribute Database
        :return:
        """
        # pylint: disable=too-many-arguments, too-many-locals

        num_rank_indices = self._select_candidate_ranks(num_rank_indices)
        cost_calc = cc.CostCalculator()
        network_cost = cost_calc.compute_model_cost(database)
        # Ranks are in order from least compression to highest
        best_index = None
        min_objective_score = None

        # List to hold the SVD Statistics for all the Rank indices
        rank_stats_list = list()

        for rank_index in range(num_rank_indices):
            svd_rank_pair_dict = {}
            for layer in database.get_selected_layers():

                # Get the candidate ranks for given rank index
                svd_ranks = self._svd_lib_ref.GetCandidateRanks(str(layer.name), rank_index)
                svd_rank_pair_dict[layer.name] = (svd_ranks[0], 0)
            # Compress the model given a rank index
            compressed_model, compressed_layers, layer_stats_list = svd_pruner_deprecated.ModelPruner().create_compressed_model(
                svd_rank_pair_dict=svd_rank_pair_dict, model=model,
                compressible_layers=database.get_compressible_layers(), svd_lib_ref=self._svd_lib_ref, metric=metric)
            ms = MS.ModelStats

            # Estimate relative compression score for this rank_index
            compression_score = ms.compute_compression_ratio(compressed_layers, metric, network_cost)
            logger.debug('Rank Index: %i, Compression Score: %f', rank_index, compression_score)

            # Get accuracy for the compressed model
            model_perf = run_model(compressed_model, run_model_iterations, use_cuda)

            model_accuracy = model_perf
            model_compression_ratio = compression_score

            objective_score = ms.compute_objective_score(model_perf, compression_score, error_margin, baseline_perf)

            logger.info('Compressed network with rank_index %i/%i: accuracy = %f percent '
                        'with %f percent compression (%r option) and an objective score of %f',
                        rank_index, num_rank_indices, model_perf*100, compression_score*100,
                        metric, objective_score)

            if not min_objective_score:
                min_objective_score = objective_score
                logger.info('Initializing objective score to %f at rank index %i',
                            min_objective_score, rank_index)

            if model_perf + error_margin/100 < baseline_perf:
                logger.info('Model performance %f falls below %f percent of baseline performance %f'
                            ' Ending rank selection', model_perf*100, error_margin, baseline_perf*100)
                break
            else:
                if objective_score <= min_objective_score:
                    min_objective_score = objective_score
                    logger.info('Found a better value for the objective score %f at rank_index %i',
                                min_objective_score, rank_index)
                    best_index = rank_index
                    svd_rank_pair_dict_best_index = svd_rank_pair_dict

            # Create the Per Rank Index Statistics object.
            rank_data = stats_u.SvdStatistics.PerRankIndex(rank_index=rank_index, model_accuracy=model_accuracy,
                                                           model_compression_ratio=model_compression_ratio,
                                                           layer_stats_list=layer_stats_list)
            rank_stats_list.append(rank_data)

        if not best_index:
            raise RuntimeError('No suitable ranks found to compress model within defined error bounds.')

        return best_index, svd_rank_pair_dict_best_index, rank_stats_list

    def split_manual_rank(self, model, run_model, run_model_iterations, use_cuda, metric, database, **kw_args):
        """
        :param model: The original model
        :param run_model: Method to run evaluation on model
        :param run_model_iterations: Number of iterations for run_model
        :param use_cuda: Model is on GPU or not
        :param metric: cost metric
        :param database: Layer attribute database reference
        :param kw_args: layer rank list
        :return:
        """
        # pylint: disable=too-many-locals

        cost_calc = cc.CostCalculator()
        network_cost = cost_calc.compute_model_cost(database)
        layer_rank_list = kw_args['layer_rank_list']
        svd_rank_pair_dict = {}
        for name, module in model.named_modules():
            for layer, rank in layer_rank_list:
                if layer is module:
                    svd_rank_pair_dict[name] = (rank, 0)

        compressed_model, \
        compressed_model_layers, \
        layer_stats_list = svd_pruner_deprecated.ModelPruner().create_compressed_model(svd_rank_pair_dict=svd_rank_pair_dict,
                                                                                       model=model,
                                                                                       compressible_layers=database.get_compressible_layers(),
                                                                                       svd_lib_ref=self._svd_lib_ref,
                                                                                       metric=metric)
        model_perf = run_model(compressed_model, run_model_iterations, use_cuda)
        ms = MS.ModelStats
        compression_score = ms.compute_compression_ratio(compressed_model_layers, metric, network_cost)
        rank_data = stats_u.SvdStatistics.PerRankIndex(rank_index=0, model_accuracy=model_perf,
                                                       model_compression_ratio=compression_score,
                                                       layer_stats_list=layer_stats_list)
        rank_data_list = list()
        rank_data_list.append(rank_data)

        return rank_data_list, svd_rank_pair_dict
