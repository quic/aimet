# /usr/bin/env python3.5
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

"""Computes compression ratio for each layer and the network """
# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_common import cost_calculator as cc
from aimet_torch.svd.svd_intf_defs_deprecated import CostMetric

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class ModelStats:
    """
    A class for calculating the statistics for a model
    """
    @staticmethod
    def compute_compression_ratio(compressed_layers, cost_metric, network_cost):
        """
        Computes the compression ratio of a model
        :param compressed_layers: layers which are compressed
        :param cost_metric: cost metric is memory or mac
        :param network_cost: mac and memory cost calculated for the entire network
        :return: It returns the compression ratio for a network
        """
        cost_calc = cc.CostCalculator()
        compressed_model_cost = cost_calc.compute_network_cost(compressed_layers)

        if cost_metric is CostMetric.memory:
            savings = network_cost.memory - compressed_model_cost.memory
            ratio = savings/network_cost.memory

        else:
            savings = network_cost.mac - compressed_model_cost.mac
            ratio = savings/network_cost.mac

        return ratio

    @staticmethod
    def compute_objective_score(model_perf, compression_score, error_margin, baseline_perf):
        """
        :param model_perf: The accuracy of model
        :param compression_score: model compression ratio
        :param error_margin: permissible error
        :param baseline_perf: initial accuracy of model
        :return:
        """
        if model_perf + (error_margin / 100) >= baseline_perf:
            objective_score = 1 - model_perf + (1 - compression_score)
        else:
            objective_score = 1 + (1 - compression_score)  # treat lower accuracies as 0

        return objective_score

    @staticmethod
    def compute_per_layer_compression_ratio(orig_layer, split_layers, metric):
        """
        Updates the per layer statistics

        :param orig_layer: The layer before it was split
        :param split_layers: List of split layers
        :param metric: Cost metric
        :return: The compression ratio of split layers
        """
        cost_calc = cc.CostCalculator()
        orig_layer_cost = cost_calc.compute_layer_cost(orig_layer)

        split_layers_cost = cc.Cost(0, 0)

        for layer in split_layers:
            split_cost = cost_calc.compute_layer_cost(layer)
            split_layers_cost += split_cost

        savings = orig_layer_cost - split_layers_cost
        if metric is CostMetric.memory:
            ratio = savings.memory / orig_layer_cost.memory
            logger.debug('Original Layer Cost: %i   Memory Compression Ratio: %f', orig_layer_cost.memory, ratio)
        else:
            ratio = savings.mac / orig_layer_cost.mac
            logger.debug('Original Layer Cost: %i   MAC Compression Ratio: %f', orig_layer_cost.mac, ratio)

        return ratio
