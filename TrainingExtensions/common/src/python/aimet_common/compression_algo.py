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

""" Abstract aimet Compression Algorithm """

from decimal import Decimal
from typing import List, Tuple
import pickle
import os
from aimet_common.comp_ratio_select import GreedyCompRatioSelectAlgo
from aimet_common.defs import CostMetric, LayerCompRatioPair, EvalFunction, CompressionStats
from aimet_common import cost_calculator as cc
from aimet_common.pruner import Pruner
from aimet_common.comp_ratio_select import CompRatioSelectAlgo
from aimet_common.layer_database import LayerDatabase
from aimet_common import plotting_utils


class CompressionAlgo:
    """
    Abstract class modeling a generic compression algorithm
    """

    PICKLE_FILE_COMP_RATIO_LIST = './data/greedy_selection_comp_ratios_list.pkl'

    def __init__(self, layer_db: LayerDatabase, comp_ratio_select_algo: CompRatioSelectAlgo, pruner: Pruner,
                 eval_func: EvalFunction, layer_selector, modules_to_ignore: List,
                 cost_calculator: cc.CostCalculator, use_cuda: bool):
        # pylint: disable=too-many-arguments

        self._layer_db = layer_db
        self._comp_ratio_select_algo = comp_ratio_select_algo
        self._pruner = pruner
        self._eval_func = eval_func
        self._layer_selector = layer_selector
        self._modules_to_ignore = modules_to_ignore
        self._cost_calculator = cost_calculator
        self._use_cuda = use_cuda

    def compress_model(self, cost_metric: CostMetric, trainer) -> Tuple[LayerDatabase, CompressionStats]:
        """
        Compress model
        :param cost_metric: Cost metric to use compression (mac or memory)
        :param trainer: Training function
                        None: If per layer fine tuning is not required while creating the final compressed model
        :return: LayerDatabase object for the compressed model
        """
        # Select layers
        self._layer_selector.select(self._layer_db, self._modules_to_ignore)

        # Find optimal compression ratios for each layer
        layer_comp_ratio_list, stats = self._comp_ratio_select_algo.select_per_layer_comp_ratios()

        self._pickle_comp_ratio_list(layer_comp_ratio_list)
        if isinstance(self._comp_ratio_select_algo, GreedyCompRatioSelectAlgo) and\
                self._comp_ratio_select_algo.bokeh_session:
            # visualize comp ratios vs layers in a plot and add it to a server session document.
            comp_ratios = [i.comp_ratio for i in layer_comp_ratio_list]
            layer_names = [i.layer.name for i in layer_comp_ratio_list]
            optimal_comp_ratios_plot = plotting_utils.plot_optimal_compression_ratios(comp_ratios, layer_names)
            self._comp_ratio_select_algo.bokeh_session.document.add_root(optimal_comp_ratios_plot)

        # Create a compressed model using these optimal compression ratios per layer
        compressed_layer_db = self._pruner.prune_model(self._layer_db,
                                                       layer_comp_ratio_list,
                                                       cost_metric, trainer)
        compressed_model_cost = self._cost_calculator.compute_model_cost(compressed_layer_db)
        stats = self._compile_stats(compressed_layer_db, compressed_model_cost, layer_comp_ratio_list, stats)

        return compressed_layer_db, stats

    def _compile_stats(self, compressed_layer_db: LayerDatabase,
                       compressed_model_cost: cc.Cost,
                       layer_comp_ratio_list: List[LayerCompRatioPair],
                       compression_ratio_select_stats) -> CompressionStats:
        """
        Compile compression statistics
        :param compressed_layer_db: LayerDatabase for the compressed model
        :param layer_comp_ratio_list: List of per-layer compression ratios
        :return: CompressionStats instance
        """

        # Baseline accuracy
        baseline_accuracy = self._eval_func(self._layer_db.model, None, self._use_cuda)

        # Compressed accuracy
        compressed_accuracy = self._eval_func(compressed_layer_db.model, None, self._use_cuda)

        # Compression-ratios
        original_model_cost = cc.CostCalculator.compute_model_cost(self._layer_db)
        mem_comp_ratio = Decimal(compressed_model_cost.memory / original_model_cost.memory)
        mac_comp_ratio = Decimal(compressed_model_cost.mac / original_model_cost.mac)

        layer_stats = []
        for layer_ratio_pair in layer_comp_ratio_list:
            layer_stats.append(CompressionStats.LayerStats(layer_ratio_pair.layer.name, layer_ratio_pair.comp_ratio))

        stats = CompressionStats(baseline_accuracy, compressed_accuracy, mem_comp_ratio, mac_comp_ratio, layer_stats,
                                 compression_ratio_select_stats)

        return stats

    def _pickle_comp_ratio_list(self, layer_comp_ratio_list):
        comp_ratios_list = []
        for entry in layer_comp_ratio_list:
            layer_comp_ratio_tuple = (entry.layer.name, entry.comp_ratio)
            comp_ratios_list.append(layer_comp_ratio_tuple)

        if not os.path.exists('./data'):
            os.makedirs('./data')

        with open(self.PICKLE_FILE_COMP_RATIO_LIST, 'wb') as file:
            pickle.dump(comp_ratios_list, file)

    @staticmethod
    def unpickle_comp_ratios_list(comp_ratio_list_path: str):
        """ unpickles the optimal comp ratio list
        :param comp_ratio_list_path: path to comp ratio list
        :return: compression ratio
        """

        with open(comp_ratio_list_path, 'rb') as f:
            comp_ratios = pickle.load(f)

        return comp_ratios
