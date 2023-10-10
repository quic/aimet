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

""" Implementation of the AIMET SVD model-compression feature """

from enum import Enum
from torch import cuda

# Import AIMET specific modules
import aimet_common.libpymo as pymo
import aimet_common.defs
from aimet_common.utils import AimetLogger
from aimet_common import statistics_util as stats_u, cost_calculator as cc

from aimet_torch.utils import create_rand_tensors_given_shapes, get_device
from aimet_torch import pymo_utils
from aimet_torch.svd.model_stats_calculator import ModelStats
from aimet_torch.svd.svd_intf_defs_deprecated import CostMetric, RankSelectionScheme
from aimet_torch.svd import layer_selector_deprecated as ls
from aimet_torch.svd.svd_pruner_deprecated import ModelPruner
from aimet_torch.svd.rank_selector import RankSelector
from aimet_torch import layer_database as database


class LayerTypes(Enum):
    """ Enumeration of layer types (e.g. conv, fc) supported for SVD """
    Conv2D = pymo.LAYER_TYPE_CONV
    Linear = pymo.LAYER_TYPE_FC


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class SvdImpl:
    """A class for performing singular value decomposition on a PyTorch model.

    The Svd class enables model compression through singular value decomposition (SVD).
    It can analyze convolution and fully connected layers and perform
    some analysis to find the optimal ranks for balancing compression and the
    accuracy of the network.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, model, run_model, run_model_iterations, input_shape,
                 compression_type, cost_metric,
                 layer_selection_scheme, **kw_layer_select_params):
        """Constructor for the Svd class

        Constructs the Svd class from a set of options passed in at construction. The class takes
        a number of named arguments which are detailed below.

        :param model: The model which needs to be compressed
        :param run_model: The evaluation function that needs to be passed for one forward pass
        :param run_model_iterations: The number of iterations of forward pass for the run_model
        :param input_shape: Shape of the inputs to the model
        :param compression_type: Enum argument. Options available: svd , ssvd.
        :param cost_metric: Enum argument. Options available: mac, memory
        :param layer_selection_scheme: Enum argument. Options available: manual, top_n_layers, top_x_percent
        :param kw_layer_select_params: Params for layer selection. Params depend on modes selected
                    1) If the layer_selection_scheme is manual then user has to specify the list of layers by using- layers_to_compress= [list of layers],
                    2) If the layer_selection_scheme is top_n_layers then the user has to specify the number of layers as num_layers= <number>
                    3) If the layer_selection_scheme is top_x_percent then the user has to specify percentage threshold by using percent_thresh= <number>
        :return:
            Todo: Add description
        """

        # ------------------------------
        # Initialize state
        # ------------------------------
        self._model = model
        self._run_model = run_model
        self._run_model_iterations = run_model_iterations
        self._svd_lib_ref = pymo.GetSVDInstance()
        self._network_cost = None
        self._compression_type = compression_type
        self._metric = cost_metric
        # Layer selection related state
        self._layer_selection_scheme = layer_selection_scheme

        self._selected_layers = list()
        self._use_cuda = self._is_model_on_gpu()
        if self._use_cuda is True:
            if not cuda.is_available():
                raise ValueError('CUDA use was expected but CUDA is not available!')
        # ------------------------------
        # Creating the layer attribute database
        device = get_device(model)
        dummy_input = create_rand_tensors_given_shapes(input_shape, device)
        self._layer_database = database.LayerDatabase(model=self._model, dummy_input=dummy_input)

        # picking layers for compression based on the scheme
        ls.LayerSelectorDeprecated(layer_selection_scheme, cost_metric, self._layer_database,
                                   **kw_layer_select_params)

        # Hack for now
        if cost_metric == CostMetric.memory:
            pymo_cost_metric = aimet_common.defs.CostMetric.memory
        else:
            pymo_cost_metric = aimet_common.defs.CostMetric.mac

        pymo_utils.PymoSvdUtils.configure_layers_in_pymo_svd(self._layer_database.get_selected_layers(),
                                                             pymo_cost_metric,
                                                             self._svd_lib_ref)

        logger.info("Selected layers: %s", self._selected_layers)
        logger.info("Model is created on GPU : %s", self._use_cuda)

    def _is_model_on_gpu(self):
        """
        Function to check whether user defined model is created on GPU or CPU
        Assumption : model is on single device
        :return:
            True if the model is on GPU, False if on CPU
        """
        return all([param.is_cuda for param in self._model.parameters()])

    def _final_compressed_network(self, best_index, svd_rank_pair_dict_best_index, rank_stats_list):
        baseline_accuracy = self._run_model(self._model, self._run_model_iterations, self._use_cuda)

        best_model, best_compressed_layers, _ = ModelPruner().create_compressed_model(
            svd_rank_pair_dict=svd_rank_pair_dict_best_index, model=self._model,
            compressible_layers=self._layer_database.get_compressible_layers(), svd_lib_ref=self._svd_lib_ref,
            metric=self._metric)
        stats = self._update_statistics(baseline_accuracy, best_model, best_compressed_layers, rank_stats_list, best_index)
        return best_model, stats

    def _update_statistics(self, baseline_accuracy, best_model, best_compressed_layers, rank_stats_list, best_index):
        """ Function that updates the SVD statistics after the svd compression is completed.

        :param baseline_accuracy: Accuracy of the model before it was compressed.
        :param best_index: The best rank index that was used to compress the model.
        :param rank_stats_list: A list of Rank index specific SVD statistics
        :return: SvdStatistics object that contains all of the SVD Statistics
        """

        compressed_model_accuracy = self._run_model(best_model, self._run_model_iterations, self._use_cuda)
        cost_calc = cc.CostCalculator()
        network_cost = cost_calc.compute_model_cost(self._layer_database)
        memory_compression_ratio = ModelStats.compute_compression_ratio(best_compressed_layers,
                                                                        CostMetric.memory, network_cost)
        mac_compression_ratio = ModelStats.compute_compression_ratio(best_compressed_layers,
                                                                     CostMetric.mac, network_cost)
        stats = stats_u.SvdStatistics(base_accuracy=baseline_accuracy, comp_accuracy=compressed_model_accuracy,
                                      cost_metric=self._metric, best_index=best_index,
                                      mem_comp_ratio=memory_compression_ratio,
                                      mac_comp_ratio=mac_compression_ratio, rank_stats_list=rank_stats_list)
        return stats

    def compress_net(self, rank_selection_scheme, **kw_args):
        """

        :param rank_selection_scheme:
        :param kw_args:
        :return:
        """

        logger.info('Started SVD!')

        baseline_accuracy = self._run_model(self._model, self._run_model_iterations, self._use_cuda)
        logger.info("Baseline accuracy: %f", baseline_accuracy * 100)
        rank_selector = RankSelector(svd_lib_ref=self._svd_lib_ref)
        if rank_selection_scheme is RankSelectionScheme.auto:
            num_rank_indices = kw_args['num_rank_indices']
            error_margin = kw_args['error_margin']
            best_index, svd_rank_pair_dict_best_index, rank_stats_list = \
                rank_selector.choose_best_rank(model=self._model, run_model=self._run_model,
                                               run_model_iterations=self._run_model_iterations, use_cuda=self._use_cuda,
                                               metric=self._metric, error_margin=error_margin,
                                               baseline_perf=baseline_accuracy, num_rank_indices=num_rank_indices,
                                               database=self._layer_database)
            compressed_model, stats = self._final_compressed_network(best_index, svd_rank_pair_dict_best_index,
                                                                     rank_stats_list)

        else:
            rank_stats_list, svd_rank_pair_dict = \
                rank_selector.split_manual_rank(model=self._model, run_model=self._run_model,
                                                run_model_iterations=self._run_model_iterations,
                                                use_cuda=self._use_cuda, metric=self._metric, **kw_args,
                                                database=self._layer_database)
            compressed_model, stats = self._final_compressed_network(0, svd_rank_pair_dict, rank_stats_list)
        return compressed_model, stats
