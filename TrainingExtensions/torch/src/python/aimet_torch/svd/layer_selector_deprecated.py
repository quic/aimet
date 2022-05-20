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

"""Selects layers based on the layer selection scheme"""

from torch import nn

import aimet_common.libpymo as pymo
from aimet_common.utils import AimetLogger
from aimet_common import cost_calculator as cc
from aimet_torch.svd.svd_intf_defs_deprecated import LayerSelectionScheme, CostMetric

_MIN_LAYER_DIM_FOR_SVD = 10

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class LayerSelectorDeprecated:
    """
    Performs layer selection and marks the corresponding layer which are picked in the Layer Attribute Database
    """
    def __init__(self, layer_selection_scheme, metric, layer_database, **kw_layer_select_params):
        self._layer_selection_scheme = layer_selection_scheme
        self._metric = metric
        self._svd_lib_ref = pymo.GetSVDInstance()
        self._layer_database = layer_database
        if layer_selection_scheme is LayerSelectionScheme.manual:
            self._layers_to_compress = kw_layer_select_params['layers_to_compress']

        elif layer_selection_scheme is LayerSelectionScheme.top_n_layers:
            self._num_layers = kw_layer_select_params['num_layers']

        elif layer_selection_scheme is LayerSelectionScheme.top_x_percent:
            self._layer_selection_threshold = kw_layer_select_params['percent_thresh']

        else:
            raise ValueError("Unsupported layer_selection_scheme: {}".format(layer_selection_scheme))

        # Select layers to compress
        # ------------------------------
        logger.info("Starting layer selection..")
        self._perform_layer_selection()

        logger.info("Layer selection complete")

    @staticmethod
    def _check_layer_with_smaller_dimensions(model_module):
        weight_shape = (list(model_module.weight.size()))
        # Filter layers with dimensions smaller than
        # a certain low value as doing rank-based truncation
        # on the smaller dimensions does significant harm
        # to the final network accuracy.
        input_channels, output_channels, k_h, k_w = weight_shape[1], weight_shape[0], 1, 1
        if isinstance(model_module, nn.Conv2d):
            k_h, k_w = weight_shape[2], weight_shape[3]

        # the 2D column equivalent of output_channels * k_h * k_w
        return not ((input_channels < _MIN_LAYER_DIM_FOR_SVD) or \
                        (output_channels * k_h * k_w < _MIN_LAYER_DIM_FOR_SVD))

    def _pick_compression_layers(self, cost_metric, layer_select_scheme, **kwargs):
        """
        Function to pick top N layer based on selection threshold provided by user and then
        store layer attributes for MO(ModelOptimization) like shape of weight matrix, activation dimensions,
         weights and bias etc.
        :param run_model: The function to use for running data through the graph to calculate input and output shape
        of layer (activation dimensions). This function will be used with the custom hook to feed 1 iteration of
        data.
        """
        # pylint: disable=too-many-locals, too-many-branches

        # Sanity check
        if not isinstance(cost_metric, CostMetric):
            raise TypeError("cost_metric is not of type CostMetric")

        if not isinstance(layer_select_scheme, LayerSelectionScheme):
            raise TypeError("layer_selection_scheme is not of type Svd.LayerSelectionScheme")

        # register custom hook for the model with run_graph provided by user
        # if the user wants to experiment with custom hook, we can support that option by
        # exposing the hook parameter to compress_net method
        pruned_list = []
        # cache the layer attributes list for further processing
        for layer in self._layer_database:
            # Heuristic1: Reject any ops whose param shape does not meet a base criterion
            if self._check_layer_with_smaller_dimensions(layer.module):
                pruned_list.append(layer)
            else:
                logger.debug("Pruning out %r: shape is %r", layer.module,
                             layer.module.weight.size())

        # Reset list of layers for the next phase
        layers = pruned_list
        pruned_list = []

        # Create a list of layer, cost tuples
        layer_cost_pairs = []
        for layer in layers:
            cost = cc.CostCalculator.compute_layer_cost(layer)
            layer_cost_pairs.append((layer, cost))

        # Sort list of layer-cost pairs
        if cost_metric == CostMetric.memory:
            layer_cost_pairs.sort(key=lambda x: x[1].memory, reverse=True)
        else:
            layer_cost_pairs.sort(key=lambda x: x[1].mac, reverse=True)

        if layer_select_scheme == LayerSelectionScheme.top_n_layers:
            num_layers = kwargs['num_layers']
            pruned_list_of_pairs = layer_cost_pairs[:num_layers]
            pruned_list = [pair[0] for pair in pruned_list_of_pairs]

        elif layer_select_scheme == LayerSelectionScheme.top_x_percent:
            percent_thresh = kwargs['percent_thresh']
            # get the network cost for Memory and MAC
            cost_calc = cc.CostCalculator()
            network_cost = cost_calc.compute_model_cost(self._layer_database)
            network_cost = network_cost.memory if cost_metric == CostMetric.memory else network_cost.mac
            accum_cost = 0.
            logger.debug("Total network cost: %f", network_cost)
            logger.debug("Picking layers contributing to : %f (percent) of total network cost.", percent_thresh)

            for layer, cost in layer_cost_pairs:
                cost = cost.memory if (cost_metric == CostMetric.memory) else cost.mac

                if (100 * (cost + accum_cost)/network_cost) < percent_thresh:
                    accum_cost += cost
                    pruned_list.append(layer)
                    logger.debug("Layer Picked : %s with cost : %f", layer.module, cost)
                    logger.debug("Accumulated cost so far : %f", accum_cost)

        elif layer_select_scheme == LayerSelectionScheme.manual:
            layers_to_compress = kwargs['layers_to_compress']
            for layer, _ in layer_cost_pairs:
                if layer.module in layers_to_compress:
                    pruned_list.append(layer)

        if not pruned_list:
            raise RuntimeError('No suitable layers found in the model.')

        return pruned_list

    def _perform_layer_selection(self):

        # Layer-selection
        if self._layer_selection_scheme is LayerSelectionScheme.manual:
            self._selected_layers = self._pick_compression_layers(self._metric,
                                                                  LayerSelectionScheme.manual,
                                                                  layers_to_compress=self._layers_to_compress)

        elif self._layer_selection_scheme is LayerSelectionScheme.top_n_layers:
            self._selected_layers = self._pick_compression_layers(self._metric,
                                                                  LayerSelectionScheme.top_n_layers,
                                                                  num_layers=self._num_layers)

        else:
            percent_thresh = self._layer_selection_threshold
            self._selected_layers = self._pick_compression_layers(self._metric,
                                                                  LayerSelectionScheme.top_x_percent,
                                                                  percent_thresh=percent_thresh)

        self._layer_database.mark_picked_layers(self._selected_layers)
