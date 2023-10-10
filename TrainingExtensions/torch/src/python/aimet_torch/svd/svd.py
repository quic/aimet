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

""" Interface of the AIMET SVD model-compression feature """

# Import AIMET specific modules
from aimet_torch.svd import svd_impl
from aimet_torch.svd.svd_intf_defs_deprecated import LayerSelectionScheme, RankSelectionScheme


class Svd:
    """ Top-level SVD interface class to be invoked by AIMET users """

    @staticmethod
    def _check_params_and_throw(kw_args, expected_params, not_expected_params):
        for param in expected_params:
            if param not in kw_args:
                raise ValueError("Expected param: {} is missing".format(param))

        for param in not_expected_params:
            if param in kw_args:
                raise ValueError("Unexpected param: {} found".format(param))

    @staticmethod
    def _validate_layer_rank_params(model, layer_selection_scheme, rank_selection_scheme, **kwargs):
        """ Validates the Layer Selection and Rank Selection parameters passed to the compress_model() function.

            Args:
                layer_selection_scheme (required): Enum argument. Options available: manual, top_n_layers, top_x_percent.
                rank_selection_scheme (required):  Enum argument. Options available: manual, auto
                **kwargs (required): The Layer Selection and Rank Selection parameters.
            Raises:
                ValueError: When an invalid parameter is passed.

        """

        # Validate the Layer Selection parameters
        if layer_selection_scheme == LayerSelectionScheme.manual and rank_selection_scheme == rank_selection_scheme.auto:
            Svd._check_params_and_throw(kwargs,
                                        ['layers_to_compress'],
                                        ['num_layers', 'percent_thresh'])

        if layer_selection_scheme == LayerSelectionScheme.top_n_layers:
            Svd._check_params_and_throw(kwargs,
                                        ['num_layers'],
                                        ['percent_thresh', 'layers_to_compress'])

            num_layers = kwargs['num_layers']

            # modules() always returns the model itself as the first iterable entry
            num_modules_in_model = sum(1 for _ in model.modules()) - 1
            if num_layers < 1 or num_layers > num_modules_in_model:
                raise ValueError("KW argument num_layers: {} out-of-range".format(num_layers))

        if layer_selection_scheme == LayerSelectionScheme.top_x_percent:
            Svd._check_params_and_throw(kwargs,
                                        ['percent_thresh'],
                                        ['num_layers', 'layers_to_compress'])
            percent = kwargs['percent_thresh']
            if percent < 0 or percent > 100:
                raise ValueError("KW argument percent_thresh: {} out-of-range".format(percent))

        # Validate the Rank Selection parameters
        if rank_selection_scheme == RankSelectionScheme.manual:
            Svd._check_params_and_throw(kwargs,
                                        ['layer_rank_list'],
                                        ['error_margin', 'num_rank_indices'])

        if rank_selection_scheme == RankSelectionScheme.auto:
            Svd._check_params_and_throw(kwargs,
                                        ['error_margin', 'num_rank_indices'],
                                        ['layer_rank_list'])

    @staticmethod
    def compress_model(model, run_model, run_model_iterations, input_shape,
                       compression_type, cost_metric, layer_selection_scheme,
                       rank_selection_scheme, **kw_layer_rank_params):
        """
        Runs rank selection on the model, and compresses it using the method and parameters provided

        :param model: The model which needs to be compressed
        :param run_model: The evaluation function that needs to be passed for one forward pass
        :param run_model_iterations: The number of iterations of forward pass for the run_model
        :param input_shape: Shape of the input to the model
        :param compression_type: Enum argument. Options available: svd , ssvd.
        :param cost_metric: Enum argument. Options available: mac, memory
        :param layer_selection_scheme: Enum argument. Options available: manual, top_n_layers, top_x_percent
        :param rank_selection_scheme: Enum argument. Options available: manual, auto
        :param kw_layer_rank_params: Params for layer and rank selection. Params depend on modes selected
        :return: compressed model and Model statistics

        **Note regarding kw_layer_rank_params**:
         - If the layer_selection_scheme is manual then user has to specify the list of layers by using- layers_to_compress= [list of layers],
         - If the layer_selection_scheme is top_n_layers then the user has to specify the number of layers as num_layers= <number>
         - If the layer_selection_scheme is top_x_percent then the user has to specify percentage threshold by using percent_thresh= <number>
         - If the mode is manual then user has to specify the layers and the respective ranks by specifying a list as layer_rank = [[layer, rank]]
         - If the mode is auto then user has to specify maximum rank till the optimum rank search has to happen as max_ranks_error_margin= [maximum rank, error margin]

        """
        Svd._validate_layer_rank_params(model, layer_selection_scheme, rank_selection_scheme, **kw_layer_rank_params)

        # Sanity check for run_model_iterations
        if run_model_iterations <= 0:
            raise ValueError("run_model_iterations: {} unexpected value. "
                             "Expect at least 1 iteration".format(run_model_iterations))

        # Instantiate the SVD impl class
        if rank_selection_scheme == rank_selection_scheme.auto:
            svd_obj = svd_impl.SvdImpl(model, run_model, run_model_iterations, input_shape,
                                       compression_type, cost_metric,
                                       layer_selection_scheme,
                                       **kw_layer_rank_params)
            compressed_model, stats = svd_obj.compress_net(rank_selection_scheme=rank_selection_scheme,
                                                           **kw_layer_rank_params)

        elif rank_selection_scheme == rank_selection_scheme.manual:
            layers_to_compress = [layer for layer, _ in kw_layer_rank_params['layer_rank_list']]
            svd_obj = svd_impl.SvdImpl(model, run_model, run_model_iterations, input_shape,
                                       compression_type, cost_metric,
                                       LayerSelectionScheme.manual,
                                       layers_to_compress=layers_to_compress)
            compressed_model, stats = svd_obj.compress_net(rank_selection_scheme=rank_selection_scheme,
                                                           **kw_layer_rank_params)
        return compressed_model, stats
