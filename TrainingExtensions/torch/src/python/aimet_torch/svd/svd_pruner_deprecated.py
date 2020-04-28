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

"""Based on the scheme and rank splits a layer"""
import copy

from torch import nn

# Import AIMET specific modules
from aimet_torch.svd.svd_splitter import WeightSvdModuleSplitter
from aimet_torch import layer_database as lad
from aimet_torch.svd import model_stats_calculator as MS

from aimet_common import statistics_util as stats_u
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class ModelPruner:
    """
    Creates a compressed model by iterating over the selected layers in the model using the corresponding ranks
    """
    @classmethod
    def _copy_model(cls, model, model_layers):
        """
        Creates a copy of the original model and its layers
        :param model: the original model
        :param model_layers: original model's layers
        :return:
        """

        # Create a deep-copy of the model to return
        model_copy = copy.deepcopy(model)

        # Create an empty model_layers to fill
        model_layers_copy = {}

        modules_in_copy = list(model_copy.modules())

        # For all modules in the current model
        for index, module in enumerate(model.modules()):

            # If this module is included in the existing model_layers, we need to add a corresponding entry into
            # model_layers_copy
            if id(module) in model_layers:

                existing_layer = model_layers[id(module)]
                new_layer = lad.Layer(modules_in_copy[index], existing_layer.name,
                                      existing_layer.output_shape)
                new_layer.picked_for_compression = existing_layer.picked_for_compression
                model_layers_copy[id(modules_in_copy[index])] = new_layer

        # Now we need to set parent references
        lad.LayerDatabase.set_reference_to_parent_module(model_copy, model_layers_copy)

        return model_copy, model_layers_copy

    def create_compressed_model(self, svd_rank_pair_dict, model, compressible_layers, svd_lib_ref, metric):
        """
        Creates and returns a compressed model using self._model for the given rank index
        :param svd_rank_pair_dict: Rank index is the value and corresponding layer name is the key
        :param model: the original model
        :param compressible_layers: all the layers that can be compressed
        :param svd_lib_ref: Model Optimization library reference
        :param metric: the cost metric being used
        :return: Returns the compressed model and per_layer statistics list
        """
        # pylint: disable=too-many-locals

        # Create a copy of the model
        compressed_model, compressed_model_layers = self._copy_model(model, compressible_layers)

        # Layer attributes for compressed model
        # Start with all layers that are not selected for compression
        selected_layers = [layer for (key, layer) in compressed_model_layers.items()
                           if layer.picked_for_compression is True]
        compressed_model_layers = {key: value for (key, value) in compressed_model_layers.items()
                                   if value.picked_for_compression is False}

        # List to hold the SVD Statistics for each selected layer
        layer_stats_list = list()
        # Loop over all the selected layers
        for layer in selected_layers:
            svd_rank_pair = svd_rank_pair_dict[layer.name]
            # Split the layer
            sequential_of_split_layers, layer_a_attr, layer_b_attr = DeprecatedSvdPruner.prune_layer(layer, svd_rank_pair[0],
                                                                                                     svd_lib_ref)

            # Replace original layer with sequential of split layers
            setattr(layer.parent_module, layer.var_name_of_module_in_parent,
                    sequential_of_split_layers)

            # Add layer attributes for the split layers
            compressed_model_layers[id(layer_a_attr.module)] = layer_a_attr
            compressed_model_layers[id(layer_b_attr.module)] = layer_b_attr
            split_layers = list()
            split_layers.append(layer_a_attr)
            split_layers.append(layer_b_attr)
            ms = MS.ModelStats
            layer_compression_ratio = ms.compute_per_layer_compression_ratio(orig_layer=layer,
                                                                             split_layers=split_layers,
                                                                             metric=metric)

            per_layer = stats_u.SvdStatistics.PerSelectedLayer(name=layer.name, rank=svd_rank_pair[0],
                                                               compression_ratio=layer_compression_ratio)
            layer_stats_list.append(per_layer)
        return compressed_model, compressed_model_layers, layer_stats_list


class DeprecatedSvdPruner:
    """
    Splits layers based on SVD technique
    """
    @staticmethod
    def prune_layer(original_layer, rank, svd_lib_ref):
        """
        Splits a layer based on the splitting scheme
        :param original_layer: original layers attributes
        :param rank: rank pair for a given layer
        :param svd_lib_ref: Reference to Model optimization library
        :return:
        """
        # Delegate to the right method to split the layer
        if isinstance(original_layer.module, nn.Conv2d):
            module_a, module_b = WeightSvdModuleSplitter.split_conv_module(original_layer.module, original_layer.name,
                                                                           rank, svd_lib_ref)

        elif isinstance(original_layer.module, nn.Linear):
            module_a, module_b = WeightSvdModuleSplitter.split_fc_module(original_layer.module, original_layer.name,
                                                                         rank, svd_lib_ref)

        else:
            raise TypeError("Only Conv and FC layers are currently supported")

        # Create a sequential of the split layers
        seq = nn.Sequential(module_a, module_b)

        # layer_attr of split layers
        layer_a = lad.Layer(module_a, original_layer.name + '_a', original_layer.output_shape)
        layer_b = lad.Layer(module_b, original_layer.name + '_b', original_layer.output_shape)

        return seq, layer_a, layer_b
