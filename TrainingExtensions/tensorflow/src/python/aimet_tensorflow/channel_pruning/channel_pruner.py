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

""" Prunes layers using Channel Pruning scheme """

from typing import List, Dict, Tuple, Set

import copy
import tensorflow as tf
import numpy as np

# Import aimet specific modules
from aimet_common.defs import CostMetric, LayerCompRatioPair
from aimet_common.utils import AimetLogger
from aimet_common.pruner import Pruner
from aimet_common.channel_pruner import select_channels_to_prune
from aimet_common.cost_calculator import CostCalculator, Cost
from aimet_common.winnow.winnow_utils import update_winnowed_channels

from aimet_tensorflow.utils.graph_saver import save_and_load_graph
from aimet_tensorflow.utils.common import is_op_compressible, get_ordered_ops
from aimet_tensorflow.layer_database import Layer, LayerDatabase
from aimet_tensorflow.utils.op.conv import WeightTensorUtils
from aimet_tensorflow.winnow import winnow
from aimet_tensorflow.channel_pruning.data_subsampler import DataSubSampler
from aimet_tensorflow.channel_pruning.weight_reconstruction import WeightReconstructor
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ChannelPruning)


class InputChannelPruner(Pruner):
    """
    Pruner for Channel Pruning method
    """

    def __init__(self, input_op_names: List[str], output_op_names: List[str], data_set: tf.data.Dataset,
                 batch_size: int, num_reconstruction_samples: int, allow_custom_downsample_ops: bool):
        """
        Input Channel Pruner with given dataset, input shape, number of batches and samples per image.

        :param input_op_names: list of input op names
        :param output_op_names: List of output op names of the model, used to help ConnectedGraph determine valid ops
        (to ignore training ops for example).
        :param data_set: data set to be used with the model
        :param batch_size: batch size
        :param num_reconstruction_samples: number of reconstruction samples
        :param allow_custom_downsample_ops: allow downsample/upsample ops to be inserted
        """
        self._input_op_names = input_op_names
        self._output_op_names = output_op_names
        self._data_set = data_set
        self._batch_size = batch_size
        self._num_reconstruction_samples = num_reconstruction_samples
        self._allow_custom_downsample_ops = allow_custom_downsample_ops

    @staticmethod
    def _select_inp_channels(layer: Layer, comp_ratio: float) -> list:
        """

        :param layer: layer for which input channels to prune are selected.
        :param comp_ratio: the ratio of costs after pruning has taken place
                           0 < comp_ratio <= 1.
        :return: prune_indices: list of input channels indices to prune.
        """

        assert layer.module.type == 'Conv2D'

        weight_index = WeightTensorUtils.get_tensor_index_in_given_op(layer.module)

        weight_tensor = layer.model.run(layer.module.inputs[weight_index])

        # Conv2d weight shape in TensorFlow  [kh, kw, Nic, Noc]
        # re order in the common shape  [Noc, Nic, kh, kw]
        weight_tensor = np.transpose(weight_tensor, (3, 2, 0, 1))

        num_in_channels = weight_tensor.shape[1]

        prune_indices = select_channels_to_prune(weight_tensor, comp_ratio, num_in_channels)

        return prune_indices

    def _data_subsample_and_reconstruction(self, orig_layer: Layer, pruned_layer: Layer, output_mask: List[int],
                                           orig_layer_db: LayerDatabase, comp_layer_db: LayerDatabase):
        """
        Collect and sub sampled output data from original layer and input data from pruned layer and set
        reconstructed weight and bias to pruned layer in compressed model database

        :param orig_layer: layer from original model
        :param pruned_layer: layer from potentially compressed model
        :param output_mask  : output mask that specifies certain output channels to remove
        :param orig_layer_db: original Layer database without any compression
        :param comp_layer_db: compressed Layer database
        :return:
        """

        sub_sampled_inp, sub_sampled_out = DataSubSampler.get_sub_sampled_data(orig_layer, pruned_layer,
                                                                               self._input_op_names, orig_layer_db,
                                                                               comp_layer_db, self._data_set,
                                                                               self._batch_size,
                                                                               self._num_reconstruction_samples)

        logger.debug("Input Data size: %s, Output data size: %s", len(sub_sampled_inp), len(sub_sampled_out))

        # update the weight and bias (if any) using sub sampled input and output data
        WeightReconstructor.reconstruct_params_for_conv2d(pruned_layer, sub_sampled_inp, sub_sampled_out, output_mask)

    def _sort_on_occurrence(self, sess: tf.compat.v1.Session, layer_comp_ratio_list: List[LayerCompRatioPair]) -> \
            List[LayerCompRatioPair]:
        """
        Function takes session and list of conv layer-comp ratio to sort, and sorts them based on
        occurrence in the model.

        :param sess: tf.compat.v1.Session
        :param layer_comp_ratio_list: layer compression ratio list
        :return: sorted_layer_comp_ratio_List
        """
        sorted_layer_comp_ratio_list = []

        ordered_ops = get_ordered_ops(graph=sess.graph, starting_op_names=self._input_op_names,
                                      output_op_names=self._output_op_names)

        for op in ordered_ops:

            if is_op_compressible(op):
                for pair in layer_comp_ratio_list:

                    if op.name == pair.layer.name:
                        sorted_layer_comp_ratio_list.append(LayerCompRatioPair(pair.layer, pair.comp_ratio))

        return sorted_layer_comp_ratio_list

    def _prune_layer(self, orig_layer_db: LayerDatabase, comp_layer_db: LayerDatabase,
                     layer: Layer, comp_ratio: float, cost_metric: CostMetric):

        pass

    def calculate_compressed_cost(self, layer_db: LayerDatabase,
                                  layer_comp_ratio_list: List[LayerCompRatioPair]) -> Cost:
        """
        Calculate cost of a compressed model given a set of layers and corresponding comp-ratios
        :param layer_db: Layer database for original model
        :param layer_comp_ratio_list: List of (layer + comp-ratio) pairs
        :return: Estimated cost of the compressed model
        """

        # sort all the layers in layer_comp_ratio_list based on occurrence
        layer_comp_ratio_list = self._sort_on_occurrence(layer_db.model, layer_comp_ratio_list)

        detached_op_names = set()

        # Copy the db
        comp_layer_db = copy.deepcopy(layer_db)
        current_sess = comp_layer_db.model

        for layer_comp_ratio in layer_comp_ratio_list:

            orig_layer = layer_db.find_layer_by_name(layer_comp_ratio.layer.name)
            comp_ratio = layer_comp_ratio.comp_ratio

            if comp_ratio is not None and comp_ratio < 1.0:

                # select input channels of conv2d op to winnow
                prune_indices = self._select_inp_channels(orig_layer, comp_ratio)
                if not prune_indices:
                    continue

                # Winnow the selected op and modify it's upstream affected ops
                current_sess, ordered_modules_list = winnow.winnow_tf_model(current_sess, self._input_op_names,
                                                                            self._output_op_names,
                                                                            [(orig_layer.module, prune_indices)],
                                                                            reshape=self._allow_custom_downsample_ops,
                                                                            in_place=True, verbose=False)
                if not ordered_modules_list:
                    continue

                # Get all the detached op names from updated session graph
                for orig_op_name, _, _, _ in ordered_modules_list:
                    detached_op_names.add(orig_op_name)

        # update layer database by excluding the detached ops
        comp_layer_db.update_database(current_sess, detached_op_names, update_model=False)

        # calculate the cost of this model
        compressed_model_cost = CostCalculator.compute_model_cost(comp_layer_db)

        # close the session associated with compressed layer database
        comp_layer_db.model.close()

        return compressed_model_cost

    def prune_model(self, layer_db: LayerDatabase, layer_comp_ratio_list: List[LayerCompRatioPair],
                    cost_metric: CostMetric, trainer):

        # sort all the layers in layer_comp_ratio_list based on occurrence
        layer_comp_ratio_list = self._sort_on_occurrence(layer_db.model, layer_comp_ratio_list)

        # Copy the db
        comp_layer_db = copy.deepcopy(layer_db)
        current_sess = comp_layer_db.model

        # Dictionary to map original layer name to list of most recent pruned layer name and output mask.
        # Masks remain at the original length and specify channels winnowed after each round of winnower.
        orig_layer_name_to_pruned_name_and_mask_dict = {}
        # Dictionary to map most recent pruned layer name to the original layer name
        pruned_name_to_orig_name_dict = {}
        # List to hold original layers to reconstruct
        layers_to_reconstruct = []
        detached_op_names = set()

        # Prune layers which have comp ratios less than 1
        for layer_comp_ratio in layer_comp_ratio_list:
            orig_layer = layer_db.find_layer_by_name(layer_comp_ratio.layer.name)
            if layer_comp_ratio.comp_ratio is not None and layer_comp_ratio.comp_ratio < 1.0:
                # 1) channel selection
                prune_indices = self._select_inp_channels(orig_layer, layer_comp_ratio.comp_ratio)
                if not prune_indices:
                    continue

                # 2) Winnowing the model
                current_sess, ordered_modules_list = winnow.winnow_tf_model(current_sess, self._input_op_names,
                                                                            self._output_op_names,
                                                                            [(orig_layer.module, prune_indices)],
                                                                            reshape=self._allow_custom_downsample_ops,
                                                                            in_place=True, verbose=False)
                if not ordered_modules_list:
                    continue

                layers_to_reconstruct.append(orig_layer)
                # Update dictionaries with new info about pruned ops and new masks
                self._update_pruned_ops_and_masks_info(ordered_modules_list,
                                                       orig_layer_name_to_pruned_name_and_mask_dict,
                                                       pruned_name_to_orig_name_dict,
                                                       detached_op_names)

        # Save and reload modified graph to allow changes to take effect
        # Need to initialize uninitialized variables first since only newly winnowed conv ops are initialized during
        # winnow_tf_model, and all other newly winnowed ops are not.
        with current_sess.graph.as_default():
            initialize_uninitialized_vars(current_sess)
        current_sess = save_and_load_graph('./saver', current_sess)
        comp_layer_db.update_database(current_sess, detached_op_names, update_model=True)

        # Perform reconstruction
        self._reconstruct_layers(layers_to_reconstruct, orig_layer_name_to_pruned_name_and_mask_dict, layer_db,
                                 comp_layer_db)

        return comp_layer_db

    @staticmethod
    def _update_pruned_ops_and_masks_info(
            ordered_modules_list: List[Tuple[str, tf.Operation, List[List[int]], List[List[int]]]],
            orig_layer_name_to_pruned_name_and_mask_dict: Dict[str, Tuple[str, List[int]]],
            pruned_name_to_orig_name_dict: Dict[str, str],
            detached_op_names: Set[str]):
        """
        Update dictionaries with information about newly winnowed ops and masks
        :param ordered_modules_list: Output of winnow_tf_model holding information on winnowed ops and masks
        :param orig_layer_name_to_pruned_name_and_mask_dict: Dictionary mapping original layer names to most recent
        pruned op name and most recent output masks.
        :param pruned_name_to_orig_name_dict: Dictionary mapping pruned layer names to original layer names (if a layer
        was winnowed in multiple rounds of winnow_tf_model, there may be multiple prined layer names mapping to the same
        original layer name)
        :param detached_op_names: Set holding names of operations which are detached due to winnowing and should not be
        used.
        """

        for prepruned_op_name, pruned_op, _, output_masks in ordered_modules_list:
            detached_op_names.add(prepruned_op_name)
            if pruned_op.type == 'Conv2D':      # Currently, we only care about tracking information about conv ops
                if prepruned_op_name in pruned_name_to_orig_name_dict:
                    # the op was already pruned once prior to this most recent round of winnowing
                    original_op_name = pruned_name_to_orig_name_dict[prepruned_op_name]

                    # Get and update previous pruned op name and output mask
                    _, running_output_mask = \
                        orig_layer_name_to_pruned_name_and_mask_dict.get(original_op_name, (None, None))
                    assert running_output_mask is not None
                    # Replace previous pruned op name with most recent pruned op name
                    # Update output mask
                    update_winnowed_channels(running_output_mask, output_masks[0])
                    orig_layer_name_to_pruned_name_and_mask_dict[original_op_name] = (pruned_op.name,
                                                                                      running_output_mask)
                else:
                    # This is the first time this op is being pruned
                    # The name should not show up in either dict
                    assert prepruned_op_name not in orig_layer_name_to_pruned_name_and_mask_dict
                    assert prepruned_op_name not in pruned_name_to_orig_name_dict

                    original_op_name = prepruned_op_name
                    # Add output channel mask info to layer_to_masks_dict
                    orig_layer_name_to_pruned_name_and_mask_dict[prepruned_op_name] = (pruned_op.name,
                                                                                       output_masks[0])

                # Map pruned op's name to original op name in pruned_to_orig_name_dict
                pruned_name_to_orig_name_dict[pruned_op.name] = original_op_name

    def _reconstruct_layers(self, layers_to_reconstruct: List[Layer],
                            orig_layer_name_to_pruned_name_and_mask_dict: Dict[str, Tuple[str, List[int]]],
                            layer_db: LayerDatabase, comp_layer_db: LayerDatabase):
        """
        Reconstruct weights and biases of layers in the layers_to_reconstruct list.
        :param layers_to_reconstruct: List of layers to reconstruct weights and biases of
        :param orig_layer_name_to_pruned_name_and_mask_dict: Dictionary mapping original layer names to most recent
        pruned op name and most recent output masks.
        :param layer_db: Original layer database
        :param comp_layer_db: Compressed layer database
        """
        for layer in layers_to_reconstruct:
            # Get output mask of layer, that contains information about all channels winnowed since the start
            pruned_layer_name, output_mask = \
                orig_layer_name_to_pruned_name_and_mask_dict.get(layer.name, (None, None))
            assert pruned_layer_name is not None

            pruned_layer = comp_layer_db.find_layer_by_name(pruned_layer_name)
            self._data_subsample_and_reconstruction(layer, pruned_layer, output_mask, layer_db, comp_layer_db)


class ChannelPruningCostCalculator(CostCalculator):
    """ Cost calculation utilities for Channel Pruning """

    def __init__(self, pruner: InputChannelPruner):
        self._pruner = pruner

    def calculate_compressed_cost(self, layer_db: LayerDatabase,
                                  layer_ratio_list: List[LayerCompRatioPair], cost_metric: CostMetric) -> Cost:
        """
        Calculate compressed cost of a model given a list of layer-compression-ratio pairs
        :param layer_db: Layer database for the original model
        :param layer_ratio_list: List of layer, compression-ratio
        :param cost_metric: Cost metric to use for compression (mac or memory)
        :return: Compressed cost
        """

        # Special logic for channel pruning - we first actually prune the model and then determine its cost
        # Because it is not easy to estimate it otherwise
        compressed_cost = self._pruner.calculate_compressed_cost(layer_db, layer_ratio_list)

        return compressed_cost
