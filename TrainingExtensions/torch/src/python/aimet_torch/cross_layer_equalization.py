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

""" Cross Layer Equalization

Some terminology for this code.
CLS set: Set of layers (2 or 3) that can be used for cross-layer scaling
Layer groups: Groups of layers that are immediately connected and can be decomposed further into CLS sets
"""

from typing import Tuple, List, Union, Dict
from enum import Enum
import numpy as np
import torch

import aimet_common.libpymo as libpymo      # pylint: disable=import-error

from aimet_common.utils import AimetLogger
from aimet_torch import utils
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.utils import get_device, get_ordered_list_of_modules, create_rand_tensors_given_shapes

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


ClsSet = Union[Tuple[torch.nn.Conv2d, torch.nn.Conv2d],
               Tuple[torch.nn.Conv2d, torch.nn.Conv2d, torch.nn.Conv2d]]

ClsSupportedLayer = Union[torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d]

ScaleFactor = Union[np.ndarray, Tuple[np.ndarray]]

cls_supported_layers = (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Conv1d, torch.nn.ConvTranspose1d)
cls_supported_activations = (torch.nn.ReLU, torch.nn.PReLU)


class ClsLayerType(Enum):
    """Enum class to represent CLS layer types"""
    Unsupported = 0
    Conv = 1  # Overloaded for conv and ConvTranspose
    DepthwiseConv = 2


class ClsSetInfo:
    """
    This class hold information about the layers in a CLS set, along with corresponding scaling factors
    and other information like if there is a ReLU activation function between the CLS set layers
    """

    class ClsSetLayerPairInfo:
        """
        Models a pair of layers that were scaled using CLS. And related information.
        """

        def __init__(self, layer1: torch.nn.Conv2d, layer2: torch.nn.Conv2d, scale_factor: np.ndarray,
                     relu_activation_between_layers: bool):
            """
            :param layer1: Layer whose bias is folded
            :param layer2: Layer to which bias of previous layer's bias is folded
            :param scale_factor: Scale Factor found from Cross Layer Scaling to scale BN parameters
            :param relu_activation_between_layers: If the activation between layer1 and layer2 is Relu
            """
            self.layer1 = layer1
            self.layer2 = layer2
            self.scale_factor = scale_factor
            self.relu_activation_between_layers = relu_activation_between_layers

    def __init__(self, cls_pair_1: ClsSetLayerPairInfo, cls_pair_2: ClsSetLayerPairInfo = None):
        """
        Constructor takes 2 pairs if Depth-wise separable layer is being folded

        :param cls_pair_1: Pair between two conv or conv and depth-wise conv
        :param cls_pair_2: Pair between depth-wise conv and point-wise conv
        """
        if cls_pair_2:
            self.cls_pair_info_list = [cls_pair_1, cls_pair_2]
        else:
            self.cls_pair_info_list = [cls_pair_1]


def get_ordered_list_of_conv_modules(model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple]) -> List:
    """
    Finds order of nodes in graph
    :param model: model
    :param dummy_input: Dummy input to the model. Used to parse model graph.
    :return: List of names in graph in order
    """
    module_list = get_ordered_list_of_modules(model, dummy_input)
    module_list = [[name, module] for name, module in module_list if isinstance(module, cls_supported_layers)]
    return module_list


class GraphSearchUtils:
    """
    Code to search a model graph to find nodes to use for cross-layer-scaling and high-bias-fold
    """

    def __init__(self, model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]], dummy_input: Union[torch.Tensor, List[torch.Tensor]] = None):

        if dummy_input is None:
            inp_tensor_list = tuple(utils.create_rand_tensors_given_shapes(input_shapes, get_device(model)))
        else:
            inp_tensor_list = dummy_input
        self._connected_graph = ConnectedGraph(model, inp_tensor_list)
        self._ordered_module_list = get_ordered_list_of_conv_modules(model, inp_tensor_list)


    @staticmethod
    def find_downstream_layer_groups_to_scale(op, layer_groups, current_group=None, visited_nodes=None):
        """
        Recursive function to find cls layer groups downstream from a given op
        :param op: Starting op to search from
        :param layer_groups: Running list of layer groups
        :param current_group: Running current layer group
        :param visited_nodes: Running list of visited nodes (to short-circuit recursion)
        :return: None
        """

        if not visited_nodes:
            visited_nodes = []
        if not current_group:
            current_group = []

        if op in visited_nodes:
            return
        visited_nodes.append(op)
        # print("Visiting node: {}".format(op.dotted_name))

        # If current node is Conv2D, add to the current group
        if op.model_module and isinstance(op.model_module.get_module(), cls_supported_layers):
            current_group.append(op.model_module.get_module())

        # Terminating condition for current group
        if not op.model_module or not isinstance(op.model_module.get_module(),
                                                 cls_supported_layers + cls_supported_activations):

            if (len(current_group) > 1) and (current_group not in layer_groups):
                layer_groups.append(current_group)
            current_group = []

        if op.output:
            for consumer in op.output.consumers:
                GraphSearchUtils.find_downstream_layer_groups_to_scale(consumer, layer_groups,
                                                                       current_group, visited_nodes)

        # Reached a leaf.. See if the current group has something to grab
        if (len(current_group) > 1) and (current_group not in layer_groups):
            layer_groups.append(current_group)

    @staticmethod
    def convert_layer_group_to_cls_sets(layer_group):
        """
        Helper function to convert a layer group to a list of cls sets
        :param layer_group: Given layer group to generate cls sets
        :return: List of cls sets

        Supported layer combinations for CLS are:
        1. Conv + Conv
        2. DepthwiseConv + Conv
        3. Conv + DepthwiseConv + Conv

        Can be rewritten as,
        Conv
            -> Conv
            -> DepthwiseConv
                -> Conv
        DepthwiseConv
            -> Conv

        If a combination is partially supported, the cls_set is completely omitted and restarted from the next
        supported layer
        For example: Consider Conv + DepthwiseConv + Depthwise(unsupported)
        - Since Depthwise(unsupported) is the last layer encountered, we need to omit all the three layers and restart
        the cls sets from the next supported layer.

        """

        # pylint: disable=too-many-branches
        def convert_to_cls_layer_type(layer: ClsSupportedLayer) -> Tuple[ClsLayerType, ClsSupportedLayer]:
            """
            Given the layer, check if its supported in CLS
            :param layer: layer to check
            :return: Tuple of ClsLayerType and the layer
            """
            if layer.groups == 1:
                layer_type = ClsLayerType.Conv
            elif layer.groups == layer.in_channels and layer.in_channels == layer.out_channels:
                # depthwiseConv layer with depth multiplier = 1
                layer_type = ClsLayerType.DepthwiseConv
            else:
                layer_type = ClsLayerType.Unsupported
            return layer_type, layer

        def get_next_layer() -> Union[Tuple[ClsLayerType, Union[ClsSupportedLayer, None]]]:
            """
            :return: Tuple of ClsLayerType and the next layer in layer_group
            """
            if not layer_group:
                return ClsLayerType.Unsupported, None
            layer = layer_group.pop(0)
            return convert_to_cls_layer_type(layer)

        cls_sets = []

        first_layer_to_scale = (ClsLayerType.Unsupported, None)
        while layer_group:
            while layer_group and first_layer_to_scale[0] is ClsLayerType.Unsupported:
                first_layer_to_scale = get_next_layer()
                if first_layer_to_scale[0] is ClsLayerType.Unsupported:
                    logger.info('Layer %s is not supported. Ignoring for cls', first_layer_to_scale[1])

            second_layer_to_scale = get_next_layer()
            if first_layer_to_scale[0] == ClsLayerType.Conv:
                if second_layer_to_scale[0] == ClsLayerType.Conv:
                    cls_sets.append((first_layer_to_scale[1], second_layer_to_scale[1]))
                    first_layer_to_scale = second_layer_to_scale
                elif second_layer_to_scale[0] == ClsLayerType.DepthwiseConv:
                    if layer_group:
                        # do not pop third layer yet, determine its type and then pop it
                        third_layer_to_scale = convert_to_cls_layer_type(layer_group[0])
                        if third_layer_to_scale[0] == ClsLayerType.Conv:
                            cls_sets.append(
                                (first_layer_to_scale[1], second_layer_to_scale[1], third_layer_to_scale[1]))
                            # adding third_layer_to_scale for the next round of CLS set determination
                            first_layer_to_scale = get_next_layer()
                        else:
                            # unsupported combination encountered
                            first_layer_to_scale = second_layer_to_scale
                else:
                    logger.info('Layer %s is not supported. Ignoring for cls', second_layer_to_scale[1])
                    first_layer_to_scale = (ClsLayerType.Unsupported, None)
            elif first_layer_to_scale[0] == ClsLayerType.DepthwiseConv:
                if second_layer_to_scale[0] == ClsLayerType.Conv:
                    cls_sets.append((first_layer_to_scale[1], second_layer_to_scale[1]))
                first_layer_to_scale = second_layer_to_scale
            else:
                logger.info('Layer %s is not supported. Ignoring for cls', first_layer_to_scale[1])
                first_layer_to_scale = second_layer_to_scale

        return cls_sets

    def find_layer_groups_to_scale(self) -> List[List[torch.nn.Conv2d]]:
        """
        :return: List of groups of layers. Each group can be independently equalized
        """

        # Find the input node(s) in the graph
        input_nodes = []
        for op in self._connected_graph.get_all_ops().values():
            if op.inputs and op.inputs[0].is_model_input:
                input_nodes.append(op)

        layer_groups = []
        for op in input_nodes:
            self.find_downstream_layer_groups_to_scale(op, layer_groups)

        # Sort the layer groups in order of occurrence in the model
        ordered_layer_groups = []
        for _, module in self._ordered_module_list:
            for layer_group in layer_groups:
                if layer_group[0] is module:
                    ordered_layer_groups.append(layer_group)

        return ordered_layer_groups

    @staticmethod
    def does_module_have_relu_activation(connected_graph: ConnectedGraph, module: torch.nn.Module) -> bool:
        """
        Finds if a given module has a ReLU activation
        :param connected_graph: Reference to ConnectedGraph instance
        :param module: PyTorch module to find activation for
        :return: True if module has a relu activation
        """

        for op in connected_graph.get_all_ops().values():

            if op.model_module and op.model_module.get_module() is module:
                assert len(op.output.consumers) == 1
                is_relu_activation = isinstance(op.output.consumers[0].model_module.get_module(),
                                                (torch.nn.ReLU, torch.nn.PReLU))
                return is_relu_activation

        return False

    def is_relu_activation_present_in_cls_sets(self, cls_sets: List[ClsSet]):
        """
        :param cls_sets: CLS sets to find relu activations in
        :return: List of groups of layers. Each group can be independently equalized
        """

        is_relu_activation_in_cls_sets = []
        for cls_set in cls_sets:

            # We need to check activation functions for all layers but the last one in the set
            # Because we are only interested in checking activation functions between the layers we will scale
            cls_set = cls_set[:-1]

            is_relu_activation_in_cls_set = ()
            for module in cls_set:
                is_relu_activation_in_cls_set += (self.does_module_have_relu_activation(self._connected_graph,
                                                                                        module), )

            if len(is_relu_activation_in_cls_set) == 1:
                is_relu_activation_in_cls_set = is_relu_activation_in_cls_set[0]

            is_relu_activation_in_cls_sets.append(is_relu_activation_in_cls_set)

        return is_relu_activation_in_cls_sets


class CrossLayerScaling:
    """
    Code to apply the cross-layer-scaling technique to a model
    """

    @staticmethod
    def scale_cls_sets(cls_sets: List[ClsSet]) -> List[ScaleFactor]:
        """
        Scale multiple CLS sets

        :param cls_sets: List of CLS sets
        :return: Scaling factors calculated and applied for each CLS set in order
        """
        scale_factor_list = []
        for cls_set in cls_sets:
            scale_factor = CrossLayerScaling.scale_cls_set(cls_set)
            scale_factor_list.append(scale_factor)
        return scale_factor_list

    @staticmethod
    def scale_cls_set(cls_set: ClsSet) -> ScaleFactor:
        """
        Scale a CLS set
        :param cls_set: Either a pair or regular conv layers or a triplet of depthwise separable layers
        :return: Scaling factor calculated and applied
        """
        if len(cls_set) == 3:
            scale_factor = CrossLayerScaling.scale_cls_set_with_depthwise_layers(cls_set)
        else:
            scale_factor = CrossLayerScaling.scale_cls_set_with_conv_layers(cls_set)

        return scale_factor

    @classmethod
    def scale_cls_set_with_conv_layers(cls, cls_set: ClsSet) -> np.ndarray:
        """
        API to invoke equalize layer params (update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized
        :return: Scaling factor S_12 for each conv layer pair: numpy array
        """
        on_gpu = False
        for module in cls_set:
            if not isinstance(module, cls_supported_layers):
                raise ValueError("Only Conv or Transposed Conv layers are supported for cross layer equalization")
            if module.weight.is_cuda:
                on_gpu = True
                module.cpu()

        # Create structs for holding layer weights and bias parameters
        prev_layer_params = libpymo.EqualizationParams()
        curr_layer_params = libpymo.EqualizationParams()

        # Prepare and pack data structures for cls set.
        cls._pack_params_for_conv(cls_set, prev_layer_params, curr_layer_params)

        # Scales weights and bias for consecutive layers and updates data structures in-place.
        scaling_factor = libpymo.scaleLayerParams(prev_layer_params, curr_layer_params)

        # Update weight and biases for cls set using updated data structures.
        cls._update_params_for_conv(cls_set, prev_layer_params, curr_layer_params)

        if on_gpu:
            for module in cls_set:
                module.to(device="cuda")

        return scaling_factor

    @classmethod
    def scale_cls_set_with_depthwise_layers(cls, cls_set: ClsSet) -> [np.ndarray, np.ndarray]:
        """
        API to invoke equalize layer params for depth wise separable layers(update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers whose weights and biases need to be equalized.
                        Second Conv layer is a depth-wise conv and third conv layer is point-wise conv
        :return: Scaling factors S_12 and S_23 : numpy arrays
        """
        on_gpu = False
        for module in cls_set:
            if not isinstance(module, cls_supported_layers):
                raise ValueError("Only conv layers are supported for cross layer equalization")
            if module.weight.is_cuda:
                on_gpu = True
                module.cpu()

        # Create structs for holding layer weights and bias parameters
        prev_layer_params = libpymo.EqualizationParams()
        curr_layer_params = libpymo.EqualizationParams()
        next_layer_params = libpymo.EqualizationParams()

        # Prepare and pack data structures for cls set.
        cls._pack_params_for_depthwise_conv(cls_set, prev_layer_params, curr_layer_params, next_layer_params)

        # Scales weights and bias for consecutive layers and updates data structures in-place.
        scaling_params = libpymo.scaleDepthWiseSeparableLayer(prev_layer_params, curr_layer_params, next_layer_params)

        # Update weight and biases for cls set using updated data structures.
        cls._update_params_for_depthwise_conv(cls_set, prev_layer_params, curr_layer_params, next_layer_params)

        if on_gpu:
            for module in cls_set:
                module.to(device="cuda")

        return scaling_params.scalingMatrix12, scaling_params.scalingMatrix23

    @staticmethod
    def create_cls_set_info_list(cls_sets: List[ClsSet], scale_factors: List[ScaleFactor],
                                 is_relu_activation_in_cls_sets):
        """
        Binds information from there separate lists into one [ClsInfoSet] data-structure
        :param cls_sets: List of CLS sets
        :param scale_factors: Scale-factors for each cls-set
        :param is_relu_activation_in_cls_sets: Information if there is relu activation in each cls-set
        :return: List of ClsSetInfo
        """
        cls_set_info_list = []
        assert len(cls_sets) == len(scale_factors) == len(is_relu_activation_in_cls_sets)

        for index, cls_set in enumerate(cls_sets):

            if isinstance(scale_factors[index], tuple):
                # If we are dealing with a triplet of layers, then we should have 2 scale factors and 2 relu flags
                # Assert that this is true
                assert len(cls_set) == 3
                assert len(scale_factors[index]) == len(is_relu_activation_in_cls_sets[index]) == 2

                cls_pair_1 = ClsSetInfo.ClsSetLayerPairInfo(cls_set[0], cls_set[1], scale_factors[index][0],
                                                            is_relu_activation_in_cls_sets[index][0])
                cls_pair_2 = ClsSetInfo.ClsSetLayerPairInfo(cls_set[1], cls_set[2], scale_factors[index][1],
                                                            is_relu_activation_in_cls_sets[index][1])

                cls_set_info = ClsSetInfo(cls_pair_1, cls_pair_2)

            else:
                cls_pair = ClsSetInfo.ClsSetLayerPairInfo(cls_set[0], cls_set[1], scale_factors[index],
                                                          is_relu_activation_in_cls_sets[index])

                cls_set_info = ClsSetInfo(cls_pair)

            cls_set_info_list.append(cls_set_info)

        return cls_set_info_list

    @staticmethod
    def scale_model(model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]], dummy_input: Union[torch.Tensor, List[torch.Tensor]] = None) -> List[ClsSetInfo]:
        """
        Uses cross-layer scaling to scale all applicable layers in the given model

        :param model: Model to scale
        :param input_shapes: Input shape for the model (can be one or multiple inputs)
        :param dummy_input: Dummy input to the model. Used to parse model graph. User is expected to place the tensors on the appropriate device.
        :return: CLS information for each CLS set
        """
        if isinstance(model, torch.nn.DataParallel):
            return CrossLayerScaling.scale_model(model.module, input_shapes, dummy_input=dummy_input)
        device = get_device(model)
        model.cpu()

        # Find layer groups
        graph_search = GraphSearchUtils(model, input_shapes, dummy_input=dummy_input)
        layer_groups = graph_search.find_layer_groups_to_scale()

        # Find cls sets from the layer groups
        cls_sets = []
        for layer_group in layer_groups:
            cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
            cls_sets += cls_set

        # Scale the CLS sets
        scale_factors = CrossLayerScaling.scale_cls_sets(cls_sets)

        # Find if there were relu activations between layers of each cls set
        is_relu_activation_in_cls_sets = graph_search.is_relu_activation_present_in_cls_sets(cls_sets)

        # Convert to a list of cls-set-info elements
        cls_set_info_list = CrossLayerScaling.create_cls_set_info_list(cls_sets, scale_factors,
                                                                       is_relu_activation_in_cls_sets)

        model.to(device=device)
        return cls_set_info_list

    @staticmethod
    def _pack_params_for_conv(cls_set: ClsSet,
                              prev_layer_params: libpymo.EqualizationParams,
                              curr_layer_params: libpymo.EqualizationParams):
        """
        Prepare and pack data structure for previous and current layer in given cls set.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        """
        weight_set_0 = cls_set[0].weight

        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        if isinstance(cls_set[0], torch.nn.ConvTranspose2d):
            weight_set_0 = weight_set_0.permute(1, 0, 2, 3)
        if isinstance(cls_set[0], torch.nn.ConvTranspose1d):
            weight_set_0 = weight_set_0.permute(1, 0, 2)

        prev_layer_params.weight = weight_set_0.detach().numpy().reshape(-1)
        prev_layer_params.weightShape = np.array(weight_set_0.shape)
        if len(prev_layer_params.weightShape) == 3:
            prev_layer_params.weightShape = prev_layer_params.weightShape + [1]

        weight_set_1 = cls_set[1].weight

        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        if isinstance(cls_set[1], torch.nn.ConvTranspose2d):
            weight_set_1 = weight_set_1.permute(1, 0, 2, 3)
        if isinstance(cls_set[1], torch.nn.ConvTranspose1d):
            weight_set_1 = weight_set_1.permute(1, 0, 2)

        curr_layer_params.weight = weight_set_1.detach().numpy().reshape(-1)
        curr_layer_params.weightShape = np.array(weight_set_1.shape)
        if len(curr_layer_params.weightShape) == 3:
            curr_layer_params.weightShape = curr_layer_params.weightShape + [1]

        if cls_set[0].bias is not None:
            prev_layer_params.bias = cls_set[0].bias.detach().numpy()
        else:
            prev_layer_params.isBiasNone = True

    @staticmethod
    def _update_params_for_conv(cls_set: ClsSet,
                                prev_layer_params: libpymo.EqualizationParams,
                                curr_layer_params: libpymo.EqualizationParams):
        """
        Update weight and biases for cls set using updated data structures.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        """
        if isinstance(cls_set[0], (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            prev_layer_params.weightShape = prev_layer_params.weightShape[:-1]
        cls_set[0].weight.data = torch.from_numpy(np.reshape(prev_layer_params.weight,
                                                             prev_layer_params.weightShape))
        cls_set[0].weight.data = cls_set[0].weight.data.type(torch.FloatTensor)


        # Transpose weight back to N, C, H, W for transposed Conv2D
        if isinstance(cls_set[0], torch.nn.ConvTranspose2d):
            cls_set[0].weight.data = cls_set[0].weight.data.permute(1, 0, 2, 3).contiguous()
        if isinstance(cls_set[0], torch.nn.ConvTranspose1d):
            cls_set[0].weight.data = cls_set[0].weight.data.permute(1, 0, 2).contiguous()

        if isinstance(cls_set[1], (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            curr_layer_params.weightShape = curr_layer_params.weightShape[:-1]
        cls_set[1].weight.data = torch.from_numpy(np.reshape(curr_layer_params.weight,
                                                             curr_layer_params.weightShape))
        cls_set[1].weight.data = cls_set[1].weight.data.type(torch.FloatTensor)

        # Transpose weight back to N, C, H, W for transposed Conv2D
        if isinstance(cls_set[1], torch.nn.ConvTranspose2d):
            cls_set[1].weight.data = cls_set[1].weight.data.permute(1, 0, 2, 3).contiguous()
        if isinstance(cls_set[1], torch.nn.ConvTranspose1d):
            cls_set[1].weight.data = cls_set[1].weight.data.permute(1, 0, 2).contiguous()

        if cls_set[0].bias is not None:
            cls_set[0].bias.data = torch.from_numpy(np.reshape(prev_layer_params.bias,
                                                               prev_layer_params.weightShape[0]))
            cls_set[0].bias.data = cls_set[0].bias.data.type(torch.FloatTensor)

    @staticmethod
    def _pack_params_for_depthwise_conv(cls_set: ClsSet,
                                        prev_layer_params: libpymo.EqualizationParams,
                                        curr_layer_params: libpymo.EqualizationParams,
                                        next_layer_params: libpymo.EqualizationParams):
        """
        Prepare and pack data structure for previous, current and next layer in given cls set.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        :param next_layer_params: Data structure holding weight and bias for next layer in cls set.
        """
        if isinstance(cls_set[0], torch.nn.ConvTranspose2d):
            cls_set[0].weight.data = cls_set[0].weight.data.permute(1, 0, 2, 3).contiguous()
        if isinstance(cls_set[0], torch.nn.ConvTranspose1d):
            cls_set[0].weight.data = cls_set[0].weight.data.permute(1, 0, 2).contiguous()

        if isinstance(cls_set[2], torch.nn.ConvTranspose2d):
            cls_set[2].weight.data = cls_set[2].weight.data.permute(1, 0, 2, 3).contiguous()
        if isinstance(cls_set[2], torch.nn.ConvTranspose1d):
            cls_set[2].weight.data = cls_set[2].weight.data.permute(1, 0, 2).contiguous()

        assert cls_set[1].groups > 1

        prev_layer_params.weight = cls_set[0].weight.detach().numpy().flatten()
        prev_layer_params.weightShape = np.array(cls_set[0].weight.shape)
        if len(prev_layer_params.weightShape) == 3:
            prev_layer_params.weightShape = prev_layer_params.weightShape + [1]

        curr_layer_params.weight = cls_set[1].weight.detach().numpy().flatten()
        curr_layer_params.weightShape = np.array(cls_set[1].weight.shape)
        if len(curr_layer_params.weightShape) == 3:
            curr_layer_params.weightShape = curr_layer_params.weightShape + [1]

        next_layer_params.weight = cls_set[2].weight.detach().numpy().flatten()
        next_layer_params.weightShape = np.array(cls_set[2].weight.shape)
        if len(next_layer_params.weightShape) == 3:
            next_layer_params.weightShape = next_layer_params.weightShape + [1]


        if cls_set[0].bias is not None:
            prev_layer_params.bias = cls_set[0].bias.detach().numpy()
        else:
            prev_layer_params.isBiasNone = True

        if cls_set[1].bias is not None:
            curr_layer_params.bias = cls_set[1].bias.detach().numpy()
        else:
            curr_layer_params.isBiasNone = True

    @staticmethod
    def _update_params_for_depthwise_conv(cls_set: ClsSet,
                                          prev_layer_params: libpymo.EqualizationParams,
                                          curr_layer_params: libpymo.EqualizationParams,
                                          next_layer_params: libpymo.EqualizationParams):
        """
        Update weight and biases for cls set using updated data structures.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        :param next_layer_params: Data structure holding weight and bias for next layer in cls set.
        """
        if isinstance(cls_set[0], (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            prev_layer_params.weightShape = prev_layer_params.weightShape[:-1]
        cls_set[0].weight.data = torch.from_numpy(np.reshape(prev_layer_params.weight,
                                                             prev_layer_params.weightShape))
        cls_set[0].weight.data = cls_set[0].weight.data.type(torch.FloatTensor)
        if isinstance(cls_set[0], torch.nn.ConvTranspose2d):
            cls_set[0].weight.data = cls_set[0].weight.data.permute(1, 0, 2, 3).contiguous()
        if isinstance(cls_set[0], torch.nn.ConvTranspose1d):
            cls_set[0].weight.data = cls_set[0].weight.data.permute(1, 0, 2).contiguous()

        if isinstance(cls_set[1], (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            curr_layer_params.weightShape = curr_layer_params.weightShape[:-1]
        cls_set[1].weight.data = torch.from_numpy(np.reshape(curr_layer_params.weight,
                                                             curr_layer_params.weightShape))
        cls_set[1].weight.data = cls_set[1].weight.data.type(torch.FloatTensor)

        if isinstance(cls_set[2], (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            next_layer_params.weightShape = next_layer_params.weightShape[:-1]

        cls_set[2].weight.data = torch.from_numpy(np.reshape(next_layer_params.weight,
                                                             next_layer_params.weightShape))
        cls_set[2].weight.data = cls_set[2].weight.data.type(torch.FloatTensor)
        if isinstance(cls_set[2], torch.nn.ConvTranspose2d):
            cls_set[2].weight.data = cls_set[2].weight.data.permute(1, 0, 2, 3).contiguous()
        if isinstance(cls_set[2], torch.nn.ConvTranspose1d):
            cls_set[2].weight.data = cls_set[2].weight.data.permute(1, 0, 2).contiguous()

        if cls_set[0].bias is not None:
            cls_set[0].bias.data = torch.from_numpy(np.reshape(prev_layer_params.bias,
                                                               prev_layer_params.weightShape[0]))
            cls_set[0].bias.data = cls_set[0].bias.data.type(torch.FloatTensor)

        if cls_set[1].bias is not None:
            cls_set[1].bias.data = torch.from_numpy(np.reshape(curr_layer_params.bias,
                                                               curr_layer_params.weightShape[0]))
            cls_set[1].bias.data = cls_set[1].bias.data.type(torch.FloatTensor)


class HighBiasFold:
    """
    Code to apply the high-bias-fold technique to a model
    """

    ActivationIsReluForFirstModule = bool
    ScaleForFirstModule = np.ndarray

    @classmethod
    def bias_fold(cls, cls_set_info_list: List[ClsSetInfo],
                  bn_layers: Dict[Union[torch.nn.Conv2d, torch.nn.ConvTranspose2d], torch.nn.BatchNorm2d]):
        """
        Folds bias values greater than 3 * sigma to next layer's bias

        :param cls_set_info_list: List of info elements for each cls set
        :param bn_layers: Key: Conv/Linear layer Value: Corresponding folded BN layer
        :return: None
        """
        if not bn_layers:
            logger.info('High Bias folding is not supported for models without BatchNorm Layers')
            return

        for cls_set_info in cls_set_info_list:
            for cls_pair_info in cls_set_info.cls_pair_info_list:

                if (cls_pair_info.layer1.bias is None) or (cls_pair_info.layer2.bias is None) or \
                        (cls_pair_info.layer1 not in bn_layers):
                    continue

                # Create data structures for holding layer weights and bias parameters.
                prev_layer_params = libpymo.LayerParams()
                curr_layer_params = libpymo.LayerParams()
                prev_layer_bn_params = libpymo.BNParamsHighBiasFold()

                # Prepare and pack data structures for high bias fold.
                cls._pack_bn_layer_params(cls_pair_info, bn_layers, prev_layer_bn_params)
                cls._pack_previous_and_current_layer_params(cls_pair_info, prev_layer_params, curr_layer_params)

                # Update bias for previous and current layer and data structures in-place.
                libpymo.updateBias(prev_layer_params, curr_layer_params, prev_layer_bn_params)

                # Set updated biases for previous and current layer.
                cls._update_previous_and_current_layer_bias(cls_pair_info, prev_layer_params, curr_layer_params)

    @staticmethod
    def _pack_bn_layer_params(cls_pair_info: ClsSetInfo.ClsSetLayerPairInfo,
                              bn_layers: Dict[torch.nn.Module, torch.nn.BatchNorm2d],
                              prev_layer_bn_params: libpymo.BNParamsHighBiasFold):
        """
        Helper method to pack batch norm layer parameter for high bias fold.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param bn_layers: Dictionary with Key being Conv/Linear layer and value being corresponding folded BN layer.
        :param prev_layer_bn_params: Data structure to pack batch norm parameter.
        """
        scaling_parameter = cls_pair_info.scale_factor

        # Scaling gamma and beta parameter of batch norm layer
        prev_layer_bn_params.gamma = bn_layers[cls_pair_info.layer1].weight.detach().numpy().reshape(-1)
        prev_layer_bn_params.beta = bn_layers[cls_pair_info.layer1].bias.detach().numpy().reshape(-1)

        if len(scaling_parameter) != len(prev_layer_bn_params.gamma) or \
                len(scaling_parameter) != len(prev_layer_bn_params.beta):
            raise ValueError("High Bias absorption is not supported for networks with fold-forward BatchNorms")
        prev_layer_bn_params.gamma = np.divide(prev_layer_bn_params.gamma, scaling_parameter)
        prev_layer_bn_params.beta = np.divide(prev_layer_bn_params.beta, scaling_parameter)

    @staticmethod
    def _pack_previous_and_current_layer_params(cls_pair_info, prev_layer_params, curr_layer_params):
        """
        Helper method to pack information of previous and current layer.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param prev_layer_params: Data structure to pack previous layer parameters.
        :param curr_layer_params: Data structure to pack current layer parameters.
        """
        prev_layer_params.activationIsRelu = cls_pair_info.relu_activation_between_layers
        prev_layer_params.bias = cls_pair_info.layer1.bias.detach().numpy()

        weight = cls_pair_info.layer2.weight

        if isinstance(cls_pair_info.layer2, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            weight = torch.unsqueeze(weight, dim=-1)

        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        if isinstance(cls_pair_info.layer2, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)) and \
                cls_pair_info.layer2.groups == 1:
            weight = weight.permute(1, 0, 2, 3)

        curr_layer_params.bias = cls_pair_info.layer2.bias.detach().numpy()
        curr_layer_params.weight = weight.detach().numpy().reshape(-1)
        curr_layer_params.weightShape = np.array(weight.shape)

    @staticmethod
    def _update_previous_and_current_layer_bias(cls_pair_info: ClsSetInfo.ClsSetLayerPairInfo,
                                                prev_layer_params: libpymo.LayerParams,
                                                curr_layer_params: libpymo.LayerParams):
        """
        Update biases for previous and current layer.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        """
        prev_layer_bias_shape = cls_pair_info.layer1.weight.shape[0]
        if (isinstance(cls_pair_info.layer1, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d))) and \
                (cls_pair_info.layer1.groups == 1):
            prev_layer_bias_shape = cls_pair_info.layer1.weight.shape[1]

        cls_pair_info.layer1.bias.data = torch.from_numpy(np.reshape(prev_layer_params.bias,
                                                                     prev_layer_bias_shape))
        cls_pair_info.layer1.bias.data = cls_pair_info.layer1.bias.data.type(torch.FloatTensor)

        cls_pair_info.layer2.bias.data = torch.from_numpy(np.reshape(curr_layer_params.bias,
                                                                     curr_layer_params.weightShape[0]))
        cls_pair_info.layer2.bias.data = cls_pair_info.layer2.bias.data.type(torch.FloatTensor)


def equalize_model(model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]],
                   dummy_input: Union[torch.Tensor, Tuple] = None):
    """
    High-level API to perform Cross-Layer Equalization (CLE) on the given model. The model is equalized in place.

    :param model: Model to equalize
    :param input_shapes: Shape of the input (can be a tuple or a list of tuples if multiple inputs)
    :param dummy_input: A dummy input to the model. Can be a Tensor or a Tuple of Tensors
    :return: None
    """
    if dummy_input is None:
        # The use of input_shapes will be removed in a future release. It is maintained now for backward compatibility.
        # Note, create_rand_tensors_given_shapes() creates all FP32 tensors where as some multi-input models might
        # additionally use Integer Tensors.
        dummy_input = create_rand_tensors_given_shapes(input_shapes, torch.device('cpu'))
    if isinstance(dummy_input, (list, tuple)):
        input_shapes = [i.shape for i in dummy_input]
    else:
        input_shapes = dummy_input.shape

    if isinstance(model, torch.nn.DataParallel):
        equalize_model(model.module, input_shapes, dummy_input)
    else:
        device = get_device(model)
        model.cpu()
        # fold batchnorm layers
        folded_pairs = fold_all_batch_norms(model, input_shapes, dummy_input)
        equalize_bn_folded_model(model, input_shapes, folded_pairs, dummy_input=dummy_input)

        model.to(device=device)

def equalize_bn_folded_model(model: torch.nn.Module,
                             input_shapes: Union[Tuple, List[Tuple]],
                             folded_pairs: List[Tuple[torch.nn.Module, torch.nn.BatchNorm2d]],
                             dummy_input: Union[torch.Tensor, Tuple] = None
                             ):
    """
    Perform Cross-Layer Scaling (CLS) and High Bias Folding (HBF) on a batchnorm-folded model.
    The model is equalized in place.

    :param model: Batchnorm-folded model to equalize
    :param input_shapes: Shape of the input (can be a tuple or a list of tuples if multiple inputs)
    :param dummy_input: Dummy input to the model. Used to parse model graph. User is expected to place the tensors on the appropriate device.
    :param folded_pairs: List of pairs of folded layers
    :return: None
    """
    if isinstance(model, torch.nn.DataParallel):
        equalize_bn_folded_model(model.module, input_shapes, folded_pairs, dummy_input=dummy_input)
    else:
        device = get_device(model)
        model.cpu()
        bn_dict = {}
        for conv_bn in folded_pairs:
            bn_dict[conv_bn[0]] = conv_bn[1]

        # replace any ReLU6 layers with ReLU
        utils.replace_modules_of_type1_with_type2(model, torch.nn.ReLU6, torch.nn.ReLU)

        # perform cross-layer scaling on applicable layer sets
        cls_set_info_list = CrossLayerScaling.scale_model(model, input_shapes, dummy_input=dummy_input)

        # high-bias fold
        HighBiasFold.bias_fold(cls_set_info_list, bn_dict)

        model.to(device=device)
