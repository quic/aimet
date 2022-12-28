# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

from typing import Tuple, List, Union
from enum import Enum
import numpy as np
from onnx import onnx_pb

from aimet_common.utils import AimetLogger
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops
from aimet_common.connected_graph.connectedgraph import get_ordered_ops
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.meta.operations import Op

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


ScaleFactor = Union[np.ndarray, Tuple[np.ndarray]]
cls_supported_layer_types = ['Conv', 'ConvTranspose']
cls_supported_activation_types = ['Relu', 'PRelu']
ClsSupportedLayer = Union['Conv']


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

        def __init__(self, layer1: onnx_pb.NodeProto, layer2: onnx_pb.NodeProto, scale_factor: np.ndarray,
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


def get_ordered_list_of_conv_modules(list_of_starting_ops: List) -> List:
    """
    Finds order of nodes in graph
    :param list_of_starting_ops: list of starting ops for the model
    :return: List of names in graph in order
    """
    module_list = get_ordered_ops(list_of_starting_ops)
    module_list = [[module.dotted_name, module] for module in module_list if module.type in cls_supported_layer_types]
    return module_list


class GraphSearchUtils:
    """
    Code to search a model graph to find nodes to use for cross-layer-scaling and high-bias-fold
    """

    def __init__(self, model):
        self._connected_graph = ConnectedGraph(model)
        self._ordered_module_list = get_ordered_list_of_conv_modules(self._connected_graph.starting_ops)

    def convert_layer_group_to_cls_sets(self, layer_group):
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
        def convert_to_cls_layer_type(layer: Op) -> Tuple[ClsLayerType, ClsSupportedLayer]:
            """
            Given the layer, check if its supported in CLS
            :param layer: layer to check
            :return: Tuple of ClsLayerType and the layer
            """
            if layer.groups == 1:
                layer_type = ClsLayerType.Conv
                # Check if it is a depthwise layer (the last two dimensions of weight param will be equal to groups
                weight_param_shape = [param for param, param_type in layer.parameters.values() if param_type == 'weight'][0].shape
                if layer.groups == weight_param_shape[-1] and weight_param_shape[-1] == weight_param_shape[-2]:
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

    @staticmethod
    def find_downstream_layer_groups_to_scale(op: Op, layer_groups: List, current_group=None, visited_nodes=None):
        """
        Recursive function to find cls layer groups downstream from a given op
        :param op: Starting op to search from
        :param layer_groups: Running list of layer groups
        :param current_group: Running current layer group
        :param visited_nodes: Running list of visited nodes (to short-circuit recursion)
        """

        if not visited_nodes:
            visited_nodes = []
        if not current_group:
            current_group = []

        if op in visited_nodes:
            return
        visited_nodes.append(op)

        # If current node is Conv2D, add to the current group
        if op.get_module() and op.type in cls_supported_layer_types:
            current_group.append(op)

        # Terminating condition for current group
        if not op.get_module() or not op.type in cls_supported_layer_types + cls_supported_activation_types:
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

    def find_layer_groups_to_scale(self) -> List[List]:
        """
        :return: List of groups of layers. Each group can be independently equalized
        """

        # Find the input node(s) in the graph
        input_nodes = get_all_input_ops(self._connected_graph)

        layer_groups = []
        for op in input_nodes:
            self.find_downstream_layer_groups_to_scale(op, layer_groups)

        # Sort the layer groups in order of occurrence in the model
        ordered_layer_groups = []
        for module_name, _ in self._ordered_module_list:
            for layer_group in layer_groups:
                if layer_group[0].dotted_name == module_name:
                    ordered_layer_groups.append(layer_group)

        return ordered_layer_groups
