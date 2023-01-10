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
import numpy as np
from onnx import onnx_pb, numpy_helper

from aimet_common.utils import AimetLogger
from aimet_common.connected_graph.connectedgraph import get_ordered_ops
from aimet_common.cross_layer_equalization import GraphSearchUtils, CrossLayerScaling as CLS, ClsSetInfo
import aimet_common.libpymo as libpymo      # pylint: disable=import-error

from aimet_onnx.meta.connectedgraph import ConnectedGraph, WEIGHT_INDEX, BIAS_INDEX
from aimet_onnx.utils import transpose_tensor, ParamUtils, get_node_attribute

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

ClsSet = Union[Tuple['Conv', 'Conv'],
               Tuple['Conv', 'Conv', 'Conv']]
ScaleFactor = Union[np.ndarray, Tuple[np.ndarray]]
cls_supported_layer_types = ['Conv', 'ConvTranspose']
cls_supported_activation_types = ['Relu', 'PRelu']


def get_ordered_list_of_conv_modules(list_of_starting_ops: List) -> List:
    """
    Finds order of nodes in graph
    :param list_of_starting_ops: list of starting ops for the model
    :return: List of names in graph in order
    """
    module_list = get_ordered_ops(list_of_starting_ops)
    module_list = [[module.dotted_name, module] for module in module_list if module.type in cls_supported_layer_types]
    return module_list


class CrossLayerScaling(CLS):
    """
    Scales a model's layers to equalize the weights between consecutive layers
    """
    def __init__(self, model: onnx_pb.ModelProto):
        """
        :param model: ONNX model
        """
        super().__init__()
        self._model = model

    def scale_model(self) -> List[ClsSetInfo]:
        """
        Uses cross-layer scaling to scale all applicable layers in the given model

        :param model: Model to scale
        :return: CLS information for each CLS set
        """
        # Find layer groups
        connected_graph = ConnectedGraph(self._model)
        ordered_module_list = get_ordered_list_of_conv_modules(connected_graph.starting_ops)
        graph_search = GraphSearchUtils(connected_graph, ordered_module_list, cls_supported_layer_types,
                                        cls_supported_activation_types)
        layer_groups = graph_search.find_layer_groups_to_scale()

        # Find cls sets from the layer groups
        cls_sets = []
        for layer_group in layer_groups:
            cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
            cls_sets += cls_set

        # Scale the CLS sets
        scale_factors = self.scale_cls_sets(cls_sets)

        # Find if there were relu activations between layers of each cls set
        is_relu_activation_in_cls_sets = graph_search.is_relu_activation_present_in_cls_sets(cls_sets)

        # Convert to a list of cls-set-info elements
        cls_set_info_list = CrossLayerScaling.create_cls_set_info_list(cls_sets, scale_factors,
                                                                       is_relu_activation_in_cls_sets)

        return cls_set_info_list

    def _populate_libpymo_params(self, module: onnx_pb.NodeProto,
                                 layer_param: libpymo.EqualizationParams):
        """
        Populates libpymo weight parameter
        """
        weight = ParamUtils.get_param(self._model.model, module, WEIGHT_INDEX)
        groups = get_node_attribute(module, "group")

        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        if module.op_type == "ConvTranspose" and groups == 1:
            weight = transpose_tensor(weight, (1, 0, 2, 3))

        layer_param.weight = numpy_helper.to_array(weight).reshape(-1)
        layer_param.weightShape = np.array(weight.dims)

    def _pack_params_for_conv(self,
                              cls_set,
                              prev_layer_params: libpymo.EqualizationParams,
                              curr_layer_params: libpymo.EqualizationParams):
        """
        Prepare and pack data structure for previous and current layer in given cls set.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        """
        self._populate_libpymo_params(cls_set[0].get_module(), prev_layer_params)
        self._populate_libpymo_params(cls_set[1].get_module(), curr_layer_params)

        cls_set_0_bias = ParamUtils.get_param(self._model.model, cls_set[0].get_module(), BIAS_INDEX)
        if cls_set_0_bias is not None:
            prev_layer_params.bias = numpy_helper.to_array(cls_set_0_bias).reshape(-1)
        else:
            prev_layer_params.isBiasNone = True

    def _update_weight_for_layer_from_libpymo_obj(self, layer_param: libpymo.EqualizationParams,
                                                  module: onnx_pb.NodeProto):
        """
        Update weight parameter from libpymo object
        """
        weight = ParamUtils.get_param(self._model.model, module, WEIGHT_INDEX)
        weight.raw_data = np.asarray(layer_param.weight, dtype=np.float32).tobytes()
        groups = get_node_attribute(module, "group")
        # Transpose weight back to original configuration
        if module.op_type == "ConvTranspose" and groups == 1:
            weight = transpose_tensor(weight, (1, 0, 2, 3))

        weight_param = ParamUtils.get_param(self._model.model, module, WEIGHT_INDEX)
        weight_param.raw_data = weight.raw_data

    def _update_params_for_conv(self,
                                cls_set,
                                prev_layer_params: libpymo.EqualizationParams,
                                curr_layer_params: libpymo.EqualizationParams):
        """
        Update weight and biases for cls set using updated data structures.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        """
        self._update_weight_for_layer_from_libpymo_obj(prev_layer_params, cls_set[0].get_module())
        self._update_weight_for_layer_from_libpymo_obj(curr_layer_params, cls_set[1].get_module())

        if not prev_layer_params.isBiasNone:
            bias_param = ParamUtils.get_param(self._model.model, cls_set[0].get_module(),
                                              BIAS_INDEX)
            bias_param.raw_data = np.asarray(prev_layer_params.bias, dtype=np.float32).tobytes()

    @staticmethod
    def _pack_params_for_depthwise_conv(cls_set,
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

    @staticmethod
    def _update_params_for_depthwise_conv(cls_set,
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
