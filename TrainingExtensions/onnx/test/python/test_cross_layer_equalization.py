# /usr/bin/env python3.8
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
import numpy as np
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from onnxruntime_extensions import get_library_path
from onnx import numpy_helper

from aimet_common.cross_layer_equalization import GraphSearchUtils
from aimet_onnx.meta.connectedgraph import ConnectedGraph, WEIGHT_INDEX
from aimet_onnx.cross_layer_equalization import get_ordered_list_of_conv_modules, \
    cls_supported_layer_types, cls_supported_activation_types, CrossLayerScaling
from aimet_onnx.utils import ParamUtils
import test_models


class TestCLS:
    def test_graph_search_utils_single_residual_model(self):
        model = test_models.single_residual_model()
        connected_graph = ConnectedGraph(model)
        ordered_module_list = get_ordered_list_of_conv_modules(connected_graph.starting_ops)
        graph_search_utils = GraphSearchUtils(connected_graph, ordered_module_list, cls_supported_layer_types, cls_supported_activation_types)
        ordered_layer_groups = graph_search_utils.find_layer_groups_to_scale()[0]
        ordered_layer_groups_names = [op.dotted_name for op in ordered_layer_groups]
        assert ordered_layer_groups_names == ['Conv_3', 'Conv_5']

    def test_find_cls_sets_depthwise_model(self):
        model = test_models.depthwise_conv_model()

        connected_graph = ConnectedGraph(model)
        ordered_module_list = get_ordered_list_of_conv_modules(connected_graph.starting_ops)
        graph_search_utils = GraphSearchUtils(connected_graph, ordered_module_list, cls_supported_layer_types,
                                              cls_supported_activation_types)

        ordered_layer_groups = graph_search_utils.find_layer_groups_to_scale()[0]
        # Find cls sets from the layer groups
        cls_sets = graph_search_utils.convert_layer_group_to_cls_sets(ordered_layer_groups)
        cls_sets_names = []
        for cls_set in cls_sets:
            cls_sets_name = tuple([op.dotted_name for op in cls_set])
            cls_sets_names.append(cls_sets_name)
        assert cls_sets_names == [('Conv_0', 'Conv_2', 'Conv_4'), ('Conv_4', 'Conv_6', 'Conv_8'), ('Conv_8', 'Conv_10', 'Conv_12'), ('Conv_12', 'Conv_14', 'Conv_16'), ('Conv_16', 'Conv_18', 'Conv_20'), ('Conv_20', 'Conv_22', 'Conv_24'), ('Conv_24', 'Conv_26', 'Conv_28'), ('Conv_28', 'Conv_30', 'Conv_32')]

    def test_find_cls_sets_resnet_model(self):
        model = test_models.single_residual_model()

        connected_graph = ConnectedGraph(model)
        ordered_module_list = get_ordered_list_of_conv_modules(connected_graph.starting_ops)
        graph_search_utils = GraphSearchUtils(connected_graph, ordered_module_list, cls_supported_layer_types,
                                              cls_supported_activation_types)

        ordered_layer_groups = graph_search_utils.find_layer_groups_to_scale()[0]
        # Find cls sets from the layer groups
        cls_sets = graph_search_utils.convert_layer_group_to_cls_sets(ordered_layer_groups)
        cls_sets_names = []
        for cls_set in cls_sets:
            cls_sets_name = tuple([op.dotted_name for op in cls_set])
            cls_sets_names.append(cls_sets_name)
        assert cls_sets_names == [('Conv_3', 'Conv_5')]

    def test_scale_model_residual(self):
        model = test_models.single_residual_model()

        input_shape = (1, 3, 32, 32)
        test_data = np.random.randn(*input_shape).astype(np.float32)
        session = _build_session(model)
        output_before_cls = session.run(None, {'input': test_data})
        cls = CrossLayerScaling(model)
        cls_set_info = cls.scale_model()
        output_after_cls = session.run(None, {'input': test_data})
        assert np.allclose(output_after_cls, output_before_cls)
        conv_3 = cls_set_info[0].cls_pair_info_list[0].layer1.get_module()
        conv_5 = cls_set_info[0].cls_pair_info_list[0].layer2.get_module()
        weight_3 = numpy_helper.to_array(ParamUtils.get_param(model.model, conv_3, WEIGHT_INDEX))
        weight_5 = numpy_helper.to_array(ParamUtils.get_param(model.model, conv_5, WEIGHT_INDEX))
        assert np.allclose(np.amax(np.abs(weight_3), axis=(1, 2, 3)), np.amax(np.abs(weight_5), axis=(0, 2, 3)))

    def test_scale_model_tranposed_conv(self):
        model = test_models.transposed_conv_model_without_bn()
        input_shape = (10,10,4,4)
        test_data = np.random.randn(*input_shape).astype(np.float32)
        session = _build_session(model)
        output_before_cls = session.run(None, {'input': test_data})
        cls = CrossLayerScaling(model)
        cls_set_info = cls.scale_model()
        output_after_cls = session.run(None, {'input': test_data})
        assert np.allclose(output_after_cls, output_before_cls)
        conv_3 = cls_set_info[0].cls_pair_info_list[0].layer1.get_module()
        conv_5 = cls_set_info[0].cls_pair_info_list[0].layer2.get_module()
        weight_3 = numpy_helper.to_array(ParamUtils.get_param(model.model, conv_3, WEIGHT_INDEX))
        weight_5 = numpy_helper.to_array(ParamUtils.get_param(model.model, conv_5, WEIGHT_INDEX))
        assert np.allclose(np.amax(np.abs(weight_3), axis=(0, 2, 3)), np.amax(np.abs(weight_5), axis=(1, 2, 3)))

    def test_scale_model_depthwise(self):
        model = test_models.depthwise_conv_model()
        input_shape = (1, 3, 224, 224)
        test_data = np.random.randn(*input_shape).astype(np.float32)
        session = _build_session(model)
        output_before_cls = session.run(None, {'input': test_data})
        cls = CrossLayerScaling(model)
        cls_set_infos = cls.scale_model()
        output_after_cls = session.run(None, {'input': test_data})
        assert np.allclose(output_after_cls, output_before_cls)
        assert len(cls_set_infos) == 8

def _build_session(model):
    """
    Build and return onnxruntime inference session
    :param providers: providers to execute onnxruntime
    """
    sess_options = SessionOptions()
    sess_options.register_custom_ops_library(get_library_path())
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    session = InferenceSession(
        path_or_bytes=model.model.SerializeToString(),
        sess_options=sess_options,
        providers=['CPUExecutionProvider'],
    )
    return session