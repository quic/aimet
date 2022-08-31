# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""
This file contains unit tests for testing ConnectedGraph
"""
import pytest

import test_models_keras
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops, get_all_output_ops
from aimet_tensorflow.keras.connectedgraph import ConnectedGraph


class TestConnectedGraph:
    """
    Test methods for ConnectedGraph
    """

    def test_simple_sequential_with_input_shape(self):
        model = test_models_keras.simple_sequential_with_input_shape()

        # If input shape was given when creating Sequential model, there is layer connection information
        for layer in model.layers:
            assert layer.inbound_nodes
        connected_graph = ConnectedGraph(model)
        for layer in connected_graph._model.layers:
            assert layer.inbound_nodes

        assert len(connected_graph.get_all_ops().keys()) == 2

    def test_simple_sequential_without_input_shape(self):
        model = test_models_keras.simple_sequential_without_input_shape()

        # There is no connection information between layers before building
        for layer in model.layers:
            assert not layer.inbound_nodes

        # Raise RuntimeError if passed model is not built
        with pytest.raises(RuntimeError):
            _ = ConnectedGraph(model)

    def test_simple_functional(self):
        model = test_models_keras.simple_functional()

        # When creating Functional model, it should have to define input information
        # Therefore, there is no case where there is no input information in the Functional model
        for layer in model.layers:
            assert layer.inbound_nodes
        connected_graph = ConnectedGraph(model)
        for layer in connected_graph._model.layers:
            assert layer.inbound_nodes

        assert len(connected_graph.get_all_ops().keys()) == 2

    def test_concat_functional(self):
        model = test_models_keras.concat_functional()

        for layer in model.layers:
            assert layer.inbound_nodes
        connected_graph = ConnectedGraph(model)
        for layer in connected_graph._model.layers:
            assert layer.inbound_nodes

        assert len(connected_graph.get_all_ops().keys()) == 6
        concat_op = connected_graph.get_all_ops()['Concat_3']
        assert len(concat_op.inputs) == 3

    def test_single_residual(self):
        """Test building ConnectedGraph on single residual model"""
        model = test_models_keras.single_residual()

        connected_graph = ConnectedGraph(model)
        # 15 usual ops, 1 split ops
        assert len(connected_graph.get_all_ops().keys()) == 15 + 1
        assert connected_graph._split_count == 1

        product_dict = connected_graph.get_all_products()
        assert "Conv_8_to_Add_10" in product_dict
        assert "AveragePool_9_to_Add_10" in product_dict

        input_ops = get_all_input_ops(connected_graph)
        assert len(input_ops) == 1
        assert input_ops[0].get_module() == model.layers[1]
        output_ops = get_all_output_ops(connected_graph)
        assert len(output_ops) == 1
        assert output_ops[0].get_module() == model.layers[-1]

    def test_nested_sequential(self):
        """Test building ConnectedGraph on a model constructed with nested Sequential"""
        model = test_models_keras.nested_sequential_model()
        model.build((None, 8, 8, 3))

        connected_graph = ConnectedGraph(model)
        assert len(connected_graph.get_all_ops().keys()) == 8

        products = connected_graph.get_all_products()
        assert "Conv_0_to_BatchNormalization_1" in products

    def test_nested_functional(self):
        """Test building ConnectedGraph on a model constructed with nested Functional"""
        model = test_models_keras.nested_functional_model()

        connected_graph = ConnectedGraph(model)
        assert len(connected_graph.get_all_ops().keys()) == 5

        product_dict = connected_graph.get_all_products()
        assert "Conv_0_to_BatchNormalization_1" in product_dict
        assert "Relu_2_to_MaxPool_3" in product_dict

    def test_sequential_in_functional(self):
        model = test_models_keras.sequential_in_functional()

        connected_graph = ConnectedGraph(model)
        assert len(connected_graph.get_all_ops().keys()) == 6

        product_dict = connected_graph.get_all_products()
        assert "BatchNormalization_1_to_Relu_2" in product_dict
        assert "Relu_4_to_MaxPool_5" in product_dict
