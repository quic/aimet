# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import tensorflow as tf

from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops, get_all_output_ops
from aimet_tensorflow.keras.connectedgraph import ConnectedGraph
from aimet_tensorflow.keras.amp.quantizer_groups import find_op_groups, find_quantizer_group
from models.test_models import keras_model, simple_sequential_with_input_shape, simple_functional, \
        concat_functional, single_residual_model, nested_sequential_model, nested_functional_model, \
        sequential_in_functional, tiny_conv_net
from aimet_tensorflow.keras.quantsim import QuantizationSimModel


class TestQuantizerGroups:
    """
    Test methods for OpGraph
    """

    def test_simple_sequential_with_input_shape(self):
        model = simple_sequential_with_input_shape()

        # If input shape was given when creating Sequential model, there is layer connection information
        for layer in model.layers:
            assert layer.inbound_nodes
        connected_graph = ConnectedGraph(model)
        find_op_groups(connected_graph)

        assert len(connected_graph.get_all_ops().keys()) == 2

        op_groups = find_op_groups(connected_graph)
        print(op_groups)
        # Check if there is one parent and one child
        for parent, child in op_groups.items():
            assert isinstance(parent, tuple) == False
            assert len(child) == 1

        assert op_groups['input_ops'] == ['Gemm_0']
        assert op_groups['Gemm_0'] == ['Gemm_1']
        assert op_groups['output_ops'] == ['Gemm_1']


    def test_simple_functional(self):
        model = simple_functional()

        # When creating Functional model, it should have to define input information
        # Therefore, there is no case where there is no input information in the Functional model
        for layer in model.layers:
            assert layer.inbound_nodes
        connected_graph = ConnectedGraph(model)
        for layer in connected_graph._model.layers:
            assert layer.inbound_nodes

        assert len(connected_graph.get_all_ops().keys()) == 2
        op_groups = find_op_groups(connected_graph)
        print(op_groups)
        for parent, child in op_groups.items():
            assert isinstance(parent, tuple) == False
            assert len(child) == 1

        assert op_groups['Gemm_0'] == ['Gemm_1']



    def test_concat_functional(self):
        model = concat_functional()

        for layer in model.layers:
            assert layer.inbound_nodes
        connected_graph = ConnectedGraph(model)
        for layer in connected_graph._model.layers:
            assert layer.inbound_nodes

        assert len(connected_graph.get_all_ops().keys()) == 6
        concat_op = connected_graph.get_all_ops()['Concat_3']
        assert len(concat_op.inputs) == 3
        op_groups = find_op_groups(connected_graph)
        print(op_groups)

    def test_single_residual(self):
        """Test building ConnectedGraph on single residual model"""
        model = single_residual_model()

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
        op_groups = find_op_groups(connected_graph)
        print(op_groups)

    def test_nested_sequential(self):
        """Test building ConnectedGraph on a model constructed with nested Sequential"""
        model = nested_sequential_model()

        connected_graph = ConnectedGraph(model)
        assert len(connected_graph.get_all_ops().keys()) == 8

        products = connected_graph.get_all_products()
        assert "Conv_0_to_BatchNormalization_1" in products
        op_groups = find_op_groups(connected_graph)
        print(op_groups)

    def test_nested_functional(self):
        """Test building ConnectedGraph on a model constructed with nested Functional"""
        model = nested_functional_model()

        connected_graph = ConnectedGraph(model)
        assert len(connected_graph.get_all_ops().keys()) == 5

        product_dict = connected_graph.get_all_products()
        assert "Conv_0_to_BatchNormalization_1" in product_dict
        assert "Relu_2_to_MaxPool_3" in product_dict
        op_groups = find_op_groups(connected_graph)
        print(op_groups)

    def test_sequential_in_functional(self):
        model = sequential_in_functional()

        connected_graph = ConnectedGraph(model)
        assert len(connected_graph.get_all_ops().keys()) == 6

        product_dict = connected_graph.get_all_products()
        assert "BatchNormalization_1_to_Relu_2" in product_dict
        assert "Relu_4_to_MaxPool_5" in product_dict
        op_groups = find_op_groups(connected_graph)
        print(op_groups)


    def test_sequential_in_functional(self):
        model = tiny_conv_net()

        connected_graph = ConnectedGraph(model)
        #assert len(connected_graph.get_all_ops().keys()) == 25

        product_dict = connected_graph.get_all_products()
        #assert "Conv_4_to_BatchNormalization_5" in product_dict
        #assert "batch_normalization_6.weight" in product_dict
        op_groups = find_op_groups(connected_graph)
        print(op_groups)


    def test_keras_model(self):
        model = keras_model()

        connected_graph = ConnectedGraph(model)
        op_groups = find_op_groups(connected_graph)

        for parent, child in op_groups.items():
            assert not isinstance(parent, tuple)
            assert len(child) == 1

        assert op_groups['Conv_0'] == ['BatchNormalization_1']
        assert op_groups['AveragePool_2'] == ['MaxPool_3']
        assert 'Conv_0' in op_groups

        assert 9 == len(op_groups)


    def test_find_quantizer_groups(self):
        model = keras_model()
        sim = QuantizationSimModel(model, quant_scheme='tf')

        _, quantizer_groups = find_quantizer_group(sim)

        assert len(quantizer_groups) == 9


    def test_find_quantizer_residual(self):
        model = single_residual_model()

        sim = QuantizationSimModel(model, quant_scheme='tf')
        _, quantizer_groups = find_quantizer_group(sim)

        assert len(quantizer_groups[2].parameter_quantizers) == 4
