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
from packaging import version
import torch
from packaging import version

from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops, get_all_ops_with_constant_inputs
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from models import models_for_tests


class TestConnectedGraph:
    def test_simple_model(self):
        model = models_for_tests.build_dummy_model()
        cg = ConnectedGraph(model)
        ops = cg.get_all_ops()
        assert len(ops) == 5
        assert ['conv', 'relu', 'maxpool', 'flatten', 'fc'] == [op_name for op_name in ops]
        products = cg.get_all_products()
        assert len(products) == 10
        assert ['input_to_conv', 'conv_to_relu', 'relu_to_maxpool', 'maxpool_to_flatten', 'flatten_to_fc', 'fc_to_output',
                'conv_w', 'conv_b', 'fc_w', 'fc_b'] == [product for product in products]

    def test_single_residual_model(self):
        if version.parse(torch.__version__) >= version.parse("1.13"):
            model = models_for_tests.single_residual_model()
            conn_graph = ConnectedGraph(model)
            assert len(conn_graph.get_all_ops()) == 16
            products = conn_graph.get_all_products()
            assert len(products) == 29
            assert {'/conv1/Conv_to_/relu1/Relu', '/relu1/Relu_to_/maxpool/MaxPool'}.issubset({product for product in products})
            assert {'/fc/Gemm_to_output', 'onnx::Conv_43', 'onnx::Conv_44', 'conv4.weight'}.issubset({product for product in products})
            input_ops = get_all_input_ops(conn_graph)
            assert len(input_ops) == 1
            assert conn_graph._branch_count == 1
            assert conn_graph.ordered_ops[15].transposed_params == True

    def test_multi_inputs_model(self):
        if version.parse(torch.__version__) >= version.parse("1.13"):
            model = models_for_tests.multi_input_model()
            conn_graph = ConnectedGraph(model)
            assert len(conn_graph.get_all_ops()) == 15

            products = conn_graph.get_all_products()
            assert len(products) == 27
            assert {'/conv1_a/Conv_to_/maxpool1_a/MaxPool', '/conv1_b/Conv_to_/maxpool1_b/MaxPool', '/conv2/Conv_to_/maxpool2/MaxPool'}.issubset(
                {product for product in products})
            assert {'conv1_a.weight', 'conv1_a.bias', 'conv1_b.weight'}.issubset({product for product in products})
            input_ops = get_all_input_ops(conn_graph)
            assert len(input_ops) == 2

    def test_transposed_conv_model(self):
        if version.parse(torch.__version__) >= version.parse("1.13"):
            model = models_for_tests.transposed_conv_model()
            conn_graph = ConnectedGraph(model)
            assert len(conn_graph.get_all_ops()) == 5

            products = conn_graph.get_all_products()
            assert len(products) == 12
            assert {'bn1.weight',
                    'bn1.bias'}.issubset({product for product in products})

    def test_concat_model(self):
        if version.parse(torch.__version__) >= version.parse("1.13"):
            model = models_for_tests.concat_model()
            conn_graph = ConnectedGraph(model)
            ops = conn_graph.get_all_ops()
            assert len(ops) == 6
            assert len(ops['/Concat'].inputs) == 3
            products = conn_graph.get_all_products()
            assert len(products) == 17
            assert conn_graph._branch_count == 0

    def test_hierarchical_model(self):
        if version.parse(torch.__version__) >= version.parse("1.13"):
            model = models_for_tests.hierarchical_model()
            conn_graph = ConnectedGraph(model)
            ops = conn_graph.get_all_ops()
            assert len(ops) == 68
            assert conn_graph._branch_count == 0
            ordered_ops = conn_graph.ordered_ops
            name_to_index = {}
            for index, op in enumerate(ordered_ops):
                name_to_index[op.name] = index

            # Check in the graph that if A & B are connected and A comes before B in the graph then that should be the case
            # in ordered graphs as well
            assert name_to_index['/conv1/conv/Conv'] < name_to_index['/nm1/tm1/Reshape']
            assert name_to_index['/sq/seq_list/seq_list.0/Conv'] < name_to_index['/sq/seq_list/seq_list.5/Conv']
            assert name_to_index['/conv2/conv/Conv'] < name_to_index['/nm2/tm1/conv3/Conv']

    def test_matmul_layer_param_creation(self):
        torch.manual_seed(10)
        torch_model = models_for_tests.BNBeforeFlattenLinear()

        torch_model.eval()

        input_shape = (2, 10, 24, 24)

        model = models_for_tests._convert_to_onnx_no_fold(torch_model, torch.randn(input_shape))

        cg = ConnectedGraph(model)
        assert cg.ordered_ops[4].type == 'Transpose'
        assert cg.ordered_ops[5].type == 'MatMul'
        assert 'fc2.weight' in cg.ordered_ops[5].parameters

    def test_constant_elementwise_inputs(self):
        """ Test that constant inputs to elementwise ops are identified correctly """
        model = models_for_tests.elementwise_op_model()
        cg = ConnectedGraph(model)

        assert len(get_all_ops_with_constant_inputs(cg)) == 2

        assert not cg.ordered_ops[0].inputs[0].is_const
        assert cg.ordered_ops[0].inputs[1].is_const
        assert cg.ordered_ops[1].inputs[0].is_model_input == False
        assert not cg.ordered_ops[1].inputs[0].is_const
        assert cg.ordered_ops[1].inputs[1].is_const
