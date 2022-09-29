# /usr/bin/env python3.8
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
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops
from TrainingExtensions.onnx.test.python.test_models import build_dummy_model, single_residual_model, multi_input_model
from aimet_onnx.meta.connectedgraph import ConnectedGraph


class TestConnectedGraph:
    def test_simple_model(self):
        model = build_dummy_model()
        cg = ConnectedGraph(model)
        ops = cg.get_all_ops()
        assert len(ops) == 5
        assert ['conv', 'relu', 'maxpool', 'flatten', 'fc'] == [op_name for op_name in ops]
        products = cg.get_all_products()
        assert len(products) == 5
        assert ['input_to_conv', 'conv_to_relu', 'relu_to_maxpool', 'maxpool_to_flatten', 'flatten_to_fc'] == [product for product in products]

    def test_single_residual_model(self):
        model = single_residual_model()
        conn_graph = ConnectedGraph(model)
        assert len(conn_graph.get_all_ops()) == 19
        products = conn_graph.get_all_products()
        assert len(products) == 21
        assert {'Conv_0_to_Relu_1', 'Relu_1_to_MaxPool_2', 'MaxPool_2_to_Conv_6'}.issubset({product for product in products})
        input_ops = get_all_input_ops(conn_graph)
        assert len(input_ops) == 1

    def test_multi_inputs_model(self):
        model = multi_input_model()
        conn_graph = ConnectedGraph(model)
        assert len(conn_graph.get_all_ops()) == 15

        products = conn_graph.get_all_products()
        assert len(products) == 16
        assert {'Conv_0_to_MaxPool_1', 'Conv_3_to_MaxPool_4', 'Conv_7_to_MaxPool_8'}.issubset(
            {product for product in products})
        input_ops = get_all_input_ops(conn_graph)
        assert len(input_ops) == 2
