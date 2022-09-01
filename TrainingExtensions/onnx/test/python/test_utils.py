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
import onnx
from aimet_onnx import utils, test_models

class TestUtils:
    """
    Test functions in utils
    """
    def test_remove_nodes(self):
        """
        Test remove nodes by given type
        """
        model = test_models.build_dummy_model()
        node_ls = [node.op_type for node in model.graph.node]
        assert node_ls == ['Conv', 'Relu', 'MaxPool', 'Flatten', 'Gemm']
        # Remove first layer of dummy model
        utils.remove_nodes_with_type('Conv', model.graph)
        new_node_ls = [node.op_type for node in model.graph.node]
        assert new_node_ls == ['Relu', 'MaxPool', 'Flatten', 'Gemm']
        # Remove last layer of dummy model
        utils.remove_nodes_with_type('Gemm', model.graph)
        new_node_ls = [node.op_type for node in model.graph.node]
        assert new_node_ls == ['Relu', 'MaxPool', 'Flatten']
        # Check connection of each layer
        name = model.graph.input[0].name
        for node in model.graph.node:
            assert node.input[0] == name
            name = node.output[0]

    def test_replace_nodes(self):
        """
        Test replace op type of nodes with given op type
        """
        model = test_models.build_dummy_model()
        node_ls = [node.op_type for node in model.graph.node]
        assert node_ls == ['Conv', 'Relu', 'MaxPool', 'Flatten', 'Gemm']

        utils.replace_node_with_op('Conv', 'CustomOp', model.graph)
        new_node_ls = [node.op_type for node in model.graph.node]
        assert new_node_ls == ['CustomOp', 'Relu', 'MaxPool', 'Flatten', 'Gemm']

    def test_get_weights(self):
        """
        Test get weights
        """
        model = test_models.build_dummy_model()
        for node in model.graph.initializer:
            assert node.raw_data == utils.get_weights(node.name, model.graph)

    def test_list_nodes(self):
        """
        Test get nodes with ordered
        """
        model = test_models.build_dummy_model()
        node_dict = utils.get_ordered_dict_of_nodes(model.graph)
        node_keys = list(node_dict.keys())

        for i, node in enumerate(model.graph.node):
            assert node_keys[i] == node.name
            assert node_dict[node.name] == node
