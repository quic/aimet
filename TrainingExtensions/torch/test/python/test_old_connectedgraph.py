# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from aimet_torch.meta.old_connectedgraph import ConnectedGraph


class TwoLinearsModel(nn.Module):

    def __init__(self, per_sample_shape: list, hidden_size: int, output_size: int):
        super(TwoLinearsModel, self).__init__()
        assert len(per_sample_shape) == 3
        self.per_sample_shape = per_sample_shape
        input_size = per_sample_shape[0]
        for dim in per_sample_shape[1:]:
            input_size *= dim
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        # input shape: [batch, channels, height, width]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.l1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.l2 = BasicConv2d(32, 32, kernel_size=3)

    def forward(self, x):
        x = self.l1(x)
        # 149 x 149 x 32
        x = self.l2(x)
        return x


class TestTrainingExtensionsConnectedGraph(unittest.TestCase):

    @unittest.skip
    def test_two_linears_model_prime(self):
        """
        Test for old deprecated code. Please delete
        """
        model = TwoLinearsModel([1, 23, 23], 47, 7)
        batch_size = 13
        input_shape = [batch_size] + model.per_sample_shape
        input_tensor = torch.rand(input_shape)
        graph = ConnectedGraph(model, input_tensor)

        leaf_ops = graph.get_leaf_operations()
        assert len(leaf_ops) == 1
        leaf_op = leaf_ops[0]
        nm = leaf_op.name
        assert nm.split('/')[-2] == 'Linear[linear2]'
        nm = leaf_op.dotted_name
        assert nm.split('.')[-1] == 'linear2'

        assert graph.num_operations == 4
        assert graph.num_products == 8
        assert len(leaf_op.inputs) == 3  # 2 parms and 1 input
        assert leaf_op.output is None

    @unittest.skip
    def test_parse_x_name(self):
        """
        Test for old deprecated code. Please delete
        """
        model = DummyModel()
        input_shape = [1, 3, 299, 299]
        x_input = torch.rand(input_shape)
        graph = ConnectedGraph(model, x_input)