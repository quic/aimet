# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

import os
import unittest.mock
import torch
import torch.nn as nn
from aimet_torch.elementwise_ops import AddOp, SubtractOp, MultiplyOp, DivideOp, ConcatOp
from aimet_torch import utils


class Model(nn.Module):
    def __init__(self, op):
        super(Model, self).__init__()
        self.relu1 = nn.ReLU()
        self.op1 = op

    def forward(self, input):
        x = self.relu1(input)
        x = self.op1(x, input)
        return x


class TestTrainingExtensionElementwiseOps(unittest.TestCase):
    def test_add_op(self):
        model = Model(AddOp())
        input_shape = (5, 10, 10, 20)
        input = torch.rand((5, 10, 10, 20))
        model(input)
        onnx_path = os.path.join( 'data/add_model'+ '.onnx')
        dummy_input = utils.create_rand_tensors_given_shapes(input_shape)
        torch.onnx.export(model, tuple(dummy_input), onnx_path)

    def test_subtract_op(self):
        model = Model(SubtractOp())
        input_shape = (5, 10, 10, 20)
        input = torch.rand((5, 10, 10, 20))
        model(input)
        onnx_path = os.path.join( 'data/subtract_model'+ '.onnx')
        dummy_input = utils.create_rand_tensors_given_shapes(input_shape)
        torch.onnx.export(model, tuple(dummy_input), onnx_path)

    def test_multiply_op(self):
        model = Model(MultiplyOp())
        input_shape = (5, 10, 10, 20)
        input = torch.rand((5, 10, 10, 20))
        model(input)
        onnx_path = os.path.join( 'data/multiply_model'+ '.onnx')
        dummy_input = utils.create_rand_tensors_given_shapes(input_shape)
        torch.onnx.export(model, tuple(dummy_input), onnx_path)

    def test_divide_op(self):
        model = Model(DivideOp())
        input_shape = (5, 10, 10, 20)
        input = torch.rand((5, 10, 10, 20))
        model(input)
        onnx_path = os.path.join( 'data/divide_model'+ '.onnx')
        dummy_input = utils.create_rand_tensors_given_shapes(input_shape)
        torch.onnx.export(model, tuple(dummy_input), onnx_path)

    def test_concat_op(self):
        model = Model(ConcatOp(axis=0))
        input_shape = (5, 10, 10, 20)
        input = torch.rand((5, 10, 10, 20))
        model(input)
        onnx_path = os.path.join( 'data/concat_model'+ '.onnx')
        dummy_input = utils.create_rand_tensors_given_shapes(input_shape)
        torch.onnx.export(model, tuple(dummy_input), onnx_path)