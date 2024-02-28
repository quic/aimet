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
""" Utilities for implementing blockwise quantization using grouped conv approach """

import torch
from aimet_torch import utils

class ReduceSum(torch.nn.Module):
    """
    ReduceSum implementation
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        """
        Forward pass.

        :param x: Input tensor
        :return: Output of forward pas
        """
        return torch.sum(x, dim=self.dim)

class LinearWithGroupedConv(torch.nn.Module):
    '''
    Linear(out_feature, in_feature)
    x:[n, seq, in_feature] * w.T:[in_feature, out_feature] = [n, seq, out_feature]
    where, in_feature = num_blocks * block_size
    w:[out_feature, in_feature].reshape()
    => [out_feature, num_blocks, block_size].permute(1,0,2)
    => [num_blocks, out_feature, block_size].reshape()
    => [num_blocks * out_feature, block_size, 1, 1]
    x:[n, seq, in_feature].permute(0, 2, 1)
    => [n, in_feature, seq].reshape()
    => [n, in_feature, 1, seq]
    => torch.conv(weight=w, input=x, groups=num_blocks)
    => [n, num_blocks * out_feature, 1, seq].reshape()
    => [n, num_blocks, out_feature, seq].sum(dim=1)
    => [n, out_feature, seq].permute(0,2,1)
    [n, seq, out_feature]
    '''
    def __init__(self, linear: torch.nn.Module, block_size: int):
        super().__init__()
        out_feature, in_feature = linear.weight.shape
        assert in_feature % block_size == 0, f"in_feature:{in_feature} should be divisible by block_size:{block_size}"

        self.block_size = block_size
        self.num_blocks = in_feature // block_size

        weight = linear.weight.detach().reshape(-1, self.num_blocks, self.block_size)
        weight = weight.permute(1, 0, 2)
        weight = weight.reshape(-1, self.block_size, 1, 1)

        self.grouped_conv = torch.nn.Conv2d(in_channels=in_feature,
                                            out_channels=out_feature * self.num_blocks,
                                            kernel_size=1,
                                            groups=self.num_blocks,
                                            bias=False,
                                            device=linear.weight.device)
        self.grouped_conv.weight.data.copy_(weight)
        self.reducesum = ReduceSum(dim=1)


    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        :param x: Input tensor
        :return: Output of forward pas
        """
        batch, seq, in_feature = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(batch, in_feature, 1, seq)
        x = self.grouped_conv(x)
        x = x.reshape(batch, self.num_blocks, -1, seq)
        x = self.reducesum(x)
        x = x.permute(0, 2, 1)

        return x

def replace_linears_for_blockwise_quant(model: torch.nn.Module, block_size: int):
    """
    Replace all instances of torch.nn.Linears in model with equivalent LinearWithMatMul modules.

    :param model: Model to replace nn.Linears for
    :param block_size: Block size to use
    """
    linear_layers = []
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append((name, module))
        elif not utils.is_leaf_module(module):
            replace_linears_for_blockwise_quant(module, block_size)

    for name, module in linear_layers:
        setattr(model, name, LinearWithGroupedConv(module, block_size))
