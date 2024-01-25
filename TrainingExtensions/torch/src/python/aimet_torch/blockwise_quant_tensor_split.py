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
""" Utilities for implementing blockwise quantization using tensor splitting approach """

import torch
from aimet_torch import elementwise_ops, utils
from aimet_torch.quantsim import QuantizationSimModel


class BlockwiseLinear(torch.nn.Module):
    """
    Blockwise Linear implementation.
    This module replaces a single nn.Linear module, breaking it into a number of smaller linear modules depending on
    block size. Each separate linear module can operate with per channel quantization independently, and the outputs
    of each linear module are summed up.
    """
    def __init__(self, linear_module: torch.nn.Linear, block_size: int):
        super(BlockwiseLinear, self).__init__()
        self.block_size = block_size
        self.linears = torch.nn.ModuleList()
        self.elementwise_adds = torch.nn.ModuleList()
        split_indices = list(range(block_size, linear_module.weight.shape[1], block_size))
        self.split = elementwise_ops.Split()
        split_weights = torch.tensor_split(linear_module.weight, split_indices, 1)
        for idx, split_weight in enumerate(split_weights):
            linear = torch.nn.Linear(split_weight.shape[1],
                                     split_weight.shape[0],
                                     bias=(linear_module.bias is not None and idx == 0))
            linear.weight = torch.nn.Parameter(split_weight)
            if linear.bias is not None:
                linear.bias = linear_module.bias
            self.linears.append(linear)
            self.elementwise_adds.append(elementwise_ops.Add())
        self.elementwise_adds = self.elementwise_adds[:-1]
        if not self.elementwise_adds:
            self.elementwise_adds = None

    def forward(self, inp):
        """ Forward pass """
        if len(self.linears) == 1:
            return self.linears[0](inp)

        split_inputs = self.split(inp, self.block_size, -1)
        out = None
        for idx, split_input in enumerate(split_inputs):
            linear_out = self.linears[idx](split_input)
            if out is None:
                out = linear_out
            else:
                out = self.elementwise_adds[idx-1](out, linear_out)
        return out


def replace_linears_for_blockwise_quant(model: torch.nn.Module, block_size: int):
    """
    Replace all instances of torch.nn.Linears in model with equivalent BlockwiseLinear modules.
    The linear weights are split on the input channel dimension such that all constituent linear modules have weights
    with input channel dimension <= block_size.

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
        setattr(model, name, BlockwiseLinear(module, block_size))


def tie_blockwise_linear_quantizers(quantsim: QuantizationSimModel):
    """
    Tie all output quantizers within a BlockwiseLinear block together so they share the same encoding. In other words,
    all output quantizers of constituent linear layers as well as output quantizers of elementwise add modules will
    share the same quantization parameters.

    :param quantsim: Quantsim model containing BlockwiseLinear modules to tie output quantizers for.
    """
    for module in quantsim.model.modules():
        if isinstance(module, BlockwiseLinear) and module.elementwise_adds is not None:
            quantizer_to_use = module.elementwise_adds[-1].output_quantizers[0]
            for linear in module.linears:
                linear.output_quantizers[0] = quantizer_to_use
            for add in module.elementwise_adds:
                add.output_quantizers[0] = quantizer_to_use
