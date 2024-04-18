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
""" Utilities for implementing blockwise quantization using batched matmul approach """

import functools
from typing import List, Union
import torch

import aimet_common.AimetTensorQuantizer as AimetTensorQuantizer
import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme, QuantizationDataType, MAP_QUANT_SCHEME_TO_PYMO
from aimet_torch import utils
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.quantsim_straight_through_grad import get_computed_encodings
from aimet_torch.tensor_quantizer import StaticGridTensorQuantizer, StaticGridPerChannelQuantizer


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

class BatchMatMulWithWeight(torch.nn.Module):
    """
    BatchMatMulWithWeight implementation
    """
    def __init__(self, weight: torch.Tensor, weight_first: bool):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.weight_first = weight_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: Input tensor
        :return: Output of forward pas
        """
        if self.weight_first:
            return torch.matmul(self.weight, x)
        return torch.matmul(x, self.weight)


class LinearWithMatMul(torch.nn.Module):
    '''
    Linear(out_feature, in_feature)
    x:[n, seq, in_feature] * w.T:[in_feature, out_feature] = [n, seq, out_feature]
    where, in_feature = num_blocks * block_size
    Approach #1: weight is the 1st input of torch.matmul
    w:[out_feature, in_feature].reshape()
    => [out_feature, num_blocks, block_size].permute(1, 0, 2)
    => [num_blocks, out_feature, block_size]
    x:[n, seq, in_feature].reshape()
    => [n * seq, num_blocks, block_size].permute(1, 2, 0)
    => [num_blocks, block_size, n * seq]
    torch.matmul(w, x)
    => [num_blocks, out_feature, n * seq]
    => [num_blocks, out_feature, n * seq].sum(dim=0)
    => [out_feature, n * seq].reshape()
    => [out_feature, n, seq].permute(1, 2, 0)
    Approach #2: weight is the 2nd input of torch.matmul
    w:[out_feature, in_feature].reshape()
    => [out_feature, num_blocks, block_size].permute(1, 2, 0)
    => [num_blocks, block_size, out_feature]
    x:[n, seq, in_feature].reshape()
    => [n * seq, num_blocks, block_size].permute(1, 0, 2)
    => [num_blocks, n * seq, block_size]
    torch.matmul(x, w)
    => [num_blocks, n * seq, out_feature].sum(dim=0)
    => [1, n* seq, out_feature].reshape()
    => [n, seq, out_feature]
    '''
    def __init__(self, linear: torch.nn.Module, block_size: int, weight_first=True):
        super().__init__()
        _, in_feature = linear.weight.shape
        assert in_feature % block_size == 0, f"in_feature:{in_feature} should be divisible by block_size:{block_size}"

        self.block_size = block_size
        self.num_blocks = in_feature // block_size

        self.weight_first = weight_first
        if self.weight_first:
            weight = linear.weight.detach().reshape(-1, self.num_blocks, block_size).permute(1, 0, 2)
        else:
            weight = linear.weight.detach().reshape(-1, self.num_blocks, block_size).permute(1, 2, 0)

        self.batch_matmul = BatchMatMulWithWeight(weight, weight_first=self.weight_first)
        self.reducesum = ReduceSum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: Input tensor
        :return: Output of forward pas
        """

        batch, seq = x.shape[:2]

        if self.weight_first:
            x = x.reshape(batch * seq, self.num_blocks, self.block_size).permute(1, 2, 0)
            x = self.batch_matmul(x)
            x = self.reducesum(x).reshape(batch, seq, -1)
        else:
            x = x.reshape(batch * seq, self.num_blocks, self.block_size).permute(1, 0, 2)
            x = self.batch_matmul(x)
            x = self.reducesum(x).reshape(batch, seq, -1)
        return x

class StaticGridMultiDimPerChannelQuantizer(StaticGridTensorQuantizer):
    """
    Quantizer allowing multiple channels to be specified as channel axes. The number of channels will be the product of
    sizes of all channels specified as channel axes.
    """
    # pylint: disable=too-many-arguments
    def __init__(self, bitwidth: int, round_mode: libpymo.RoundingMode, quant_scheme: QuantScheme,
                 use_symmetric_encodings: bool, num_channels: int, enabled_by_default: bool, ch_axes: Union[int, list],
                 data_type: QuantizationDataType = QuantizationDataType.int):
        super().__init__(
            bitwidth, round_mode, quant_scheme, use_symmetric_encodings, enabled_by_default, data_type=data_type)

        quant_scheme = MAP_QUANT_SCHEME_TO_PYMO[quant_scheme]
        self._cppOp = [AimetTensorQuantizer.AimetTensorQuantizer(quant_scheme) for _ in range(num_channels)]
        if not isinstance(ch_axes, list):
            self._ch_axes = [ch_axes]
        else:
            self._ch_axes = ch_axes
        self._permute_order = None
        self._encoding_reshape_order = None

    @property
    def encoding(self) -> Union[None, List[libpymo.TfEncoding]]:
        """
        Property to get encoding.

        :return: List of Encoding(s).
        """
        return self._encoding

    @encoding.setter
    def encoding(self, encoding: List[libpymo.TfEncoding]):
        """
        Property to set encoding.

        :param encoding: List of Encoding(s).
        """
        if self._is_encoding_frozen:
            raise RuntimeError("Encoding can be set only when it is not frozen.")

        self._encoding = encoding

    def _get_permute_order(self, num_dims: int) -> List[int]:
        """
        Return the permute order for permuting all channel axes to the front dimensions.

        :param num_dims: Number of dimensions of tensor to permute
        :return: Tensor permute order
        """
        if self._permute_order is not None:
            return self._permute_order

        remaining_dims = set(range(num_dims))
        permute_order = []
        for ch_axis in self._ch_axes:
            assert ch_axis < num_dims
            permute_order.append(ch_axis)
            remaining_dims.remove(ch_axis)
        for i in range(num_dims):
            if i in remaining_dims:
                permute_order.append(i)
        self._permute_order = permute_order
        return self._permute_order

    def _get_encoding_reshape_order(self, tensor_shape: torch.Size) -> List[int]:
        """
        Return the reshape order encodings should be reshaped to. Indices corresponding to channel axes will have proper
        sizes and all remaining indices will be of size 1.

        :param tensor_shape: Shape of tensor to quantize
        :return: Encoding reshape order
        """
        if self._encoding_reshape_order is not None:
            return self._encoding_reshape_order

        encoding_reshape_order = []
        for idx, _ in enumerate(tensor_shape):
            if idx in self._ch_axes:
                encoding_reshape_order.append(tensor_shape[idx])
            else:
                encoding_reshape_order.append(1)
        self._encoding_reshape_order = encoding_reshape_order
        return self._encoding_reshape_order

    def update_encoding_stats(self, tensor: torch.Tensor):
        """
        Update the stats for computing encodings. In the case of fp16 or int with bw=32, this is skipped.

        :param tensor: Tensor to use for updating the encodings stats
        """
        if self.enabled and not self._is_encoding_frozen:
            if self.bitwidth == 32:
                return
            if self.data_type == QuantizationDataType.float:
                raise AssertionError('Float quantization not supported')

            if self.encoding_min_max_fixed_vals is not None:
                # pylint: disable=unsubscriptable-object
                tensor = torch.tensor([self.encoding_min_max_fixed_vals[0],
                                       self.encoding_min_max_fixed_vals[1]])

                for op in self._cppOp:
                    op.updateStats(tensor, tensor.is_cuda)
            else:
                # If we have a half-float tensor, just upcast it to float
                # Need to change this when libpymo can handle half-float tensors
                if tensor.dtype == torch.float16:
                    tensor = tensor.to(torch.float32)

                # Permute tensor so all channels to coalesce are at the front dimensions
                tensor_slice = torch.permute(tensor, dims=self._get_permute_order(len(tensor.shape)))

                # Reshape tensor to combine all channels to coalesce into a single dimension; all remaining axes
                # are also coalesced into a second dimension
                tensor_slice = torch.reshape(tensor_slice,
                                             (functools.reduce(lambda x, y: x * y,
                                                               tensor_slice.shape[:len(self._ch_axes)]),
                                              -1))
                for channel_idx, op in enumerate(self._cppOp):
                    channel_slice = tensor_slice.select(0, channel_idx).contiguous(
                        memory_format=torch.contiguous_format)
                    op.updateStats(channel_slice, tensor.is_cuda)

    def quantize_dequantize(self, tensor: torch.Tensor, round_mode: libpymo.RoundingMode) -> torch.Tensor:
        """
        Quantize-dequantize the tensor, using the saved encoding for this tensor

        :param tensor: Tensor to quantize-dequantize
        :param round_mode: Rounding mode
        :return: Resulting tensor
        """
        encoding_mins = [encoding.min for encoding in self._encoding]
        encoding_maxes = [encoding.max for encoding in self._encoding]
        encoding_min_tensor = torch.tensor(encoding_mins, device=tensor.device)
        encoding_max_tensor = torch.tensor(encoding_maxes, device=tensor.device)
        delta, offset, num_steps = get_computed_encodings(self.bitwidth, encoding_min_tensor, encoding_max_tensor,
                                                          self.use_symmetric_encodings, self.use_strict_symmetric,
                                                          self.is_unsigned_symmetric)
        offset = offset.expand_as(delta)     # In case of symmetric encodings, offset will be a scalar before expansion
        delta = torch.reshape(delta, self._get_encoding_reshape_order(tensor.shape))
        offset = torch.reshape(offset, self._get_encoding_reshape_order(tensor.shape))

        zero = torch.zeros_like(num_steps)

        x_round = torch.round(tensor / delta) - offset
        x_quant = x_round.clamp(zero, num_steps)
        return (x_quant + offset) * delta

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
        setattr(model, name, LinearWithMatMul(module, block_size))

def enable_bmm_per_channel_quantizers(quantsim: QuantizationSimModel):
    """
    Loop through quantsim modules and replace weight param quantizers of BatchMatMulWithWeight modules with multi
    channel quantizers.

    :param quantsim: Quantsim containing BatchMatMulWithWeight modules
    """
    for module in quantsim.model.modules():
        # pylint: disable=protected-access
        if isinstance(module, QcQuantizeWrapper) and isinstance(module._module_to_wrap, BatchMatMulWithWeight):
            if isinstance(module.param_quantizers['weight'], StaticGridMultiDimPerChannelQuantizer):
                continue

            if not isinstance(module.param_quantizers['weight'], StaticGridPerChannelQuantizer):
                module.enable_per_channel_quantization()

            # pylint: disable=protected-access
            module.param_quantizers['weight'] = create_multi_channel_quantizer(module.param_quantizers['weight'],
                                                                               module._module_to_wrap.weight_first,
                                                                               module._module_to_wrap.weight.shape)

def create_multi_channel_quantizer(per_channel_quantizer: StaticGridPerChannelQuantizer, weight_first: bool,
                                   weight_shape: torch.Size) -> StaticGridMultiDimPerChannelQuantizer:
    """
    Create multi channel quantizer given an original per channel quantizer.

    :param per_channel_quantizer: Original per channel quantizer to obtain quantizer settings from
    :param weight_first: True if the LinearWithMatMul is using weight first, False otherwise
    :param weight_shape: Shape of weight parameter
    :return: Multi channel quantizer
    """
    if weight_first:
        channel_axes = [0, 1]
        num_channels = weight_shape[0] * weight_shape[1]
    else:
        channel_axes = [0, 2]
        num_channels = weight_shape[0] * weight_shape[2]

    return StaticGridMultiDimPerChannelQuantizer(per_channel_quantizer.bitwidth,
                                                 per_channel_quantizer.round_mode,
                                                 per_channel_quantizer.quant_scheme,
                                                 per_channel_quantizer.use_symmetric_encodings,
                                                 num_channels,
                                                 per_channel_quantizer.enabled,
                                                 channel_axes,
                                                 per_channel_quantizer.data_type
                                                 )
