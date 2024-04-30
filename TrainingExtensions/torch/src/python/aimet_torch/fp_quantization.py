# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" FP8 quantization and supporting range setting functions """

import torch
import numpy as np

from aimet_common.defs import QuantScheme


NUM_MANTISSA_BITS = 3


def init_minmax(tensor, tensor_quantizer, per_channel):
    """
    Minmax initialization.
    """
    tensor = torch.abs(tensor)
    if per_channel:
        channel = tensor_quantizer.channel_axis

        for c in reversed(range(len(tensor.shape))):
            if c != channel:
                tensor = tensor.max(c)[0]

        tensor_quantizer.fp8_maxval = tensor.clone().to(tensor.device)
        maxval = tensor

    else:
        maxval = max(torch.abs(tensor.min()), tensor.max()).to(tensor.device)

    return maxval


def mantissa_bits_to_device(tensor):
    """
    Ensure NUM_MANTISSA_BITS is copied to the same device as tensor (only once)
    """
    global NUM_MANTISSA_BITS  # pylint: disable=global-statement
    if not isinstance(NUM_MANTISSA_BITS, torch.Tensor):
        NUM_MANTISSA_BITS = torch.Tensor([NUM_MANTISSA_BITS]).to(tensor.device)


def init_mse(tensor, tensor_quantizer, per_channel):
    """
    MSE initialization. Nearly equivalent to tf_enhanced
    """
    channel = tensor_quantizer.channel_axis
    if per_channel:
        mxs = [torch.max(torch.abs(xc.min()), torch.abs(xc.max())) for xc in tensor.split(1, dim=channel)]
        lsp = [torch.linspace(0.1 * mx.item(), 1.2 * mx.item(), 111) for mx in mxs]
        # 111 x n_channels (or 1 in case not --per-channel)
        linspaces = torch.stack(lsp).to(tensor.device).transpose(0, 1)
    else:
        mx = torch.max(torch.abs(tensor.min()), torch.abs(tensor.max()))
        lsp = [torch.linspace(0.1 * mx.item(), 1.2 * mx.item(), 111)]
        # 111 x 1
        linspaces = torch.stack(lsp).to(tensor.device).transpose(0, 1)

    mses = torch.zeros_like(linspaces)

    meandims = list(torch.arange(len(tensor.shape)))
    if per_channel:
        meandims.remove(channel)

    mantissa_bits_to_device(tensor)

    for i, maxval in enumerate(linspaces):
        xfp = quantize_to_fp8(tensor, maxval, NUM_MANTISSA_BITS, channel)
        mse = ((tensor - xfp) ** 2).mean(dim=meandims)
        mses[i, :] = mse

    best_mse = mses.argmin(0)
    maxval = torch.tensor([linspaces[best_mse[i], i] for i in range(linspaces.shape[-1])]).to(tensor.device)

    return maxval


def init_percentile(*_):
    """
    Percentile range initialization
    """
    raise NotImplementedError("Percentile scheme is not supported for FP8")


INIT_MAP = {
    QuantScheme.post_training_tf: init_minmax, # minmax
    QuantScheme.post_training_tf_enhanced: init_mse, # MSE
    QuantScheme.post_training_percentile: init_percentile
}


def fp8_quantizer(tensor, tensor_quantizer, _):
    """
    FP8 quantization entry function.
    """
    mantissa_bits_to_device(tensor)

    if not hasattr(tensor_quantizer, 'fp8_maxval') or tensor_quantizer.fp8_maxval is None:
        raise ValueError('tensor_quantizer.fp8_maxval not present or not initialized!')

    return quantize_to_fp8(
        tensor, tensor_quantizer.fp8_maxval, NUM_MANTISSA_BITS, tensor_quantizer.channel_axis)


def quantize_to_fp8(x_float: torch.Tensor,
                    maxval: torch.Tensor,
                    mantissa_bits: torch.Tensor,
                    per_channel_axis: int = 0,
                    ) -> torch.Tensor:
    """
    FP8 quantizer that exploits the fact that FP quantization is just INT quantization with
    scales that depend on the input.

    """
    # For learning: ensure that the number of mantissa bits is rounded and clamped to
    # allowed values. NB: torch.round should be replaced by ste_round_func (not included
    # here yet)
    # TODO for learning we need this as well:
    # mantissa_bits = torch.clamp(torch.round(mantissa_bits), 1, 7)

    # Compute exponent bits from the (learned) number of exponent bits. NB: assumes FP8
    exponent_bits = 7 - mantissa_bits

    # Tensorized per-channel quantization: ensure that maxval has the same number of
    # dimensions as x, where the channel that is individually quantized has size C,
    # and all other channels have size 1. E.g. for a conv tensor with C output channels,
    # maxval will have shape [C, 1, 1, 1]. This allows broadcasting maxval over the
    # input tensor in steps below.
    if maxval.shape and maxval.shape[0] != 1 and len(maxval.shape) != len(x_float.shape):
        new_shape = [1] * len(x_float.shape)
        new_shape[per_channel_axis] = -1
        maxval = maxval.view(new_shape)

    return fake_cast_to_ieee_float(x_float, maxval, exponent_bits, mantissa_bits)


def fake_cast_to_ieee_float(x_float, maxval, exponent_bits, mantissa_bits):
    """
    Fake-cast to the given exponent and mantissa bits based on IEEE float representation.
    IEEE float representation follows the following equation:
      maximum_representiable_value = (2 - 2**-M) * 2 ** (2**E - bias - 2)
      (E: exponent bits, M: mantissa bits)

    This function derives the bias from exponent bits, mantissa bits, and
    maximum representable value based on the above equation.
    """
    def log2(x):
        if isinstance(x, torch.Tensor):
            return torch.log2(x)
        return np.log2(x)
    # Math explanation of what happens here:
    # Bias is computed from maxval: $B=2^E - \log_2(M) + \log_2(2 - 2^{-M}) - 1$
    # This follows from maxval $M=(2 - 2^{-M}) \cdot 2^{2^E-1-B}$.
    bias = 2 ** exponent_bits - log2(maxval) + log2(2 - 2 ** (-mantissa_bits)) - 1

    # Ensure no values are greater than the maximum value represented by an 8 bit float system
    # with M mantissa and E exponent bits. torch.min/torch.max are used to allow gradients to
    # flow to maxval
    x_clipped = x_float.clamp(-maxval, maxval)

    # FP quantization scale is determined per-element, and is computed as
    # \log_2 s = \left\lfloor \log_2 |x_c| + B \right\rfloor - M - B
    # the addition of bias inside the floor and subtraction outside ensures that a
    # tensor scaling $\alpha \neq 1$ is correctly incorporated
    log_scales = torch.floor(log2(torch.abs(x_clipped)) + bias).detach()

    # This ensures scales are never smaller than the subnormal scale
    log_scales = torch.clamp(log_scales, 1.)

    # Second step of computing scale $s$
    scales = 2. ** (log_scales - mantissa_bits - bias)

    # Using the per-element scale we can quantize the clipped input tensor to the FP grid
    return torch.round(x_clipped / scales) * scales
