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
""" Implements straight through gradient computation for Quant op"""

import torch


def broadcast_to_tensor(tensor, encoding, ch_axis):
    """
    This helper method takes n-dimension tensor and a 1-dimension encoding. And the encoding is broad-casted to
    match the n-dimensional tensor
    :param tensor: Tensor to use as target for the broadcasting operation
    :param encoding: Encoding 1-dimensional tensor to broadcast
    :param ch_axis: Channel axis along which broadcasting happens
    :return: Broad-casted tensor
    """
    if not isinstance(encoding, torch.Tensor):
        encoding = torch.Tensor(encoding).to(tensor.device)                  # convert encoding to a tensor

    # Original tensor shape is OIHW/IOHW, we change the shape to IHWO. Encoding (which is of shape O) can naturally
    # broadcast to this shape
    # This will work if the original tensor shape was any dimensions as long as the first dimension matches the
    # encoding tensor shape
    shape = list(tensor.shape)
    num_channels = shape.pop(ch_axis)
    encoding = encoding * torch.ones(shape + [num_channels]).to(tensor.device)

    # we permute the resulting tensor back to OIHW/IOHW shape
    permute_dims = list(range(len(shape)))
    permute_dims.insert(ch_axis, len(shape))
    encoding = encoding.permute(permute_dims)

    return encoding


def compute_dloss_by_dmin_dmax(x, grad, scaling, offset, n, p, ch_axis=0):
    """
    implemenatation based on LSQ+
    reference: https://arxiv.org/pdf/2004.09576.pdf

    in_cond:
        dqdmax = (round(x/s) - x/s) / p
        dqdmin = (round(x/s) - x/s) / -p

    out_cond (n>fw):
        dqdmax = n/p + o/p - round(o)/p
        dqdmin = -n/p + 1 - (o/p - round(o)/p)

    """

    fw = torch.round(x / scaling) + torch.round(offset)
    rounding_error_q = torch.round(x / scaling) - (x / scaling)
    rounding_error_o = torch.round(offset) - offset

    dqdmax = torch.where(
        torch.le(fw.data, p), rounding_error_q / p, torch.ones_like(p) - rounding_error_o / p,
    )
    dqdmax = torch.where(
        torch.le(n, fw.data), dqdmax, n / p - rounding_error_o / p,
    )

    dqdmin = torch.where(
        torch.le(fw.data, p), -rounding_error_q / p, rounding_error_o / p
    )
    dqdmin = torch.where(
        torch.le(n, fw.data), dqdmin, -n/p + 1 + rounding_error_o / p
    )
    dloss_by_dmax = dqdmax * grad
    dloss_by_dmin = dqdmin * grad
    if len(scaling) > 1 and len(x.shape) > 1:
        dim = list(range(len(x.shape)))
        # Remove the output axis
        dim.pop(ch_axis)
        dloss_by_dmax = torch.sum(dloss_by_dmax, dim=dim)
        dloss_by_dmin = torch.sum(dloss_by_dmin, dim=dim)
    elif len(scaling) == 1:
        dloss_by_dmax = torch.sum((dloss_by_dmax).flatten(), dim=0, keepdim=True)
        dloss_by_dmin = torch.sum((dloss_by_dmin).flatten(), dim=0, keepdim=True)

    return dloss_by_dmin, dloss_by_dmax


def compute_dloss_by_dx(x, grad, encoding_min, encoding_max, ch_axis=0):
    """
    compute derivative w.r.t input using straight through estimator.
    :param x: input tensor
    :param grad: gradient flowing
    :param encoding_min: encoding min grid param used on forward pass
    :param encoding_max: encoding max grid param used on forward pass
    :param ch_axis: Channel axis to use for per-channel quant
    :return: gradient w.r.t input
    """

    # Broadcast the encoding min and max tensors if they were single dimensioned. If they were scalars, the
    # broadcast is automatic and more optimal in runtime, so we skip calling the helper above
    if isinstance(encoding_max, list) and len(x.shape) > 1:
        encoding_max = broadcast_to_tensor(x, encoding_max, ch_axis)

    if isinstance(encoding_min, list) and len(x.shape) > 1:
        encoding_min = broadcast_to_tensor(x, encoding_min, ch_axis)
    else:
        encoding_min = torch.Tensor([encoding_min]).to(x.device)

    # compute dloss_by_dx = dq_by_dx * grad
    inner_cond = torch.where(torch.le(x, encoding_max),  # condition to check per value
                             torch.ones_like(x),  # execute if true
                             torch.zeros_like(x))  # execute if false

    dloss_by_dx = torch.where(torch.le(encoding_min, x),  # condition to check per value
                              inner_cond,  # execute if true
                              torch.zeros_like(x)) * grad

    return dloss_by_dx


def compute_dloss_by_dx_using_scale_offset(x, grad, scaling, offset, n, p):
    """
    compute derivative w.r.t input
    :param grad: gradient
    :param scaling: scaling factor computed for given encoding min/max
    :param offset: offset computed
    :param n: lower bound
    :param p: upper bound
    :return: gradient w.r.t input
    """

    # R(x/s) + R(o)
    r_x_by_s_plus_round_o = torch.round(x / scaling) + offset

    # compute dloss_by_dx = dq_by_dx * grad
    inner_cond = torch.where(torch.le(r_x_by_s_plus_round_o.data, p.data),  # condition to check per value
                             torch.ones_like(r_x_by_s_plus_round_o),  # execute if true
                             torch.zeros_like(r_x_by_s_plus_round_o))  # execute if false

    dloss_by_dx = torch.where(torch.le(n.data, r_x_by_s_plus_round_o.data),  # condition to check per value
                              inner_cond,  # execute if true
                              torch.zeros_like(r_x_by_s_plus_round_o.data)) * grad

    return dloss_by_dx


class RoundStraightThrough(torch.autograd.Function):
    """
    Defining gradient of rounding function as passthrogh since round is a non-linearity
    """

    @staticmethod
    # pylint: disable=arguments-differ
    def forward(ctx, *x):
        return torch.round(*x)

    @staticmethod
    def backward(ctx, *output_grad):
        return output_grad
