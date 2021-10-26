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

    def _broadcast_to_tensor(tensor, encoding):
        """
        This helper method takes n-dimension tensor and a 1-dimension encoding. And the encoding is broad-casted to
        match the n-dimensional tensor
        :param tensor: Tensor to use as target for the broadcasting operation
        :param encoding: Encoding 1-dimensional tensor to broadcast
        :return: Broad-casted tensor
        """
        shape = list(tensor.shape)
        encoding = torch.Tensor(encoding).to(x.device)                  # convert encoding to a tensor

        # Original tensor shape is OIHW/IOHW, we change the shape to IHWO. Encoding (which is of shape O) can naturally
        # broadcast to this shape
        # This will work if the original tensor shape was any dimensions as long as the first dimension matches the
        # encoding tensor shape
        num_channels = shape.pop(ch_axis)
        encoding = encoding * torch.ones(shape + [num_channels]).to(x.device)

        # we permute the resulting tensor back to OIHW/IOHW shape
        permute_dims = list(range(len(shape)))
        permute_dims.insert(ch_axis, len(shape))
        encoding = encoding.permute(permute_dims)

        return encoding

    # Broadcast the encoding min and max tensors if they were single dimensioned. If they were scalars, the
    # broadcast is automatic and more optimal in runtime, so we skip calling the helper above
    if isinstance(encoding_max, list) and len(x.shape) > 1:
        encoding_max = _broadcast_to_tensor(x, encoding_max)

    if isinstance(encoding_min, list) and len(x.shape) > 1:
        encoding_min = _broadcast_to_tensor(x, encoding_min)
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
