# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=all

import math
from typing import overload, Union, Tuple
import torch
from .utils import *


@overload
def quantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
             bitwidth: Union[int, float], signed: bool = False):
    ...

@overload
def quantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, *,
             num_bins: int, signed: bool = False):
    ...

@overload
def quantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, *,
             qmin: int, qmax: int):
    """
    return x_q := clamp(x/scale - offset, qmin, qmax)
    """
    ...


def quantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
             *args, **kwargs):
    qmin, qmax = _parse_args(args, kwargs)
    return get_backend().quantize(tensor, scale, offset, qmin, qmax)


@overload
def quantize_dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
                        bitwidth: Union[int, float], signed: bool = False):
    ...

@overload
def quantize_dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, *,
                        num_bins: int, signed: bool = False):
    ...

@overload
def quantize_dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, *,
                        qmin: int, qmax: int):
    """
    return x_qdq := (x_q + offset) * scale
        where x_q := clamp(x/scale - offset, qmin, qmax)
    """
    ...


def quantize_dequantize(tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
                        *args, **kwargs):
    qmin, qmax = _parse_args(args, kwargs)
    return get_backend().quantize_dequantize(tensor, scale, offset, qmin, qmax)


def _parse_args(args, kwargs) -> Tuple[int, int]:
    bitwidth = num_bins = signed = qmin = qmax = None

    if len(args) == 2:
        bitwidth, signed = args
    elif len(args) == 1:
        bitwidth = args[0]
        signed = kwargs['signed']
    else:
        if 'bitwidth' in kwargs:
            bitwidth, signed = kwargs['bitwidth'], kwargs['signed']
        elif 'num_bins' in kwargs:
            num_bins, signed = kwargs['num_bins'], kwargs['signed']
        else:
            qmin, qmax = kwargs['qmin'], kwargs['qmax']

    if bitwidth is not None:
        num_bins = 2 ** bitwidth - 1

    if num_bins is not None:
        if signed:
            qmin = -math.ceil(num_bins/2)
            qmax = math.floor(num_bins/2)
        else:
            qmin = 0
            qmax = num_bins

    assert qmin is not None
    assert qmax is not None

    return qmin, qmax
