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
""" Quantized definitions for custom modules of AIMET """

import torch
from torch import nn
import torch.nn.functional as F
from aimet_torch.nn.modules.custom import * # pylint: disable=wildcard-import, unused-wildcard-import
from ..true_quant import QuantizationMixin, _DispatchMixin


def _binary_quant_init(self):
    super(type(self), self).__quant_init__()
    self.input_quantizers = nn.ModuleList([None, None])


@QuantizationMixin.implements(Sin)
class QuantizedSin(_DispatchMixin, QuantizationMixin, Sin):
    """ Quantized Sin """
    _builtin_torch_fn = torch.sin


@QuantizationMixin.implements(Cos)
class QuantizedCos(_DispatchMixin, QuantizationMixin, Cos):
    """ Quantized Cos """
    _builtin_torch_fn = torch.cos


@QuantizationMixin.implements(AvgPool2d)
class QuantizedAvgPool2d(_DispatchMixin, QuantizationMixin, AvgPool2d):
    """ Quantized AvgPool2d """
    _builtin_torch_fn = F.avg_pool2d


@QuantizationMixin.implements(Reshape)
class QuantizedReshape(_DispatchMixin, QuantizationMixin, Reshape):
    """ Quantized Reshape """
    _builtin_torch_fn = torch.reshape


@QuantizationMixin.implements(RSqrt)
class QuantizedRSqrt(_DispatchMixin, QuantizationMixin, RSqrt):
    """ Quantized RSqrt """
    _builtin_torch_fn = torch.rsqrt


@QuantizationMixin.implements(MatMul)
class QuantizedMatMul(_DispatchMixin, QuantizationMixin, MatMul):
    """ Quantized MatMul """
    __quant_init__ = _binary_quant_init
    _builtin_torch_fn = torch.matmul


@QuantizationMixin.implements(Add)
class QuantizedAdd(_DispatchMixin, QuantizationMixin, Add):
    """ Quantized Add """
    __quant_init__ = _binary_quant_init
    _builtin_torch_fn = torch.add


@QuantizationMixin.implements(Multiply)
class QuantizedMultiply(_DispatchMixin, QuantizationMixin, Multiply):
    """ Quantized Multiply """
    __quant_init__ = _binary_quant_init
    _builtin_torch_fn = torch.mul


@QuantizationMixin.implements(Subtract)
class QuantizedSubtract(_DispatchMixin, QuantizationMixin, Subtract):
    """ Quantized Subtract """
    __quant_init__ = _binary_quant_init
    _builtin_torch_fn = torch.sub


@QuantizationMixin.implements(Divide)
class QuantizedDivide(_DispatchMixin, QuantizationMixin, Divide):
    """ Quantized Divide """
    __quant_init__ = _binary_quant_init
    _builtin_torch_fn = torch.div
