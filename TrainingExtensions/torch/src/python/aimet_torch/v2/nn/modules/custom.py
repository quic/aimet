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

from typing import Optional
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from aimet_torch.nn.modules.custom import * # pylint: disable=wildcard-import, unused-wildcard-import
from aimet_torch.v2.quantization.tensor import QuantizedTensorBase
from ..true_quant import (
    QuantizationMixin,
    _DispatchMixin,
    _quantize_if_applicable,
    _quantize_dequantize_if_applicable,
)


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


@QuantizationMixin.implements(Concat)
class QuantizedConcat(_DispatchMixin, QuantizationMixin, Concat):
    """ Quantized Concat """
    _builtin_torch_fn = torch.cat

    # pylint: disable=attribute-defined-outside-init
    def __quant_init__(self):
        super().__quant_init__()
        self._num_inputs = 1

    def export_input_encodings(self):
        """
        Extends super().export to repeat input quantizer's encodings :attr:`self._num_inputs` times
        """
        input_encodings = super().export_input_encodings()
        return input_encodings * self._num_inputs

    def import_input_encodings(self,
                               encodings,
                               strict: bool,
                               partial: bool,
                               requires_grad: Optional[bool],
                               allow_overwrite: bool):
        """
        Extends super().import_input_encodings to set `self._num_inputs` based on length of encodings.
        """
        self._num_inputs = len(encodings)
        super().import_input_encodings(encodings,
                                       strict=strict,
                                       partial=partial,
                                       requires_grad=requires_grad,
                                       allow_overwrite=allow_overwrite)

    def forward(self, *x): # pylint: disable=arguments-differ
        """
        Quantized forward impl for custom.Concat.
        """
        self._num_inputs = len(x)
        return super().forward(*x)

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        def cat(tensors, dim=0, *, out=None):
            input_qtzr = self.input_quantizers[0]
            tensors = tuple(_quantize_dequantize_if_applicable(x, input_qtzr) for x in tensors)
            output = fn(tensors, dim=dim, out=out)
            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return cat

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        def cat(tensors, dim=0, *, out=None):
            input_qtzr = self.input_quantizers[0]
            tensors = tuple(_quantize_if_applicable(x, input_qtzr) for x in tensors)
            output_encodings = self.output_quantizers[0].get_encoding() if self.output_quantizers[0] else None
            return fn(tensors, dim=dim, out=out, output_encodings=output_encodings)

        return cat
