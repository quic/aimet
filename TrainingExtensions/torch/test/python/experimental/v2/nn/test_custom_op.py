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

import pytest
import torch
from aimet_torch.v2.nn.fake_quant import FakeQuantizationMixin


class CustomOp(torch.nn.Module):
    """Dummy custom module"""
    def forward(self, input):
        return input * 2 + 1


class TestFakeQuantizedCustomOp:
    def test_custom_op_from_module_unregistered(self):
        with pytest.raises(RuntimeError):
            _ = FakeQuantizationMixin.from_module(CustomOp())

    def test_custom_op_from_module_registered(self):
        try:
            @FakeQuantizationMixin.implements(CustomOp)
            class FakeQuantizedCustomOp(FakeQuantizationMixin, CustomOp):
                def quantized_forward(self, x):
                    x = super().forward(x)
                    return self.output_quantizers[0](x)

            quantized_custom_op = FakeQuantizationMixin.from_module(CustomOp())
            assert isinstance(quantized_custom_op, FakeQuantizedCustomOp)

            quantized_custom_op_ = FakeQuantizationMixin.from_module(CustomOp())
            assert isinstance(quantized_custom_op_, FakeQuantizedCustomOp)

        finally:
            # Unregister CustomOp so as not to affect other test functions
            FakeQuantizationMixin.cls_to_qcls.pop(CustomOp)

    def test_custom_op_wrap_registered(self):
        try:
            @FakeQuantizationMixin.implements(CustomOp)
            class FakeQuantizedCustomOp(FakeQuantizationMixin, CustomOp):
                def quantized_forward(self, x):
                    x = super().forward(x)
                    return self.output_quantizers[0](x)

            quantized_custom_op_cls = FakeQuantizationMixin.wrap(CustomOp)
            assert quantized_custom_op_cls is FakeQuantizedCustomOp

            quantized_custom_op_cls_ = FakeQuantizationMixin.wrap(CustomOp)
            assert quantized_custom_op_cls_ is FakeQuantizedCustomOp

        finally:
            # Unregister CustomOp so as not to affect other test functions
            FakeQuantizationMixin.cls_to_qcls.pop(CustomOp)
