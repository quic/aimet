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

import pytest
import torch
import torch.nn.functional as F
from aimet_torch.experimental.v2.quantization.backends import get_backend
from aimet_torch.experimental.v2.quantization.modules.quantize import QuantizeDequantize
from aimet_torch.experimental.v2.nn.fake_quant import FakeQuantizedSoftmax
from aimet_torch.experimental.v2.quantization.encoding_analyzer import CalibrationMethod


@pytest.fixture
def input():
    return torch.arange(100).view(10, 10) / 100


class TestFakeQuantizedSoftmax:
    def test_no_spec(self, input):
        """
        Given: Instantiate a fake-, true-, or auto-quantized module without specific spec
        """

        """
        When: Inspect `*_quantizer` attributes
        Then: All of them are set to `None`
        """
        quant_softmax = FakeQuantizedSoftmax()
        assert quant_softmax.input_quantizers[0] is None
        assert quant_softmax.output_quantizers[0] is None

        """
        When: Run forward with an input x
        Then: The output should be equal to that of the base FP module.
        """
        expected_output = F.softmax(input, quant_softmax.dim)
        assert torch.equal(quant_softmax(input), expected_output)

    def test_input_qtzn(self, input):
        """
        Given: Instantiate a fake-quantized module with input quantizer spec specified
        """
        quant_softmax = FakeQuantizedSoftmax()
        quant_softmax.input_quantizers[0] = QuantizeDequantize((1,),
                                                               bitwidth=8,
                                                               symmetric=False,
                                                               qscheme=CalibrationMethod.MinMax)

        """
        When: Inspect `input_quantizer` attribute.
        Then: `input_quantizer` is set to `QuantizeDequantize` as a submodule
        """
        assert isinstance(quant_softmax.input_quantizers[0], QuantizeDequantize)
        assert quant_softmax.output_quantizers[0] is None

        """
        When: Invoke forward before the encodings are initialized with `compute_encodings()`
        Then: Throw runtime error
        """
        with pytest.raises(RuntimeError):
            _ = quant_softmax(input)

        """
        When: Invoke forward with input x after encodings are initialized
              with `compute_encodings()`
        Then: The output should be equal to FP softmax of quantize-dequantized x
        """
        with quant_softmax.compute_encodings():
            _ = quant_softmax(input)

        quant_output = quant_softmax(input)

        scale = quant_softmax.input_quantizers[0].get_scale()
        offset = quant_softmax.input_quantizers[0].get_offset()
        bitwidth = quant_softmax.input_quantizers[0].bitwidth
        input_qdq = get_backend().quantize_dequantize(input, scale, offset, bitwidth)

        expected_output = F.softmax(input_qdq, quant_softmax.dim)
        assert torch.equal(quant_output, expected_output)

    def test_output_qtzn(self, input):
        """
        Given: Instantiate a fake-quantized module with output quantizer spec specified
        """
        quant_softmax = FakeQuantizedSoftmax()
        quant_softmax.output_quantizers[0] = QuantizeDequantize((1,),
                                                                bitwidth=8,
                                                                symmetric=False,
                                                                qscheme=CalibrationMethod.MinMax)

        """
        When: Inspect `output_quantizer` attribute.
        Then: `output_quantizer` is set to `QuantizeDequantize` as a submodule
        """
        assert quant_softmax.input_quantizers[0] is None
        assert isinstance(quant_softmax.output_quantizers[0], QuantizeDequantize)

        """
        When: Invoke forward before the encodings are initialized with `compute_encodings()`
        Then: Throw runtime error
        """
        with pytest.raises(RuntimeError):
            _ = quant_softmax(input)

        """
        When: Invoke forward with input x after encodings are initialized
              with `compute_encodings()`
        Then: The output should be equal to quantize-dequantized FP softmax output
        """
        with quant_softmax.compute_encodings():
            _ = quant_softmax(input)

        quant_output = quant_softmax(input)

        scale = quant_softmax.output_quantizers[0].get_scale()
        offset = quant_softmax.output_quantizers[0].get_offset()
        bitwidth = quant_softmax.output_quantizers[0].bitwidth

        fp_output = F.softmax(input, quant_softmax.dim)
        expected_output = get_backend().quantize_dequantize(fp_output,
                                                            scale,
                                                            offset,
                                                            bitwidth)
        assert torch.equal(quant_output, expected_output)
