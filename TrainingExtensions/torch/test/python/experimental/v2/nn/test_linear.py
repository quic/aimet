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
from aimet_torch.experimental.v2.quantization.fake_quant import FakeQuantizedLinear, _ModuleSpec, _TensorSpec


@pytest.fixture
def input():
    return torch.arange(-5, 5) / 10


@pytest.fixture
def input_spec():
    return [_TensorSpec((1,), 4, False, 'minmax')]


@pytest.fixture
def output_spec():
    return [_TensorSpec((1,), 4, False, 'minmax')]


@pytest.fixture
def param_spec():
    return {
        'weight': _TensorSpec((1,), 4, True, 'minmax'),
        'bias': None,
    }


class TestFakeQuantizedLinear:
    def test_no_spec(self, input):
        quant_linear = FakeQuantizedLinear(10, 10)
        assert quant_linear.input_quantizers is None
        assert quant_linear.output_quantizers is None
        assert quant_linear.param_quantizers['weight'] is None
        assert quant_linear.param_quantizers['bias'] is None

        expected_output = F.linear(input, quant_linear.weight, quant_linear.bias)
        assert torch.equal(quant_linear(input), expected_output)

    def test_input_qtzn(self, input, input_spec):
        """
        Given: Instantiate a fake-quantized module with input quantizer spec specified
        """
        module_spec = _ModuleSpec(input_spec=input_spec,
                                  param_spec=None,
                                  output_spec=None)
        quant_linear = FakeQuantizedLinear(10, 10, spec=module_spec)
        """
        When: Inspect `input_quantizer` attribute.
        Then: `input_quantizer` is set to `QuantizeDequantize` as a submodule
        """
        assert isinstance(quant_linear.input_quantizers[0], QuantizeDequantize)
        assert quant_linear.param_quantizers['weight'] is None
        assert quant_linear.param_quantizers['bias'] is None
        assert quant_linear.output_quantizers is None

        """
        When: Invoke forward before the encodings are initialized with `compute_encodings()`
        Then: Throw runtime error
        """
        with pytest.raises(RuntimeError):
            _ = quant_linear(input)

        """
        When: Invoke forward with input x after encodings are initialized
              with `compute_encodings()`
        Then: The output should be equal to FP linear with quantize-dequantized x
        """
        with quant_linear.compute_encodings():
            _ = quant_linear(input)

        quant_output = quant_linear(input)

        scale = quant_linear.input_quantizers[0].get_scale()
        offset = quant_linear.input_quantizers[0].get_offset()
        bitwidth = quant_linear.input_quantizers[0].bitwidth
        input_qdq = get_backend().quantize_dequantize(input, scale, offset, bitwidth)

        expected_output = F.linear(input_qdq, quant_linear.weight, quant_linear.bias)
        assert torch.equal(quant_output, expected_output)

    def test_output_qtzn(self, input, output_spec):
        """
        Given: Instantiate a fake-quantized module with output quantizer spec specified
        """
        module_spec = _ModuleSpec(input_spec=None,
                                  param_spec=None,
                                  output_spec=output_spec)
        quant_linear = FakeQuantizedLinear(10, 10, spec=module_spec)

        """
        When: Inspect `output_quantizer` attribute.
        Then: `output_quantizer` is set to `QuantizeDequantize` as a submodule
        """
        assert quant_linear.input_quantizers is None
        assert isinstance(quant_linear.output_quantizers, QuantizeDequantize)
        assert quant_linear.param_quantizers['weight'] is None
        assert quant_linear.param_quantizers['bias'] is None

        """
        When: Invoke forward before the encodings are initialized with `compute_encodings()`
        Then: Throw runtime error
        """
        with pytest.raises(RuntimeError):
            _ = quant_linear(input)

        """
        When: Invoke forward with input x after encodings are initialized
              with `compute_encodings()`
        Then: The output should be equal to quantize-dequantized FP linear output
        """
        with quant_linear.compute_encodings():
            _ = quant_linear(input)

        quant_output = quant_linear(input)

        scale = quant_linear.output_quantizers[0].get_scale()
        offset = quant_linear.output_quantizers[0].get_offset()
        bitwidth = quant_linear.output_quantizers[0].bitwidth

        fp_output = F.linear(input, quant_linear.weight, quant_linear.bias)
        expected_output = get_backend().quantize_dequantize(fp_output,
                                                            scale,
                                                            offset,
                                                            bitwidth)
        assert torch.equal(quant_output, expected_output)

    def test_param_qtzn(self, input, param_spec):
        """
        Given: Instantiate a fake-quantized module with weight quantizer spec specified
        """
        module_spec = _ModuleSpec(input_spec=None,
                                  param_spec=param_spec,
                                  output_spec=None)
        quant_linear = FakeQuantizedLinear(10, 10, spec=module_spec)

        """
        When: Inspect `weight_quantizer` attribute.
        Then: `weight_quantizer` is set to `QuantizeDequantize` as a submodule
        """
        assert quant_linear.input_quantizers is None
        assert quant_linear.output_quantizers is None
        assert isinstance(quant_linear.param_quantizers['weight'], QuantizeDequantize)
        assert quant_linear.param_quantizers['bias'] is None

        with quant_linear.compute_encodings():
            _ = quant_linear(input)

        quant_output = quant_linear(input)

        weight = quant_linear.weight
        scale = quant_linear.param_quantizers['weight'].get_scale()
        offset = quant_linear.param_quantizers['weight'].get_offset()
        bitwidth = quant_linear.param_quantizers['weight'].bitwidth
        weight_qdq = get_backend().quantize_dequantize(weight, scale, offset, bitwidth)

        expected_output = F.linear(input, weight_qdq, quant_linear.bias)
        assert torch.equal(quant_output, expected_output)
