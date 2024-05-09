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
from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize
from aimet_torch.v2.quantization.float.quantizer import _ieee_float_max_representable_value
from aimet_torch.fp_quantization import fake_cast_to_ieee_float


@pytest.fixture()
def x():
    """
    Returns [
        [-2., -1.99, -1.98, ..., -1.01],
        [-1., -0.99, -0.98, ..., -0.01],
        [ 0.,  0.01,  0.02, ...,  0.99],
        [ 1.,  1.01,  1.02, ...,  1.99],
    ]
    """
    return torch.arange(-200, 200).view(4, 100) / 100


@torch.no_grad()
def test_qdq_output_standard_dtypes(x):
    """
    Given: Instantiated FloatQuantizeDequantize with a well-known dtype of pytorch
    When: Run forward
    Then: Output should be equal to downcasting and upcasting the input
    """
    float16_qdq = FloatQuantizeDequantize(dtype=torch.float16)
    expected_output = x.to(torch.float16).float()
    assert torch.equal(float16_qdq(x), expected_output)

    bfloat16_qdq = FloatQuantizeDequantize(dtype=torch.bfloat16)
    expected_output = x.to(torch.bfloat16).float()
    assert torch.equal(bfloat16_qdq(x), expected_output)

    """
    Given: Instantiated two quantizers:
        - FloatQuantizeDequantize(dtype=dtype)
        - FloatQuantizeDequantize(exponent_bits, mantissa_bits)

        where exponent_bits and mantissa_bits corresponds to dtype
    When: Run forward
    Then: The two quantizers should produce same output
    """
    float16_qdq_1 = FloatQuantizeDequantize(dtype=torch.float16)
    float16_qdq_2 = FloatQuantizeDequantize(exponent_bits=5, mantissa_bits=10)
    assert torch.equal(float16_qdq_1(x), float16_qdq_2(x))

    bfloat16_qdq_1 = FloatQuantizeDequantize(dtype=torch.bfloat16)
    bfloat16_qdq_2 = FloatQuantizeDequantize(exponent_bits=8, mantissa_bits=7)
    assert torch.equal(bfloat16_qdq_1(x), bfloat16_qdq_2(x))


@torch.no_grad()
@pytest.mark.parametrize('exponent_bits', [3, 4])
@pytest.mark.parametrize('mantissa_bits', [3, 4])
def test_qdq_output_non_standard_dtypes(x, exponent_bits, mantissa_bits):
    """
    Given: Instantiated FloatQuantizeDequantize with a non-standard float dtype
    When: Run forward
    Then: Output should be equal to fake-casting the input to the non-standard float
    """
    float_qdq = FloatQuantizeDequantize(exponent_bits, mantissa_bits)
    max_representable_value = _ieee_float_max_representable_value(exponent_bits, mantissa_bits)
    expected_output = fake_cast_to_ieee_float(x,
                                              max_representable_value,
                                              exponent_bits,
                                              mantissa_bits)
    assert torch.equal(float_qdq(x), expected_output)


@torch.no_grad()
def test_qdq_compute_encodings(x):
    """
    Given: Instantiated FloatQuantizeDequantize with a min-max encoding analyzer
    When: compute_encodings() and run forwad
    Then: Output should be equal to fake-casting the input
          with maximum representable value = observed maximum input
    """
    encoding_analyzer = MinMaxEncodingAnalyzer((1, 100))
    float16_qdq = FloatQuantizeDequantize(dtype=torch.float16,
                                          encoding_analyzer=encoding_analyzer)
    with float16_qdq.compute_encodings():
        _ = float16_qdq(x)

    maxval = x.abs().max(dim=0, keepdims=True).values
    expected_output = fake_cast_to_ieee_float(x, maxval, exponent_bits=5, mantissa_bits=10)
    assert torch.equal(float16_qdq(x), expected_output)


def test_allow_overwrite(x):
    exponent_bits, mantissa_bits = 3, 4
    q = FloatQuantizeDequantize(exponent_bits, mantissa_bits, encoding_analyzer=MinMaxEncodingAnalyzer((1, 100)))
    with q.compute_encodings():
        q(x)

    """
    Given: _allow_overwrite set to False
    When: Try to recompute encodings
    Then: Encoding does NOT get overwritten by compute_encodings
    """
    q_max = q.maxval.detach().clone()
    q._allow_overwrite = False
    with q.compute_encodings():
        q(x * 10)

    assert torch.equal(q_max, q.maxval)
