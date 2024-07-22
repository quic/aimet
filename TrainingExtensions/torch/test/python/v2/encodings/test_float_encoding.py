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

from aimet_torch.v2.quantization.float import FloatEncoding

@pytest.fixture
def maxval():
    return torch.tensor(1.0)

class TestFloatEncoding:

    @pytest.mark.parametrize("mantissa_bits, exponent_bits", ((5, 10), (8, 23), (4, 8)))
    def test_create_encoding(self, mantissa_bits, exponent_bits, maxval):
        """
        When: Create an encoding from a single maxval, mantissa_bits, and exponent_bits
        Then: 1) all encoding parameters are torch.Tensor objects
              2) mantissa_bits is the same as passed value
              3) exponent_bits is the same as passed value
              4) bitwidth is mantissa_bits + exponent_bits + 1
              5) mapping is "float"
        """
        encoding = FloatEncoding(mantissa_bits, exponent_bits, maxval)
        assert isinstance(encoding.maxval, torch.Tensor)
        assert encoding.mantissa_bits == mantissa_bits
        assert encoding.exponent_bits == exponent_bits
        assert encoding.bitwidth == mantissa_bits + exponent_bits + 1
        assert encoding.mapping == "float"

    @pytest.mark.cuda()
    @pytest.mark.parametrize("device, new_device", (("cuda:0", "cpu"),
                                                    ("cpu", "cuda:0")))
    def test_create_encoding_correct_device(self, device, new_device):
        """
        When: Create an encoding with tensors on device
        Then: encoding.maxval is on device
        """
        mantissa_bits = 5
        exponent_bits = 10
        maxval = torch.tensor(1.0).to(device)
        encoding = FloatEncoding(mantissa_bits, exponent_bits, maxval)
        assert encoding.maxval.device == torch.device(device)

        """
        When: call encoding.to(new_device)
        Then: 1) original encoding.maxval is on device
              2) returned encoding.maxval is on new_device
        """
        new_encoding = encoding.to(new_device)
        assert encoding.maxval.device == torch.device(device)
        assert new_encoding.maxval.device == torch.device(new_device)

    @pytest.mark.parametrize("dtype, new_dtype", ((torch.float16, torch.float32),
                                                  (torch.float32, torch.float16)))
    def test_create_encoding_correct_dtype(self, dtype, new_dtype):
        """
        When: Create an encoding with tensors of type dtype in {torch.float16, torch.float32}
        Then: encoding.maxval is dtype
        """
        mantissa_bits = 5
        exponent_bits = 10
        maxval = torch.tensor(1.0).to(dtype)
        encoding = FloatEncoding(mantissa_bits, exponent_bits, maxval)
        assert encoding.maxval.dtype == dtype

        """
        When: call encoding.to(new_dtype)
        Then: 1) original encoding.maxval is dtype
              2) returned encoding.maxval is new_dtype
        """
        new_encoding = encoding.to(new_dtype)
        assert encoding.maxval.dtype == dtype
        assert new_encoding.maxval.dtype == new_dtype

    @pytest.mark.parametrize("shape", ((10, 1), (10,), (1,)))
    def test_perchannel_encoding(self, shape):
        """
        When: Create an encoding with maxval whose shape has more than one element
        Then: encoding.maxval have shape == shape
              and granularity == "perchannel"
        """
        mantissa_bits = 5
        exponent_bits = 10
        maxval = torch.randn(shape)
        encoding = FloatEncoding(mantissa_bits, exponent_bits, maxval)
        assert encoding.maxval.shape == shape
        assert encoding.granularity == "perchannel"
        assert encoding.mapping == "float"

    def test_pertensor_encoding(self):
        """
        When: Create an encoding with 0-D maxval
        Then: encoding.maxval have shape == shape
              and granularity == "pertensor"
        """
        mantissa_bits = 5
        exponent_bits = 10
        maxval = torch.randn([])
        encoding = FloatEncoding(mantissa_bits, exponent_bits, maxval)
        assert encoding.maxval.shape == tuple()
        assert encoding.granularity == "pertensor"
        assert encoding.mapping == "float"
