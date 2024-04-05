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

from aimet_torch.v2.quantization.affine import AffineEncoding


@pytest.fixture
def scale():
    return torch.tensor(1.0)

class TestAffineEncoding:

    def test_create_asymmetric_encoding_from_scale_offset_bitwidth(self, scale):
        """
        When: Create an affine encoding from a single scale and nonzero offset value
        Then: 1) all encoding parameters are torch.Tensor objects
              2) min = scale * offset
              3) max = scale * (offset + num_bins)
              4) granularity is "pertensor"
              5) symmetry is "asymmetric"
              6) mapping is "affine"
              7) dtype is torch.uint8
              8) num_steps is 2 ** bitwidth - 1
        """
        bitwidth = 8
        offset = torch.tensor(-5)
        encoding = AffineEncoding(scale, offset, bitwidth=bitwidth)
        for property in [encoding.min, encoding.max, encoding.scale, encoding.offset]:
            assert isinstance(property, torch.Tensor)
            assert property.shape == ()
        assert encoding.min == -5.0
        assert encoding.max == 2 ** bitwidth - 1 + offset
        assert encoding.offset == -5
        assert encoding.bitwidth == bitwidth
        assert encoding.granularity == "pertensor"
        assert encoding.mapping == "affine"
        assert encoding.symmetry == False
        assert encoding.dtype == torch.uint8
        assert encoding.num_steps == 2 ** bitwidth - 1

    def test_create_signed_symmetric_encoding(self, scale):
        """
        When: Create an 8-bit encoding with offset=0 and signed=True
        Then: 1) encoding.min is scale * num_negative_bins
              2) encoding.max is scale * num_positive_bins
              3) symmetry is "signed_symmetric"
              4) dtype is torch.int8
              5) num_steps is 2 ** bitwidth - 1
        """
        bitwidth = 8
        offset = torch.tensor(0)
        encoding = AffineEncoding(scale, offset, bitwidth=bitwidth, signed=True, symmetry=True)
        assert encoding.min == -128
        assert encoding.max == 127
        assert encoding.offset == 0
        assert encoding.scale == 1.0
        assert encoding.symmetry == True
        assert encoding.dtype == torch.int8
        assert encoding.num_steps == 2 ** bitwidth - 1

    def test_create_unsigned_symmetric_encoding(self):
        """
        When: Create an 8-bit encoding with offset=0 and signed=True
        Then: 1) encoding.min is scale * num_negative_bins
              2) encoding.max is scale * num_positive_bins
              3) symmetry is "signed_symmetric"
              4) dtype is torch.int8
              5) num_steps is 2 ** bitwidth - 1
        """
        bitwidth = 8
        offset = torch.tensor(0)
        scale = torch.tensor(0.5)
        encoding = AffineEncoding(scale, offset, bitwidth=bitwidth, signed=False, symmetry=True)
        assert encoding.min == 0
        assert encoding.max == 255.0/2
        assert encoding.offset == 0
        assert encoding.scale == 0.5
        assert encoding.symmetry == True
        assert encoding.dtype == torch.uint8
        assert encoding.num_steps == 2 ** bitwidth - 1

    @pytest.mark.cuda()
    @pytest.mark.parametrize("device, new_device", (("cuda:0", "cpu"),
                                                    ("cpu", "cuda:0")))
    def test_create_encoding_correct_device(self, scale, device, new_device):
        """
        When: Create an encoding with tensors on device
        Then: encoding.{min, max, scale, offset} are on device
        """
        scale = torch.ones((1,)).to(device)
        offset = torch.ones((1,)).to(device)
        encoding = AffineEncoding(scale, offset, bitwidth=8)
        for property in [encoding.min, encoding.max, encoding.scale, encoding.offset]:
            assert property.device == torch.device(device)

        """
        When: call encoding.to(new_device)
        Then: 1) original encoding.{min, max, scale, offset} are on device
              2) returned encoding.{min, max, scale, offset} are on new_device
        """
        new_encoding = encoding.to(new_device)

        for property in [encoding.min, encoding.max, encoding.scale, encoding.offset]:
            assert property.device == torch.device(device)

        for property in [new_encoding.min, new_encoding.max, new_encoding.scale, new_encoding.offset]:
            assert property.device == torch.device(new_device)

    @pytest.mark.parametrize("dtype, new_dtype", ((torch.float16, torch.float32),
                                                  (torch.float32, torch.float16)))
    def test_create_encoding_correct_dtype(self, dtype, new_dtype):
        """
        When: Create an encoding with tensors of type dtype in {torch.float16, torch.float32}
        Then: encoding.{min, max, scale, offset} are dtype
        """
        scale = torch.ones((1,)).to(dtype)
        offset = torch.ones((1,)).to(dtype)
        encoding = AffineEncoding(scale, offset, bitwidth=8)
        for property in [encoding.min, encoding.max, encoding.scale, encoding.offset]:
            assert property.dtype == dtype

        """
        When: call encoding.to(new_dtype)
        Then: 1) original encoding.{min, max, scale, offset} are dtype
              2) returned encoding.{min, max, scale, offset} are new_dtype
        """
        new_encoding = encoding.to(new_dtype)

        for property in [new_encoding.min, new_encoding.max, new_encoding.scale, new_encoding.offset]:
            assert property.dtype == new_dtype

        for property in [encoding.min, encoding.max, encoding.scale, encoding.offset]:
            assert property.dtype == dtype

    @pytest.mark.parametrize("dtype", (torch.uint8, torch.int32))
    def test_change_encoding_to_invalid_dtype(self, dtype):
        """
        When: call encoding.to() with invalid dtype in {torch.uint8, torch.int32}
        Then: raises RuntimeError
        """
        scale = torch.ones((1,), dtype=torch.float32)
        offset = torch.ones((1,), dtype=torch.float32)
        encoding = AffineEncoding(scale, offset, bitwidth=8)
        with pytest.raises(RuntimeError):
            encoding.to(dtype)

    @pytest.mark.parametrize("shape", ((10, 1), (10,)))
    def test_perchannel_encoding(self, shape):
        """
        When: Create an encoding with scale whose shape has more than one element
        Then: encoding.{scale, offset, min, max} have shape == shape
              and granularity == "perchannel"
        """
        scale = torch.randn(shape)
        offset = torch.randn(shape)
        encoding = AffineEncoding(scale, offset, 8)
        for property in [encoding.min, encoding.max, encoding.scale, encoding.offset]:
            assert property.shape == shape
        assert encoding.granularity == "perchannel"
        assert encoding.mapping == "affine"

    @pytest.mark.parametrize("shape", (tuple(), (1,)))
    def test_pertensor_encoding(self, shape):
        """
        When: Create an encoding with scale whose shape in {[], [1]}
        Then: encoding.{scale, offset, min, max} have shape == shape
              and granularity == "pertensor"
        """
        scale = torch.randn(shape)
        offset = torch.randn(shape)
        encoding = AffineEncoding(scale, offset, 8)
        for property in [encoding.min, encoding.max, encoding.scale, encoding.offset]:
            assert property.shape == shape
        assert encoding.granularity == "pertensor"
        assert encoding.mapping == "affine"
