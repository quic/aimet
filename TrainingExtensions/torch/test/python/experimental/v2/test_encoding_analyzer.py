# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 Qualcomm Innovation Center, Inc. All rights reserved.
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
import torch
import pytest
from aimet_torch.experimental.v2.quantization.encoding_analyzer import get_encoding_analyzer_cls, CalibrationMethod
from aimet_torch.experimental.v2.quantization.backends.default import quantize, quantize_dequantize


class TestEncodingAnalyzer():
   @pytest.mark.parametrize('symmetric', [True, False])
   def test_overflow(self, symmetric):
       encoding_shape = (1,)
       float_input = (torch.arange(10) * torch.finfo(torch.float).tiny)

       encoding_analyzer = get_encoding_analyzer_cls(CalibrationMethod.MinMax, encoding_shape)
       encoding_analyzer.update_stats(float_input)
       min, max = encoding_analyzer.compute_encodings(bitwidth=8, is_symmetric=symmetric)

       scale = (max - min) / 255
       # Scale should be at least as large as torch.tiny
       assert torch.all(torch.isfinite(scale))
       assert torch.allclose(scale, torch.tensor(torch.finfo(scale.dtype).tiny))

   @pytest.mark.parametrize('dtype', [torch.float, torch.half])
   @pytest.mark.parametrize('symmetric', [True, False])
   def test_continuity(self, symmetric, dtype):
       encoding_shape = (1,)
       normal_range = torch.arange(-128, 128).to(dtype) / 256
       encoding_analyzer = get_encoding_analyzer_cls(CalibrationMethod.MinMax, encoding_shape)
       eps = torch.finfo(dtype).eps

       min_1, max_1 = encoding_analyzer.compute_dynamic_encodings(normal_range * (1 - eps),
                                                                  bitwidth=8, is_symmetric=symmetric)
       min_2, max_2 = encoding_analyzer.compute_dynamic_encodings(normal_range,
                                                                  bitwidth=8, is_symmetric=symmetric)
       min_3, max_3 = encoding_analyzer.compute_dynamic_encodings(normal_range * (1 + eps),
                                                                  bitwidth=8, is_symmetric=symmetric)

       assert min_3 <= min_2 <= min_1 <= max_1 <= max_2 <= max_3
       assert torch.allclose(max_1, max_2, atol=eps)
       assert torch.allclose(min_1, min_2, atol=eps)
       assert torch.allclose(max_2, max_3, atol=eps)
       assert torch.allclose(min_2, min_3, atol=eps)


class TestMinMaxEncodingAnalyzer():
   def test_compute_encodings_with_negative_bitwidth(self):
        encoding_min_max = torch.randn(3, 4)
        encoding_analyzer = get_encoding_analyzer_cls(CalibrationMethod.MinMax, encoding_min_max.shape)
        encoding_analyzer.update_stats(torch.randn(3, 4))
        with pytest.raises(ValueError):
                encoding_analyzer.compute_encodings(bitwidth = 0, is_symmetric = False)
        
   def test_compute_encodings_asymmetric(self):
        encoding_min_max = torch.randn(1)
        encoding_analyzer = get_encoding_analyzer_cls(CalibrationMethod.MinMax, encoding_min_max.shape)
        input_tensor =  torch.arange(start=0, end=26, step=0.5, dtype=torch.float)
        encoding_analyzer.update_stats(input_tensor)

        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False)
        assert torch.all(torch.isclose(asymmetric_min, torch.zeros(tuple(encoding_analyzer.observer.shape))))
        assert torch.all(torch.isclose(asymmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), 25.5)))
        
   def test_compute_encodings_signed_symmetric(self):
        encoding_min_max = torch.randn(1)
        encoding_analyzer = get_encoding_analyzer_cls(CalibrationMethod.MinMax, encoding_min_max.shape)
        input_tensor =  torch.arange(start=0, end=26, step=0.5, dtype=torch.float)
        encoding_analyzer.update_stats(input_tensor)

        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True)
        assert torch.all(torch.isclose(symmetric_min, torch.full(tuple(encoding_analyzer.observer.shape), -25.5)))
        assert torch.all(torch.isclose(symmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), 25.5)))

   def test_reset_stats(self):
        encoding_min_max = torch.randn(3, 4)
        encoding_analyzer = get_encoding_analyzer_cls(CalibrationMethod.MinMax, encoding_min_max.shape)
        encoding_analyzer.update_stats(torch.randn(3, 4))
        assert torch.all(encoding_analyzer.observer.stats.min)
        assert torch.all(encoding_analyzer.observer.stats.max)
        encoding_analyzer.reset_stats()
        assert not encoding_analyzer.observer.stats.min
        assert not encoding_analyzer.observer.stats.max
        
   def test_compute_encodings_with_no_stats(self):
        encoding_min_max = torch.randn(3, 4)
        encoding_analyzer = get_encoding_analyzer_cls(CalibrationMethod.MinMax, encoding_min_max.shape)
        with pytest.raises(RuntimeError):
                encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False)

   def test_compute_encodings_with_only_zero_tensor(self):
        encoding_min_max = torch.randn(3, 4)
        encoding_analyzer = get_encoding_analyzer_cls(CalibrationMethod.MinMax, encoding_min_max.shape)
        encoding_analyzer.update_stats(torch.zeros(tuple(encoding_analyzer.observer.shape)))
        
        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False)
        updated_min = torch.finfo(asymmetric_min.dtype).tiny * (2 ** (8 - 1))
        updated_max = torch.finfo(asymmetric_min.dtype).tiny * ((2 **(8 - 1)) - 1)
        assert torch.all(torch.eq(asymmetric_min,  torch.full(tuple(encoding_analyzer.observer.shape), -updated_min)))
        assert torch.all(torch.eq(asymmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), updated_max)))
        
        symmetric_min , symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True)
        updated_symmetric_min = min(-updated_min, -updated_max)
        updated_symmetric_max = max(updated_min, updated_max)
        assert torch.all(torch.eq(symmetric_min, torch.full(tuple(encoding_analyzer.observer.shape), updated_symmetric_min)))
        assert torch.all(torch.eq(symmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), updated_symmetric_max)))
        
   def test_compute_encodings_with_same_nonzero_tensor(self):
        encoding_min_max = torch.randn(3, 4)
        encoding_analyzer = get_encoding_analyzer_cls(CalibrationMethod.MinMax, encoding_min_max.shape)
        encoding_analyzer.update_stats(torch.full((3, 4), 3.0))

        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False)
        assert torch.allclose(asymmetric_min, torch.full(tuple(encoding_analyzer.observer.shape), 0.0), atol = torch.finfo().tiny)
        assert torch.allclose(asymmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), 3.0), atol = torch.finfo().tiny)

        symmetric_min , symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True)
        assert torch.allclose(symmetric_min, torch.full(tuple(encoding_analyzer.observer.shape), -3.0), atol = torch.finfo().tiny)
        assert torch.allclose(symmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), 3.0), atol = torch.finfo().tiny)
        
   @pytest.mark.parametrize("min_max_size", [[3,4], [2, 3, 1], [4], [1]])
   def test_update_stats_with_different_dimensions(self,min_max_size):
        for i in range(4):
              encoding_analyzer =  get_encoding_analyzer_cls(CalibrationMethod.MinMax, min_max_size)
              encoding_analyzer.update_stats(torch.randn(2, 3, 4))
              assert list(encoding_analyzer.observer.stats.min.shape) == min_max_size
              assert list(encoding_analyzer.observer.stats.max.shape) == min_max_size

   def test_update_stats_incompatible_dimension(self):
        encoding_analyzer_1 = get_encoding_analyzer_cls(CalibrationMethod.MinMax, [3, 4])
        with pytest.raises(RuntimeError):
                encoding_analyzer_1.update_stats(torch.randn(2, 3, 5))
