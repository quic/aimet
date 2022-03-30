# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

from aimet_common.defs import QuantScheme
from aimet_torch.tensor_quantizer import StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer,\
    StaticGridTensorQuantizer

BUCKET_SIZE = 512


class TestTensorQuantizer:

    def test_get_stats_histogram_per_tensor(self):
        """
        test get_stats_histogram() for per tensor quantizer.
        """
        tensor = torch.randn(5, 5)
        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                 use_symmetric_encodings=False, enabled_by_default=True)
        quantizer.update_encoding_stats(tensor)
        quantizer.compute_encoding()
        assert quantizer.encoding, "Encoding shouldn't be None."
        histograms = quantizer.get_stats_histogram()
        assert len(histograms) == 1
        for histogram in histograms:
            print(histogram)
            assert len(histogram) == BUCKET_SIZE

    def test_get_stats_histogram_per_channel(self):
        """
        test get_stats_histogram() for per channel quantizer.
        """
        tensor = torch.randn(10, 5)
        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                  num_channels=5, use_symmetric_encodings=False,
                                                  enabled_by_default=True)
        quantizer.update_encoding_stats(tensor)
        quantizer.compute_encoding()
        assert quantizer.encoding, "Encoding shouldn't be None."
        histograms = quantizer.get_stats_histogram()
        assert len(histograms) == 5
        for histogram in histograms:
            assert len(histogram) == BUCKET_SIZE

    def test_get_stats_histogram_with_invalid_combination(self):
        """
        test get_stats_histogram() with invalid inputs.
        """
        # Encoding should be computed.
        quantizer = StaticGridTensorQuantizer(bitwidth=8, round_mode='nearest',
                                              quant_scheme=QuantScheme.post_training_tf_enhanced,
                                              use_symmetric_encodings=False, enabled_by_default=True)
        with pytest.raises(RuntimeError):
            quantizer.get_stats_histogram()

        # quant scheme should be TF-Enhanced.
        quantizer = StaticGridTensorQuantizer(bitwidth=8, round_mode='nearest',
                                              quant_scheme=QuantScheme.post_training_tf,
                                              use_symmetric_encodings=False, enabled_by_default=True)
        with pytest.raises(RuntimeError):
            quantizer.get_stats_histogram()