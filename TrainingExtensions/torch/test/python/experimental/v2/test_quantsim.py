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

import torch
import torch.nn as nn
from aimet_torch.experimental.v2.quantization.quantsim import QuantizationSimModel
from aimet_torch.experimental.v2.quantization.encoding_analyzer import PercentileEncodingAnalyzer
from .models_ import test_models


class TestPercentileScheme:
    """ Test Percentile quantization scheme """ 

    def test_set_percentile_value(self):
        """ Test pecentile scheme by setting different percentile values """

        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input, quant_scheme="percentile")
        weight_quantizer = sim.model.conv.param_quantizers["weight"]
        assert isinstance(weight_quantizer.encoding_analyzer, PercentileEncodingAnalyzer)

        sim.set_percentile_value(99.9)
        assert weight_quantizer.encoding_analyzer.percentile == 99.9

        sim.compute_encodings(forward_pass, None)
        weight_max_99p9 = weight_quantizer.get_max()

        sim.set_percentile_value(90.0)
        assert weight_quantizer.encoding_analyzer.percentile == 90.0
        sim.compute_encodings(forward_pass, None)
        weight_max_90p0 = weight_quantizer.get_max()

        assert torch.all(weight_max_99p9.gt(weight_max_90p0))
