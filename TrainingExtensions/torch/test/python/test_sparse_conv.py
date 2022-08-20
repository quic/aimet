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

import unittest
import json
import torch
from torch import nn
import spconv.pytorch as spconv

from aimet_torch.qc_quantize_op import StaticGridQuantWrapper, StaticGridPerTensorQuantizer,\
    StaticGridPerChannelQuantizer
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.libpymo import TfEncoding


class SpconvModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=10):
        super().__init__()
        self.conv = spconv.SparseConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        # torch tensor is stored in nchw while spconv requires it to be nhwc
        nhwc_x = x.permute(0, *[i for i in range(2, len(x.shape))], 1)
        spconv_input = spconv.SparseConvTensor.from_dense(nhwc_x)
        output = self.conv(spconv_input)
        return output.dense()


class TestSparseConv(unittest.TestCase):
    def test_sparse_conv_quantsim(self):
        dummy_input = torch.rand(2, 1, 5, 5)

        spconv_model = SpconvModel()
        sim = QuantizationSimModel(spconv_model, dummy_input)

        def dummy_forward(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        sim.compute_encodings(dummy_forward, None)

        # Check if Quantizers were created
        self.assertTrue(isinstance(sim.model.conv, StaticGridQuantWrapper))
        self.assertTrue(isinstance(sim.model.conv.param_quantizers['weight'], StaticGridPerTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.param_quantizers['bias'], StaticGridPerTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.output_quantizer, StaticGridPerTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.input_quantizer, StaticGridPerTensorQuantizer))

        # Check if encodings were created
        self.assertTrue(isinstance(sim.model.conv.param_quantizers['weight'].encoding, TfEncoding))
        self.assertTrue(isinstance(sim.model.conv.output_quantizer.encoding, TfEncoding))

