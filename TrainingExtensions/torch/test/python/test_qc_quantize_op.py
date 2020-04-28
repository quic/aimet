# /usr/bin/env python2.7
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

import torch
from torch import nn

from aimet_torch.qc_quantize_op import QcPostTrainingWrapper, QcQuantizeOpMode


class TestQcQuantizeOp(unittest.TestCase):

    def test_update_stats_with_pymo(self):

        device = torch.device('cpu')
        conv1 = torch.nn.Conv2d(4, 4, 1)
        quantize = QcPostTrainingWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                         quant_scheme='tf_enhanced')

        input_var = torch.autograd.Variable(torch.randn(4, 4, 2, 2), requires_grad=False).to(device)
        print(input_var)

        quantize.set_mode(QcQuantizeOpMode.ANALYSIS)

        output = quantize.forward(input_var)
        quantize.compute_encoding()
        actual_encoding = quantize.output_quantizer.encoding
        print("Encoding returned: min={}, max={}, offset={}. delta={}, bw={}"
              .format(actual_encoding.min, actual_encoding.max,
                      actual_encoding.offset, actual_encoding.delta, actual_encoding.bw))

    def test_quantize_dequantize_with_pymo(self):

        device = torch.device('cpu')
        conv1 = torch.nn.Conv2d(4, 4, 1)
        quantize = QcPostTrainingWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                         quant_scheme='tf_enhanced')

        input_var = torch.autograd.Variable(torch.randn(4, 4, 2, 2), requires_grad=True).to(device)

        quantize.set_mode(QcQuantizeOpMode.ANALYSIS)
        output = quantize.forward(input_var)
        quantize.compute_encoding()
        actual_encoding = quantize.output_quantizer.encoding

        print("Encoding returned: min={}, max={}, offset={}. delta={}, bw={}"
              .format(quantize.output_quantizer.encoding.min,
                      quantize.output_quantizer.encoding.max,
                      quantize.output_quantizer.encoding.offset,
                      quantize.output_quantizer.encoding.delta,
                      quantize.output_quantizer.encoding.bw))

        quantize.set_mode(QcQuantizeOpMode.ACTIVE)
        output = quantize.forward(input_var)
