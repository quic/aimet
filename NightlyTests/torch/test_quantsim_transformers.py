#!/usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" contains unit tests to validate transformer quantization support """

import unittest
import torch

from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.transformers.utils import get_quantizable_pt_transformer_model


class TestQuantizationSimTransformers(unittest.TestCase):
    def test_word_langauge_model(self):
        from transformer_models.word_language_model import TransformerModel
        n_layers = 2
        model = TransformerModel(33278, 200, 2, 200, n_layers)

        model.eval()
        get_quantizable_pt_transformer_model(model)

        # create quantsim object on updated model
        dummy_input = torch.randint(33278, size=(35, 20))
        sim = QuantizationSimModel(model, dummy_input)

        def forward_pass(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)
        sim.compute_encodings(forward_pass, None)

        for i in range(n_layers):
            # validate MHA layers have quantizers
            self.assertTrue(sim.model.transformer_encoder.layers[i].self_attn.linear_Q.output_quantizers[0].encoding)
            self.assertTrue(sim.model.transformer_encoder.layers[i].self_attn.linear_K.output_quantizers[0].encoding)
            self.assertTrue(sim.model.transformer_encoder.layers[i].self_attn.linear_V.output_quantizers[0].encoding)
            self.assertTrue(sim.model.transformer_encoder.layers[i].self_attn.matmul_1.output_quantizers[0].encoding)
            self.assertTrue(sim.model.transformer_encoder.layers[i].self_attn.matmul_2.output_quantizers[0].encoding)
            self.assertTrue(sim.model.transformer_encoder.layers[i].self_attn.softmax.output_quantizers[0].encoding)

            # validate mask encoding
            mask_add_quantizer = sim.model.transformer_encoder.layers[i].self_attn.mask_add.output_quantizers[0]
            self.assertAlmostEqual(mask_add_quantizer.encoding.min, -6, 1)

        del sim
