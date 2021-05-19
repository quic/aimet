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

import pytest
import unittest.mock
import torch

from aimet_torch.qc_quantize_op import QcPostTrainingWrapper, QcQuantizeOpMode, QuantScheme, MAP_QUANT_SCHEME_TO_PYMO, \
    MAP_ROUND_MODE_TO_PYMO
from aimet_torch.tensor_quantizer import PostTrainingTensorQuantizer
import libpymo


class TestQcQuantizeOp(unittest.TestCase):

    def test_update_stats_with_pymo(self):

        device = torch.device('cpu')
        conv1 = torch.nn.Conv2d(4, 4, 1)
        quantize = QcPostTrainingWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                         quant_scheme=QuantScheme.post_training_tf_enhanced)

        input_var = torch.autograd.Variable(torch.randn(4, 4, 2, 2), requires_grad=False).to(device)
        print(input_var)

        quantize.set_mode(QcQuantizeOpMode.ANALYSIS)

        output = quantize.forward(input_var)
        quantize.compute_encoding()
        actual_encoding = quantize.output_quantizers[0].encoding
        print("Encoding returned: min={}, max={}, offset={}. delta={}, bw={}"
              .format(actual_encoding.min, actual_encoding.max,
                      actual_encoding.offset, actual_encoding.delta, actual_encoding.bw))

    def test_quantize_dequantize_with_pymo(self):

        device = torch.device('cpu')
        conv1 = torch.nn.Conv2d(4, 4, 1)
        quantize = QcPostTrainingWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                         quant_scheme=QuantScheme.post_training_tf_enhanced)

        input_var = torch.autograd.Variable(torch.randn(4, 4, 2, 2), requires_grad=True).to(device)

        quantize.set_mode(QcQuantizeOpMode.ANALYSIS)
        output = quantize.forward(input_var)
        quantize.compute_encoding()
        actual_encoding = quantize.output_quantizers[0].encoding

        print("Encoding returned: min={}, max={}, offset={}. delta={}, bw={}"
              .format(quantize.output_quantizers[0].encoding.min,
                      quantize.output_quantizers[0].encoding.max,
                      quantize.output_quantizers[0].encoding.offset,
                      quantize.output_quantizers[0].encoding.delta,
                      quantize.output_quantizers[0].encoding.bw))

        quantize.set_mode(QcQuantizeOpMode.ACTIVE)
        output = quantize.forward(input_var)

    def test_qc_post_training_wrapper(self):
        torch.manual_seed(0)

        encodings = libpymo.TfEncoding()
        encodings.bw, encodings.max, encodings.min, encodings.delta, encodings.offset = 8, 0.5, -1, 1, 0.2

        encodings_new = libpymo.TfEncoding()
        encodings_new.bw, encodings_new.max, encodings_new.min, encodings_new.delta, encodings_new.offset = 8, 0.4, -0.98, 1, 0.2

        output_grad = []
        def hook_fn(m, _, i):

            for grad in i:
                try:
                    output_grad.append(grad)
                except AttributeError:
                    print ("None found for Gradient")

        conv1 = torch.nn.Conv2d(1, 2, 1)
        quantize = QcPostTrainingWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                         quant_scheme=QuantScheme.post_training_tf_enhanced)
        quantize.train()
        quantize._module_to_wrap.register_backward_hook(hook_fn)

        quantize.input_quantizer.enabled = True
        quantize.output_quantizers[0].enabled = True
        quantize.input_quantizer.encoding = encodings
        quantize.output_quantizers[0].encoding = encodings

        new_input = torch.autograd.Variable(torch.tensor([[[[0.6469]]], [[[-0.9]]]]), requires_grad=True)
        quantize.set_mode(QcQuantizeOpMode.ACTIVE)
        out = quantize(new_input)

        quantize.input_quantizer.encoding = encodings_new
        quantize.output_quantizers[0].encoding = encodings_new
        quantize.param_quantizers['weight'].encoding = encodings_new

        loss = out.flatten().sum()
        loss.backward()

        # Check if input gradient got clipped
        for i, val in enumerate(new_input):
            if encodings_new.min > val or val > encodings_new.max:
                self.assertTrue(new_input.grad[0][i] == 0.0)

        # Check if output gradient got clipped
        output_grad = output_grad[0].flatten()
        self.assertTrue(output_grad[0] == 1.0)
        self.assertTrue(output_grad[1] == 1.0)
        self.assertTrue(output_grad[2] == 1.0)
        self.assertTrue(output_grad[3] == 0.0)

        # Check if weight gradient got clipped
        weight_tensor = quantize._module_to_wrap.weight.flatten()
        weight_tensor_grad = quantize._module_to_wrap.weight.grad.flatten()
        for i, val in enumerate(weight_tensor):
            if encodings_new.min > val or val > encodings_new.max:
                self.assertTrue(weight_tensor_grad[i] == 0.0)

    def test_quantize_maxpool_with_indices(self):
        """ Test that maxpool2d returning int tensor can be quantized """
        maxpool = torch.nn.MaxPool2d(2, return_indices=True)
        quantize_op = QcPostTrainingWrapper(maxpool, weight_bw=8, activation_bw=8, round_mode='nearest',
                                            quant_scheme=QuantScheme.post_training_tf_enhanced)
        inp = torch.rand((1, 3, 8, 8))
        quantize_op.set_mode(QcQuantizeOpMode.ANALYSIS)
        quantize_op(inp)
        quantize_op.compute_encoding()
        quantize_op.set_mode(QcQuantizeOpMode.ACTIVE)
        out, indices = quantize_op(inp)

        # Check that one of the outputs of quantize_op is the indices with dtype int64
        self.assertEqual(indices.dtype, torch.int64)
        self.assertTrue(quantize_op.output_quantizers[0] is not None)

    def test_quantize_only_cpu(self):
        """ Test tensor quantizer quantize only functionality """

        post_training_tensor_quantizer = \
            PostTrainingTensorQuantizer(bitwidth=8, round_mode='nearest',
                                        quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                        use_symmetric_encodings=False, enabled_by_default=True)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 2.23
        encodings.min = -5.19
        post_training_tensor_quantizer.encoding = encodings

        inp_tensor = torch.tensor([-7, -5, -3, 0, .1, 2.5])
        quant_out = post_training_tensor_quantizer.quantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([0, 6, 75, 178, 181, 255], dtype=torch.float32)
        self.assertTrue(torch.equal(quant_out, expected_out))

    @pytest.mark.cuda
    def test_quantize_only_gpu(self):
        """ Test tensor quantizer quantize only functionality on gpu """
    
        post_training_tensor_quantizer = \
            PostTrainingTensorQuantizer(bitwidth=8, round_mode='nearest',
                                        quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                        use_symmetric_encodings=False, enabled_by_default=True)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 2.23
        encodings.min = -5.19
        post_training_tensor_quantizer.encoding = encodings
    
        # Test quantize only on gpu
        inp_tensor_gpu = torch.tensor([-7, -5, -3, 0, .1, 2.5], device=torch.device('cuda'))
        quant_out = post_training_tensor_quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([0, 6, 75, 178, 181, 255], dtype=torch.float32, device=torch.device('cuda'))
        self.assertTrue(torch.equal(quant_out, expected_out))

    def test_qc_post_training_wrapper_mem_leak(self):
        torch.manual_seed(0)

        rand_tensor = torch.rand(1, 10, 20, 20)
        quant = PostTrainingTensorQuantizer(bitwidth=8, round_mode='nearest',
                                            quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf_enhanced],
                                            use_symmetric_encodings=False, enabled_by_default=True)
        import psutil
        import os
        process = psutil.Process(os.getpid())
        baseline_mem = None

        for i in range(1000):
            quant.reset_encoding_stats()
            quant.update_encoding_stats(rand_tensor)
            quant.compute_encoding()
            if not baseline_mem:
                baseline_mem = process.memory_info().rss

        quant.reset_encoding_stats()
        delta = process.memory_info().rss - baseline_mem
        self.assertEqual(0, delta)

