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
import copy

import pytest
import torch

from aimet_torch.qc_quantize_op import StaticGridQuantWrapper, QcQuantizeOpMode, \
    QuantScheme, MAP_QUANT_SCHEME_TO_PYMO, MAP_ROUND_MODE_TO_PYMO
from aimet_torch.tensor_quantizer import StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer
import libpymo


class TestQcQuantizeOp:

    def test_update_stats_with_pymo(self):

        device = torch.device('cpu')
        conv1 = torch.nn.Conv2d(4, 4, 1)
        quantize = StaticGridQuantWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
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
        quantize = StaticGridQuantWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
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
        quantize = StaticGridQuantWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
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
                assert new_input.grad[0][i] == 0.0

        # Check if output gradient got clipped
        output_grad = output_grad[0].flatten()
        assert output_grad[0] == 1.0
        assert output_grad[1] == 1.0
        assert output_grad[2] == 1.0
        assert output_grad[3] == 0.0

        # Check if weight gradient got clipped
        weight_tensor = quantize._module_to_wrap.weight.flatten()
        weight_tensor_grad = quantize._module_to_wrap.weight.grad.flatten()
        for i, val in enumerate(weight_tensor):
            if encodings_new.min > val or val > encodings_new.max:
                assert weight_tensor_grad[i] == 0.0

    def test_quantize_maxpool_with_indices(self):
        """ Test that maxpool2d returning int tensor can be quantized """
        maxpool = torch.nn.MaxPool2d(2, return_indices=True)
        quantize_op = StaticGridQuantWrapper(maxpool, weight_bw=8, activation_bw=8, round_mode='nearest',
                                             quant_scheme=QuantScheme.post_training_tf_enhanced)
        inp = torch.rand((1, 3, 8, 8))
        quantize_op.set_mode(QcQuantizeOpMode.ANALYSIS)
        quantize_op(inp)
        quantize_op.compute_encoding()
        quantize_op.set_mode(QcQuantizeOpMode.ACTIVE)
        out, indices = quantize_op(inp)

        # Check that one of the outputs of quantize_op is the indices with dtype int64
        assert indices.dtype == torch.int64
        assert quantize_op.output_quantizers[0] is not None

    def test_quantize_only_asymmetric_cpu(self):
        """ Test tensor quantizer quantize only asymmetric functionality """
        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                 use_symmetric_encodings=False, enabled_by_default=True)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 2.23
        encodings.min = -5.19
        encodings.offset = -178
        quantizer.encoding = encodings

        inp_tensor = torch.tensor([-7, -5, -3, 0, .1, 2.5])
        quant_out = quantizer.quantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([0, 6, 75, 178, 181, 255], dtype=torch.float32)
        assert torch.equal(quant_out, expected_out)

    def test_per_channel_symmetric_qdq(self):
        """ Test tensor quantizer symmetric quantize-dequantize functionality on cpu """

        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                  use_symmetric_encodings=True, enabled_by_default=True,
                                                  num_channels=4)
        encodings = [libpymo.TfEncoding() for _ in range(4)]
        for index in range(3):
            encodings[index].bw = 8
            encodings[index].max = 3.81
            encodings[index].min = -3.84
            encodings[index].delta = 0.03
            encodings[index].offset = -128

        encodings[3].bw = 8
        encodings[3].max = 6.35
        encodings[3].min = -6.4
        encodings[3].delta = 0.05
        encodings[3].offset = -128

        # delta is 0.040745098
        quantizer.encoding = encodings

        # Test quantize only on gpu
        inp_tensor = torch.tensor([[-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5]])

        quant_out = quantizer.quantize_dequantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([[-3.84, -3.84, -3, 0, .09, 2.49],
                                     [-3.84, -3.84, -3, 0, .09, 2.49],
                                     [-3.84, -3.84, -3, 0, .09, 2.49],
                                     [-6.4, -5, -3, 0, .1, 2.5]],
                                    dtype=torch.float32)
        assert torch.equal(quant_out, expected_out)

    def test_per_channel_asymmetric_qdq(self):
        """ Test tensor quantizer asymmetric quantize-dequantize functionality on cpu """

        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                  use_symmetric_encodings=False, enabled_by_default=True,
                                                  num_channels=4)
        encodings = [libpymo.TfEncoding() for _ in range(4)]
        for index in range(3):
            encodings[index].bw = 8
            encodings[index].max = 1.9999956
            encodings[index].min = -2.9999934
            encodings[index].delta = 0.0196078
            encodings[index].offset = -153

        encodings[3].bw = 8
        encodings[3].max = 2.404693
        encodings[3].min = -5.995262
        encodings[3].delta = 0.032941
        encodings[3].offset = -182

        # delta is 0.040745098
        quantizer.encoding = encodings

        # Test quantize only on gpu
        inp_tensor = torch.tensor([[-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5],
                                   [-7, -5, -3, 0, .1, 2.5]])

        quant_out = quantizer.quantize_dequantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([[-3.0, -3.0, -3.0, 0, .098, 2.0],
                                     [-3.0, -3.0, -3.0, 0, .098, 2.0],
                                     [-3.0, -3.0, -3.0, 0, .098, 2.0],
                                     [-5.9953, -5.0070, -2.9976, 0, .09888, 2.4047]],
                                    dtype=torch.float32)
        assert torch.allclose(quant_out, expected_out, atol=0.0001)

    def test_per_channel_symmetric_compute_encodings(self):
        """ Test tensor quantizer symmetric compute-encodings functionality on cpu """

        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                  use_symmetric_encodings=True, enabled_by_default=True,
                                                  num_channels=4)

        inp_tensor = torch.tensor([[-7, -5, -3, 0, .1, 2.5],
                                   [-5, -5, -3, 0, .1, 2.7],
                                   [-6, -5, -3, 0, .1, 2.8],
                                   [-5, -5, -3, 0, .1, 2]])
        quantizer.update_encoding_stats(inp_tensor)
        quantizer.compute_encoding()

        assert len(quantizer.encoding) == 4
        assert quantizer.encoding[0].max == 7
        assert round(quantizer.encoding[0].min, 2) == -7.06

        assert quantizer.encoding[3].max == 5
        assert round(quantizer.encoding[3].min, 2) == -5.04

    def test_per_channel_asymmetric_compute_encodings(self):
        """ Test tensor quantizer asymmetric compute-encodings functionality on cpu """

        quantizer = StaticGridPerChannelQuantizer(bitwidth=8, round_mode='nearest',
                                                  quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                  use_symmetric_encodings=False, enabled_by_default=True,
                                                  num_channels=4)

        inp_tensor = torch.tensor([[-7, -5, -3, 0, .1, 2.5],
                                   [-5, -5, -3, 0, .1, 2.7],
                                   [-6, -5, -3, 0, .1, 2.8],
                                   [-5, -5, -3, 0, .1, 2]])
        quantizer.update_encoding_stats(inp_tensor)
        quantizer.compute_encoding()

        assert len(quantizer.encoding) == 4
        assert round(quantizer.encoding[0].max, 3) == 2.496
        assert round(quantizer.encoding[0].min, 3) == -7.004

        assert round(quantizer.encoding[3].max, 3) == 2.004
        assert round(quantizer.encoding[3].min, 3) == -4.996

    def test_quantize_only_symmetric_signed_cpu(self):
        """ Test tensor quantizer quantize only symmetric signed functionality on cpu """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                 use_symmetric_encodings=True, enabled_by_default=True)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.19
        encodings.min = -5.20
        encodings.offset = -128

        # delta is 0.040745098
        quantizer.encoding = encodings

        # Test quantize only on gpu
        inp_tensor_gpu = torch.tensor([-7, -5, -3, 0, .1, 2.5])
        quant_out = quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([-128, -123, -74, 0, 2, 61], dtype=torch.float32)
        assert torch.equal(quant_out, expected_out)

    def test_quantize_only_symmetric_unsigned_cpu(self):
        """ Test tensor quantizer quantize only symmetric unsigned functionality on cpu """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                 use_symmetric_encodings=True, enabled_by_default=True)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.19
        encodings.min = 0.0
        encodings.offset = 0

        # delta is 0.020352941
        quantizer.encoding = encodings

        # Test quantize only on gpu
        inp_tensor_gpu = torch.tensor([0, 1.2, 1.5, 4.0, 4.9, 5.3])
        quant_out = quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([0, 59, 74, 197, 241, 255], dtype=torch.float32)
        assert torch.equal(quant_out, expected_out)

    @pytest.mark.cuda
    def test_quantize_only_asymmetric_gpu(self):
        """ Test tensor quantizer quantize only asymmetric functionality on gpu """
    
        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                 use_symmetric_encodings=False, enabled_by_default=True)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 2.23
        encodings.min = -5.19
        encodings.offset = -178
        quantizer.encoding = encodings
    
        # Test quantize only on gpu
        inp_tensor_gpu = torch.tensor([-7, -5, -3, 0, .1, 2.5], device=torch.device('cuda'))
        quant_out = quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([0, 6, 75, 178, 181, 255], dtype=torch.float32, device=torch.device('cuda'))
        assert torch.equal(quant_out, expected_out)

    @pytest.mark.cuda
    def test_quantize_only_symmetric_signed_gpu(self):
        """ Test tensor quantizer quantize only symmetric signed functionality on gpu """

        post_training_tensor_quantizer = \
            StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                         quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                         use_symmetric_encodings=True, enabled_by_default=True)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.19
        encodings.min = -5.20
        encodings.offset = -128

        # delta is 0.040745098
        post_training_tensor_quantizer.encoding = encodings

        # Test quantize only on gpu
        inp_tensor_gpu = torch.tensor([-7, -5, -3, 0, .1, 2.5], device=torch.device('cuda'))
        quant_out = post_training_tensor_quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([-128, -123, -74, 0, 2, 61], dtype=torch.float32, device=torch.device('cuda'))
        assert torch.equal(quant_out, expected_out)

    @pytest.mark.cuda
    def test_quantize_only_symmetric_unsigned_gpu(self):
        """ Test tensor quantizer quantize only symmetric unsigned functionality on gpu """

        post_training_tensor_quantizer = \
            StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                         quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                         use_symmetric_encodings=True, enabled_by_default=True)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.19
        encodings.min = 0.0
        encodings.offset = 0

        # delta is 0.020352941
        post_training_tensor_quantizer.encoding = encodings

        # Test quantize only on gpu
        inp_tensor_gpu = torch.tensor([0, 1.2, 1.5, 4.0, 4.9, 5.3], device=torch.device('cuda'))
        quant_out = post_training_tensor_quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([0, 59, 74, 197, 241, 255], dtype=torch.float32, device=torch.device('cuda'))
        assert torch.equal(quant_out, expected_out)

    def test_qc_post_training_wrapper_mem_leak(self):
        torch.manual_seed(0)

        rand_tensor = torch.rand(1, 10, 20, 20)
        quant = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
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
        assert 100000 >= delta

    def test_compute_encoding_for_tensor_quantizer_with_no_stats(self):
        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=MAP_QUANT_SCHEME_TO_PYMO[
                                                     QuantScheme.post_training_tf_enhanced],
                                                 use_symmetric_encodings=False, enabled_by_default=True)
        quantizer.compute_encoding()
        assert quantizer._encoding == []
