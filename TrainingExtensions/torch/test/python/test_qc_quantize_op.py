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
import unittest.mock
import torch

from aimet_common.defs import QuantScheme
from aimet_torch.qc_quantize_op import QcPostTrainingWrapper, QcQuantizeOpMode
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
        actual_encoding = quantize.output_quantizer.encoding
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
        actual_encoding = quantize.output_quantizer.encoding

        print("Encoding returned: min={}, max={}, offset={}. delta={}, bw={}"
              .format(quantize.output_quantizer.encoding.min,
                      quantize.output_quantizer.encoding.max,
                      quantize.output_quantizer.encoding.offset,
                      quantize.output_quantizer.encoding.delta,
                      quantize.output_quantizer.encoding.bw))

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
        quantize.output_quantizer.enabled = True
        quantize.input_quantizer.encoding = encodings
        quantize.output_quantizer.encoding = encodings

        new_input = torch.autograd.Variable(torch.tensor([[[[0.6469]]], [[[-0.9]]]]), requires_grad=True)
        quantize.set_mode(QcQuantizeOpMode.ACTIVE)
        out = quantize(new_input)

        quantize.input_quantizer.encoding = encodings_new
        quantize.output_quantizer.encoding = encodings_new
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

    def test(self):
        torch.manual_seed(0)
        tensor = torch.rand(2, 2)
        t = torch.nn.Parameter(tensor, requires_grad=True)
        quantizer = Dummy()
        out = quantizer(t)
        out.retain_grad()

        loss = out.flatten().sum()
        loss.backward()
        print()


class Dummy(torch.nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.module = torch.nn.ReLU()

    def forward(self, input):
        out = QuantizeDequantize.apply(input)
        out = self.module(out)
        return out


from aimet_torch.quantsim_straight_through_grad import compute_dloss_by_dx


class QuantizeDequantize(torch.autograd.Function):

    # pylint:disable = arguments-differ
    @staticmethod
    def forward(ctx, tensor):
        """
        Quantize-dequantize the tensor, using the saved encoding for this tensor
        :param tensor: Tensor to quantize-dequantize
        :return: Resulting tensor
        """

        tensor = tensor * 2 + 0.1
        ctx.save_for_backward(tensor)
        return tensor

    @staticmethod
    def backward(ctx, output_grad):
        tensor = ctx.saved_tensors
        # x = tensor[0]
        # encoding_max = 0.2
        # encoding_min = -1
        # inner_cond = torch.where(torch.le(x, encoding_max),  # condition to check per value
        #                          torch.ones_like(x),  # execute if true
        #                          torch.zeros_like(x))  # execute if false
        #
        # grad = (torch.where(torch.le(torch.tensor(encoding_min), x),  # condition to check per value
        #                            inner_cond,  # execute if true
        #                            torch.zeros_like(x))) * output_grad
        grad = compute_dloss_by_dx(tensor[0], output_grad, -1, 1)
        return grad