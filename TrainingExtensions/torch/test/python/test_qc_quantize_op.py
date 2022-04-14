# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
import time

import pytest
import numpy as np
import torch

from aimet_common.defs import MAP_ROUND_MODE_TO_PYMO, QuantizationDataType
from aimet_torch.qc_quantize_op import StaticGridQuantWrapper, LearnedGridQuantWrapper, SteGatingFuncForParameters, \
    QcQuantizeOpMode
from aimet_torch.qc_quantize_op import QuantScheme
from aimet_torch.quantsim_straight_through_grad import compute_dloss_by_dx_using_scale_offset, compute_dloss_by_dmin_using_dmax, compute_dloss_by_dmax
from aimet_torch.tensor_quantizer import StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer
from aimet_torch.tensor_quantizer import LearnedGridTensorQuantizer, ParameterQuantizer
import libpymo


class TestQcQuantizeOpStaticGrid:

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
                                             quant_scheme=QuantScheme.post_training_tf_enhanced, num_outputs=2)
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
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=False, enabled_by_default=True,
                                                 data_type=QuantizationDataType.int)
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

    def test_quantize_only_symmetric_signed_cpu(self):
        """ Test tensor quantizer quantize only symmetric signed functionality on cpu """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=True, enabled_by_default=True,
                                                 data_type=QuantizationDataType.int)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.19
        encodings.min = -5.20
        encodings.offset = -128

        # delta is 0.040745098
        quantizer.encoding = encodings

        # Test quantize only on cpu
        inp_tensor_gpu = torch.tensor([-7, -5, -3, 0, .1, 2.5])
        quant_out = quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([-128, -123, -74, 0, 2, 61], dtype=torch.float32)
        assert torch.equal(quant_out, expected_out)

    def test_quantizer_ignore_data_type_to_quantize_static_grid(self):
        relu = torch.nn.ReLU()
        quantizer = StaticGridQuantWrapper(relu, weight_bw=8, activation_bw=8, round_mode='nearest',
                                          quant_scheme=QuantScheme.post_training_tf_enhanced)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.19
        encodings.min = 0.0
        encodings.offset = 0

        # delta is 0.020352941
        quantizer.encoding = encodings

        inputs = torch.tensor([[True, True, False, False]])
        quant_out = quantizer._quantize_activation(quantizer.output_quantizers, inputs)
        expected_output = torch.tensor([True, True, False, False])
        assert torch.equal(quant_out, expected_output)

    def test_quantize_only_symmetric_unsigned_cpu(self):
        """ Test tensor quantizer quantize only symmetric unsigned functionality on cpu """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=True, enabled_by_default=True,
                                                 data_type=QuantizationDataType.int)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.19
        encodings.min = 0.0
        encodings.offset = 0

        # delta is 0.020352941
        quantizer.encoding = encodings

        # Test quantize only on cpu
        inp_tensor_gpu = torch.tensor([0, 1.2, 1.5, 4.0, 4.9, 5.3])
        quant_out = quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([0, 59, 74, 197, 241, 255], dtype=torch.float32)
        assert torch.equal(quant_out, expected_out)

    def test_quantize_dequantize_fp16_cpu(self):
        """ Test tensor quantizer quantize only symmetric unsigned functionality on cpu """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=16, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=True, enabled_by_default=True,
                                                 data_type=QuantizationDataType.float)

        # Test quantize-dequantize only on cpu
        inp_tensor = torch.tensor([0.3, 2.1, 1.1, 0.9, 0.1, 1.3])
        quant_out = quantizer.quantize_dequantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        assert torch.allclose(inp_tensor, quant_out, rtol=0.1)

    @pytest.mark.cuda
    def test_quantize_dequantize_fp16_gpu(self):
        """ Test tensor quantizer quantize only symmetric unsigned functionality on cpu """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=16, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=True, enabled_by_default=True,
                                                 data_type=QuantizationDataType.float)

        # Test quantize-dequantize only on cpu
        inp_tensor = torch.tensor([0.3, 2.1, 1.1, 0.9, 0.1, 1.3])
        quant_out = quantizer.quantize_dequantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        assert torch.allclose(inp_tensor, quant_out, rtol=0.1)

    def test_compute_encodings_fp16_disable_tensor(self):
        """ Negative test to make sure the encodings are not computed if 'enabled' field is set to False """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=16, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=True, enabled_by_default=True,
                                                 data_type=QuantizationDataType.float)

        # Test if encodings are computed for a float tensor
        quantizer.compute_encoding()
        assert quantizer.encoding == None

    @pytest.mark.cuda
    def test_quantize_only_asymmetric_gpu(self):
        """ Test tensor quantizer quantize only asymmetric functionality on gpu """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=False, enabled_by_default=True,
                                                 data_type=QuantizationDataType.int)
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
                                         quant_scheme=QuantScheme.post_training_tf,
                                         use_symmetric_encodings=True, enabled_by_default=True,
                                         data_type=QuantizationDataType.int)
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
                                         quant_scheme=QuantScheme.post_training_tf,
                                         use_symmetric_encodings=True, enabled_by_default=True,
                                         data_type=QuantizationDataType.int)
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
                                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                                           use_symmetric_encodings=False, enabled_by_default=True,
                                           data_type=QuantizationDataType.int)
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
                                                 quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                 use_symmetric_encodings=False, enabled_by_default=True,
                                                 data_type=QuantizationDataType.int)
        quantizer.compute_encoding()
        assert quantizer._encoding == []

    def test_custom_stegatingfuncforparameters(self):
        """
        test SteGatingFuncForParameters function
        """
        inputs = torch.randn(3, requires_grad=True)
        tensor_to_be_copied = torch.randn(3)
        print(tensor_to_be_copied)
        outputs = SteGatingFuncForParameters.apply(None, inputs)
        print(outputs)
        outputs = [out.clone() for out in outputs]
        # After clone(), following copy operation should succeed, otherwise throws RuntimeError
        outputs = [out.copy_(tensor_to_be_copied) for out in outputs]
        print(outputs)


class RoundStraightThrough(torch.autograd.Function):
    """
    Defining gradient of rounding function as pass-though since round is a non-linearity.
    """
    @staticmethod
    def forward(ctx, *x):
        return torch.round(*x)

    @staticmethod
    def backward(ctx, *output_grad):
        return output_grad


class CustomFunc(torch.autograd.Function):
    """
    This a custom function to perform custom gradient computation using
    to be performed for range learning.
    """

    @staticmethod
    def compute_scaling_offset(weight_min, weight_max):
        scaling = (weight_max - weight_min) / 255
        offset = RoundStraightThrough.apply(-weight_min / scaling)

        return scaling, offset

    @staticmethod
    def quant_dequant_param(x, weight_min, weight_max):
        delta = (weight_max - weight_min) / 255
        offset = RoundStraightThrough.apply(-weight_min / delta)
        x = x.clamp(weight_min.item(), weight_max.item())
        x = RoundStraightThrough.apply(x / delta) + offset
        x = (x - offset) * delta
        return x

    @staticmethod
    def forward(ctx, tensor, enc_min, enc_max, n, p):
        """
        Forward pass function
        :param ctx: context
        :param tensor: input tensor
        :param enc_min: encoding min
        :param enc_max: encoding max
        :param n: lower bound for a given bitwidth and symmetric/asymmetric encoding
        :param p: upper bound for a given bitwidth and symmetric/asymmetric encoding
        :return: computed quant dequant output
        """
        # save tensors for backward pass
        ctx.save_for_backward(tensor, enc_min, enc_max)
        scaling, offset = CustomFunc.compute_scaling_offset(enc_min, enc_max)
        # save params for backward pass
        ctx.offset = offset
        ctx.scaling = scaling
        ctx.n = n
        ctx.p = p
        y = CustomFunc.quant_dequant_param(tensor, enc_min, enc_max)

        return y

    @staticmethod
    def backward(ctx, grad):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the inputs.
        :param ctx: current context
        :param grad: gradients as a tensor containing gradient of loss w.r.t output
        :return: gradients w.r.t input, encoding_min and encoding_max and None for two extra params.
        """

        input, weight_min, weight_max = ctx.saved_tensors
        # print('--------------------------------')
        # print('gradient computation code ')
        # print('input, min, max passed are ', input, weight_min, weight_max)
        # print('--------------------------------')

        grad_input = compute_dloss_by_dx_using_scale_offset(input, grad, ctx.scaling, ctx.offset, ctx.n, ctx.p)
        grad_max = compute_dloss_by_dmax(input, grad, ctx.scaling, ctx.offset, ctx.n, ctx.p)
        grad_min = compute_dloss_by_dmin_using_dmax(grad_max)

        return grad_input, grad_min, grad_max, None, None


class TestQcQuantizeOpLearnedGrid:

    def test_trainable_tensor_quantizer_forward_backward(self):
        tensor_quantizer = LearnedGridTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                      quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                      use_symmetric_encodings=True,
                                                      enabled_by_default=True,
                                                      data_type=QuantizationDataType.int)

        encoding_min = torch.nn.Parameter(torch.FloatTensor([-5]))
        encoding_max = torch.nn.Parameter(torch.FloatTensor([5]))
        tensor = 10 * torch.rand((2, 1, 3, 5))
        tensor = tensor_quantizer.quantize_dequantize(tensor, encoding_min, encoding_max)

        assert np.amax(np.abs(tensor.detach().numpy()), axis=(0, 1, 2, 3)) <= 5
        assert np.amin(np.abs(tensor.detach().numpy()), axis=(0, 1, 2, 3)) >= -5

    @staticmethod
    def perform_auto_grad_computation(custom_input, min_value, max_value):
        """
        helper to perform auto grad computation
        :return:
        """

        def quant_dequant_param(x, weight_min, weight_max):
            delta = (weight_max - weight_min) / 255
            offset = RoundStraightThrough.apply(-weight_min / delta)
            x = x.clamp(weight_min.item(), weight_max.item())
            x = RoundStraightThrough.apply(x / delta) + offset
            x = (x - offset) * delta
            return x

        # use same tensor for auto and custom grad
        a_input_tensor = copy.deepcopy(custom_input)
        a_enc_min = torch.nn.Parameter(torch.Tensor([min_value]), requires_grad=True)
        a_enc_max = torch.nn.Parameter(torch.Tensor([max_value]), requires_grad=True)

        # after forward pass for quant op, we get output y
        y = quant_dequant_param(a_input_tensor, a_enc_min, a_enc_max)
        loss = y.flatten().sum()
        loss.backward()

        return a_input_tensor.grad, a_enc_min.grad, a_enc_max.grad

    def test_quantizer_ignore_data_type_to_quantize_learned_grid(self):
        relu = torch.nn.ReLU()
        quantizer = LearnedGridQuantWrapper(relu, weight_bw=8, activation_bw=8, round_mode='nearest',
                                            quant_scheme=QuantScheme.training_range_learning_with_tf_init, device='cpu')

        inputs = torch.tensor([[True, True, False, False]])
        quant_out = quantizer._quantize_activation(inputs, quantizer.output_quantizers, 'output')
        expected_output = torch.tensor([True, True, False, False])
        assert torch.equal(quant_out, expected_output)

    @staticmethod
    def perform_custom_grad_computation(custom_input, min_value, max_value, bw, sym_flag):

        c_input_tensor = copy.deepcopy(custom_input)
        c_enc_min = torch.nn.Parameter(torch.Tensor([min_value]), requires_grad=True)
        c_enc_max = torch.nn.Parameter(torch.Tensor([max_value]), requires_grad=True)
        # n = torch.nn.Parameter(torch.Tensor([0]), requires_grad=False)

        # To apply our Function, we use Function.apply method.
        # We alias this as 'custom_op'.
        custom_op = CustomFunc.apply

        # Forward pass: compute predicted y using operations; we compute
        # using our "custom" autograd operation.
        y_pred = custom_op(c_input_tensor, c_enc_min, c_enc_max, bw, sym_flag)

        # Compute and print loss
        loss = y_pred.flatten().sum()
        # Use custom grad to compute the backward pass.ctx.p = p
        loss.backward()

        # print(' grads after custom computation ...')
        # print(c_input_tensor.grad)
        # print(c_enc_min.grad)
        # print(c_enc_max.grad)

        return c_input_tensor.grad, c_enc_min.grad, c_enc_max.grad

    def test_custom_gradient_math_for_range_learning(self):
        """
        Unit test to validate custom gradient computation with auto grad computation.
        :return: None
        """

        dtype = torch.float
        device = torch.device("cpu")

        torch.manual_seed(0)
        custom_input = torch.rand((2, 2), requires_grad=True, dtype=dtype, device=device)

        a_input_grad, a_min_grad, a_max_grad = TestQcQuantizeOpLearnedGrid.perform_auto_grad_computation(custom_input,
                                                                                                         0.0015, 1.0)
        # custom gradient computation
        torch.manual_seed(0)
        custom_input = torch.rand((2, 2), requires_grad=True, dtype=dtype, device=device)

        # compute this one time and pass it to forward function
        n, p = LearnedGridTensorQuantizer.get_n_and_p(bitwidth=8, use_symmetric_encoding=False)
        c_input_grad, c_min_grad, c_max_grad = TestQcQuantizeOpLearnedGrid.perform_custom_grad_computation(custom_input,
                                                                                                           0.0015, 1.0,
                                                                                                           n, p)

        # validate gradients computed
        assert torch.allclose(c_input_grad.data[0], a_input_grad.data[0])
        assert torch.isclose(c_min_grad.data[0], a_min_grad.data[0])
        assert torch.isclose(c_max_grad.data[0], a_max_grad.data[0])

    def test_custom_gradient_for_range_learning_time_taken(self):
        """
        Unit test to check the time taken by custom gradient computation against auto grad computation
        :return: None
        """

        dtype = torch.float
        device = torch.device("cpu")

        torch.manual_seed(0)
        custom_input = torch.rand((2, 2), requires_grad=True, dtype=dtype, device=device)

        time_taken_by_auto = 0
        iterations = 10
        for i in range(1, iterations):
            start_time = time.perf_counter()
            _ = TestQcQuantizeOpLearnedGrid.perform_auto_grad_computation(custom_input, 0.0015, 1.0)
            exec_time = time.perf_counter() - start_time
            time_taken_by_auto = time_taken_by_auto + exec_time
        auto_average_time = time_taken_by_auto / iterations
        print('Avg time taken by auto grad', auto_average_time)

        # custom gradient computation
        time_taken_by_custom = 0
        torch.manual_seed(0)
        custom_input = torch.rand((2, 2), requires_grad=True, dtype=dtype, device=device)
        for i in range(1, iterations):
            # compute this one time and pass it to forward function
            n, p = LearnedGridTensorQuantizer.get_n_and_p(bitwidth=8, use_symmetric_encoding=False)
            start_time = time.perf_counter()
            _ = TestQcQuantizeOpLearnedGrid.perform_custom_grad_computation(custom_input, 0.0015, 1.0, n, p)
            exec_time = time.perf_counter() - start_time
            time_taken_by_custom = time_taken_by_custom + exec_time

        custom_avg_time = time_taken_by_custom/iterations

        print('Avg time taken by custom grad', custom_avg_time)
        print('Total % increase is ', ((custom_avg_time-auto_average_time)/auto_average_time) * 100)

    @pytest.mark.skip
    def test_compare_quantize_dequantize_cpp_python(self):
        torch.manual_seed(10)
        random_tensor = torch.rand((2, 3), requires_grad=True)

        encoding_min = torch.nn.Parameter(torch.FloatTensor([-5]))
        encoding_max = torch.nn.Parameter(torch.FloatTensor([5]))
        tensor_quantizer = LearnedGridTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                      quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                      use_symmetric_encodings=False,
                                                      enabled_by_default=True,
                                                      data_type=QuantizationDataType.int)
        out1 = tensor_quantizer.quantize_dequantize(random_tensor, encoding_min, encoding_max)

        post_training_tensor_quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                                      quant_scheme=QuantScheme.post_training_tf,
                                                                      use_symmetric_encodings=False,
                                                                      enabled_by_default=True)

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5
        encodings.min = -5
        encodings.delta = 1
        encodings.offset = 0.2
        post_training_tensor_quantizer.encoding = encodings

        out2 = post_training_tensor_quantizer.quantize_dequantize(random_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        assert np.allclose(out1.detach().numpy(), out2.detach().numpy())

    def test_n_p_computation(self):
        """
        validate n and p values computed for symmetric and asymmetric case.
        :return:
        """
        bitwidth = 8

        sym_n, sym_p = LearnedGridTensorQuantizer.get_n_and_p(bitwidth, True)

        # for 8 bit , -127 to +127
        expected_sym_n = (-2 ** (bitwidth - 1)) + 1
        expected_sym_p = (2 ** (bitwidth - 1)) - 1

        comp_symmetric_n = sym_n.data[0].item()
        comp_symmetric_p = sym_p.data[0].item()

        asym_n, asym_p = LearnedGridTensorQuantizer.get_n_and_p(bitwidth, False)

        # for 8 bit , 0 to 255
        expected_asym_n = 0
        expected_asym_p = (2 ** bitwidth) - 1
        comp_asymmetric_n = asym_n.data[0].item()
        comp_asymmetric_p = asym_p.data[0].item()

        assert expected_asym_n == comp_asymmetric_n
        assert expected_asym_p == comp_asymmetric_p

        assert expected_sym_n == comp_symmetric_n
        assert expected_sym_p == comp_symmetric_p

    def test_ste_gating_for_learnable_grid_wrapper(self):
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
                    print("None found for Gradient")

        conv1 = torch.nn.Conv2d(1, 2, 1)
        conv1.weight.data = torch.Tensor([[[[-0.8]]], [[[0.9]]]])
        quantize = LearnedGridQuantWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                           quant_scheme=QuantScheme.training_range_learning_with_tf_init, device='cpu',
                                           data_type=QuantizationDataType.int)
        quantize.train()
        quantize._module_to_wrap.register_backward_hook(hook_fn)

        quantize.input_quantizer.enabled = True
        quantize.output_quantizers[0].enabled = True
        quantize.input_quantizer.encoding = encodings
        quantize.output_quantizers[0].encoding = encodings
        quantize.param_quantizers['weight'].encoding = encodings
        quantize.param_quantizers['bias'].enabled = False

        new_input = torch.autograd.Variable(torch.tensor([[[[-0.8469]]], [[[0.9]]]]), requires_grad=True)
        out = quantize(new_input)

        quantize.input_quantizer.encoding = encodings_new
        quantize.output_quantizers[0].encoding = encodings_new
        quantize.param_quantizers['weight'].encoding = encodings_new

        loss = out.flatten().sum()
        loss.backward()

        # Check if input gradient got clipped
        for i, val in enumerate(new_input):
            if encodings_new.min > val or val > encodings_new.max:
                assert new_input.grad.flatten()[1] == 0.0

        # Check if output gradient got clipped
        output_grad = output_grad[0].flatten()
        assert output_grad[0] == 1.0
        assert output_grad[1] == 0.0
        assert output_grad[2] == 0.0
        assert output_grad[3] == 1.0

        # Check if weight gradient got clipped
        weight_tensor_grad = quantize._module_to_wrap.weight.grad.flatten()
        assert weight_tensor_grad[1] == 0.0

    def test_wrapper_for_in_place_operation(self):
        """
        Test wrapper for following in-place operation
        """
        module = torch.nn.Conv2d(3, 4, 2)
        wrapper = StaticGridQuantWrapper(module, weight_bw=8, activation_bw=8, round_mode='nearest',
                                         quant_scheme=QuantScheme.post_training_tf_enhanced)
        input_shape = (1, 3, 8, 8)
        input_var = torch.autograd.Variable(torch.randn(*input_shape), requires_grad=True)
        wrapper.set_mode(QcQuantizeOpMode.ANALYSIS)
        output = wrapper.forward(input_var)
        output += output # in-place operation should succeed
        wrapper.compute_encoding()
        wrapper.set_mode(QcQuantizeOpMode.ACTIVE)
        output = wrapper.forward(input_var)
        output += output    # in-place operation should succeed

    def test_set_enabled_for_param_quantizers(self):
        """
        Test set enabled flag for parameter quantizers
        """
        module = torch.nn.Conv2d(3, 4, 2)
        wrapper = StaticGridQuantWrapper(module, weight_bw=8, activation_bw=8, round_mode='nearest',
                                         quant_scheme=QuantScheme.post_training_tf_enhanced)
        # Disable bias quantization
        wrapper.param_quantizers['bias'].enabled = False
        wrapper.enable_param_quantizers(enabled=True)

        assert wrapper.param_quantizers['weight'].enabled == True
        assert wrapper.param_quantizers['bias'].enabled == False

        wrapper.enable_param_quantizers(enabled=False, param_name_to_exclude=("weight",))
        assert wrapper.param_quantizers['weight'].enabled == True
        assert wrapper.param_quantizers['bias'].enabled == False

        # Enable bias quantization
        wrapper.param_quantizers['bias'].enabled = True
        wrapper.enable_param_quantizers(enabled=False, param_name_to_exclude=None)
        assert wrapper.param_quantizers['weight'].enabled == False
        assert wrapper.param_quantizers['bias'].enabled == False