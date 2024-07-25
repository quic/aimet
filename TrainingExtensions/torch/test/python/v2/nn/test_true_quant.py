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

import functools

import pytest
import torch
from torch import randn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from torch.overrides import get_ignored_functions
from aimet_torch.v2.quantization.affine.backends import quantize, quantize_dequantize, dequantize
from aimet_torch.v2.quantization.affine import Quantize, QuantizeDequantize
from aimet_torch.v2.nn import (
    QuantizedConv1d,
    QuantizedConv2d,
    QuantizedConv3d,
    QuantizedConvTranspose1d,
    QuantizedConvTranspose2d,
    QuantizedConvTranspose3d,
    QuantizedGELU,
    QuantizedLinear,
    QuantizationMixin,
    QuantizedSigmoid,
    QuantizedSoftmax,
    QuantizedLayerNorm,
    QuantizedGroupNorm,
    QuantizedReLU,
    QuantizedPReLU,
    QuantizedConstantPad2d as QConstantPad2d,
    QuantizedHardtanh,
    QuantizedMaxPool2d,
    QuantizedUpsamplingBilinear2d as QUpsamplingBilinear2d,
    QuantizedPixelShuffle,
    QuantizedAvgPool2d,
    FakeQuantizationMixin,
)
from aimet_torch.v2.nn.true_quant import _dispatch
from aimet_torch.v2.quantization.affine import AffineEncoding
from aimet_torch.v2.quantization.tensor import QuantizedTensorBase, QuantizedTensor, DequantizedTensor
from aimet_torch.v2.utils import enable_recompute
import aimet_torch.nn.modules.custom as aimet_ops
import aimet_torch.v2.nn.quantized.custom as custom


def affine_quantize(tensor: torch.Tensor,
                    scale: torch.Tensor,
                    offset: torch.Tensor,
                    bitwidth: int) -> QuantizedTensor:
    """
    Quantizes the input tensor into a QuantizedTensor using the quantization parameters
    """
    tensor_q = quantize(tensor, scale, offset, bitwidth)
    encoding = AffineEncoding(scale, offset, bitwidth)
    qtensor = tensor_q.as_subclass(QuantizedTensor)
    qtensor.encoding = encoding
    return qtensor


def _input(*shape):
    numel = functools.reduce(lambda x, y: x * y, shape)
    return torch.arange(1,numel+1).view(*shape) / numel


@pytest.fixture
def input():
    return _input(10, 10)


@pytest.fixture(autouse=True)
def register_int_linear():
    def int_linear(input, weight, bias=None, *, output_encodings=None):
        # Implicit dequantization is not supported yet
        if not isinstance(input, QuantizedTensor):
            raise RuntimeError
        if not isinstance(weight, QuantizedTensor):
            raise RuntimeError

        input = input.dequantize()
        weight = weight.dequantize()

        return affine_quantize(input.mm(weight.t()) + bias,
                               output_encodings.scale,
                               output_encodings.offset,
                               output_encodings.bitwidth)

    QuantizedLinear.set_default_kernel(int_linear)
    yield
    QuantizedLinear.set_default_kernel(None)


@pytest.fixture(autouse=True)
def register_int_conv():
    def int_convnd(kernel, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, output_encodings=None):
        # Implicit dequantization is not supported yet
        if not isinstance(input, QuantizedTensor):
            raise RuntimeError
        if not isinstance(weight, QuantizedTensor):
            raise RuntimeError

        input = input.dequantize()
        weight = weight.dequantize()
        output = kernel(input, weight, bias, stride, padding, dilation, groups)
        return affine_quantize(output,
                               output_encodings.scale,
                               output_encodings.offset,
                               output_encodings.bitwidth)

    QuantizedConv1d.set_default_kernel(functools.partial(int_convnd, F.conv1d))
    QuantizedConv2d.set_default_kernel(functools.partial(int_convnd, F.conv2d))
    QuantizedConv3d.set_default_kernel(functools.partial(int_convnd, F.conv3d))
    yield
    QuantizedConv3d.set_default_kernel(None)
    QuantizedConv2d.set_default_kernel(None)
    QuantizedConv1d.set_default_kernel(None)


@pytest.fixture(autouse=True)
def register_int_conv():
    def int_convtransposend(kernel, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, *, output_encodings=None):
        # Implicit dequantization is not supported yet
        if not isinstance(input, QuantizedTensor):
            raise RuntimeError
        if not isinstance(weight, QuantizedTensor):
            raise RuntimeError

        input = input.dequantize()
        weight = weight.dequantize()
        output = kernel(input, weight, bias, stride, padding, output_padding, groups, dilation)
        return affine_quantize(output,
                               output_encodings.scale,
                               output_encodings.offset,
                               output_encodings.bitwidth)

    QuantizedConvTranspose1d.set_default_kernel(functools.partial(int_convtransposend, F.conv_transpose1d))
    QuantizedConvTranspose2d.set_default_kernel(functools.partial(int_convtransposend, F.conv_transpose2d))
    QuantizedConvTranspose3d.set_default_kernel(functools.partial(int_convtransposend, F.conv_transpose3d))
    yield
    QuantizedConvTranspose1d.set_default_kernel(None)
    QuantizedConvTranspose2d.set_default_kernel(None)
    QuantizedConvTranspose3d.set_default_kernel(None)


@pytest.fixture(autouse=True)
def register_int_activation():
    def wrap_functional(func):
        def wrapped_func(*args, output_encodings=None, **kwargs):
            # Implicit dequantization is not supported yet
            x, *others = args
            if not isinstance(x, QuantizedTensor):
                raise RuntimeError
            output = func(x.dequantize(), *others, **kwargs)
            return affine_quantize(output, output_encodings.scale, output_encodings.offset, output_encodings.bitwidth)

        return wrapped_func

    QuantizedSoftmax.set_default_kernel(wrap_functional(F.softmax))
    QuantizedSigmoid.set_default_kernel(wrap_functional(torch.sigmoid))
    QuantizedLayerNorm.set_default_kernel(wrap_functional(F.layer_norm))
    QuantizedGELU.set_default_kernel(wrap_functional(F.gelu))
    yield
    QuantizedGELU.set_default_kernel(None)
    QuantizedLayerNorm.set_default_kernel(None)
    QuantizedSigmoid.set_default_kernel(None)
    QuantizedSoftmax.set_default_kernel(None)


@pytest.fixture(autouse=True)
def register_int_norm():
    def wrap_functional(func):
        def int_norm(input, normalized_shape, weight, bias, eps, *, output_encodings=None):
            # Implicit dequantization is not supported yet
            if not isinstance(input, QuantizedTensor):
                raise RuntimeError
            if not isinstance(weight, QuantizedTensor):
                raise RuntimeError

            input = input.dequantize()
            weight = weight.dequantize()

            output = func(input.dequantize(), normalized_shape, weight, bias, eps)
            return affine_quantize(output, output_encodings.scale, output_encodings.offset, output_encodings.bitwidth)

        return int_norm

    QuantizedLayerNorm.set_default_kernel(wrap_functional(F.layer_norm))
    QuantizedGroupNorm.set_default_kernel(wrap_functional(F.group_norm))
    yield
    QuantizedGroupNorm.set_default_kernel(None)
    QuantizedLayerNorm.set_default_kernel(None)


@pytest.fixture(autouse=True)
def register_int_custom():
    def int_elementwise(kernel, x, y, *, output_encodings=None):
        # Implicit dequantization is not supported yet
        if not isinstance(x, QuantizedTensor):
            raise RuntimeError
        if not isinstance(y, QuantizedTensor):
            raise RuntimeError
        output = kernel(x.dequantize(), y.dequantize())
        return affine_quantize(output, output_encodings.scale, output_encodings.offset, output_encodings.bitwidth)

    custom.QuantizedAdd.set_default_kernel(functools.partial(int_elementwise, torch.add))
    custom.QuantizedMultiply.set_default_kernel(functools.partial(int_elementwise, torch.multiply))
    custom.QuantizedSubtract.set_default_kernel(functools.partial(int_elementwise, torch.subtract))
    custom.QuantizedDivide.set_default_kernel(functools.partial(int_elementwise, torch.div))
    custom.QuantizedMatMul.set_default_kernel(functools.partial(int_elementwise, torch.matmul))
    yield
    custom.QuantizedMultiply.set_default_kernel(None)
    custom.QuantizedSubtract.set_default_kernel(None)
    custom.QuantizedAdd.set_default_kernel(None)
    custom.QuantizedDivide.set_default_kernel(None)
    custom.QuantizedMatMul.set_default_kernel(None)



class TestTrueQuantLinear:

    def test_no_quantizers(self, input):
        """
        Given: TrueQuantLinear with no input, output, or param quantizers
        """
        quant_linear = QuantizedLinear(10, input.shape[-1])
        """
        When: inspect input/output/param quantizers
        Then: quantizers are None
        """
        assert quant_linear.input_quantizers[0] is None
        assert quant_linear.output_quantizers[0] is None
        assert quant_linear.param_quantizers["weight"] is None
        assert quant_linear.param_quantizers["bias"] is None
        """
        When: call forward pass within compute encodings context
        Then: output is equal to floating point output
        """
        expected_output = F.linear(input, quant_linear.weight, quant_linear.bias)
        with quant_linear.compute_encodings():
            output = quant_linear(input)
        assert torch.all(output == expected_output)
        """
        When: call forward pass outside of compute encodings context
        Then: raise RuntimeError
        """
        with pytest.raises(RuntimeError):
            quant_linear(input)

    def test_fully_specified_quantizers(self, input):
        """
        Given: TrueQuantLinear with input, output, and param quantizers
        """
        quant_linear = QuantizedLinear(10, input.shape[-1])
        quant_linear.input_quantizers[0] = Quantize((1, ), bitwidth=8, symmetric=False)
        quant_linear.output_quantizers[0] = Quantize((1, ), bitwidth=8, symmetric=False)
        quant_linear.param_quantizers["weight"] = Quantize((10, ), bitwidth=8, symmetric=True)
        """
        When: Call forward pass before computing encodings
        Then: raise RuntimeError
        """
        with pytest.raises(RuntimeError):
            quant_linear(input)

        """
        When: Invoke forward pass within compute_encodings context
        Then: Output should be equal to fake quant forward pass with activation quantizers disabled
        """
        with quant_linear.compute_encodings():
            output = quant_linear(input)

        input_enc = (quant_linear.input_quantizers[0].get_scale(),
                     quant_linear.input_quantizers[0].get_offset(),
                     quant_linear.input_quantizers[0].bitwidth)
        output_enc = (quant_linear.output_quantizers[0].get_scale(),
                      quant_linear.output_quantizers[0].get_offset(),
                      quant_linear.output_quantizers[0].bitwidth)
        weight_enc = (quant_linear.param_quantizers["weight"].get_scale(),
                      quant_linear.param_quantizers["weight"].get_offset(),
                      quant_linear.param_quantizers["weight"].bitwidth)
        weight_qdq = quantize_dequantize(quant_linear.weight, *weight_enc, signed=True)
        output_expected = F.linear(input, weight_qdq, bias=quant_linear.bias)
        assert torch.equal(output, output_expected)

        """
        When: Invoke forward pass outside of compute_encodings context with an unquantized tensor
        Then: 1) output should be computed using the global true quant backend
              2) output should be a quantized tensor
              3) output should be close to fake quant output after dequantization
        """
        input_qdq = quantize_dequantize(input, *input_enc)
        output_fp = F.linear(input_qdq, weight_qdq, bias=quant_linear.bias)
        output_expected = quantize_dequantize(output_fp, *output_enc)
        output_quant = quant_linear(input)
        assert isinstance(output_quant, DequantizedTensor)
        assert torch.allclose(output_quant.dequantize(), output_expected)

        """
        When: Invoke forward pass outside of compute_encodings context with a quantized tensor
        Then: Dequantized output should be close to running fake quant on the dequantized input tensor
        """
        quantized_input = affine_quantize(input, *input_enc)
        output = quant_linear(quantized_input)
        input_qdq = dequantize(quantized_input, *input_enc[:2])
        output_fp = F.linear(input_qdq, weight_qdq, bias=quant_linear.bias)
        output_expected = quantize_dequantize(output_fp, *output_enc)
        assert torch.allclose(output.dequantize(), output_expected)

    def test_no_input_quantizer(self, input):
        """
        Given: TrueQuantLinear with output and param quantizers and computed encodings
        """
        quant_linear = QuantizedLinear(10, input.shape[-1])
        quant_linear.output_quantizers[0] = Quantize((1, ), bitwidth=8, symmetric=False)
        quant_linear.param_quantizers["weight"] = Quantize((10, ), bitwidth=8, symmetric=True)
        with quant_linear.compute_encodings():
            quant_linear(input)
        """
        When: Invoke forward pass outside of compute_encodings with an unquantized tensor
        Then: raise RuntimeError
        """
        with pytest.raises(RuntimeError):
            quant_linear(input)

        """
        When: Invoke forward pass with a quantized tensor
        Then: return a tensor quantized with quant_linear.output_quantizer[0].encoding
        """
        quantizer = Quantize((1, ), bitwidth=8, symmetric=False)
        with quantizer.compute_encodings():
            quantizer(input)

        input_q = quantizer(input)
        output = quant_linear(input_q)
        assert isinstance(output, DequantizedTensor)
        assert output.encoding.scale == quant_linear.output_quantizers[0].get_scale()
        assert output.encoding.offset == quant_linear.output_quantizers[0].get_offset()


    def test_from_module(self, input):
        # Analogous to FakeQuantMixin.from_module test case
        """
        Given: Instantiate a true-quantized module using `TrueQuantMixin.from_module` and compute_encodings
        When: Inspect {input, output, param}_quantizers, they are the correct length
        """
        fp_linear = torch.nn.Linear(10, input.shape[-1])
        quant_linear = QuantizationMixin.from_module(fp_linear)

        assert len(quant_linear.input_quantizers) == 1
        assert len(quant_linear.output_quantizers) == 1
        assert len(quant_linear.param_quantizers) == 2

        """
        When: Inspect the parameters of the TrueQuant layer
        Then: They are identical to the parameters of the original layer
        """
        assert fp_linear.weight is quant_linear.weight
        assert fp_linear.bias is quant_linear.bias

        """
        When: Update to the parameter/buffer of the base FP module (or its submodule) using in-place operators.
              For example,
                1) fp_module.{param_or_buffer_name}.add_(1)
                2) fp_module.{submodule_name}.{param_or_buffer_name}.add_(1)
        Then: The result of in-place operation affects the parameters/buffers of the quantized module.
              In other words, the parameters/buffers of the quantized module will have been incremented by 1.
        """
        with torch.no_grad():
            fp_linear.weight.add_(1)
        assert torch.equal(fp_linear.weight, quant_linear.weight)
        with quant_linear.compute_encodings():
            quant_linear(input)

        """
        When: Reassign a new submodule/parameter/buffer to the base FP module using assignment stmt.
              For example,
                1) fp_module.{submodule_name} = torch.nn.Linear(...)
                2) fp_module.{param_or_buffer_name} = torch.empty(...)
        Then: The reassignment shouldn't affect the quantized module derived from the FP module.
              The vice versa should also hold.
        """
        fp_linear.weight = torch.nn.Parameter(torch.zeros(10, 10))
        assert not torch.all(fp_linear.weight == quant_linear.weight)


class TestQuantizedLayers:

    @pytest.mark.parametrize("layer,inputs", ((torch.nn.Softmax(dim=1), (_input(10, 10),)),
                                              (torch.nn.Sigmoid(), (_input(10, 10),)),
                                              (torch.nn.GELU(), (_input(10, 10),)),
                                              (aimet_ops.Add(), (_input(10, 10), _input(10, 10))),
                                              (aimet_ops.Multiply(), (_input(10, 10), _input(10, 10))),
                                              (aimet_ops.Subtract(), (_input(10, 10), _input(10, 10))),
                                              (aimet_ops.MatMul(), (_input(10, 10), _input(10, 10))),
                                              (aimet_ops.Divide(), (_input(10, 10), _input(10, 10)))))
    def test_layers_no_params(self, layer, inputs):
        fq_layer = FakeQuantizationMixin.from_module(layer)
        tq_layer = QuantizationMixin.from_module(layer)
        for i, _ in enumerate(inputs):
            fq_layer.input_quantizers[i] = QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
            tq_layer.input_quantizers[i] = Quantize(shape=(), bitwidth=8, symmetric=False)

        fq_layer.output_quantizers[0] = QuantizeDequantize(shape=(1, ), bitwidth=8, symmetric=False)
        tq_layer.output_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)

        with fq_layer.compute_encodings():
            fq_layer(*inputs)

        fq_output = fq_layer(*inputs)

        with tq_layer.compute_encodings():
            tq_layer(*inputs)
        tq_output = tq_layer(*inputs)

        assert torch.allclose(fq_output, tq_output.dequantize())

    @pytest.mark.parametrize("layer,input", ((torch.nn.Linear(10, 10), _input(10, 10)),
                                             (torch.nn.LayerNorm(10), _input(10, 10)),
                                             (torch.nn.GroupNorm(2, 10), _input(10, 10)),
                                             (torch.nn.Conv1d(3, 3, 3), _input(1, 3, 10)),
                                             (torch.nn.Conv2d(3, 3, 3), _input(1, 3, 10, 10)),
                                             (torch.nn.Conv3d(3, 3, 3), _input(1, 3, 10, 10, 10)),
                                             (torch.nn.ConvTranspose1d(3, 3, 3), _input(1, 3, 10)),
                                             (torch.nn.ConvTranspose2d(3, 3, 3), _input(1, 3, 10, 10)),
                                             (torch.nn.ConvTranspose3d(3, 3, 3), _input(1, 3, 10, 10, 10))))
    def test_layers_with_weight(self, layer, input):
        fq_layer = FakeQuantizationMixin.from_module(layer)
        tq_layer = QuantizationMixin.from_module(layer)
        fq_layer.input_quantizers[0] = QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
        fq_layer.output_quantizers[0] = QuantizeDequantize(shape=(1, ), bitwidth=8, symmetric=False)
        fq_layer.param_quantizers["weight"] = QuantizeDequantize(shape=(), bitwidth=8, symmetric=True)
        tq_layer.input_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)
        tq_layer.output_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)
        tq_layer.param_quantizers["weight"] = Quantize(shape=(), bitwidth=8, symmetric=True)

        with fq_layer.compute_encodings():
            fq_layer(input)

        fq_output = fq_layer(input)


        with tq_layer.compute_encodings():
            tq_layer(input)
        tq_output = tq_layer(input)

        assert torch.allclose(fq_output, tq_output.dequantize())

    @pytest.mark.cuda
    def test_layers_with_recompute(self):
        qlinear = QuantizedLinear(4096, 4096)
        qlinear.input_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)
        qlinear.output_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)
        qlinear.param_quantizers["weight"] = Quantize(shape=(), bitwidth=8, symmetric=True)
        qlinear.cuda()

        # Using dummy backend is no good for testing memory saving in real life.
        # Set kernel to None so as to use FakeQuantizedLinear under the hood.
        qlinear.set_kernel(None)

        x = torch.randn((100, 4096), device="cuda:0")

        with qlinear.compute_encodings():
            qlinear(x)

        torch.cuda.empty_cache()
        with enable_recompute():
            out = qlinear(x)
        torch.cuda.synchronize()
        mem_with_recompute = torch.cuda.memory_allocated()

        out.backward(torch.ones_like(out))
        grads_with_recompute = [param.grad.clone().detach().cpu() for param in qlinear.parameters()]
        for param in qlinear.parameters():
            param.grad = None

        del out

        torch.cuda.empty_cache()
        out = qlinear(x)
        torch.cuda.synchronize()
        mem_without_recompute = torch.cuda.memory_allocated()

        out.backward(torch.ones_like(out))
        grads_without_recompute = [param.grad.clone().detach().cpu() for param in qlinear.parameters()]
        for param in qlinear.parameters():
            param.grad = None

        # Expected memory saving:
        #   - Input quantizer save:
        #      - mask of shape [100, 4096] * 1 byte
        #      - quantized uint8 tensor of shape [100, 4096] * 1 byte
        #   - Weight quantizer saves:
        #      - mask of shape [4096, 4096] * 1 byte
        #      - quantized uint8 tensor of shape [4096, 4096] * 1 byte
        #   - F.linear saves:
        #      - quantized weight of shape [4096, 4096] * 4 bytes
        #      - quantized input of shape [100, 4096] * 4 bytes
        #   - Output quantizer saves:
        #      - linear output of shape [100, 4096] * 4 bytes
        #      - mask of shape [100, 4096] * 1 byte
        #      - quantized uint8 tensor of shape [100, 4096] * 1 byte
        expected_memory_saving = 0
        expected_memory_saving += (1 + 1) * x.numel() # input quantizer
        expected_memory_saving += (1 + 1) * qlinear.weight.numel() # weight quantizer
        expected_memory_saving += 4 * (qlinear.weight.numel() + x.numel()) # F.linear
        expected_memory_saving += (4 + 1 + 1) * out.numel() # output quantizer
        actual_memory_saving = mem_without_recompute - mem_with_recompute

        # Considering noise factors, actual memory saving should be no less than
        # 90% of the expected memory saving
        assert expected_memory_saving * 0.9 <= actual_memory_saving

        for grad_0, grad_1 in zip(grads_with_recompute, grads_without_recompute):
            assert torch.equal(grad_0, grad_1)

    def test_remove_quantizers(self, input):
        qlinear = QuantizedLinear(10, 10, bias=False)
        qlinear.input_quantizers[0] = input_qtzr = Quantize(shape=(), bitwidth=8, symmetric=False)
        qlinear.output_quantizers[0] = output_qtzr = Quantize(shape=(), bitwidth=8, symmetric=False)
        qlinear.param_quantizers["weight"] = weight_qtzr = Quantize(shape=(), bitwidth=8, symmetric=True)
        with qlinear.compute_encodings():
            qlinear(input)

        qlinear.set_kernel(None) # Set kernel to None since removing quantizers does not work with integer kernels

        """
        When: ``with _remove_{input, param, output, activation, all}_quantizers``
        Then:
            1) The corresponding quantizers are set to None under the context.
               (Output should be computed without input, param, and output quantization respectively)
            2) The corresponding quantizers are restored when exiting the context.
        """
        with qlinear._remove_input_quantizers(0):
            assert qlinear.input_quantizers[0] is None
            expected_out = output_qtzr(
                F.linear(
                    input, weight_qtzr(qlinear.weight).dequantize()
                )
            ).dequantize()
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.input_quantizers[0] is input_qtzr

        with qlinear._remove_param_quantizers('weight'):
            assert qlinear.param_quantizers['weight'] is None
            expected_out = output_qtzr(
                F.linear(
                    input_qtzr(input).dequantize(), qlinear.weight
                )
            ).dequantize()
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.param_quantizers['weight'] is weight_qtzr

        with qlinear._remove_output_quantizers(0):
            assert qlinear.output_quantizers[0] is None
            expected_out = F.linear(input_qtzr(input).dequantize(),
                                    weight_qtzr(qlinear.weight).dequantize())
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.output_quantizers[0] is output_qtzr

        with qlinear._remove_activation_quantizers():
            assert qlinear.input_quantizers[0] is None
            assert qlinear.output_quantizers[0] is None
            expected_out = F.linear(input, weight_qtzr(qlinear.weight).dequantize())
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.input_quantizers[0] is input_qtzr
        assert qlinear.output_quantizers[0] is output_qtzr

        with qlinear._remove_all_quantizers():
            assert qlinear.input_quantizers[0] is None
            assert qlinear.output_quantizers[0] is None
            assert qlinear.param_quantizers['weight'] is None
            expected_out = F.linear(input, qlinear.weight)
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.input_quantizers[0] is input_qtzr
        assert qlinear.output_quantizers[0] is output_qtzr
        assert qlinear.param_quantizers['weight'] is weight_qtzr

        """
        When: Call ``_remove_{input, param, output}_quantizers`` without ``with`` statement
        Then: The corresponding quantizers are set to None permanently
        """
        qlinear._remove_input_quantizers(0)
        assert qlinear.input_quantizers[0] is None
        qlinear._remove_param_quantizers('weight')
        assert qlinear.param_quantizers['weight'] is None
        qlinear._remove_output_quantizers(0)
        assert qlinear.output_quantizers[0] is None


def _pseudo_integer_kernel_helper(fn):
    """
    Helper function for creating a pseudo-integer kernel of ``fn``.
    Pseudo-integer kernel is a function that takes/returns QuantizedTensors,
    but internally delegates actual computation to ``fn``.
    """
    def pseudo_kernel(*args, **kwargs):
        assert "output_encodings" in kwargs
        output_encodings = kwargs.pop("output_encodings")

        def dequantize(t: torch.Tensor):
            if isinstance(t, torch.Tensor):
                if not isinstance(t, QuantizedTensorBase):
                    raise RuntimeError
                t = t.dequantize()
            return t

        # Dequantize quantized input tensors
        args = tree_map(dequantize, args)
        kwargs = tree_map(dequantize, kwargs)

        # Delegate actual computation to ``fn``
        out = fn(*args, **kwargs)

        # Quantize output
        out = output_encodings.quantize(out).as_subclass(QuantizedTensor)
        out.encoding = output_encodings
        return out

    return pseudo_kernel


pseudo_relu_kernel = _pseudo_integer_kernel_helper(F.relu)
pseudo_prelu_kernel = _pseudo_integer_kernel_helper(F.prelu)
pseudo_pad_kernel = _pseudo_integer_kernel_helper(F.pad)
pseudo_hardtanh_kernel = _pseudo_integer_kernel_helper(F.hardtanh)
pseudo_max_pool2d_kernel = _pseudo_integer_kernel_helper(F.max_pool2d)
pseudo_interpolate_kernel = _pseudo_integer_kernel_helper(F.interpolate)
pseudo_pixel_shuffle_kernel = _pseudo_integer_kernel_helper(F.pixel_shuffle)
pseudo_sin_kernel = _pseudo_integer_kernel_helper(torch.sin)
pseudo_cos_kernel = _pseudo_integer_kernel_helper(torch.cos)
pseudo_avg_pool2d_kernel = _pseudo_integer_kernel_helper(F.avg_pool2d)
pseudo_reshape_kernel = _pseudo_integer_kernel_helper(torch.reshape)
pseudo_rsqrt_kernel = _pseudo_integer_kernel_helper(torch.rsqrt)
pseudo_add_kernel = _pseudo_integer_kernel_helper(torch.add)
pseudo_mul_kernel = _pseudo_integer_kernel_helper(torch.mul)
pseudo_sub_kernel = _pseudo_integer_kernel_helper(torch.sub)
pseudo_div_kernel = _pseudo_integer_kernel_helper(torch.div)


@pytest.mark.parametrize(
    "qmodule,                       pseudo_kernel,              inputs",           [
     (QuantizedReLU(),              pseudo_relu_kernel,         randn(100)),
     (QuantizedPReLU(),             pseudo_prelu_kernel,        randn(100)),
     (QConstantPad2d([1,1,1,1], 0), pseudo_pad_kernel,          randn(10,10)),
     (QuantizedHardtanh(),          pseudo_hardtanh_kernel,     randn(100)),
     (QuantizedMaxPool2d([3,3]),    pseudo_max_pool2d_kernel,   randn(1,10,10)),
     (QUpsamplingBilinear2d([2,2]), pseudo_interpolate_kernel,  randn(1,1,10,10)),
     (QuantizedPixelShuffle(1),     pseudo_pixel_shuffle_kernel,randn(1,1,10,10)),
     (QuantizedAvgPool2d(2),        pseudo_avg_pool2d_kernel,   randn(1,10,10)),
     (custom.QuantizedSin(),        pseudo_sin_kernel,          randn(100)),
     (custom.QuantizedCos(),        pseudo_cos_kernel,          randn(100)),
     (custom.QuantizedAvgPool2d(),  pseudo_avg_pool2d_kernel,   (randn(1,10,10), 2)),
     (custom.QuantizedReshape(),    pseudo_reshape_kernel,      (randn(10,10), (100, 1))),
     (custom.QuantizedRSqrt(),      pseudo_rsqrt_kernel,        randn(100).abs()),
     (custom.QuantizedAdd(),        pseudo_add_kernel,          (randn(100), randn(100))),
     (custom.QuantizedMultiply(),   pseudo_mul_kernel,          (randn(100), randn(100))),
     (custom.QuantizedSubtract(),   pseudo_sub_kernel,          (randn(100), randn(100))),
     (custom.QuantizedDivide(),     pseudo_div_kernel,          (randn(100), randn(100))),
])
def test_sanity(qmodule, pseudo_kernel, inputs):
    """
    When: Set qmodule's kernel as pseudo-integer kernel
    Then: Output of qmodule should be same as that of fake-quantization fallback
    """
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)

    for i, _ in enumerate(qmodule.input_quantizers):
        qmodule.input_quantizers[i] = Quantize((), 8, False)

    for i, _ in enumerate(qmodule.output_quantizers):
        qmodule.output_quantizers[i] = Quantize((), 8, False)

    if 'weight' in qmodule.param_quantizers:
        qmodule.param_quantizers['weight'] = Quantize((), 8, True)

    with qmodule.compute_encodings():
        _ = qmodule(*inputs)

    out_fakequant = qmodule(*inputs)

    qmodule.set_kernel(pseudo_kernel)
    out_truequant = qmodule(*inputs)

    assert torch.equal(out_fakequant, out_truequant)


def test_dispatch_sanity():
    custom_add = lambda *args, **kwargs: torch.add(*args, **kwargs) + 1

    """
    When: Dispatch torch.add with custom_add(x, y) := x + y + 1
    Then: Output of torch.add(x, y) should be equal to x + y + 1
    """
    zeros = torch.zeros(10)
    with _dispatch(torch.add, custom_add):
        out = torch.add(zeros, zeros)
    assert torch.all(out == 1)

    with _dispatch(torch.Tensor.add, custom_add):
        out = zeros + zeros
    assert torch.all(out == 1)

    """
    When: Dispatch torch.add with custom_add(x, y) := x + y + 1
    Then: Output of the other functions should not be affected
    """
    with _dispatch(torch.add, custom_add):
        zeros = torch.zeros(10)
        ones = torch.ones(10)
        twos = ones * 2
        fours = twos.square()
        threes = fours - twos / 2

    assert torch.all(zeros == 0)
    assert torch.all(ones == 1)
    assert torch.all(twos == 2)
    assert torch.all(threes == 3)
    assert torch.all(fours == 4)

    """
    When: Try to dispatch unsupported functions
    Then: Throw runtime error
    """
    for func in get_ignored_functions():
        dummy_impl = lambda *args, **kwargs: func(*args, **kwargs)
        with pytest.raises(RuntimeError):
            with _dispatch(func, dummy_impl): pass
