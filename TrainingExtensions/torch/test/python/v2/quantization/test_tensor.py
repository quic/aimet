# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
from aimet_torch.v2.quantization.affine import quantize, quantize_dequantize
from aimet_torch.v2.quantization.tensor import QuantizedTensor, DequantizedTensor
from aimet_torch.v2.quantization.affine import AffineEncoding


@pytest.fixture
def scale():
    return torch.tensor(0.1)

@pytest.fixture
def offset():
    return torch.tensor(-10.)

@pytest.fixture
def bitwidth():
    return 8


def affine_quantize(tensor: torch.Tensor,
                    scale: torch.Tensor,
                    offset: torch.Tensor,
                    bitwidth: int,
                    signed: bool = False) -> QuantizedTensor:
    """
    Quantizes the input tensor into a QuantizedTensor using the quantization parameters
    """
    tensor_q = quantize(tensor, scale, offset, bitwidth)
    encoding = AffineEncoding(scale, offset, bitwidth)
    qtensor = tensor_q.as_subclass(QuantizedTensor)
    qtensor.encoding = encoding
    return qtensor


class TestQuantizedTensor:
    @pytest.mark.parametrize('qtensor_cls', [QuantizedTensor, DequantizedTensor])
    @pytest.mark.cuda
    def test_qtensor_sanity(self, qtensor_cls, scale, offset, bitwidth):
        """
        When: Instantiate QuantizedTensor/DequantizedTensor from a torch.Tensor.as_subclass
        Then: The created QuantizedTensor/DequantizedTensor is equal to the input tensor
        """
        data = torch.arange(256, dtype=torch.float) # actual content of qtensor
        qtensor = data.clone().as_subclass(qtensor_cls)
        qtensor.encoding = AffineEncoding(scale, offset, bitwidth)
        assert torch.equal(qtensor, data)

        """
        Given: QuantizedTensor/DequantizedTensor with fp32 dtype on cpu
        When: Call float() / cpu() / to(device='cpu', dtype=torch.float32)
        Then: The output tensor is the identical object as itself
        """
        assert qtensor.float() is qtensor
        assert qtensor.cpu() is qtensor
        assert qtensor.to(torch.float32) is qtensor
        assert qtensor.to('cpu') is qtensor
        assert qtensor.to('cpu', torch.float32) is qtensor

        """
        When: Call quantize()/dequantize()
        Then: 1) The output tensor is an instance of DequantizedTensor
              2) The output tensor inherits a shallow copy of its input tensor's encoding
        """
        qtensor_q = qtensor.quantize()
        assert isinstance(qtensor_q, QuantizedTensor)
        # assert qtensor_q.encoding is not qtensor.encoding
        assert torch.equal(qtensor_q.encoding.scale, qtensor.encoding.scale)
        assert torch.equal(qtensor_q.encoding.offset, qtensor.encoding.offset)
        assert qtensor_q.encoding.bitwidth == qtensor.encoding.bitwidth
        assert qtensor_q.encoding.signed == qtensor.encoding.signed
        qtensor_dq = qtensor.dequantize()
        assert isinstance(qtensor_dq, DequantizedTensor)
        # assert qtensor_dq.encoding is not qtensor.encoding
        assert torch.equal(qtensor_dq.encoding.scale, qtensor.encoding.scale)
        assert torch.equal(qtensor_dq.encoding.offset, qtensor.encoding.offset)
        assert qtensor_dq.encoding.bitwidth == qtensor.encoding.bitwidth
        assert qtensor_dq.encoding.signed == qtensor.encoding.signed


        for cast_fn in [torch.Tensor.half, torch.Tensor.double, torch.Tensor.cuda]:
            """
            Given: QuantizedTensor with fp32 dtype on cpu
            When: Cast to different dtype or device
            Then: 1) The output tensor is an instance of QuantizedTensor with the same value
            """
            data = torch.arange(256, dtype=torch.float) # actual content of qtensor
            qtensor = data.clone().as_subclass(qtensor_cls)
            qtensor.encoding = AffineEncoding(scale, offset, bitwidth)
            qtensor_casted = cast_fn(qtensor)

            assert qtensor_casted is not qtensor
            assert isinstance(qtensor_casted, qtensor_cls)
            assert torch.equal(qtensor_casted, cast_fn(data))

            """
            Then: 2) The output tensor inherits a shallow copy of its input tensor's encoding
            """
            # assert qtensor_casted.encoding is not qtensor.encoding
            assert torch.equal(qtensor_casted.encoding.scale.cpu(), qtensor.encoding.scale)
            assert torch.equal(qtensor_casted.encoding.offset.cpu(), qtensor.encoding.offset)
            assert qtensor_casted.encoding.bitwidth == qtensor.encoding.bitwidth
            assert qtensor_casted.encoding.signed == qtensor.encoding.signed

            """
            Then: 3) The result of quantize()/dequantize() should be similar before/after casting
            """
            if cast_fn != torch.Tensor.half: # NOTE: torch.round is not supported for cpu tensors of float16 dtype
                assert torch.equal(qtensor_casted.quantize().cpu().float(), qtensor.quantize())
            assert torch.allclose(qtensor_casted.dequantize().cpu().float(), qtensor.dequantize(), atol=scale)

            """
            When: Cast twice
            Then: The output tensor is the identical object as itself
            """
            assert cast_fn(qtensor_casted) is qtensor_casted


        """
        When: Instantiate QuantizedTensor with wrong non-floating point dtype directly using __new__
        Then: Throw error
        """
        data = torch.arange(256, dtype=torch.long) # actual content of qtensor
        with pytest.raises(RuntimeError):
            _ = qtensor_cls(data)

        for cast_fn in [torch.Tensor.char, torch.Tensor.short, torch.Tensor.int, torch.Tensor.long]:
            """
            Given: QuantizedTensor
            When: Cast to non-floating point dtypes
            Then: Throw error
            """
            data = torch.arange(256, dtype=torch.float) # actual content of qtensor
            qtensor = data.clone().as_subclass(qtensor_cls)
            qtensor.encoding = AffineEncoding(scale, offset, bitwidth)

            with pytest.raises(RuntimeError):
                _ = cast_fn(qtensor)


    @pytest.mark.parametrize("scale, offset, bitwidth, signed", 
                             [(torch.tensor(0.1), torch.tensor(-125.), 8, False),
                              (torch.tensor(0.1), torch.tensor(-4.), 8, True),
                              (torch.tensor(0.1), torch.tensor(-10.), 16, False),
                              (torch.tensor(0.1), torch.tensor(0.), 4, True)])
    def test_quantized_representation(self, scale, offset, bitwidth, signed):
        """
        Given: QuantizedTensor with an encoding object
        When: Call qtensor.quantized_repr()
        Then: 1) Return value is a tensor of type encoding.dtype
        """
        tensor = torch.arange(-10, 10, 0.1)
        qtensor = affine_quantize(tensor, scale, offset, bitwidth, signed)
        quant_repr = qtensor.quantized_repr()
        assert quant_repr.dtype == qtensor.encoding.dtype
        """
        Then: 2) Quantized values are equal to calling quantize(tensor, encoding.scale, encoding.offset, encoding.bitwidth)
        """
        assert torch.allclose(quant_repr.to(torch.float32), quantize(tensor, scale, offset, bitwidth))

    @pytest.mark.cuda()
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("scale, offset, bitwidth, signed",
                             [(torch.tensor(0.1), torch.tensor(-125.), 8, False)])
    def test_dequantize_self(self, scale, offset, bitwidth, signed, dtype):
        """
        Given: A tensor and encoding of the same dtype
        """
        device = 'cuda'
        tensor = torch.randn(10, 10).to(dtype=dtype, device=device)
        scale, offset = scale.to(dtype=dtype, device=device), offset.to(dtype=dtype, device=device)
        qtensor = affine_quantize(tensor, scale, offset, bitwidth, signed)
        """
        When: Create qtensor: QuantizedTensor from the tensor and encoding
        Then: 1) qtensor.dequantize() has the same dtype as the original tensor
        """
        dq_tensor = qtensor.dequantize()
        assert not isinstance(dq_tensor, QuantizedTensor)
        assert dq_tensor.dtype == dtype
        """
        Then: 2) Returned tensor has values equivalent to calling quantize_dequantize(tensor, encoding.scale, encoding.offset, encoding.bitwidth)
        """
        assert torch.allclose(dq_tensor, quantize_dequantize(tensor, scale, offset, bitwidth))
       
    @pytest.mark.cuda() 
    @pytest.mark.parametrize("devices", [(torch.device("cpu"), torch.device("cuda:0")),
                                         (torch.device("cuda:0"), torch.device("cpu"))])
    @pytest.mark.parametrize("scale, offset, bitwidth, signed",
                             [(torch.tensor(0.1), torch.tensor(-125.), 8, False)])
    def test_qtensor_device(self, devices, scale, offset, bitwidth, signed):
        """
        Given: Instantiated a qtensor on device_1
        """
        device_1, device_2 = devices
        tensor = torch.randn((10, )).to(device_1)
        scale, offset = scale.to(device_1), offset.to(device_1)
        qtensor = affine_quantize(tensor, scale, offset, bitwidth, signed)
        """
        When: Inspect qtensor.device
        Then: Device is device_1
        """
        assert qtensor.device == device_1
        """
        When: Send qtensor to device 2
        Then: 1) qtensor_d2.device is device_2
              2) original qtensor.device is still device_1
        """
        qtensor_d2 = qtensor.to(device_2)
        assert qtensor_d2.device == device_2
        assert qtensor.device == device_1
        """
        Then: 1) qtensor_d2.encoding is on the correct device
              2) qtensor.encoding is still on the correct device
        """
        assert qtensor_d2.encoding.scale.device == device_2
        assert qtensor.encoding.scale.device == device_1
        """
        Then: qtensor_d2.dequantize() produces a tensor on the correct device
        """
        assert qtensor_d2.dequantize().device == device_2
        """
        Then: qtensor_d2.quantized_repr() produces a tensor on the correct device
        """
        assert qtensor_d2.quantized_repr().device == device_2

    @pytest.mark.skip(reason="implicit dequantization is not supported yet")
    @pytest.mark.parametrize("scale, offset, bitwidth, signed",
                             [(torch.tensor(0.1), torch.tensor(-125.), 8, False)])
    def test_propagate_gradient(self, scale, offset, bitwidth, signed):
        """
        Given: Create a QuantizedTensor from a tensor with tensor.requires_grad = True
        When: 1) Pass the QuantizedTensor into an autograd function
              2) Call backward() on the output
        Then: The gradient propagates correctly to the original tensor
        """
        
        class MultiplyBy2(torch.autograd.Function):
            
            def forward(ctx, qtensor):
                dq_tensor = qtensor.dequantize()
                return affine_quantize(dq_tensor * 2, scale, offset, bitwidth, signed)
            
            def backward(ctx, *grad_outputs):
                return grad_outputs[0] * 2
            
        shape = (5, 5)
        tensor = torch.rand(shape, requires_grad=True)
        grad_in = torch.randn_like(tensor)
        qtensor = affine_quantize(tensor, scale, offset, bitwidth, signed)
        qoutput = MultiplyBy2.apply(qtensor)
        # This should actually be calling qoutput.dequantize().backward(grad_in)
        qoutput.backward(grad_in)
        tensor_grad = tensor.grad.clone()
        tensor.grad.zero_()
        fp_output = quantize_dequantize(tensor, scale, offset, bitwidth) * 2
        fp_output.backward(grad_in)
        assert torch.allclose(tensor_grad, tensor.grad)

    @pytest.mark.skip(reason="implicit dequantization is not supported yet")
    @pytest.mark.parametrize("scale, offset, bitwidth, signed",
                             [(torch.tensor(0.1), torch.tensor(-125.), 8, False)])
    def test_fallback_to_dequantize(self, scale, offset, bitwidth, signed):
        """
        Given: a QuantizedTensor qtensor
        When: Pass qtensor to a floating point operation
        Then: 1) Dequantize qtensor before calling the operation
              2) Output is not a QuantizedTensor
        """
        qdq = quantize_dequantize
        input_shape = (16, 16)
        tensor = torch.randn(input_shape)
        tensor_qdq = qdq(tensor, scale, offset, bitwidth)
        qtensor = affine_quantize(tensor, scale, offset, bitwidth, signed)
        other = torch.randn_like(tensor)
        
        qtensor_out = qtensor + other
        assert not isinstance(qtensor_out, QuantizedTensor)
        expected_out = tensor_qdq + other
        assert torch.allclose(qtensor_out, expected_out)
        
        qtensor_out = torch.nn.functional.linear(other, qtensor, None)
        expected_out = torch.nn.functional.linear(other, tensor_qdq, None)
        assert torch.allclose(qtensor_out, expected_out)
        
        """
        When: Call a native pytorch op on multiple QuantizedTensor objects
        Then: 1) Dequantize all QuantizedTensors before calling the op
              2) Output is not a QuantizedTensor
        """
        qother = affine_quantize(other, scale / 2, offset, bitwidth, signed)
        other_qdq = qdq(other, scale / 2, offset, bitwidth)
        qtensor_out = torch.nn.functional.linear(qother, qtensor)
        assert not isinstance(qtensor_out, QuantizedTensor)
        assert torch.allclose(qtensor_out, torch.nn.functional.linear(other_qdq, tensor_qdq))

    @pytest.mark.parametrize('qtensor_cls', [QuantizedTensor, DequantizedTensor])
    @pytest.mark.parametrize('op, args, kwargs', [
        (torch.Tensor.clone, (), {}),
        (torch.Tensor.expand, ((2, 128, 2), ), {}),
        (torch.Tensor.expand_as, (torch.randn(2, 128, 2), ), {}),
        (torch.Tensor.permute, (0, 2, 1), {}),
        (torch.Tensor.repeat, ((2, 128, 2)), {}),
        (torch.Tensor.reshape, (128, 2, 1), {}),
        (torch.Tensor.resize, ((2, 64, 2)), {}),
        (torch.Tensor.unsqueeze, (-1, ), {}),
        (torch.Tensor.squeeze, (-1, ), {}),
        (torch.Tensor.view, (-1, ), {}),
        (torch.flatten, (), {}),
        (torch.detach, (), {}),
        (torch.view_copy, ((-1, ), ), {})
    ])
    def test_propagate_pertensor_encoding(self, qtensor_cls, op, args, kwargs, scale, offset, bitwidth):
        shape = (2, 128, 1)
        data = torch.empty(shape)
        qtensor = data.clone().as_subclass(qtensor_cls)
        qtensor.encoding = AffineEncoding(scale, offset, bitwidth)
        """
        Given: Per-tensor quantized tensor object
        When: Call a 'math invariant' tensor operation on the quantized tensor
        Then: 1) Output is also a quantized tensor
              2) Output encoding matches input encoding
              3) Output encoding is not the same object as input encoding
        """
        output = op(qtensor, *args, **kwargs)

        assert isinstance(output, qtensor_cls)
        assert torch.equal(output.encoding.scale, qtensor.encoding.scale)
        assert torch.equal(output.encoding.offset, qtensor.encoding.offset)
        assert output.encoding.bitwidth == qtensor.encoding.bitwidth
        assert output.encoding.signed == qtensor.encoding.signed
        assert output.encoding is not qtensor.encoding

    @pytest.mark.parametrize('qtensor_cls', [QuantizedTensor, DequantizedTensor])
    @pytest.mark.parametrize('op, args, kwargs', [
        (torch.Tensor.add, (torch.randn(2, 128, 1), ), {}),
        (torch.Tensor.bmm, (torch.randn(2, 1, 128),), {})
    ])
    def test_dont_propagate_pertensor_encoding(self, qtensor_cls, op, args, kwargs, scale, offset, bitwidth):
        shape = (2, 128, 1)
        data = torch.empty(shape)
        qtensor = data.clone().as_subclass(qtensor_cls)
        qtensor.encoding = AffineEncoding(scale, offset, bitwidth)
        """
        Given: Per-tensor quantized tensor object
        When: Call non 'math invariant' tensor operation on the quantized tensor
        Then: Output is not a quantized tensor
        """
        output = op(qtensor, *args, **kwargs)
        assert not isinstance(output, qtensor_cls)

    @pytest.mark.parametrize('qtensor_cls', [QuantizedTensor, DequantizedTensor])
    @pytest.mark.parametrize('op, args, kwargs', [
        (torch.Tensor.expand, ((2, 128, 2), ), {}),
        (torch.Tensor.expand_as, (torch.randn(2, 128, 2), ), {}),
        (torch.Tensor.permute, (0, 2, 1), {}),
        (torch.Tensor.repeat, ((2, 128, 2)), {}),
        (torch.Tensor.reshape, (128, 2, 1), {}),
        (torch.Tensor.resize, ((2, 64, 2)), {}),
        (torch.Tensor.unsqueeze, (-1, ), {}),
        (torch.Tensor.squeeze, (-1, ), {}),
        (torch.Tensor.view, (-1, ), {}),
        (torch.flatten, (), {}),
        (torch.view_copy, ((-1, ), ), {})
    ])
    def test_dont_propagate_perchannel_encoding(self, qtensor_cls, op, args, kwargs, bitwidth):
        scale = torch.randn(2, 1, 1)
        offset = torch.zeros_like(scale)
        shape = (2, 128, 1)
        data = torch.empty(shape)
        qtensor = data.clone().as_subclass(qtensor_cls)
        qtensor.encoding = AffineEncoding(scale, offset, bitwidth)
        """
        Given: Per-channel quantized tensor object
        When: Call an op which changes the dimensions of the tensor
        Then: Output is not a quantized tensor
        """
        output = op(qtensor, *args, **kwargs)
        assert not isinstance(output, qtensor_cls)