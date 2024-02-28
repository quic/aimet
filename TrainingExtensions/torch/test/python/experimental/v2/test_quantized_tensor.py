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
import torch.nn.functional as F
from aimet_torch.experimental.v2.quantization.backends import get_backend
from aimet_torch.experimental.v2.quantization.quantized_tensor import QuantizedTensor
from aimet_torch.experimental.v2.quantization.encodings import AffineEncoding


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
    tensor_q = get_backend().quantize(tensor, scale, offset, bitwidth)
    encoding = AffineEncoding(scale, offset, bitwidth)
    dequant = get_backend().dequantize
    dequant_fn = lambda t: dequant(torch.Tensor(t), t.encoding.scale, t.encoding.offset)
    qtensor = QuantizedTensor(tensor_q, encoding, dequant_fn=dequant_fn)
    return qtensor

class TestQuantizedTensor:

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
        assert torch.allclose(quant_repr.to(torch.float32), get_backend().quantize(tensor, scale, offset, bitwidth))

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
        assert torch.allclose(dq_tensor, get_backend().quantize_dequantize(tensor, scale, offset, bitwidth))
       
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
        fp_output = get_backend().quantize_dequantize(tensor, scale, offset, bitwidth) * 2
        fp_output.backward(grad_in)
        assert torch.allclose(tensor_grad, tensor.grad)

    @pytest.mark.parametrize("scale, offset, bitwidth, signed",
                             [(torch.tensor(0.1), torch.tensor(-125.), 8, False)])
    def test_fallback_to_dequantize(self, scale, offset, bitwidth, signed):
        """
        Given: a QuantizedTensor qtensor
        When: Pass qtensor to a floating point operation
        Then: 1) Dequantize qtensor before calling the operation
              2) Output is not a QuantizedTensor
        """
        qdq = get_backend().quantize_dequantize
        input_shape = (16, 16)
        tensor = torch.randn(input_shape)
        tensor_qdq = qdq(tensor, scale, offset, bitwidth)
        qtensor = affine_quantize(tensor, scale, offset, bitwidth, signed)
        print(qtensor)
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
        