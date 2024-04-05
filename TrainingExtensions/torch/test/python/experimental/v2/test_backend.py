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
import random
import torch
import pytest
from collections import namedtuple
from aimet_torch.v2.quantization.affine.backends import torch_builtins
from aimet_torch.v2.utils import ste_round

VectorSetForTest = namedtuple("VectorSetForTest", ["tensor", "tensor_q", "tensor_qdq", "mask", "delta", "offset", "bitwidth"])

bfloat16_compat_per_tensor_4b_test_set = VectorSetForTest(
    tensor=torch.tensor([
            [-1004.0, -1000.0, -15.375, -11.25, -3.25, 0.25, 500.0],
            [-1.375, -0.75, -0.125, 0, 1.125, 3, 10]
        ]),
    tensor_q=torch.tensor([
            [0, 0, 0, 0, 0, 5, 15],
            [2, 3, 5, 5, 7, 11, 15]
        ]),
    tensor_qdq=torch.tensor([
            [-2.5, -2.5, -2.5, -2.5, -2.5, 0.0, 5.0],
            [-1.5, -1.0, 0.0, 0.0, 1.0, 3.0, 5.0]
        ]),
    mask=torch.tensor([
        [1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1]
    ],  dtype=torch.bool),
    delta=torch.tensor([0.5]),
    offset=torch.tensor([-5]),
    bitwidth=4
)

bfloat16_compat_per_tensor_8b_test_set = VectorSetForTest(
    tensor=torch.tensor([
            [-1004.0, -1000.0, -15.375, -11.25, -3.25, 0.25, 500.0],
            [-1.375, -0.75, -0.125, 0, 1.125, 3, 10]
        ]),
    tensor_q=torch.tensor([
            [0, 0, 102, 111, 127, 133, 255],
            [130, 131, 133, 133, 135, 139, 153]
        ]),
    tensor_qdq=torch.tensor([
            [-66.5, -66.5, -15.5, -11, -3.0, 0, 61],
            [-1.5, -1, 0, 0, 1, 3, 10.0],
        ]),
    mask=torch.tensor([
        [1, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=torch.bool),
    delta=torch.tensor([0.5]),
    offset=torch.tensor([-133]),
    bitwidth=8
)

per_tensor_4b_test_set = VectorSetForTest(
    tensor=torch.tensor([
            [-1005.8, -1000.1, -15.4, -11.24, -3.3, 0.2, 500.4],
            [-1.3, -0.76, -0.11, 0, 1.01, 3, 10.4]
            ]),
    tensor_q=torch.tensor([
            [0, 0, 0, 0, 0, 5, 15],
            [2, 3, 5, 5, 7, 11, 15]
        ]),
    tensor_qdq=torch.tensor([
            [-2.5, -2.5, -2.5, -2.5, -2.5, 0.0, 5.0],
            [-1.5, -1.0, 0.0, 0.0, 1.0, 3.0, 5.0]
        ]),
    mask=torch.tensor([
        [1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1]
    ],  dtype=torch.bool),
    delta=torch.tensor([0.5]),
    offset=torch.tensor([-5]),
    bitwidth=4
)

per_tensor_8b_test_set = VectorSetForTest(
    tensor=torch.tensor([
            [-1005.8, -1000.1, -15.4, -11.24, -3.3, 0.2, 500.4],
            [-1.3, -0.76, -0.11, 0, 1.01, 3, 10.4]
            ]),
    tensor_q=torch.tensor([
            [0, 0, 102, 111, 126, 133, 255],
            [130, 131, 133, 133, 135, 139, 154]
        ]),
    tensor_qdq=torch.tensor([
            [-66.5, -66.5, -15.5, -11, -3.5, 0, 61],
            [-1.5, -1, 0, 0, 1, 3, 10.5],
        ]),
    mask=torch.tensor([
        [1, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=torch.bool),
    delta=torch.tensor([0.5]),
    offset=torch.tensor([-133]),
    bitwidth=8
)

per_channel_4b_test_set = VectorSetForTest(
    tensor=torch.tensor([
            [-1005.8, -1000.1, -15.4, -11.24, -3.3, 0.2, 500.4],
            [-1.3, -0.76, -0.11, 0, 1.01, 3, 10.4]
            ]),
    tensor_q=torch.tensor([
            [0, 0, 0, 0, 6, 13, 15],
            [0, 0, 5, 7, 15, 15, 15]
        ]),
    tensor_qdq=torch.tensor([
            [-6.5, -6.5, -6.5, -6.5, -3.5, 0, 1.0],
            [-0.4375, -0.4375, -0.125, 0.0, 0.5, 0.5, 0.5],
        ]),
    mask=torch.tensor([
        [1, 1, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 1, 1]
    ], dtype=torch.bool),
    delta=torch.tensor([[0.5], [0.0625]]),
    offset=torch.tensor([[-13], [-7]]),
    bitwidth=4
)

per_channel_8b_test_set = VectorSetForTest(
    tensor=torch.tensor([
            [-1005.8, -1000.1, -15.4, -11.24, -3.3, 0.2, 500.4],
            [-1.3, -0.76, -0.11, 0, 1.01, 3, 10.4]
            ]),
    tensor_q=torch.tensor([
            [0, 0, 102, 111, 126, 133, 255],
            [106, 115, 125, 127, 143, 175, 255]
        ]),
    tensor_qdq=torch.tensor([
            [-66.5, -66.5, -15.5, -11, -3.5, 0, 61],
            [-1.3125, -0.75, -0.125, 0., 1.0, 3.0, 8.0],
        ]),
    mask=torch.tensor([
        [1, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1]
    ], dtype=torch.bool),
    delta=torch.tensor([[0.5], [0.0625]]),
    offset=torch.tensor([[-133], [-127]]),
    bitwidth=8
)

class AutogradQuantizationModule(torch.nn.Module):
    def __init__(self, scale, offset, bitwidth):
        super().__init__()
        self.bitwidth = bitwidth
        self.scale = torch.nn.Parameter(scale.clone())
        self.offset = torch.nn.Parameter(offset.clone())

    def forward(self, x):
        return torch.clamp(
            ste_round(x / self.scale) - ste_round(self.offset),
            0, 2 ** self.bitwidth - 1
        )

class AutogradDequantizationModule(torch.nn.Module):
    def __init__(self, scale, offset, bitwidth):
        super().__init__()
        self.bitwidth = bitwidth
        self.scale = torch.nn.Parameter(scale.clone())
        self.offset = torch.nn.Parameter(offset.clone())

    def forward(self, x):
        return (x + ste_round(self.offset)) * self.scale

class AutogradQuantDequantModule(torch.nn.Module):
    def __init__(self, scale, offset, bitwidth):
        super().__init__()
        self.bitwidth = bitwidth
        self.scale = torch.nn.Parameter(scale.clone())
        self.offset = torch.nn.Parameter(offset.clone())

    def forward(self, x):
        x_q = torch.clamp(
            ste_round(x / self.scale) - ste_round(self.offset),
            0, 2 ** self.bitwidth - 1
        )
        x_dq = (x_q + ste_round(self.offset)) * self.scale
        return x_dq

def copy_test_set(test_set: namedtuple, device: torch.device = torch.device("cpu"),
                  dtype: torch.dtype = torch.float32):
    new_test_set = VectorSetForTest(
        tensor=test_set.tensor.clone().detach().to(device, dtype),
        tensor_q=test_set.tensor_q.clone().detach().to(device, dtype),
        tensor_qdq=test_set.tensor_qdq.clone().detach().to(device, dtype),
        mask=test_set.mask.clone().to(device),
        delta=test_set.delta.clone().detach().to(device, dtype),
        offset=test_set.offset.clone().detach().to(device, dtype),
        bitwidth=test_set.bitwidth
    )
    return new_test_set

def get_round_safe_quantizable_tensor(size: tuple, scale: torch.Tensor, bitwidth: int):
    """
    Returns round-safe quantizable random tensor by forcing
    fractional part of tensor divided by scale not to be near 0.5
    """
    return scale.cpu() * (torch.randint(0, 2 ** bitwidth - 1, size).to(torch.float32) \
        + torch.rand(size) * 0.8 - 0.4)

def get_random_quantized_tensor(size: tuple, bitwidth: int):
    return torch.randint(0, 2 ** bitwidth, size).to(torch.float32)

@pytest.fixture(autouse=True)
def set_seed():
    random.seed(19521)
    torch.random.manual_seed(19521)

@pytest.fixture(scope='session')
def offset():
    return torch.randint(-5, 5, []).to(torch.float32)

@pytest.mark.parametrize('backend_module', [torch_builtins])
class TestQuantizationBackends:
    def _test_quantization_backend(self, backend_module, random_tensor, scale, offset, bitwidth):
        expected_quantized_tensor = torch.clamp(torch.round(random_tensor / scale) - torch.round(offset), 0, 2 ** bitwidth - 1)
        quantized_tensor = backend_module.quantize(random_tensor, scale, offset, bitwidth)
        assert torch.allclose(quantized_tensor, expected_quantized_tensor)

        dequantized_tensor = backend_module.dequantize(expected_quantized_tensor, scale, offset)
        expected_dequantized_tensor = (expected_quantized_tensor + torch.round(offset)) * scale
        assert torch.allclose(dequantized_tensor, expected_dequantized_tensor)

        qdq_tensor = backend_module.quantize_dequantize(random_tensor, scale, offset, bitwidth)
        expected_qdq_tensor = (expected_quantized_tensor + torch.round(offset)) * scale
        assert torch.allclose(qdq_tensor, expected_qdq_tensor)

    @pytest.mark.parametrize('scale_shape', [(4), (2,), (3, 1), (2, 1, 1)])
    @pytest.mark.parametrize('bitwidth', [8])
    def test_quantize_using_not_broadcastable_scale(self, backend_module, offset, scale_shape, bitwidth):
        # Add small value to scale to make scale not equal to 0
        scale = torch.rand(scale_shape)
        scale[scale == 0.0] = 0.1
        random_tensor = torch.randn(2, 3, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)

        with pytest.raises(RuntimeError):
            backend_module.quantize(random_tensor, scale, offset, bitwidth)

        with pytest.raises(RuntimeError):
            backend_module.dequantize(random_quantized_tensor, scale, offset)

        with pytest.raises(RuntimeError):
            backend_module.quantize_dequantize(random_tensor, scale, offset, bitwidth)

    @pytest.mark.parametrize('bitwidth', [8])
    @pytest.mark.parametrize('scale_dtype, input_dtype', [(torch.float32, torch.float16), (torch.float16, torch.float32)])
    def test_quantize_using_invalid_dtype(self, backend_module, offset, bitwidth, scale_dtype, input_dtype):
        scale = torch.tensor([0.2], dtype=scale_dtype)
        random_tensor = torch.randn(2, 3, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)
        random_tensor = random_tensor.to(input_dtype)
        random_quantized_tensor = random_quantized_tensor.to(input_dtype)

        with pytest.raises(RuntimeError):
            backend_module.quantize(random_tensor, scale, offset, bitwidth)

        with pytest.raises(RuntimeError):
            backend_module.dequantize(random_quantized_tensor, scale, offset)

        with pytest.raises(RuntimeError):
            backend_module.quantize_dequantize(random_tensor, scale, offset, bitwidth)

    @pytest.mark.parametrize('input_dtype, bitwidth', [(torch.bfloat16, 32), (torch.float16, 32), (torch.float32, 64)])
    def test_quantize_using_wider_quantization_bitwidth(self, backend_module, offset, bitwidth, input_dtype):
        scale = torch.tensor([0.2], dtype=torch.float32)
        random_tensor = torch.randn(2, 3, 4, 5)
        random_tensor = random_tensor.to(input_dtype)

        with pytest.raises(RuntimeError):
            backend_module.quantize(random_tensor, scale, offset, bitwidth)

        with pytest.raises(RuntimeError):
            backend_module.quantize_dequantize(random_tensor, scale, offset, bitwidth)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='cannot test as there\'s only cpu on this machine')
    @pytest.mark.parametrize('bitwidth', [8])
    @pytest.mark.parametrize(
        'input_device, scale_device, offset_device',
        [
            (torch.device('cuda'), torch.device('cpu'), torch.device('cpu')),
            (torch.device('cpu'), torch.device('cuda'), torch.device('cpu')),
            (torch.device('cpu'), torch.device('cpu'), torch.device('cuda')),
        ]
    )
    def test_quantize_using_parameters_on_different_device(self, backend_module, offset, bitwidth, input_device, scale_device, offset_device):
        scale = torch.tensor([0.2], dtype=torch.float32, device=scale_device)
        random_tensor = torch.randn(2, 3, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)
        random_tensor = random_tensor.to(input_device)
        random_quantized_tensor = random_quantized_tensor.to(input_device)
        offset = offset.to(offset_device)

        with pytest.raises(RuntimeError):
            backend_module.quantize(random_tensor, scale, offset, bitwidth)

        with pytest.raises(RuntimeError):
            backend_module.dequantize(random_quantized_tensor, scale, offset)

        with pytest.raises(RuntimeError):
            backend_module.quantize_dequantize(random_tensor, scale, offset, bitwidth)

    @pytest.mark.parametrize('bitwidth', [8])
    @pytest.mark.parametrize('memory_format', [torch.channels_last, torch.channels_last_3d])
    def test_quantize_using_non_contiguous_tensor(self, backend_module, offset, bitwidth, memory_format):
        scale = torch.tensor([0.2], dtype=torch.float32)
        random_tensor = get_round_safe_quantizable_tensor((2, 3, 4, 5), scale, bitwidth)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)

        # Rank 5 tensor is required to use channels_last_3d format
        if memory_format == torch.channels_last_3d:
            random_tensor = random_tensor[..., None]
            random_quantized_tensor = random_quantized_tensor[..., None]

        channel_last_random_tensor = random_tensor.to(memory_format=memory_format)
        channel_last_random_quantized_tensor = random_quantized_tensor.to(memory_format=memory_format)

        channel_last_quantized_tensor = backend_module.quantize(channel_last_random_tensor, scale, offset, bitwidth)
        assert channel_last_quantized_tensor.is_contiguous(memory_format=memory_format)
        
        channel_last_dequantized_tensor = backend_module.dequantize(channel_last_random_quantized_tensor, scale, offset)
        assert channel_last_dequantized_tensor.is_contiguous(memory_format=memory_format)
        
        channel_last_qdq_tensor = backend_module.quantize_dequantize(channel_last_random_tensor, scale, offset, bitwidth)
        assert channel_last_qdq_tensor.is_contiguous(memory_format=memory_format)

        expected_quantized_tensor = torch.clamp(torch.round(random_tensor / scale) - torch.round(offset), 0, 2 ** bitwidth - 1)
        assert torch.allclose(channel_last_quantized_tensor, expected_quantized_tensor)

        expected_dequantized_tensor = (random_quantized_tensor + torch.round(offset)) * scale
        assert torch.allclose(channel_last_dequantized_tensor, expected_dequantized_tensor)

        expected_qdq_tensor = (expected_quantized_tensor + torch.round(offset)) * scale
        assert torch.allclose(channel_last_qdq_tensor, expected_qdq_tensor)

    @pytest.mark.parametrize('bitwidth', [8])
    @pytest.mark.parametrize('scale_shape', [(5, 1, 1, 1, 1), (3, 1, 1)])
    def test_quantize_using_inversely_broadcastable_scale(self, backend_module, offset, scale_shape, bitwidth):
        # Add small value to scale to make scale not equal to 0
        scale = torch.rand(scale_shape)
        scale[scale == 0.0] = 0.1
        random_tensor = torch.randn(2, 1, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 1, 4, 5), bitwidth)

        with pytest.raises(RuntimeError):
            backend_module.quantize(random_tensor, scale, offset, bitwidth)

        with pytest.raises(RuntimeError):
            backend_module.dequantize(random_quantized_tensor, scale, offset)

        with pytest.raises(RuntimeError):
            backend_module.quantize_dequantize(random_tensor, scale, offset, bitwidth)

    @pytest.mark.parametrize('bitwidth', [8])
    @pytest.mark.parametrize('scale_requires_grad', [True, False])
    @pytest.mark.parametrize('offset_requires_grad', [True, False])
    @pytest.mark.parametrize('input_requires_grad', [True, False])
    def test_quantize_backward_pass(self, backend_module, offset, bitwidth, scale_requires_grad, offset_requires_grad, input_requires_grad):
        scale = torch.rand([])
        scale[scale == 0.0] = 0.1
        offset = offset.detach().clone()
        random_tensor = get_round_safe_quantizable_tensor((2, 3, 4, 5), scale, bitwidth)
        scale.requires_grad = scale_requires_grad
        offset.requires_grad = offset_requires_grad
        random_tensor.requires_grad = input_requires_grad

        qdq_tensor = backend_module.quantize_dequantize(random_tensor, scale, offset, bitwidth)
        loss = torch.sum((random_tensor - qdq_tensor) ** 2)
        if loss.requires_grad:
            loss.backward()

        if scale_requires_grad:
            assert scale.grad is not None
        else:
            assert scale.grad is None

        if offset_requires_grad:
            assert offset.grad is not None
        else:
            assert offset.grad is None

        if input_requires_grad:
            assert random_tensor.grad is not None
        else:
            assert random_tensor.grad is None

    @pytest.mark.parametrize("signed", (False, True))
    @pytest.mark.parametrize("test_set", (per_tensor_4b_test_set,
                                          per_tensor_8b_test_set,
                                          per_channel_4b_test_set,
                                          per_channel_8b_test_set))
    @pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
    @pytest.mark.skipif(not torch.cuda.is_available(), reason='cannot test as there\'s only cpu on this machine')
    def test_quantize_with_predefined_values(self, backend_module, test_set, dtype, signed):
        test_set = copy_test_set(test_set, device="cuda:0", dtype=dtype)
        test_set.tensor.requires_grad = True
        offset = test_set.offset if not signed else test_set.offset + 2 ** (test_set.bitwidth - 1)
        tensor_q = backend_module.quantize(test_set.tensor, test_set.delta, offset, test_set.bitwidth, signed)
        expected_tensor_q = test_set.tensor_q if not signed else test_set.tensor_q - 2 ** (test_set.bitwidth - 1)
        assert torch.all(tensor_q == expected_tensor_q)
        grad_in = torch.randn_like(test_set.tensor)
        tensor_q.backward(grad_in)
        assert torch.all(test_set.tensor.grad[test_set.mask] == 0)
        assert torch.allclose(test_set.tensor.grad[~test_set.mask], (grad_in / test_set.delta)[~test_set.mask])

    @pytest.mark.parametrize("test_set", (per_tensor_4b_test_set,
                                          per_tensor_8b_test_set,
                                          per_channel_4b_test_set,
                                          per_channel_8b_test_set))
    @pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
    @pytest.mark.skipif(not torch.cuda.is_available(), reason='cannot test as there\'s only cpu on this machine')
    def test_dequantize_with_predefined_values(self, backend_module, test_set, dtype):
        test_set = copy_test_set(per_tensor_8b_test_set, dtype=dtype, device="cuda")
        tensor_q = test_set.tensor_q
        tensor_qdq = backend_module.dequantize(tensor_q, test_set.delta, test_set.offset)
        assert torch.all(tensor_qdq == test_set.tensor_qdq)

    @pytest.mark.parametrize("signed", (False, True))
    @pytest.mark.parametrize("test_set", (per_tensor_4b_test_set,
                                          per_tensor_8b_test_set,
                                          per_channel_4b_test_set,
                                          per_channel_8b_test_set))
    @pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
    @pytest.mark.skipif(not torch.cuda.is_available(), reason='cannot test as there\'s only cpu on this machine')
    def test_qdq_with_predefined_values(self, backend_module, test_set, dtype, signed):
        test_set = copy_test_set(test_set, dtype=dtype, device="cuda")
        test_set.tensor.requires_grad = True
        offset = test_set.offset if not signed else test_set.offset + 2 ** (test_set.bitwidth - 1)
        tensor_qdq = backend_module.quantize_dequantize(test_set.tensor, test_set.delta, offset, test_set.bitwidth, signed)
        assert torch.allclose(tensor_qdq, test_set.tensor_qdq)
        grad_in = torch.randn_like(test_set.tensor)
        tensor_qdq.backward(grad_in)
        assert torch.all(test_set.tensor.grad[test_set.mask] == 0)
        assert torch.all(test_set.tensor.grad[~test_set.mask] == grad_in[~test_set.mask])

    @pytest.mark.parametrize("test_set", (bfloat16_compat_per_tensor_4b_test_set,
                                          bfloat16_compat_per_tensor_8b_test_set))
    @pytest.mark.skipif(not torch.cuda.is_available(), reason='cannot test as there\'s only cpu on this machine')
    def test_quantize_with_predefined_bfloat_values(self, backend_module, test_set):
        test_set = copy_test_set(test_set, device="cuda:0", dtype=torch.bfloat16)
        test_set.tensor.requires_grad = True
        tensor_q = backend_module.quantize(test_set.tensor, test_set.delta, test_set.offset, test_set.bitwidth)
        assert torch.all(tensor_q == test_set.tensor_q)
        grad_in = torch.randn_like(test_set.tensor)
        tensor_q.backward(grad_in)
        assert torch.all(test_set.tensor.grad[test_set.mask] == 0)
        assert torch.allclose(test_set.tensor.grad[~test_set.mask], (grad_in / test_set.delta)[~test_set.mask])

    @pytest.mark.parametrize("test_set", (bfloat16_compat_per_tensor_4b_test_set,
                                          bfloat16_compat_per_tensor_8b_test_set))
    @pytest.mark.skipif(not torch.cuda.is_available(), reason='cannot test as there\'s only cpu on this machine')
    def test_dequantize_with_predefined_bfloat_values(self, backend_module, test_set):
        test_set = copy_test_set(per_tensor_8b_test_set, dtype=torch.bfloat16, device="cuda")
        tensor_q = test_set.tensor_q
        tensor_qdq = backend_module.dequantize(tensor_q, test_set.delta, test_set.offset)
        assert torch.all(tensor_qdq == test_set.tensor_qdq)

    @pytest.mark.parametrize("test_set", (bfloat16_compat_per_tensor_4b_test_set,
                                          bfloat16_compat_per_tensor_8b_test_set))
    @pytest.mark.skipif(not torch.cuda.is_available(), reason='cannot test as there\'s only cpu on this machine')
    def test_qdq_with_predefined_bfloat_values(self, backend_module, test_set):
        test_set = copy_test_set(test_set, dtype=torch.bfloat16, device="cuda")
        test_set.tensor.requires_grad = True
        tensor_qdq = backend_module.quantize_dequantize(test_set.tensor, test_set.delta, test_set.offset, test_set.bitwidth)
        assert torch.allclose(tensor_qdq, test_set.tensor_qdq)
        grad_in = torch.randn_like(test_set.tensor)
        tensor_qdq.backward(grad_in)
        assert torch.all(test_set.tensor.grad[test_set.mask] == 0)
        assert torch.all(test_set.tensor.grad[~test_set.mask] == grad_in[~test_set.mask])

    @pytest.mark.parametrize('bitwidth', [8])
    def test_compare_quantize_gradients_with_autograd_results(self, backend_module, offset, bitwidth):
        scale = torch.rand([])
        scale[scale == 0.0] = 0.1
        offset = offset.detach().clone()
        random_tensor = get_round_safe_quantizable_tensor((2, 3, 4, 5), scale, bitwidth)
        random_tensor_for_autograd = random_tensor.detach().clone()

        scale.requires_grad = True
        offset.requires_grad = True
        random_tensor.requires_grad = True
        random_tensor_for_autograd.requires_grad = True
        
        autograd_based_module = AutogradQuantizationModule(scale, offset, bitwidth)
        expected_tensor_q = autograd_based_module(random_tensor_for_autograd)
        tensor_q = backend_module.quantize(random_tensor, scale, offset, bitwidth)
        assert torch.allclose(tensor_q, expected_tensor_q)

        grad_in = torch.randn_like(random_tensor)
        expected_tensor_q.backward(grad_in)
        tensor_q.backward(grad_in)

        expected_tensor_grad = random_tensor_for_autograd.grad
        expected_scale_grad = autograd_based_module.scale.grad
        expected_offset_grad = autograd_based_module.offset.grad

        assert torch.allclose(random_tensor.grad, expected_tensor_grad)
        assert torch.allclose(scale.grad, expected_scale_grad)
        assert torch.allclose(offset.grad, expected_offset_grad)

    @pytest.mark.parametrize('bitwidth', [8])
    def test_compare_dequantize_gradients_with_autograd_results(self, backend_module, offset, bitwidth):
        scale = torch.rand([])
        scale[scale == 0.0] = 0.1
        offset = offset.detach().clone()
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)
        random_quantized_tensor_for_autograd = random_quantized_tensor.detach().clone()

        scale.requires_grad = True
        offset.requires_grad = True
        random_quantized_tensor.requires_grad = True
        random_quantized_tensor_for_autograd.requires_grad = True
        
        autograd_based_module = AutogradDequantizationModule(scale, offset, bitwidth)
        expected_tensor_dq = autograd_based_module(random_quantized_tensor_for_autograd)
        tensor_dq = backend_module.dequantize(random_quantized_tensor, scale, offset)
        assert torch.allclose(tensor_dq, expected_tensor_dq)

        grad_in = torch.randn_like(random_quantized_tensor)
        expected_tensor_dq.backward(grad_in)
        tensor_dq.backward(grad_in)

        expected_tensor_grad = random_quantized_tensor_for_autograd.grad
        expected_scale_grad = autograd_based_module.scale.grad
        expected_offset_grad = autograd_based_module.offset.grad

        assert torch.allclose(random_quantized_tensor.grad, expected_tensor_grad)
        assert torch.allclose(scale.grad, expected_scale_grad)
        assert torch.allclose(offset.grad, expected_offset_grad)

    @pytest.mark.parametrize('bitwidth', [8])
    def test_compare_qdq_gradients_with_autograd_results(self, backend_module, offset, bitwidth):
        scale = torch.rand([])
        scale[scale == 0.0] = 0.1
        offset = offset.detach().clone()
        random_tensor = get_round_safe_quantizable_tensor((2, 3, 4, 5), scale, bitwidth)
        random_tensor_for_autograd = random_tensor.detach().clone()

        scale.requires_grad = True
        offset.requires_grad = True
        random_tensor.requires_grad = True
        random_tensor_for_autograd.requires_grad = True
        
        autograd_based_module = AutogradQuantDequantModule(scale, offset, bitwidth)
        expected_tensor_qdq = autograd_based_module(random_tensor_for_autograd)
        tensor_qdq = backend_module.quantize_dequantize(random_tensor, scale, offset, bitwidth)
        assert torch.allclose(tensor_qdq, expected_tensor_qdq)

        grad_in = torch.randn_like(random_tensor)
        expected_tensor_qdq.backward(grad_in)
        tensor_qdq.backward(grad_in)

        expected_tensor_grad = random_tensor_for_autograd.grad
        expected_scale_grad = autograd_based_module.scale.grad
        expected_offset_grad = autograd_based_module.offset.grad

        assert torch.allclose(random_tensor.grad, expected_tensor_grad, rtol=1e-3)
        assert torch.allclose(scale.grad, expected_scale_grad, rtol=1e-3)
        assert torch.allclose(offset.grad, expected_offset_grad, rtol=1e-3)
