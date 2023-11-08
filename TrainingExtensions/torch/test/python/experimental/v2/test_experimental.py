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

VectorSetForTest = namedtuple("VectorSetForTest", ["tensor", "tensor_q", "tensor_qdq", "mask", "delta", "offset", "bitwidth"])

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

per_tensor_16b_test_set = VectorSetForTest(
    tensor=torch.tensor([
            [-1005.8, -1000.1, -15.4, -11.24, -3.3, 0.2, 500.4],
            [-1.3, -0.76, -0.11, 0, 1.01, 3, 10.4]
            ]),
    tensor_q=torch.tensor([
            [0, 0, 0, 0, 0, 5, 1006],
            [2, 3, 5, 5, 7, 11, 26]
        ]),
    tensor_qdq=torch.tensor([
            [-2.5, -2.5, -2.5, -2.5, -2.5, 0.0, 500.5],
            [-1.5, -1. , 0.0, 0.0, 1.0, 3.0, 10.5]
        ]),
    mask=torch.tensor([
        [1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],  dtype=torch.bool),
    delta=torch.tensor([0.5]),
    offset=torch.tensor([-5]),
    bitwidth=16
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

per_channel_16b_test_set = VectorSetForTest(
    tensor=torch.tensor([
            [-1005.8, -1000.1, -15.4, -11.24, -3.3, 0.2, 500.4],
            [-1.3, -0.76, -0.11, 0, 1.01, 3, 100]
            ]),
    tensor_q=torch.tensor([
            [0, 0, 0, 0, 0, 5, 1006],
            [0, 0, 0, 0, 15, 47, 1599]
        ]),
    tensor_qdq=torch.tensor([
            [-2.5, -2.5, -2.5, -2.5, -2.5, 0.0, 500.5],
            [0.0625, 0.0625, 0.0625, 0.0625, 1.0, 3.0, 100.0]
        ]),
    mask=torch.tensor([
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0]
    ], dtype=torch.bool),
    delta=torch.tensor([[0.5], [0.0625]]),
    offset=torch.tensor([[-5], [1]]),
    bitwidth=16
)

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
    return scale * (torch.randint(0, 2**bitwidth - 1, size).to(torch.float32) \
        + torch.rand(size) * 0.8 - 0.4)

def get_random_quantized_tensor(size: tuple, bitwidth: int):
    return torch.randint(0, 2**bitwidth, size).to(torch.float32)

@pytest.fixture(autouse=True)
def set_seed():
    random.seed(19521)
    torch.random.manual_seed(19521)

@pytest.fixture(scope='session')
def offset():
    return torch.randint(-5, 5, []).to(torch.float32)

@pytest.mark.parametrize('backend_class', [])
class TestQuantizationBackends:
    def _test_quantization_backend(self, backend_class, random_tensor, random_quantized_tensor, scale, offset, bitwidth):
        tensor_after_quantize = torch.clamp(torch.round(random_tensor / scale) - torch.round(offset), 0, 2 ** bitwidth - 1)
        quantized_tensor = backend_class.quantize(random_tensor, scale, offset, bitwidth)
        assert torch.allclose(quantized_tensor, tensor_after_quantize)

        dequantized_tensor = backend_class.dequantize(random_quantized_tensor, scale, offset)
        tensor_after_dequantize = (random_quantized_tensor + torch.round(offset)) * scale
        assert torch.allclose(dequantized_tensor, tensor_after_dequantize)

        qdq_tensor = backend_class.quantize_dequantize(random_tensor, scale, offset, bitwidth)
        tensor_after_qdq = (tensor_after_quantize + torch.round(offset)) * scale
        assert torch.allclose(qdq_tensor, tensor_after_qdq)

    @pytest.mark.parametrize(
            'scale', 
            [torch.tensor(0.3, dtype=torch.float32), torch.tensor([0.3], dtype=torch.float32)]
    )
    @pytest.mark.parametrize('bitwidth', [8])
    def test_quantize_using_scale_with_single_element(self, backend_class, scale, offset, bitwidth):
        extracted_scale = scale if scale.ndim == 0 else scale[0]
        random_tensor = get_round_safe_quantizable_tensor((2, 3, 4, 5), scale, bitwidth)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)

        self._test_quantization_backend(backend_class, random_tensor, random_quantized_tensor, extracted_scale, offset, bitwidth)

    @pytest.mark.parametrize('scale_shape', [(4), (2,), (3, 1), (2, 1, 1)])
    @pytest.mark.parametrize('bitwidth', [8])
    def test_quantize_using_not_broadcastable_scale(self, backend_class, offset, scale_shape, bitwidth):
        # Add small value to scale to make scale not equal to 0
        scale = torch.rand(scale_shape) + 0.1
        random_tensor = torch.randn(2, 3, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)

        with pytest.raises(RuntimeError):
            backend_class.quantize(random_tensor, scale, offset, bitwidth)

        with pytest.raises(RuntimeError):
            backend_class.dequantize(random_quantized_tensor, scale, offset)

        with pytest.raises(RuntimeError):
            backend_class.quantize_dequantize(random_tensor, scale, offset, bitwidth)

    @pytest.mark.parametrize('scale_shape, pcq_axis', [((5,), 3), ((4, 1), 2), ((3, 1, 1), 1),])
    @pytest.mark.parametrize('bitwidth', [8])
    def test_quantize_using_broadcastable_scale(self, backend_class, scale_shape, offset, bitwidth, pcq_axis):
        scale = torch.rand(scale_shape) + 0.1
        random_tensor = get_round_safe_quantizable_tensor((2, 3, 4, 5), scale, bitwidth)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)
        broadcasted_scale_shape = tuple(dim if axis == pcq_axis else 1
                                       for (axis, dim) in enumerate(random_tensor.shape))
        broadcasted_scale = scale.view(broadcasted_scale_shape)

        self._test_quantization_backend(backend_class, random_tensor, random_quantized_tensor, broadcasted_scale, offset, bitwidth)

    @pytest.mark.parametrize('bitwidth', [8])
    @pytest.mark.parametrize('scale_dtype, input_dtype', [(torch.float32, torch.float16), (torch.float16, torch.float32)])
    def test_quantize_using_invalid_dtype(self, backend_class, offset, bitwidth, scale_dtype, input_dtype):
        scale = torch.tensor([0.2], dtype=scale_dtype)
        random_tensor = torch.randn(2, 3, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)
        random_tensor = random_tensor.to(input_dtype)
        random_quantized_tensor = random_quantized_tensor.to(input_dtype)

        with pytest.raises(RuntimeError):
            backend_class.quantize(random_tensor, scale, offset, bitwidth)

        with pytest.raises(RuntimeError):
            backend_class.dequantize(random_quantized_tensor, scale, offset)

        with pytest.raises(RuntimeError):
            backend_class.quantize_dequantize(random_tensor, scale, offset, bitwidth)

    @pytest.mark.parametrize('input_dtype, bitwidth', [(torch.float16, 32), (torch.float32, 64)])
    def test_quantize_using_wider_quantization_bitwidth(self, backend_class, offset, bitwidth, input_dtype):
        scale = torch.tensor([0.2], dtype=torch.float32)
        random_tensor = torch.randn(2, 3, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)
        random_tensor = random_tensor.to(input_dtype)
        random_quantized_tensor = random_quantized_tensor.to(input_dtype)

        with pytest.raises(RuntimeError):
            backend_class.quantize(random_tensor, scale, offset, bitwidth)

        with pytest.raises(RuntimeError):
            backend_class.quantize_dequantize(random_tensor, scale, offset, bitwidth)

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
    def test_quantize_using_parameters_on_different_device(self, backend_class, offset, bitwidth, input_device, scale_device, offset_device):
        scale = torch.tensor([0.2], dtype=torch.float32, device=scale_device)
        random_tensor = torch.randn(2, 3, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)
        random_tensor = random_tensor.to(input_device)
        random_quantized_tensor = random_quantized_tensor.to(input_device)
        offset = offset.to(offset_device)

        with pytest.raises(RuntimeError):
            backend_class.quantize(random_tensor, scale, offset, bitwidth)

        with pytest.raises(RuntimeError):
            backend_class.dequantize(random_quantized_tensor, scale, offset)

        with pytest.raises(RuntimeError):
            backend_class.quantize_dequantize(random_tensor, scale, offset, bitwidth)

    @pytest.mark.parametrize('bitwidth', [8])
    def test_quantize_using_input_in_channels_last_format(self, backend_class, offset, bitwidth):
        scale = torch.tensor([0.2], dtype=torch.float32)
        random_tensor = get_round_safe_quantizable_tensor((2, 3, 4, 5), scale, bitwidth)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)
        channel_last_random_tensor = random_tensor.to(memory_format=torch.channels_last)
        channel_last_random_quantized_tensor = random_quantized_tensor.to(memory_format=torch.channels_last)

        channel_last_quantized_tensor = backend_class.quantize(channel_last_random_tensor, scale, offset, bitwidth)
        assert channel_last_quantized_tensor.is_contiguous(memory_format=torch.channels_last)
        
        channel_last_dequantized_tensor = backend_class.dequantize(channel_last_random_quantized_tensor, scale, offset)
        assert channel_last_dequantized_tensor.is_contiguous(memory_format=torch.channels_last)
        
        channel_last_qdq_tensor = backend_class.quantize_dequantize(channel_last_random_tensor, scale, offset, bitwidth)
        assert channel_last_qdq_tensor.is_contiguous(memory_format=torch.channels_last)

        tensor_after_quantize = torch.clamp(torch.round(random_tensor / scale) - torch.round(offset), 0, 2 ** bitwidth - 1)
        assert torch.allclose(channel_last_quantized_tensor, tensor_after_quantize)

        tensor_after_dequantize = (random_quantized_tensor + torch.round(offset)) * scale
        assert torch.allclose(channel_last_dequantized_tensor, tensor_after_dequantize)

        tensor_after_qdq = (tensor_after_quantize + torch.round(offset)) * scale
        assert torch.allclose(channel_last_qdq_tensor, tensor_after_qdq)

    @pytest.mark.parametrize('bitwidth', [0])
    def test_quantize_using_invalid_bitwidth(self, backend_class, offset, bitwidth):
        scale = torch.tensor([0.2], dtype=torch.float32)
        random_tensor = torch.randn(2, 3, 4, 5)
        random_quantized_tensor = torch.zeros((2, 3, 4, 5), dtype=torch.float32)

        with pytest.raises(RuntimeError):
            backend_class.quantize(random_tensor, scale, offset, bitwidth)

        with pytest.raises(RuntimeError):
            backend_class.dequantize(random_quantized_tensor, scale, offset)

        with pytest.raises(RuntimeError):
            backend_class.quantize_dequantize(random_tensor, scale, offset, bitwidth)

    @pytest.mark.parametrize('bitwidth', [8])
    def test_quantize_using_scale_with_multiple_channels(self, backend_class, offset, bitwidth):
        scale_shape = (2, 1, 4, 1)
        # Add small value to scale to make scale not equal to 0
        scale = torch.rand(scale_shape) + 0.1
        random_tensor = torch.randn(2, 3, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)

        with pytest.raises(RuntimeError):
            backend_class.quantize(random_tensor, scale, offset, bitwidth)

        with pytest.raises(RuntimeError):
            backend_class.dequantize(random_quantized_tensor, scale, offset)

        with pytest.raises(RuntimeError):
            backend_class.quantize_dequantize(random_tensor, scale, offset, bitwidth)

    @pytest.mark.parametrize('bitwidth', [8])
    def test_quantize_using_inversely_broadcastable_scale(self, backend_class, offset, bitwidth):
        # Add small value to scale to make scale not equal to 0
        scale = torch.rand((5, 1, 1, 1, 1)) + 0.1
        random_tensor = torch.randn(2, 3, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), bitwidth)

        with pytest.raises(RuntimeError):
            backend_class.quantize(random_tensor, scale, offset, bitwidth)

        with pytest.raises(RuntimeError):
            backend_class.dequantize(random_quantized_tensor, scale, offset)

        with pytest.raises(RuntimeError):
            backend_class.quantize_dequantize(random_tensor, scale, offset, bitwidth)

    @pytest.mark.parametrize('bitwidth', [8])
    @pytest.mark.parametrize('scale_requires_grad', [True, False])
    @pytest.mark.parametrize('offset_requires_grad', [True, False])
    @pytest.mark.parametrize('input_requires_grad', [True, False])
    def test_quantize_backward_pass(self, backend_class, offset, bitwidth, scale_requires_grad, offset_requires_grad, input_requires_grad):
        scale = torch.rand([]) + 0.1
        random_tensor = get_round_safe_quantizable_tensor((2, 3, 4, 5), scale, bitwidth)
        scale.requires_grad = scale_requires_grad
        offset.requires_grad = offset_requires_grad
        random_tensor.requires_grad = input_requires_grad

        qdq_tensor = backend_class.quantize_dequantize(random_tensor, scale, offset, bitwidth)
        loss = (random_tensor - qdq_tensor) ** 2
        if loss.requires_grad:
            loss.backward()

        if scale_requires_grad:
            assert any(scale.grad != 0.0)

        if offset_requires_grad:
            assert any(offset.grad != 0.0)

        if input_requires_grad:
            assert any(random_tensor.grad != 0.0)

    @pytest.mark.parametrize("test_set", (per_tensor_8b_test_set,
                                          per_tensor_16b_test_set,
                                          per_channel_4b_test_set,
                                          per_channel_8b_test_set,
                                          per_channel_16b_test_set))
    @pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
    def test_quantize_with_predefined_values(self, backend_class, test_set, dtype):
        test_set = copy_test_set(test_set, device="cuda:0", dtype=dtype)
        test_set.tensor.requires_grad = True
        tensor_q = backend_class.quantize(test_set.tensor, test_set.delta, test_set.offset, test_set.bitwidth)
        assert torch.all(tensor_q == test_set.tensor_q), f"expected: {test_set.tensor_q}\ngot: {tensor_q}"
        grad_in = torch.randn_like(test_set.tensor)
        tensor_q.backward(grad_in)
        assert torch.all(test_set.tensor.grad[test_set.mask] == 0)
        assert torch.all(test_set.tensor.grad[~test_set.mask] == grad_in[~test_set.mask])

    @pytest.mark.parametrize("test_set", (per_tensor_8b_test_set,
                                          per_tensor_16b_test_set,
                                          per_channel_4b_test_set,
                                          per_channel_8b_test_set,
                                          per_channel_16b_test_set))
    @pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
    def test_dequantize_with_predefined_values(self, backend_class, test_set, dtype):
        test_set = copy_test_set(per_tensor_8b_test_set, dtype=dtype, device="cuda")
        tensor_q = test_set.tensor_q
        tensor_qdq = backend_class.dequantize(tensor_q, test_set.delta, test_set.offset)
        assert torch.all(tensor_qdq == test_set.tensor_qdq)

    @pytest.mark.parametrize("test_set", (per_tensor_8b_test_set,
                                          per_tensor_16b_test_set,
                                          per_channel_4b_test_set,
                                          per_channel_8b_test_set,
                                          per_channel_16b_test_set))
    @pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
    def test_qdq_with_predefined_values(self, backend_class, test_set, dtype):
        test_set = copy_test_set(test_set, dtype=dtype, device="cuda")
        test_set.tensor.requires_grad = True
        tensor_qdq = backend_class.quantize_dequantize(test_set.tensor, test_set.delta, test_set.offset, test_set.bitwidth)
        assert torch.allclose(tensor_qdq, test_set.tensor_qdq), f"{tensor_qdq}, \n{test_set.tensor_qdq}"
        grad_in = torch.randn_like(test_set.tensor)
        tensor_qdq.backward(grad_in)
        assert torch.all(test_set.tensor.grad[test_set.mask] == 0)
        assert torch.all(test_set.tensor.grad[~test_set.mask] == grad_in[~test_set.mask])
