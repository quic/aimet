# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import os
import copy
import math
import pickle
import pytest
import torch
from torch import nn
from torch.optim import SGD, RMSprop, Adagrad, Adam, AdamW
from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
from aimet_torch.v2.quantization.affine import AffineQuantizerBase, Quantize, \
    QuantizeDequantize, Dequantize, GroupedBlockQuantizeDequantize
from aimet_torch.v2.quantization import affine
import aimet_torch.v2.quantization as Q


_PARAMETER_SHAPE = (100,)

def _initialize(q, symmetric):
    min = torch.empty(_PARAMETER_SHAPE)
    max = torch.empty(_PARAMETER_SHAPE)

    bw = q.bitwidth
    total_bins = 2**bw - 1
    negative_bins = math.ceil(total_bins / 2)
    positive_bins = math.floor(total_bins / 2)
    min.copy_(-1)
    max.copy_(1 * positive_bins/negative_bins) # max is one tick smaller

    if not symmetric:
        # Move the center to 1
        min.add_(1)
        max.add_(1)

    q.min = torch.nn.Parameter(min)
    q.max = torch.nn.Parameter(max)


def quantize(symmetric, initialized, bitwidth=8):
    encoding_analyzer = MinMaxEncodingAnalyzer(shape=_PARAMETER_SHAPE)
    quantize = Quantize(shape=_PARAMETER_SHAPE,
                        bitwidth=bitwidth,
                        symmetric=symmetric,
                        encoding_analyzer=encoding_analyzer)
    if initialized:
        _initialize(quantize, symmetric)
    return quantize


def quantize_dequantize(symmetric, initialized, bitwidth=8):
    encoding_analyzer = MinMaxEncodingAnalyzer(shape=_PARAMETER_SHAPE)
    quantize_dequantize = QuantizeDequantize(shape=_PARAMETER_SHAPE,
                                             bitwidth=bitwidth,
                                             symmetric=symmetric,
                                             encoding_analyzer=encoding_analyzer)
    if initialized:
        _initialize(quantize_dequantize, symmetric)
    return quantize_dequantize


@pytest.fixture()
def x():
    """
    Returns [
        [-2., -1.99, -1.98, ..., -1.01],
        [-1., -0.99, -0.98, ..., -0.01],
        [ 0.,  0.01,  0.02, ...,  0.99],
        [ 1.,  1.01,  1.02, ...,  1.99],
    ]
    """
    return torch.arange(-200, 200).view(4, 100) / 100


@pytest.fixture
def init_process_group():
    import torch.distributed as dist

    LOCAL_RANK = os.getenv('LOCAL_RANK', None)
    try:
        # Create process group of size 1
        dist.init_process_group(backend='gloo',
                                store=dist.HashStore(),
                                world_size=1,
                                rank=0)
        os.environ['LOCAL_RANK'] = '0'
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if LOCAL_RANK is not None:
            os.environ['LOCAL_RANK'] = LOCAL_RANK


@pytest.fixture
def deepspeed_zero3_config():
    return {
        "zero_optimization": {"stage": 3},
        "train_batch_size": 1,
    }


def minmax_to_scaleoffset(min, max, symmetric, bitwidth):
    total_bins = 2 ** bitwidth - 1
    scale = (max - min) / total_bins
    if symmetric:
        offset = torch.zeros_like(scale)
    else:
        offset = torch.round(min / scale)
    return scale, offset


@pytest.mark.parametrize('quantize', [
    quantize(symmetric=True, initialized=False),
    quantize(symmetric=True, initialized=True),
    quantize(symmetric=False, initialized=False),
    quantize(symmetric=False, initialized=True),
])
def test_quantize_compute_encodings(quantize: Quantize, x: torch.Tensor):
    """
    :param quantize: Quantize module
    :param x: Input tensor

    Given: During compute_encodings
    When:
      1. forward() invoked with input x
      2. Exit compute_encodings() context
    Then:
      1. forward() returns dynamic quantization output
      2. self.get_scale(), self.get_offset() == dynamic scale/offset of x
    """
    num_quant_bins = math.pow(2, quantize.bitwidth) - 1
    dynamic_min, dynamic_max =\
            quantize.encoding_analyzer.compute_dynamic_encodings(x, num_quant_bins, quantize.symmetric)
    dynamic_scale, dynamic_offset = minmax_to_scaleoffset(dynamic_min,
                                                          dynamic_max,
                                                          quantize.symmetric,
                                                          bitwidth=8)
    expected_x_int = Q.affine.quantize(x,
                                       dynamic_scale,
                                       dynamic_offset,
                                       quantize.bitwidth,
                                       quantize._signed)

    with quantize.compute_encodings():
        x_int = quantize(x)

    assert torch.allclose(x_int.quantized_repr(), expected_x_int.to(x_int.encoding.dtype))
    assert torch.allclose(x_int.encoding.scale, dynamic_scale)
    assert torch.allclose(x_int.encoding.offset, dynamic_offset)
    assert torch.allclose(quantize.min, dynamic_min)
    assert torch.allclose(quantize.max, dynamic_max)
    assert torch.allclose(quantize.get_scale(), dynamic_scale)
    assert torch.allclose(quantize.get_offset(), dynamic_offset)


@pytest.mark.parametrize('quantize_dequantize', [
    quantize_dequantize(symmetric=True, initialized=False),
    quantize_dequantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=False, initialized=False),
    quantize_dequantize(symmetric=False, initialized=True),
])
def test_qdq_compute_encodings(quantize_dequantize: QuantizeDequantize, x: torch.Tensor):
    """
    :param q: QuantizeDequantize module
    :param x: Input tensor

    Given: During compute_encodings
    When:
      1. forward() invoked with input x
      2. Exit compute_encodings() context
    Then:
      1. forward() returns dynamic quantization output
      2. self.get_scale(), self.get_offset() == dynamic scale/offset of x
    """
    num_quant_bins = math.pow(2, quantize_dequantize.bitwidth) - 1
    dynamic_min, dynamic_max =\
            quantize_dequantize.encoding_analyzer.compute_dynamic_encodings(x,
                                                                            num_quant_bins,
                                                                            quantize_dequantize.symmetric)
    dynamic_scale, dynamic_offset = minmax_to_scaleoffset(dynamic_min,
                                                          dynamic_max,
                                                          quantize_dequantize.symmetric,
                                                          bitwidth=8)
    expected_output = Q.affine.quantize_dequantize(x,
                                                   dynamic_scale,
                                                   dynamic_offset,
                                                   quantize_dequantize.bitwidth,
                                                   quantize_dequantize._signed)

    with quantize_dequantize.compute_encodings():
        output = quantize_dequantize(x)

    assert torch.allclose(output, expected_output)
    assert torch.allclose(quantize_dequantize.min, dynamic_min)
    assert torch.allclose(quantize_dequantize.max, dynamic_max)
    assert torch.allclose(quantize_dequantize.get_scale(), dynamic_scale)
    assert torch.allclose(quantize_dequantize.get_offset(), dynamic_offset)


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=False),
    quantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=True, initialized=False),
    quantize_dequantize(symmetric=True, initialized=True),
])
def test_compute_encodings_with_no_input(q: AffineQuantizerBase):
    """
    :param q: Quantize or QuantizeDequantize module

    Given: During compute_encodings
    When:
      1. forward() never invoked
      2. Exit compute_encodings() context
    Then: self.get_min(), self.get_max() doesn't change
    """

    original_min = q.get_min()
    if original_min is not None:
        original_min = original_min.clone().detach()

    original_max = q.get_max()
    if original_max is not None:
        original_max = original_max.clone().detach()

    with q.compute_encodings():
        pass

    if original_min is None:
        assert q.get_min() is original_min
    else:
        assert torch.equal(q.get_min(), original_min)

    if original_max is None:
        assert q.get_max() is None
    else:
        assert torch.equal(q.get_max(), original_max)


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=True, initialized=True),
    quantize(symmetric=False, initialized=True),
    quantize_dequantize(symmetric=False, initialized=True),
])
def test_backward_during_compute_encodings(q: AffineQuantizerBase, x: torch.Tensor):
    """
    :param q: Quantize or QuantizeDequantize module
    :param x: Input tensor

    Given: During compute_encodings
    When:
      1. forward() invoked
      2. backward() invoked
    Then: self.min.grad == self.max.grad == None
          (min/max are not trainable during compute_encodings)
    """
    x = x.clone().requires_grad_(True)

    with q.compute_encodings():
        if isinstance(q, Quantize):
            output = q(x)
        else:
            output = q(x)
        output.backward(torch.zeros_like(output))

    assert q.min.grad is None
    assert q.max.grad is None


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=False),
    quantize_dequantize(symmetric=True, initialized=False),
])
def test_compute_encodings_updates_parameters_upon_exit(q: AffineQuantizerBase, x: torch.Tensor):
    """
    :param q: Quantize or QuantizeDequantize module
    :param x: Input tensor

    Given: During compute_encodings
    When:
      1. forward() invoked
      2. Exit compute_encodings() context
    Then: min/max/scale/offset are updated when exiting compute_encodings
    """
    assert q.get_min() is None
    assert q.get_max() is None
    assert q.get_scale() is None
    assert q.get_offset() is None

    with q.compute_encodings():
        assert q.get_min() is None
        assert q.get_max() is None
        assert q.get_scale() is None
        assert q.get_offset() is None

        _ = q(x)

        assert q.get_min() is None
        assert q.get_max() is None
        assert q.get_scale() is None
        assert q.get_offset() is None


    assert q.get_min() is not None
    assert q.get_max() is not None
    assert q.get_scale() is not None
    assert q.get_offset() is not None


@pytest.mark.parametrize('quantize', [
    quantize(symmetric=True, initialized=True),
    quantize(symmetric=False, initialized=True),
])
def test_quantize_forward(quantize: Quantize, x: torch.Tensor):
    """
    :param q: Quantize module
    :param x: Input tensor

    Given:
      1. Outside compute_encodings
      2. Quantization parmeters are initialized
    When: forward() invoked
    Then: forward() returns parametric quantization output.
    """
    output = quantize(x)
    expected_output = Q.affine.quantize(x,
                                        quantize.get_scale(),
                                        quantize.get_offset(),
                                        quantize.bitwidth,
                                        quantize._signed)
    assert torch.allclose(output.quantized_repr(), expected_output.to(output.encoding.dtype))


@pytest.mark.parametrize('quantize_dequantize', [
    quantize_dequantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=False, initialized=True),
])
def test_qdq_forward(quantize_dequantize: QuantizeDequantize, x: torch.Tensor):
    """
    :param q: QuantizeDequantize module
    :param x: Input tensor

    Given:
      1. Outside compute_encodings
      2. Quantization parmeters are initialized
    When: forward() invoked
    Then: forward() returns parametric quantization output.
    """
    output = quantize_dequantize(x)
    expected_output = Q.affine.quantize_dequantize(x,
                                                   quantize_dequantize.get_scale(),
                                                   quantize_dequantize.get_offset(),
                                                   quantize_dequantize.bitwidth,
                                                   quantize_dequantize._signed)
    assert torch.allclose(output, expected_output)


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=True),
    quantize(symmetric=False, initialized=True),
    quantize_dequantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=False, initialized=True),
])
def test_backward(q: AffineQuantizerBase, x: torch.Tensor):
    """
    :param q: Quantize or QuantizeDequantize module
    :param x: Input tensor

    Given:
      1. Outside compute_encodings
      2. Quantization parmeters are initialized
    When:
      1. forward() invoked
      2. backward() invoked
    Then: self.min.grad and self.max.grad should be computed
    """
    if isinstance(q, Quantize):
        output = q(x)
    else:
        output = q(x)
    output.backward(torch.zeros_like(output))
    assert q.min.grad is not None
    assert q.max.grad is not None


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=True),
    quantize(symmetric=False, initialized=True),
    quantize_dequantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=False, initialized=True),
])
def test_backward_with_no_grad(q, x: torch.Tensor):
    """
    :param q: Quantize or QuantizeDequantize module
    :param x: Input tensor

    Given:
      1. Outside compute_encodings
      2. Quantization parmeters are initialized
    When:
      1. forward() invoked with torch.no_grad()
      2. backward() invoked
    Then: self.min.grad and self.max.grad should not be computed
    """
    x = x.clone().requires_grad_(True)
    with torch.no_grad():
        if isinstance(q, Quantize):
            output = q(x)
        else:
            output = q(x)
    output = output + x
    output.backward(torch.zeros_like(output))
    assert q.min.grad is None
    assert q.max.grad is None


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=False),
    quantize_dequantize(symmetric=True, initialized=False),
])
def test_uninitialized_quantize(q: AffineQuantizerBase, x: torch.Tensor):
    """
    :param q: Quantize or QuantizeDequantize module
    :param x: Input tensor

    Given:
      1. Outside compute_encodings
      2. Quantization parameters not initialized yet
    When: forward() invoked
    Then: Throw runtime error
    """
    assert q.get_min() is None
    assert q.get_max() is None
    assert q.get_scale() is None
    assert q.get_offset() is None

    with pytest.raises(RuntimeError):
        _ = q(x)


def _test_symmetric_invariants(q):
    """
    symmetric invaraints:
      1. min = scale * offest
      2. max = scale * -(offset + 1)
      3. scale = range / total_bins
         where range = max(-min * total_bins/negative_bins,
                            max * total_bins/positive_bins)
               total_bins = 2**bw - 1
               negative_bins = 2 ** (bw - 1)
               positive_bins = negative_bins - 1
      4. offset = 0
      5. offset is fixed (offset.requires_grad = False)
    """
    min = q.get_min()
    max = q.get_max()
    scale = q.get_scale()
    offset = q.get_offset()
    bw = q.bitwidth

    total_bins = 2**bw - 1
    positive_bins = math.floor(total_bins / 2)
    negative_bins = math.ceil(total_bins / 2)

    # min == scale * offset
    assert torch.allclose(min, - scale * negative_bins,
                          rtol=1e-3, atol=scale.abs().max().item() * 1e-5)

    # max == scale * -(offset+1)
    assert torch.allclose(max, scale * positive_bins,
                          rtol=1e-3, atol=scale.abs().max().item() * 1e-5)

    range = torch.maximum(-min * total_bins/negative_bins,
                          max * total_bins/positive_bins)
    assert torch.allclose(scale, range / total_bins,
                          rtol=1e-3, atol=scale.abs().max().item() * 1e-5)

    # offset == -1 * 2 ** (bw -1)
    assert torch.equal(offset, torch.zeros_like(offset))
    # offset is fixed in symmetric quantizer
    assert not offset.requires_grad


def _test_asymmetric_invariants(q):
    """
    asymmetric invaraints:
      1. min = scale * offest
      2. max = min + (2**bw - 1)
      3. scale = (max - min) / (2**bw - 1)
      4. offset = round(min / scale)
      5. offset is trainable (offset.requires_grad = True)
    """
    min = q.get_min()
    max = q.get_max()
    scale = q.get_scale()
    offset = q.get_offset()
    bw = q.bitwidth

    # min == scale * offset
    assert torch.allclose(min, scale * offset,
                          rtol=1e-3, atol=min.abs().max().item() * 1e-5)

    # max == min + scale * (2**bw - 1)
    assert torch.allclose(max, min + scale * (2**bw - 1),
                          rtol=1e-3, atol=max.abs().max().item() * 1e-5)

    # scale == (max - min) / (2**bw - 1)
    assert torch.allclose(scale, (max - min) / (2**bw - 1),
                          rtol=1e-3, atol=scale.abs().max().item() * 1e-5)

    # offsets == round(min / scale)
    assert torch.equal(torch.round(min/scale), offset)
    # offset is learned in asymmetric quantizer
    assert offset.requires_grad


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=False),
    quantize_dequantize(symmetric=True, initialized=False),
])
def test_symmetric_invariants(q, x: torch.Tensor):
    """
    Given: Symmetric quantizer
    When: Quantization parameters initialized with compute_encodings
    Then: Should satisfy all the symmetric quantization invariants
    """
    with q.compute_encodings():
        _ = q(x)

    _test_symmetric_invariants(q)


@pytest.mark.parametrize("optim_cls", [SGD, RMSprop, Adagrad, Adam, AdamW])
@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=True, initialized=True),
])
def test_symmetric_learning(q, x, optim_cls):
    """
    Given:
      1. Symmetric quantizer
      2. Quantization parameters are initialized
    When:
      1. forward() invoked
      2. backward() invoked
      3. optimizer.step() invoked
    Then: Should satisfy all the symmetric quantization invariants
    """

    original_min = q.get_min().clone().detach()
    original_max = q.get_max().clone().detach()
    original_scale = q.get_scale().clone().detach()
    original_offset = q.get_offset().clone().detach()

    optimizer = optim_cls(q.parameters(), lr=1.0)

    for _ in range(10):
        if isinstance(q, Quantize):
            output = q(x)
        else:
            output = q(x)
        output.backward(torch.randn_like(output))
        optimizer.step()
        _test_symmetric_invariants(q)

    assert not torch.equal(q.get_min(), original_min)
    assert not torch.equal(q.get_max(), original_max)
    assert not torch.equal(q.get_scale(), original_scale)
    assert torch.equal(q.get_offset(), original_offset)


@pytest.mark.parametrize('q', [
    quantize(symmetric=False, initialized=False),
    quantize_dequantize(symmetric=False, initialized=False),
])
def test_asymmetric_invariants(q: AffineQuantizerBase, x: torch.Tensor):
    """
    Given: Asymmetric quantizer
    When: Quantization parameters initialized with compute_encodings
    Then: Should satisfy all the symmetric quantization invariants
    """
    with q.compute_encodings():
        _ = q(x)

    _test_asymmetric_invariants(q)


@pytest.mark.parametrize("optim_cls", [SGD, RMSprop, Adagrad, Adam, AdamW])
@pytest.mark.parametrize('q', [
    quantize(symmetric=False, initialized=True),
    quantize_dequantize(symmetric=False, initialized=True),
])
def test_asymmetric_learning(q, x, optim_cls):
    """
    Given:
      1. Asymmetric quantizer
      2. Quantization parameters are initialized
    When:
      1. forward() invoked
      2. backward() invoked
      3. optimizer.step() invoked
    Then: Should satisfy all the asymmetric quantization invariants
    """
    original_min = q.get_min().clone().detach()
    original_max = q.get_max().clone().detach()
    original_scale = q.get_scale().clone().detach()
    original_offset = q.get_offset().clone().detach()

    optimizer = optim_cls(q.parameters(), lr=1.0)

    for _ in range(10):
        if isinstance(q, Quantize):
            output = q(x)
        else:
            output = q(x)
        output.backward(torch.randn_like(output))
        optimizer.step()
        _test_asymmetric_invariants(q)

    assert not torch.equal(q.get_min(), original_min)
    assert not torch.equal(q.get_max(), original_max)
    assert not torch.equal(q.get_scale(), original_scale)
    assert not torch.equal(q.get_offset(), original_offset)


def test_invalid_encoding_analyzer():
    """
    When: Instantiate a quantizer with an encoding analyzer of unmatching shape
    Then: Throw runtime error
    """
    dummy_input = torch.randn((30, 10, 11))
    param_shape = (10, 11)

    encoding_shape = (12,)
    with pytest.raises(RuntimeError):
        _ = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))

    encoding_shape = (10, 11)
    qdq = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))
    with qdq.compute_encodings():
        _ = qdq(dummy_input)

    encoding_shape = (11,)
    qdq = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))
    with qdq.compute_encodings():
        _ = qdq(dummy_input)

    encoding_shape = (10, 1)
    qdq = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))
    with qdq.compute_encodings():
        _ = qdq(dummy_input)

    encoding_shape = 11
    qdq = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))
    with qdq.compute_encodings():
        _ = qdq(dummy_input)

    encoding_shape = 1
    qdq = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))
    with qdq.compute_encodings():
        _ = qdq(dummy_input)


@torch.no_grad()
@pytest.mark.cuda
def test_is_initialized():
    """
    When: Instantiate a quantizer object
    Then:
      1) All the parameters readily exist as nn.Parameters (not as None or nn.UninitializedParameters)
      2) quantizer.is_initialized() returns False
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    assert isinstance(qdq.min, nn.Parameter) and not isinstance(qdq.min, nn.UninitializedParameter)
    assert isinstance(qdq.max, nn.Parameter) and not isinstance(qdq.max, nn.UninitializedParameter)
    assert not qdq.is_initialized()

    qdq.to(device="cuda", dtype=torch.float16)
    assert not qdq.is_initialized()

    """
    When: Update the parameters using in-place operation
    Then: is_initialized() returns True
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.min.copy_(torch.zeros(10))
    assert not qdq.is_initialized() # False; max is not initialized yet
    qdq.max.add_(3)
    assert qdq.is_initialized()

    """
    When: Update the parameters with assignment statement
    Then: is_initialized() returns True
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.min = nn.Parameter(-torch.ones(10) * 2)
    assert not qdq.is_initialized() # False; max is not initialized yet
    qdq.max = nn.Parameter(torch.ones(10) * 2)
    assert qdq.is_initialized()

    """
    When: Update the parameters with compute_encodings()
    Then: is_initialized() returns True
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    with qdq.compute_encodings():
        _ = qdq(torch.arange(-5, 5, dtype=torch.float))
    assert qdq.is_initialized()

    """
    When: Invoke load_state_dict() with a state dict that contains all parameters
    Then: quantizer.is_initialized() returns True
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.load_state_dict({'min': -torch.ones(10), 'max': torch.ones(10)})
    assert qdq.is_initialized()

    """
    When: Invoke load_state_dict with insufficient parameters
    Then: quantizer.is_initialized() returns False
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.load_state_dict({'min': -torch.ones(10)}, strict=False)
    assert not qdq.is_initialized() # False; max is not initialized yet
    qdq.load_state_dict({'max': torch.ones(10)}, strict=False)
    assert qdq.is_initialized()

    """
    When: Invoke load_state_dict() with a state dict that contains uninitialized parameters
    Then: quantizer.is_initialized() returns False
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    uninitialized_state_dict = qdq.state_dict()
    qdq.load_state_dict(uninitialized_state_dict)
    assert not qdq.is_initialized()

    qdq.min.mul_(1.)
    partially_initialized_state_dict = qdq.state_dict()
    qdq.load_state_dict(partially_initialized_state_dict)
    assert not qdq.is_initialized()

    qdq.max.mul_(1.)
    fully_initialized_state_dict = qdq.state_dict()
    qdq.load_state_dict(fully_initialized_state_dict)
    assert qdq.is_initialized()

    """
    When: Create a deepcopy of quantizer
    Then: quantizer.is_initialized() flag should be preserved
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq = copy.deepcopy(qdq)
    assert not qdq.is_initialized()

    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.load_state_dict({'min': -torch.ones(10), 'max': torch.ones(10)})
    qdq = copy.deepcopy(qdq)
    assert qdq.is_initialized()

    """
    When: Pickle and unpickle quantizer
    Then: quantizer.is_initialized() flag should be preserved
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    res = pickle.dumps(qdq)
    qdq = pickle.loads(res)
    assert not qdq.is_initialized()

    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.load_state_dict({'min': -torch.ones(10), 'max': torch.ones(10)})
    res = pickle.dumps(qdq)
    qdq = pickle.loads(res)
    assert qdq.is_initialized()


@pytest.mark.cuda
def test_is_initialized_with_deepspeed_zero3(init_process_group, deepspeed_zero3_config):
    import deepspeed as ds

    """
    When: Partition a quantizer with deepspeed zero 3
    Then: quantizer.is_initialized() flag should be preserved after pertitioning
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module
    assert not qdq_zero3.is_initialized()

    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.set_range(-1, 1)
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module
    assert qdq_zero3.is_initialized()

    """
    When: Gather the partitioned quantization parameters in read-only mode
    Then: quantizer.is_initialized() flag should be preserved during/after gathering
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module

    with ds.zero.GatheredParameters(qdq_zero3.parameters()):
        assert not qdq_zero3.is_initialized()
    assert not qdq_zero3.is_initialized()

    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.set_range(-1, 1)
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module

    with ds.zero.GatheredParameters(qdq_zero3.parameters()):
        assert qdq_zero3.is_initialized()
    assert qdq_zero3.is_initialized()

    """
    When: Modify the partitioned quantization parameters
    Then: quantizer.is_initialized() returns True
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module
    with ds.zero.GatheredParameters(qdq_zero3.parameters(), modifier_rank=0):
        qdq_zero3.set_range(-1, 1)
        assert qdq_zero3.is_initialized()
    assert qdq_zero3.is_initialized()

    # TODO (kyunggeu): Support the below use case
    # qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    # engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    # qdq_zero3 = engine.module
    # with ds.zero.GatheredParameters(qdq_zero3.parameters(), modifier_rank=0):
    #     with qdq_zero3.compute_encodings():
    #         _ = qdq_zero3(torch.arange(-5, 5, dtype=torch.float, device='cuda:0'))
    #     assert qdq_zero3.is_initialized()
    # assert qdq_zero3.is_initialized()

    """
    When: Gather the partitioned quantization parameters in writable mode but don't modify them
    Then: quantizer.is_initialized() flag should be preserved during/after gathering
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module

    with ds.zero.GatheredParameters(qdq_zero3.parameters(), modifier_rank=0):
        assert not qdq_zero3.is_initialized()
    assert not qdq_zero3.is_initialized()

    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.set_range(-1, 1)
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module

    with ds.zero.GatheredParameters(qdq_zero3.parameters(), modifier_rank=0):
        assert qdq_zero3.is_initialized()
    assert qdq_zero3.is_initialized()


@torch.no_grad()
@pytest.mark.parametrize('symmetric', [True, False])
def test_quantize_dequantize_then_quantize_and_dequantize_equality(x, symmetric):
    qdq = QuantizeDequantize((1,), 8, symmetric)
    q = Quantize((1,), 8, symmetric)
    dq = Dequantize()

    with qdq.compute_encodings(), q.compute_encodings():
        _ = qdq(x)
        _ = q(x)

    a = qdq(x)
    b = dq(q(x))
    assert torch.allclose(a, b)


@pytest.mark.cuda
@pytest.mark.parametrize("symmetric", [True, False])
def test_high_bitwidth(x, symmetric):
    """
    Given: QuantizeDequantize of bitwidth=16
    When: Run forward with input of dtype float16, float32, and bfloat16
    Then:
      1) All of them should produce outputs normally without any nan value.
      2) The output dtype should be the same as the input dtype
    """
    x = x.cuda()
    for param_dtype in (torch.float32, torch.bfloat16, torch.float16):
        for input_dtype in (torch.float16, torch.float32, torch.bfloat16):
            qdq = quantize_dequantize(symmetric=symmetric, initialized=True, bitwidth=16).to(param_dtype).cuda()
            out = qdq(x.to(input_dtype))
            assert not torch.any(out.isnan())
            assert out.dtype == input_dtype

    """
    Given: Quantize of bitwidth=16
    When: Run forward with input of dtype float32, and bfloat16
    Then:
      1) All of them should produce outputs normally without any nan value.
      2) The output dtype should be the same as the input dtype
    """
    for param_dtype in (torch.float32, torch.bfloat16, torch.float16):
        for input_dtype in (torch.float32, torch.bfloat16):
            q = quantize(symmetric=symmetric, initialized=True, bitwidth=16).to(param_dtype).cuda()
            out = q(x.to(input_dtype))
            assert not torch.any(out.isnan())
            assert out.dtype == input_dtype

    """
    Given: Quantize of bitwidth=16
    When: Run forward with input of dtype float16
    Then: Throw runtime error. float16 cannot represent the range [0, 2**16-1].
    """
    for param_dtype in (torch.float32, torch.bfloat16, torch.float16):
        for input_dtype in (torch.float16,):
            q = quantize(symmetric=symmetric, initialized=True, bitwidth=16).to(param_dtype).cuda()
            with pytest.raises(RuntimeError):
                out = q(x.to(input_dtype))

@pytest.mark.parametrize("q", (Quantize(_PARAMETER_SHAPE, 8, False),
                               QuantizeDequantize(_PARAMETER_SHAPE, 8, True)))
def test_freeze_encodings(x, q):
    with q.compute_encodings():
        q(x)

    q_min, q_max = q.min.detach().clone(), q.max.detach().clone()
    assert q.min.requires_grad
    assert q.max.requires_grad
    assert not q._is_encoding_frozen()

    q._freeze_encoding()
    assert q._is_encoding_frozen()
    """
    Given: Called quantizer.freeze_encoding()
    When: Inspect parameter requires_grad() attributes
    Then: requires_grad = False for all parameters
    """
    assert not q.min.requires_grad
    assert not q.max.requires_grad

    """
    When: Try to recompute encodings
    Then: Encodings do not change
    """
    with q.compute_encodings():
        q(x * 10)

    assert torch.equal(q_min, q.min)
    assert torch.equal(q_max, q.max)

def test_bq_compute_encodings_and_forward():
    torch.manual_seed(0)
    shape = (2, 2, 4)
    bq = QuantizeDequantize(shape=shape,
                            bitwidth=4,
                            symmetric=True,
                            block_size=(2, 4, 3))
    assert bq.encoding_analyzer.observer.shape == [2, 1, 2, 1, 4, 1]

    bq.eval()
    param_tensor = torch.randn(4, 8, 12)
    with bq.compute_encodings():
        _ = bq(param_tensor)

    out = bq(param_tensor)
    assert bq.get_min().shape == shape
    assert out.shape == param_tensor.shape

    qdq_out = affine.quantize_dequantize(param_tensor, bq.get_scale(), bq.get_offset(), bitwidth=bq.bitwidth,
                                         signed=bq.signed, block_size=bq.block_size)
    assert torch.equal(out, qdq_out)

@pytest.mark.parametrize('shape, block_sizes', [[(4, 1, 1), (1, 4, 4)],
                                                [(1, 4, 1), (4, 1, 4)],
                                                [(1, 1, 4), (4, 4, 1)]])
def test_bq_vs_per_channel_sanity(shape, block_sizes):
    torch.manual_seed(0)
    bq = QuantizeDequantize(shape=shape,
                            bitwidth=4,
                            symmetric=True,
                            block_size=block_sizes)

    pc = QuantizeDequantize(shape=shape,
                            bitwidth=4,
                            symmetric=True)

    bq.eval()
    pc.eval()
    param_tensor = torch.randn(4, 4, 4)
    with bq.compute_encodings():
        _ = bq(param_tensor)

    with pc.compute_encodings():
        _ = pc(param_tensor)

    assert torch.equal(bq(param_tensor), pc(param_tensor))

def test_quantized_tensor_with_block_size():
    torch.manual_seed(0)
    shape = (2, 2, 4)
    tensor = torch.randn(4, 8, 12)
    bq = Quantize(shape=shape,
                  bitwidth=4,
                  symmetric=True,
                  block_size=(2, 4, 3))
    with bq.compute_encodings():
        _ = bq(tensor)
    assert bq.get_encoding().block_size == bq.block_size
    q = bq(tensor)
    assert q.encoding.block_size == bq.block_size
    assert torch.equal(q.dequantize(), affine.dequantize(q, bq.get_scale(), bq.get_offset(), bq.block_size))

def test_gbbq_sanity():
    torch.manual_seed(0)
    tensor = torch.randn(8, 12)
    gbbq = GroupedBlockQuantizeDequantize(shape=(8, 4),
                                          bitwidth=4,
                                          symmetric=True,
                                          decompressed_bw=8,
                                          block_size=(-1, -1),
                                          block_grouping=(1, -1))
    pc = QuantizeDequantize(shape=(8, 1),
                            bitwidth=4,
                            symmetric=True)

    with gbbq.compute_encodings():
        _ = gbbq(tensor)

    with pc.compute_encodings():
        _ = pc(tensor)

    assert gbbq.get_scale().shape == (8, 4)

    # The largest scale for any given channel GBBQ should equal the scale for per channel
    assert torch.equal(torch.amax(gbbq.get_scale(), dim=1, keepdim=True), pc.get_scale())

    assert not torch.equal(gbbq(tensor), pc(tensor))

@pytest.mark.parametrize('bitwidth, decompressed_bw', [[4, 8], [4, 16], [4, 12], [3, 5], [5, 9], [6, 6]])
def test_gbbq_per_block_sanity(bitwidth, decompressed_bw):
    torch.manual_seed(0)
    tensor = torch.randn(4, 8, 12)
    gbbq = GroupedBlockQuantizeDequantize(shape=(2, 4, 6),
                                          bitwidth=bitwidth,
                                          symmetric=True,
                                          block_size=(2, 2, 2),
                                          decompressed_bw=decompressed_bw,
                                          block_grouping=(2, 2, 3))
    qdq = QuantizeDequantize(shape=(2, 4, 6),
                             bitwidth=bitwidth,
                             symmetric=True,
                             block_size=(2, 2, 2))
    with gbbq.compute_encodings():
        _ = gbbq(tensor)

    with qdq.compute_encodings():
        _ = qdq(tensor)

    for i in range(gbbq.shape[0] // gbbq.block_grouping[0]):
        for j in range(gbbq.shape[1] // gbbq.block_grouping[1]):
            for k in range(gbbq.shape[2] // gbbq.block_grouping[2]):
                gbbq_block_group = gbbq.get_scale()[i * gbbq.block_grouping[0]:(i + 1) * gbbq.block_grouping[0],
                                                    j * gbbq.block_grouping[1]:(j + 1) * gbbq.block_grouping[1],
                                                    k * gbbq.block_grouping[2]:(k + 1) * gbbq.block_grouping[2]]
                qdq_block_group = qdq.get_scale()[i * gbbq.block_grouping[0]:(i + 1) * gbbq.block_grouping[0],
                                                  j * gbbq.block_grouping[1]:(j + 1) * gbbq.block_grouping[1],
                                                  k * gbbq.block_grouping[2]:(k + 1) * gbbq.block_grouping[2]]
                max_scale = torch.max(qdq_block_group)
                compression_factor = 2 ** (decompressed_bw - bitwidth)
                gamma = max_scale / compression_factor
                int_rounded_scales = torch.maximum(torch.tensor([1.0]), torch.round(qdq_block_group / gamma))
                rounded_scales = int_rounded_scales * gamma
                assert torch.equal(rounded_scales, gbbq_block_group)

def test_gbbq_quantizer_default_grouping():
    torch.manual_seed(0)
    tensor = torch.randn(4, 8, 12)
    gbbq_default_grouping = GroupedBlockQuantizeDequantize(shape=(2, 4, 6),
                                                           bitwidth=4,
                                                           symmetric=True,
                                                           block_size=(2, 2, 2),
                                                           decompressed_bw=8)
    gbbq_no_grouping = GroupedBlockQuantizeDequantize(shape=(2, 4, 6),
                                                      bitwidth=4,
                                                      symmetric=True,
                                                      block_size=(2, 2, 2),
                                                      decompressed_bw=8,
                                                      block_grouping=(1, 1, 1))
    with gbbq_default_grouping.compute_encodings():
        _ = gbbq_default_grouping(tensor)

    with gbbq_no_grouping.compute_encodings():
        _ = gbbq_no_grouping(tensor)

    assert torch.equal(gbbq_default_grouping.get_scale(), gbbq_no_grouping.get_scale())
    assert torch.equal(gbbq_default_grouping(tensor), gbbq_no_grouping(tensor))

@pytest.mark.parametrize('gbbq_shape, gbbq_decompressed_bw, gbbq_block_size, gbbq_block_grouping, qdq_shape,'
                         'qdq_block_size',
                         [[(2, 4, 6), 8, (2, 2, 2), (1, 1, 1), (2, 4, 6), (2, 2, 2)],
                          [(2, 4, 6), 4, (2, 2, 2), (-1, -1, -1), (1, 1, 1), None],
                          [(2, 8, 6), 4, (2, 1, 2), (-1, 1, -1), (1, 8, 1), None]])
def test_gbbq_equivalences(gbbq_shape, gbbq_decompressed_bw, gbbq_block_size, gbbq_block_grouping, qdq_shape,
                           qdq_block_size):
    # Test 1: GBBQ should be equal to BQ in the case when block_grouping is 1 for all dims.
    # Test 2: GBBQ should be equal to per tensor in the case when block_grouping is -1 for all dims and decompressed_bw
    #         is equal to bitwidth.
    # Test 3: GBBQ should be equal to per channel in the case when block_grouping is -1 for all dims except the channel
    #         dimension and decompressed_bw is equal to bitwidth.
    torch.manual_seed(0)
    tensor = torch.randn(4, 8, 12)
    gbbq = GroupedBlockQuantizeDequantize(shape=gbbq_shape,
                                          bitwidth=4,
                                          symmetric=True,
                                          block_size=gbbq_block_size,
                                          decompressed_bw=gbbq_decompressed_bw,
                                          block_grouping=gbbq_block_grouping)
    qdq = QuantizeDequantize(shape=qdq_shape,
                             bitwidth=4,
                             symmetric=True,
                             block_size=qdq_block_size)
    with gbbq.compute_encodings():
        _ = gbbq(tensor)

    with qdq.compute_encodings():
        _ = qdq(tensor)

    assert torch.equal(gbbq.get_scale(), qdq.get_scale().expand(gbbq.get_scale().shape))
    gbbq_out = gbbq(tensor)
    qdq_out = qdq(tensor)
    assert torch.equal(gbbq_out, qdq_out)

    if all(group_size == 1 for group_size in gbbq_block_grouping):
        grad = torch.randn(size=gbbq_out.shape)
        gbbq_out.backward(grad)
        qdq_out.backward(grad)

        assert torch.equal(gbbq.min.grad, qdq.min.grad)
        assert torch.equal(gbbq.max.grad, qdq.max.grad)

def test_invalid_gbbq_settings():
    with pytest.raises(RuntimeError):
        _ = GroupedBlockQuantizeDequantize(shape=(2, 4, 6),
                                           bitwidth=4,
                                           symmetric=False,
                                           decompressed_bw=8)
    with pytest.raises(RuntimeError):
        _ = GroupedBlockQuantizeDequantize(shape=(2, 4, 6),
                                           bitwidth=4,
                                           symmetric=True,
                                           decompressed_bw=8,
                                           block_grouping=(-1, -1, -1, -1))

    with pytest.raises(RuntimeError):
        _ = GroupedBlockQuantizeDequantize(shape=(2, 4, 6),
                                           bitwidth=4,
                                           symmetric=True,
                                           decompressed_bw=3)
