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
import copy
import math
import pickle
import pytest
import torch
from torch import nn
from torch.optim import SGD, RMSprop, Adagrad, Adam, AdamW
from aimet_torch.experimental.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
from aimet_torch.experimental.v2.quantization.quantizers.affine import AffineQuantizerBase, Quantize, \
    QuantizeDequantize, Dequantize
from aimet_torch.experimental.v2.quantization.backends import get_backend


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


def quantize(symmetric, initialized):
    encoding_analyzer = MinMaxEncodingAnalyzer(shape=_PARAMETER_SHAPE)
    quantize = Quantize(shape=_PARAMETER_SHAPE,
                        bitwidth=8,
                        symmetric=symmetric,
                        encoding_analyzer=encoding_analyzer)
    if initialized:
        _initialize(quantize, symmetric)
    return quantize


def quantize_dequantize(symmetric, initialized):
    encoding_analyzer = MinMaxEncodingAnalyzer(shape=_PARAMETER_SHAPE)
    quantize_dequantize = QuantizeDequantize(shape=_PARAMETER_SHAPE,
                                             bitwidth=8,
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



def minmax_to_scaleoffset(min, max, symmetric, bitwidth):
    total_bins = 2 ** bitwidth - 1
    scale = (max - min) / total_bins
    if symmetric:
        positive_bins = total_bins // 2
        negative_bins = positive_bins + 1
        offset = torch.ones_like(scale) * -negative_bins
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
    dynamic_min, dynamic_max =\
            quantize.encoding_analyzer.compute_dynamic_encodings(x, quantize.bitwidth, quantize.symmetric)
    dynamic_scale, dynamic_offset = minmax_to_scaleoffset(dynamic_min,
                                                          dynamic_max,
                                                          quantize.symmetric,
                                                          bitwidth=8)
    expected_x_int = get_backend().quantize(x,
                                            dynamic_scale,
                                            dynamic_offset,
                                            quantize.bitwidth)

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
    dynamic_min, dynamic_max =\
            quantize_dequantize.encoding_analyzer.compute_dynamic_encodings(x,
                                                                            quantize_dequantize.bitwidth,
                                                                            quantize_dequantize.symmetric)
    dynamic_scale, dynamic_offset = minmax_to_scaleoffset(dynamic_min,
                                                          dynamic_max,
                                                          quantize_dequantize.symmetric,
                                                          bitwidth=8)
    expected_output = get_backend().quantize_dequantize(x,
                                                        dynamic_scale,
                                                        dynamic_offset,
                                                        quantize_dequantize.bitwidth)

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
    expected_output = get_backend().quantize(x,
                                             quantize.get_scale(),
                                             quantize.get_offset(),
                                             quantize.bitwidth)
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
    expected_output = get_backend().quantize_dequantize(x,
                                                        quantize_dequantize.get_scale(),
                                                        quantize_dequantize.get_offset(),
                                                        quantize_dequantize.bitwidth)
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
      4. offset = -1 * 2 ** (bw - 1)
      5. offset is fixed (offset.requires_grad = False)
    """
    min = q.get_min()
    max = q.get_max()
    scale = q.get_scale()
    offset = q.get_offset()
    bw = q.bitwidth

    # min == scale * offset
    assert torch.allclose(min, scale * offset,
                          rtol=1e-3, atol=scale.abs().max().item() * 1e-5)

    # max == scale * -(offset+1)
    assert torch.allclose(max, scale * -(offset+1),
                          rtol=1e-3, atol=scale.abs().max().item() * 1e-5)

    total_bins = 2**bw - 1
    positive_bins = math.floor(total_bins / 2)
    negative_bins = math.ceil(total_bins / 2)
    range = torch.maximum(-min * total_bins/negative_bins,
                          max * total_bins/positive_bins)
    assert torch.allclose(scale, range / total_bins,
                          rtol=1e-3, atol=scale.abs().max().item() * 1e-5)

    # offset == -1 * 2 ** (bw -1)
    assert torch.equal(offset, torch.ones_like(offset) * -2 ** (bw-1))
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
