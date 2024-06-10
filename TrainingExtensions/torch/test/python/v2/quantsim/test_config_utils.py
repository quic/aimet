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
""" Test commonly used utilities for configuring quantsim """

import pytest
import torch
from aimet_torch.examples.test_models import SingleResidualWithAvgPool
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.quantsim.config_utils import set_activation_quantizers_to_float, \
    set_blockwise_quantization_for_weights, set_grouped_blockwise_quantization_for_weights
from aimet_torch.v2.quantization.affine import QuantizeDequantize, GroupedBlockQuantizeDequantize
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize
import aimet_torch.v2.nn as aimet_nn

def test_set_activation_quantizers_to_float():
    model = SingleResidualWithAvgPool().eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    qsim = QuantizationSimModel(model, dummy_input)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)

    other_layers = []
    relu_layers = []
    for _, module in qsim.named_qmodules():
        if isinstance(module, aimet_nn.FakeQuantizedReLU):
            relu_layers.append(module)
        elif hasattr(module, 'output_quantizers') and len(module.output_quantizers) > 0 and \
                module.output_quantizers[0] is not None:
            other_layers.append(module)
    assert relu_layers
    assert other_layers

    for layer in relu_layers + other_layers:
        assert layer.output_quantizers[0].is_initialized()
        assert isinstance(layer.output_quantizers[0], QuantizeDequantize)

    with pytest.raises(RuntimeError):
        set_activation_quantizers_to_float(qsim, [1], dtype=torch.float16)

    set_activation_quantizers_to_float(qsim, [relu_layers[0]], dtype=torch.float16)
    assert isinstance(relu_layers[0].output_quantizers[0], FloatQuantizeDequantize)

    for relu_layer in relu_layers[1:]:
        assert isinstance(relu_layer.output_quantizers[0], QuantizeDequantize)

    set_activation_quantizers_to_float(qsim,
                                       lambda m: m == relu_layers[1],
                                       dtype=torch.float16)

    for relu_layer in relu_layers[:2]:
        assert isinstance(relu_layer.output_quantizers[0], FloatQuantizeDequantize)

    for relu_layer in relu_layers[2:]:
        assert isinstance(relu_layer.output_quantizers[0], QuantizeDequantize)

    set_activation_quantizers_to_float(qsim, [aimet_nn.FakeQuantizedReLU], dtype=torch.float16)
    for relu_layer in relu_layers:
        assert isinstance(relu_layer.output_quantizers[0], FloatQuantizeDequantize)

    for other_layer in other_layers:
        assert isinstance(other_layer.output_quantizers[0], QuantizeDequantize)

def test_set_blockwise_quantization_for_weights():
    model = SingleResidualWithAvgPool().eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    qsim = QuantizationSimModel(model, dummy_input)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)

    conv_layers = [module for module in qsim.model.modules() if isinstance(module, aimet_nn.QuantizedConv2d)]

    # exclude the 1st conv layers since its in channels of 3 makes it inconvenient to set blockwise
    conv_layers = conv_layers[1:]

    for conv_layer in conv_layers:
        assert conv_layer.param_quantizers['weight'].is_initialized()
        assert len(conv_layer.param_quantizers['weight'].shape) == 1
        assert conv_layer.param_quantizers['weight'].bitwidth == 8

    set_blockwise_quantization_for_weights(qsim, [conv_layers[0]], 4, True, [1, 4, -1, -1])

    assert not conv_layers[0].param_quantizers['weight'].is_initialized()
    assert conv_layers[0].param_quantizers['weight'].bitwidth == 4
    assert tuple(conv_layers[0].param_quantizers['weight'].shape) == (conv_layers[0].weight.shape[0],
                                                                      conv_layers[0].weight.shape[1] // 4,
                                                                      1,
                                                                      1)

    for conv_layer in conv_layers[1:]:
        assert conv_layer.param_quantizers['weight'].is_initialized()
        assert len(conv_layer.param_quantizers['weight'].shape) == 1
        assert conv_layer.param_quantizers['weight'].bitwidth == 8

    set_blockwise_quantization_for_weights(qsim, lambda m: m == conv_layers[1], 4, True, [1, 4, -1, -1])

    for conv_layer in conv_layers[:2]:
        assert not conv_layer.param_quantizers['weight'].is_initialized()
        assert conv_layer.param_quantizers['weight'].bitwidth == 4
        assert tuple(conv_layer.param_quantizers['weight'].shape) == (conv_layer.weight.shape[0],
                                                                      conv_layer.weight.shape[1] // 4,
                                                                      1,
                                                                      1)

    for conv_layer in conv_layers[2:]:
        assert conv_layer.param_quantizers['weight'].is_initialized()
        assert len(conv_layer.param_quantizers['weight'].shape) == 1
        assert conv_layer.param_quantizers['weight'].bitwidth == 8


    assert qsim.model.fc.param_quantizers['weight'].is_initialized()
    assert qsim.model.fc.param_quantizers['weight'].bitwidth == 8
    assert len(qsim.model.fc.param_quantizers['weight'].shape) == 1

    set_blockwise_quantization_for_weights(qsim, [aimet_nn.QuantizedLinear], 4, True, [1, 4])

    assert not qsim.model.fc.param_quantizers['weight'].is_initialized()
    assert qsim.model.fc.param_quantizers['weight'].bitwidth == 4
    assert tuple(qsim.model.fc.param_quantizers['weight'].shape) == (qsim.model.fc.weight.shape[0],
                                                                     qsim.model.fc.weight.shape[1] // 4)

def test_set_grouped_blockwise_quantization_for_weights():
    model = SingleResidualWithAvgPool().eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    qsim = QuantizationSimModel(model, dummy_input)
    qsim.compute_encodings(lambda m, _: m(dummy_input), None)

    conv_layers = [module for module in qsim.model.modules() if isinstance(module, aimet_nn.QuantizedConv2d)]

    # exclude the 1st conv layers since its in channels of 3 makes it inconvenient to set blockwise
    conv_layers = conv_layers[1:]

    for conv_layer in conv_layers:
        assert isinstance(conv_layer.param_quantizers['weight'], QuantizeDequantize)
        assert conv_layer.param_quantizers['weight'].is_initialized()
        assert len(conv_layer.param_quantizers['weight'].shape) == 1
        assert conv_layer.param_quantizers['weight'].bitwidth == 8

    set_grouped_blockwise_quantization_for_weights(qsim, [conv_layers[0]], 4, True, 8, [1, 4, -1, -1], [1, -1, 1, 1])

    assert isinstance(conv_layers[0].param_quantizers['weight'], GroupedBlockQuantizeDequantize)
    assert not conv_layers[0].param_quantizers['weight'].is_initialized()
    assert conv_layers[0].param_quantizers['weight'].bitwidth == 4
    assert tuple(conv_layers[0].param_quantizers['weight'].shape) == (conv_layers[0].weight.shape[0],
                                                                      conv_layers[0].weight.shape[1] // 4,
                                                                      1,
                                                                      1)
    assert conv_layers[0].param_quantizers['weight'].decompressed_bw == 8
    assert tuple(conv_layers[0].param_quantizers['weight'].block_grouping) == (1, -1, 1, 1)

    for conv_layer in conv_layers[1:]:
        assert isinstance(conv_layer.param_quantizers['weight'], QuantizeDequantize)
        assert conv_layer.param_quantizers['weight'].is_initialized()
        assert len(conv_layer.param_quantizers['weight'].shape) == 1
        assert conv_layer.param_quantizers['weight'].bitwidth == 8

    set_grouped_blockwise_quantization_for_weights(qsim, lambda m: m == conv_layers[1], 4, True, 8, [1, 4, -1, -1],
                                                   [1, -1, 1, 1])

    for conv_layer in conv_layers[:2]:
        assert isinstance(conv_layer.param_quantizers['weight'], GroupedBlockQuantizeDequantize)
        assert not conv_layer.param_quantizers['weight'].is_initialized()
        assert conv_layer.param_quantizers['weight'].bitwidth == 4
        assert tuple(conv_layer.param_quantizers['weight'].shape) == (conv_layer.weight.shape[0],
                                                                      conv_layer.weight.shape[1] // 4,
                                                                      1,
                                                                      1)
        assert conv_layer.param_quantizers['weight'].decompressed_bw == 8
        assert tuple(conv_layer.param_quantizers['weight'].block_grouping) == (1, -1, 1, 1)

    for conv_layer in conv_layers[2:]:
        assert conv_layer.param_quantizers['weight'].is_initialized()
        assert len(conv_layer.param_quantizers['weight'].shape) == 1
        assert conv_layer.param_quantizers['weight'].bitwidth == 8

    assert isinstance(qsim.model.fc.param_quantizers['weight'], QuantizeDequantize)
    assert qsim.model.fc.param_quantizers['weight'].is_initialized()
    assert qsim.model.fc.param_quantizers['weight'].bitwidth == 8
    assert len(qsim.model.fc.param_quantizers['weight'].shape) == 1

    set_grouped_blockwise_quantization_for_weights(qsim, [aimet_nn.QuantizedLinear], 4, True, 8, [1, 4], [1, -1])

    assert isinstance(qsim.model.fc.param_quantizers['weight'], GroupedBlockQuantizeDequantize)
    assert not qsim.model.fc.param_quantizers['weight'].is_initialized()
    assert qsim.model.fc.param_quantizers['weight'].bitwidth == 4
    assert tuple(qsim.model.fc.param_quantizers['weight'].shape) == (qsim.model.fc.weight.shape[0],
                                                                     qsim.model.fc.weight.shape[1] // 4)
    assert qsim.model.fc.param_quantizers['weight'].decompressed_bw == 8
    assert tuple(qsim.model.fc.param_quantizers['weight'].block_grouping) == (1, -1)
