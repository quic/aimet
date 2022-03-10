# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
from unittest.mock import MagicMock
from aimet_torch.examples.test_models import TinyModel
from aimet_common.defs import QuantScheme
from aimet_torch.qc_quantize_op import QcQuantizeWrapper, StaticGridQuantWrapper
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.quant_analyzer import QuantAnalyzer, CallbackFunc


def calibrate(model: torch.nn.Module, dummy_input: torch.Tensor):
    """
    Helper function to calibrate model given dummy input
    :param model: PyTorch model.
    :param dummy_input: dummy input to model.
    """
    model.eval()
    with torch.no_grad():
        model(dummy_input)

def evaluate(model: torch.nn.Module, dummy_input: torch.Tensor):
    """
    Helper function to evaluate model performance given dummy input
    :param model: PyTorch model
    :param dummy_input: dummy input to model.
    """
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    return 0.8


class TestQuantAnalyzer:


    def test_quant_analyzer(self):
        """ test analyze_model_sensitivity API """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        forward_pass_callback = CallbackFunc(calibrate, dummy_input)
        eval_callback = CallbackFunc(evaluate, dummy_input)
        analyzer = QuantAnalyzer(model, dummy_input, forward_pass_callback, eval_callback)
        fp32_acc, weight_quantized_acc, act_quantized_acc = analyzer.check_model_sensitivity_to_quantization(
            default_quant_scheme=QuantScheme.post_training_tf_enhanced, default_param_bw=8, default_output_bw=8)

        assert fp32_acc >= weight_quantized_acc
        assert fp32_acc >= act_quantized_acc
        assert analyzer._model is model

    def test_sort_quant_wrappers_based_on_occurrence(self):
        """ test sort quant wrappers based on occurrence """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(model, dummy_input)
        analyzer = QuantAnalyzer(model, dummy_input, CallbackFunc(None), CallbackFunc(None))
        sorted_quant_wrappers_dict = analyzer._sort_quant_wrappers_based_on_occurrence(sim)
        assert isinstance(sorted_quant_wrappers_dict, dict)
        for quant_wrapper in sorted_quant_wrappers_dict.values():
            assert isinstance(quant_wrapper, QcQuantizeWrapper)
        assert len(sorted_quant_wrappers_dict) == 12

    def test_perform_per_layer_analysis_by_enabling_quant_wrappers(self):
        """ test perform per layer analysis by enabling quant wrappers """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        forward_pass_callback = CallbackFunc(calibrate, dummy_input)
        eval_callback = CallbackFunc(evaluate, dummy_input)
        analyzer = QuantAnalyzer(model, dummy_input, forward_pass_callback, eval_callback)
        layer_wise_eval_score_dict = analyzer.perform_per_layer_analysis_by_enabling_quant_wrappers()
        print(layer_wise_eval_score_dict)
        assert type(layer_wise_eval_score_dict) == dict
        assert len(layer_wise_eval_score_dict) == 12

    def test_perform_per_layer_analysis_by_disabling_quant_wrappers(self):
        """ test perform per layer analysis by disabling quant wrappers """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        forward_pass_callback = CallbackFunc(calibrate, dummy_input)
        eval_callback = CallbackFunc(evaluate, dummy_input)
        analyzer = QuantAnalyzer(model, dummy_input, forward_pass_callback, eval_callback)
        layer_wise_eval_score_dict = analyzer.perform_per_layer_analysis_by_disabling_quant_wrappers()
        print(layer_wise_eval_score_dict)
        assert type(layer_wise_eval_score_dict) == dict
        assert len(layer_wise_eval_score_dict) == 12