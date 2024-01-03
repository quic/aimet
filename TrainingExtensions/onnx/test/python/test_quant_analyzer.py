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
import copy
import pytest
import json
import os.path
import shutil
from typing import Dict

import numpy as np
import torch
import onnxruntime as ort

from aimet_common.utils import CallbackFunc

from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.quant_analyzer import QuantAnalyzer

from models import models_for_tests

def calibrate(session: ort.InferenceSession, dummy_input: Dict[str, np.ndarray]):
    """
    Helper function to calibrate model given dummy input
    :param model: ONNX model session.
    :param dummy_input: dummy input to model.
    """
    _ = session.run(None, dummy_input)

def evaluate(session: ort.InferenceSession, dummy_input: Dict[str, np.ndarray]):
    """
    Helper function to evaluate model performance given dummy input
    :param model: ONNX model session.
    :param dummy_input: dummy input to model.
    """
    for i in range(2):
        _ = session.run(None, dummy_input)
    return 0.8

class TestQuantAnalyzer:

    def test_check_model_sensitivity_to_quantization(self):
        """ test analyze_model_sensitivity API """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        sim.compute_encodings(calibrate, dummy_input_dict)
        forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
        eval_callback = CallbackFunc(evaluate, dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)
        fp32_acc, weight_quantized_acc, act_quantized_acc = quant_analyzer._check_model_sensitivity_to_quantization(sim)

        assert fp32_acc >= weight_quantized_acc
        assert fp32_acc >= act_quantized_acc
        assert quant_analyzer._model is model

    def test_get_enabled_activation_quantizers(self):
        """ test get_enabled_activation_quantizers()  """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, CallbackFunc(None), CallbackFunc(None))
        enabled_quantizers = quant_analyzer._get_enabled_activation_quantizers(sim)

        # total 8 activation quantizers (conv + relu is a supergroup) are enabled as per default config file.
        assert len(enabled_quantizers) == 8

    def test_get_enabled_param_quantizers(self):
        """ test get_enabled_param_quantizers() """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, CallbackFunc(None), CallbackFunc(None))
        enabled_quantizers = quant_analyzer._get_enabled_param_quantizers(sim)

        # total 5 param quantizers are enabled as per default config file.
        assert len(enabled_quantizers) == 5
