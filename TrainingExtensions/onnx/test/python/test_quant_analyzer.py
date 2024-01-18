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
from packaging import version

import numpy as np
import torch
import onnxruntime as ort

from aimet_common.utils import CallbackFunc

from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.qc_quantize_op import QcQuantizeOp
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
        fp32_acc, weight_quantized_acc, act_quantized_acc = quant_analyzer.check_model_sensitivity_to_quantization(sim)

        assert fp32_acc >= weight_quantized_acc
        assert fp32_acc >= act_quantized_acc
        assert quant_analyzer._onnx_model is model

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

    def test_get_enabled_quantizers(self):
        """ test get_enabled_quantizers() """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, CallbackFunc(None), CallbackFunc(None))
        op_to_quantizers_dict = quant_analyzer._get_enabled_quantizers(sim)

        # Verify all the ops having enabled quantizers are being captured
        assert len(op_to_quantizers_dict) == 10
        for op_name, enabled_quantizers in op_to_quantizers_dict.items():
            assert isinstance(op_name, str)
            assert all(isinstance(quantizer, QcQuantizeOp) for quantizer in enabled_quantizers)

        # Verify the order of ops in op_to_quantizers_dict is according to its occurence in the model graph
        if version.parse(torch.__version__) >= version.parse("1.13"):
            layer_names = list(op_to_quantizers_dict.keys())
            assert layer_names.index("/conv2/Conv") < layer_names.index("/relu2/Relu")
            assert layer_names.index("/conv4/Conv") < layer_names.index("/fc/Gemm")

        # Disable all the quantizers and verify op_to_quantizers_dict is empty.
        for _, qc_quantize_op in sim.qc_quantize_op_dict.items():
            qc_quantize_op.enabled = False
        op_to_quantizers_dict = quant_analyzer._get_enabled_quantizers(sim)
        assert not op_to_quantizers_dict

    def test_perform_per_layer_analysis_by_enabling_quantizers(self):
        """ test perform per layer analysis by enabling quantizers """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        layer_names = []
        for node in model.nodes():
            layer_names.append(node.name)

        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        sim.compute_encodings(evaluate, dummy_input_dict)
        forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
        eval_callback = CallbackFunc(evaluate, dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)
        try:
            layer_wise_eval_score_dict = \
                quant_analyzer.perform_per_layer_analysis_by_enabling_quantizers(sim, results_dir="./tmp/")
            print(layer_wise_eval_score_dict)
            assert type(layer_wise_eval_score_dict) == dict
            assert len(layer_wise_eval_score_dict) == 10

            # Test whether layer_wise_eval_score_dict consists of correct keys (op names).
            for op_name in layer_wise_eval_score_dict.keys():
                assert op_name in layer_names

            # Check if it is exported to correct html file.
            assert os.path.isfile("./tmp/per_layer_quant_enabled.html")
        finally:
            if os.path.isdir("./tmp/"):
                shutil.rmtree("./tmp/")

    def test_perform_per_layer_analysis_by_disabling_quantizers(self):
        """ test perform per layer analysis by disabling quantizers """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        layer_names = []
        for node in model.nodes():
            layer_names.append(node.name)

        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        sim.compute_encodings(evaluate, dummy_input_dict)
        forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
        eval_callback = CallbackFunc(evaluate, dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)
        try:
            layer_wise_eval_score_dict = \
                quant_analyzer.perform_per_layer_analysis_by_disabling_quantizers(sim, results_dir="./tmp/")
            print(layer_wise_eval_score_dict)
            assert type(layer_wise_eval_score_dict) == dict
            assert len(layer_wise_eval_score_dict) == 10

            # Test whether layer_wise_eval_score_dict consists of correct keys (op names).
            for op_name in layer_wise_eval_score_dict.keys():
                assert op_name in layer_names

            # Check if it is exported to correct html file.
            assert os.path.isfile("./tmp/per_layer_quant_disabled.html")
        finally:
            if os.path.isdir("./tmp/"):
                shutil.rmtree("./tmp/")

    def test_export_per_layer_encoding_min_max_range(self):
        """ test export_per_layer_encoding_min_max_range() """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict)
        sim.compute_encodings(evaluate, dummy_input_dict)
        forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
        eval_callback = CallbackFunc(evaluate, dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)
        try:
            quant_analyzer.export_per_layer_encoding_min_max_range(sim, results_dir="./tmp/")
            assert os.path.isfile("./tmp/min_max_ranges/weights.html")
            assert os.path.isfile("./tmp/min_max_ranges/activations.html")
        finally:
            if os.path.isdir("./tmp/"):
                shutil.rmtree("./tmp/")

    def test_export_per_layer_encoding_min_max_range_per_channel(self):
        """ test export_per_layer_encoding_min_max_range() for per channel quantization """
        results_dir = os.path.abspath("./tmp/")
        os.makedirs(results_dir, exist_ok=True)

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "per_channel_quantization": "True",
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {"Gemm": {"per_channel_quantization": "False"}},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with open("./tmp/quantsim_config.json", 'w') as f:
            json.dump(quantsim_config, f)

        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = models_for_tests._convert_to_onnx(models_for_tests.TinyModel(), dummy_input)
        dummy_input_dict = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
        fold_all_batch_norms_to_weight(model)
        sim = QuantizationSimModel(copy.deepcopy(model), dummy_input_dict, config_file="./tmp/quantsim_config.json")
        sim.compute_encodings(evaluate, dummy_input_dict)
        forward_pass_callback = CallbackFunc(calibrate, dummy_input_dict)
        eval_callback = CallbackFunc(evaluate, dummy_input_dict)
        quant_analyzer = QuantAnalyzer(model, dummy_input_dict, forward_pass_callback, eval_callback)
        try:
            quant_analyzer.export_per_layer_encoding_min_max_range(sim, results_dir="./tmp/")
            assert os.path.isfile("./tmp/min_max_ranges/activations.html")
            if version.parse(torch.__version__) >= version.parse("1.13"):
                assert os.path.isfile("./tmp/min_max_ranges/_conv1_Conv_onnx_Conv_39.html")
            # Dense (Gemm) is disabled to per-channel quantization, it should be in weights.html
            assert os.path.isfile("./tmp/min_max_ranges/weights.html")
        finally:
            if os.path.isdir("./tmp/"):
                shutil.rmtree("./tmp/")
