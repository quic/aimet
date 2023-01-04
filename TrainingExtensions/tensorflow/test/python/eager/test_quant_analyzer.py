# /usr/bin/env python3.8
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
"""Test Quant Analyzer"""
import json
import os
import shutil

import numpy as np
import pytest
import tensorflow as tf
from aimet_tensorflow.keras.quant_analyzer import QuantAnalyzer

from aimet_common.utils import CallbackFunc
from aimet_tensorflow.keras.quantsim import QuantizationSimModel

from aimet_tensorflow.examples.test_models import keras_functional_conv_net, keras_sequential_conv_net


def forward_pass_func(model: tf.keras.Model, dummy_input):
    _ = model(dummy_input)


def eval_func(model: tf.keras.Model, dummy_input):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=tf.keras.metrics.CategoricalAccuracy())

    model.evaluate(dummy_input)
    return 0.8


@pytest.fixture()
def clear_session() -> None:
    tf.keras.backend.clear_session()


class TestQuantAnalyzer:
    def test_perform_per_layer_analysis_by_enabling_quant_wrappers(self):
        """ test perform per layer analysis by enabling quant wrappers """
        model = keras_functional_conv_net()
        layer_names = [layer.name for layer in model.layers]

        dummy_input = np.random.rand(1, 28, 28, 3)
        sim = QuantizationSimModel(model)
        sim.compute_encodings(forward_pass_func, dummy_input)

        forward_pass_callback = CallbackFunc(forward_pass_func, dummy_input)
        eval_callback = CallbackFunc(eval_func, dummy_input)
        quant_analyzer = QuantAnalyzer(model, forward_pass_callback, eval_callback)

        try:
            layer_wise_eval_score_dict = \
                quant_analyzer.perform_per_layer_analysis_by_enabling_quant_wrappers(sim, results_dir="./tmp/")
            assert type(layer_wise_eval_score_dict) == dict
            assert len(layer_wise_eval_score_dict) == 6

            # test whether layer_wise_eval_score_dict consists of correct keys (module names).
            for quant_wrapper_name in layer_wise_eval_score_dict.keys():
                assert quant_wrapper_name in layer_names

                # Check if it is exported to correct html file.
                assert os.path.isfile("./tmp/per_layer_quant_enabled.html")
        finally:
            if os.path.isdir("./tmp/"):
                shutil.rmtree("./tmp/")

    def test_perform_per_layer_analysis_by_disabling_quant_wrappers(self):
        """ test perform per layer analysis by disabling quant wrappers """
        model = keras_functional_conv_net()
        layer_names = [layer.name for layer in model.layers]

        dummy_input = np.random.rand(1, 28, 28, 3)
        sim = QuantizationSimModel(model)
        sim.compute_encodings(forward_pass_func, dummy_input)

        forward_pass_callback = CallbackFunc(forward_pass_func, dummy_input)
        eval_callback = CallbackFunc(eval_func, dummy_input)
        quant_analyzer = QuantAnalyzer(model, forward_pass_callback, eval_callback)
        try:
            layer_wise_eval_score_dict = \
                quant_analyzer.perform_per_layer_analysis_by_disabling_quant_wrappers(sim, results_dir="./tmp/")
            assert type(layer_wise_eval_score_dict) == dict
            assert len(layer_wise_eval_score_dict) == 6

            # test whether layer_wise_eval_score_dict consists of correct keys (module names).
            for quant_wrapper_name in layer_wise_eval_score_dict.keys():
                assert quant_wrapper_name in layer_names

                # Check if it is exported to correct html file.
                assert os.path.isfile("./tmp/per_layer_quant_disabled.html")
        finally:
            if os.path.isdir("./tmp/"):
                shutil.rmtree("./tmp/")

    def test_export_per_layer_encoding_min_max_range(self, clear_session):
        """ test export_per_layer_encoding_min_max_range() """
        model = keras_functional_conv_net()

        dummy_input = np.random.rand(1, 28, 28, 3)
        sim = QuantizationSimModel(model)
        sim.compute_encodings(forward_pass_func, dummy_input)

        forward_pass_callback = CallbackFunc(forward_pass_func, dummy_input)
        eval_callback = CallbackFunc(eval_func, dummy_input)
        quant_analyzer = QuantAnalyzer(model, forward_pass_callback, eval_callback)
        try:
            quant_analyzer.export_per_layer_encoding_min_max_range(sim, results_dir="./tmp/")
            assert os.path.isfile("./tmp/min_max_ranges/weights.html")
            assert os.path.isfile("./tmp/min_max_ranges/activations.html")
        finally:
            if os.path.isdir("./tmp/"):
                shutil.rmtree("./tmp/")

    def test_export_per_layer_encoding_min_max_range_per_channel(self, clear_session):
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
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }

        with open("./tmp/quantsim_config.json", "w") as f:
            json.dump(quantsim_config, f)

        model = keras_sequential_conv_net()

        dummy_input = np.random.rand(1, 28, 28, 3)
        sim = QuantizationSimModel(model, config_file="./tmp/quantsim_config.json")
        sim.compute_encodings(forward_pass_func, dummy_input)

        forward_pass_callback = CallbackFunc(forward_pass_func, dummy_input)
        eval_callback = CallbackFunc(eval_func, dummy_input)
        quant_analyzer = QuantAnalyzer(model, forward_pass_callback, eval_callback)
        try:
            quant_analyzer.export_per_layer_encoding_min_max_range(sim, results_dir="./tmp/")
            assert os.path.isfile("./tmp/min_max_ranges/activations.html")
            assert os.path.isfile("./tmp/min_max_ranges/conv2d_conv2d-kernel.html")
            assert os.path.isfile("./tmp/min_max_ranges/dense_dense-kernel.html")
        finally:
            if os.path.isdir("./tmp/"):
                shutil.rmtree("./tmp/")
