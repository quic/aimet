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

""" Unit tests for Quant Analyzer """
import shutil
import json
from typing import Any
import os
import pytest
import numpy as np
import tensorflow as tf
from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.quant_analyzer import QuantAnalyzer, CallbackFunc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()

@pytest.fixture
def cpu_session():
    """
    Function to get Tensorflow cpu session
    :return: session
    """
    tf.compat.v1.reset_default_graph()
    with tf.device('/cpu:0'):
        _ = keras_model()
        init = tf.compat.v1.global_variables_initializer()

    session = tf.compat.v1.Session()
    session.run(init)
    return session

def forward_pass_callback(session: tf.compat.v1.Session, _: Any = None):
    """
    Helper function to calibrate model for quantization.
    :param session: TensorFlow session.
    :param _:
    :return: model performance.
    """
    model_output = session.graph.get_tensor_by_name('keras_model/Softmax:0')
    model_input = session.graph.get_tensor_by_name('conv2d_input:0')
    for _ in range(2):
        dummy_input = np.random.randn(1, 16, 16, 3)
        session.run(model_output, feed_dict={model_input: dummy_input})
    return 0.8


#pylint: disable=redefined-outer-name
def test_export_per_layer_stats_histogram(cpu_session):
    """ test export_per_layer_stats_histogram() """

    sim = QuantizationSimModel(cpu_session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
    sim.compute_encodings(forward_pass_callback, None)
    quant_analyzer = QuantAnalyzer(cpu_session, start_op_names=['conv2d_input'],
                                   output_op_names=['keras_model/Softmax'],
                                   forward_pass_callback=CallbackFunc(forward_pass_callback),
                                   eval_callback=CallbackFunc(forward_pass_callback), use_cuda=False)

    try:
        #pylint: disable=protected-access
        quant_analyzer._export_per_layer_stats_histogram(sim, results_dir="./tmp/")

        # Check if it is exported to correct html file.
        assert os.path.exists("./tmp/activations_pdf")
        assert os.path.exists("./tmp/weights_pdf")
        assert os.path.isfile("./tmp/activations_pdf/conv2d_input_quantized_0.html")
        assert os.path.isfile("./tmp/activations_pdf/conv2d_BiasAdd_quantized_0.html")
        assert os.path.isfile("./tmp/weights_pdf/conv2d_Conv2D_ReadVariableOp_quantized/"
                              "conv2d_Conv2D_ReadVariableOp_quantized_0.html")
    finally:
        sim.session.close()
        cpu_session.close()
        if os.path.isdir("./tmp/"):
            shutil.rmtree("./tmp/")

#pylint: disable=redefined-outer-name
def test_export_per_layer_stats_histogram_per_channel(cpu_session):
    """ test export_per_layer_stats_histogram() for per channel quantization"""

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
        "params": {},
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }
    with open("./tmp/quantsim_config.json", 'w') as f:
        json.dump(quantsim_config, f)

    sim = QuantizationSimModel(cpu_session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False,
                               config_file="./tmp/quantsim_config.json")
    sim.compute_encodings(forward_pass_callback, None)
    quant_analyzer = QuantAnalyzer(cpu_session, start_op_names=['conv2d_input'],
                                   output_op_names=['keras_model/Softmax'],
                                   forward_pass_callback=CallbackFunc(forward_pass_callback),
                                   eval_callback=CallbackFunc(forward_pass_callback), use_cuda=False)

    try:
        #pylint: disable=protected-access
        quant_analyzer._export_per_layer_stats_histogram(sim, results_dir="./tmp/")

        # Check if it is exported to correct html file.
        assert os.path.exists("./tmp/activations_pdf")
        assert os.path.exists("./tmp/weights_pdf")
        assert os.path.isfile("./tmp/activations_pdf/conv2d_BiasAdd_quantized_0.html")
        assert os.path.isfile("./tmp/weights_pdf/conv2d_Conv2D_ReadVariableOp_quantized/"
                              "conv2d_Conv2D_ReadVariableOp_quantized_0.html")
        assert os.path.isfile("./tmp/weights_pdf/conv2d_Conv2D_ReadVariableOp_quantized/"
                              "conv2d_Conv2D_ReadVariableOp_quantized_7.html")
    finally:
        sim.session.close()
        cpu_session.close()
        if os.path.isdir("./tmp/"):
            shutil.rmtree("./tmp/")

#pylint: disable=redefined-outer-name
def test_export_per_layer_encoding_min_max_range(cpu_session):
    """ test export_per_layer_encoding_min_max_range() """

    sim = QuantizationSimModel(cpu_session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
    sim.compute_encodings(forward_pass_callback, None)
    quant_analyzer = QuantAnalyzer(cpu_session, start_op_names=['conv2d_input'],
                                   output_op_names=['keras_model/Softmax'],
                                   forward_pass_callback=CallbackFunc(forward_pass_callback),
                                   eval_callback=CallbackFunc(forward_pass_callback), use_cuda=False)
    try:
        #pylint: disable=protected-access
        quant_analyzer._export_per_layer_encoding_min_max_range(sim, results_dir="./tmp/")
        assert os.path.isfile("./tmp/min_max_ranges/weights.html")
        assert os.path.isfile("./tmp/min_max_ranges/activations.html")
    finally:
        sim.session.close()
        cpu_session.close()
        if os.path.isdir("./tmp/"):
            shutil.rmtree("./tmp/")

#pylint: disable=redefined-outer-name
def test_export_per_layer_encoding_min_max_range_per_channel(cpu_session):
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
    with open("./tmp/quantsim_config.json", 'w') as f:
        json.dump(quantsim_config, f)

    sim = QuantizationSimModel(cpu_session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False,
                               config_file="./tmp/quantsim_config.json")
    sim.compute_encodings(forward_pass_callback, None)
    quant_analyzer = QuantAnalyzer(cpu_session, start_op_names=['conv2d_input'],
                                   output_op_names=['keras_model/Softmax'],
                                   forward_pass_callback=CallbackFunc(forward_pass_callback),
                                   eval_callback=CallbackFunc(forward_pass_callback), use_cuda=False)
    try:
        #pylint: disable=protected-access
        quant_analyzer._export_per_layer_encoding_min_max_range(sim, results_dir="./tmp/")
        assert os.path.isfile("./tmp/min_max_ranges/activations.html")
        assert os.path.isfile("./tmp/min_max_ranges/conv2d_Conv2D_ReadVariableOp_quantized.html")
        assert os.path.isfile("./tmp/min_max_ranges/keras_model_MatMul_ReadVariableOp_quantized.html")
    finally:
        sim.session.close()
        cpu_session.close()
        if os.path.isdir("./tmp/"):
            shutil.rmtree("./tmp/")

#pylint: disable=redefined-outer-name
def test_model_sensitivity_to_quantization(cpu_session):
    """ tests _check_model_sensitivity_to_quantization() which perform the sensitivity analysis to
    parameter and activation quantization individually """

    sim = QuantizationSimModel(cpu_session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
    sim.compute_encodings(forward_pass_callback, None)
    quant_analyzer = QuantAnalyzer(cpu_session, start_op_names=['conv2d_input'],
                                   output_op_names=['keras_model/Softmax'],
                                   forward_pass_callback=CallbackFunc(forward_pass_callback),
                                   eval_callback=CallbackFunc(forward_pass_callback), use_cuda=False)
    #pylint: disable=protected-access
    fp32_acc, weight_quantized_acc, act_quantized_acc = quant_analyzer._check_model_sensitivity_to_quantization(sim)
    try:
        assert fp32_acc >= weight_quantized_acc
        assert fp32_acc >= act_quantized_acc
        #pylint: disable=protected-access
        assert quant_analyzer._session is cpu_session
    finally:
        sim.session.close()
        cpu_session.close()

#pylint: disable=redefined-outer-name
def test_get_enabled_activation_quantizers(cpu_session):
    """ test get_enabled_activation_quantizers()  """

    sim = QuantizationSimModel(cpu_session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
    sim.compute_encodings(forward_pass_callback, None)
    enabled_quantizers = sim.get_enabled_activation_quantizers()
    # total 9 activation quantizers are enabled as per default config file.
    try:
        assert len(enabled_quantizers) == 9
    finally:
        sim.session.close()
        cpu_session.close()

#pylint: disable=redefined-outer-name
def test_get_enabled_param_quantizers(cpu_session):
    """ test get_enabled_param_quantizers() """

    sim = QuantizationSimModel(cpu_session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
    sim.compute_encodings(forward_pass_callback, None)
    enabled_quantizers = sim.get_enabled_parameter_quantizers()
    # total 3 param quantizers are enabled as per default config file.
    try:
        assert len(enabled_quantizers) == 3
    finally:
        sim.session.close()
        cpu_session.close()

def test_get_enabled_quantizer_groups(cpu_session):
    """ test get_enabled_quantizers() """

    sim = QuantizationSimModel(cpu_session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
    sim.compute_encodings(forward_pass_callback, None)
    quant_analyzer = QuantAnalyzer(cpu_session, start_op_names=['conv2d_input'],
                                   output_op_names=['keras_model/Softmax'],
                                   forward_pass_callback=CallbackFunc(forward_pass_callback),
                                   eval_callback=CallbackFunc(forward_pass_callback), use_cuda=False)
    # total 10 quantizer groups are created.
    #pylint: disable=protected-access
    #pylint: disable=no-member
    try:
        assert len(quant_analyzer._get_enabled_quantizer_groups(sim)) == 10
    finally:
        sim.session.close()
        cpu_session.close()

#pylint: disable=redefined-outer-name
def test_perform_per_layer_analysis_by_enabling_quant_ops(cpu_session):
    """test _perform_per_op_analysis_by_enabling_quant_ops()"""
    sim = QuantizationSimModel(cpu_session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
    sim.compute_encodings(forward_pass_callback, None)
    quant_analyzer = QuantAnalyzer(cpu_session, start_op_names=['conv2d_input'],
                                   output_op_names=['keras_model/Softmax'],
                                   forward_pass_callback=CallbackFunc(forward_pass_callback),
                                   eval_callback=CallbackFunc(forward_pass_callback), use_cuda=False)
    try:
        #pylint: disable=protected-access
        #pylint: disable=no-member
        quant_analyzer._perform_per_op_analysis_by_enabling_quant_ops(sim, results_dir="./tmp/")
        assert os.path.isfile("./tmp/per_op_quant_enabled.html")
        assert os.path.isfile("./tmp/per_op_quant_enabled.json")
    finally:
        sim.session.close()
        cpu_session.close()
        if os.path.isdir("./tmp/"):
            shutil.rmtree("./tmp/")

#pylint: disable=redefined-outer-name
def test_perform_per_layer_analysis_by_disabling_quant_ops(cpu_session):
    """test _perform_per_op_analysis_by_disabling_quant_ops()"""
    sim = QuantizationSimModel(cpu_session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
    sim.compute_encodings(forward_pass_callback, None)
    quant_analyzer = QuantAnalyzer(cpu_session, start_op_names=['conv2d_input'],
                                   output_op_names=['keras_model/Softmax'],
                                   forward_pass_callback=CallbackFunc(forward_pass_callback),
                                   eval_callback=CallbackFunc(forward_pass_callback), use_cuda=False)
    try:
        #pylint: disable=protected-access
        #pylint: disable=no-member
        quant_analyzer._perform_per_op_analysis_by_disabling_quant_ops(sim, results_dir="./tmp/")
        assert os.path.isfile("./tmp/per_op_quant_disabled.html")
        assert os.path.isfile("./tmp/per_op_quant_disabled.json")
    finally:
        sim.session.close()
        cpu_session.close()
        if os.path.isdir("./tmp/"):
            shutil.rmtree("./tmp/")
