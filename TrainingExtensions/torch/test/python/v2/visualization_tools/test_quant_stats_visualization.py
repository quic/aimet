# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
import tempfile
import os.path
import torch
from models.test_models import TinyModel
from models.test_models import ModelWithUnusedMatmul
from aimet_torch.examples.test_models import SingleResidualWithAvgPool
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.quantsim import QuantScheme
from aimet_torch.v2.visualization_tools import visualize_stats
import aimet_torch.v2.nn as aimet_nn
from aimet_torch.v2.quantsim.config_utils import set_blockwise_quantization_for_weights
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config


def evaluate(model: torch.nn.Module, dummy_input: torch.Tensor):
    """
    Helper function to evaluate model performance given dummy input
    :param model: PyTorch model
    :param dummy_input: dummy input to model.
    """
    model.eval()
    for _ in range(2):
        with torch.no_grad():
            model(dummy_input)
    return 0.8

def evaluate_2(model: torch.nn.Module, dummy_input: tuple):
    """
    Helper function to evaluate model performance given dummy inputs as a tuple
    :param model: PyTorch model
    :param dummy_input: dummy inputs to model as a tuple.
    """
    model.eval()
    for _ in range(2):
        with torch.no_grad():
            model(*dummy_input)
    return 0.8


class TestQuantStatsVisualization:

    def test_visualize_stats(self):
        """ test visualize_stats with minmax observers"""
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf)
        sim.compute_encodings(evaluate, dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            visualize_stats(sim, dummy_input, os.path.join(tmp_dir, "test_visualize_stats.html"))
            assert os.path.isfile(os.path.join(tmp_dir, "test_visualize_stats.html"))

    def test_visualize_stats_histogram(self):
        """ test visualize_stats with histogram observers"""
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced)
        sim.compute_encodings(evaluate, dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            visualize_stats(sim, dummy_input, os.path.join(tmp_dir, "test_visualize_stats.html"))
            assert os.path.isfile(os.path.join(tmp_dir, "test_visualize_stats.html"))

    def test_per_channel(self):
        """ test visualize_stats in per channel case"""
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf, config_file=get_path_for_per_channel_config())
        sim.compute_encodings(evaluate, dummy_input)
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     visualize_stats(sim, dummy_input, os.path.join(tmp_dir, "test_visualize_stats.html"))
        #     assert os.path.isfile(os.path.join(tmp_dir, "test_visualize_stats.html"))
        tmp_dir = "/local/mnt/workspace/ipendse/mirror_sync/dev/minmax_visualization"
        visualize_stats(sim, dummy_input, os.path.join(tmp_dir, "test_perchannel_vis.html"))
        assert os.path.isfile(os.path.join(tmp_dir, "test_perchannel_vis.html"))

    def test_per_channel_histogram(self):
        """ test visualize_stats in per channel case with histogram observers"""
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced, config_file=get_path_for_per_channel_config())
        sim.compute_encodings(evaluate, dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            visualize_stats(sim, dummy_input, os.path.join(tmp_dir, "test_visualize_stats.html"))
            assert os.path.isfile(os.path.join(tmp_dir, "test_visualize_stats.html"))

    def test_blockwise(self):
        model = SingleResidualWithAvgPool().eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf)
        sim.compute_encodings(lambda m, _: m(dummy_input), None)
        conv_layers = [module for module in sim.model.modules() if isinstance(module, aimet_nn.QuantizedConv2d)]

        # exclude the 1st conv layers since its in channels of 3 makes it inconvenient to set blockwise
        conv_layers = conv_layers[1:]
        set_blockwise_quantization_for_weights(sim, [conv_layers[0]], 4, True, [1, 4, -1, -1])
        set_blockwise_quantization_for_weights(sim, lambda m: m == conv_layers[1], 4, True, [1, 4, -1, -1])
        set_blockwise_quantization_for_weights(sim, [aimet_nn.QuantizedLinear], 4, True, [1, 4])
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     visualize_stats(sim, dummy_input, os.path.join(tmp_dir, "test_visualize_stats.html"))
        #     assert os.path.isfile(os.path.join(tmp_dir, "test_visualize_stats.html"))
        tmp_dir = "/local/mnt/workspace/ipendse/mirror_sync/dev/minmax_visualization"
        visualize_stats(sim, dummy_input, os.path.join(tmp_dir, "test_blockwise_vis.html"))
        assert os.path.isfile(os.path.join(tmp_dir, "test_blockwise_vis.html"))


    def test_not_calibrated_error(self):
        """ Check whether an exception is raised if QuantSim is not calibrated """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(RuntimeError):
                visualize_stats(sim, dummy_input, os.path.join(tmp_dir, "test_visualize_stats.html"))

    def test_not_quantsim_object_error(self):
        """ Check whether the input is a QuantizationSimModel instance """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(TypeError):
                visualize_stats(model, dummy_input, os.path.join(tmp_dir, "test_visualize_stats.html"))

    def test_not_a_directory_error(self):
        """ Raise exception if directory does not exist """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf)
        sim.compute_encodings(evaluate, dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = tmp_dir
        with pytest.raises(NotADirectoryError, match=f"'{tmp}' is not a directory."):
            visualize_stats(sim, dummy_input, os.path.join(tmp, "test_visualize_stats.html"))

    def test_no_html_extension_error(self):
        """ Raise exception if provided path does not end with .html """
        input_shape = (1, 3, 32, 32)
        dummy_input = torch.randn(*input_shape)
        model = TinyModel().eval()
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf)
        sim.compute_encodings(evaluate, dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="'save_path' must end with '.html'."):
                visualize_stats(sim, dummy_input, os.path.join(tmp_dir, "test_visualize_stats.jpg"))

    def test_model_with_unused_matmul(self):
        """ Check that the visualization is generated even when there exists an unused matmul """
        model = ModelWithUnusedMatmul()
        dummy_input = (torch.randn(10), torch.randn(10))
        sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf)
        sim.compute_encodings(evaluate_2, dummy_input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            visualize_stats(sim, dummy_input, os.path.join(tmp_dir, "test_visualize_stats.html"))
            assert os.path.isfile(os.path.join(tmp_dir, "test_visualize_stats.html"))
