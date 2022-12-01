# /usr/bin/env python3.6
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
"""Quant Analyzer"""
import os

import tensorflow as tf

from aimet_common.defs import QuantScheme
from aimet_common.utils import CallbackFunc, AimetLogger
from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.utils.quantizer_utils import get_enabled_activation_quantizers, enable_disable_quantizers, \
    get_enabled_param_quantizers

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class QuantAnalyzer:
    """
    QuantAnalyzer tool provides

     1) model sensitivity to weight and activation quantization
     2) per layer sensitivity analysis
     3) per layer encoding (min - max range)
     4) per PDF analysis and
     4) per layer MSE analysis
    """

    def __init__(self,
                 model: tf.keras.Model,
                 forward_pass_callback: CallbackFunc,
                 eval_callback: CallbackFunc):
        """
        :param model: FP32 model to analyze for quantization.
        :param forward_pass_callback: A callback function for model calibration that simply runs
                forward passes on the model to compute encoding (delta/offset). This
                callback function should use representative data and should be subset of
                entire train/validation dataset (~1000 images/samples).
        :param eval_callback: A callback function for model evaluation that determines model
                performance. This callback function is expected to return scalar value
                representing the model performance evaluated against entire test/evaluation dataset.
        """
        if not isinstance(forward_pass_callback, CallbackFunc):
            raise ValueError('forward_pass_callback and its argument(s) are not encapsulated by CallbackFunc class.')
        if not isinstance(eval_callback, CallbackFunc):
            raise ValueError('eval_callback and its argument(s) are not encapsulated by CallbackFunc class.')

        self._model = model
        self._forward_pass_callback = forward_pass_callback
        self._eval_callback = eval_callback

    # pylint: disable=unused-argument, no-self-use
    def analyze(self,
                quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                rounding_mode: str = "nearest",
                default_param_bw: int = 8,
                default_output_bw: int = 8,
                config_file: str = None,
                results_dir: str = "./tmp/"):
        """
        Analyze model for quantization and point out sensitive parts/hotspots of the model by performing
            1) model sensitivity to quantization,
            2) perform per layer sensitivity analysis by enabling and disabling quant wrappers,
            3) export per layer encodings min - max ranges,
            4) export per layer statistics histogram (PDF) when quant scheme is TF-Enhanced,
            5) per layer MSE analysis

        :param quant_scheme: Quantization scheme. Supported values are
                QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced.
        :param rounding_mode: The round scheme to used. One of: 'nearest' or 'stochastic', defaults to 'nearest'
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param config_file: Path to configuration file for model quantizers.
        :param results_dir: Directory to save the results.
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        sim = self._create_quantsim_and_encodings(quant_scheme,
                                                  rounding_mode,
                                                  default_param_bw,
                                                  default_output_bw,
                                                  config_file)

        # Check model sensitivity to weight and activation quantization individually.
        self._check_model_sensitivity_to_quantization(sim, default_param_bw, default_output_bw)


    def _create_quantsim_and_encodings(self,
                                       quant_scheme: QuantScheme,
                                       rounding_mode: str,
                                       default_param_bw: int,
                                       default_output_bw: int,
                                       config_file: str) -> QuantizationSimModel:
        """
        Create Quantsim and compute encodings.

        :param quant_scheme: Quantization scheme.
        :param rounding_mode: The round scheme to used. One of: 'nearest' or 'stochastic', defaults to 'nearest'
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param config_file: Path to configuration file for model quantizers.
        :return: Quantsim model.
        """
        _ = fold_all_batch_norms(self._model)
        sim = QuantizationSimModel(self._model,
                                   quant_scheme=quant_scheme,
                                   rounding_mode=rounding_mode,
                                   default_output_bw=default_output_bw,
                                   default_param_bw=default_param_bw,
                                   config_file=config_file)

        sim.compute_encodings(forward_pass_callback=self._forward_pass_callback.func,
                              forward_pass_callback_args=self._forward_pass_callback.args)

        return sim

    def _check_model_sensitivity_to_quantization(self,
                                                 sim: QuantizationSimModel,
                                                 default_param_bw: int,
                                                 default_output_bw: int):
        """
        Perform the sensitivity analysis to weight and activation quantization
        individually.

        :param sim: Quantsim model.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :return: FP32 eval score, weight-quantized eval score, act-quantized eval score.
        """
        fp32_eval_score = self._eval_model(self._model)
        _logger.info("FP32 eval score (W32A32): %f", fp32_eval_score)

        weight_quantized_eval_score = self._eval_weight_quantized_model(sim)
        _logger.info("Weight-quantized eval score (W%dA32): %f", default_param_bw,
                     weight_quantized_eval_score)

        act_quantized_eval_score = self._eval_activation_quantized_model(sim)
        _logger.info("Activation-quantized eval score (W32A%d): %f", default_output_bw,
                     act_quantized_eval_score)

    def _eval_model(self, model: tf.keras.Model) -> float:
        """
        Evaluate the model performance.
        :param model: tf.keras.Model to be evaluated
        :return: Scalar value representing model performance
        """
        return self._eval_callback.func(model, self._eval_callback.args)

    def _eval_weight_quantized_model(self, sim: QuantizationSimModel) -> float:
        """
        Evaluate weight quantized model performance.
        For weight quantized model performance, disable enabled activation quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        enabled_activation_quantizers = get_enabled_activation_quantizers(sim)
        enable_disable_quantizers(enabled_activation_quantizers, enabled=False)
        eval_score = self._eval_model(sim.model)
        enable_disable_quantizers(enabled_activation_quantizers, enabled=True)
        return eval_score

    def _eval_activation_quantized_model(self, sim: QuantizationSimModel) -> float:
        """
        Evaluate activation quantized model performance.
        For activation quantized model performance, disable enabled param quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        enabled_param_quantizers = get_enabled_param_quantizers(sim)
        enable_disable_quantizers(enabled_param_quantizers, enabled=False)
        eval_score = self._eval_model(sim.model)
        enable_disable_quantizers(enabled_param_quantizers, enabled=True)
        return eval_score
