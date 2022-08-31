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

"""Automatic Post-Training Quantization"""
import os
from dataclasses import dataclass
from typing import List, Callable, Dict, Any, Tuple

import tensorflow as tf

from aimet_common.cache import Cache
from aimet_common.defs import QuantScheme
from aimet_common.quantsim import validate_quantsim_inputs
from aimet_common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AutoQuant)

cache = Cache()


class AutoQuant:
    """
    Integrate and apply post-training quantization techniques.

    AutoQuant includes 1) batchnorm folding, 2) cross-layer equalization,
    and 3) Adaround.
    These techniques will be applied in a best-effort manner until the model
    meets the evaluation goal given as allowed_accuracy_drop.
    """

    def __init__(self,
                 allowed_accuracy_drop: float,
                 default_param_bw: int = 8,
                 default_output_bw: int = 8,
                 default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                 default_rounding_mode: str = "nearest",
                 default_config_file: str = None):
        """
        :param allowed_accuracy_drop: Maximum allowed accuracy drop.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param default_quant_scheme: Quantization scheme. Supported values are
                QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced.
        :param default_rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
        :param default_config_file: Path to configuration file for model quantizers
        """
        if allowed_accuracy_drop < 0:
            raise ValueError(
                "`allowed_accuracy_drop` must be a positive value. Got {:.2f}"
                .format(allowed_accuracy_drop)
            )

        validate_quantsim_inputs(default_quant_scheme,
                                 default_rounding_mode,
                                 default_output_bw,
                                 default_param_bw)

        self.allowed_accuracy_drop = allowed_accuracy_drop
        self.default_param_bw = default_param_bw
        self.default_output_bw = default_output_bw
        self.default_quant_scheme = default_quant_scheme
        self.default_rounding_mode = default_rounding_mode
        self.default_config_file = default_config_file

    def apply(self,
              fp32_model: tf.keras.Model,
              results_dir: str = "/tmp",
              cache_id: str = None) -> Tuple[tf.keras.Model, float, str]:
        """
        Apply post-training quantization techniques.

        :param fp32_model: Model to apply PTQ techniques.
        :param results_dir: Directory to save the results.
        :param cache_id: A string that composes a cache id in combination with results_dir.
            If specified, AutoQuant will load/save the PTQ results from/to the file system
            if previous PTQ results produced under the same results_dir and cache_id exist,
        :return: Tuple of (best model, eval score, encoding path).
        """
        result = self._apply_helper(self._auto_quant_main,
                                    fp32_model,
                                    results_dir,
                                    cache_id)

        return result["model"], result["accuracy"], result["encoding_path"]

    def _apply_helper(self,
                      auto_quant_main_fn: Callable,
                      fp32_model: tf.keras.Model,
                      results_dir: str = "/tmp",
                      cache_id: str = None) -> Dict[str, Any]:
        """

        :param auto_quant_main_fn: Function that implements the main logic of AutoQuant.
        :param fp32_model: Model to apply PTQ techniques.
        :param results_dir: Directory to save the results.
        :param cache_id: A string that composes a cache id in combination with results_dir.
            If specified, AutoQuant will load/save the PTQ results from/to the file system
            if previous PTQ results produced under the same results_dir and cache_id exist,
        :return: The best ptq result as a dictionary.
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        if cache_id is None:
            cache_dir = None
        else:
            cache_dir = os.path.join(results_dir, ".auto_quant_cache", cache_id)

        with cache.enable(cache_dir):
            _logger.info("Starting AutoQuant")

            # TODO: Implement evaluation of model performance
            fp32_acc = ...
            target_acc = fp32_acc - self.allowed_accuracy_drop

            _logger.info("Target eval score: %f", target_acc)
            _logger.info("FP32 eval score (W32A32): %f", fp32_acc)

            eval_manager = _EvalManager()

            ret = auto_quant_main_fn(fp32_model, target_acc,
                                     eval_manager, results_dir)

            return ret

    # TODO: Remove temporary pylint disable after full implementation
    # pylint: disable=no-self-use, unused-argument
    def _auto_quant_main(self,
                         fp32_model: tf.keras.Model,
                         target_acc: float,
                         eval_manager: "_EvalManager",
                         results_dir: str = "/tmp") -> Dict[str, Any]:
        """
        Helper function of apply().

        :param fp32_model: Model to apply PTQ techniques.
        :param target_acc: Target eval score.
        :param eval_manager: _Evalmanager object.
        :param results_dir: Directory to save the results.
        :return: The best ptq result as a dictionary.
        """

        return eval_manager.get_best_ptq_result().as_dict()


@dataclass
class PtqResult:
    """
    Evaluation results
    :param model_path: Path to the serialized model.
    :param encoding_path: Path to the encoding file.
    :param accuracy: Accuracy of the model.
    :param applied_techniques: Applied ptq techniques.
    """
    model_path: str
    encoding_path: str
    accuracy: float
    applied_techniques: List[str]

    def load_model(self):
        """
        Load model
        :return: Loaded model
        """
        return tf.keras.models.load_model(self.model_path)

    def as_dict(self):
        """Convert to dictionary"""
        return dict(model=self.load_model(),
                    accuracy=self.accuracy,
                    encoding_path=self.encoding_path,
                    applied_techniques=self.applied_techniques)


class _EvalManager:
    """
    Evaluation manager for AutoQuant.
    """

    def __init__(self):
        # TODO: Implement _PtqSession
        # self._ptq_sessions: List[_PtqSession] = []
        self._ptq_sessions = []

    def get_best_ptq_result(self) -> PtqResult:
        """
        Get the results with the highest evaluation score among the ptq results evaluated so far.
        :return: The best evaluation result so far.
        """
        if not self._ptq_sessions:
            raise RuntimeError

        ptq_results = [sess.ptq_result for sess in self._ptq_sessions]
        return max(ptq_results, key=lambda ptq_result: ptq_result.accuracy)
