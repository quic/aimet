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

""" Quant Analyzer """

import os
from typing import Union, Tuple, Dict, List
import copy
from collections import defaultdict

import numpy as np
from onnx import ModelProto
import onnxruntime as ort
from onnxruntime.quantization.onnx_model import ONNXModel

from aimet_common.utils import AimetLogger, CallbackFunc
from aimet_common.defs import QuantScheme
from aimet_common.quant_analyzer import save_json, export_per_layer_sensitivity_analysis_plot

from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.QuantAnalyzer)


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
                 model: Union[ModelProto, ONNXModel],
                 dummy_input: Dict[str, np.ndarray],
                 forward_pass_callback: CallbackFunc,
                 eval_callback: CallbackFunc,
                 ):
        """
        :param model: FP32 model to analyze for quantization.
        :param dummy_input: Dummy input to model.
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

        self._onnx_model = model
        if not isinstance(self._onnx_model, ONNXModel):
            self._onnx_model = ONNXModel(self._onnx_model)
        self._dummy_input = dummy_input
        self._forward_pass_callback = forward_pass_callback
        self._eval_callback = eval_callback
        self._unlabeled_dataset_iterable = None
        self._num_batches = None

    def analyze(self,
                quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                default_param_bw: int = 8,
                default_activation_bw: int = 8,
                config_file: str = None,
                results_dir: str = "./tmp/",
                ):
        """
        Analyze model for quantization and point out sensitive parts/hotspots of the model by performing
            1) model sensitivity to quantization,
            2) perform per layer sensitivity analysis by enabling and disabling quant wrappers,

        :param quant_scheme: Quantization scheme. Supported values are
                QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_activation_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param config_file: Path to configuration file for model quantizers.
        :param results_dir: Directory to save the results.
        """
        sim = self.create_quantsim_and_encodings(quant_scheme,
                                                 default_param_bw,
                                                 default_activation_bw,
                                                 config_file)

        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        # Check model sensitivity to weight and activation quantization individually.
        self.check_model_sensitivity_to_quantization(sim)

        # Perform per layer analysis by enabling its quantizers (OPTION-1).
        self._perform_per_layer_analysis_by_enabling_quantizers(sim, results_dir)

        # Perform per layer analysis by disabling its quantizers (OPTION-2).
        self._perform_per_layer_analysis_by_disabling_quantizers(sim, results_dir)

    def create_quantsim_and_encodings(self, quant_scheme: QuantScheme, default_param_bw: int,
                                       default_activation_bw: int, config_file: str) \
            -> QuantizationSimModel:
        """
        Create Quantsim and compute encodings.

        :param quant_scheme: Quantization scheme.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_activation_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param config_file: Path to configuration file for model quantizers.
        :return: Quantsim model.
        """
        _ = fold_all_batch_norms_to_weight(self._onnx_model)
        kwargs = dict(
            quant_scheme=quant_scheme,
            default_activation_bw=default_activation_bw,
            default_param_bw=default_param_bw,
            config_file=config_file,
        )
        sim = QuantizationSimModel(copy.deepcopy(self._onnx_model), self._dummy_input, **kwargs)
        sim.compute_encodings(self._forward_pass_callback.func, self._forward_pass_callback.args)
        return sim

    def check_model_sensitivity_to_quantization(self,
                                                sim: QuantizationSimModel,
                                                ) -> Tuple[float, float, float]:
        """
        Perform the sensitivity analysis to weight and activation quantization
        individually.

        :param sim: Quantsim model.
        :return: FP32 eval score, weight-quantized eval score, act-quantized eval score.
        """
        # pylint: disable=protected-access
        fp32_eval_score = self._eval_model(self._onnx_model.model)
        _logger.info("FP32 eval score (W32A32): %f", fp32_eval_score)

        weight_quantized_eval_score = self._eval_weight_quantized_model(sim)
        _logger.info("Weight-quantized eval score (W%dA32): %f", sim._default_param_bw,
                     weight_quantized_eval_score)

        act_quantized_eval_score = self._eval_activation_quantized_model(sim)
        _logger.info("Activation-quantized eval score (W32A%d): %f", sim._default_activation_bw,
                     act_quantized_eval_score)

        return fp32_eval_score, weight_quantized_eval_score, act_quantized_eval_score

    def _eval_model(self, model):
        """
        Evaluate the model performance.

        :param model: ONNX model to be evaluated.
        :return: Scaler value representing model performance.
        """
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(
            path_or_bytes=model.SerializeToString(),
            providers=providers,
        )
        return self._eval_callback.func(session, self._eval_callback.args)

    def _eval_weight_quantized_model(self, sim: QuantizationSimModel)-> float:
        """
        Evaluate weight quantized model performance.
        For weight quantized model performance, disable enabled activation quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        enabled_activation_quantizers = self._get_enabled_activation_quantizers(sim)
        self._enable_disable_quantizers(enabled_activation_quantizers, enabled=False)
        eval_score = self._eval_callback.func(sim.session, self._eval_callback.args)
        self._enable_disable_quantizers(enabled_activation_quantizers, enabled=True)
        return eval_score

    def _eval_activation_quantized_model(self, sim: QuantizationSimModel)-> float:
        """
        Evaluate activation quantized model performance.
        For activation quantized model performance, disable enabled param quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        enabled_param_quantizers = self._get_enabled_param_quantizers(sim)
        self._enable_disable_quantizers(enabled_param_quantizers, enabled=False)
        eval_score = self._eval_callback.func(sim.session, self._eval_callback.args)
        self._enable_disable_quantizers(enabled_param_quantizers, enabled=True)
        return eval_score

    @staticmethod
    def _get_enabled_param_quantizers(sim: QuantizationSimModel) -> List[QcQuantizeOp]:
        """
        For given quantsim model, get all enabled param quantizers.
        :param sim: Quantsim model.
        :return: List of enabled param quantizers.
        """
        param_quantizers, _ = sim.get_all_quantizers()
        enabled_param_quantizers = [quantizer for quantizer in param_quantizers if quantizer.enabled]
        return enabled_param_quantizers

    @staticmethod
    def _get_enabled_activation_quantizers(sim: QuantizationSimModel) -> List[QcQuantizeOp]:
        """
        For given quantsim model, get all enabled activation quantizers.
        :param sim: Quantsim model.
        :return: List of enabled activation quantizers.
        """
        _, act_quantizers = sim.get_all_quantizers()
        enabled_activation_quantizers = [quantizer for quantizer in act_quantizers if quantizer.enabled]
        return enabled_activation_quantizers

    @staticmethod
    def _enable_disable_quantizers(quantizers: List[QcQuantizeOp], enabled: bool):
        """
        For given list of quantizers, set (enable/disable) quantizer's enabled.

        :param quantizers: List of quantizers.
        :param enabled: Enabled flag.
        """
        for quantizer in quantizers:
            quantizer.enabled = enabled

    def _perform_per_layer_analysis_by_enabling_quantizers(self,
                                                           sim: QuantizationSimModel,
                                                           results_dir: str,
                                                           ) -> Dict:
        """
        NOTE: Option 1

        1. All parameter and activation quantizers are disabled.
        2. For every layer, based on occurrence:
              i. Each layer's parameters and activations quantizers are enabled as per JSON config file
                 and set to bit-width specified.
             ii. Measure and record eval score on subset of dataset.
            iii. Disable enabled quantizers in step i.
        3. Returns dictionary containing layer name and corresponding eval score.

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        _logger.info("OPTION-1: All the quantizers are disabled. "
                     "Starting per-layer analysis by enabling layer-specific quantizers as per config file.")
        layer_wise_eval_score_dict = self._perform_per_layer_analysis(sim,
                                                                      disable_all_quantizers=True,
                                                                      enabled_before=True,
                                                                      enabled_after=False)
        export_per_layer_sensitivity_analysis_plot(layer_wise_eval_score_dict,
                                                   results_dir,
                                                   title="per_layer_quant_enabled")
        save_json(layer_wise_eval_score_dict,
                  results_dir,
                  title="per_layer_quant_enabled.json")
        _logger.info("Exported per-layer quant analysis (enabled) plot.")
        return layer_wise_eval_score_dict

    def _perform_per_layer_analysis_by_disabling_quantizers(self,
                                                            sim: QuantizationSimModel,
                                                            results_dir: str,
                                                            ) -> Dict:
        """
        NOTE: Option 2

        1. All parameter and activation quantizers are enabled as per JSON config file
        and set to bit-width specified.
        2. For every layer, based on occurrence:
              i. Each layer's parameters and activations quantizers are disabled.
             ii. Measure and record eval score on subset of dataset.
            iii. Enable disabled quantizers in step i.
        3. Returns dictionary containing layer name and corresponding eval score.

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        _logger.info("OPTION-2: All the quantizers are enabled as per config file. "
                     "Starting per-layer analysis by disabling layer-specific quantizers.")
        layer_wise_eval_score_dict = self._perform_per_layer_analysis(sim,
                                                                      disable_all_quantizers=False,
                                                                      enabled_before=False,
                                                                      enabled_after=True)
        export_per_layer_sensitivity_analysis_plot(layer_wise_eval_score_dict,
                                                   results_dir,
                                                   title="per_layer_quant_disabled")
        save_json(layer_wise_eval_score_dict,
                  results_dir,
                  title="per_layer_quant_disabled.json")
        _logger.info("Exported per-layer quant analysis (disabled) plot.")
        return layer_wise_eval_score_dict

    def _perform_per_layer_analysis(self,
                                    sim: QuantizationSimModel,
                                    disable_all_quantizers: bool,
                                    enabled_before: bool,
                                    enabled_after: bool,
                                    ) -> Dict:
        """
        Helper function for perform_per_layer_analysis_by_enabling_quant_wrappers() and
        perform_per_layer_analysis_by_disabling_quant_wrappers()

        :param sim: Quantsim model.
        :param disable_all_quantizers: Flag to disable all the quantizers before per-layer analysis.
        :param enabled_before: Flag to set enabled for quantizers before computing encodings.
        :param enabled_after: Flag to set enabled for quantizers after computing encodings.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score.
        """

        # Maps op to its list of enabled quantizers.
        op_to_quantizers_dict = self._get_enabled_quantizers(sim)

        if disable_all_quantizers:
            for enabled_quantizers in op_to_quantizers_dict.values():
                self._enable_disable_quantizers(enabled_quantizers, enabled=False)

        eval_score_dict = {}
        for op_name, enabled_quantizers in op_to_quantizers_dict.items():
            self._enable_disable_quantizers(enabled_quantizers, enabled=enabled_before)

            # Record eval score.
            eval_score_dict[op_name] = self._eval_callback.func(sim.session, self._eval_callback.args)
            _logger.debug("For layer: %s, the eval score is: %f", op_name, eval_score_dict[op_name])

            self._enable_disable_quantizers(enabled_quantizers, enabled=enabled_after)

        if disable_all_quantizers:
            for enabled_quantizers in op_to_quantizers_dict.values():
                self._enable_disable_quantizers(enabled_quantizers, enabled=True)

        return eval_score_dict

    @staticmethod
    def _get_enabled_quantizers(sim: QuantizationSimModel):
        enabled_quant_wrappers = defaultdict(list)
        cg_ops = sim.connected_graph.ordered_ops
        for op in cg_ops:
            # Get param quantizers
            for param in op.parameters:
                if param in sim.qc_quantize_op_dict and sim.qc_quantize_op_dict[param].enabled:
                    enabled_quant_wrappers[op.name_op].append(sim.qc_quantize_op_dict[param])
            # Get output activation quantizers
            if op.output_ops and op.output_ops[0].type == 'branch':
                cg_product = op.output_ops[0].output
            else:
                cg_product = op.output
            for output_name in set(cg_product.tensor_dict.values()):
                if output_name in sim.qc_quantize_op_dict and sim.qc_quantize_op_dict[output_name].enabled:
                    enabled_quant_wrappers[op.name_op].append(sim.qc_quantize_op_dict[output_name])
            # Get input activation quantizers if starting op
            if op in sim.connected_graph.starting_ops:
                cg_products = [cg_product for cg_product in op.inputs if cg_product.is_model_input]
                for cg_product in cg_products:
                    assert len(cg_product.tensor_dict) == 1
                    input_name = list(cg_product.tensor_dict.values())[0]
                    if input_name in sim.qc_quantize_op_dict and sim.qc_quantize_op_dict[input_name].enabled:
                        enabled_quant_wrappers[op.name_op].append(sim.qc_quantize_op_dict[input_name])
        return enabled_quant_wrappers
