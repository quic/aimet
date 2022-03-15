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

""" Quant Analyzer """

from collections import OrderedDict, defaultdict
from typing import Union, Tuple, Callable, Dict
import torch

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_torch.utils import in_eval_mode, run_hook_for_layers_with_given_input
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch.qc_quantize_recurrent import QcQuantizeRecurrent
from aimet_torch.quantsim import QuantizationSimModel

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class CallbackFunc:
    """
    Class encapsulating callback function, and it's argument(s)
    """
    def __init__(self, func: Callable, func_callback_args=None):
        """
        :param func: Callable Function
        :param func_callback_args: Arguments passed to the callable function as-is.
        """
        self.func = func
        self.args = func_callback_args


class QuantAnalyzer:
    """
    QuantAnalyzer tool provides

     1) model sensitivity to weight and activation quantization
     2) per layer sensitivity analysis
     3) per layer encoding (min - max range) and PDF analysis and
     4) per layer MSE analysis
    """
    def __init__(
            self,
            model: torch.nn.Module,
            dummy_input: Union[torch.Tensor, Tuple],
            forward_pass_callback: CallbackFunc,
            eval_callback: CallbackFunc,
    ) -> None:
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

        self._model = model
        self._dummy_input = dummy_input
        self._forward_pass_callback = forward_pass_callback
        self._eval_callback = eval_callback

    def _eval_model_with_weight_quantized(self, **kwargs) -> float:
        """
        Analyze model sensitivity to only parameter (weight) quantization.
        Disable quantizers for activations.

        :param **kwargs: Additional arguments to the Quantsim.
        :return: Weight Quantized, Activation in float model performance.
        """
        sim = QuantizationSimModel(self._model, self._dummy_input, **kwargs)
        sim.set_enabled_for_all_act_quantizers(enabled=False)
        sim.compute_encodings(self._forward_pass_callback.func, self._forward_pass_callback.args)
        acc = self._eval_model(sim.model)
        return acc

    def _eval_model_with_act_quantized(self, **kwargs) -> float:
        """
        Analyze model sensitivity to only activation quantization.
        Disable quantizers for parameters.

        :param **kwargs: Additional arguments to the quantsim.
        :return: Activations Quantized, Weights in float model performance.
        """
        sim = QuantizationSimModel(self._model, self._dummy_input, **kwargs)
        sim.set_enabled_for_all_param_quantizers(enabled=False)
        sim.compute_encodings(self._forward_pass_callback.func, self._forward_pass_callback.args)
        acc = self._eval_model(sim.model)
        return acc

    def _eval_model(self, model: torch.nn.Module) -> float:
        """
        Evaluate the model performance.

        :param model: PyTorch model to be evaluated.
        :return: Scaler value representing model performance.
        """
        with in_eval_mode(model), torch.no_grad():
            return self._eval_callback.func(model, self._eval_callback.args)

    def _sort_quant_wrappers_based_on_occurrence(self, sim: QuantizationSimModel) -> Dict:
        """
        Sort quant wrappers based on occurrence for given quantsim model.

        :param sim: Quantsim model.
        :return: Dictionary containing sorted quant wrappers. dict[name] = Quant wrapper.
        """
        def sorting_hook(quant_wrapper: torch.nn.Module, *_):
            """
            Hook-function to sort quant wrappers based on occurrence.

            :param quant_wrapper: Quant wrapper.
            :param _: Additional args.
            """
            quant_wrapper_name = module_to_name_dict[quant_wrapper]
            sorted_quant_wrappers_dict[quant_wrapper_name] = quant_wrapper

        module_to_name_dict = {}
        for name, module in sim.model.named_modules():
            module_to_name_dict[module] = name

        sorted_quant_wrappers_dict = OrderedDict()
        run_hook_for_layers_with_given_input(sim.model, self._dummy_input, sorting_hook,
                                             module_type_for_attaching_hook=(QcQuantizeWrapper, QcQuantizeRecurrent),
                                             leaf_node_only=False)
        return sorted_quant_wrappers_dict

    @staticmethod
    def _get_enabled_info_for_all_quantizers(sorted_quant_wrappers: Dict) -> Dict:
        """
        Collect enabled information for all the quantizers.

        :param sorted_quant_wrappers: Dictionary containing quant wrappers sorted based on occurrence.
        :return: Nested dictionary containing layer wise enabled info for all quantizers.
        """
        layer_wise_enabled_for_all_quantizers = {}
        for name, quant_wrapper in sorted_quant_wrappers.items():
            all_quantizers = {}
            param_quantizers = {}
            for param_name, quantizer in quant_wrapper.param_quantizers.items():
                param_quantizers[param_name] = quantizer.enabled
            all_quantizers["param_quantizers"] = param_quantizers

            output_quantizers = {}
            for index, quantizer in enumerate(quant_wrapper.output_quantizers):
                output_quantizers[index] = quantizer.enabled
            all_quantizers["output_quantizers"] = output_quantizers

            input_quantizers = {}
            for index, quantizer in enumerate(quant_wrapper.input_quantizers):
                input_quantizers[index] = quantizer.enabled
            all_quantizers["input_quantizers"] = input_quantizers
            layer_wise_enabled_for_all_quantizers[name] = all_quantizers

        return layer_wise_enabled_for_all_quantizers

    @staticmethod
    def _revert_enabled_for_quantizers(
            quant_wrapper: Union[QcQuantizeWrapper, QcQuantizeRecurrent],
            modified_quantizers: Dict,
            enabled: bool,
    ) -> None:
        """
        For given quant wrapper, revert enabled flag for modified quantizers.

        :param quant_wrapper: Quant wrapper.
        :param modified_quantizers: Dictionary of quantizers' names whose enabled flag is modified.
        :param enabled: Enabled flag.
        """
        for param_name, quantizer in quant_wrapper.param_quantizers.items():
            if param_name in modified_quantizers["param_quantizers"]:
                quantizer.enabled = enabled
        for index, quantizer in enumerate(quant_wrapper.input_quantizers):
            if index in modified_quantizers["input_quantizers"]:
                quantizer.enabled = enabled
        for index, quantizer in enumerate(quant_wrapper.output_quantizers):
            if index in modified_quantizers["output_quantizers"]:
                quantizer.enabled = enabled

    @staticmethod
    def _set_enabled_for_quantizers(
            quant_wrapper: Union[QcQuantizeWrapper, QcQuantizeRecurrent],
            name: str,
            modified_quantizers: Dict,
            enabled_for_all_quantizers: Dict,
            enabled: bool,
    ) -> None:
        """
        For given quant wrapper, set enabled for its parameters and activation quantizers.
        Also keep track of wrapper's quantizers whose enabled flag is modified/set.

        :param quant_wrapper: Quant wrapper.
        :param name: Wrapped module name.
        :param modified_quantizers: Dictionary of quantizers' names whose enabled flag is modified.
        :param enabled_for_all_quantizers: Nested dictionary containing layer wise enabled info for all quantizers.
        :param enabled: Enabled flag.
        """
        for param_name, quantizer in quant_wrapper.param_quantizers.items():
            if enabled_for_all_quantizers[name]["param_quantizers"][param_name]:
                quantizer.enabled = enabled
                modified_quantizers["param_quantizers"].append(param_name)
        for index, quantizer in enumerate(quant_wrapper.input_quantizers):
            if enabled_for_all_quantizers[name]["input_quantizers"][index]:
                quantizer.enabled = enabled
                modified_quantizers["input_quantizers"].append(index)
        for index, quantizer in enumerate(quant_wrapper.output_quantizers):
            if enabled_for_all_quantizers[name]["output_quantizers"][index]:
                quantizer.enabled = enabled
                modified_quantizers["output_quantizers"].append(index)

    def _perform_per_layer_analysis(
            self,
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
        sorted_quant_wrappers = self._sort_quant_wrappers_based_on_occurrence(sim)

        # Get enabled flag info for all the quantizers as per set by user's JSON config file.
        enabled_for_all_quantizers = self._get_enabled_info_for_all_quantizers(sorted_quant_wrappers)

        if disable_all_quantizers:
            sim.set_enabled_for_all_param_quantizers(enabled=False)
            sim.set_enabled_for_all_act_quantizers(enabled=False)

        layer_wise_eval_score_dict = {}
        for name, quant_wrapper in sorted_quant_wrappers.items():
            modified_quantizers = defaultdict(list)
            self._set_enabled_for_quantizers(quant_wrapper, name, modified_quantizers, enabled_for_all_quantizers,
                                             enabled=enabled_before)

            # Compute encodings and record eval score.
            sim.compute_encodings(self._forward_pass_callback.func, self._forward_pass_callback.args)
            layer_wise_eval_score_dict[name] = self._eval_model(sim.model)
            _logger.info("For layer: %s, the eval score is: %.02f", name, layer_wise_eval_score_dict[name])

            self._revert_enabled_for_quantizers(quant_wrapper, modified_quantizers, enabled=enabled_after)

        return layer_wise_eval_score_dict

    def check_model_sensitivity_to_quantization(
            self,
            default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
            default_rounding_mode: str = 'nearest',
            default_param_bw: int = 8,
            default_output_bw: int = 8,
            default_config_file: str = None,
            default_data_type: QuantizationDataType = QuantizationDataType.int,
    ) -> Tuple[float, float, float]:
        """
        Perform the sensitivity analysis to weight and activation quantization
        individually.

        :param default_quant_scheme: Quantization scheme. Supported values are
                QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced.
        :param default_rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param default_config_file: Path to configuration file for model quantizers.
        :param default_data_type: Default data type to use for quantizing all layer inputs, outputs and parameters.
                                 Possible options are QuantizationDataType.int and QuantizationDataType.float.
                                 Note that the mode default_data_type=QuantizationDataType.float is only supported with
                                 default_output_bw=16 and default_param_bw=16.
        :return: FP32 eval score, weight-quantized eval score, act-quantized eval score.
        """
        kwargs = dict(
            quant_scheme=default_quant_scheme,
            rounding_mode=default_rounding_mode,
            default_output_bw=default_output_bw,
            default_param_bw=default_param_bw,
            config_file=default_config_file,
            default_data_type=default_data_type,
        )
        fp32_eval_score = self._eval_model(self._model)
        _logger.info("FP32 eval score (W32A32): %.02f", fp32_eval_score)

        weight_quantized_eval_score = self._eval_model_with_weight_quantized(**kwargs)
        _logger.info("Weight-quantized eval score (W%dA32): %.02f", default_param_bw, weight_quantized_eval_score)

        act_quantized_eval_score = self._eval_model_with_act_quantized(**kwargs)
        _logger.info("Activation-quantized eval score (W32A%d): %.02f", default_output_bw, act_quantized_eval_score)

        return fp32_eval_score, weight_quantized_eval_score, act_quantized_eval_score

    def perform_per_layer_analysis_by_enabling_quant_wrappers(
            self,
            default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
            default_rounding_mode: str = 'nearest',
            default_param_bw: int = 8,
            default_output_bw: int = 8,
            default_config_file: str = None,
            default_data_type: QuantizationDataType = QuantizationDataType.int,
    ) -> Dict:
        """
        NOTE: Option 1

        1. All quant wrappers' parameters and activations quantizers are disabled.
        2. For every quant wrappers, based on occurrence:
              i. Each quant wrapper's parameters and activations quantizers are enabled as per JSON config file
                 and set to bit-width specified.
             ii. Measure and record eval score on subset of dataset.
            iii. Disable enabled quantizers in step i.
        3. Returns dictionary containing quant wrapper name and corresponding eval score.

        :param default_quant_scheme: Quantization scheme. Supported values are
                QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced.
        :param default_rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param default_config_file: Path to configuration file for model quantizers.
        :param default_data_type: Default data type to use for quantizing all layer inputs, outputs and parameters.
                                 Possible options are QuantizationDataType.int and QuantizationDataType.float.
                                 Note that the mode default_data_type=QuantizationDataType.float is only supported with
                                 default_output_bw=16 and default_param_bw=16.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score
        """
        kwargs = dict(
            quant_scheme=default_quant_scheme,
            rounding_mode=default_rounding_mode,
            default_output_bw=default_output_bw,
            default_param_bw=default_param_bw,
            config_file=default_config_file,
            default_data_type=default_data_type,
        )
        sim = QuantizationSimModel(self._model, self._dummy_input, **kwargs)

        _logger.info("OPTION-1: Starting per-layer analysis by enabling quant wrappers.")
        layer_wise_eval_score_dict = self._perform_per_layer_analysis(sim,
                                                                      disable_all_quantizers=True,
                                                                      enabled_before=True,
                                                                      enabled_after=False)
        return layer_wise_eval_score_dict

    def perform_per_layer_analysis_by_disabling_quant_wrappers(
            self,
            default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
            default_rounding_mode: str = 'nearest',
            default_param_bw: int = 8,
            default_output_bw: int = 8,
            default_config_file: str = None,
            default_data_type: QuantizationDataType = QuantizationDataType.int,
    ) -> Dict:
        """
        NOTE: Option 2

        1. All quant wrappers' parameters and activations quantizers are enabled as per JSON config file
        and set to bit-width specified.
        2. For every quant wrappers, based on occurrence:
              i. Each quant wrapper's parameters and activations quantizers are disabled.
             ii. Measure and record eval score on subset of dataset.
            iii. Enable disabled quantizers in step i.
        3. Returns dictionary containing quant wrapper name and corresponding eval score.

        :param default_quant_scheme: Quantization scheme. Supported values are
                QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced.
        :param default_rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param default_config_file: Path to configuration file for model quantizers.
        :param default_data_type: Default data type to use for quantizing all layer inputs, outputs and parameters.
                                 Possible options are QuantizationDataType.int and QuantizationDataType.float.
                                 Note that the mode default_data_type=QuantizationDataType.float is only supported with
                                 default_output_bw=16 and default_param_bw=16.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score
        """
        kwargs = dict(
            quant_scheme=default_quant_scheme,
            rounding_mode=default_rounding_mode,
            default_output_bw=default_output_bw,
            default_param_bw=default_param_bw,
            config_file=default_config_file,
            default_data_type=default_data_type,
        )
        sim = QuantizationSimModel(self._model, self._dummy_input, **kwargs)

        _logger.info("OPTION-2: Starting per-layer analysis by disabling quant wrappers.")
        layer_wise_eval_score_dict = self._perform_per_layer_analysis(sim,
                                                                      disable_all_quantizers=False,
                                                                      enabled_before=False,
                                                                      enabled_after=True)
        return layer_wise_eval_score_dict
