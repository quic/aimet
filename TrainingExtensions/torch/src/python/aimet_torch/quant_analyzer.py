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

""" Quant Analyzer """

from typing import Union, Tuple, Callable
import torch

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_torch.utils import in_eval_mode
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

    @staticmethod
    def _disable_act_quantizers(sim: QuantizationSimModel) -> None:
        """
        Disable quantizers for activations for given Quantsim.
        :param sim: Quantsim model.
        """
        for module in sim.model.modules():
            if isinstance(module, (QcQuantizeWrapper, QcQuantizeRecurrent)):
                for quantizer in module.output_quantizers:
                    if quantizer.enabled:
                        quantizer.enabled = False
                for quantizer in module.input_quantizers:
                    if quantizer.enabled:
                        quantizer.enabled = False

    @staticmethod
    def _disable_param_quantizers(sim: QuantizationSimModel) -> None:
        """
        Disable quantizers for parameters for given Quantsim.
        :param sim: Quantsim model.
        """
        for module in sim.model.modules():
            if isinstance(module, (QcQuantizeWrapper, QcQuantizeRecurrent)):
                for quantizer in module.param_quantizers.values():
                    if quantizer.enabled:
                        quantizer.enabled = False

    def _eval_model_with_weight_quantized(self, **kwargs) -> float:
        """
        Analyze model sensitivity to only parameter (weight) quantization.
        Disable quantizers for activations.

        :param **kwargs: Additional arguments to the Quantsim.
        :return: Weight Quantized, Activation in float model performance.
        """
        sim = QuantizationSimModel(self._model, self._dummy_input, **kwargs)
        self._disable_act_quantizers(sim)
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
        self._disable_param_quantizers(sim)
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

    def analyze_model_sensitivity_to_quantization(
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
        """

        QuantizationSimModel._validate_quantsim_inputs(default_quant_scheme, # pylint: disable=protected-access
                                                       default_rounding_mode,
                                                       default_output_bw,
                                                       default_param_bw,
                                                       default_data_type)
        kwargs = dict(
            quant_scheme=default_quant_scheme,
            rounding_mode=default_rounding_mode,
            default_output_bw=default_output_bw,
            default_param_bw=default_param_bw,
            config_file=default_config_file,
            default_data_type=default_data_type,
        )

        fp32_acc = self._eval_model(self._model)
        _logger.info("FP32 eval score (W32A32): %.02f", fp32_acc)

        weight_quantized_acc = self._eval_model_with_weight_quantized(**kwargs)
        _logger.info("Weight-quantized eval score (W%dA32): %.02f", default_param_bw, weight_quantized_acc)

        act_quantized_acc = self._eval_model_with_act_quantized(**kwargs)
        _logger.info("Activation-quantized eval score (W32A%d): %.02f", default_output_bw, act_quantized_acc)

        return fp32_acc, weight_quantized_acc, act_quantized_acc
