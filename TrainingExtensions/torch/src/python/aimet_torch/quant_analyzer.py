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

import os
import shutil
from collections import OrderedDict, defaultdict
from typing import Union, Tuple, Callable, Dict, List, Any
from bokeh import plotting
from bokeh.models import ColumnDataSource, Band, Span, tickers
import torch

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_torch.utils import in_eval_mode, run_hook_for_layers_with_given_input, is_leaf_module
from aimet_torch.tensor_quantizer import TensorQuantizer, StaticGridTensorQuantizer
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch.qc_quantize_recurrent import QcQuantizeRecurrent
from aimet_torch.quantsim import QuantizationSimModel

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

DEFAULT_BOKEH_FIGURE_HEIGHT = 300


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

    def _eval_weight_quantized_model(
            self,
            sim: QuantizationSimModel,
    )-> float:
        """
        Evaluate weight quantized model performance.
        For weight quantized model performance, disable enabled activation quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        enabled_activation_quantizers = self._get_enabled_activation_quantizers(sim)
        self._enable_quantizers(enabled_activation_quantizers, enabled=False)
        eval_score = self._eval_model(sim.model)
        self._enable_quantizers(enabled_activation_quantizers, enabled=True)
        return eval_score

    def _eval_activation_quantized_model(
            self,
            sim: QuantizationSimModel,
    )-> float:
        """
        Evaluate activation quantized model performance.
        For activation quantized model performance, disable enabled param quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        enabled_param_quantizers = self._get_enabled_param_quantizers(sim)
        self._enable_quantizers(enabled_param_quantizers, enabled=False)
        eval_score = self._eval_model(sim.model)
        self._enable_quantizers(enabled_param_quantizers, enabled=True)
        return eval_score

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
        :return: Ordered dictionary which maps wrapped module name to quant wrapper.
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
    def _get_enabled_quantizers(sorted_quant_wrappers: Dict)\
            -> Dict[Union[QcQuantizeWrapper, QcQuantizeRecurrent], List[TensorQuantizer]]:
        """
        For given sorted quant wrappers dict, get enabled quantizers.

        :param sorted_quant_wrappers: Dictionary containing quant wrappers sorted based on occurrence.
        :return: Dictionary which maps a quant wrapper to a list of enabled quantizers in it.
        """
        enabled_quant_wrappers = defaultdict(list)

        for quant_wrapper in sorted_quant_wrappers.values():
            for quantizer in quant_wrapper.param_quantizers.values():
                if quantizer.enabled:
                    enabled_quant_wrappers[quant_wrapper].append(quantizer)
            for quantizer in quant_wrapper.output_quantizers:
                if quantizer.enabled:
                    enabled_quant_wrappers[quant_wrapper].append(quantizer)
            for quantizer in quant_wrapper.input_quantizers:
                if quantizer.enabled:
                    enabled_quant_wrappers[quant_wrapper].append(quantizer)

        return enabled_quant_wrappers

    @staticmethod
    def _get_enabled_param_quantizers(sim: QuantizationSimModel) -> List[TensorQuantizer]:
        """
        For given quantsim model, get all enabled param quantizers.
        :param sim: Quantsim model.
        :return: List of enabled param quantizers.
        """
        enabled_param_quantizers = []
        for quant_wrapper in sim.quant_wrappers():
            for quantizer in quant_wrapper.param_quantizers.values():
                if quantizer.enabled:
                    enabled_param_quantizers.append(quantizer)

        return enabled_param_quantizers

    @staticmethod
    def _get_enabled_activation_quantizers(sim: QuantizationSimModel) -> List[TensorQuantizer]:
        """
        For given quantsim model, get all enabled activation quantizers.
        :param sim: Quantsim model.
        :return: List of enabled activation quantizers.
        """
        enabled_activation_quantizers = []
        for quant_wrapper in sim.quant_wrappers():
            for quantizer in quant_wrapper.input_quantizers:
                if quantizer.enabled:
                    enabled_activation_quantizers.append(quantizer)
            for quantizer in quant_wrapper.output_quantizers:
                if quantizer.enabled:
                    enabled_activation_quantizers.append(quantizer)

        return enabled_activation_quantizers

    @staticmethod
    def _enable_quantizers(quantizers: List[TensorQuantizer], enabled: bool) -> None:
        """
        For given list of quantizers, set (enable/disable) quantizer's enabled.

        :param quantizers: List of quantizers.
        :param enabled: Enabled flag.
        """
        for quantizer in quantizers:
            quantizer.enabled = enabled

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
        # Sorted quant wrappers based on occurrence.
        # maps wrapped module name to a quant wrapper.
        sorted_quant_wrappers = self._sort_quant_wrappers_based_on_occurrence(sim)

        # Enabled quant wrappers.
        # maps quant wrapper to a list of enabled quantizers in it.
        enabled_quant_wrappers = self._get_enabled_quantizers(sorted_quant_wrappers)

        if disable_all_quantizers:
            for enabled_quantizers in enabled_quant_wrappers.values():
                self._enable_quantizers(enabled_quantizers, enabled=False)

        eval_score_dict = {}
        for name, quant_wrapper in sorted_quant_wrappers.items():
            if quant_wrapper in enabled_quant_wrappers:
                enabled_quantizers = enabled_quant_wrappers[quant_wrapper]
                self._enable_quantizers(enabled_quantizers, enabled=enabled_before)

                # Record eval score.
                eval_score_dict[name] = self._eval_model(sim.model)
                _logger.info("For layer: %s, the eval score is: %.02f", name, eval_score_dict[name])

                self._enable_quantizers(enabled_quantizers, enabled=enabled_after)

        if disable_all_quantizers:
            for enabled_quantizers in enabled_quant_wrappers.values():
                self._enable_quantizers(enabled_quantizers, enabled=True)

        return eval_score_dict

    @staticmethod
    def _export_per_layer_sensitivity_analysis_plot(layer_wise_eval_score_dict: Dict, results_dir: str,
                                                    title: str) -> plotting.Figure:
        """
        Export per layer sensitivity analysis in html format.

        :param layer_wise_eval_score_dict: layer wise eval score dictionary. dict[layer_name] = eval_score.
        :param results_dir: Directory to save the results.
        :param title: Title of the plot.
        """
        layer_names = []
        eval_scores = []
        for layer_name, eval_score in layer_wise_eval_score_dict.items():
            layer_names.append(layer_name)
            eval_scores.append(eval_score)

        # Configure the output file to be saved.
        filename = os.path.join(results_dir, f"{title}.html")
        plotting.output_file(filename)
        plot = plotting.figure(x_range=layer_names,
                               plot_height=DEFAULT_BOKEH_FIGURE_HEIGHT,
                               title=title,
                               x_axis_label="Layers",
                               y_axis_label="Eval score")
        plot.line(x=layer_names, y=eval_scores)
        plot.y_range.start = 0
        plot.xaxis.major_label_orientation = "vertical"
        plot.sizing_mode = "scale_width"
        plotting.save(plot)
        return plot

    @staticmethod
    def _export_stats_histogram_plot(histogram: List, encoding, results_dir: str, title: str) -> plotting.Figure:
        """
        Export histogram (PDF) of statistics with overlaying encoding min and max
        values in html format.

        :param histogram: List of buckets where each bucket is (xLeft, PDF).
        :param encoding: Encoding.
        :param results_dir: Directory to save the results.
        :param title: Title of the plot.
        :return: Histogram plot.
        """
        entries = []
        pdfs = []
        for entry, pdf in histogram:
            entries.append(entry)
            pdfs.append(pdf)

        # Configure the output file to be saved.
        filename = os.path.join(results_dir, f"{title}.html")
        plotting.output_file(filename)
        plot = plotting.figure(plot_height=DEFAULT_BOKEH_FIGURE_HEIGHT,
                               title=title)
        # Add line and underlying color for histogram.
        plot_source = ColumnDataSource(data=dict(entries=entries, pdfs=pdfs))
        plot.line("entries", "pdfs", source=plot_source, color="blue", legend="PDF")
        band = Band(base='entries', upper='pdfs', source=plot_source, level='underlay', fill_color='blue')
        plot.add_layout(band)

        # Overlay encoding min and max values.
        line = Span(location=encoding.min, dimension='height', line_color='green', line_dash='dashed')
        plot.line([], [], line_dash='dashed', line_color="green", legend='MIN_VAL')
        plot.add_layout(line)
        line = Span(location=encoding.max, dimension='height', line_color='red', line_dash='dashed')
        plot.line([], [], line_dash='dashed', line_color="red", legend='MAX_VAL')
        plot.add_layout(line)

        plotting.save(plot)
        return plot

    @staticmethod
    def _export_per_layer_min_max_ranges_plot(layer_wise_min_max_ranges_dict: Dict, results_dir: str, title: str) \
            -> plotting.Figure:
        """
        Export per layer encoding min-max range in html format.

        :param layer_wise_min_max_ranges_dict: layer wise eval score dictionary.
         dict[layer_name] = (encoding min, encoding max)
        :param results_dir:  Directory to save the results.
        :param title: Title of the plot.
        :return: Encoding min-max range plot.
        """
        layer_names = []
        enc_min_values = []
        enc_max_values = []
        for layer_name, (enc_min, enc_max) in layer_wise_min_max_ranges_dict.items():
            layer_names.append(layer_name)
            enc_min_values.append(enc_min)
            enc_max_values.append(enc_max)

        # Configure the output file to be saved.
        filename = os.path.join(results_dir, f"{title}.html")
        plotting.output_file(filename)
        plot = plotting.figure(x_range=layer_names,
                               plot_height=DEFAULT_BOKEH_FIGURE_HEIGHT,
                               title=title)
        plot.vbar(x=layer_names, width=0.2, bottom=enc_min_values, top=enc_max_values)
        plot.xaxis.major_label_orientation = "vertical"
        plot.sizing_mode = "scale_width"
        plot.yaxis.ticker = tickers.SingleIntervalTicker(interval=1)
        plotting.save(plot)
        return plot

    @staticmethod
    def _export_per_layer_mse_plot(mse_loss_dict: Dict, results_dir: str, title: str) -> plotting.Figure:
        """
        Export per layer MSE loss between fp32 and quantized output activations in html format.

        :param mse_loss_dict: layer wise MSE loss.
        :param results_dir:  Directory to save the results.
        :param title: Title of the plot.
        :return: Layer-wise MSE loss plot.
        """
        layer_names = []
        mse_losses = []
        for layer_name, mse_loss in mse_loss_dict.items():
            layer_names.append(layer_name)
            mse_losses.append(mse_loss)

        # Configure the output file to be saved.
        filename = os.path.join(results_dir, f"{title}.html")
        plotting.output_file(filename)
        plot = plotting.figure(x_range=layer_names,
                               plot_height=DEFAULT_BOKEH_FIGURE_HEIGHT,
                               title=title,
                               x_axis_label="Layers",
                               y_axis_label="MSE loss")
        plot.circle(x=layer_names, y=mse_losses, size=10)
        plot.line(x=layer_names, y=mse_losses)
        plot.xaxis.major_label_orientation = "vertical"
        plot.sizing_mode = "scale_width"
        plotting.save(plot)
        return plot

    def _create_and_export_stats_histogram_plot(
            self,
            quantizer: StaticGridTensorQuantizer,
            results_dir: str,
            title: str,
    ) -> None:
        """
        For given quantizer, create and export histogram (PDF) of statistics in html format.

        :param quantizer: Quantizer.
        :param results_dir: Directory to save the results.
        :param title: Title of the plot.
        """
        os.makedirs(results_dir, exist_ok=True)

        histograms = quantizer.get_stats_histogram()
        encodings = quantizer.encoding
        if not isinstance(encodings, List):
            encodings = [encodings]

        for index, (histogram, encoding) in enumerate(zip(histograms, encodings)):
            filename_suffix = f"{title}_{index}"
            self._export_stats_histogram_plot(histogram, encoding, results_dir, filename_suffix)

    def _collect_and_save_output_acts(
            self,
            model: torch.nn.Module,
            module_type_for_attaching_hook: Any,
            leaf_module_only: bool,
            results_dir: str,
    ) -> None:
        """
        Collect and save output activations for given model.

        :param model: Model to attach hooks.
        :param module_type_for_attaching_hook: torch.nn module types for which hook has to be attached.
        :param leaf_module_only: Flag to set only leaf modules vs all modules to attach hooks.
        :param results_dir: Directory to save the results.
        """
        def _hook_to_tap_activations(module, _, out):
            """
            Hook-function to tap activation data.
            :param module: torch.nn.Module type module.
            :param _: Input activations to module.
            :param out: Output activations from module.
            """
            filename = os.path.join(results_dir, module_to_name_dict[module])
            with open(filename, 'wb') as f:
                torch.save(out.cpu(), f)

        os.makedirs(results_dir, exist_ok=True)
        hooks = []
        module_to_name_dict = {}
        for name, module in model.named_modules():
            module_to_name_dict[module] = name

        modules = [module for module in model.modules() if not leaf_module_only or is_leaf_module(module)]
        if module_type_for_attaching_hook:
            modules = [module for module in modules if isinstance(module, module_type_for_attaching_hook)]
        for module in modules:
            hooks.append(module.register_forward_hook(_hook_to_tap_activations))

        # Forward pass.
        self._eval_model(model)

        # Remove all the added hooks from the model.
        for hook in hooks:
            hook.remove()

    @staticmethod
    def _load_out_acts(results_dir: str, name: str) -> torch.Tensor:
        """
        Load output activations from results_dir for given name.
        :param results_dir: Directory to load the output activations.
        :param name: Name of the file.
        :return: Output activations tensor.
        """
        filename = os.path.join(results_dir, name)
        with open(filename, 'rb') as f:
            out_acts = torch.load(f)
        return out_acts

    def _check_model_sensitivity_to_quantization(
            self,
            sim: QuantizationSimModel,
    ) -> Tuple[float, float, float]:
        """
        Perform the sensitivity analysis to weight and activation quantization
        individually.

        :param sim: Quantsim model.
        :return: FP32 eval score, weight-quantized eval score, act-quantized eval score.
        """
        # pylint: disable=protected-access
        fp32_eval_score = self._eval_model(self._model)
        _logger.info("FP32 eval score (W32A32): %.02f", fp32_eval_score)

        weight_quantized_eval_score = self._eval_weight_quantized_model(sim)
        _logger.info("Weight-quantized eval score (W%dA32): %.02f", sim._default_param_bw,
                     weight_quantized_eval_score)

        act_quantized_eval_score = self._eval_activation_quantized_model(sim)
        _logger.info("Activation-quantized eval score (W32A%d): %.02f", sim._default_output_bw,
                     act_quantized_eval_score)

        return fp32_eval_score, weight_quantized_eval_score, act_quantized_eval_score

    def _perform_per_layer_analysis_by_enabling_quant_wrappers(
            self,
            sim: QuantizationSimModel,
            results_dir: str = "./tmp/",
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

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        _logger.info("\nOPTION-1:\nAll the quant wrappers are disabled.\n"
                     "Starting per-layer analysis by enabling quant wrappers as per config file.")
        layer_wise_eval_score_dict = self._perform_per_layer_analysis(sim,
                                                                      disable_all_quantizers=True,
                                                                      enabled_before=True,
                                                                      enabled_after=False)
        self._export_per_layer_sensitivity_analysis_plot(layer_wise_eval_score_dict,
                                                         results_dir,
                                                         title="per_layer_quant_enabled")
        return layer_wise_eval_score_dict

    def _perform_per_layer_analysis_by_disabling_quant_wrappers(
            self,
            sim: QuantizationSimModel,
            results_dir: str = "./tmp/",
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

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise eval score dictionary. dict[layer_name] = eval_score
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        _logger.info("\nOPTION-2:\nAll the quant wrappers are enabled as per config file.\n"
                     "Starting per-layer analysis by disabling quant wrappers.")
        layer_wise_eval_score_dict = self._perform_per_layer_analysis(sim,
                                                                      disable_all_quantizers=False,
                                                                      enabled_before=False,
                                                                      enabled_after=True)
        self._export_per_layer_sensitivity_analysis_plot(layer_wise_eval_score_dict,
                                                         results_dir,
                                                         title="per_layer_quant_disabled")
        return layer_wise_eval_score_dict

    def _export_per_layer_encoding_min_max_range(
            self,
            sim: QuantizationSimModel,
            results_dir: str = "./tmp/",
    ) -> Tuple[Dict, Dict]:
        """
        Export encoding min and max range for all weights and activations.

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise min-max range for weights and activations.
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        _logger.info("\nExporting per layer encoding min-max ranges.")
        module_to_name_dict = {}
        for name, module in sim.model.named_modules():
            module_to_name_dict[module] = name

        min_max_range_for_activations_dict = {}
        min_max_range_for_weights_dict = {}
        for quant_wrapper in sim.quant_wrappers():
            wrapped_module_name = module_to_name_dict[quant_wrapper]
            for index, quantizer in enumerate(quant_wrapper.input_quantizers):
                if quantizer.encoding:
                    name = f"{wrapped_module_name}_input_{index}"
                    min_max_range_for_activations_dict[name] = (quantizer.encoding.min, quantizer.encoding.max)
            for index, quantizer in enumerate(quant_wrapper.output_quantizers):
                if quantizer.encoding:
                    name = f"{wrapped_module_name}_output_{index}"
                    min_max_range_for_activations_dict[name] = (quantizer.encoding.min, quantizer.encoding.max)
            for param_name, quantizer in quant_wrapper.param_quantizers.items():
                if quantizer.encoding:
                    # TODO: Add support for per channel quantization.
                    name = f"{wrapped_module_name}_{param_name}"
                    min_max_range_for_weights_dict[name] = (quantizer.encoding.min, quantizer.encoding.max)

        self._export_per_layer_min_max_ranges_plot(min_max_range_for_weights_dict,
                                                   results_dir,
                                                   title="min_max_range_all_weights")
        self._export_per_layer_min_max_ranges_plot(min_max_range_for_activations_dict,
                                                   results_dir,
                                                   title="min_max_range_all_activations")

        return min_max_range_for_weights_dict, min_max_range_for_activations_dict

    def _export_per_layer_stats_histogram(
            self,
            sim: QuantizationSimModel,
            results_dir: str = "./tmp/",
    ) -> None:
        """
        NOTE: Not to invoke when quantization scheme is not TF-Enhanced.

        Export histogram that represents a PDF of collected statistics by a quantizer for every
        quant wrapper. After invoking this API, results_dir should have html files in following
        format for every quantizers of quant wrappers.

        -results_dir
            -activations_pdf
                name_{input/output}_{index}.html
            -weights_pdf
                -name
                    param_name_{channel_index}.html

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        """
        weights_pdf_dir = os.path.join(results_dir, "weights_pdf")
        activations_pdf_dir = os.path.join(results_dir, "activations_pdf")

        _logger.info("\nExporting per layer stats histogram.")
        module_to_name_dict = {}
        for name, module in sim.model.named_modules():
            module_to_name_dict[module] = name

        for quant_wrapper in sim.quant_wrappers():
            wrapped_module_name = module_to_name_dict[quant_wrapper]
            for quantizer in quant_wrapper.input_quantizers:
                if quantizer.encoding:
                    self._create_and_export_stats_histogram_plot(quantizer,
                                                                 activations_pdf_dir,
                                                                 title=f"{wrapped_module_name}_input")
            for quantizer in quant_wrapper.output_quantizers:
                if quantizer.encoding:
                    self._create_and_export_stats_histogram_plot(quantizer,
                                                                 activations_pdf_dir,
                                                                 title=f"{wrapped_module_name}_output")
            for param_name, quantizer in quant_wrapper.param_quantizers.items():
                if quantizer.encoding:
                    self._create_and_export_stats_histogram_plot(quantizer,
                                                                 os.path.join(weights_pdf_dir, wrapped_module_name),
                                                                 title=f"{wrapped_module_name}_{param_name}")
            _logger.info("Exported stats histogram for layer: %s", wrapped_module_name)

    def _export_per_layer_mse_loss(
            self,
            sim: QuantizationSimModel,
            results_dir: str = "./tmp/",
    ) -> None:
        """
        NOTE: Need to pass same model input data through both fp32 and quantsim model to
        tap output activations of each layer.

        Export MSE loss between fp32 and quantized output activations for each layer.
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        fp32_out_acts_dir = os.path.join(results_dir, "fp32_out_acts")
        quantized_out_acts_dir = os.path.join(results_dir, "quantized_out_acts")

        _logger.info("\nExporting per layer MSE loss.")
        self._collect_and_save_output_acts(self._model,
                                           module_type_for_attaching_hook=None,
                                           leaf_module_only=True,
                                           results_dir=fp32_out_acts_dir)
        self._collect_and_save_output_acts(sim.model,
                                           module_type_for_attaching_hook=(QcQuantizeWrapper,
                                                                           QcQuantizeRecurrent),
                                           leaf_module_only=False,
                                           results_dir=quantized_out_acts_dir)
        mse_loss_dict = {}
        for name in os.listdir(quantized_out_acts_dir):
            fp32_out_acts = self._load_out_acts(fp32_out_acts_dir, name)
            quantized_out_acts = self._load_out_acts(quantized_out_acts_dir, name)
            loss = torch.nn.functional.mse_loss(fp32_out_acts, quantized_out_acts)
            mse_loss_dict[name] = loss.item()

        # Delete temporary directories.
        shutil.rmtree(fp32_out_acts_dir, ignore_errors=True)
        shutil.rmtree(quantized_out_acts_dir, ignore_errors=True)

        self._export_per_layer_mse_plot(mse_loss_dict,
                                        results_dir,
                                        title="per_layer_mse_loss")

    def analyze(
            self,
            quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
            default_param_bw: int = 8,
            default_output_bw: int = 8,
            config_file: str = None,
            results_dir: str = "./tmp/",
    ) -> None:
        """
        Analyze model for quantization and point out sensitive parts/hotspots of the model by performing
            1) model sensitivity to quantization,
            2) perform per layer sensitivity analysis by enabling and disabling quant wrappers,
            3) export per layer statistics histogram (PDF) when quant scheme is TF-Enhanced.

        :param quant_scheme: Quantization scheme. Supported values are
                QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs.
        :param config_file: Path to configuration file for model quantizers.
        :param results_dir: Directory to save the results.
        """
        kwargs = dict(
            quant_scheme=quant_scheme,
            default_output_bw=default_output_bw,
            default_param_bw=default_param_bw,
            config_file=config_file,
        )
        sim = QuantizationSimModel(self._model, self._dummy_input, **kwargs)
        sim.compute_encodings(self._forward_pass_callback.func, self._forward_pass_callback.args)

        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        # Check model sensitivity to weight and activation quantization individually.
        self._check_model_sensitivity_to_quantization(sim)

        # Perform per layer analysis by enabling each quant wrapper (OPTION-1).
        self._perform_per_layer_analysis_by_enabling_quant_wrappers(sim, results_dir)

        # Perform per layer analysis by disabling each quant wrapper (OPTION-2).
        self._perform_per_layer_analysis_by_disabling_quant_wrappers(sim, results_dir)

        # Export encoding min-max range.
        self._export_per_layer_encoding_min_max_range(sim, results_dir)

        # Export PDF of statistics.
        if quant_scheme == QuantScheme.post_training_tf_enhanced:
            self._export_per_layer_stats_histogram(sim, results_dir)

        # Export per layer MSE loss between fp32 and quantized output activations.
        self._export_per_layer_mse_loss(sim, results_dir)
