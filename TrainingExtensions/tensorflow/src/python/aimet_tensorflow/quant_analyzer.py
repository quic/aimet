# /usr/bin/env python3.8
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
from typing import List, Tuple, Dict
import tensorflow.compat.v1 as tf
from bokeh import plotting
from bokeh.models import ColumnDataSource, Band, Span
from aimet_common.defs import QuantScheme
from aimet_common.quant_analyzer import save_json, export_per_layer_sensitivity_analysis_plot, \
    create_and_export_min_max_ranges_plot
from aimet_common.utils import AimetLogger, CallbackFunc
from aimet_tensorflow.common.operation import Op
from aimet_tensorflow.quantizer_info import QuantizerInfo
from aimet_tensorflow.quantsim import QuantizationSimModel

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

DEFAULT_BOKEH_FIGURE_HEIGHT = 300


class QuantAnalyzer:
    """
    QuantAnalyzer tool provides
        1) Model sensitivity to weight and activation quantization
        2) Per layer encoding (min - max range) and PDF analysis
        3) per op sensitivity analysis
    """

    def __init__(self, session: tf.compat.v1.Session, start_op_names: List[str], output_op_names: List[str],
                 forward_pass_callback: CallbackFunc, eval_callback: CallbackFunc, use_cuda: bool = True):
        """
        :param session: The input model as session to add quantize ops to
        :param start_op_names: List of starting op names of the model
        :param output_op_names: List of output op names of the model
        :param forward_pass_callback: A callback function that is expected to run forward passes on a session.
               This callback function should use representative data for the forward pass, so the calculated
               encodings work for all data samples. This callback internally chooses the number of data samples
               it wants to use for calculating encodings.
        :param eval_callback: A callback function for model evaluation that determines model
                performance. This callback function is expected to return scalar value
                representing the model performance evaluated against entire test/evaluation dataset.
        :param use_cuda: If True, places quantization ops on GPU. Defaults to True
        """
        if not isinstance(forward_pass_callback, CallbackFunc):
            raise ValueError('forward_pass_callback and its argument(s) are not encapsulated by CallbackFunc class.')
        if not isinstance(eval_callback, CallbackFunc):
            raise ValueError('eval_callback and its argument(s) are not encapsulated by CallbackFunc class.')

        self._session = session
        self._start_op_names = start_op_names
        self._output_op_names = output_op_names
        self._forward_pass_callback = forward_pass_callback
        self._eval_callback = eval_callback
        self._use_cuda = use_cuda
        self._default_output_bw = None
        self._default_param_bw = None

    def analyze(self,
                quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                rounding_mode: str = 'nearest',
                default_param_bw: int = 8,
                default_output_bw: int = 8,
                config_file: str = None,
                results_dir: str = "./tmp/"):
        """
        Analyze model for quantization and point out sensitive parts/hotspots of the model by performing
            1) model sensitivity to quantization
            2) export per layer encoding (min - max range)
            3) export per layer statistics histogram (PDF) when quant scheme is TF-Enhanced
            4) perform per op sensitivity analysis by enabling and disabling quant ops

        :param quant_scheme: Quantization Scheme, currently supported schemes are post_training_tf and
               post_training_tf_enhanced, defaults to post_training_tf_enhanced
        :param rounding_mode: The round scheme to used. One of: 'nearest' or 'stochastic', defaults to 'nearest'
        :param default_param_bw: bitwidth to use for parameter tensors, defaults to 8
        :param default_output_bw: bitwidth to use for activation tensors, defaults to 8
        :param config_file: Path to a config file to use to specify rules for placing quant ops in the model
        :param results_dir: Directory to save the results.
        """
        self._default_param_bw = default_param_bw
        self._default_output_bw = default_output_bw

        sim = self._create_quantsim_and_encodings(quant_scheme, rounding_mode, config_file)
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        # Check model sensitivity to weight and activation quantization individually.
        self._check_model_sensitivity_to_quantization(sim)

        # Export encoding min-max range.
        self._export_per_layer_encoding_min_max_range(sim, results_dir)

        # Export PDF of statistics.
        if quant_scheme == QuantScheme.post_training_tf_enhanced:
            self._export_per_layer_stats_histogram(sim, results_dir)

        # Perform per op analysis by enabling each quant op (OPTION-1).
        self._perform_per_op_analysis_by_enabling_quant_ops(sim, results_dir)

        # Perform per op analysis by disabling each quant op (OPTION-2).
        self._perform_per_op_analysis_by_disabling_quant_ops(sim, results_dir)

    def _create_quantsim_and_encodings(self, quant_scheme: QuantScheme, rounding_mode: str,
                                       config_file: str) -> QuantizationSimModel:
        """"
        Create Quantsim and compute encodings.

        :param quant_scheme: Quantization Scheme
        :param rounding_mode: The round scheme to used
        :param config_file: Path to a config file
        :return: Quantsim model
        """
        quant_sim_model = QuantizationSimModel(session=self._session,
                                               starting_op_names=self._start_op_names,
                                               output_op_names=self._output_op_names,
                                               quant_scheme=quant_scheme, rounding_mode=rounding_mode,
                                               default_output_bw=self._default_output_bw,
                                               default_param_bw=self._default_param_bw,
                                               use_cuda=self._use_cuda,
                                               config_file=config_file)

        quant_sim_model.compute_encodings(forward_pass_callback=self._forward_pass_callback.func,
                                          forward_pass_callback_args=self._forward_pass_callback.args)

        return quant_sim_model

    def _check_model_sensitivity_to_quantization(self, sim: QuantizationSimModel) -> Tuple[float, float, float]:
        """
        Perform the sensitivity analysis to weight and activation quantization
        individually.

        :param sim: Quantsim model.
        :return: FP32 eval score, weight-quantized eval score, act-quantized eval score.
        """
        fp32_eval_score = self._eval_model(self._session)
        _logger.info("FP32 eval score (W32A32): %f", fp32_eval_score)

        act_quantized_eval_score = self._eval_activation_quantized_model(sim)
        _logger.info("Activation-quantized eval score (W32A%d): %f", self._default_output_bw,
                     act_quantized_eval_score)

        weight_quantized_eval_score = self._eval_weight_quantized_model(sim)
        _logger.info("Weight-quantized eval score (W%dA32): %f", self._default_param_bw,
                     weight_quantized_eval_score)

        return fp32_eval_score, weight_quantized_eval_score, act_quantized_eval_score

    def _eval_model(self, session: tf.compat.v1.Session) -> float:
        """
        Evaluate the model performance.

        :param session: TensorFlow session to be evaluated.
        :return: Scaler value representing model performance.
        """
        return self._eval_callback.func(session, self._eval_callback.args)

    def _eval_weight_quantized_model(self, sim):
        """
        Evaluate weight quantized model performance.
        For weight quantized model performance, disable enabled activation quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        enabled_activation_quantizers = sim.get_enabled_activation_quantizers()
        self._enable_disable_quantizers(enabled_activation_quantizers, enabled=False)
        eval_score = self._eval_model(sim.session)
        self._enable_disable_quantizers(enabled_activation_quantizers, enabled=True)
        return eval_score

    def _eval_activation_quantized_model(self, sim):
        """
        Evaluate activation quantized model performance.
        For activation quantized model performance, disable enabled param quantizers, measure
        eval score and enable again.

        :param sim: Quantsim model.
        :return: Quantized model performance.
        """
        enabled_param_quantizers = sim.get_enabled_parameter_quantizers()
        self._enable_disable_quantizers(enabled_param_quantizers, enabled=False)
        eval_score = self._eval_model(sim.session)
        self._enable_disable_quantizers(enabled_param_quantizers, enabled=True)
        return eval_score

    def _perform_per_op_analysis_by_enabling_quant_ops(self,
                                                       sim: QuantizationSimModel,
                                                       results_dir: str = "./tmp/",
                                                       ) -> Dict:
        """
        1. All activations and parameters quantizers are disabled.
        2. For every activations and parameters quantizers:
              i. Quantizer is enabled
             ii. Measure and record eval score on subset of dataset.
            iii. Disable enabled quantizer in step i.
        3. Returns dictionary containing quant op name and corresponding eval score.

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise eval score dictionary. dict[op_name] = eval_score
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        _logger.info("\nOPTION-1:\nAll the quant ops are disabled.\n"
                     "Starting per-layer analysis by enabling quant ops as per config file.")
        op_wise_eval_score_dict = self._perform_per_op_analysis(sim,
                                                                disable_all_quantizers=True,
                                                                enabled_before=True,
                                                                enabled_after=False)
        export_per_layer_sensitivity_analysis_plot(op_wise_eval_score_dict,
                                                   results_dir,
                                                   title="per_op_quant_enabled")
        save_json(op_wise_eval_score_dict,
                  results_dir,
                  title="per_op_quant_enabled.json")
        return op_wise_eval_score_dict

    def _perform_per_op_analysis_by_disabling_quant_ops(self,
                                                        sim: QuantizationSimModel,
                                                        results_dir: str = "./tmp/",
                                                        ) -> Dict:
        """
        1. All activations and parameters quantizers are enabled as per JSON config file.
        2. For every activations and parameters quantizers:
              i. Quantizer is disabled
             ii. Measure and record eval score on subset of dataset.
            iii. Enable disabled quantizer in step i.
        3. Returns dictionary containing quant op name and corresponding eval score.

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise eval score dictionary. dict[op_name] = eval_score
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        _logger.info("\nOPTION-2:\nAll the quant ops are enabled as per config file.\n"
                     "Starting per-layer analysis by disabling quant ops.")
        op_wise_eval_score_dict = self._perform_per_op_analysis(sim,
                                                                disable_all_quantizers=False,
                                                                enabled_before=False,
                                                                enabled_after=True)
        export_per_layer_sensitivity_analysis_plot(op_wise_eval_score_dict,
                                                   results_dir,
                                                   title="per_op_quant_disabled")
        save_json(op_wise_eval_score_dict,
                  results_dir,
                  title="per_op_quant_disabled.json")
        return op_wise_eval_score_dict

    def _perform_per_op_analysis(self,
                                 sim: QuantizationSimModel,
                                 disable_all_quantizers: bool,
                                 enabled_before: bool,
                                 enabled_after: bool,
                                 ) -> Dict:
        """
        Helper function for perform_per_layer_analysis_by_enabling_quant_ops() and
        perform_per_layer_analysis_by_disabling_quant_ops()

        :param sim: Quantsim model.
        :param disable_all_quantizers: Flag to disable all the quantizers before per-layer analysis.
        :param enabled_before: Flag to set enabled for quantizers before computing encodings.
        :param enabled_after: Flag to set enabled for quantizers after computing encodings.
        :return: layer wise eval score dictionary. dict[conn_graph_op] = eval_score.
        """

        enabled_quant_ops = self._get_enabled_quantizer_groups(sim)

        if disable_all_quantizers:
            for quantizer_group_list in enabled_quant_ops.values():
                if quantizer_group_list:
                    self._enable_disable_quantizers(quantizer_group_list, enabled=False)

        eval_score_dict = {}
        for conn_graph_op, quantizer_info_list in enabled_quant_ops.items():
            if quantizer_info_list:
                conn_graph_op = str(conn_graph_op)
                self._enable_disable_quantizers(quantizer_info_list, enabled=enabled_before)

                # Record eval score.
                eval_score_dict[conn_graph_op] = self._eval_model(sim.session)
                _logger.info("For connected graph op: %s, the eval score is: %f", conn_graph_op, eval_score_dict[conn_graph_op])

                self._enable_disable_quantizers(quantizer_info_list, enabled=enabled_after)

        if disable_all_quantizers:
            for quantizer_group_list in enabled_quant_ops.values():
                if quantizer_group_list:
                    self._enable_disable_quantizers(quantizer_group_list, enabled=True)

        return eval_score_dict

    @staticmethod
    def _get_enabled_quantizer_groups(sim: QuantizationSimModel)-> Dict[Op, List[QuantizerInfo]]:
        """
        For given quantsim model, get all enabled activation and parameter quantizers.
        :param sim: Quantsim model.
        :return: Dictionary which maps a connected graph op to a list of enabled quantizer info in it.
        """
        enabled_quantizers_dict = {}
        # pylint: disable=protected-access
        for conn_graph_op, quantizer_group in sim._op_to_quant_ops_dict.items():
            group = []
            # pylint: disable=protected-access
            param_quant_op_dict, act_quant_op = quantizer_group
            activation_quantize_info = sim._activation_quantizers.get(act_quant_op.name)
            if activation_quantize_info.enabled:
                group.append(activation_quantize_info)
            for param_op_set in param_quant_op_dict.values():
                for param_op in param_op_set:
                    # pylint: disable=protected-access
                    param_quantize_info = sim._param_quantizers.get(param_op.name)
                    if param_quantize_info.enabled:
                        group.append(param_quantize_info)
            enabled_quantizers_dict[conn_graph_op] = group
        return enabled_quantizers_dict

    @staticmethod
    def _enable_disable_quantizers(quantizer_list: List[QuantizerInfo], enabled: bool):
        """
        For given list of quantizers, set (enable/disable) quantizer's enabled.

        :param quantizer_list: List of quantizers.
        :param enabled: Enabled flag.
        """
        for quantizer_info in quantizer_list:
            if enabled:
                quantizer_info.enable_keeping_encoding()
            else:
                quantizer_info.enabled = enabled

    def _export_per_layer_stats_histogram(self, sim: QuantizationSimModel,
                                          results_dir: str = "./tmp/"):
        """
        NOTE: Not to invoke when quantization scheme is not TF-Enhanced.

        Export histogram that represents a PDF of collected statistics by a quantizer for every
        quant op. After invoking this API, results_dir should have html files in following
        format for every quantizers of quant ops.

        -results_dir
            -activations_pdf
                quant_op_name.html
            -weights_pdf
                -quant_op_name
                    quant_op_name_{channel_index}.html

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        """
        # pylint: disable=protected-access
        weights_pdf_dir = os.path.join(results_dir, "weights_pdf")
        activations_pdf_dir = os.path.join(results_dir, "activations_pdf")

        for quant_op_name, quantizer_info in sim._activation_quantizers.items():
            quant_op_name = quant_op_name.replace("/", "_")
            if quantizer_info.is_encoding_valid():
                self._create_and_export_stats_histogram_plot(quantizer_info,
                                                             activations_pdf_dir,
                                                             title=f"{quant_op_name}")
        for quant_op_name, quantizer_info in sim._param_quantizers.items():
            quant_op_name = quant_op_name.replace("/", "_")
            if quantizer_info.is_encoding_valid():
                self._create_and_export_stats_histogram_plot(quantizer_info,
                                                             os.path.join(weights_pdf_dir, quant_op_name),
                                                             title=f"{quant_op_name}")

        _logger.info("Exported per layer stats histogram.")

    # pylint: disable=no-self-use
    def _export_per_layer_encoding_min_max_range(self, sim: QuantizationSimModel,
                                                 results_dir: str = "./tmp/"
                                                 ) -> Tuple[Dict, Dict]:
        """
        Export encoding min and max range for all weights and activations. results_dir should have
        html files in following format.

        -results_dir
            -activations.html
            -weights.html

        If per channel quantization(PCQ) is enabled then,

        -results_dir
            -activations.html
            -{quant_op_name}_{param_name}.html

        :param sim: Quantsim model.
        :param results_dir: Directory to save the results.
        :return: layer wise min-max range for weights and activations.
        """
        # pylint: disable=protected-access
        min_max_ranges_dir = os.path.join(results_dir, "min_max_ranges")

        min_max_range_for_activations_dict = {}
        min_max_range_for_weights_dict = {}
        for quant_op_name, quantizer_info in sim._activation_quantizers.items():
            quant_op_name = quant_op_name.replace("/", "_")
            if quantizer_info.enabled:
                encoding = quantizer_info.get_encoding()
                min_max_range_for_activations_dict[quant_op_name] = (encoding.min, encoding.max)

        for quant_op_name, quantizer_info in sim._param_quantizers.items():
            quant_op_name = quant_op_name.replace("/", "_")
            if quantizer_info.enabled:
                encoding = quantizer_info.get_encoding()
                if isinstance(encoding, List):  # per-channel
                    per_channel_encodings = {}
                    for index, enc in enumerate(encoding):
                        per_channel_encodings[f"{quant_op_name}_{index}"] = (enc.min, enc.max)
                    min_max_range_for_weights_dict[quant_op_name] = per_channel_encodings
                else:  # per-tensor
                    min_max_range_for_weights_dict[quant_op_name] = (encoding.min, encoding.max)

        create_and_export_min_max_ranges_plot(min_max_range_for_weights_dict,
                                              min_max_ranges_dir,
                                              title="weights")
        create_and_export_min_max_ranges_plot(min_max_range_for_activations_dict,
                                              min_max_ranges_dir,
                                              title="activations")

        _logger.info("Exported per layer encoding min-max ranges.")
        return min_max_range_for_weights_dict, min_max_range_for_activations_dict


    def _create_and_export_stats_histogram_plot(self, quantizer_info: QuantizerInfo,
                                                results_dir: str,
                                                title: str):
        """
        For given quantizer, create and export histogram (PDF) of statistics in html format.

        :param quantizer_info: Quantizer.
        :param results_dir: Directory to save the results.
        :param title: Title of the plot.
        """
        os.makedirs(results_dir, exist_ok=True)

        histograms = quantizer_info.get_stats_histogram()
        encodings = quantizer_info.get_encoding()
        if not isinstance(encodings, List):
            encodings = [encodings]

        for index, (histogram, encoding) in enumerate(zip(histograms, encodings)):
            self._export_stats_histogram_plot(histogram, encoding, results_dir,
                                              title=f"{title}_{index}")

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
