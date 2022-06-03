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
from typing import List, Tuple, Dict
from bokeh import plotting
from bokeh.models import ColumnDataSource, Band, Span, tickers

from aimet_common.utils import AimetLogger
from aimet_tensorflow.quantizer_info import QuantizerInfo
from aimet_tensorflow.quantsim import QuantizationSimModel

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

DEFAULT_BOKEH_FIGURE_HEIGHT = 300


def export_per_layer_stats_histogram(sim: QuantizationSimModel,
                                     results_dir: str = "./tmp/",
                                     ):
    """
    NOTE: Not to invoke when quantization scheme is not TF-Enhanced.

    Export histogram that represents a PDF of collected statistics by a quantizer for every
    quant wrapper. After invoking this API, results_dir should have html files in following
    format for every quantizers of quant wrappers.

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
            _create_and_export_stats_histogram_plot(quantizer_info,
                                                    activations_pdf_dir,
                                                    title=f"{quant_op_name}")
    for quant_op_name, quantizer_info in sim._param_quantizers.items():
        quant_op_name = quant_op_name.replace("/", "_")
        if quantizer_info.is_encoding_valid():
            _create_and_export_stats_histogram_plot(quantizer_info,
                                                    os.path.join(weights_pdf_dir, quant_op_name),
                                                    title=f"{quant_op_name}")

    _logger.info("Exported per layer stats histogram.")


def export_per_layer_encoding_min_max_range(sim: QuantizationSimModel,
                                            results_dir: str = "./tmp/",
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

    _create_and_export_min_max_ranges_plot(min_max_range_for_weights_dict,
                                           min_max_ranges_dir,
                                           title="weights")
    _create_and_export_min_max_ranges_plot(min_max_range_for_activations_dict,
                                           min_max_ranges_dir,
                                           title="activations")

    _logger.info("Exported per layer encoding min-max ranges.")
    return min_max_range_for_weights_dict, min_max_range_for_activations_dict


def _create_and_export_stats_histogram_plot(quantizer_info: QuantizerInfo,
                                            results_dir: str,
                                            title: str,
                                            ):
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
        _export_stats_histogram_plot(histogram, encoding, results_dir,
                                     title=f"{title}_{index}")


def _create_and_export_min_max_ranges_plot(min_max_ranges_dict: Dict,
                                           results_dir: str,
                                           title: str
                                           ):
    """
    Create and export per layer encoding(s) min-max ranges in html format.

    :param min_max_ranges_dict: Dictionary containing encoding min and max ranges.
    :param results_dir: Directory to save the results.
    :param title: Title of the plot.
    """
    os.makedirs(results_dir, exist_ok=True)

    if set(map(type, min_max_ranges_dict.values())) == {dict}:
        for name, per_channel_encodings_dict in min_max_ranges_dict.items():
            _export_per_layer_min_max_ranges_plot(per_channel_encodings_dict,
                                                  results_dir=results_dir,
                                                  title=name)
    elif set(map(type, min_max_ranges_dict.values())) == {tuple}:
        _export_per_layer_min_max_ranges_plot(min_max_ranges_dict,
                                              results_dir=results_dir,
                                              title=title)
    else:
        raise RuntimeError("Per channel quantization should be enabled for all the layers.")


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
    plot.yaxis.ticker = tickers.SingleIntervalTicker(interval=0.25)
    plotting.save(plot)
    return plot
