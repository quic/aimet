# /usr/bin/env python3.8
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""Common methods used in QuantAnalyzer across frameworks"""
import json
import os
from typing import Dict

from bokeh import plotting
from bokeh.models import tickers

DEFAULT_BOKEH_FIGURE_HEIGHT = 300


def export_per_layer_sensitivity_analysis_plot(layer_wise_eval_score_dict: Dict, results_dir: str,
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


def save_json(dictionary: Dict, results_dir: str, title: str):
    """
    Save dictionary in JSON format.
    :param dictionary: Dictionary to be saved.
    :param results_dir: Directory to save the results.
    :param title: Title of the file.
    """
    filename = os.path.join(results_dir, title)
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4)


def create_and_export_min_max_ranges_plot(min_max_ranges_dict: Dict,
                                          results_dir: str,
                                          title: str):
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
