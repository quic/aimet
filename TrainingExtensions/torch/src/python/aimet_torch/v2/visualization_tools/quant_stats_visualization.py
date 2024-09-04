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

""" Tool to visualize min and max activations/weights of quantized modules in a given model"""


import os
from pathlib import Path
import torch
from bokeh.events import DocumentReady, Reset
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, TextInput, CustomJS, Range1d, HoverTool, CustomJSHover, Div, \
    BooleanFilter, CDSView, Spacer, DataTable, StringFormatter, ScientificFormatter, TableColumn, Tooltip, Select
from bokeh.models.tools import ResetTool
from bokeh.models.dom import HTML
from bokeh.plotting import figure, save, curdoc
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.utils import get_ordered_list_of_modules
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.encoding_analyzer import _MinMaxObserver, _HistogramObserver


PERCENTILES = [1, 25, 50, 75, 99]


def _visualize(sim: QuantizationSimModel, dummy_input, mode: str, save_path: str = "./quant_stats_visualization.html") -> None:
    """
    Helper function for the visualization APIs.

    :param sim: Calibrated QuantSim Object.
    :param dummy_input: Dummy Input.
    :param mode: Whether to plot basic or advanced stats.
    :param save_path: Path for saving the visualization. Format is 'path_to_dir/file_name.html'. Default is './quant_stats_visualization.html'.
    """

    # Ensure that sim is an instance of aimet_torch.quantsim.QuantizationSimModel
    if not isinstance(sim, QuantizationSimModel):
        raise TypeError(f"Expected type 'aimet_torch.v2.quantsim.QuantizationSimModel', got '{type(sim)}'.")

    # Ensure that the save path is valid
    _check_path(save_path)

    # Topologically sort the quantized modules into an ordered list for easier indexing in the plots
    ordered_list = (get_ordered_list_of_modules(sim.model, dummy_input))
    stats_list = []

    if mode == "basic":
        percentile_list = []
    elif mode == "advanced":
        percentile_list = _add_key_percentiles_to_list(PERCENTILES)
        percentile_list = sorted(percentile_list)
    else:
        raise ValueError(f"Expected mode to be 'basic' or 'advanced', got '{mode}'.")

    # Collect stats from observers
    for module in ordered_list:
        module_stats = _get_observer_stats(module, percentile_list=percentile_list)
        if module_stats is not None:
            stats_list.append(module_stats)

    # Raise an error if no stats were found
    if len(stats_list) == 0:
        raise RuntimeError(
            "No stats found to plot. Either there were no quantized modules, or calibration was not performed before calling this function, or no observers of type _MinMaxObserver or _HistogramObserver were present.")

    stats_dict = dict()
    keys_list = ["name", 0, 100] + percentile_list
    stats_dict["idx"] = list(range(len(stats_list)))
    for key in keys_list:
        stats_dict[key] = [None] * len(stats_list)
    for idx, stats in enumerate(stats_list):
        for key in keys_list:
            stats_dict[key][idx] = stats[key]
    visualizer = QuantStatsVisualizer(stats_dict)

    # Save an interactive bokeh plot as a standalone html
    visualizer.export_plot_as_html(save_path)


def visualize_stats(sim: QuantizationSimModel, dummy_input, save_path: str = "./quant_stats_visualization.html") -> None:
    """Produces an interactive html to view the stats collected by each quantizer during calibration

    .. note::

        The QuantizationSimModel input is expected to have been calibrated before using this function. Stats will only
        be plotted for activations/parameters with quantizers containing calibration statistics.

    Creates an interactive visualization of min and max activations/weights of all quantized modules in the input
    QuantSim object. The features include:

        - Adjustable threshold values to flag layers whose min or max activations/weights exceed the set thresholds
        - Tables containing names and ranges for layers exceeding threshold values

    Saves the visualization as a .html at the given path.

    Example:

        >>> sim = aimet_torch.v2.quantsim.QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf)
        >>> with aimet_torch.v2.nn.compute_encodings(sim.model):
        ...     for data, _ in data_loader:
        ...         sim.model(data)
        ...
        >>> visualize_stats(sim, dummy_input, "./quant_stats_visualization.html")

    :param sim: Calibrated QuantizationSimModel
    :param dummy_input: Sample input used to trace the model
    :param save_path: Path for saving the visualization. Default is "./quant_stats_visualization.html"
    """

    _visualize(sim, dummy_input, mode="basic", save_path=save_path)


def visualize_advanced_stats(sim: QuantizationSimModel, dummy_input, save_path: str = "./quant_advanced_stats_visualization.html") -> None:
    """Produces an interactive html to view the advanced stats collected by each quantizer during calibration

    .. note::

        The QuantizationSimModel input is expected to have been calibrated before using this function. Stats will only
        be plotted for activations/parameters with quantizers containing calibration statistics.

    Creates an interactive visualization of min and max activations/weights of all quantized modules in the input
    QuantSim object. The features include:

        - Adjustable threshold values to flag layers whose min or max activations/weights exceed the set thresholds
        - Table containing names and ranges for layers exceeding threshold values
        - Select different views of the table to group layers exceeding threshold values
        - Filter layers listed in the table by name
        - Select one or more layers from the table for advanced analysis

    Saves the visualization as a .html at the given path.

    Example:

        >>> sim = aimet_torch.v2.quantsim.QuantizationSimModel(model, dummy_input, quant_scheme=QuantScheme.post_training_tf_enhanced)
        >>> with aimet_torch.v2.nn.compute_encodings(sim.model):
        ...     for data, _ in data_loader:
        ...         sim.model(data)
        ...
        >>> visualize_advanced_stats(sim, dummy_input, "./quant_advanced_stats_visualization.html")

    :param sim: Calibrated QuantizationSimModel
    :param dummy_input: Sample input used to trace the model
    :param save_path: Path for saving the visualization. Default is "./quant_advanced_stats_visualization.html"
    """

    _visualize(sim, dummy_input, mode="advanced", save_path=save_path)


def _check_path(path: str):
    """ Function for sanity check on the given path """
    path_to_directory = os.path.dirname(path)
    if path_to_directory != '' and not os.path.exists(path_to_directory):
        raise NotADirectoryError(f"'{path_to_directory}' is not a directory.")
    if not path.endswith('.html'):
        raise ValueError("'save_path' must end with '.html'.")


def _get_observer_stats(module, percentile_list):
    """
    Function to extract stats from an observer.
    Handles observers of types _MinMaxObserver, _HistogramObserver.
    """
    module_name, module_quantizer = module[0], module[1]
    if isinstance(module_quantizer, QuantizerBase):
        if isinstance(module_quantizer.encoding_analyzer.observer, _MinMaxObserver):
            rng = module_quantizer.encoding_analyzer.observer.get_stats()
            if rng.min is not None:
                stats = dict()
                stats["name"] = module_name
                stats[0] = torch.min(rng.min).item()
                stats[100] = torch.max(rng.max).item()
                for p in percentile_list:
                    stats[p] = None
                return stats

        elif isinstance(module_quantizer.encoding_analyzer.observer, _HistogramObserver):
            histogram_list = module_quantizer.encoding_analyzer.observer.get_stats()
            if len(histogram_list) == 1:
                histogram = histogram_list[0]
                if histogram.min is not None:
                    stats = dict()
                    stats["name"] = module_name
                    stats[0] = histogram.min.item()
                    stats[100] = histogram.max.item()
                    _get_advanced_stats_from_histogram(histogram, stats, percentile_list)
                    return stats
            elif len(histogram_list) > 1:
                stats = dict()
                stats["name"] = module_name
                curmin = float("inf")
                curmax = float("-inf")
                for histogram in histogram_list:
                    if histogram.min is not None:
                        curmin = min(curmin, histogram.min.item())
                        curmax = max(curmax, histogram.max.item())
                if curmin < float("inf"):
                    stats[0] = curmin
                    stats[100] = curmax
                    for p in percentile_list:
                        stats[p] = None
                    return stats

    return None


def _add_key_percentiles_to_list(percentiles):
    """ Add percentiles required for boxplot if not already present """
    percentile_list = percentiles[:]
    for p in [25, 50, 75]:
        flag = True
        for i in percentile_list:
            if i == p:
                flag = False
        if flag:
            percentile_list.append(p)
    return percentile_list


def _get_advanced_stats_from_histogram(histogram, stats, percentile_list):
    """ High level function to extract advanced stats from a histogram object """
    if len(percentile_list) > 0:
        percentile_stats = _get_percentile_stats_from_histogram(histogram, percentile_list)
        for i, percentile in enumerate(percentile_list):
            stats[percentile] = percentile_stats[i]


def _get_percentile_stats_from_histogram(histogram, percentile_list):
    """ Function to extract percentile stats from a histogram object """
    if len(percentile_list) == 0:
        raise RuntimeError("'percentile_list' cannot be empty.'")
    if not _is_sorted(percentile_list):
        raise RuntimeError("'percentile_list' must be sorted before calling this function.")

    n = torch.sum(histogram.histogram).item()
    cum_f = 0
    idx = 0
    percentile_stats = []
    for i in range(len(histogram.histogram)):
        f = histogram.histogram[i].item()
        if f > 0:
            bin_low = histogram.bin_edges[i].item()
            bin_high = histogram.bin_edges[i + 1].item()
            while True:
                if (cum_f + f) / n >= percentile_list[idx] / 100:
                    percentile_stats.append(bin_low + ((n * percentile_list[idx] / 100 - cum_f) / f) * (bin_high - bin_low))
                    idx += 1
                    if idx == len(percentile_list):
                        return percentile_stats
                else:
                    break
        cum_f += f

    return None


def _is_sorted(arr: list):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True


class DataSources:
    """
    Class to hold the Bokeh ColumnDataSource objects needed in the visualization.
    """

    def __init__(self,
                 stats_dict: dict,
                 plot: figure,
                 default_values: dict,
                 ):
        self.data_source = ColumnDataSource(
            data=dict(idx=stats_dict["idx"], namelist=stats_dict["name"], minlist=stats_dict[0],
                      maxlist=stats_dict[100],
                      marker_yminlist=[default_values['default_ymin']] * len(stats_dict["idx"]),
                      marker_ymaxlist=[default_values['default_ymax']] * len(stats_dict["idx"]),
                      selected=[False] * len(stats_dict["idx"])))
        for key in stats_dict.keys():
            if key not in ["idx", "name", 0, 100]:
                self.data_source.add(data=stats_dict[key], name=str(key) + "%ilelist")

        self.default_values_source = ColumnDataSource(
            data=dict(default_ymax=[default_values['default_ymax']],
                      default_ymin=[default_values['default_ymin']],
                      default_maxclip=[default_values['default_maxclip']],
                      default_minclip=[default_values['default_minclip']],
                      default_xmax=[default_values['default_xmax']],
                      default_xmin=[default_values['default_xmin']]))

        self.limits_source = ColumnDataSource(
            data=dict(ymax=[default_values['default_ymax']], ymin=[default_values['default_ymin']],
                      xmin=[plot.x_range.start], xmax=[plot.x_range.end],
                      minclip=[default_values['default_minclip']],
                      maxclip=[default_values['default_maxclip']]))

        self.table_data_source = ColumnDataSource(
            data=dict(idx=[], namelist=[], minlist=[], maxlist=[]))
        for key in stats_dict.keys():
            if key not in ["idx", "name", 0, 100]:
                self.table_data_source.add(data=[], name=str(key) + "%ilelist")

        self.selected_data_source = ColumnDataSource(
            data=dict(idx=[], namelist=[], floor=[], ceil=[], minlist=[], maxlist=[])
        )
        for key in stats_dict.keys():
            if key not in ["idx", "name", 0, 100]:
                self.selected_data_source.add(data=[], name=str(key) + "%ilelist")


class TableFilters:
    """
    Class for holding data filters.
    """

    def __init__(self, data_sources: DataSources):
        self.name_filter = BooleanFilter()
        self.name_filter.booleans = [True for _ in range(len(data_sources.data_source.data['idx']))]
        self.min_thresh_filter = BooleanFilter()
        self.min_thresh_filter.booleans = [True for _ in range(len(data_sources.data_source.data['idx']))]
        self.max_thresh_filter = BooleanFilter()
        self.max_thresh_filter.booleans = [True for _ in range(len(data_sources.data_source.data['idx']))]


class TableViews:
    """
    Class for holding views of the data sources.
    """

    def __init__(self, tablefilters: TableFilters):
        self.min_thresh_view = CDSView(filter=tablefilters.min_thresh_filter)
        self.max_thresh_view = CDSView(filter=tablefilters.max_thresh_filter)


class TableObjects:
    """
    Class for holding various objects related to the table elements in the visualization.
    """

    def __init__(self, datasources: DataSources):
        self.filters = TableFilters(datasources)
        self.views = TableViews(self.filters)

        columns = [
            TableColumn(field="idx", title="Layer Index",
                        width=QuantStatsVisualizer.table_column_widths["Layer Index"]),
            TableColumn(field="namelist", title="Layer Name",
                        formatter=StringFormatter(font_style="bold"),
                        width=QuantStatsVisualizer.table_column_widths["Layer Name"]),
            TableColumn(field="minlist", title="Min Activation",
                        formatter=ScientificFormatter(precision=3),
                        width=QuantStatsVisualizer.table_column_widths["Min Activation"]),
            TableColumn(field="maxlist", title="Max Activation",
                        formatter=ScientificFormatter(precision=3),
                        width=QuantStatsVisualizer.table_column_widths["Max Activation"]),
        ]

        self.data_table = DataTable(source=datasources.table_data_source, columns=columns,
                                    sortable=True, width=QuantStatsVisualizer.plot_dims["table_width"],
                                    selectable="checkbox",
                                    index_position=None,
                                    )


class InputWidgets:
    """
    Class to hold various input widgets.
    """

    def __init__(self, default_values: dict):
        self.ymin_input = TextInput(value=str(default_values['default_ymin']),
                                    title="Enter lower display limit of the plot")
        self.ymax_input = TextInput(value=str(default_values['default_ymax']),
                                    title="Enter upper display limit of the plot")
        self.minclip_input = TextInput(value=str(default_values['default_minclip']),
                                       title="Enter lower threshold value for activations/weights")
        self.maxclip_input = TextInput(value=str(default_values['default_maxclip']),
                                       title="Enter upper threshold value for activations/weights")

        self.name_input = TextInput(value="", title="Enter Name Filter")

        tooltip_table_mode = Tooltip(content=HTML("""
                                                <h3> Select Table View </h3>
                                                <p> Following table views are available </p>
                                                <ol>
                                                <li> <b> All: </b> All quantized layers </li>
                                                <li> <b> Min: </b> Quantized layers with min activation below lower threshold value </li>
                                                <li> <b> Max: </b> Quantized layers with max activation above upper threshold value </li>
                                                <li> <b> Min | Max: </b> Union of Min and Max </li>
                                                <li> <b> Min & Max: </b> Intersection of Min and Max </li>
                                                </ol>  
                                            """),
                                     position="right")
        self.table_view_select = Select(title="Select Table View",
                                        value="Min | Max",
                                        options=["All", "Min", "Max", "Min | Max", "Min & Max"],
                                        width=200,
                                        description=tooltip_table_mode
                                        )


class CustomCallbacks:
    """
    Class to hold Custom JavaScript Callbacks for interactivity in the visualization.
    """

    def __init__(self):
        self.limit_change_callback = None
        self.reset_callback = None
        self.name_filter_callback = None
        self.select_table_view_callback = None
        self.table_selection_callback = None


class QuantStatsVisualizer:
    """
    Class for constructing the visualization with functionality to export the plot as

    :param stats_dict: Dictionary containing the module names, indices, and other extracted statistics
    """

    # Class level constants
    plot_dims = {"plot_width": 700, "plot_height": 400, "table_width": 800}
    initial_vals = {"default_ymin": -1e5, "default_ymax": 1e5}
    spacer_dims = {"sp1_width": 50, "sp1_height": 40}
    table_column_widths = {"Layer Index": 100,
                           "Layer Name": 400,
                           "Min Activation": 100,
                           "Max Activation": 100}

    def __init__(self, stats_dict: dict):
        self.stats_dict = stats_dict
        self.plot = figure(
            title="Min Max Activations/Weights of quantized modules for given model",
            x_axis_label="Layer index",
            y_axis_label="Activation/Weight",
            tools="pan,wheel_zoom,box_zoom")
        self.default_values = dict()

    def _add_plot_lines(self, datasources: DataSources):
        self.plot.segment(x0='xmin', x1='xmax', y0='ymin', y1='ymin', line_width=4, line_color='black',
                          source=datasources.limits_source)
        self.plot.segment(x0='xmin', x1='xmax', y0='ymax', y1='ymax', line_width=4, line_color='black',
                          source=datasources.limits_source)
        self.plot.segment(x0='xmin', x1='xmax', y0='minclip', y1='minclip', line_width=2, line_color='black',
                          line_dash='dashed',
                          source=datasources.limits_source)
        self.plot.segment(x0='xmin', x1='xmax', y0='maxclip', y1='maxclip', line_width=2, line_color='black',
                          line_dash='dashed',
                          source=datasources.limits_source)
        self.plot.line('idx', 'maxlist', source=datasources.data_source, legend_label="Max Activation", line_width=2,
                       line_color="red")
        self.plot.line('idx', 'minlist', source=datasources.data_source, legend_label="Min Activation", line_width=2,
                       line_color="blue")
        selections = self.plot.segment(x0='idx', x1='idx', y0='floor', y1='ceil', line_width=2, line_color='goldenrod',
                                       line_alpha=0.5, source=datasources.selected_data_source)

        return selections

    def _add_min_max_markers(self, datasources: DataSources, tableobjects: TableObjects):
        min_markers = self.plot.circle_x('idx', 'marker_yminlist', source=datasources.data_source, size=10,
                                         color='orange',
                                         line_color="navy")
        min_markers.view = tableobjects.views.min_thresh_view
        max_markers = self.plot.circle_x('idx', 'marker_ymaxlist', source=datasources.data_source, size=10,
                                         color='orange',
                                         line_color="navy")
        max_markers.view = tableobjects.views.max_thresh_view

        return min_markers, max_markers

    @staticmethod
    def _get_marker_hovertool(min_markers, max_markers):
        format_code = """
                    if (Math.abs(value) < 1e-3 || Math.abs(value) > 1e5) {
                    return value.toExponential(3);
                    } else {
                    return value.toFixed(3);
                    }
                """

        format_hover = CustomJSHover(code=format_code)

        marker_hover = HoverTool(renderers=[min_markers, max_markers], tooltips=[
            ("Layer Index", "@idx"),
            ("Name", "@namelist"),
            ("Max Activation", "@maxlist{custom}"),
            ("Min Activation", "@minlist{custom}"),
        ], formatters={
            "@minlist": format_hover,
            "@maxlist": format_hover,
        })

        return marker_hover

    @staticmethod
    def _get_selection_hovertool(selections):
        format_code = """
                    if (Math.abs(value) < 1e-3 || Math.abs(value) > 1e5) {
                    return value.toExponential(3);
                    } else {
                    return value.toFixed(3);
                    }
                """

        format_hover = CustomJSHover(code=format_code)

        selection_hover = HoverTool(renderers=[selections], tooltips=[
            ("Layer Index", "@idx"),
            ("Name", "@namelist"),
            ("Max Activation", "@maxlist{custom}"),
            ("Min Activation", "@minlist{custom}"),
        ], formatters={
            "@minlist": format_hover,
            "@maxlist": format_hover,
        })

        return selection_hover

    def _define_callbacks(self, datasources, tableobjects, inputwidgets):
        customcallbacks = CustomCallbacks()

        customcallbacks.limit_change_callback = CustomJS(args=dict(
            limits_source=datasources.limits_source,
            data_source=datasources.data_source,
            table_data_source=datasources.table_data_source,
            selected_data_source=datasources.selected_data_source,
            min_marker_source=datasources.data_source,
            max_marker_source=datasources.data_source,
            ymax_input=inputwidgets.ymax_input,
            ymin_input=inputwidgets.ymin_input,
            maxclip_input=inputwidgets.maxclip_input,
            minclip_input=inputwidgets.minclip_input,
            plot=self.plot,
            min_thresh_filter=tableobjects.filters.min_thresh_filter,
            max_thresh_filter=tableobjects.filters.max_thresh_filter,
            name_filter=tableobjects.filters.name_filter,
            select=inputwidgets.table_view_select,
        ), code=(Path(__file__).parent / "quant_stats_visualization_JS_code/utils.js").read_text("utf8") + (Path(__file__).parent / "quant_stats_visualization_JS_code/limit_change_callback.js").read_text("utf8"))

        customcallbacks.reset_callback = CustomJS(args=dict(
            limits_source=datasources.limits_source,
            data_source=datasources.data_source,
            table_data_source=datasources.table_data_source,
            selected_data_source=datasources.selected_data_source,
            default_values_source=datasources.default_values_source,
            min_marker_source=datasources.data_source,
            max_marker_source=datasources.data_source,
            ymax_input=inputwidgets.ymax_input,
            ymin_input=inputwidgets.ymin_input,
            maxclip_input=inputwidgets.maxclip_input,
            minclip_input=inputwidgets.minclip_input,
            select=inputwidgets.table_view_select,
            name_input=inputwidgets.name_input,
            plot=self.plot,
            min_thresh_filter=tableobjects.filters.min_thresh_filter,
            max_thresh_filter=tableobjects.filters.max_thresh_filter,
            name_filter=tableobjects.filters.name_filter,
        ), code=(Path(__file__).parent / "quant_stats_visualization_JS_code/utils.js").read_text("utf8") + (Path(__file__).parent / "quant_stats_visualization_JS_code/reset_callback.js").read_text("utf8"))

        customcallbacks.name_filter_callback = CustomJS(args=dict(
            data_source=datasources.data_source,
            table_data_source=datasources.table_data_source,
            limits_source=datasources.limits_source,
            min_thresh_filter=tableobjects.filters.min_thresh_filter,
            max_thresh_filter=tableobjects.filters.max_thresh_filter,
            name_filter=tableobjects.filters.name_filter,
            select=inputwidgets.table_view_select,
        ), code=(Path(__file__).parent / "quant_stats_visualization_JS_code/utils.js").read_text("utf8") + (Path(__file__).parent / "quant_stats_visualization_JS_code/name_filter_callback.js").read_text("utf8"))

        customcallbacks.select_table_view_callback = CustomJS(args=dict(
            data_source=datasources.data_source,
            table_data_source=datasources.table_data_source,
            select=inputwidgets.table_view_select,
            min_thresh_filter=tableobjects.filters.min_thresh_filter,
            max_thresh_filter=tableobjects.filters.max_thresh_filter,
            name_filter=tableobjects.filters.name_filter,
            table=tableobjects.data_table
        ), code=(Path(__file__).parent / "quant_stats_visualization_JS_code/utils.js").read_text("utf8") + (Path(__file__).parent / "quant_stats_visualization_JS_code/select_table_view_callback.js").read_text("utf8"))

        customcallbacks.table_selection_callback = CustomJS(args=dict(
            data_source=datasources.data_source,
            table_data_source=datasources.table_data_source,
            selected_data_source=datasources.selected_data_source,
            limits_source=datasources.limits_source,
        ), code=(Path(__file__).parent / "quant_stats_visualization_JS_code/utils.js").read_text("utf8") + (Path(__file__).parent / "quant_stats_visualization_JS_code/table_selection_callback.js").read_text("utf8"))

        return customcallbacks

    def _attach_callbacks(self, datasources, inputwidgets, customcallbacks):
        self.plot.js_on_event(Reset, customcallbacks.reset_callback)
        inputwidgets.ymax_input.js_on_change('value', customcallbacks.limit_change_callback)
        inputwidgets.ymin_input.js_on_change('value', customcallbacks.limit_change_callback)
        inputwidgets.maxclip_input.js_on_change('value', customcallbacks.limit_change_callback)
        inputwidgets.minclip_input.js_on_change('value', customcallbacks.limit_change_callback)
        inputwidgets.name_input.js_on_change("value", customcallbacks.name_filter_callback)
        inputwidgets.table_view_select.js_on_change('value', customcallbacks.select_table_view_callback)
        datasources.table_data_source.selected.js_on_change('indices', customcallbacks.table_selection_callback)

    def _create_layout(self, inputwidgets, tableobjects):
        heading_1 = Div(text="<h2>Quant Stats Visualizer</h2>")
        heading_2 = Div(text="<h2>Quant Stats Data Table</h2>")

        sp1 = Spacer(width=QuantStatsVisualizer.spacer_dims["sp1_width"],
                     height=QuantStatsVisualizer.spacer_dims["sp1_height"])
        row1 = row(inputwidgets.ymin_input, inputwidgets.ymax_input)
        row2 = row(inputwidgets.minclip_input, inputwidgets.maxclip_input)
        inputs1 = column(row1, row2)
        layout = column(heading_1, inputs1, sp1, self.plot,
                        column(heading_2, row(inputwidgets.table_view_select, inputwidgets.name_input),
                               tableobjects.data_table))

        return layout

    def export_plot_as_html(self, save_path: str) -> None:
        """
        Method for constructing the visualization and saving it to the given path.

        :param save_path: Path for saving the visualization.
        """

        curdoc().theme = 'light_minimal'

        self.plot.width = QuantStatsVisualizer.plot_dims["plot_width"]
        self.plot.height = QuantStatsVisualizer.plot_dims["plot_height"]

        # Defining the default values of plotting parameters
        self.default_values['default_ymax'] = QuantStatsVisualizer.initial_vals["default_ymax"]
        self.default_values['default_ymin'] = QuantStatsVisualizer.initial_vals["default_ymin"]
        self.default_values['default_xmax'] = len(self.stats_dict["idx"]) - 1
        self.default_values['default_xmin'] = 0
        self.default_values['default_maxclip'] = self.default_values['default_ymax'] / 2
        self.default_values['default_minclip'] = self.default_values['default_ymin'] / 2

        self.plot.x_range = Range1d(0, len(self.stats_dict["idx"]))
        self.plot.y_range = Range1d(self.default_values['default_ymax'] * 1.05,
                                    self.default_values['default_ymin'] * 1.05)

        # Creating and adding a reset tool
        rt = ResetTool()
        self.plot.add_tools(rt)

        # Defining Bokeh ColumnDataSources
        datasources = DataSources(stats_dict=self.stats_dict,
                                  plot=self.plot,
                                  default_values=self.default_values,
                                  )

        # Creating plot objects
        selections = self._add_plot_lines(datasources)

        # Defining the table objects and name filter views
        tableobjects = TableObjects(datasources)

        # Marker points to see which layers cross the thresholds
        min_markers, max_markers = self._add_min_max_markers(datasources, tableobjects)

        # Defining a hover functionality to see layer details on hovering on the marker points and selections
        marker_hover = self._get_marker_hovertool(min_markers, max_markers)
        selection_hover = self._get_selection_hovertool(selections)
        self.plot.add_tools(marker_hover, selection_hover)

        # Creating the input widgets
        inputwidgets = InputWidgets(self.default_values)

        # Defining Custom JavaScript callbacks
        customcallbacks = self._define_callbacks(datasources, tableobjects, inputwidgets)

        # Attach events to corresponding callbacks
        curdoc().js_on_event(DocumentReady, customcallbacks.reset_callback)
        self._attach_callbacks(datasources, inputwidgets, customcallbacks)

        # Define the formatting
        layout = self._create_layout(inputwidgets, tableobjects)

        # Save as standalone html
        save(layout, save_path)
