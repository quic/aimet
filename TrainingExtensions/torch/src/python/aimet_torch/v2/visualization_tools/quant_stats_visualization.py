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
import torch
from bokeh.events import DocumentReady, Reset
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, TextInput, CustomJS, Range1d, HoverTool, CustomJSHover, Div, \
    BooleanFilter, CDSView, Spacer, DataTable, StringFormatter, TableColumn
from bokeh.models.tools import ResetTool
from bokeh.plotting import figure, save, curdoc
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.utils import get_ordered_list_of_modules
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.encoding_analyzer import _MinMaxObserver


def visualize_stats(sim: QuantizationSimModel, dummy_input, save_path: str = None) -> None:
    """Produces an interactive html to view the stats collected by each quantizer during calibration

    .. note::

        The QuantizationSimModel input is expected to have been calibrated before using this function. Stats will only
        be plotted for activations/parameters with quantizers containing calibration statistics.

        Currently, this tool is only compatible with quantizers containing :class:`MinMaxEncodingAnalyzer` encoding
        analyzers (i.e., :attr:`QuantScheme.post_training_tf` and :attr:`QuantScheme.training_range_learning_with_tf_init`
        quant schemes).

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
    if not isinstance(sim, QuantizationSimModel):
        raise TypeError(f"Expected type 'aimet_torch.v2.quantsim.QuantizationSimModel', got '{type(sim)}'.")

    check_path(save_path)

    # Flatten the quantized modules into an ordered list for easier indexing in the plots
    ordered_list = (get_ordered_list_of_modules(sim.model, dummy_input))
    namelist = []
    minlist = []
    maxlist = []

    for module in ordered_list:
        if isinstance(module[1], QuantizerBase):
            if isinstance(module[1].encoding_analyzer.observer, _MinMaxObserver):
                rng = module[1].encoding_analyzer.observer.get_stats()
                if (rng.min is not None) and (rng.max is not None):
                    namelist.append(module[0])
                    minlist.append(torch.min(rng.min).item())
                    maxlist.append(torch.max(rng.max).item())
            # TODO - Handle other quant schemes

    if len(namelist) == 0:
        raise RuntimeError(
            "No stats found to plot. Either there were no quantized modules, or calibration was not performed before calling this function, or observers of type other than _MinMaxObserver were used.")
    idx = list(range(len(namelist)))

    # Save an interactive bokeh plot as a standalone html in the specified directory with the specified name if provided
    if not save_path:
        save_path = "quant_stats_visualization.html"
    check_path(save_path)

    visualizer = QuantStatsVisualizer(idx, namelist, minlist, maxlist)
    visualizer.export_plot_as_html(save_path)

def check_path(path: str):
    """ Function for sanity check on the given path """
    path_to_directory = os.path.dirname(path)
    if path_to_directory != '' and not os.path.exists(path_to_directory):
        raise NotADirectoryError(f"'{path_to_directory}' is not a directory.")
    if not path.endswith('.html'):
        raise ValueError("'save_path' must end with '.html'.")


class DataSources:
    """
    Class to hold the Bokeh ColumnDataSource objects needed in the visualization.
    """
    def __init__(self,
                 idx:list,
                 namelist:list,
                 minlist:list,
                 maxlist:list,
                 p: figure,
                 default_values: dict,
                 ):

        self.data_source = ColumnDataSource(
            data=dict(idx=idx, namelist=namelist, minlist=minlist, maxlist=maxlist))
        self.default_values_source = ColumnDataSource(
            data=dict(default_ymax=[default_values['default_ymax']],
                      default_ymin=[default_values['default_ymin']],
                      default_maxclip=[default_values['default_maxclip']],
                      default_minclip=[default_values['default_minclip']],
                      default_xmax=[default_values['default_xmax']],
                      default_xmin=[default_values['default_xmin']]))
        self.limits_source = ColumnDataSource(
            data=dict(ymax=[default_values['default_ymax']], ymin=[default_values['default_ymin']],
                      xmin=[p.x_range.start], xmax=[p.x_range.end],
                      minclip=[default_values['default_minclip']],
                      maxclip=[default_values['default_maxclip']]))
        self.min_marker_source = ColumnDataSource(
            data=dict(x=[], y=[], names=[], min_activations=[], max_activations=[], fmt_min_activations=[],
                      fmt_max_activations=[]))
        self.max_marker_source = ColumnDataSource(
            data=dict(x=[], y=[], names=[], min_activations=[], max_activations=[], fmt_min_activations=[],
                      fmt_max_activations=[]))


class TableObjects:
    """
    Class for holding various objects related to the table elements in the visualization.
    """
    def __init__(self, datasources: DataSources):
        self.min_name_filter = BooleanFilter()
        self.max_name_filter = BooleanFilter()
        self.min_name_view = CDSView(filter=self.min_name_filter)
        self.max_name_view = CDSView(filter=self.max_name_filter)

        min_columns = [
            TableColumn(field="names", title="Layer Name",
                        formatter=StringFormatter(font_style="bold"), width=400),
            TableColumn(field="fmt_min_activations", title="Min Activation", width=100),
            TableColumn(field="fmt_max_activations", title="Max Activation", width=100),
        ]

        self.min_data_table = DataTable(source=datasources.min_marker_source, view=self.min_name_view, columns=min_columns,
                                        editable=True,
                                        sortable=True, selectable="checkbox", width=800,
                                        index_position=-1, index_header="row index", index_width=60,
                                        )

        max_columns = [
            TableColumn(field="names", title="Layer Name",
                        formatter=StringFormatter(font_style="bold"), width=400),
            TableColumn(field="fmt_min_activations", title="Min Activation", width=100),
            TableColumn(field="fmt_max_activations", title="Max Activation", width=100),
        ]

        self.max_data_table = DataTable(source=datasources.max_marker_source, view=self.max_name_view, columns=max_columns,
                                        editable=True,
                                        sortable=True, selectable="checkbox", width=800,
                                        index_position=-1, index_header="row index", index_width=60,
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

        self.min_name_input = TextInput(value="", title="Enter name filter")
        self.max_name_input = TextInput(value="", title="Enter name filter")


class CustomCallbacks:
    """
    Class to hold Custom JavaScript Callbacks for interactivity in the visualization.
    """
    def __init__(self):
        self.limit_change_callback = None
        self.reset_callback = None
        self.min_name_filter_callback = None
        self.max_name_filter_callback = None


class QuantStatsVisualizer:
    """
    Class for constructing the visualization with functionality to export the plot as

    :param idx: List with indexing for the ordered list of quantized modules.
    :param namelist: List containing names of the ordered list of quantized modules.
    :param minlist: List containing min activations of the ordered list of quantized modules.
    :param maxlist: List containing max activations of the ordered list of quantized modules.
    """
    def __init__(self, idx: list, namelist: list, minlist: list, maxlist: list):
        self.idx = idx
        self.namelist = namelist
        self.minlist = minlist
        self.maxlist = maxlist
        self.p = None
        self.default_values = dict()

    def _add_plot_lines(self, datasources: DataSources):
        self.p.segment(x0='xmin', x1='xmax', y0='ymin', y1='ymin', line_width=4, line_color='black', source=datasources.limits_source)
        self.p.segment(x0='xmin', x1='xmax', y0='ymax', y1='ymax', line_width=4, line_color='black', source=datasources.limits_source)
        self.p.segment(x0='xmin', x1='xmax', y0='minclip', y1='minclip', line_width=2, line_color='black',
                  line_dash='dashed',
                  source=datasources.limits_source)
        self.p.segment(x0='xmin', x1='xmax', y0='maxclip', y1='maxclip', line_width=2, line_color='black',
                  line_dash='dashed',
                  source=datasources.limits_source)
        self.p.line('idx', 'maxlist', source=datasources.data_source, legend_label="Max Activation", line_width=2, line_color="red")
        self.p.line('idx', 'minlist', source=datasources.data_source, legend_label="Min Activation", line_width=2, line_color="blue")


    def _add_min_max_markers(self, datasources: DataSources):
        min_markers = self.p.circle_x('x', 'y', source=datasources.min_marker_source, size=10, color='orange', line_color="navy")
        max_markers = self.p.circle_x('x', 'y', source=datasources.max_marker_source, size=10, color='orange', line_color="navy")

        return  min_markers, max_markers

    @staticmethod
    def _get_min_max_hovertools(min_markers, max_markers):
        format_code = """
                    if (Math.abs(value) < 1e-3 || Math.abs(value) > 1e3) {
                    return value.toExponential(2);
                    } else {
                    return value.toFixed(2);
                    }
                """

        format_hover = CustomJSHover(code=format_code)

        min_hover = HoverTool(renderers=[min_markers], tooltips=[
            ("Name", "@names"),
            ("Max Activation", "@max_activations{custom}"),
            ("Min Activation", "@min_activations{custom}"),
        ], formatters={
            "@min_activations": format_hover,
            "@max_activations": format_hover,
        })

        max_hover = HoverTool(renderers=[max_markers], tooltips=[
            ("Name", "@names"),
            ("Max Activation", "@max_activations{custom}"),
            ("Min Activation", "@min_activations{custom}"),
        ], formatters={
            "@min_activations": format_hover,
            "@max_activations": format_hover,
        })

        return min_hover, max_hover


    def _define_callbacks(self, datasources, tableobjects, inputwidgets):
        customcallbacks = CustomCallbacks()

        customcallbacks.limit_change_callback = CustomJS(args=dict(limits_source=datasources.limits_source,
                                                   data_source=datasources.data_source,
                                                   min_marker_source=datasources.min_marker_source,
                                                   max_marker_source=datasources.max_marker_source,
                                                   ymax_input=inputwidgets.ymax_input,
                                                   ymin_input=inputwidgets.ymin_input,
                                                   maxclip_input=inputwidgets.maxclip_input,
                                                   minclip_input=inputwidgets.minclip_input,
                                                   plot=self.p,
                                                   min_name_filter=tableobjects.min_name_filter,
                                                   max_name_filter=tableobjects.max_name_filter,
                                                   ), code="""
                // Function to adaptively format numerical values in scientific notation
                // if they are large in magnitude
                function formatValue(value) {
                    if (Math.abs(value) < 1e-3 || Math.abs(value) > 1e3) {
                        return value.toExponential(2);
                    } else {
                        return value.toFixed(2);
                    }
                }
                
                // Reading values from input widgets and setting plot y axis range  
                const limits_data = limits_source.data;
                limits_data['ymax'] = [parseFloat(ymax_input.value)];
                limits_data['ymin'] = [parseFloat(ymin_input.value)];
                plot.y_range.start = limits_data['ymin'][0];
                plot.y_range.end = limits_data['ymax'][0];
                limits_data['maxclip'] = [parseFloat(maxclip_input.value)];
                limits_data['minclip'] = [parseFloat(minclip_input.value)];
                
                // Updating the min and max marker sources
                const activation_data = data_source.data;
                const idx = activation_data['idx'];
                const minlist = activation_data['minlist'];
                const maxlist = activation_data['maxlist'];
                const namelist = activation_data['namelist'];
                const min_marker_x = [];
                const min_marker_y = [];
                const min_marker_names = [];
                const min_marker_min_activations = [];
                const min_marker_max_activations = [];
                const min_marker_fmt_min_activations = [];
                const min_marker_fmt_max_activations = [];
                for (let i = 0; i < idx.length; i++) {
                    if (minlist[i] < limits_data['minclip'][0]) {
                        min_marker_x.push(idx[i]);
                        min_marker_y.push(limits_data['minclip'][0]);
                        min_marker_names.push(namelist[i]);
                        min_marker_min_activations.push(minlist[i]);
                        min_marker_max_activations.push(maxlist[i]);
                        min_marker_fmt_min_activations.push(formatValue(minlist[i]));
                        min_marker_fmt_max_activations.push(formatValue(maxlist[i]));
                    }
                }
                min_marker_source.data['x'] = min_marker_x;
                min_marker_source.data['y'] = min_marker_y;
                min_marker_source.data['names'] = min_marker_names;
                min_marker_source.data['min_activations'] = min_marker_min_activations;
                min_marker_source.data['max_activations'] = min_marker_max_activations;
                min_marker_source.data['fmt_min_activations'] = min_marker_fmt_min_activations;
                min_marker_source.data['fmt_max_activations'] = min_marker_fmt_max_activations;
                min_name_filter.booleans = new Array(min_marker_names.length).fill(true);

                const max_marker_x = [];
                const max_marker_y = [];
                const max_marker_names = [];
                const max_marker_min_activations = [];
                const max_marker_max_activations = [];
                const max_marker_fmt_min_activations = [];
                const max_marker_fmt_max_activations = [];
                for (let i = 0; i < idx.length; i++) {
                    if (maxlist[i] > limits_data['maxclip'][0]) {
                        max_marker_x.push(idx[i]);
                        max_marker_y.push(limits_data['maxclip'][0]);
                        max_marker_names.push(namelist[i]);
                        max_marker_min_activations.push(minlist[i]);
                        max_marker_max_activations.push(maxlist[i]);
                        max_marker_fmt_min_activations.push(formatValue(minlist[i]));
                        max_marker_fmt_max_activations.push(formatValue(maxlist[i]));
                    }
                }

                max_marker_source.data['x'] = max_marker_x;
                max_marker_source.data['y'] = max_marker_y;
                max_marker_source.data['names'] = max_marker_names;
                max_marker_source.data['min_activations'] = max_marker_min_activations;
                max_marker_source.data['max_activations'] = max_marker_max_activations;
                max_marker_source.data['fmt_min_activations'] = max_marker_fmt_min_activations;
                max_marker_source.data['fmt_max_activations'] = max_marker_fmt_max_activations;
                max_name_filter.booleans = new Array(max_marker_names.length).fill(true);
                
                // Emitting the changes made to ColumnDataSources
                limits_source.change.emit();
                min_marker_source.change.emit();
                max_marker_source.change.emit();
            """)

        customcallbacks.reset_callback = CustomJS(args=dict(limits_source=datasources.limits_source,
                                            data_source=datasources.data_source,
                                            default_values_source=datasources.default_values_source,
                                            min_marker_source=datasources.min_marker_source,
                                            max_marker_source=datasources.max_marker_source,
                                            ymax_input=inputwidgets.ymax_input,
                                            ymin_input=inputwidgets.ymin_input,
                                            maxclip_input=inputwidgets.maxclip_input,
                                            minclip_input=inputwidgets.minclip_input,
                                            plot=self.p,
                                            min_name_filter=tableobjects.min_name_filter,
                                            max_name_filter=tableobjects.max_name_filter,
                                            ), code="""
                // Function to adaptively format numerical values in scientific notation
                // if they are large in magnitude
                function formatValue(value) {
                    if (Math.abs(value) < 1e-3 || Math.abs(value) > 1e3) {
                        return value.toExponential(2);
                    } else {
                        return value.toFixed(2);
                    }
                }
                
                // Resetting the limits source with default values
                limits_source.data['ymax'] = default_values_source.data['default_ymax'];
                limits_source.data['ymin'] = default_values_source.data['default_ymin'];
                limits_source.data['xmax'] = default_values_source.data['default_xmax'];
                limits_source.data['xmin'] = default_values_source.data['default_xmin'];
                limits_source.data['maxclip'] = default_values_source.data['default_maxclip'];
                limits_source.data['minclip'] = default_values_source.data['default_minclip'];
                const limits_data = limits_source.data;

                // Resetting the plot ranges
                plot.y_range.start = limits_data['ymin'][0];
                plot.y_range.end = limits_data['ymax'][0];
                plot.x_range.start = limits_data['xmin'][0];
                plot.x_range.end = limits_data['xmax'][0];

                // Resetting the input widget values
                ymax_input.value = limits_data['ymax'][0].toString();
                ymin_input.value = limits_data['ymin'][0].toString();
                maxclip_input.value = limits_data['maxclip'][0].toString();
                minclip_input.value = limits_data['minclip'][0].toString();
                
                // Updating the min and max marker sources
                const activation_data = data_source.data;
                const idx = activation_data['idx'];
                const minlist = activation_data['minlist'];
                const maxlist = activation_data['maxlist'];
                const namelist = activation_data['namelist'];
                const min_marker_x = [];
                const min_marker_y = [];
                const min_marker_names = [];
                const min_marker_min_activations = [];
                const min_marker_max_activations = [];
                const min_marker_fmt_min_activations = [];
                const min_marker_fmt_max_activations = [];
                for (let i = 0; i < idx.length; i++) {
                    if (minlist[i] < limits_data['minclip'][0]) {
                        min_marker_x.push(idx[i]);
                        min_marker_y.push(limits_data['minclip'][0]);
                        min_marker_names.push(namelist[i]);
                        min_marker_min_activations.push(minlist[i]);
                        min_marker_max_activations.push(maxlist[i]);
                        min_marker_fmt_min_activations.push(formatValue(minlist[i]));
                        min_marker_fmt_max_activations.push(formatValue(maxlist[i]));
                    }
                }
                min_marker_source.data['x'] = min_marker_x;
                min_marker_source.data['y'] = min_marker_y;
                min_marker_source.data['names'] = min_marker_names;
                min_marker_source.data['min_activations'] = min_marker_min_activations;
                min_marker_source.data['max_activations'] = min_marker_max_activations;
                min_marker_source.data['fmt_min_activations'] = min_marker_fmt_min_activations;
                min_marker_source.data['fmt_max_activations'] = min_marker_fmt_max_activations;
                min_name_filter.booleans = new Array(min_marker_names.length).fill(true);

                const max_marker_x = [];
                const max_marker_y = [];
                const max_marker_names = [];
                const max_marker_min_activations = [];
                const max_marker_max_activations = [];
                const max_marker_fmt_min_activations = [];
                const max_marker_fmt_max_activations = [];
                for (let i = 0; i < idx.length; i++) {
                    if (maxlist[i] > limits_data['maxclip'][0]) {
                        max_marker_x.push(idx[i]);
                        max_marker_y.push(limits_data['maxclip'][0]);
                        max_marker_names.push(namelist[i]);
                        max_marker_min_activations.push(minlist[i]);
                        max_marker_max_activations.push(maxlist[i]);
                        max_marker_fmt_min_activations.push(formatValue(minlist[i]));
                        max_marker_fmt_max_activations.push(formatValue(maxlist[i]));
                    }
                }

                max_marker_source.data['x'] = max_marker_x;
                max_marker_source.data['y'] = max_marker_y;
                max_marker_source.data['names'] = max_marker_names;
                max_marker_source.data['min_activations'] = max_marker_min_activations;
                max_marker_source.data['max_activations'] = max_marker_max_activations;
                max_marker_source.data['fmt_min_activations'] = max_marker_fmt_min_activations;
                max_marker_source.data['fmt_max_activations'] = max_marker_fmt_max_activations;
                max_name_filter.booleans = new Array(max_marker_names.length).fill(true);
                
                // Emitting the changes made to ColumnDataSources
                limits_source.change.emit();
                min_marker_source.change.emit();
                max_marker_source.change.emit();
            """)

        customcallbacks.min_name_filter_callback = CustomJS(args=dict(marker_source=datasources.min_marker_source,
                                                      text_filter=tableobjects.min_name_filter,
                                                      ),
                                            code="""
                // Filter all names having entered pattern as a substring
                text_filter.booleans = Array.from(marker_source.data['names']).map(t => t.includes(cb_obj.value));
                marker_source.change.emit();
            """)

        customcallbacks.max_name_filter_callback = CustomJS(args=dict(marker_source=datasources.max_marker_source,
                                                      text_filter=tableobjects.max_name_filter,
                                                      ),
                                            code="""
                // Filter all names having entered pattern as a substring
                text_filter.booleans = Array.from(marker_source.data['names']).map(t => t.includes(cb_obj.value));
                marker_source.change.emit();
            """)

        return customcallbacks

    def _attach_callbacks(self, inputwidgets, customcallbacks):
        self.p.js_on_event(Reset, customcallbacks.reset_callback)
        inputwidgets.ymax_input.js_on_change('value', customcallbacks.limit_change_callback)
        inputwidgets.ymin_input.js_on_change('value', customcallbacks.limit_change_callback)
        inputwidgets.maxclip_input.js_on_change('value', customcallbacks.limit_change_callback)
        inputwidgets.minclip_input.js_on_change('value', customcallbacks.limit_change_callback)
        inputwidgets.min_name_input.js_on_change("value", customcallbacks.min_name_filter_callback)
        inputwidgets.max_name_input.js_on_change("value", customcallbacks.max_name_filter_callback)


    def _create_layout(self, inputwidgets, tableobjects):
        heading_1 = Div(text="<h2>Quant Stats Visualizer</h2>")
        heading_2 = Div(text="<h2>List of layers with Min activation/weight lesser than lower threshold</h2>")
        heading_3 = Div(text="<h2>List of layers with Max activation/weight higher than upper threshold</h2>")

        sp1 = Spacer(width=50, height=40)
        row1 = row(inputwidgets.ymin_input, inputwidgets.ymax_input)
        row2 = row(inputwidgets.minclip_input, inputwidgets.maxclip_input)
        inputs1 = column(row1, row2)
        layout = column(heading_1, inputs1, sp1, self.p,
                        row(column(heading_2, inputwidgets.min_name_input, tableobjects.min_data_table),
                            column(heading_3, inputwidgets.max_name_input, tableobjects.max_data_table)))

        return layout

    def export_plot_as_html(self, save_path: str) -> None:
        """
        Method for constructing the visualization and saving it to the given path.

        :param save_path: Path for saving the visualization.
        """

        curdoc().theme = 'light_minimal'

        self.p = figure(width=700,
               height=400,
               title="Min Max Activations/Weights of quantized modules for given model",
               x_axis_label="Layer index",
               y_axis_label="Activation/Weight",
               tools="pan,wheel_zoom,box_zoom")

        # Defining the default values of plotting parameters
        self.default_values['default_ymax'] = 1e5
        self.default_values['default_ymin'] = -1e5
        self.default_values['default_xmax'] = len(self.idx) - 1
        self.default_values['default_xmin'] = 0
        self.default_values['default_maxclip'] = self.default_values['default_ymax'] / 2
        self.default_values['default_minclip'] = self.default_values['default_ymin'] / 2

        self.p.x_range = Range1d(0, len(self.idx))
        self.p.y_range = Range1d(self.default_values['default_ymax'], self.default_values['default_ymin'])

        # Creating and adding a reset tool
        rt = ResetTool()
        self.p.add_tools(rt)

        # Defining Bokeh ColumnDataSources
        datasources = DataSources(idx=self.idx,
                                  namelist=self.namelist,
                                  minlist=self.minlist,
                                  maxlist=self.maxlist,
                                  p=self.p,
                                  default_values=self.default_values,
                                  )

        # Creating plot objects
        self._add_plot_lines(datasources)

        # Marker points to see which layers cross the thresholds
        min_markers, max_markers = self._add_min_max_markers(datasources)

        # Defining a hover functionality to see layer details on hovering on the marker points
        min_hover, max_hover = self._get_min_max_hovertools(min_markers, max_markers)
        self.p.add_tools(min_hover, max_hover)

        # Defining the table objects and name filter views
        tableobjects = TableObjects(datasources)

        # Creating the input widgets
        inputwidgets = InputWidgets(self.default_values)

        # Defining Custom JavaScript callbacks
        customcallbacks = self._define_callbacks(datasources, tableobjects, inputwidgets)

        # Attach events to corresponding callbacks
        curdoc().js_on_event(DocumentReady, customcallbacks.reset_callback)
        self._attach_callbacks(inputwidgets, customcallbacks)

        # Define the formatting
        layout = self._create_layout(inputwidgets, tableobjects)

        # Save as standalone html
        save(layout, save_path)
