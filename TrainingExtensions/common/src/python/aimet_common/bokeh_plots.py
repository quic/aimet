# /usr/bin/env python3.5
# -*- mode: python -*-
# pylint: disable=E1136,E1137
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Classes for bokeh plots"""

import numpy as np
import pandas as pd
from bokeh.client import push_session
from bokeh.document import Document
from bokeh.layouts import column
from bokeh.models import TableColumn
from bokeh.models import DataTable as BokehDataTable
from bokeh.models import Div
from bokeh.models import Plot
from bokeh.models.glyphs import Rect

from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Title
from bokeh.plotting import figure


class BokehServerSession:
    """ Creates a unique bokeh session specified by a bokeh session id as input name and a port number"""

    def __init__(self, url: str, session_id: str = None, display: bool = True) -> object:
        """
        Create a BokehServerSession object. We will link this server session with a document, self.document,
        which allows changes to be pushed to the server.
        :param url: url for server
        :param session_id: unique session id for server session
        :param display: a bool variable to indicate if output is to be displayed immediately.
        """
        self.document = Document()
        self.server_session = push_session(self.document, url=url, session_id=session_id)
        if display:
            self.server_session.show()


class ProgressBar:
    """
    Progress bar showing both the percentage and the area wise completion of a task. Displays the progress of a long
    running operation, providing a visual cue that processing is underway.
    """

    def __init__(self, total: int, title: str, color: str, bokeh_session: object):
        """
        Initialize a ProgressBar instance.
        :param total: number of steps the progress bar is divided into. In other words, the number of times update should be called.
        :param title: the title of the progress bar
        :param color: the color of the progress bar
        :param bokeh_session: bokeh server session
        """
        self.color = color
        self.title = title
        self.total = total
        self.current_source_index = 0
        self.source = self.create_column_data_source()

        plot = Plot(plot_width=1000, plot_height=50, min_border=0, toolbar_location=None, outline_line_color=None)

        glyph = Rect(x="x_coordinate", y=0, width=1, height=1, angle=-0.0, fill_color="color", line_color="color",
                     line_alpha=0.3, fill_alpha=0.3)

        plot.add_glyph(self.source, glyph)

        self.title_object = Title()
        self.title_object.text = "     " + self.title
        plot.title = self.title_object

        bokeh_session.document.add_root(plot)

    def update(self):
        """
        Updates the table to represent the new area wise progress with self.color color.
        :return: None.
        """
        if self.current_source_index == self.total:
            return
        self.source.data["color"][self.current_source_index] = self.color
        self.source.data["color"] = list(self.source.data['color'])
        self.update_title()
        self.current_source_index += 1

    def update_title(self):
        """
        Updates the title of the progress bar to show a percentage complete.
        :return: None
        """
        self.title_object.text = "     " + self.title + "   " + str(self.calculate_percentage_complete()) + "%"

    def calculate_percentage_complete(self):
        """Calculate percentage complete and round it to the hundreths place"""
        if self.total == 0:
            return 100
        if self.current_source_index == self.total:
            return 100
        completed_portion = self.current_source_index + 1
        percentage_complete = completed_portion / self.total * 100
        percentage_truncated_after_hundreds = round(percentage_complete, 2)
        return percentage_truncated_after_hundreds

    def create_column_data_source(self):
        """
        Used when a ProgressBar is first initialized to create a data frame filling the table with necessary data to
        represent a white, or empty progress bar.
        :return: pandas data frame object.
        """
        data_frame = pd.DataFrame(index=np.arange(self.total), columns=["x", "color"])
        data_frame["color"] = ["white" for i in range(self.total)]
        data_frame["x_coordinate"] = [i for i in range(self.total)]
        return ColumnDataSource(data=data_frame)


class DataTable:
    """
    Datatable object synced with a server session that updates on the bokeh server every time update_table is called.
    """

    def __init__(self, num_rows: int, num_columns: int, column_names: list, bokeh_session: object,
                 row_index_names: list = None):
        """
        initialize a
        :param num_rows:
        :param num_columns:
        :param column_names:
        :param bokeh_session:
        :param row_index_names:
        """
        self.total = num_rows * num_columns
        self.row_names = row_index_names
        if row_index_names:
            data_frame = pd.DataFrame(index=np.arange(num_rows), columns=["index"] + column_names)
            data_frame["index"] = row_index_names
            self.row_index_to_row_name_map = self.map_row_names()
        else:
            data_frame = pd.DataFrame(index=np.arange(num_rows), columns=column_names)
        data_frame.fillna('', inplace=True)

        self.source = ColumnDataSource(data=data_frame)
        columns = [TableColumn(field=column_str, title=column_str) for column_str in data_frame.columns]  # bokeh columns
        self.data_table = BokehDataTable(source=self.source, columns=columns, width=1500)

        bokeh_session.document.add_root(self.data_table)

    def update_table(self, column_name, row, value):
        """
        Updates the data table to value at row, column entry
        :param column_name: Name of the column
        :param row: Row index
        :param value: Value to place in row, column entry in data table.
        :return: None
        """
        if isinstance(row, int):
            self.source.data[column_name][row] = value
            self.source.data[column_name] = list(self.source.data[column_name])
            return
        row_index = self.row_index_to_row_name_map[row]
        self.source.data[column_name][row_index] = value
        self.source.data[column_name] = list(self.source.data[column_name])

    def map_row_names(self):
        """
        Creates a maping between the user defined index, row_names, and the numerical index that it refers to in the
        'data table
        :param row_names: list of row names
        :return: mapping between row name strings and a numerical index
        """
        row_name_to_numerical_index_map = {}
        row_index_counter = 0
        for row in self.row_names:
            row_name_to_numerical_index_map[row] = row_index_counter
            row_index_counter += 1
        return row_name_to_numerical_index_map


class LinePlot:
    """ Creates an updating line plot with x,y coordinates for each point marked with dots."""

    def __init__(self, x_axis_label, y_axis_label, title, bokeh_session):
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.title = title
        self.bokeh_session = bokeh_session

        self.source = ColumnDataSource(data=dict(x=[], y=[]))

        self.plot = figure(x_axis_label=self.x_axis_label, y_axis_label=self.y_axis_label,
                           title=self.title,
                           tools="pan,box_zoom, crosshair,reset, save", width=1500)
        self.plot.circle(x="x", y="y", size=10, alpha=0.7, color="black", source=self.source)
        self.plot.line(x="x", y="y", source=self.source)

        self.title_object = Title()
        self.title_object.text = self.title
        self.plot.title = self.title_object

        LinePlot.style(self.plot)

        self.bokeh_session.document.add_root(self.plot)

    def update(self, new_x_coordinate, new_y_coordinate):
        """
        Updates the plot by adding a new x,y coordinate pair
        :param new_x_coordinate: New x value
        :param new_y_coordinate: New y value
        :return: None
        """
        # has new, identical-length updates for all columns in source
        new_data = {'x': [new_x_coordinate], 'y': [new_y_coordinate]}
        self.source.stream(new_data)

    def remove_plot(self):
        """
        Removes current plot object from document
        :return: None
        """
        self.bokeh_session.document.remove_root(self.plot)

    def update_title(self, new_title):
        """
        Update the title of the plot
        :param new_title: string title
        :return: None
        """
        self.title_object.text = new_title

    @staticmethod
    def style(p):
        """
        Style bokeh figure object p and return the styled object
        :param p: Bokeh figure object
        :return: Bokeh figure object
        """
        # Title
        p.title.align = 'center'
        p.title.text_font_size = '14pt'
        p.title.text_font = 'serif'

        # Axis titles
        p.xaxis.axis_label_text_font_size = '12pt'
        p.yaxis.axis_label_text_font_size = '12pt'

        # Tick labels
        p.xaxis.major_label_text_font_size = '10pt'
        p.yaxis.major_label_text_font_size = '10pt'

        return p



class PlotsLayout:
    """
    Keeps track of a layout (rows and columns of plot objects) and pushes them to a bokeh session once the layout is complete
    """

    def __init__(self):
        self.title = None
        self.layout = []

    def add_row(self, figures_list):
        """
        adds a row to self.layout
        :param figures_list: list of figure objects.
        :return: None.
        """
        self.layout.append(figures_list)

    def complete_layout(self):
        """
        complete a layout by adding self.layout to a server session document.
        :return:
        """
        if self.title is None:
            print(type(self.layout))
            if isinstance(self.layout, list):
                plot = self.layout
            else:
                plot = column(self.layout)
        else:
            my_session_with_title = self.add_title()
            return my_session_with_title
        return plot

    def add_title(self):
        """
        Add a title to the current layout.
        :return: layout wrapped with title div.
        """
        text_str = "<b>" + self.title + "</b>"
        wrap_layout_with_div = column(Div(text=text_str), column(self.layout))
        return wrap_layout_with_div
