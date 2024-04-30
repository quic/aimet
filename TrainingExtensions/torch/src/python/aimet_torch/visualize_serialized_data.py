# -*- mode: python -*-
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

""" Class for visualizing after compression is completed"""
import pickle
import pandas as pd
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from aimet_common.compression_algo import CompressionAlgo
from aimet_common.bokeh_plots import BokehServerSession
from aimet_common import plotting_utils


class VisualizeCompression:
    """ Updates bokeh server session document and publishes graphs/tables to the server with session id compression. """

    def __init__(self, visualization_url):
        self.bokeh_session = BokehServerSession(visualization_url, session_id="compression")
        self.__document = self.bokeh_session.document

    def display_eval_scores(self, saved_eval_scores_dict_path):
        """
        Publishes the evaluation scores table to the server.

        :param saved_eval_scores_dict_path: file path to the evaluation scores for each layer
        :return: None
        """
        with open(saved_eval_scores_dict_path, 'rb') as infile:
            eval_scores_dict = pickle.load(infile)

        eval_scores_data_frame = pd.DataFrame.from_dict(eval_scores_dict).T
        eval_scores_data_frame.columns = eval_scores_data_frame.columns.map(str)
        eval_scores_data_frame.insert(0, 'layers', eval_scores_data_frame.index)

        source = ColumnDataSource(data=eval_scores_data_frame)
        columns = [TableColumn(field=Ci, title=Ci) for Ci in eval_scores_data_frame.columns]  # bokeh columns
        eval_scores_data_table = DataTable(source=source, columns=columns, width=1500)

        self.__document.add_root(eval_scores_data_table)

    def display_comp_ratio_plot(self, comp_ratio_list_path):
        """
        Publishes the optimal compression ratios to the server.

        :param comp_ratio_list_path: Path to the pkl file with compression ratios for each layer
        :return: None
        """
        layer_comp_ratio_list = CompressionAlgo.unpickle_comp_ratios_list(comp_ratio_list_path=comp_ratio_list_path)

        # visualize comp ratios vs layers in a plot and add it to a server session document.
        comp_ratios = []
        layer_names = []
        for layer_name, comp_ratio in layer_comp_ratio_list:
            comp_ratios.append(comp_ratio)
            layer_names.append(layer_name)

        plot = plotting_utils.plot_optimal_compression_ratios(comp_ratios, layer_names)
        self.__document.add_root(plot)
