# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020-2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

import os
import signal
import unittest
import numpy as np
import torch
import torch.nn as nn
from bokeh.models import Range1d
from bokeh.plotting import figure
from aimet_common.utils import AimetLogger, kill_process_with_name_and_port_number, start_bokeh_server_session
from aimet_common import bokeh_plots
from aimet_torch import plotting_utils
from aimet_torch import visualize_model
from aimet_common.bokeh_plots import BokehServerSession
from aimet_common.bokeh_plots import ProgressBar

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class CNNModel(torch.nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        # input channel, output channels, 5x5 square convolution
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 150, 5)
        self.conv3 = torch.nn.Conv2d(1, 60, 5)
        self.conv4 = torch.nn.Conv2d(6, 80, 5)


model = CNNModel()


class VisualizeNetwork(unittest.TestCase):
    def test_tensor_input(self):
        # 2 input channels, 6 output channels, 5*5 square convolution
        conv1 = torch.nn.Conv2d(2, 4, 5)

        # each column contains all the weights from one output channel
        conv1_weights = plotting_utils.get_weights(conv1)
        # print("conv1 weights after numpy reshaping", conv1_weights)

        # number of output channels, or columns in the reshaped matrix
        num_weights_out_channel1 = len(conv1_weights[0])
        total_weights_expected = np.prod(list(conv1.weight.shape))
        total_weights_actual = np.prod(conv1_weights.shape)

        # the length of any row should equal the number of output channels
        self.assertEqual(conv1.weight.shape[0], num_weights_out_channel1)

        # ensure the number of weights is the same before and after
        self.assertEqual(total_weights_expected, total_weights_actual)

    def test_progress_bar(self):
        visualization_url, process = start_bokeh_server_session(8002)
        bokeh_session = BokehServerSession(url=visualization_url, session_id="test")
        progress_bar = ProgressBar(total=10, bokeh_document=bokeh_session, title="testing", color="green")
        for i in range(10):
            progress_bar.update()
        progress_bar.update()
        self.assertEqual(progress_bar.calculate_percentage_complete(), 100.0)
        bokeh_session.server_session.close("test complete")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    def test_show_zoomed_in_plot_from_start(self):
        layout = bokeh_plots.PlotsLayout()

        # create a new plot with a range set with a tuple
        p = figure(plot_width=400, plot_height=400, x_range=(0, 20))

        # set a range using a Range1d
        p.y_range = Range1d(0, 15)

        p.circle([1, 2, 3, 4, 5, 25], [2, 5, 8, 2, 7, 50], size=10)
        # r = row(p)
        layout.layout = p
        # layout.add_row(p)
        layout.complete_layout()

    def test_invoke_progress_bar(self):
        visualization_url, process = start_bokeh_server_session(8002)
        bokeh_session = BokehServerSession(url=visualization_url, session_id="test")
        progress_bar = ProgressBar(80, title="Some Title Goes Here", color="green", bokeh_document=bokeh_session)

        for i in range(80):
            progress_bar.update()
        progress_bar.update()
        bokeh_session.server_session.close("test complete")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    def test_module_data_frame_mapping(self):
        layer_weights_map = plotting_utils.map_all_module_weights_to_data_frame(model)

        num_conv_and_linear_layers = 0
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear)):
                num_conv_and_linear_layers += 1

        # verify that there are the same number of data frames as there are conv and linear layers
        self.assertEqual(num_conv_and_linear_layers, len(layer_weights_map))

    def test_line_plot_visualizations_per_layer(self):
        results_dir = 'artifacts'
        if not os.path.exists('artifacts'):
            os.makedirs('artifacts')
        plot = visualize_model.visualize_relative_weight_ranges_to_identify_problematic_layers(model, results_dir)
