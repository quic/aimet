# /usr/bin/env python3.5
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

""" Top level API for visualizing a pytorch model. """

import torch
from aimet_torch import plotting_utils
from aimet_torch.utils import get_layer_by_name
from aimet_common.bokeh_plots import BokehServerSession


def visualize_changes_after_optimization(old_model, new_model, visualization_url, selected_layers=None):
    """
    Visualizes changes before and after some optimization has been applied to a model.
    Visualization_url is in the form: http://<host name>:<port number>/

    :param old_model: pytorch model before optimization
    :param new_model: pytorch model after optimization
    :param visualization_url: user inputted url with session id set as optimization for the visualizations.
    :param selected_layers: a list of layers a user can choose to have visualized. If selected layers is None,
        all Linear and Conv layers will be visualized.
    :return: None
    """

    bokeh_session = BokehServerSession(url=visualization_url, session_id="optimization")
    server_document = bokeh_session.document
    if selected_layers:
        for name, module in new_model.named_modules():
            if name in selected_layers and hasattr(module, "weight"):
                old_model_module = get_layer_by_name(old_model, name)
                new_model_module = module
                plot = plotting_utils.visualize_changes_after_optimization_single_layer(name, old_model_module,
                                                                                        new_model_module)
                server_document.add_root(plot)

    else:
        for name, module in new_model.named_modules():
            if hasattr(module, "weight") and isinstance(module,
                                                        (torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear)):
                old_model_module = get_layer_by_name(old_model, name)
                new_model_module = module
                plot = plotting_utils.visualize_changes_after_optimization_single_layer(name, old_model_module,
                                                                                        new_model_module)
                server_document.add_root(plot)

    # returns bokeh session object, mostly for testing purposes, so the session can be closed in a test case.
    return bokeh_session


def visualize_weight_ranges(model, visualization_url, selected_layers=None):
    """
    Visualizes weight ranges for each layer through a scatter plot showing mean plotted against the standard deviation,
    the minimum plotted against the max, and a line plot with min, max, and mean for each output channel.
    Visualization_url is in the form: http://<host name>:<port number>/

    :param model: pytorch model
    :param visualization_url: user inputted url with session id set as optimization for the visualizations.
    :param selected_layers:  a list of layers a user can choose to have visualized. If selected layers is None,
        all Linear and Conv layers will be visualized.
    :return: None
    """

    bokeh_session = BokehServerSession(url=visualization_url, session_id="optimization")
    server_document = bokeh_session.document
    if selected_layers:
        for name, module in model.named_modules():
            if name in selected_layers and hasattr(module, "weight"):
                plot = plotting_utils.visualize_weight_ranges_single_layer(module, name)
                server_document.add_root(plot)
    else:
        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module,
                                                        (torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear)):
                plot = plotting_utils.visualize_weight_ranges_single_layer(module, name)
                server_document.add_root(plot)

    # returns bokeh session object, mostly for testing purposes, so the session can be closed in a test case.
    return bokeh_session


def visualize_relative_weight_ranges_to_identify_problematic_layers(model, visualization_url, selected_layers=None):
    """
    For each of the selected layers, publishes a line plot showing  weight ranges for each layer, summary statistics
    for relative weight ranges, and a histogram showing weight ranges of output channels
    with respect to the minimum weight range.
    Visualization_url is in the form: http://<host name>:<port number>/

    :param model: pytorch model
    :param visualization_url: user inputted url with session id set as optimization for the visualizations.
    :param selected_layers: a list of layers a user can choose to have visualized. If selected layers is None,
        all Linear and Conv layers will be visualized.
    :return: None
    """

    bokeh_session = BokehServerSession(url=visualization_url, session_id="optimization")
    server_document = bokeh_session.document
    # layer name -> module weights data frame mapping
    if not selected_layers:
        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module,
                                                        (torch.nn.modules.conv.Conv2d,
                                                         torch.nn.modules.linear.Linear)):
                plot = plotting_utils.visualize_relative_weight_ranges_single_layer(module, name)
                server_document.add_root(plot)
    else:
        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module,
                                                        (torch.nn.modules.conv.Conv2d,
                                                         torch.nn.modules.linear.Linear)) and name in selected_layers:
                plot = plotting_utils.visualize_relative_weight_ranges_single_layer(module, name)
                server_document.add_root(plot)

    return bokeh_session
