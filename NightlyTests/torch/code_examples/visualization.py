# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" code examples for visualization APIs """

import copy
from decimal import Decimal
import torch
from torchvision import models
import aimet_common.defs
import aimet_torch.defs
import aimet_torch.utils
from aimet_common.utils import start_bokeh_server_session
from aimet_torch.compress import ModelCompressor
from aimet_torch.visualize_serialized_data import VisualizeCompression


from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.examples.imagenet_dataloader import ImageNetDataLoader
from aimet_torch.examples.supervised_classification_pipeline import \
    create_stand_alone_supervised_classification_evaluator
from aimet_torch.utils import IterFirstX
from aimet_torch import batch_norm_fold
from aimet_torch import visualize_model

image_dir = './data/tiny-imagenet-200'
image_size = 224
batch_size = 5
num_workers = 1


def evaluate(model, early_stopping_iterations, use_cuda):
    """
    :param model: model to be evaluated
    :param early_stopping_iterations: if None, data loader will iterate over entire validation data
    :return: top_1_accuracy on validation data
    """

    data_loader = ImageNetDataLoader(image_dir, image_size, batch_size, num_workers)
    if early_stopping_iterations is not None:
        # wrapper around validation data loader to run only 'X' iterations to save time
        val_loader = IterFirstX(data_loader.val_loader, early_stopping_iterations)
    else:
        # iterate over entire validation data set
        val_loader = data_loader.val_loader

    criterion = torch.nn.CrossEntropyLoss().cuda()
    evaluator = create_stand_alone_supervised_classification_evaluator(model, criterion, use_cuda=use_cuda)
    evaluator.run(val_loader)
    return evaluator.state.metrics['top_1_accuracy']


def visualize_changes_in_model_after_and_before_cle():
    """
    Code example for visualizating model before and after Cross Layer Equalization optimization
    """
    visualization_url, process = start_bokeh_server_session(8002)
    model = models.resnet18(pretrained=True).to(torch.device('cpu'))
    model = model.eval()
    model_copy = copy.deepcopy(model)

    batch_norm_fold.fold_all_batch_norms(model_copy, (1, 3, 224, 224))

    equalize_model(model, (1, 3, 224, 224))
    visualize_model.visualize_changes_after_optimization(model_copy, model, visualization_url)


def visualize_weight_ranges_model():
    """
    Code example for model visualization
    """
    visualization_url, process = start_bokeh_server_session(8002)
    model = models.resnet18(pretrained=True).to(torch.device('cpu'))
    model = model.eval()

    batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224))

    # Usually it is observed that if we do BatchNorm fold the layer's weight range increases.
    # This helps in visualizing layer's weight
    visualize_model.visualize_weight_ranges(model, visualization_url)


def visualize_relative_weight_ranges_model():
    """
    Code example for model visualization
    """
    visualization_url, process = start_bokeh_server_session(8002)
    model = models.resnet18(pretrained=True).to(torch.device('cpu'))
    model = model.eval()

    batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224))

    # Usually it is observed that if we do BatchNorm fold the layer's weight range increases.
    # This helps in finding layers which can be equalized to get better performance on hardware
    visualize_model.visualize_relative_weight_ranges_to_identify_problematic_layers(model, visualization_url)


def model_compression_with_visualization():
    """
    Code example for compressing a model with a visualization url provided.
    """
    visualization_url, process = start_bokeh_server_session(8002)

    ImageNetDataLoader(image_dir, image_size, batch_size, num_workers)
    input_shape = (1, 3, 224, 224)
    model = models.resnet18(pretrained=True).to(torch.device('cuda'))

    modules_to_ignore = [model.conv1]

    greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.65),
                                                                num_comp_ratio_candidates=10,
                                                                saved_eval_scores_dict=
                                                               '../data/resnet18_eval_scores.pkl')

    auto_params = aimet_torch.defs.SpatialSvdParameters.AutoModeParams(greedy_params,
                                                                       modules_to_ignore=modules_to_ignore)

    params = aimet_torch.defs.SpatialSvdParameters(aimet_torch.defs.SpatialSvdParameters.Mode.auto, auto_params,
                                                   multiplicity=8)

    # If no visualization URL is provided, during model compression execution no visualizations will be published.
    ModelCompressor.compress_model(model=model, eval_callback=evaluate, eval_iterations=5,
                                   input_shape=input_shape,
                                   compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
                                   cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                   visualization_url=None)

    comp_ratios_file_path = './data/greedy_selection_comp_ratios_list.pkl'
    eval_scores_path = '../data/resnet18_eval_scores.pkl'

    # A user can visualize the eval scores dictionary and optimal compression ratios by executing the following code.
    compression_visualizations = VisualizeCompression(visualization_url)
    compression_visualizations.display_eval_scores(eval_scores_path)
    compression_visualizations.display_comp_ratio_plot(comp_ratios_file_path)