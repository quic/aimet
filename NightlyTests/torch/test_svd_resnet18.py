#/usr/bin/env python3.5
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

import unittest
import torch
import torch.nn as nn

from torchvision import models

import aimet_torch.svd.svd_intf_defs_deprecated
from aimet_torch.svd import svd as svd_intf
from aimet_torch.examples.imagenet_dataloader import ImageNetDataLoader
from aimet_torch.utils import IterFirstX
from aimet_torch.examples.supervised_classification_pipeline import create_stand_alone_supervised_classification_evaluator


image_dir = './data/tiny-imagenet-200'
image_size = 224
batch_size = 256
num_workers = 16
# change iterations for early stopping
# if set to 'None', then iterate over entire validation data set
iterations = 3


def model_eval(model, early_stopping_iterations, use_cuda):
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
    criterion = nn.CrossEntropyLoss().cuda()
    evaluator = create_stand_alone_supervised_classification_evaluator(model, criterion, use_cuda)
    evaluator.run(val_loader)
    return evaluator.state.metrics['top_1_accuracy']


class SvdAcceptanceTest(unittest.TestCase):
    @unittest.skip
    def test_resnet_compression_using_svd(self):

        torch.cuda.empty_cache()

        # load pretrained resnet-18 on Imagenet dataset
        model_resnet18 = models.resnet18(pretrained=True)
        model_resnet18 = model_resnet18.to(torch.device('cuda'))
        compressed_model, stats = svd_intf.Svd.compress_model(model=model_resnet18, run_model=model_eval,
                                                              run_model_iterations=iterations,
                                                              input_shape=(1, 3, 224, 224),
                                                              compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                                                              cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                                                              layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme
                                                              .top_x_percent,
                                                              rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                                              percent_thresh=60, error_margin=2.0, num_rank_indices=20)

        path = '../data/resnet.pth.tar'
        torch.save(compressed_model, path)
        torch.save(compressed_model.state_dict(), '../data/resnet_state.pth')
        compressed_model_copy = torch.load(path)
        compressed_model_copy.load_state_dict(torch.load('../data/resnet_state.pth'))
        for (layer1, layer2)in zip( compressed_model_copy.state_dict(), compressed_model.state_dict()):
            if layer1 != layer2:
                raise ValueError('Model is not the same')

        # 1) compare compressed model with base model
        self.assertTrue(model_resnet18 is not compressed_model)

        for i in range(len(stats.per_rank_index) - 1):

            current_index_compression_ratio = stats.per_rank_index[i].model_compression_ratio
            next_index_compression_ratio = stats.per_rank_index[i + 1].model_compression_ratio
            # 2) compression ratio should increase as rank index is increased
            self.assertTrue(current_index_compression_ratio <= next_index_compression_ratio)

        # 3) 5 layers should be selected in this case MEMORY
        self.assertEqual(5, len(stats.per_rank_index[stats.best_rank_index].per_selected_layer))
