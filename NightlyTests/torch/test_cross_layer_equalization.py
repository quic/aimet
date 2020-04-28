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
""" Cross Layer Equalization acceptance tests for ResNet model. """

import os
import signal
import unittest
import copy
import torch
import numpy as np
import math
import torch.nn as nn
from torchvision import models
from aimet_torch import batch_norm_fold
from aimet_torch.cross_layer_equalization import CrossLayerScaling, HighBiasFold, equalize_model
from aimet_common.utils import start_bokeh_server_session
from aimet_common.bokeh_plots import BokehServerSession
from aimet_torch import visualize_model
from aimet_torch.examples.mobilenet import MobileNetV2


class TestCrossLayerEqualization(unittest.TestCase):
    """ Acceptance tests related to winnowing ResNet models. """

    def test_cross_layer_equalization_resnet(self):

        torch.manual_seed(10)
        model = models.resnet18(pretrained=True)

        model = model.eval()

        folded_pairs = batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224))
        bn_dict = {}
        for conv_bn in folded_pairs:
            bn_dict[conv_bn[0]] = conv_bn[1]

        self.assertFalse(isinstance(model.layer2[0].bn1, torch.nn.BatchNorm2d))

        w1 = model.layer1[0].conv1.weight.detach().numpy()
        w2 = model.layer1[0].conv2.weight.detach().numpy()
        w3 = model.layer1[1].conv1.weight.detach().numpy()

        cls_set_info_list = CrossLayerScaling.scale_model(model, (1, 3, 224, 224))

        # check if weights are updating
        assert not np.allclose(model.layer1[0].conv1.weight.detach().numpy(), w1)
        assert not np.allclose(model.layer1[0].conv2.weight.detach().numpy(), w2)
        assert not np.allclose(model.layer1[1].conv1.weight.detach().numpy(), w3)

        b1 = model.layer1[0].conv1.bias.data
        b2 = model.layer1[1].conv2.bias.data

        HighBiasFold.bias_fold(cls_set_info_list, bn_dict)

        for i in range(len(model.layer1[0].conv1.bias.data)):
            self.assertTrue(model.layer1[0].conv1.bias.data[i] <= b1[i])

        for i in range(len(model.layer1[1].conv2.bias.data)):
            self.assertTrue(model.layer1[1].conv2.bias.data[i] <= b2[i])

    def test_cross_layer_equalization_mobilenet_v2(self):
        torch.manual_seed(10)

        model = MobileNetV2().to(torch.device('cpu'))
        print(model)

        model = model.eval()
        equalize_model(model, (1, 3, 224, 224))

    def test_cross_layer_equalization_vgg(self):
        torch.manual_seed(10)
        model = models.vgg16(pretrained=True).to(torch.device('cpu'))
        model = model.eval()
        equalize_model(model, (1, 3, 224, 224))

    @unittest.skip("Takes 1 min 42 secs to run")
    def test_cross_layer_equalization_mobilenet_v2_visualize_after_optimization(self):
        bokeh_visualizations_url, process = start_bokeh_server_session(8006)
        torch.manual_seed(10)
        model = MobileNetV2().to(torch.device('cpu'))
        bokeh_session = BokehServerSession(bokeh_visualizations_url, session_id="cle")
        model = model.eval()
        model_copy = copy.deepcopy(model)

        # model_copy_again = copy.deepcopy(model)
        batch_norm_fold.fold_all_batch_norms(model_copy, (1, 3, 224, 224))
        equalize_model(model, (1, 3, 224, 224))
        visualize_model.visualize_changes_after_optimization(model_copy, model, bokeh_visualizations_url)
        bokeh_session.server_session.close("test complete")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    def test_cross_layer_equalization_resnet18_visualize_to_identify_problem_layers(self):
        bokeh_visualizations_url, process = start_bokeh_server_session(6008)
        torch.manual_seed(10)
        model = models.resnet18(pretrained=True)
        model = model.eval()

        batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224))

        bokeh_server_session = visualize_model.visualize_relative_weight_ranges_to_identify_problematic_layers(model,
                                                                                                               bokeh_visualizations_url)
        bokeh_server_session.server_session.close("test complete")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)




