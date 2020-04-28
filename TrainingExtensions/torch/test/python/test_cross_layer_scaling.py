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

import unittest.mock
import torch
from torchvision import models

from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import CrossLayerScaling, GraphSearchUtils
from aimet_torch.utils import get_layer_name
from aimet_torch.examples.mobilenet import MockMobileNetV2, MockMobileNetV1

import torch.nn as nn
import numpy as np


class MyModel(torch.nn.Module):
    def __init__(self):

        super(MyModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(10, 20, 3)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(20, 20, 3)
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(20, 20, 3)

        self.conv4 = torch.nn.Conv2d(20, 20, 3)

        self.conv5 = torch.nn.Conv2d(20, 20, 3)

        self.fc1 = torch.nn.Linear(5120, 10)

    def forward(self, x):

        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.relu1(x)

        # Non-linearity between conv and bn, not a candidate for fold
        x = self.conv2(x)
        x = self.relu2(x)

        # Case where BN can fold into an immediate downstream conv
        x = self.conv3(x)

        # No fold if there is a split between conv and BN
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


class TestTrainingExtensionsCrossLayerScaling(unittest.TestCase):

    def test_verify_cross_layer_scaling(self):
        # Get trained MNIST model

        torch.manual_seed(10)
        model = MyModel()
        # Call API
        model = model.eval()
        random_input = torch.rand(2, 10, 24, 24)
        # model.features[0].bias.data = torch.rand(64)
        baseline_output = model(random_input).detach().numpy()

        CrossLayerScaling.scale_cls_set_with_conv_layers((model.conv1, model.conv2))

        output_after_scaling = model(random_input).detach().numpy()

        range_conv1_after_scaling = np.amax(np.abs(model.conv1.weight.detach().cpu().numpy()), axis=(1, 2, 3))
        range_conv2_after_scaling = np.amax(np.abs(model.conv2.weight.detach().cpu().numpy()), axis=(0, 2, 3))

        assert (np.allclose(range_conv1_after_scaling, range_conv2_after_scaling))
        assert(np.allclose(baseline_output, output_after_scaling, rtol=1.e-2))

    def test_verify_cross_layer_for_multiple_pairs(self):
        # Get trained MNIST model

        model = MyModel()
        # Call API
        consecutive_layer_list = [(model.conv1, model.conv2),
                                  (model.conv3, model.conv4)]
        w1 = model.conv1.weight.detach().numpy()
        w2 = model.conv2.weight.detach().numpy()
        w3 = model.conv3.weight.detach().numpy()

        CrossLayerScaling.scale_cls_sets(consecutive_layer_list)

        # check if weights are updating
        assert not np.allclose(model.conv1.weight.detach().numpy(), w1)
        assert not np.allclose(model.conv2.weight.detach().numpy(), w2)
        assert not np.allclose(model.conv3.weight.detach().numpy(), w3)

    def test_verify_cross_layer_scaling_depthwise_separable_layer_mobilnet(self):
        torch.manual_seed(10)

        model = MockMobileNetV1()
        model = model.eval()

        model = model.to(torch.device('cpu'))
        model.model[0][0].bias = torch.nn.Parameter(torch.rand(model.model[0][0].weight.data.size()[0]))
        model.model[1][0].bias = torch.nn.Parameter(torch.rand(model.model[1][0].weight.data.size()[0]))
        model.model[1][3].bias = torch.nn.Parameter(torch.rand(model.model[1][3].weight.data.size()[0]))
        model.model[2][0].bias = torch.nn.Parameter(torch.rand(model.model[2][0].weight.data.size()[0]))
        model.model[2][3].bias = torch.nn.Parameter(torch.rand(model.model[2][3].weight.data.size()[0]))

        random_input = torch.rand(1, 3, 224, 224)
        baseline_output = model(random_input).detach().numpy()

        consecutive_layer_list = [(model.model[0][0], model.model[1][0], model.model[1][3]),
                                  (model.model[1][3], model.model[2][0], model.model[2][3])]

        for consecutive_layer in consecutive_layer_list:
            CrossLayerScaling.scale_cls_set_with_depthwise_layers(consecutive_layer)
            r1 = np.amax(np.abs(consecutive_layer[0].weight.detach().cpu().numpy()), axis=(1, 2, 3))
            r2 = np.amax(np.abs(consecutive_layer[1].weight.detach().cpu().numpy()), axis=(1, 2, 3))
            r3 = np.amax(np.abs(consecutive_layer[2].weight.detach().cpu().numpy()), axis=(0, 2, 3))
            assert (np.allclose(r1, r2))
            assert (np.allclose(r2, r3))

        output_after_scaling = model(random_input).detach().numpy()

        assert(np.allclose(baseline_output, output_after_scaling, rtol=1.e-2))

    def test_verify_cross_layer_scaling_depthwise_separable_layer_multiple_triplets(self):

        torch.manual_seed(10)

        model = MockMobileNetV1()
        model = model.eval()

        consecutive_layer_list = [(model.model[0][0], model.model[1][0], model.model[1][3]),
                                  (model.model[1][3], model.model[2][0], model.model[2][3])]

        w1 = model.model[0][0].weight.detach().numpy()
        w2 = model.model[1][3].weight.detach().numpy()
        w3 = model.model[2][3].weight.detach().numpy()

        CrossLayerScaling.scale_cls_sets(consecutive_layer_list)

        assert not np.allclose(model.model[0][0].weight.detach().numpy(), w1)
        assert not np.allclose(model.model[1][3].weight.detach().numpy(), w2)
        assert not np.allclose(model.model[2][3].weight.detach().numpy(), w3)

    def test_find_layer_groups_to_scale_for_network_with_residuals(self):

        torch.manual_seed(10)
        model = MockMobileNetV2()
        model.eval()

        fold_all_batch_norms(model, (1, 3, 224, 224))
        graph_search = GraphSearchUtils(model, (1, 3, 224, 224))
        layer_groups = graph_search.find_layer_groups_to_scale()
        self.assertEqual(4, len(layer_groups))
        self.assertIn([model.features[3].conv[0], model.features[3].conv[3], model.features[3].conv[6]], layer_groups)
        self.assertIn([model.features[4].conv[0], model.features[4].conv[3], model.features[4].conv[6]], layer_groups)
        self.assertIn([model.features[5].conv[0], model.features[5].conv[3], model.features[5].conv[6],
                       model.features[6][0]], layer_groups)


        for layer_group in layer_groups:
            print("Group ------- ")
            for module in layer_group:
                print("   " + get_layer_name(model, module))

    def test_find_cls_sets_vgg16(self):

        torch.manual_seed(10)
        model = models.vgg16()
        print(model)
        model.eval()

        graph_search = GraphSearchUtils(model, (1, 3, 224, 224))
        layer_groups = graph_search.find_layer_groups_to_scale()
        self.assertEqual(5, len(layer_groups))
        self.assertIn([model.features[0], model.features[2]], layer_groups)
        self.assertIn([model.features[5], model.features[7]], layer_groups)
        self.assertIn([model.features[10], model.features[12], model.features[14]], layer_groups)
        self.assertIn([model.features[17], model.features[19], model.features[21]], layer_groups)
        self.assertIn([model.features[24], model.features[26], model.features[28]], layer_groups)

        cls_sets = []
        for layer_group in layer_groups:

            cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
            cls_sets += cls_set

        for cls_set in cls_sets:
            print(cls_set)

        self.assertEqual(8, len(cls_sets))
        self.assertIn((model.features[0], model.features[2]), cls_sets)
        self.assertIn((model.features[5], model.features[7]), cls_sets)
        self.assertIn((model.features[10], model.features[12]), cls_sets)
        self.assertIn((model.features[12], model.features[14]), cls_sets)
        self.assertIn((model.features[17], model.features[19]), cls_sets)
        self.assertIn((model.features[19], model.features[21]), cls_sets)
        self.assertIn((model.features[24], model.features[26]), cls_sets)
        self.assertIn((model.features[26], model.features[28]), cls_sets)

        result = graph_search.is_relu_activation_present_in_cls_sets(cls_sets)
        print(result)

    def test_find_cls_sets_mobilenetv1(self):

        torch.manual_seed(10)

        model = MockMobileNetV1()
        model.eval()

        fold_all_batch_norms(model, (1, 3, 224, 224))

        graph_search = GraphSearchUtils(model, (1, 3, 224, 224))
        layer_groups = graph_search.find_layer_groups_to_scale()

        self.assertEqual(1, len(layer_groups))
        self.assertIn([model.model[0][0],
                       model.model[1][0],
                       model.model[1][3],
                       model.model[2][0],
                       model.model[2][3],
                       model.model[3][0],
                       model.model[3][3],
                       model.model[4][0],
                       model.model[4][3],
                       model.model[5][0],
                       model.model[5][3],
                       model.model[6][0],
                       model.model[6][3],
                       model.model[7][0],
                       model.model[7][3],
                       model.model[8][0],
                       model.model[8][3],
                       ], layer_groups)

        layer_pairs = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_groups[0])
        for layer_tuple in layer_pairs:
            print(layer_tuple)

    def test_auto_mobilenetv1(self):

        torch.manual_seed(10)
        model = MockMobileNetV1()
        model.eval()

        # BN fold
        fold_all_batch_norms(model, (1, 3, 224, 224))

        scale_factors = CrossLayerScaling.scale_model(model, (1, 3, 224, 224))
        self.assertEqual(8, len(scale_factors))
