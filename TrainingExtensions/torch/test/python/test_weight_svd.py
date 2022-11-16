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

import unittest
from unittest.mock import create_autospec
import logging
from decimal import Decimal

import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import pytest
import copy

import aimet_common.libpymo as pymo
import aimet_common.defs
from aimet_common import cost_calculator as cc
from aimet_common.defs import LayerCompRatioPair
from aimet_common.utils import AimetLogger
import aimet_torch.svd.svd_intf_defs_deprecated
from aimet_torch.examples import mnist_torch_model as mnist_model
from aimet_torch.utils import create_rand_tensors_given_shapes, get_device
from aimet_torch import pymo_utils
from aimet_torch.svd import layer_selector_deprecated as ls, svd as svd_intf, svd_impl as s
from aimet_torch.layer_database import LayerDatabase, Layer
from aimet_torch.svd import svd_pruner_deprecated
from aimet_torch.svd import rank_selector as rank_select
from aimet_torch.svd.svd_pruner import WeightSvdPruner

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=(2, 2))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = functional.relu(functional.max_pool2d(self.conv1(x), 2))
        x = functional.relu(functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = functional.relu(self.fc1(x))
        x = functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return functional.log_softmax(x, dim=1)


class MnistSequentialModel(nn.Module):
    def __init__(self):
        super(MnistSequentialModel, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=(2, 2)),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.classfier = nn.Sequential(
            nn.Linear(3136, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.classfier(x)
        return x


class TestTrainingExtensionsSvd(unittest.TestCase):

    def test_pick_compression_layers_top_x_percent(self):

        logger.debug(self.id())
        model = MnistModel().to("cpu")

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        with unittest.mock.patch('aimet_torch.svd.layer_selector_deprecated.LayerSelectorDeprecated._perform_layer_selection'):
            layer_selector = ls.LayerSelectorDeprecated(
                aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_x_percent, None, layer_db,
                percent_thresh=None)

        # 100 % threshold
        picked_layers = layer_selector._pick_compression_layers(run_model=mnist_model.evaluate, cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                                                                layer_select_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_x_percent,
                                                                percent_thresh=100)

        self.assertEqual(model.fc1, picked_layers[0].module)
        self.assertEqual(model.conv2, picked_layers[1].module)
        self.assertEqual(model.fc2, picked_layers[2].module)
        self.assertEqual(3, len(picked_layers))

        # 80% criterion

        picked_layers = layer_selector._pick_compression_layers(run_model=mnist_model.evaluate, cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                                                                layer_select_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_x_percent,
                                                                percent_thresh=80)

        self.assertEqual(model.conv2, picked_layers[0].module)
        self.assertEqual(model.fc2, picked_layers[1].module)
        self.assertEqual(2, len(picked_layers))

    def test_pick_compression_layers_top_n_layers(self):

        # Memory
        logger.debug(self.id())
        model = MnistModel().to("cpu")

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        with unittest.mock.patch('aimet_torch.svd.layer_selector_deprecated.LayerSelectorDeprecated._perform_layer_selection'):
            layer_selector = ls.LayerSelectorDeprecated(
                aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers, None, layer_db,
                num_layers=2)

        picked_layers = layer_selector._pick_compression_layers(run_model=mnist_model.evaluate,
                                                                cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                                                                layer_select_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                                                                num_layers=2)

        self.assertEqual(picked_layers[0].module, model.fc1)
        self.assertEqual(picked_layers[1].module, model.conv2)
        self.assertEqual(2, len(picked_layers))

        # MAC
        picked_layers = layer_selector._pick_compression_layers(run_model=mnist_model.evaluate, cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.mac,
                                                                layer_select_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                                                                num_layers=2)

        self.assertEqual(picked_layers[0].module, model.conv2)
        self.assertEqual(picked_layers[1].module, model.fc1)
        self.assertEqual(2, len(picked_layers))

    def test_pick_compression_layers_manual(self):

        logger.debug(self.id())
        model = MnistModel().to("cpu")

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        with unittest.mock.patch('aimet_torch.svd.layer_selector_deprecated.LayerSelectorDeprecated._perform_layer_selection'):
            layer_selector = ls.LayerSelectorDeprecated(
                aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual, None, layer_db,
                layers_to_compress=[model.conv2])

        picked_layers = layer_selector._pick_compression_layers(run_model=mnist_model.evaluate, cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                                                                layer_select_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                                                layers_to_compress=[model.conv2])
        self.assertEqual(1, len(picked_layers))
        self.assertEqual(picked_layers[0].module, model.conv2)

    def test_split_conv_layer_with_mo(self):

        logger.debug(self.id())
        model = mnist_model.Net().to("cpu")

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        with unittest.mock.patch('aimet_torch.svd.layer_selector_deprecated.LayerSelectorDeprecated'):
            svd = s.SvdImpl(model=model, run_model=mnist_model.evaluate, run_model_iterations=1,
                            input_shape=(1, 1, 28, 28),
                            compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                            cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                            layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                            num_layers=2)

        conv2 = layer_db.find_layer_by_module(model.conv2)
        pymo_utils.PymoSvdUtils.configure_layers_in_pymo_svd([conv2], aimet_common.defs.CostMetric.mac, svd._svd_lib_ref)

        split_layer = svd_pruner_deprecated.DeprecatedSvdPruner
        seq, conv_a, conv_b = split_layer.prune_layer(conv2, 28, svd._svd_lib_ref)

        print('\n')
        weight_arr = conv_a.module.weight.detach().numpy().flatten()
        weight_arr = weight_arr[0:10]
        print(weight_arr)

        self.assertEqual((28, model.conv2.in_channels, 1, 1), conv_a.module.weight.shape)
        self.assertEqual([28], list(conv_a.module.bias.shape))
        self.assertEqual((model.conv2.out_channels, 28, 5, 5), conv_b.module.weight.shape)
        self.assertEqual([model.conv2.out_channels], list(conv_b.module.bias.shape))

        self.assertEqual(model.conv2.stride, conv_a.module.stride)
        self.assertEqual(model.conv2.stride, conv_b.module.stride)

        self.assertEqual((0, 0), conv_a.module.padding)
        self.assertEqual(model.conv2.padding, conv_b.module.padding)

        self.assertEqual((1, 1), conv_a.module.kernel_size)
        self.assertEqual(model.conv2.kernel_size, conv_b.module.kernel_size)

    def test_split_fc_layer_without_mo(self):

        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        logger.debug(self.id())
        model = MnistModel().to("cpu")

        with unittest.mock.patch('aimet_torch.layer_database.LayerDatabase'):
            with unittest.mock.patch('aimet_torch.svd.layer_selector_deprecated.LayerSelectorDeprecated'):
                svd = s.SvdImpl(model=model, run_model=None, run_model_iterations=1, input_shape=(1, 1, 28, 28),
                                compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                                cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                                layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                                num_layers=2)

        layer_attr = Layer(model.fc1, id(model.fc1), [3136, 1024, 1, 1])

        svd._svd_lib_ref = create_autospec(pymo.Svd, instance=True)
        split_weights = [np.zeros((400, model.fc1.in_features)).flatten().tolist(),
                         np.zeros((model.fc1.out_features, 400)).flatten().tolist()]
        svd._svd_lib_ref.SplitLayerWeights.return_value = split_weights

        split_biases = [np.zeros(400).flatten().tolist(),
                        np.zeros(model.fc1.out_features).flatten().tolist()]
        svd._svd_lib_ref.SplitLayerBiases.return_value = split_biases

        split_layer = svd_pruner_deprecated.DeprecatedSvdPruner

        seq, layer_a_attr, layer_b_attr = split_layer.prune_layer(layer_attr, 400, svd_lib_ref=svd._svd_lib_ref)

        self.assertEqual((400, model.fc1.in_features), seq[0].weight.shape)
        self.assertEqual([400], list(seq[0].bias.shape))
        self.assertEqual((model.fc1.out_features, 400), seq[1].weight.shape)
        self.assertEqual([model.fc1.out_features], list(seq[1].bias.shape))

        self.assertEqual(layer_a_attr.module, seq[0])
        self.assertEqual(layer_b_attr.module, seq[1])

    @unittest.skip
    def test_create_compressed_model(self):
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        logger.debug(self.id())
        model = MnistModel().to("cpu")

        with unittest.mock.patch('aimet_torch.svd.layer_database.LayerDatabase'):
            with unittest.mock.patch('aimet_torch.svd.layer_selector_deprecated.LayerSelectorDeprecated'):
                svd = s.SvdImpl(model=model, run_model=None, run_model_iterations=1, input_shape=(1, 1, 28, 28),
                                compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                                cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                                layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                                num_layers=2)

        ls.LayerSelectorDeprecated._pick_compression_layers = create_autospec(ls.LayerSelectorDeprecated._pick_compression_layers)
        layer_attr1 = Layer(model.fc2, id(model.fc2), model.fc2.weight.shape)
        layer_attr1.parent_module = model
        layer_attr1.var_name_of_module_in_parent = "fc2"
        layer_attr1.output_shape = [0, 0, 1, 1]
        layer_attr1.name = 'fc2'

        layer_attr2 = Layer(model.conv2, id(model.conv2), model.conv2.weight.shape)
        layer_attr2.parent_module = model
        layer_attr2.var_name_of_module_in_parent = "conv2"
        layer_attr2.name = 'conv2'
        layer_attr1.output_shape = [0, 0, 14, 14]

        ls.LayerSelectorDeprecated._pick_compression_layers.return_value = [layer_attr1, layer_attr2]

        svd._compressible_layers = {id(model.conv2): layer_attr2,
                                    id(model.fc2):   layer_attr1}

        ls.LayerSelectorDeprecated._perform_layer_selection(model)

        svd._select_candidate_ranks(20)
        svd_rank_pair_dict = {'conv2': (31,0), 'fc2': (9,0)}
        c_model, c_layer_attr, _ = svd._create_compressed_model(svd_rank_pair_dict)

        self.assertTrue(c_model is not model)
        self.assertTrue(c_model.conv1 is not model.conv1)
        self.assertTrue(c_model.conv2 is not model.conv2)

        self.assertFalse(isinstance(svd._model, nn.Sequential))
        self.assertEqual((9, 1024), c_model.fc2[0].weight.shape)
        self.assertEqual([9], list(c_model.fc2[0].bias.shape))
        self.assertEqual((10, 9), c_model.fc2[1].weight.shape)
        self.assertEqual([10], list(c_model.fc2[1].bias.shape))

        self.assertEqual((31, 32, 1, 1), c_model.conv2[0].weight.shape)
        self.assertEqual([31], list(c_model.conv2[0].bias.shape))
        self.assertEqual((64, 31, 5, 5), c_model.conv2[1].weight.shape)
        self.assertEqual([64], list(c_model.conv2[1].bias.shape))

        self.assertEqual(svd._model.conv1.weight.shape, c_model.conv1.weight.shape)
        self.assertEqual(svd._model.fc1.weight.shape, c_model.fc1.weight.shape)

        # Expect double the number of layers in layer_attr_list
        self.assertEqual(4, len(c_layer_attr))

    def test_svd_with_mo(self):

        logger.debug(self.id())
        model = MnistModel().to("cpu")

        svd = s.SvdImpl(model=model, run_model=mnist_model.evaluate, run_model_iterations=1, input_shape=(1, 1, 28, 28),
                        compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                        cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                        layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_x_percent,
                        percent_thresh=60)

        c_model, svd_stats = svd.compress_net(rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                              num_rank_indices=20, error_margin=10)

        # Log the SVD Statistics
        # Test passing an existing logger.
        # In this case the TestLogger is being passed in as the logger.
        svd_stats.pretty_print(logger=logger)

    def test_svd_sequential_with_mo(self):

        logger.debug(self.id())
        model = MnistSequentialModel().to("cpu")
        svd = s.SvdImpl(model=model, run_model=mnist_model.evaluate, run_model_iterations=1, input_shape=(1, 1, 28, 28),
                        compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                        cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                        layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_x_percent,
                        percent_thresh=60)

        c_model, svd_stats = svd.compress_net(rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                              num_rank_indices=20, error_margin=10)

        # Log the  SVD Statistics
        # Do not pass in a logger.
        # In this case the default root logger will be used.
        svd_stats.pretty_print(logger=None)

    def test_set_parent_attribute_two_deep(self):
        """With a two-deep model"""
        class SubNet(nn.Module):
            def __init__(self):
                super(SubNet, self).__init__()
                self.conv1 = nn.Conv2d(30, 40, 5)
                self.conv2 = nn.Conv2d(40, 50, kernel_size=5)

            def forward(self, *inputs):
                pass

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = SubNet()
                self.fc1 = nn.Linear(320, 50)
                self.subnet2 = SubNet()
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                pass

        net = Net()
        model = net.to("cpu")

        # create layer attribute
        output_activation_shape = None

        layers = {id(model.subnet1.conv2): Layer(model.subnet1.conv2, id(model.subnet1.conv2),
                                                 output_activation_shape),
                  id(model.subnet2.conv1): Layer(model.subnet2.conv1, id(model.subnet2.conv1),
                                                 output_activation_shape),
                  id(model.fc2):           Layer(model.fc2, id(model.fc2),
                                                 output_activation_shape)}

        LayerDatabase.set_reference_to_parent_module(model, layers)

        # child : model.subnet1.conv2 --> parent : model.subnet1
        self.assertEqual(model.subnet1, layers[id(model.subnet1.conv2)].parent_module)
        # child : model.subnet2.conv1 --> parent : model.subnet2
        self.assertEqual(model.subnet2, layers[id(model.subnet2.conv1)].parent_module)
        # child : model.fc2 --> parent : model
        self.assertEqual(model, layers[id(model.fc2)].parent_module)

    def test_set_attributes_with_sequentials(self):
        """With a one-deep model"""
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Conv2d(10, 20, 5)
                )
                self.fc1 = nn.Linear(320, 50)
                self.subnet2 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                pass

        net = Net()
        model = net.to("cpu")

        # create layer attribute
        output_activation_shape = None

        layers = {id(model.subnet1[2]): Layer(model.subnet1[2], id(model.subnet1[2]),
                                              output_activation_shape),
                  id(model.subnet2[0]): Layer(model.subnet2[0], id(model.subnet2[0]),
                                              output_activation_shape),
                  id(model.fc2): Layer(model.fc2, id(model.fc2),
                                       output_activation_shape)}

        LayerDatabase.set_reference_to_parent_module(model, layers)

        # child : model.subnet1.2 --> parent : model.subnet1
        self.assertEqual(model.subnet1, layers[id(model.subnet1[2])].parent_module)
        # child : model.subnet2.1 --> parent : model.subnet2
        self.assertEqual(model.subnet2, layers[id(model.subnet2[0])].parent_module)
        # child : model.fc2 --> parent : model
        self.assertEqual(model, layers[id(model.fc2)].parent_module)

    def test_set_parent_attribute_with_sequential_two_deep(self):
        """With a two-deep model"""
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, 5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.subnet1 = nn.Sequential(
                    nn.Conv2d(1, 10, 5),
                    nn.ReLU(),
                    nn.Sequential(
                        nn.Conv2d(1, 10, 5),
                        nn.ReLU(),
                        nn.Conv2d(20, 50, 5)
                    ),
                    nn.Conv2d(1, 10, 5)
                )
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, *inputs):
                pass

        net = Net()
        model = net.to("cpu")

        # create layer attribute
        output_activation_shape = None

        layers = {id(model.subnet1[0]):     Layer(model.subnet1[0], None, output_activation_shape),
                  id(model.subnet1[2][0]):  Layer(model.subnet1[2][0], None, output_activation_shape),
                  id(model.subnet1[2][2]):  Layer(model.subnet1[2][2], None, output_activation_shape)}

        LayerDatabase.set_reference_to_parent_module(model, layers)
        # child : model.subnet1.0 --> parent : model.subnet1
        self.assertEqual(model.subnet1, layers[id(model.subnet1[0])].parent_module)
        # child : model.subnet1.2.0 --> parent : model.subnet1.2
        self.assertEqual(model.subnet1[2], layers[id(model.subnet1[2][0])].parent_module)
        # child : model.subnet1.2.2 --> parent : model.subnet1.2
        self.assertEqual(model.subnet1[2], layers[id(model.subnet1[2][2])].parent_module)

    def test_choose_best_ranks(self):

        model = MnistModel().to("cpu")

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        run_model_return_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        run_model = unittest.mock.Mock(side_effect=run_model_return_values)

        with unittest.mock.patch('aimet_torch.layer_database.LayerDatabase'):
            with unittest.mock.patch('aimet_torch.svd.layer_selector_deprecated.LayerSelectorDeprecated'):
                svd = s.SvdImpl(model=model, run_model=run_model, run_model_iterations=1, input_shape=(1, 1, 28, 28),
                                compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                                cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                                layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                                num_layers=2)

        svd._network_cost = (500, 500)

        svd._svd_lib_ref = create_autospec(pymo.Svd, instance=True)
        with unittest.mock.patch('aimet_torch.svd.model_stats_calculator.ModelStats.compute_compression_ratio') as compute_compression_ratio:
            with unittest.mock.patch('aimet_torch.svd.svd_pruner_deprecated.ModelPruner.create_compressed_model') as create_compressed_model:
                with unittest.mock.patch('aimet_torch.svd.rank_selector.RankSelector._select_candidate_ranks') as select_candidate_ranks:
                    select_candidate_ranks.return_value = 20
                    compute_compression_ratio.side_effect = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                    create_compressed_model.return_value = None, None, None
                    rank_selector = rank_select.RankSelector(svd_lib_ref=svd._svd_lib_ref)
                    rank_selector.choose_best_rank(model=model, run_model=run_model, run_model_iterations=1,
                                                   use_cuda=False, metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                                                   error_margin=1,
                                                   baseline_perf=0.5, num_rank_indices=20, database=layer_db)

    def test_validate_params(self):

        si = svd_intf
        model = MnistModel()

        # All the sunny day possibilities
        # Manual - Manual
        si.Svd._validate_layer_rank_params(model,
                                           layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                           layers_to_compress=[],
                                           rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.manual,
                                           layer_rank_list=[])

        # Manual - Auto
        si.Svd._validate_layer_rank_params(model,
                                           layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                           layers_to_compress=[],
                                           rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                           error_margin=None, num_rank_indices=None)

        # top_x_percent - Manual
        si.Svd._validate_layer_rank_params(model,
                                           layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_x_percent,
                                           percent_thresh=0,
                                           rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                           error_margin=None, num_rank_indices=None)

        # top_n_layers - Manual
        si.Svd._validate_layer_rank_params(model,
                                           layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                                           num_layers=1,
                                           rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                           error_margin=None, num_rank_indices=None)

        # All the error possibilities
        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                               aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.manual)

        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                               layers_to_compress=None,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.manual,
                                               error_margin=None)

        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                               layers_to_compress=None,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.manual,
                                               num_rank_indices=None)

        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                               layers_to_compress=None,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                               num_rank_indices=None)

        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                               layers_to_compress=None,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                               error_margin=None)

        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                               layers_to_compress=None,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                               layer_rank_list=None)

        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                                               percent_thresh=None,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                               error_margin=None, num_rank_indices=None)

        # negative num_layers
        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                                               num_layers=-1,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                               error_margin=None, num_rank_indices=None)

        # need to select atleast one layer
        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                                               num_layers=0,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                               error_margin=None, num_rank_indices=None)

        # more layers than in the model (5)
        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                                               num_layers=6,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                               error_margin=None, num_rank_indices=None)

        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_x_percent,
                                               num_layers=None,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                               error_margin=None, num_rank_indices=None)

        # negative percent
        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_x_percent,
                                               percent_thresh=-1,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                               error_margin=None, num_rank_indices=None)

        # greater than 100%
        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_x_percent,
                                               percent_thresh=101,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                               error_margin=None, num_rank_indices=None)

        with pytest.raises(ValueError):
            si.Svd._validate_layer_rank_params(model,
                                               aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                                               num_layers=None, percent_thresh=None,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                               error_margin=None, num_rank_indices=None)

    def test_compress_model_no_iterations(self):

        model = MnistModel().to("cpu")

        with pytest.raises(ValueError):
            _, _ = svd_intf.Svd.compress_model(model=model, run_model=mnist_model.evaluate, run_model_iterations=0,
                                               input_shape=(1, 1, 28, 28),
                                               compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                                               cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.mac,
                                               layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                               rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                               layers_to_compress=[model.conv2, model.fc2], num_rank_indices=20,
                                               error_margin=100)

    def test_compress_model(self):

        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        logger.debug(self.id())
        model = MnistModel().to("cpu")

        c_model, stats = svd_intf.Svd.compress_model(model=model, run_model=mnist_model.evaluate,
                                                     run_model_iterations=1,
                                                     input_shape=(1, 1, 28, 28),
                                                     compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                                                     cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.mac,
                                                     layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                                     rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                                     layers_to_compress=[model.conv2, model.fc2], num_rank_indices=20,
                                                     error_margin=100)

        self.assertTrue(c_model.conv2[0].bias is not None)
        self.assertTrue(c_model.conv2[1].bias is not None)

        self.assertTrue(c_model.fc2[0].bias is not None)
        self.assertTrue(c_model.fc2[1].bias is not None)

        self.assertEqual(2, len(stats.per_rank_index[0].per_selected_layer))
        self.assertEqual('conv2', stats.per_rank_index[0].per_selected_layer[0].layer_name)
        self.assertEqual('fc2', stats.per_rank_index[0].per_selected_layer[1].layer_name)

    def test_compress_model_no_bias(self):

        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        logger.debug(self.id())
        model = MnistModel().to("cpu")
        model.conv2.bias = None
        model.fc2.bias = None

        c_model, stats = svd_intf.Svd.compress_model(model=model, run_model=mnist_model.evaluate,
                                                     run_model_iterations=1,
                                                     input_shape=(1, 1, 28, 28),
                                                     compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                                                     cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.mac,
                                                     layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                                     rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                                     layers_to_compress=[model.conv2, model.fc2], num_rank_indices=20,
                                                     error_margin=100)

        self.assertTrue(c_model.conv2[0].bias is None)
        self.assertTrue(c_model.conv2[1].bias is None)

        self.assertTrue(c_model.fc2[0].bias is None)
        self.assertTrue(c_model.fc2[1].bias is None)

        self.assertEqual(2, len(stats.per_rank_index[0].per_selected_layer))
        self.assertEqual('conv2', stats.per_rank_index[0].per_selected_layer[0].layer_name)
        self.assertEqual('fc2', stats.per_rank_index[0].per_selected_layer[1].layer_name)

    def test_compress_model_with_stride(self):
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        logger.debug(self.id())
        model = MnistModel().to("cpu")

        # Change the model to add a stride to conv2, and adjust the input dimensions of the next layer accordingly
        model.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=(2, 2), stride=(2, 2))
        model.fc1 = nn.Linear(3*3*64, 1024)

        c_model, stats = svd_intf.Svd.compress_model(model=model, run_model=mnist_model.evaluate,
                                                     run_model_iterations=1,
                                                     input_shape=(1, 1, 28, 28),
                                                     compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                                                     cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.mac,
                                                     layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.manual,
                                                     rank_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.RankSelectionScheme.auto,
                                                     layers_to_compress=[model.conv2, model.fc2], num_rank_indices=20,
                                                     error_margin=100)

        self.assertEqual(2, len(stats.per_rank_index[0].per_selected_layer))
        self.assertEqual('conv2', stats.per_rank_index[0].per_selected_layer[0].layer_name)
        self.assertEqual('fc2', stats.per_rank_index[0].per_selected_layer[1].layer_name)

    @pytest.mark.cuda
    def test_model_allocation_gpu(self):

        model = MnistModel().to("cuda")
        svd = s.SvdImpl(model=model, run_model=mnist_model.evaluate, run_model_iterations=1, input_shape=(1, 1, 28, 28),
                        compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                        cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                        layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                        num_layers=2)

        self.assertTrue(svd._is_model_on_gpu())
        # copy one layer to CPU
        model.conv1.to("cpu")
        self.assertFalse(svd._is_model_on_gpu())

        model = MnistModel().to("cpu")
        svd = s.SvdImpl(model=model, run_model=mnist_model.evaluate, run_model_iterations=1, input_shape=(1, 1, 28, 28),
                        compression_type=aimet_torch.svd.svd_intf_defs_deprecated.CompressionTechnique.svd,
                        cost_metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                        layer_selection_scheme=aimet_torch.svd.svd_intf_defs_deprecated.LayerSelectionScheme.top_n_layers,
                        num_layers=2)

        self.assertFalse(svd._is_model_on_gpu())
        # copy entire model on GPU
        model.cuda()
        self.assertTrue(svd._is_model_on_gpu())

    def test_split_manual_rank(self):
        model = MnistModel().to("cpu")
        run_model = mnist_model.evaluate
        logger.debug(self.id())

        intf_defs = aimet_torch.svd.svd_intf_defs_deprecated

        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        with unittest.mock.patch('aimet_torch.layer_database.LayerDatabase'):
            with unittest.mock.patch('aimet_torch.svd.layer_selector_deprecated.LayerSelectorDeprecated'):
                svd = s.SvdImpl(model=model, run_model=None, run_model_iterations=1, input_shape=(1, 1, 28, 28),
                                compression_type=intf_defs.CompressionTechnique.svd,
                                cost_metric=intf_defs.CostMetric.memory,
                                layer_selection_scheme=intf_defs.LayerSelectionScheme.manual,
                                layers_to_compress=[model.fc1])
        layer_rank_list = [[model.fc1, 9]]
        with unittest.mock.patch('aimet_common.cost_calculator.CostCalculator.compute_network_cost') as compute_network_cost:
            compute_network_cost.return_value = cc.Cost(100, 200)
            svd._svd_lib_ref = create_autospec(pymo.Svd, instance=True)
            split_weights = [np.zeros((400, model.fc1.in_features)).flatten().tolist(),
                             np.zeros((model.fc1.out_features, 400)).flatten().tolist()]
            svd._svd_lib_ref.SplitLayerWeights.return_value = split_weights

            split_biases = [np.zeros(400).flatten().tolist(),
                            np.zeros(model.fc1.out_features).flatten().tolist()]
            svd._svd_lib_ref.SplitLayerBiases.return_value = split_biases
            rank_selector = rank_select.RankSelector(svd_lib_ref=svd._svd_lib_ref)
            rank_data_list, svd_rank_pair_dict = rank_selector.split_manual_rank(model=model, run_model=run_model,
                                                                                 run_model_iterations=1, use_cuda=False,
                                                                                 metric=aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory,
                                                                                 database=layer_db,
                                                                                 layer_rank_list=layer_rank_list)
            self.assertEqual(len(svd_rank_pair_dict), 1)


class TestWeightSvdPruning(unittest.TestCase):

    def test_prune_layer(self):

        model = mnist_model.Net()

        # Create a layer database
        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        orig_layer_db = LayerDatabase(model, dummy_input)

        # Copy the db
        comp_layer_db = copy.deepcopy(orig_layer_db)

        conv2 = comp_layer_db.find_layer_by_name('conv2')
        weight_svd_pruner = WeightSvdPruner()
        weight_svd_pruner._prune_layer(orig_layer_db, comp_layer_db, conv2, 0.5, aimet_common.defs.CostMetric.mac)

        conv2_a = comp_layer_db.find_layer_by_name('conv2.0')
        conv2_b = comp_layer_db.find_layer_by_name('conv2.1')

        self.assertEqual((1, 1), conv2_a.module.kernel_size)
        self.assertEqual(32, conv2_a.module.in_channels)
        self.assertEqual(15, conv2_a.module.out_channels)

        self.assertEqual((5, 5), conv2_b.module.kernel_size)
        self.assertEqual(15, conv2_b.module.in_channels)
        self.assertEqual(64, conv2_b.module.out_channels)

        self.assertTrue(isinstance(comp_layer_db.model.conv2, nn.Sequential))

        for layer in comp_layer_db:
            print("Layer: " + layer.name)
            print("   Module: " + str(layer.module))

        print(comp_layer_db.model)

    def test_prune_model_2_layers(self):

        model = mnist_model.Net()

        # Create a layer database
        input_shape = (1, 1, 28, 28)
        dummy_input = create_rand_tensors_given_shapes(input_shape, get_device(model))
        layer_db = LayerDatabase(model, dummy_input)

        fc1 = layer_db.find_layer_by_name('fc1')
        conv2 = layer_db.find_layer_by_name('conv2')
        pruner = WeightSvdPruner()

        layer_db = pruner.prune_model(layer_db, [LayerCompRatioPair(fc1, Decimal(0.5)),
                                                 LayerCompRatioPair(conv2, Decimal(0.5))], aimet_common.defs.CostMetric.mac,
                                      trainer=None)

        fc1_a = layer_db.find_layer_by_name('fc1.0')
        fc1_b = layer_db.find_layer_by_name('fc1.1')

        self.assertEqual(3136, fc1_a.module.in_features)
        self.assertEqual(1024, fc1_b.module.out_features)

        conv2_a = layer_db.find_layer_by_name('conv2.0')
        conv2_b = layer_db.find_layer_by_name('conv2.1')

        self.assertEqual((1, 1), conv2_a.module.kernel_size)
        self.assertEqual(32, conv2_a.module.in_channels)
        self.assertEqual(15, conv2_a.module.out_channels)

        self.assertEqual((5, 5), conv2_b.module.kernel_size)
        self.assertEqual(15, conv2_b.module.in_channels)
        self.assertEqual(64, conv2_b.module.out_channels)

        self.assertTrue(isinstance(layer_db.model.fc1, nn.Sequential))
        self.assertTrue(isinstance(layer_db.model.conv2, nn.Sequential))

        for layer in layer_db:
            print("Layer: " + layer.name)
            print("   Module: " + str(layer.module))

        print(layer_db.model)
