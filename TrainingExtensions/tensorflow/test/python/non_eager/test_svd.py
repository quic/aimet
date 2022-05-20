# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import aimet_common.libpymo as pymo
from aimet_tensorflow import svd as s

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()


def conv(x, shape, scope):
    # initialize conv weights, if not done yet
    W = tf.compat.v1.get_variable(scope + '_w', initializer=tf.random.truncated_normal(shape, stddev=0.1, seed=0))
    b = tf.compat.v1.get_variable(scope + '_b', initializer=tf.constant(0.1, shape=shape[3:]))
    # do CONV forward path
    acts = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)
    return acts


def fc(x, shape, scope):
    # initialize fc weights, if not done yet
    W = tf.compat.v1.get_variable(scope + '_w', initializer=tf.random.truncated_normal(shape, stddev=0.1, seed=0))
    b = tf.compat.v1.get_variable(scope + '_b', initializer=tf.constant(0.1, shape=shape[1:2]))
    # do FC forward path
    y = tf.matmul(x, W) + b
    return y


def model(x):
    x = tf.reshape(x, [-1,28,28,1])
    acts = conv(x, [5,5,1,32], 'conv1')
    acts = tf.nn.max_pool2d(acts, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    acts = conv(acts, [5,5,32,64], 'conv2')
    acts = tf.nn.max_pool2d(acts, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    acts = tf.reshape(acts, [-1,7*7*64])
    acts = fc(acts, [7*7*64,1024], 'fc1')
    acts = tf.nn.relu(acts)
    y = fc(acts, [1024,10], 'fc2')
    return y


class TestSvdTrainingExtensions(unittest.TestCase):

    def test_svd_layer_selection_without_mo(self):
        tf.compat.v1.reset_default_graph()
        svd = s.Svd(None, None, s.CostMetric.memory)
        svd._svd = create_autospec(pymo.Svd, instance=True)

        x = tf.compat.v1.placeholder(tf.float32, [None, 784], 'data')
        y_hat = model(x)
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        svd._svd.GetCompressionType.return_value = pymo.SVD_COMPRESS_TYPE.TYPE_SINGLE
        svd._store_net_stats(sess)
        self.assertTrue(svd._svd.GetCompressionType.called)

    def test_svd_layer_selection_with_mo(self):
        tf.compat.v1.reset_default_graph()
        svd = s.Svd(None, None, s.CostMetric.memory)

        x = tf.compat.v1.placeholder(tf.float32, [None, 784], 'data')
        y_hat = model(x)
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        svd._store_net_stats(sess)

        layers = svd._svd.GetLayerNames()
        print("Layers added: ", layers)


    def test_compute_layer_cost(self):

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, [None, 784], 'data')
            y_hat = model(x)

            sess.run(tf.compat.v1.global_variables_initializer())

            ops = tf.compat.v1.get_default_graph().get_operations()
            for op in ops:
                if op.name == "Conv2D":
                    conv1 = op

                elif op.name == "MatMul":
                    fc1 = op

            mem_cost, mac_cost = s.Svd._compute_layer_cost(conv1.inputs[1].shape, conv1.outputs[0].shape, conv1.type)
            self.assertEqual(mem_cost, 800)
            self.assertEqual(mac_cost, 627200)

            mem_cost, mac_cost = s.Svd._compute_layer_cost(fc1.inputs[1].shape, fc1.outputs[0].shape, fc1.type)
            self.assertEqual(mem_cost, 3211264)
            self.assertEqual(mac_cost, 3211264)

    def test_create_layer_attributes_list(self):

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, [None, 784], 'data')
            y_hat = model(x)

            sess.run(tf.compat.v1.global_variables_initializer())

            ops = tf.compat.v1.get_default_graph().get_operations()
            ops_to_use_for_cost = []
            for op in ops:
                if op.type in ['Conv2D', 'MatMul']:
                    ops_to_use_for_cost.append(op)

            layer_attributes_list = s.Svd._create_layer_attributes_list(ops_to_use_for_cost, sess)
            print(layer_attributes_list)
            self.assertEqual((800, 627200), layer_attributes_list[0].cost)
            self.assertEqual((51200, 10035200), layer_attributes_list[1].cost)
            self.assertEqual(len(layer_attributes_list), 4)

    def test_compute_network_cost(self):

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, [None, 784], 'data')
            y_hat = model(x)

            sess.run(tf.compat.v1.global_variables_initializer())

            ops = tf.compat.v1.get_default_graph().get_operations()
            ops_to_use_for_cost = []
            for op in ops:
                if op.type in ['Conv2D', 'MatMul']:
                    ops_to_use_for_cost.append(op)

            layer_attributes_list = s.Svd._create_layer_attributes_list(ops_to_use_for_cost, sess)
            mem_cost, mac_cost = s.Svd._compute_network_cost(layer_attributes_list)

            self.assertEqual(mem_cost, 800 + 51200 + 3211264 + 10240)
            self.assertEqual(mac_cost, 627200 + 10035200 + 3211264 + 10240)

    def test_pick_compression_layers_top_n_layers(self):

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, [None, 784], 'data')
            y_hat = model(x)

            sess.run(tf.compat.v1.global_variables_initializer())

            picked_layers, network_cost = s.Svd._pick_compression_layers(sess, cost_metric=s.CostMetric.mac,
                                                                         layer_select_scheme=s.Svd.LayerSelectionScheme.top_n_layers,
                                                                         num_layers=2)
            self.assertEqual('Conv2D_1', picked_layers[0].layer_ref.name)
            self.assertEqual('MatMul', picked_layers[1].layer_ref.name)
            self.assertEqual(2, len(picked_layers))
            self.assertEqual((3273504, 13883904), network_cost)

            picked_layers, network_cost = s.Svd._pick_compression_layers(sess, cost_metric=s.CostMetric.memory,
                                                                         layer_select_scheme=s.Svd.LayerSelectionScheme.top_n_layers,
                                                                         num_layers=2)
            for layer in picked_layers:
                print(layer.layer_ref.name, layer.cost)
            self.assertEqual('MatMul', picked_layers[0].layer_ref.name)
            self.assertEqual('Conv2D_1', picked_layers[1].layer_ref.name)
            self.assertEqual(2, len(picked_layers))
            self.assertEqual((3273504, 13883904), network_cost)

    def test_pick_compression_layers_top_x_percent(self):

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, [None, 784], 'data')
            y_hat = model(x)

            sess.run(tf.compat.v1.global_variables_initializer())

            # 100% criterion
            picked_layers, network_cost = s.Svd._pick_compression_layers(sess, cost_metric=s.CostMetric.memory,
                                                                         layer_select_scheme=s.Svd.LayerSelectionScheme.top_x_percent,
                                                                         percent_thresh=100)
            for layer in picked_layers:
                print(layer.layer_ref.name, layer.cost)

            self.assertEqual('MatMul', picked_layers[0].layer_ref.name)
            self.assertEqual('Conv2D_1', picked_layers[1].layer_ref.name)
            self.assertEqual('MatMul_1', picked_layers[2].layer_ref.name)
            self.assertEqual(3, len(picked_layers))
            self.assertEqual((3273504, 13883904), network_cost)

            # 80% criterion
            picked_layers, network_cost = s.Svd._pick_compression_layers(sess, cost_metric=s.CostMetric.memory,
                                                                         layer_select_scheme=s.Svd.LayerSelectionScheme.top_x_percent,
                                                                         percent_thresh=80)
            for layer in picked_layers:
                print(layer.layer_ref.name, layer.cost)

            self.assertEqual('Conv2D_1', picked_layers[0].layer_ref.name)
            self.assertEqual('MatMul_1', picked_layers[1].layer_ref.name)
            self.assertEqual(2, len(picked_layers))
            self.assertEqual((3273504, 13883904), network_cost)

            # 99% criterion
            picked_layers, network_cost = s.Svd._pick_compression_layers(sess, cost_metric=s.CostMetric.memory,
                                                                         layer_select_scheme=s.Svd.LayerSelectionScheme.top_x_percent,
                                                                         percent_thresh=98.5)
            for layer in picked_layers:
                print(layer.layer_ref.name, layer.cost)

            self.assertEqual('MatMul', picked_layers[0].layer_ref.name)
            self.assertEqual('MatMul_1', picked_layers[1].layer_ref.name)
            self.assertEqual(2, len(picked_layers))
            self.assertEqual((3273504, 13883904), network_cost)

    def test_pick_compression_layers_manual(self):

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, [None, 784], 'data')
            y_hat = model(x)

            sess.run(tf.compat.v1.global_variables_initializer())

            # Manual criterion
            picked_layers, network_cost = s.Svd._pick_compression_layers(sess, cost_metric=s.CostMetric.memory,
                                                                         layer_select_scheme=s.Svd.LayerSelectionScheme.manual,
                                                                         layers_to_compress=['Conv2D_1'])
            for layer in picked_layers:
                print(layer.layer_ref.name, layer.cost)

            self.assertEqual('Conv2D_1', picked_layers[0].layer_ref.name)
            self.assertEqual(1, len(picked_layers))
            self.assertEqual((3273504, 13883904), network_cost)

    @unittest.skip
    def test_automatic_rank_selection(self):

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, [None, 784], 'data')
            y_hat = model(x)
            sess.run(tf.compat.v1.global_variables_initializer())
            s.Svd._baseline_perf = 1
            s.Svd._svd = create_autospec(pymo.Svd, instance=True)

            s.Svd._compute_compression_ratio = create_autospec(s.Svd._compute_compression_ratio)
            s.Svd._compute_compression_ratio.side_effect = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            s.Svd._create_compressed_network = create_autospec(s.Svd._create_compressed_network)
            s.Svd._create_compressed_network.return_value = None, None
            s.Svd._network_cost = (500, 500)
            run_graph_return_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            s.Svd.run_graph = unittest.mock.Mock(side_effect=run_graph_return_values)
            s.Svd._svd.SetCandidateRanks = create_autospec(s.Svd._svd.SetCandidateRanks)
            s.Svd._svd.SetCandidateRanks.side_effect = 20
            s.Svd._perform_rank_selection(self)
