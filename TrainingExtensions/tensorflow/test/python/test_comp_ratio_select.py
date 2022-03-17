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

import unittest
import unittest.mock
from decimal import Decimal
import math
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import signal
import tensorflow as tf

from aimet_common.defs import CostMetric, LayerCompRatioPair
from aimet_common.cost_calculator import SpatialSvdCostCalculator
from aimet_common import comp_ratio_select
from aimet_tensorflow.layer_database import LayerDatabase, Layer
from aimet_tensorflow.examples import mnist_tf_model
from aimet_tensorflow.svd_pruner import SpatialSvdPruner
from aimet_common.utils import start_bokeh_server_session
from aimet_common.bokeh_plots import BokehServerSession
from aimet_common.bokeh_plots import DataTable, ProgressBar

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()


class TestTrainingExtensionsCompRatioSelect(unittest.TestCase):

    def test_per_layer_eval_scores(self):

        pruner = unittest.mock.MagicMock()
        eval_func = unittest.mock.MagicMock()

        # create tf.compat.v1.Session and initialize the weights and biases with zeros
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # by default, model will be constructed in default graph
            _ = mnist_tf_model.create_model(data_format='channels_last')
            sess.run(tf.compat.v1.global_variables_initializer())

        # Create a layer database
        layer_db = LayerDatabase(model=sess, input_shape=(1, 28, 28, 1), working_dir=None)
        layer1 = layer_db.find_layer_by_name('conv2d/Conv2D')

        layer_db.mark_picked_layers([layer1])
        eval_func.side_effect = [90, 80, 70, 60, 50, 40, 30, 20, 10]

        url, process = start_bokeh_server_session(8006)
        bokeh_session = BokehServerSession(url=url, session_id="compression")

        # Instantiate child
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo(layer_db=layer_db, pruner=pruner,
                                                                  cost_calculator=SpatialSvdCostCalculator(),
                                                                  eval_func=eval_func, eval_iterations=20,
                                                                  cost_metric=CostMetric.mac, target_comp_ratio=0.5,
                                                                  num_candidates=10, use_monotonic_fit=True,
                                                                  saved_eval_scores_dict=None,
                                                                  comp_ratio_rounding_algo=None, use_cuda=False,
                                                                  bokeh_session=bokeh_session)
        progress_bar = ProgressBar(1, "eval scores", "green", bokeh_document=bokeh_session)
        data_table = DataTable(num_columns=3, num_rows=1,
                               column_names=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],
                               row_index_names=[layer1.name], bokeh_document=bokeh_session)

        pruner.prune_model.return_value = layer_db
        eval_dict = greedy_algo._compute_layerwise_eval_score_per_comp_ratio_candidate(data_table, progress_bar, layer1)

        self.assertEqual(90, eval_dict[Decimal('0.1')])

        tf.compat.v1.reset_default_graph()
        sess.close()

        bokeh_session.server_session.close("test complete")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    def test_eval_scores(self):

        pruner = unittest.mock.MagicMock()
        eval_func = unittest.mock.MagicMock()
        eval_func.side_effect = [90, 80, 70, 60, 50, 40, 30, 20, 10,
                                 91, 81, 71, 61, 51, 41, 31, 21, 11]

        # create tf.compat.v1.Session and initialize the weights and biases with zeros
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # by default, model will be constructed in default graph
            _ = mnist_tf_model.create_model(data_format='channels_last')
            sess.run(tf.compat.v1.global_variables_initializer())

        # Create a layer database
        layer_db = LayerDatabase(model=sess, input_shape=(1, 28, 28, 1), working_dir=None)
        layer1 = layer_db.find_layer_by_name('conv2d/Conv2D')
        layer2 = layer_db.find_layer_by_name('conv2d_1/Conv2D')

        layer_db.mark_picked_layers([layer1, layer2])

        url, process = start_bokeh_server_session(8006)
        bokeh_session = BokehServerSession(url=url, session_id="compression")

        # Instantiate child
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo(layer_db=layer_db, pruner=pruner,
                                                                  cost_calculator=SpatialSvdCostCalculator(),
                                                                  eval_func=eval_func, eval_iterations=20,
                                                                  cost_metric=CostMetric.mac, target_comp_ratio=0.5,
                                                                  num_candidates=10, use_monotonic_fit=True,
                                                                  saved_eval_scores_dict=None,
                                                                  comp_ratio_rounding_algo=None, use_cuda=False,
                                                                  bokeh_session=bokeh_session)

        eval_dict = greedy_algo._compute_eval_scores_for_all_comp_ratio_candidates()

        self.assertEqual(50, eval_dict['conv2d/Conv2D'][Decimal('0.5')])
        self.assertEqual(60, eval_dict['conv2d/Conv2D'][Decimal('0.4')])

        self.assertEqual(11, eval_dict['conv2d_1/Conv2D'][Decimal('0.9')])

        tf.compat.v1.reset_default_graph()
        sess.close()

        bokeh_session.server_session.close("test complete")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    def test_eval_scores_with_spatial_svd_pruner(self):

        pruner = SpatialSvdPruner()
        eval_func = unittest.mock.MagicMock()
        eval_func.side_effect = [90, 80, 70, 60, 50, 40, 30, 20, 10,
                                 91, 81, 71, 61, 51, 41, 31, 21, 11]

        # create tf.compat.v1.Session and initialize the weights and biases with zeros
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # by default, model will be constructed in default graph
            _ = mnist_tf_model.create_model(data_format='channels_last')
            sess.run(tf.compat.v1.global_variables_initializer())

        # Create a layer database
        layer_db = LayerDatabase(model=sess, input_shape=(1, 28, 28, 1), working_dir=None)
        layer1 = layer_db.find_layer_by_name('conv2d/Conv2D')
        layer2 = layer_db.find_layer_by_name('conv2d_1/Conv2D')

        layer_db.mark_picked_layers([layer1, layer2])

        url, process = start_bokeh_server_session(8006)
        bokeh_session = BokehServerSession(url=url, session_id="compression")

        # Instantiate child
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo(layer_db=layer_db, pruner=pruner,
                                                                  cost_calculator=SpatialSvdCostCalculator(),
                                                                  eval_func=eval_func, eval_iterations=20,
                                                                  cost_metric=CostMetric.mac, target_comp_ratio=0.5,
                                                                  num_candidates=10, use_monotonic_fit=True,
                                                                  saved_eval_scores_dict=None,
                                                                  comp_ratio_rounding_algo=None, use_cuda=False,
                                                                  bokeh_session=bokeh_session)

        dict = greedy_algo._compute_eval_scores_for_all_comp_ratio_candidates()

        print()
        print(dict)
        self.assertEqual(90, dict['conv2d/Conv2D'][Decimal('0.1')])

        self.assertEqual(51, dict['conv2d_1/Conv2D'][Decimal('0.5')])
        self.assertEqual(21, dict['conv2d_1/Conv2D'][Decimal('0.8')])

        tf.compat.v1.reset_default_graph()
        sess.close()

        bokeh_session.server_session.close("test complete")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    def test_find_min_max_eval_scores(self):

        eval_scores_dict = {'layer1': {Decimal('0.1'): 90, Decimal('0.5'): 50, Decimal('0.7'): 30, Decimal('0.8'): 20},
                            'layer2': {Decimal('0.2'): 91, Decimal('0.3'): 45, Decimal('0.7'): 30, Decimal('0.9'): 11}}

        min_score, max_score = comp_ratio_select.GreedyCompRatioSelectAlgo._find_min_max_eval_scores(eval_scores_dict)

        self.assertEqual(11, min_score)
        self.assertEqual(91, max_score)

        eval_scores_dict = {'layer1': {Decimal('0.1'): 10, Decimal('0.5'): 92, Decimal('0.7'): 30, Decimal('0.8'): 20},
                            'layer2': {Decimal('0.2'): 91, Decimal('0.3'): 45, Decimal('0.7'): 30, Decimal('0.9'): 11}}

        min_score, max_score = comp_ratio_select.GreedyCompRatioSelectAlgo._find_min_max_eval_scores(eval_scores_dict)

        self.assertEqual(10, min_score)
        self.assertEqual(92, max_score)

    def test_find_layer_comp_ratio_given_eval_score(self):

        eval_scores_dict = {'layer1': {Decimal('0.1'): 90, Decimal('0.5'): 50, Decimal('0.7'): 30, Decimal('0.8'): 20},

                            'layer2': {Decimal('0.1'): 11,
                                       Decimal('0.3'): 23,
                                       Decimal('0.5'): 47,
                                       Decimal('0.7'): 85,
                                       Decimal('0.9'): 89}
                            }

        # data format : NHWC
        inp_tensor = tf.Variable(tf.random.normal([1, 28, 28, 32]))
        filter_tensor = tf.Variable(tf.random.normal([5, 5, 32, 64]))
        conv = tf.nn.conv2d(inp_tensor, filter_tensor, strides=[1, 2, 2, 1], padding='SAME',
                            data_format="NHWC", name='layer2')

        conv_op = tf.compat.v1.get_default_graph().get_operation_by_name('layer2')

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        # output shape in NCHW format
        output_shape = conv_op.outputs[0].shape

        layer2 = Layer(model=sess, op=conv_op, output_shape=output_shape)

        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo
        comp_ratio = greedy_algo._find_layer_comp_ratio_given_eval_score(eval_scores_dict,
                                                                         45,
                                                                         layer2)
        self.assertEqual(Decimal('0.5'), comp_ratio)

        comp_ratio = greedy_algo._find_layer_comp_ratio_given_eval_score(eval_scores_dict,
                                                                         48,
                                                                         layer2)
        self.assertEqual(Decimal('0.7'), comp_ratio)

        comp_ratio = greedy_algo._find_layer_comp_ratio_given_eval_score(eval_scores_dict,
                                                                         90,
                                                                         layer2)
        self.assertEqual(None, comp_ratio)

        tf.compat.v1.reset_default_graph()
        sess.close()

    def test_select_per_layer_comp_ratios(self):

        pruner = unittest.mock.MagicMock()
        eval_func = unittest.mock.MagicMock()
        rounding_algo = unittest.mock.MagicMock()
        rounding_algo.round.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        eval_func.side_effect = [10, 20, 30, 40, 50, 60, 70, 80, 90,
                                 11, 21, 31, 35, 40, 45, 50, 55, 60]

        # create tf.compat.v1.Session and initialize the weights and biases with zeros
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # by default, model will be constructed in default graph
            _ = mnist_tf_model.create_model(data_format='channels_last')
            sess.run(tf.compat.v1.global_variables_initializer())

        # Create a layer database
        layer_db = LayerDatabase(model=sess, input_shape=(1, 28, 28, 1), working_dir=None)
        layer1 = layer_db.find_layer_by_name('conv2d/Conv2D')
        layer2 = layer_db.find_layer_by_name('conv2d_1/Conv2D')

        selected_layers = [layer1, layer2]
        layer_db.mark_picked_layers([layer1, layer2])

        try:
            os.remove('./data/greedy_selection_eval_scores_dict.pkl')
        except OSError:
            pass

        url, process = start_bokeh_server_session(8006)
        bokeh_session = BokehServerSession(url=url, session_id="compression")

        # Instantiate child
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo(layer_db=layer_db, pruner=pruner,
                                                                  cost_calculator=SpatialSvdCostCalculator(),
                                                                  eval_func=eval_func, eval_iterations=20,
                                                                  cost_metric=CostMetric.mac,
                                                                  target_comp_ratio=Decimal(0.6), num_candidates=10,
                                                                  use_monotonic_fit=True, saved_eval_scores_dict=None,
                                                                  comp_ratio_rounding_algo=rounding_algo,
                                                                  use_cuda=False,
                                                                  bokeh_session=bokeh_session)

        layer_comp_ratio_list, stats = greedy_algo.select_per_layer_comp_ratios()

        original_cost = SpatialSvdCostCalculator.compute_model_cost(layer_db)

        for layer in layer_db:
            if layer not in selected_layers:
                layer_comp_ratio_list.append(LayerCompRatioPair(layer, None))
        compressed_cost = SpatialSvdCostCalculator.calculate_compressed_cost(layer_db, layer_comp_ratio_list,
                                                                             CostMetric.mac)
        rounding_algo.round.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        actual_compression_ratio = compressed_cost.mac / original_cost.mac
        self.assertTrue(math.isclose(Decimal(0.6), actual_compression_ratio, abs_tol=0.05))
        self.assertTrue(os.path.isfile('./data/greedy_selection_eval_scores_dict.pkl'))

        print('\n')
        for pair in layer_comp_ratio_list:
            print(pair)

        # lets repeat with a saved eval_dict
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo(layer_db=layer_db, pruner=pruner,
                                                                  cost_calculator=SpatialSvdCostCalculator(),
                                                                  eval_func=eval_func, eval_iterations=20,
                                                                  cost_metric=CostMetric.mac,
                                                                  target_comp_ratio=Decimal(0.6), num_candidates=10,
                                                                  use_monotonic_fit=True,
                                                                  saved_eval_scores_dict='./data/greedy_selection_eval_scores_dict.pkl',
                                                                  comp_ratio_rounding_algo=rounding_algo,
                                                                  use_cuda=False,
                                                                  bokeh_session=bokeh_session)

        layer_comp_ratio_list, stats = greedy_algo.select_per_layer_comp_ratios()

        original_cost = SpatialSvdCostCalculator.compute_model_cost(layer_db)

        for layer in layer_db:
            if layer not in selected_layers:
                layer_comp_ratio_list.append(LayerCompRatioPair(layer, None))
        compressed_cost = SpatialSvdCostCalculator.calculate_compressed_cost(layer_db, layer_comp_ratio_list,
                                                                             CostMetric.mac)

        actual_compression_ratio = compressed_cost.mac / original_cost.mac
        self.assertTrue(math.isclose(Decimal(0.6), actual_compression_ratio, abs_tol=0.05))

        print('\n')
        for pair in layer_comp_ratio_list:
            print(pair)

        tf.compat.v1.reset_default_graph()
        sess.close()

        bokeh_session.server_session.close("test complete")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    def test_select_per_layer_comp_ratios_with_spatial_svd_pruner(self):

        pruner = SpatialSvdPruner()
        eval_func = unittest.mock.MagicMock()
        rounding_algo = unittest.mock.MagicMock()
        eval_func.side_effect = [10, 20, 30, 40, 50, 60, 70, 80, 90,
                                 11, 21, 31, 35, 40, 45, 50, 55, 60]
        rounding_algo.round.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # create tf.compat.v1.Session and initialize the weights and biases with zeros
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # create session with graph
        sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)

        with sess.graph.as_default():
            # by default, model will be constructed in default graph
            _ = mnist_tf_model.create_model(data_format='channels_last')
            sess.run(tf.compat.v1.global_variables_initializer())

        # Create a layer database
        layer_db = LayerDatabase(model=sess, input_shape=(1, 28, 28, 1), working_dir=None)

        selected_layers = [layer for layer in layer_db if layer.module.type == 'Conv2D']
        layer_db.mark_picked_layers(selected_layers)

        url, process = start_bokeh_server_session(8006)
        bokeh_session = BokehServerSession(url=url, session_id="compression")

        # Instantiate child
        greedy_algo = comp_ratio_select.GreedyCompRatioSelectAlgo(layer_db=layer_db, pruner=pruner,
                                                                  cost_calculator=SpatialSvdCostCalculator(),
                                                                  eval_func=eval_func, eval_iterations=20,
                                                                  cost_metric=CostMetric.mac,
                                                                  target_comp_ratio=Decimal(0.4), num_candidates=10,
                                                                  use_monotonic_fit=True, saved_eval_scores_dict=None,
                                                                  comp_ratio_rounding_algo=rounding_algo,
                                                                  use_cuda=False,
                                                                  bokeh_session=bokeh_session)

        layer_comp_ratio_list, stats = greedy_algo.select_per_layer_comp_ratios()

        original_cost = SpatialSvdCostCalculator.compute_model_cost(layer_db)

        for layer in layer_db:
            if layer not in selected_layers:
                layer_comp_ratio_list.append(LayerCompRatioPair(layer, None))
        compressed_cost = SpatialSvdCostCalculator.calculate_compressed_cost(layer_db, layer_comp_ratio_list,
                                                                             CostMetric.mac)

        actual_compression_ratio = compressed_cost.mac / original_cost.mac
        self.assertTrue(math.isclose(Decimal(0.3), actual_compression_ratio, abs_tol=0.8))

        print('\n')
        for pair in layer_comp_ratio_list:
            print(pair)

        tf.compat.v1.reset_default_graph()
        sess.close()

        bokeh_session.server_session.close("test complete")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
