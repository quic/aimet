# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" This file contains unit tests for testing TfConnectedGraph module. """

# pylint: disable=no-name-in-module
import os
import unittest
import logging
import tensorflow as tf
from tensorflow.contrib.graph_editor.edit import detach_inputs
from tensorflow.contrib.slim.nets import vgg

from aimet_common.utils import AimetLogger
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.common.module_identifier import StructureModuleIdentifier
from aimet_tensorflow.common.sub_graph_matcher import ModuleIdentifierOpInfo
from aimet_tensorflow.examples.test_models import keras_model, keras_model_functional, tf_slim_basic_model, \
    single_residual, split_and_concat_model, concat_model, dropout_keras_model, dropout_slim_model, \
    tf_slim_with_softmax, multiple_input_model, upsample_model, model_with_upsample2d, model_with_leaky_relu, \
    model_with_global_max_pool2d, keras_model_functional_with_non_fused_batchnorms
import aimet_tensorflow.winnow.winnow as winnow

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Test, logging.DEBUG)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.ConnectedGraph, logging.DEBUG)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestTfConnectedGraph(unittest.TestCase):
    """ Test TfConnectedGraph module """

    def test_keras_model_get_op_product_graph(self):
        """ Test connected graph construction on keras model """
        tf.compat.v1.reset_default_graph()

        _ = keras_model()
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['conv2d_input'], ['keras_model/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(11, len(conn_graph.get_all_ops()))

        # 10 products from inter module connections
        # 14 products from parameters
        self.assertEqual(24, len(conn_graph.get_all_products()))

    def test_keras_model_functional_get_op_product_graph(self):
        """ Test connected graph construction on keras model functional """
        tf.compat.v1.reset_default_graph()

        _ = keras_model_functional()
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['input_1'], ['keras_model_functional/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(14, len(conn_graph.get_all_ops()))

        # 13 products from inter module connections
        # 22 products from parameters
        self.assertEqual(35, len(conn_graph.get_all_products()))

    def test_keras_model_functional_with_non_fused_batchnorms_get_op_product_graph(self):
        """ Test connected graph construction on keras model functional with non fused batchnorms """
        tf.compat.v1.reset_default_graph()

        _ = keras_model_functional_with_non_fused_batchnorms()
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['input_1'],
                                    ['keras_model_functional_with_non_fused_batchnorms/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        _ = conn_graph.get_all_ops()['batch_normalization']
        _ = conn_graph.get_all_ops()['scope_1/batch_normalization_1']
        _ = conn_graph.get_all_ops()['scope_1/batch_normalization_2']
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(14, len(conn_graph.get_all_ops()))

        # 13 products from inter module connections
        # 22 products from parameters
        self.assertEqual(35, len(conn_graph.get_all_products()))

    def test_tf_slim_model_get_op_product_graph(self):
        """ Test connected graph construction on tf_slim model """

        tf.compat.v1.reset_default_graph()

        x = tf.placeholder(tf.float32, [1, 32, 32, 3])
        _ = tf_slim_basic_model(x)
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['Placeholder'], ['tf_slim_model/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(15, len(conn_graph.get_all_ops()))
        # 14 products from interop connections
        # need to add 1 since gamma is treated as a parameter for the training = True bn, even though it is a constant
        # in the graph
        self.assertEqual(14 + len(tf.get_default_graph().get_collection('variables')) + 1,
                         len(conn_graph.get_all_products()))

    def test_tf_slim_with_softmax_model_get_op_product_graph(self):
        """ Test connected graph construction on tf_slim with softmax model """

        tf.compat.v1.reset_default_graph()

        x = tf.placeholder(tf.float32, [1, 32, 32, 3])
        _ = tf_slim_with_softmax(x)
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['Placeholder'], ['softmax/Reshape_1'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(8, len(conn_graph.get_all_ops()))
        # 7 products from interop connections
        # need to add 2 since gamma is treated as a parameter, even though it is a constant in this graph
        self.assertEqual(7 + len(tf.get_default_graph().get_collection('variables')) + 2,
                         len(conn_graph.get_all_products()))

    def test_single_residual_get_op_product_graph(self):
        """ Test connected graph construction on single residual model """

        tf.compat.v1.reset_default_graph()
        _ = single_residual()
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['input_1'], ['single_residual/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(1, conn_graph.branch_count)
        self.assertEqual(18, len(conn_graph.get_all_ops()))
        # 17 products from interop connections, 20 from parameters
        self.assertEqual(37, len(conn_graph.get_all_products()))

    def test_split_get_op_product_graph(self):
        """ Test connected graph construction on a graph with split op """

        tf.compat.v1.reset_default_graph()

        _ = split_and_concat_model()
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['input_1'], ['split_and_concat_model/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(1, conn_graph.branch_count)
        self.assertEqual(9, len(conn_graph.get_all_ops()))
        self.assertEqual(8 + len(tf.get_default_graph().get_collection('variables')),
                         len(conn_graph.get_all_products()))

    def test_concat_get_op_product_graph(self):
        """ Test connected graph construction on a graph with concat op """

        tf.compat.v1.reset_default_graph()

        _ = concat_model()
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['input_1'], ['concat_model/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(2, conn_graph.branch_count)
        self.assertEqual(13, len(conn_graph.get_all_ops()))
        self.assertEqual(12 + len(tf.get_default_graph().get_collection('variables')),
                         len(conn_graph.get_all_products()))

        # Check that the order of input products to the concat op matches the order of input tensors in the tf graph
        concat_tf_op = tf.get_default_graph().get_operation_by_name("concatenate/concat")
        concat_op = conn_graph.get_all_ops()['concatenate/concat']
        for index, product in enumerate(concat_op.get_input_products()):
            self.assertTrue(len(product.consumers) == 1)
            self.assertEqual(product.tensor_dict[product.consumers[0]], concat_tf_op.inputs[index])

    def test_dropout_keras_get_op_product_graph(self):
        """ Test connected graph construction on a keras graph with dropout op """

        tf.compat.v1.reset_default_graph()
        _ = dropout_keras_model()
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['input_1'], ['dropout_keras_model/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(8, len(conn_graph.get_all_ops()))
        self.assertEqual(7 + len(tf.get_default_graph().get_collection('variables')),
                         len(conn_graph.get_all_products()))
        self.assertTrue(conn_graph.get_all_ops()['dropout'], 'Dropout_with_training_tensor')

    def test_dropout_slim_get_op_product_graph(self):
        """ Test connected graph construction on a slim graph with dropout op """

        tf.compat.v1.reset_default_graph()
        _ = dropout_slim_model()
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['input_1'], ['dropout_slim_model/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(10, len(conn_graph.get_all_ops()))
        self.assertEqual(9 + len(tf.get_default_graph().get_collection('variables')),
                         len(conn_graph.get_all_products()))
        self.assertTrue(conn_graph.get_all_ops()['Dropout'], 'Dropout_training_True')

    def test_vgg16_slim_get_op_product_graph(self):
        """
        Test connected graph construction on vgg16 from tf slim
        This model includes dropout pattern 3 which does not appear in other models.
        """

        tf.compat.v1.reset_default_graph()
        inp = tf.placeholder(tf.float32, [1, 224, 224, 3])
        _ = vgg.vgg_16(inp)
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['Placeholder'], ['vgg_16/fc8/squeezed'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(40, len(conn_graph.get_all_ops()))
        self.assertEqual(39 + len(tf.get_default_graph().get_collection('variables')),
                         len(conn_graph.get_all_products()))
        self.assertTrue(conn_graph.get_all_ops()['vgg_16/dropout6'], 'Dropout_training_True_unknown_shape')
        self.assertTrue(conn_graph.get_all_ops()['vgg_16/dropout7'], 'Dropout_training_True_unknown_shape')

    def test_multiple_input_model_get_op_product_graph(self):
        """ Test connected graph construction on a multiple input graph """

        tf.compat.v1.reset_default_graph()
        _ = multiple_input_model()
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['input1', 'input2'], ['multiple_input_model/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(9, len(conn_graph.get_all_ops()))
        self.assertEqual(8 + len(tf.get_default_graph().get_collection('variables')),
                         len(conn_graph.get_all_products()))
        self.assertTrue(conn_graph.get_all_ops()['input1'].output.is_model_input)
        self.assertTrue(conn_graph.get_all_ops()['input2'].output.is_model_input)

    def test_connected_graph_with_detached_ops(self):
        """ Test connected graph construction on a graph with detached ops """
        tf.compat.v1.reset_default_graph()
        _ = single_residual()

        # Detach everything starting from conv2d_4/Conv2D and below
        detach_inputs(tf.get_default_graph().get_operation_by_name('conv2d_4/Conv2D'))
        conn_graph = ConnectedGraph(tf.get_default_graph(), ['input_1'], ['Relu_2'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(1, conn_graph.branch_count)
        self.assertEqual(13, len(conn_graph.get_all_ops()))
        # 12 products from interop connections, 16 from parameters
        self.assertEqual(28, len(conn_graph.get_all_products()))

    @unittest.skip('Skipping until support for upsample is added')
    def test_upsample_get_op_product_graph(self):
        """ Test connected graph construction on a graph with upsample op
        Need to perform one round of winnowing first to insert the upsample op """
        tf.compat.v1.reset_default_graph()
        sess = tf.Session()
        module_zero_channels_list = []

        _ = upsample_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ['input_1']
        output_op_names = ['upsample_model/Softmax']

        tf_op = tf.get_default_graph().get_operation_by_name("conv2d_3/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, _ = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                             list_of_modules_to_winnow=module_zero_channels_list,
                                             reshape=True, in_place=True, verbose=True)

        conn_graph = ConnectedGraph(tf.get_default_graph(), input_op_names, output_op_names)
        self.assertEqual(18, len(conn_graph.get_all_ops()))
        reduced_bn_1_op = conn_graph.get_op_from_module_name('reduced_batch_normalization_1/cond/FusedBatchNormV3_1')
        self.assertTrue(reduced_bn_1_op.output.consumers[0].type == 'Upsample')
        new_sess.close()

    def test_keras_model_functional_with_training_ops_get_op_product_graph(self):
        """ Test connected graph construction on keras model functional with training ops attached """
        tf.compat.v1.reset_default_graph()
        _ = keras_model_functional()

        # add training ops
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, name='Adam_new')
        _ = optimizer.minimize(loss=tf.get_default_graph().get_tensor_by_name('keras_model_functional/Softmax:0'),
                               name='train_step_new')
        conn_graph = ConnectedGraph(tf.get_default_graph(), ["input_1"],
                                    output_op_names=['keras_model_functional/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(14, len(conn_graph.get_all_ops()))

        # 13 products from inter module connections
        # 22 products from parameters
        self.assertEqual(35, len(conn_graph.get_all_products()))

    def test_model_with_upsample2d(self):
        """ Test connected graph construction on model with upsample2D op """
        tf.compat.v1.reset_default_graph()
        _ = model_with_upsample2d()
        conn_graph = ConnectedGraph(tf.get_default_graph(), starting_op_names=['input_1'],
                                    output_op_names=['model_with_upsample2d/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(7, len(conn_graph.get_all_ops()))

        # 6 products from inter module connections
        # 6 products from parameters
        self.assertEqual(12, len(conn_graph.get_all_products()))
        found_upsample2d = False
        for op in conn_graph.get_all_ops().values():
            if op.type == 'Upsample2D':
                found_upsample2d = True
        self.assertTrue(found_upsample2d)

        tf.compat.v1.reset_default_graph()

    def test_model_with_leaky_relu(self):
        """ Test connected graph construction on model with leaky relu op """
        tf.compat.v1.reset_default_graph()
        _ = model_with_leaky_relu()
        conn_graph = ConnectedGraph(tf.get_default_graph(), starting_op_names=['input_1'],
                                    output_op_names=['model_with_leaky_relu/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(7, len(conn_graph.get_all_ops()))

        # 6 products from inter module connections
        # 6 products from parameters
        self.assertEqual(12, len(conn_graph.get_all_products()))
        found_leaky_relu = False
        for op in conn_graph.get_all_ops().values():
            if op.type == 'LeakyRelu':
                found_leaky_relu = True
        self.assertTrue(found_leaky_relu)

        tf.compat.v1.reset_default_graph()

    def test_model_with_global_max_pool2d(self):
        """ Test connected graph construction on model with leaky relu op """
        tf.compat.v1.reset_default_graph()
        _ = model_with_global_max_pool2d()
        conn_graph = ConnectedGraph(tf.get_default_graph(), starting_op_names=['input_1'],
                                    output_op_names=['model_with_global_max_pool2d/Softmax'])
        self.assertTrue(validate_branch_ops(conn_graph))
        self.assertTrue(validate_product_tensor_lists(conn_graph))
        self.assertEqual(0, conn_graph.branch_count)
        self.assertEqual(6, len(conn_graph.get_all_ops()))

        # 5 products from inter module connections
        # 4 products from parameters
        self.assertEqual(9, len(conn_graph.get_all_products()))
        found_global_max_pool2d = False
        for op in conn_graph.get_all_ops().values():
            if op.type == 'GlobalMaxPool2D':
                found_global_max_pool2d = True
        self.assertTrue(found_global_max_pool2d)

        tf.compat.v1.reset_default_graph()

    def test_model_with_simple_rnn_layer(self):
        """ Test connected graph construction on a model with simple RNN op """
        tf.compat.v1.reset_default_graph()
        sess = tf.Session()
        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            x = tf.keras.layers.SimpleRNN(12, name='rnn0')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="matmul0")(x)

            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            # _ = tf.summary.FileWriter('./simple_rnn', sess.graph)

        # construct a connected graph
        conn_graph = ConnectedGraph(sess.graph, ['input_1'], ['matmul0/Softmax'])

        # there should be only 4 connected graph ops, input, simpleRNN , Dense and Softmax
        self.assertEqual(4, len(conn_graph.get_all_ops()))
        simple_rnn_detected = False
        for op in conn_graph.get_all_ops().values():
            if op.type == 'SimpleRNN':
                simple_rnn_detected = True
                inner_list = op.internal_ops
                self.assertEqual(49, len(inner_list))
                self.assertEqual(op.get_module(), sess.graph.get_operation_by_name('rnn0/while/MatMul'))
                self.assertEqual('rnn0', op.name)
        self.assertTrue(simple_rnn_detected)

        # check for 2 MatMuls, 1 BiasAdd and an activation function in the inner op list
        valid_matmuls = []
        valid_bias_add = []
        valid_activation = []
        for op in inner_list:
            if op.type == 'MatMul' and op not in valid_matmuls:
                valid_matmuls.append(op)
            if op.type == 'BiasAdd' and op not in valid_bias_add:
                valid_bias_add.append(op)
            if op.type == 'Tanh' and op not in valid_activation:
                valid_activation.append(op)

        self.assertEqual(2, len(valid_matmuls))
        self.assertEqual(1, len(valid_bias_add))
        self.assertEqual(1, len(valid_activation))

    def test_model_with_simple_rnn_layer_relu(self):
        """ Test connected graph construction on a model with simple RNN op with relu activation """
        tf.compat.v1.reset_default_graph()
        sess = tf.Session()
        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            x = tf.keras.layers.SimpleRNN(12, name='rnn0', activation='relu')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax, name="matmul0")(x)

            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            # _ = tf.summary.FileWriter('./simple_rnn', sess.graph)

        # construct a connected graph
        conn_graph = ConnectedGraph(sess.graph, ['input_1'], ['matmul0/Softmax'])

        # there should be only 4 connected graph ops, input, simpleRNN , Dense and Softmax
        self.assertEqual(4, len(conn_graph.get_all_ops()))
        simple_rnn_detected = False
        for op in conn_graph.get_all_ops().values():
            if op.type == 'SimpleRNN':
                simple_rnn_detected = True
                inner_list = op.internal_ops
                self.assertEqual(49, len(inner_list))
                self.assertEqual(op.get_module(), sess.graph.get_operation_by_name('rnn0/while/MatMul'))
                self.assertEqual('rnn0', op.name)
        self.assertTrue(simple_rnn_detected)

        # check for 2 MatMuls, 1 BiasAdd and an activation function in the inner op list
        valid_matmuls = []
        valid_bias_add = []
        valid_activation = []
        for op in inner_list:
            if op.type == 'MatMul' and op not in valid_matmuls:
                valid_matmuls.append(op)
            if op.type == 'BiasAdd' and op not in valid_bias_add:
                valid_bias_add.append(op)
            if op.type == 'Relu' and op not in valid_activation:
                valid_activation.append(op)

        self.assertEqual(2, len(valid_matmuls))
        self.assertEqual(1, len(valid_bias_add))
        self.assertEqual(1, len(valid_activation))


    def test_model_with_simple_rnn_multiple_layers(self):
        """ Test connected graph construction on a model with multiple simple RNN layers """
        tf.compat.v1.reset_default_graph()
        sess = tf.Session()
        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            x = tf.keras.layers.SimpleRNN(12, name='rnn0', activation='tanh', return_sequences=True)(inputs)
            x = tf.keras.layers.SimpleRNN(12, name='rnn1', activation='relu', return_sequences=True)(x)
            x = tf.keras.layers.SimpleRNN(12, name='rnn2', activation='tanh')(x)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax, name="matmul0")(x)

            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            # _ = tf.summary.FileWriter('./simple_rnn', sess.graph)

        # construct a connected graph
        conn_graph = ConnectedGraph(sess.graph, ['input_1'], ['matmul0/Softmax'])

        # there should be only 4 connected graph ops, input, simpleRNN , Dense and Softmax
        self.assertEqual(6, len(conn_graph.get_all_ops()))
        num_detected_rnns = 0
        for op in conn_graph.get_all_ops().values():
            if op.type == 'SimpleRNN':
                num_detected_rnns += 1
                inner_list = op.internal_ops
                self.assertEqual(49, len(inner_list))
        self.assertEqual(3, num_detected_rnns)

    def test_model_with_basic_lstm_layer(self):
        """ Test connected graph construction on a model with LSTM op """
        tf.reset_default_graph()
        sess = tf.Session()
        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            x = tf.keras.layers.LSTM(12, name='lstm0')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="matmul0")(x)

            init = tf.global_variables_initializer()
            sess.run(init)
            _ = tf.summary.FileWriter('./lstm', sess.graph)

        # construct a connected graph
        conn_graph = ConnectedGraph(sess.graph, ['input_1'], ['matmul0/Softmax'])

        # there should be only 4 connected graph ops, input, LSTM , Dense and Softmax
        self.assertEqual(4, len(conn_graph.get_all_ops()))
        lstm_detected = False
        for op in conn_graph.get_all_ops().values():
            if op.type == 'LSTM':
                lstm_detected = True
                inner_list = op.internal_ops
                self.assertEqual(86, len(inner_list))
                self.assertEqual(op.get_module(), sess.graph.get_operation_by_name('lstm0/while/MatMul'))
                self.assertEqual('lstm0', op.name)
        self.assertTrue(lstm_detected)

        valid_matmuls = []
        valid_muls = []
        valid_bias_add = []
        valid_activation = []
        for op in inner_list:
            if op.type == 'MatMul' and op not in valid_matmuls:
                valid_matmuls.append(op)
            if op.type == 'Mul' and op not in valid_matmuls:
                valid_muls.append(op)
            if op.type == 'BiasAdd' and op not in valid_bias_add:
                valid_bias_add.append(op)
            if op.type == 'Tanh' and op not in valid_activation:
                valid_activation.append(op)

        self.assertEqual(8, len(valid_matmuls))
        self.assertEqual(7, len(valid_muls))
        self.assertEqual(4, len(valid_bias_add))
        self.assertEqual(2, len(valid_activation))


def validate_branch_ops(conn_graph: ConnectedGraph):
    """ A helper function for validating that branch ops are inserted correctly """

    def check_for_branch_op(op_info: ModuleIdentifierOpInfo):
        """
        Look inside conn_graph ops and products for branch ops, and validate connections to parent and child ops
        """

        op = conn_graph.get_all_ops()[op_info.module_name]
        return_bool = True
        product = op.output
        if "branch" not in product.name:
            logger.error("branch not in product name")
            return_bool = False
        if len(product.consumers) > 1:
            logger.error("branch op is not parent op's only consumer")
            return_bool = False
        branch_op = product.consumers[0]
        if branch_op.type != "branch":
            logger.error("parent op's child op is not of type branch")
            return_bool = False
        branch_product = branch_op.output
        if "multiple_ops" not in branch_product.name:
            logger.error("multiple_ops not in branch op's product's name")
            return_bool = False
        if len(branch_product.consumers) <= 1:
            logger.error("branch op's product has one or fewer consumers")
            return_bool = False
        for consumer in branch_product.consumers:
            for input_product in consumer.inputs:
                if input_product.producer == op:
                    logger.error("parent op is still one of child op's inputs (as opposed to branch op)")
                    return_bool = False
        return return_bool

    # pylint: disable=protected-access
    module_identifier = StructureModuleIdentifier(conn_graph.graph, conn_graph._starting_op_names,
                                                  conn_graph._valid_ops)
    num_branches_found = 0
    for tf_op in conn_graph.graph.get_operations():
        # Ignore ops which were not found in the initial depth first search
        if tf_op not in module_identifier.processed_ops:
            continue

        found_branch = False
        for output_tensor in tf_op.outputs:
            if len(output_tensor.consumers()) > 1:
                # Potential branch op. Check if children go to separate modules
                child_module_set = set()
                for consumer_op in output_tensor.consumers():
                    if consumer_op in module_identifier._valid_ops:
                        child_module_info = module_identifier.get_op_info(consumer_op)
                        child_module_set.add(child_module_info.module_name)

                # If children go to separate modules, this should be a branch op
                if len(child_module_set) > 1:
                    found_branch = True
                    break

        if found_branch:
            num_branches_found += 1
            tf_op_info = module_identifier.get_op_info(tf_op)
            if not check_for_branch_op(tf_op_info):
                return False

    logger.info("Found %s branches", num_branches_found)
    return True


def validate_product_tensor_lists(conn_graph: ConnectedGraph):
    """
    For each product, make sure that the length of its consumer list is the same as the length of its tensor
    list (should be a one to one correspondence between the lists)
    """
    for product in conn_graph.get_all_products().values():
        # products going to branch ops will not have tensors associated with them
        if product.consumers[0].type != 'branch':
            if len(product.consumers) != len(product.tensor_dict.keys()):
                return False
    return True
