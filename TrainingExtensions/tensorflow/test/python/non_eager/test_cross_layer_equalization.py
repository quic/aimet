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
""" This file contains unit tests for testing cross layer scaling feature of CLE """

import unittest
import numpy as np
import os
import pytest
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import aimet_tensorflow.utils.graph_saver
from aimet_tensorflow.cross_layer_equalization import CrossLayerScaling, GraphSearchUtils, equalize_model, \
    fold_all_batch_norms, HighBiasFold
from aimet_tensorflow.utils.op.conv import WeightTensorUtils, BiasUtils

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()


class TestCrossLayerEqualization(unittest.TestCase):
    """ Test methods for Cross layer equalization """

    @staticmethod
    def _custom_two_conv_layer_model():
        """
        Builds a custom model with two conv layers
        :return:
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        x = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        x = tf.nn.relu(x, name='ReluInTheMiddle')
        x = tf.keras.layers.Conv2D(32, (3, 3))(x)
        x = tf.nn.relu(x, name='Relu')

        return x

    @staticmethod
    def _custom_three_layer_model_keras():
        """
        Builds a custom model with three conv layers
        :return:
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        x = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        x = tf.nn.relu(x, name='ReluInTheMiddle')
        x = tf.keras.layers.Conv2D(32, (3, 3))(x)
        x = tf.keras.layers.ReLU(name='AnotherRelu')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3))(x)
        x = tf.nn.relu(x, name='Relu')

        return x

    @staticmethod
    def _custom_three_layer_model_keras_prelu():
        """
        Builds a custom model with three conv layers
        :return:
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        x = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        x = tf.nn.relu(x, name='ReluInTheMiddle')
        x = tf.keras.layers.Conv2D(32, (3, 3))(x)
        x = tf.keras.layers.PReLU(name='prelu')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3))(x)
        x = tf.nn.relu(x, name='Relu')

        return x

    def test_find_layer_groups_to_scale_custom_model_with_candidate_layers(self):
        """
        Test find_layer_groups_to_scale() on a custom model
        """
        tf.compat.v1.reset_default_graph()
        _ = TestCrossLayerEqualization._custom_two_conv_layer_model()
        tf.compat.v1.set_random_seed(0)
        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)
        start_op = "inputs"

        graph_util = GraphSearchUtils(tf.compat.v1.get_default_graph(), start_op, 'Relu')
        _ , layer_groups = graph_util.find_layer_groups_to_scale()
        self.assertEqual(1, len(layer_groups))
        sess.close()

    def test_find_layers_groups_tp_scale_custom_model_without_candidate_layers(self):
        """
        Test find_layer_groups_to_scale() on a model without potential layers for scaling
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn_op = tf.keras.layers.BatchNormalization(trainable=False)(conv_op)
        _ = tf.nn.relu(bn_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        graph_util = GraphSearchUtils(tf.compat.v1.get_default_graph(), "inputs", 'Relu')
        _ , layer_groups = graph_util.find_layer_groups_to_scale()
        self.assertEqual(0, len(layer_groups))
        sess.close()

    def test_update_weight_tensor_for_op(self):
        """
        Test update_weight_tensor_for_op() on custom conv op
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        _ = tf.nn.relu(conv_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        initial_data = WeightTensorUtils.get_tensor_as_numpy_data(sess, conv_op)
        wt_data = initial_data + 2

        # this is block1_conv1/Conv2D in VGG16
        WeightTensorUtils.update_tensor_for_op(sess, conv_op, wt_data)
        new_sess = aimet_tensorflow.utils.graph_saver.save_and_load_graph('./temp_conv_wt_updated', sess)

        # check for if reroute was successful
        # read op from conv op should be same as one defined by new variable type
        conv_op = new_sess.graph.get_operation_by_name('conv2d/Conv2D')
        new_wt_data = WeightTensorUtils.get_tensor_as_numpy_data(new_sess, conv_op)

        assert not np.allclose(initial_data, new_wt_data)
        sess.close()
        new_sess.close()

    def test_scale_cls_set_with_conv_layers_custom_model(self):
        """
        Test scale_cls_set_with_conv_layers() on a custom model
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        _ = TestCrossLayerEqualization._custom_two_conv_layer_model()

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        graph_util = GraphSearchUtils(tf.compat.v1.get_default_graph(), "inputs", 'Relu')
        _ , layer_groups_as_tf_ops = graph_util.find_layer_groups_to_scale()
        scaling_factors = CrossLayerScaling.scale_cls_set_with_conv_layers(sess, layer_groups_as_tf_ops[0])
        self.assertEqual(32, len(scaling_factors))

        range_conv1_after_scaling = np.amax(np.abs(WeightTensorUtils.get_tensor_as_numpy_data(
            sess, layer_groups_as_tf_ops[0][0])), axis=(2, 0, 1))
        range_conv2_after_scaling = np.amax(np.abs(WeightTensorUtils.get_tensor_as_numpy_data(
            sess, layer_groups_as_tf_ops[0][1])), axis=(3, 0, 1))

        assert np.allclose(range_conv1_after_scaling, range_conv2_after_scaling)
        sess.close()

    def test_scale_cls_set_with_depthwise_conv_layer_custom_model(self):
        """
        Test test_scale_cls_set_with_depthwise_layers() on a custom model
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        inputs = tf.keras.Input(shape=(10, 10, 3,))
        x = tf.keras.layers.Conv2D(10, (1, 1))(inputs)
        y = tf.keras.layers.DepthwiseConv2D((3, 3), padding='valid',depth_multiplier=1, strides=(1,1), use_bias=False)(x)
        z = tf.keras.layers.Conv2D(10, (1, 1))(y)
        _ = tf.nn.relu(z)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph())
        sess.run(init)

        graph_util = GraphSearchUtils(tf.compat.v1.get_default_graph(), "input_1", 'Relu')
        _ , layer_groups_as_tf_ops = graph_util.find_layer_groups_to_scale()
        scaling_matrix12, scaling_matrix23 = CrossLayerScaling.scale_cls_set_with_depthwise_layers(
            sess, layer_groups_as_tf_ops[0])
        self.assertEqual(10, len(scaling_matrix12))
        self.assertEqual(10, len(scaling_matrix23))
        sess.close()

    def test_scale_model_custom(self):
        """
        Test scale_model on a custom model
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        _ = TestCrossLayerEqualization._custom_two_conv_layer_model()
        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)
        new_sess, scaling_factors = CrossLayerScaling.scale_model(sess, "inputs", 'Relu')
        # scaling factors for number of groups selected for scaling returned
        self.assertEqual(1, len(scaling_factors))
        self.assertTrue(scaling_factors[0].cls_pair_info_list[0].relu_activation_between_layers)
        sess.close()
        new_sess.close()

    def test_scale_three_layer_model(self):
        """ Test scale_model on a custom 3-layer model """
        tf.compat.v1.reset_default_graph()
        _ = TestCrossLayerEqualization._custom_three_layer_model_keras()
        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)
        new_sess, scaling_factors = CrossLayerScaling.scale_model(sess, "inputs", 'Relu')
        # scaling factors for number of groups selected for scaling returned
        self.assertEqual(2, len(scaling_factors))
        self.assertTrue(scaling_factors[0].cls_pair_info_list[0].relu_activation_between_layers)
        self.assertTrue(scaling_factors[1].cls_pair_info_list[0].relu_activation_between_layers)
        sess.close()
        new_sess.close()

    def test_scale_three_layer_model_with_prelu(self):
        """
        Test scale_model on a custom 3-layer model with prelu
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        _ = TestCrossLayerEqualization._custom_three_layer_model_keras_prelu()
        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)
        new_sess, scaling_factors = CrossLayerScaling.scale_model(sess, "inputs", 'Relu')
        # scaling factors for number of groups selected for scaling returned
        self.assertEqual(2, len(scaling_factors))
        self.assertTrue(scaling_factors[0].cls_pair_info_list[0].relu_activation_between_layers)
        self.assertTrue(scaling_factors[1].cls_pair_info_list[0].relu_activation_between_layers)
        sess.close()
        new_sess.close()

    def test_relu6_replaced_with_relu(self):
        """
        Test replacing Relu6 wth Relu
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        _ = tf.nn.relu6(conv_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        bias_add = sess.graph.get_operation_by_name('conv2d/BiasAdd')
        self.assertEqual('Relu6', bias_add.outputs[0].consumers()[0].type)

        #update Relu
        start_op = "input_1"
        graph_util = GraphSearchUtils(sess.graph, start_op, 'Relu6')
        after_relu_replace_sess = graph_util.find_and_replace_relu6_with_relu(sess)

        updated_bias_add = after_relu_replace_sess.graph.get_operation_by_name('conv2d/BiasAdd')
        self.assertEqual('Relu', updated_bias_add.outputs[0].consumers()[0].type)
        sess.close()
        after_relu_replace_sess.close()

    def test_cls_with_conv_depthwiseconv_conv_layers(self):
        """
        Test cross layer scaling with Conv, depthwiseConv2D and Conv layers
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential((
                tf.keras.Input(shape=(32, 32, 3,)),
                tf.keras.layers.Conv2D(32, (3, 3)),
                tf.keras.layers.DepthwiseConv2D(kernel_size=3, depth_multiplier=1), #only depth_multiplier=1 is supported
                tf.keras.layers.Conv2D(32, (3, 3)),
                tf.keras.layers.Dense(32, activation=tf.nn.softmax)
            ))
            model.summary()

            init = tf.compat.v1.global_variables_initializer()
            sess = tf.compat.v1.Session()
            sess.run(init)

            in_t = sess.graph.get_tensor_by_name('input_1:0')
            in_values = np.random.rand(1, 32, 32, 3)
            out_1 = sess.run(sess.graph.get_operation_by_name('dense/Softmax').outputs[0], feed_dict={in_t: in_values})

            sess_updated, cls_set_info_list = CrossLayerScaling.scale_model(sess, 'input_1', 'dense/Softmax')
            int_t_updated = sess_updated.graph.get_tensor_by_name('input_1:0')
            out_2 = sess_updated.run(sess_updated.graph.get_operation_by_name('dense/Softmax').outputs[0],
                                     feed_dict={int_t_updated: in_values})

            assert np.allclose(out_1, out_2)
            assert len(cls_set_info_list[0].cls_pair_info_list) == 2, 'expect two supported layer pair'
            assert cls_set_info_list[0].cls_pair_info_list[0].layer1.type == 'Conv2D'
            assert cls_set_info_list[0].cls_pair_info_list[0].layer2.type == 'DepthwiseConv2dNative'
            assert cls_set_info_list[0].cls_pair_info_list[1].layer1.type == 'DepthwiseConv2dNative'
            assert cls_set_info_list[0].cls_pair_info_list[1].layer2.type == 'Conv2D'

    def test_cls_with_depthwiseconv_conv_layers(self):
        """
        Test cross layer scaling with depthwiseConv2D and Conv layers
        """

        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential((
                tf.keras.Input(shape=(32, 32, 3,)),
                tf.keras.layers.DepthwiseConv2D(kernel_size=3, depth_multiplier=1),
                # only depth_multiplier=1 is supported
                tf.keras.layers.Conv2D(32, (4, 4)),
                tf.keras.layers.Dense(32, activation=tf.nn.softmax)
            ))
            model.summary()

            init = tf.compat.v1.global_variables_initializer()
            sess = tf.compat.v1.Session()
            sess.run(init)

            in_t = sess.graph.get_tensor_by_name('input_1:0')
            in_values = np.random.rand(1, 32, 32, 3)
            out_1 = sess.run(sess.graph.get_operation_by_name('dense/Softmax').outputs[0], feed_dict={in_t: in_values})

            sess_updated, cls_set_info_list = CrossLayerScaling.scale_model(sess, 'input_1', 'dense/Softmax')
            int_t_updated = sess_updated.graph.get_tensor_by_name('input_1:0')
            out_2 = sess_updated.run(sess_updated.graph.get_operation_by_name('dense/Softmax').outputs[0],
                                     feed_dict={int_t_updated: in_values})

            assert np.allclose(out_1, out_2)
            assert len(cls_set_info_list[0].cls_pair_info_list) == 1, 'expect only one supported layer pair'
            assert cls_set_info_list[0].cls_pair_info_list[0].layer1.type == 'DepthwiseConv2dNative'
            assert cls_set_info_list[0].cls_pair_info_list[0].layer2.type == 'Conv2D'

    def test_cls_with_conv_conv_layers(self):
        """
        Test cross layer scaling with Conv and Conv layers
        """

        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential((
                tf.keras.Input(shape=(32, 32, 3,)),
                tf.keras.layers.Conv2D(32, (3, 3)),
                tf.keras.layers.Conv2D(32, (3, 3)),
                tf.keras.layers.Dense(32, activation=tf.nn.softmax)
            ))
            model.summary()

            init = tf.compat.v1.global_variables_initializer()
            sess = tf.compat.v1.Session()
            sess.run(init)

            in_t = sess.graph.get_tensor_by_name('input_1:0')
            in_values = np.random.rand(1, 32, 32, 3)
            out_1 = sess.run(sess.graph.get_operation_by_name('dense/Softmax').outputs[0], feed_dict={in_t: in_values})
            # print(out_1)

            sess_updated, cls_set_info_list = CrossLayerScaling.scale_model(sess, 'input_1', 'dense/Softmax')
            int_t_updated = sess_updated.graph.get_tensor_by_name('input_1:0')
            out_2 = sess_updated.run(sess_updated.graph.get_operation_by_name('dense/Softmax').outputs[0],
                                     feed_dict={int_t_updated: in_values})

            assert np.allclose(out_1, out_2)
            assert len(cls_set_info_list[0].cls_pair_info_list) == 1, 'expect only one supported layer pair'
            assert cls_set_info_list[0].cls_pair_info_list[0].layer1.type == 'Conv2D'
            assert cls_set_info_list[0].cls_pair_info_list[0].layer2.type == 'Conv2D'

    def test_find_layer_groups_to_scale(self):
        """
        test conv+conv combination
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential((
                tf.keras.Input(shape=(32, 32, 3,)),
                tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
                tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
                tf.keras.layers.Dense(32, activation=tf.nn.softmax)
            ))
            model.summary()

            init = tf.compat.v1.global_variables_initializer()
            sess = tf.compat.v1.Session()
            sess.run(init)

            in_t = sess.graph.get_tensor_by_name('input_1:0')
            in_values = np.random.rand(1, 32, 32, 3)
            out_1 = sess.run(sess.graph.get_operation_by_name('dense/Softmax').outputs[0], feed_dict={in_t: in_values})
            # print(out_1)

            sess_updated, cls_set_info_list = CrossLayerScaling.scale_model(sess, 'input_1', 'dense/Softmax')
            int_t_updated = sess_updated.graph.get_tensor_by_name('input_1:0')
            out_2 = sess_updated.run(sess_updated.graph.get_operation_by_name('dense/Softmax').outputs[0],
                                     feed_dict={int_t_updated: in_values})

            assert np.allclose(out_1, out_2)
            assert len(cls_set_info_list[0].cls_pair_info_list) == 1, 'expect only one supported layer pair'
            assert cls_set_info_list[0].cls_pair_info_list[0].layer1.type == 'Conv2D'
            assert cls_set_info_list[0].cls_pair_info_list[0].layer2.type == 'Conv2D'

    def test_find_layer_groups_to_scale_2(self):
        """
        test conv+depthwise+conv combination
        """
        tf.compat.v1.reset_default_graph()
        with tf.device('/cpu:0'):
            model = tf.keras.Sequential((
                tf.keras.Input(shape=(32, 32, 3,)),
                tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
                tf.keras.layers.DepthwiseConv2D(kernel_size=3, depth_multiplier=1),
                tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
                tf.keras.layers.Dense(32, activation=tf.nn.softmax)
            ))
            model.summary()

            init = tf.compat.v1.global_variables_initializer()
            sess = tf.compat.v1.Session()
            sess.run(init)

            in_t = sess.graph.get_tensor_by_name('input_1:0')
            in_values = np.random.rand(1, 32, 32, 3)
            out_1 = sess.run(sess.graph.get_operation_by_name('dense/Softmax').outputs[0], feed_dict={in_t: in_values})
            # print(out_1)

            sess_updated, cls_set_info_list = CrossLayerScaling.scale_model(sess, 'input_1', 'dense/Softmax')
            int_t_updated = sess_updated.graph.get_tensor_by_name('input_1:0')
            out_2 = sess_updated.run(sess_updated.graph.get_operation_by_name('dense/Softmax').outputs[0],
                                     feed_dict={int_t_updated: in_values})

            assert np.allclose(out_1, out_2)
            assert len(cls_set_info_list[0].cls_pair_info_list) == 2, 'expect two supported layer pair'
            assert cls_set_info_list[0].cls_pair_info_list[0].layer1.type == 'Conv2D'
            assert cls_set_info_list[0].cls_pair_info_list[0].layer2.type == 'DepthwiseConv2dNative'
            assert cls_set_info_list[0].cls_pair_info_list[1].layer1.type == 'DepthwiseConv2dNative'
            assert cls_set_info_list[0].cls_pair_info_list[1].layer2.type == 'Conv2D'

    def test_high_bias_fold_two_bn_folded_convs(self):
        """
        Test high bias fold with a custom model with two BN folded convs
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        np.random.seed(0)
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn_op = tf.keras.layers.BatchNormalization(trainable=False)(conv_op)
        relu_1= tf.nn.relu(bn_op)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(relu_1)
        bn_op_2 = tf.keras.layers.BatchNormalization(trainable=False)(conv2_op)
        relu_2 = tf.nn.relu(bn_op_2)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        start_op_names = conv_op.inputs[0].op.name
        output_op_names = 'Relu_1'

        # fold batchnorm layers
        after_bn_fold_sess, folded_pairs = fold_all_batch_norms(sess, start_op_names, output_op_names)

        # replace any ReLU6 layers with ReLU
        graph_util = GraphSearchUtils(after_bn_fold_sess.graph, start_op_names, output_op_names)
        after_relu_replace_sess = graph_util.find_and_replace_relu6_with_relu(after_bn_fold_sess)

        # perform cross-layer scaling on applicable layer sets
        after_cls_sess, cls_set_info_list = CrossLayerScaling.scale_model(after_relu_replace_sess, start_op_names,
                                                                          output_op_names)

        # we want to validate that after high bias fold, bias for conv is >= bias before high bias fold.
        conv_op = after_cls_sess.graph.get_operation_by_name('conv2d_1/Conv2D')
        before_high_bias_fold_bias_data = BiasUtils.get_bias_as_numpy_data(after_cls_sess, conv_op)

        # perform high-bias fold
        after_hbf_sess = HighBiasFold.bias_fold(after_cls_sess, folded_pairs, cls_set_info_list)

        # read updated bias value
        conv_op = after_hbf_sess.graph.get_operation_by_name('conv2d_1/Conv2D')
        high_bias_folded_bias_data = BiasUtils.get_bias_as_numpy_data(after_hbf_sess, conv_op)

        for i in range(len(before_high_bias_fold_bias_data)):
            # folded bias should be greater than previous bias
            self.assertTrue(high_bias_folded_bias_data[i] >= before_high_bias_fold_bias_data[i])

        sess.close()
        after_bn_fold_sess.close()
        after_relu_replace_sess.close()
        after_cls_sess.close()
        after_hbf_sess.close()

    def test_bias_add_custom_model(self):
        """ test update bias when no bias present """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(inputs)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(inputs)
        relu2= tf.nn.relu(conv2_op)
        add = tf.keras.layers.add([conv_op, relu2])
        _ = tf.nn.relu(add)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        shape = WeightTensorUtils.get_tensor_shape(conv_op.op)
        np.random.seed(0)
        bias_data = np.random.rand(shape[3])

        assert BiasUtils.is_bias_none(conv_op.op)
        BiasUtils.update_bias_for_op(sess, conv_op.op, bias_data)
        n_sess = aimet_tensorflow.utils.graph_saver.save_and_load_graph('./test_update', sess)

        conv_op_updated = n_sess.graph.get_operation_by_name(conv_op.op.name)
        assert not BiasUtils.is_bias_none(conv_op_updated)
        updated_bias = BiasUtils.get_bias_as_numpy_data(n_sess, conv_op_updated)
        self.assertTrue(np.allclose(updated_bias, bias_data))
        sess.close()
        n_sess.close()

    def test_cls_layer_select_conv_with_identity(self):
        """
        test cross layer scaling layer selection code when convs have identity nodes in-btw.
        This was observed with TF Slim Mobilenetv2 model
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv1_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        relu_op = tf.nn.relu(conv1_op)
        identity = tf.identity(relu_op)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(identity)
        relu2_op = tf.nn.relu(conv2_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        start_op = "inputs"
        output_op = 'Relu_1'

        graph_search = GraphSearchUtils(sess.graph, start_op, output_op)
        _ , layer_groups_as_tf_ops = graph_search.find_layer_groups_to_scale()

        assert len(layer_groups_as_tf_ops) == 1

        sess.close()

    def test_high_bias_fold_custom_model(self):
        """
        Test high bias fold with a custom model
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        relu_1= tf.nn.relu(conv_op)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(relu_1)
        bn_op_2 = tf.keras.layers.BatchNormalization(trainable=False)(conv2_op)
        conv3_op = tf.keras.layers.Conv2D(32, (3, 3))(bn_op_2)
        relu_2 = tf.nn.relu(conv3_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        np.random.seed(0)
        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        start_op_names = conv_op.inputs[0].op.name
        output_op_names = relu_2.op.name

        # fold batchnorm layers
        after_bn_fold_sess, folded_pairs = fold_all_batch_norms(sess, start_op_names, output_op_names)

        # replace any ReLU6 layers with ReLU
        graph_util = GraphSearchUtils(after_bn_fold_sess.graph, start_op_names, output_op_names)
        after_relu_replace_sess = graph_util.find_and_replace_relu6_with_relu(after_bn_fold_sess)

        # perform cross-layer scaling on applicable layer sets
        after_cls_sess, cls_set_info_list = CrossLayerScaling.scale_model(after_relu_replace_sess, start_op_names,
                                                                          output_op_names)

        # we want to validate that after high bias fold, bias for conv is >= bias before high bias fold.
        conv_op = after_cls_sess.graph.get_operation_by_name('conv2d_2/Conv2D')
        before_high_bias_fold_bias_data = BiasUtils.get_bias_as_numpy_data(after_cls_sess, conv_op)

        # perform high-bias fold
        after_hbf_sess = HighBiasFold.bias_fold(after_cls_sess, folded_pairs, cls_set_info_list)

        # read updated bias value
        conv_op = after_hbf_sess.graph.get_operation_by_name('conv2d_2/Conv2D')
        high_bias_folded_bias_data = BiasUtils.get_bias_as_numpy_data(after_hbf_sess, conv_op)

        for i in range(len(before_high_bias_fold_bias_data)):
            # folded bias should be greater than previous bias
            self.assertTrue(high_bias_folded_bias_data[i] >= before_high_bias_fold_bias_data[i])

        sess.close()
        after_bn_fold_sess.close()
        after_relu_replace_sess.close()
        after_cls_sess.close()
        after_hbf_sess.close()

    def test_equalize_model_multi_input(self):
        """
        Test bn fold with multiple input nodes
        """
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(0)
        input1 = tf.keras.Input(name='input1', shape=(10, 10, 3))
        input2 = tf.keras.Input(name='input2', shape=(12, 12, 3))
        x1 = tf.keras.layers.Conv2D(8, (1, 1), name='conv1a',
                                    kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                    bias_initializer='random_uniform')(input1)
        x2 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1b',
                                    kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                    bias_initializer='random_uniform')(x1)
        x3 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1c',
                                    kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                    bias_initializer='random_uniform')(input2)
        x4 = tf.keras.layers.Conv2D(8, (3, 3), name='conv1d',
                                    kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                    bias_initializer='random_uniform')(x3)
        x = tf.keras.layers.add([x2, x4])
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(x)
        bn_op = tf.keras.layers.BatchNormalization(trainable=False)(conv2_op)
        _ = tf.nn.relu(bn_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        conv_1b_before_equalize = sess.graph.get_operation_by_name('conv1b/Conv2D')
        conv_1b_bias_data_before_fold = BiasUtils.get_bias_as_numpy_data(sess, conv_1b_before_equalize)
        conv_1d_before_equalize = sess.graph.get_operation_by_name('conv1d/Conv2D')
        conv_1d_bias_data_before_fold = BiasUtils.get_bias_as_numpy_data(sess, conv_1d_before_equalize)

        new_sess = equalize_model(sess, ["input1", "input2"], 'Relu')

        conv_1b_after_equalize = new_sess.graph.get_operation_by_name('conv1b/Conv2D')
        conv_1b_bias_data_after_fold = BiasUtils.get_bias_as_numpy_data(new_sess, conv_1b_after_equalize)
        conv_1d_after_equalize = new_sess.graph.get_operation_by_name('conv1d/Conv2D')
        conv_1d_bias_data_after_fold = BiasUtils.get_bias_as_numpy_data(new_sess, conv_1d_after_equalize)

        for i in range(len(conv_1b_bias_data_after_fold)):
            self.assertTrue(conv_1b_bias_data_after_fold[i] <= conv_1b_bias_data_before_fold[i])

        for i in range(len(conv_1d_bias_data_after_fold)):
            self.assertTrue(conv_1d_bias_data_after_fold[i] <= conv_1d_bias_data_before_fold[i])

        sess.close()
        new_sess.close()

    def test_equalize_with_custom_model_no_bias(self):
        """
        Test equalize with a custom model with conv without bias param
        """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        with sess.as_default():
            inputs = tf.keras.Input(shape=(32, 32, 3,))
            conv_op = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(inputs)
            bn_op = tf.keras.layers.BatchNormalization(trainable=False)(conv_op)
            relu_1= tf.nn.relu(bn_op)
            conv2_op = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(relu_1)
            bn_op_2 = tf.keras.layers.BatchNormalization(fused=True)(conv2_op, training=False)
            relu_2 = tf.nn.relu(bn_op_2)

            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

            old_conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
            self.assertTrue(BiasUtils.is_bias_none(old_conv_op))

            conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
            new_sess = equalize_model(sess, conv_op.inputs[0].op.name, 'Relu_1')

            new_conv_op = new_sess.graph.get_operation_by_name('conv2d/Conv2D')
            bias = BiasUtils.get_bias_as_numpy_data(new_sess, new_conv_op)
            self.assertFalse(BiasUtils.is_bias_none(new_conv_op))
        sess.close()
        new_sess.close()

    def test_equalize_fold_forward(self):
        """
        Test equalize on a model with a forward bn fold
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        r_op = tf.nn.relu(conv_op)
        bn_op = tf.keras.layers.BatchNormalization(trainable=False)(r_op)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(bn_op)
        conv3_op = tf.keras.layers.Conv2D(32, (3, 3))(conv2_op)
        _ = tf.nn.relu(conv3_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph())
        sess.run(init)
        old_conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        conv_bias_data_before_fold = BiasUtils.get_bias_as_numpy_data(sess, old_conv_op)

        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')

        new_sess = equalize_model(sess, conv_op.inputs[0].op.name, 'Relu_1')
        new_conv_op = new_sess.graph.get_operation_by_name('conv2d/Conv2D')
        self.assertFalse(BiasUtils.is_bias_none(new_conv_op))
        conv_bias_data_after_fold = BiasUtils.get_bias_as_numpy_data(new_sess, new_conv_op)

        for i in range(len(conv_bias_data_before_fold)):
            self.assertTrue(conv_bias_data_before_fold[i] <= conv_bias_data_after_fold[i])

        sess.close()
        new_sess.close()
