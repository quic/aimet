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
""" This file contains unit tests for testing bias correction """

import unittest
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
from unittest.mock import MagicMock

import aimet_tensorflow.utils.graph_saver
from aimet_tensorflow.utils.op.conv import BiasUtils, WeightTensorUtils
from aimet_tensorflow.cross_layer_equalization import equalize_model
from aimet_tensorflow.bias_correction import BiasCorrectionParams, BiasCorrection, QuantParams
from aimet_tensorflow.examples.test_models import keras_model_functional
from aimet_tensorflow.utils.graph_saver import save_and_load_graph
from aimet_common.defs import ActivationType

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()


class TestBiasCorrection(unittest.TestCase):
    """ Test methods for BiasCorrection"""

    def test_get_output_data(self):
        """
        Test get_output_data method
        """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session(graph=tf.Graph())
        input_op_names = ['input_1']
        output_op_name = 'scope_1/conv2d_2/Conv2D'
        with sess.graph.as_default():
            _ = keras_model_functional()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

        data = np.random.rand(1, 32, 32, 3)
        output = BiasCorrection._get_output_data(sess, input_op_names, output_op_name, data)
        self.assertEqual(output.shape[3], 8)
        sess.close()

    def test_bias_correction_single_layer(self):
        """
        Test bias correction for a single layer api
        """
        tf.compat.v1.reset_default_graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # create a custom model
        inputs = tf.keras.Input(shape=(32, 16, 3,))
        conv_op = tf.keras.layers.Conv2D(16, (3, 3))(inputs)
        relu_1 = tf.nn.relu(conv_op)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(relu_1)
        relu_2 = tf.nn.relu(conv2_op)

        # global initializer
        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(config=config, graph=tf.compat.v1.get_default_graph())
        sess.run(init)

        # populate conv with dummy bias and weights
        np.random.seed(0)
        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        w_shape = WeightTensorUtils.get_tensor_shape(conv_op)
        w_numpy_data = np.random.rand(w_shape[0],w_shape[1],w_shape[2],w_shape[3])
        b_shape = BiasUtils.get_shape(conv_op)
        b_numpy_data = np.random.rand(b_shape[0])

        WeightTensorUtils.update_tensor_for_op(sess, conv_op, w_numpy_data)
        BiasUtils.update_bias_for_op(sess, conv_op, b_numpy_data)
        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        bool_ = BiasUtils.is_bias_none(conv_op)

        # save and load the updated graph after high bias fold update
        n_sess = aimet_tensorflow.utils.graph_saver.save_and_load_graph('./test_update', sess)

        output_op = n_sess.graph.get_operation_by_name('Relu_1')
        conv_op = n_sess.graph.get_operation_by_name('conv2d/Conv2D')
        import copy
        bias_data = copy.deepcopy(BiasUtils.get_bias_as_numpy_data(n_sess, conv_op))

        input_op_name = conv_op.inputs[0].op.name

        bias_corr_input = BiasCorrectionParams(batch_size=1, num_quant_samples=10,
                                               num_bias_correct_samples=10,
                                               input_op_names=[input_op_name],
                                               output_op_names =[output_op.name])

        quant_params = QuantParams(use_cuda=False)

        np.random.seed(0)
        shape = conv_op.inputs[0].shape
        dataset = np.random.rand(1, shape[1], shape[2], shape[3])

        with unittest.mock.patch('aimet_tensorflow.bias_correction.iter_first_x') as iter_first_x:
            iter_first_x.return_value = [dataset]
            quantsim = BiasCorrection._get_quantized_model(n_sess, quant_params, bias_corr_input.input_op_names,
                                                           bias_corr_input.output_op_names, bias_corr_input.num_quant_samples,
                                                           bias_corr_input.batch_size, dataset)
            BiasCorrection.bias_correction_per_layer(reference_model=sess,
                                                     corrected_model=quantsim.session,
                                                     bias_correct_params= bias_corr_input,
                                                     layer_name_to_be_corrected = conv_op.name,
                                                     data_set = dataset)

            conv_op = quantsim.session.graph.get_operation_by_name('conv2d/Conv2D')
            bias_data_updated = BiasUtils.get_bias_as_numpy_data(quantsim.session, conv_op)

        # needs a validation
        self.assertFalse(np.allclose(bias_data, bias_data_updated, atol=1e-4))
        print('Test completed')

        sess.close()
        n_sess.close()
        quantsim.session.close()

    def test_bias_correction_model_tf_enhanced(self):
        """
        Test bias correction for custom model
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        relu_1 = tf.nn.relu(conv_op)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3),use_bias=False)(relu_1)
        relu_2 = tf.nn.relu(conv2_op)
        conv3_op = tf.keras.layers.Conv2D(32, (3, 3))(relu_2)
        relu_3 = tf.nn.relu(conv3_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        # updating random bias and weight for one conv
        np.random.seed(0)
        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        b_shape = BiasUtils.get_shape(conv_op)
        numpy_data = np.random.rand(b_shape[0])
        BiasUtils.update_bias_for_op(sess, conv_op, numpy_data)
        w_shape = WeightTensorUtils.get_tensor_shape(conv_op)
        w_numpy_data = np.random.rand(w_shape[0],w_shape[1],w_shape[2],w_shape[3])
        WeightTensorUtils.update_tensor_for_op(sess, conv_op, w_numpy_data)

        # save and load the updated graph after high bias fold update
        n_sess = aimet_tensorflow.utils.graph_saver.save_and_load_graph('./test_update', sess)

        output_op = n_sess.graph.get_operation_by_name('Relu_1')
        conv_op = n_sess.graph.get_operation_by_name('conv2d/Conv2D')
        bias_data = BiasUtils.get_bias_as_numpy_data(n_sess, conv_op)

        input_op_name = conv_op.inputs[0].op.name
        output_op = n_sess.graph.get_operation_by_name('Relu_2')

        input_op_names = [input_op_name]
        output_op_names = [output_op.name]

        batch_size = 1
        num_samples = 10

        np.random.seed(0)
        shape = conv_op.inputs[0].shape

        dataset = np.random.rand(10, 1, shape[1], shape[2], shape[3])
        dataset = tf.convert_to_tensor(dataset)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

        quant_params = QuantParams(use_cuda=False)
        quant_params.use_cuda = False
        bias_correction_params = BiasCorrectionParams(batch_size=batch_size,
                                                      num_quant_samples=num_samples,
                                                      num_bias_correct_samples=num_samples,
                                                      input_op_names=input_op_names,
                                                      output_op_names=output_op_names)

        conv_op = sess.graph.get_operation_by_name('conv2d_1/Conv2D')
        assert(BiasUtils.is_bias_none(conv_op))
        new_sess = BiasCorrection.correct_bias(n_sess, bias_correction_params, quant_params, dataset)

        conv_op = new_sess.graph.get_operation_by_name('conv2d_1/Conv2D')
        assert(not BiasUtils.is_bias_none(conv_op))

        sess.close()
        n_sess.close()
        new_sess.close()

    def test_bias_correction_model_tf(self):
        """
        Test bias correction for custom model
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        relu_1 = tf.nn.relu(conv_op)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(relu_1)
        relu_2 = tf.nn.relu(conv2_op)
        conv3_op = tf.keras.layers.Conv2D(32, (3, 3))(relu_2)
        relu_3 = tf.nn.relu(conv3_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        # updating random bias and weight for one conv
        np.random.seed(0)
        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        b_shape = BiasUtils.get_shape(conv_op)
        numpy_data = np.random.rand(b_shape[0])
        BiasUtils.update_bias_for_op(sess, conv_op, numpy_data)
        w_shape = WeightTensorUtils.get_tensor_shape(conv_op)
        w_numpy_data = np.random.rand(w_shape[0], w_shape[1], w_shape[2], w_shape[3])
        WeightTensorUtils.update_tensor_for_op(sess, conv_op, w_numpy_data)

        # save and load the updated graph after high bias fold update
        n_sess = save_and_load_graph('./test_update', sess)

        output_op = n_sess.graph.get_operation_by_name('Relu_1')
        conv_op = n_sess.graph.get_operation_by_name('conv2d/Conv2D')
        bias_data = BiasUtils.get_bias_as_numpy_data(n_sess, conv_op)

        input_op_name = conv_op.inputs[0].op.name
        output_op = n_sess.graph.get_operation_by_name('Relu_2')

        input_op_names = [input_op_name]
        output_op_names = [output_op.name]

        batch_size = 1
        num_samples = 10

        np.random.seed(0)
        shape = conv_op.inputs[0].shape

        dataset = np.random.rand(10, 1, shape[1], shape[2], shape[3])
        dataset = tf.convert_to_tensor(dataset)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

        quant_params = QuantParams(quant_mode='tf', use_cuda=False)
        bias_correction_params = BiasCorrectionParams(batch_size=batch_size,
                                                      num_quant_samples=num_samples,
                                                      num_bias_correct_samples=num_samples,
                                                      input_op_names=input_op_names,
                                                      output_op_names=output_op_names)

        conv_op = sess.graph.get_operation_by_name('conv2d_1/Conv2D')
        assert(BiasUtils.is_bias_none(conv_op))
        new_sess = BiasCorrection.correct_bias(sess, bias_correction_params, quant_params, dataset)

        conv_op = new_sess.graph.get_operation_by_name('conv2d_1/Conv2D')
        assert(not BiasUtils.is_bias_none(conv_op))

        sess.close()
        n_sess.close()
        new_sess.close()

    def test_bias_update_to_dense(self):
        """
        test bias correction on matmul layer
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        x = tf.keras.layers.Flatten()(inputs)
        dense = tf.keras.layers.Dense(2, use_bias=False, activation=tf.nn.softmax, name="single_residual")(x)
        _ = tf.nn.relu(dense)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph())
        sess.run(init)

        input_op = sess.graph.get_operation_by_name('input_1')
        output_op = sess.graph.get_operation_by_name('Relu')

        input_op_names = ['input_1']
        output_op_names = [output_op.name]

        batch_size = 1
        num_samples = 10

        np.random.seed(0)
        shape = input_op.outputs[0].shape

        dataset = np.random.rand(10, 1, shape[1], shape[2], shape[3])
        dataset = tf.convert_to_tensor(dataset)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

        quant_params = QuantParams(use_cuda=False)
        bias_correction_params = BiasCorrectionParams(batch_size=batch_size,
                                                      num_quant_samples=num_samples,
                                                      num_bias_correct_samples=num_samples,
                                                      input_op_names=input_op_names,
                                                      output_op_names=output_op_names)

        dense_conv_op = sess.graph.get_operation_by_name('single_residual/MatMul')
        assert(BiasUtils.is_bias_none(dense_conv_op))

        new_sess = BiasCorrection.correct_bias(sess, bias_correction_params, quant_params, dataset)
        updated_dense_conv_op = new_sess.graph.get_operation_by_name('single_residual/MatMul')
        bias = BiasUtils.get_bias_as_numpy_data(new_sess, updated_dense_conv_op)
        assert(not BiasUtils.is_bias_none(updated_dense_conv_op))

        sess.close()
        new_sess.close()

    def test_depthwise_custom(self):
        """ test depthwise conv2d layer withput bias """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(10, 10, 3,))
        x = tf.keras.layers.Conv2D(10, (1, 1))(inputs)
        with tf.compat.v1.variable_scope("standalone_depthwise"):
            x = tf.compat.v1.nn.depthwise_conv2d_native(x,
                                                        tf.compat.v1.get_variable(initializer=tf.random.truncated_normal(shape=(3, 3, 10, 1)),
                                                                                  name="depthwise_kernel"),
                                                        [1, 1, 1, 1],
                                                        'VALID')
        _ = tf.nn.relu(x)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph())
        sess.run(init)

        depthwise_conv_op = sess.graph.get_operation_by_name('standalone_depthwise/DepthwiseConv2dNative')
        input_op = sess.graph.get_operation_by_name('input_1')
        output_op = sess.graph.get_operation_by_name('Relu')

        input_op_names = ['input_1']
        output_op_names = [output_op.name]

        batch_size = 1
        num_samples = 10

        np.random.seed(0)
        shape = input_op.outputs[0].shape

        dataset = np.random.rand(10, 1, shape[1], shape[2], shape[3])
        dataset = tf.convert_to_tensor(dataset)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

        quant_params = QuantParams(use_cuda=False)
        bias_correction_params = BiasCorrectionParams(batch_size=batch_size,
                                                      num_quant_samples=num_samples,
                                                      num_bias_correct_samples=num_samples,
                                                      input_op_names=input_op_names,
                                                      output_op_names=output_op_names)

        assert(BiasUtils.is_bias_none(depthwise_conv_op))

        new_sess = BiasCorrection.correct_bias(sess, bias_correction_params, quant_params, dataset)

        updated_conv_op = new_sess.graph.get_operation_by_name('standalone_depthwise/DepthwiseConv2dNative')

        assert(not BiasUtils.is_bias_none(updated_conv_op))

        sess.close()
        new_sess.close()

    def test_bn_based_bias_correction_layer_selection_bn_conv(self):
        """
        Test layer selection for bn based bias correction
        patterns:
        BN -> Conv
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(conv_op)
        relu = tf.nn.relu(bn_op)
        conv1_op = tf.keras.layers.Conv2D(32, (3, 3))(relu)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(conv1_op)
        _ = tf.nn.relu(conv2_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        start_op = "inputs"

        bn_conv_linear_dict = BiasCorrection.find_all_convs_bn_with_activation(sess, start_op, 'Relu_1')

        assert(len(bn_conv_linear_dict) == 1)
        sess.close()

    def test_layer_selection_bn_depthwise_conv(self):
        """
        Test layer selection with depthwise_layer on a custom model
        patterns:
        BN -> Depthwise conv
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(10, 10, 3,))
        x = tf.keras.layers.Conv2D(10, (1, 1))(inputs)
        bn_op = tf.keras.layers.BatchNormalization(fused=True)(x, training=False)

        with tf.compat.v1.variable_scope("standalone_depthwise"):
            x = tf.compat.v1.nn.depthwise_conv2d_native(bn_op,
                                              tf.compat.v1.get_variable(initializer=tf.random.truncated_normal(shape=(3, 3, 10, 1)),
                                                              name="depthwise_kernel"),
                                              [1, 1, 1, 1],
                                              'VALID')
        _ = tf.nn.relu(x)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph())
        sess.run(init)

        start_op = "input_1"

        bn_conv_linear_dict = BiasCorrection.find_all_convs_bn_with_activation(sess, start_op, 'Relu')
        depthwise_op = sess.graph.get_operation_by_name('standalone_depthwise/DepthwiseConv2dNative')
        assert(1 == len(bn_conv_linear_dict))
        assert(depthwise_op in bn_conv_linear_dict.keys())
        sess.close()

    def test_bn_conv_layer_selection_bn_relu_conv(self):
        """
        Test layer selection code
        patterns:
        BN -> Relu -> conv
        BN -> (no activation) -> Conv
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")

        conv_op = tf.keras.layers.Conv2D(32, (3, 3))(inputs)

        bn_op = tf.keras.layers.BatchNormalization(fused=True)(conv_op, training=False)
        relu = tf.nn.relu(bn_op)
        conv1_op = tf.keras.layers.Conv2D(32, (3, 3))(relu)

        relu_2 = tf.nn.relu(conv1_op)
        bn_op_2 = tf.keras.layers.BatchNormalization(fused=True)(relu_2)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(bn_op_2, training=False)

        _ = tf.nn.relu(conv2_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)
        start_op = "inputs"

        bn_conv_linear_dict = BiasCorrection.find_all_convs_bn_with_activation(sess, start_op, 'Relu_2')

        assert(2 == len(bn_conv_linear_dict))
        conv1_op = sess.graph.get_operation_by_name('conv2d_1/Conv2D')
        assert bn_conv_linear_dict[conv1_op].in_activation_type == ActivationType.relu
        assert bn_conv_linear_dict[conv1_op].input_bn is not None
        conv2_op = sess.graph.get_operation_by_name('conv2d_2/Conv2D')
        assert bn_conv_linear_dict[conv2_op].in_activation_type == ActivationType.no_activation
        assert bn_conv_linear_dict[conv2_op].input_bn is not None
        sess.close()

    def test_bn_based_bias_correction_single_layer_functions_invoked(self):
        """
        Test bn based bias correction for a single layer api methods invoked correctly
        """
        # create a custom model
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv1_op = tf.keras.layers.Conv2D(32, (3, 3),
                                          kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                          bias_initializer='random_uniform')(inputs)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(conv1_op)
        _ = tf.nn.relu(conv2_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        sess.run(init)

        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        output_op = sess.graph.get_operation_by_name('Relu')
        input_op_name = "inputs"

        bias_corr_input = BiasCorrectionParams(batch_size=1, num_quant_samples=10,
                                               num_bias_correct_samples=10,
                                               input_op_names=[input_op_name],
                                               output_op_names =[output_op.name])
        quant_params = QuantParams(use_cuda=False)

        np.random.seed(0)
        shape = conv_op.inputs[0].shape
        dataset = np.random.rand(1, shape[1], shape[2], shape[3])

        bias_tensor, weight = BiasCorrection._get_conv_linear_params(sess, conv_op)
        q_weight = BiasCorrection._get_quantized_weights(weight, quant_params)

        with unittest.mock.patch('aimet_tensorflow.bias_correction.iter_first_x') as iter_first_x:
            iter_first_x.return_value = [dataset]
            with unittest.mock.patch('aimet_tensorflow.bias_correction.BiasCorrection.analytical_bias_correction_per_layer',
                                     return_value=sess) as mocked_analytical_bias_correction_per_layer:
                with unittest.mock.patch('aimet_tensorflow.bias_correction.BiasCorrection.bias_correction_per_layer',
                                         return_value=sess) as mocked_bias_correction_per_layer:
                    updated_sess = BiasCorrection.correct_bias(sess, bias_corr_input, quant_params, dataset,
                                                               perform_only_empirical_bias_corr = False)

        # check if api(s) are invoked
        assert mocked_analytical_bias_correction_per_layer.called
        called_args = mocked_analytical_bias_correction_per_layer.call_args
        assert(called_args[1]['is_first_conv'] == True)
        assert mocked_bias_correction_per_layer.called

        sess.close()
        updated_sess.close()

    def test_analytical_empirical_bias_correction_single_layer(self):
        """
        Test bn based bias correction for a single layer api
        """
        # create a custom model
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")

        conv1_op = tf.keras.layers.Conv2D(32, (3, 3),
                                          kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                          bias_initializer='random_uniform')(inputs)

        bn_op = tf.keras.layers.BatchNormalization(fused=True, beta_initializer='random_uniform',
                                                   gamma_initializer='random_uniform',
                                                   moving_mean_initializer='random_uniform',
                                                   moving_variance_initializer='random_uniform')(conv1_op, training=False)

        conv2_op = tf.keras.layers.Conv2D(32, (3, 3))(bn_op)
        conv3_op = tf.keras.layers.Conv2D(32, (3, 3))(conv2_op)
        conv4_op = tf.keras.layers.Conv2D(32, (3, 3))(conv3_op)
        conv5_op = tf.keras.layers.Conv2D(32, (3, 3))(conv4_op)
        bn_op2 = tf.keras.layers.BatchNormalization(fused=True, beta_initializer='random_uniform',
                                                    gamma_initializer='random_uniform',
                                                    moving_mean_initializer='random_uniform',
                                                    moving_variance_initializer='random_uniform')(conv5_op, training=False)
        relu_1 = tf.nn.relu(bn_op2)

        conv6_op = tf.keras.layers.Conv2D(32, (3, 3))(relu_1)

        _ = tf.nn.relu(conv6_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        sess.run(init)

        output_op = sess.graph.get_operation_by_name('Relu_1')
        input_op_name = "inputs"

        bias_corr_input = BiasCorrectionParams(batch_size=1, num_quant_samples=10,
                                               num_bias_correct_samples=10,
                                               input_op_names=[input_op_name],
                                               output_op_names =[output_op.name])
        quant_params = QuantParams(use_cuda=False)

        np.random.seed(0)

        input_tensor = sess.graph.get_tensor_by_name('inputs:0')
        shape = input_tensor.shape
        dataset = np.random.rand(1, shape[1], shape[2], shape[3])

        with unittest.mock.patch('aimet_tensorflow.bias_correction.iter_first_x') as iter_first_x:
            iter_first_x.return_value = [dataset]
            with unittest.mock.patch('aimet_tensorflow.bias_correction.BiasCorrection.analytical_bias_correction_per_layer',
                                     return_value = sess) as mocked_analytical_bias_correction_per_layer:
                with unittest.mock.patch('aimet_tensorflow.bias_correction.BiasCorrection.bias_correction_per_layer',
                                         return_value = sess) as mocked_bias_correction_per_layer:
                    updated_sess = BiasCorrection.correct_bias(sess, bias_corr_input, quant_params, dataset,
                                                               perform_only_empirical_bias_corr=False)

        assert mocked_bias_correction_per_layer.called
        assert mocked_analytical_bias_correction_per_layer.called
        self.assertEqual(mocked_analytical_bias_correction_per_layer.call_count, 3)
        self.assertEqual(mocked_bias_correction_per_layer.call_count, 3)  # conv 3,4,5

        sess.close()
        updated_sess.close()

    def test_bias_correction_model_tf_with_no_bias(self):
        """
        Test bias correction for custom model
        """
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,))
        conv_op = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(inputs)
        relu_1 = tf.nn.relu(conv_op)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3),use_bias=False)(relu_1)
        relu_2= tf.nn.relu(conv2_op)
        conv3_op = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)(relu_2)
        _ = tf.nn.relu(conv3_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        # updating random bias and weight for one conv
        np.random.seed(0)
        conv_op = sess.graph.get_operation_by_name('conv2d/Conv2D')
        w_shape = WeightTensorUtils.get_tensor_shape(conv_op)
        w_shape = WeightTensorUtils.get_tensor_shape(conv_op)
        w_numpy_data = np.random.rand(w_shape[0],w_shape[1],w_shape[2],w_shape[3])

        # save and load the updated graph after high bias fold update
        n_sess = save_and_load_graph('./test_update', sess)
        conv_op = n_sess.graph.get_operation_by_name('conv2d/Conv2D')

        input_op_name = conv_op.inputs[0].op.name
        output_op = n_sess.graph.get_operation_by_name('Relu_2')

        input_op_names = [input_op_name]
        output_op_names = [output_op.name]

        batch_size = 1
        num_samples = 10

        np.random.seed(0)
        shape = conv_op.inputs[0].shape

        dataset = np.random.rand(10, 1, shape[1], shape[2], shape[3])
        dataset = tf.convert_to_tensor(dataset)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

        quant_params = QuantParams(quant_mode='tf', use_cuda=False)
        bias_correction_params = BiasCorrectionParams(batch_size=batch_size,
                                                      num_quant_samples=num_samples,
                                                      num_bias_correct_samples=num_samples,
                                                      input_op_names=input_op_names,
                                                      output_op_names=output_op_names)

        conv_op = sess.graph.get_operation_by_name('conv2d_1/Conv2D')
        assert(BiasUtils.is_bias_none(conv_op))
        new_sess = BiasCorrection.correct_bias(n_sess, bias_correction_params, quant_params, dataset,
                                               perform_only_empirical_bias_corr=False)

        conv_op = new_sess.graph.get_operation_by_name('conv2d_1/Conv2D')
        assert(not BiasUtils.is_bias_none(conv_op))

        sess.close()
        n_sess.close()
        new_sess.close()

    def test_analytical_empirical_bias_correction(self):
        """
        Test bn based bias correction hybrid with a user passed in dictionary of conv and bn after cle.
        """
        # create a custom model
        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(32, 32, 3,), name="inputs")
        conv_op = tf.keras.layers.Conv2D(32, (3, 3),
                                          kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                          bias_initializer='random_uniform')(inputs)
        conv1_op = tf.keras.layers.Conv2D(32, (3, 3),
                                          kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                          bias_initializer='random_uniform')(conv_op)
        bn_op = tf.keras.layers.BatchNormalization(fused=True, beta_initializer='random_uniform',
                                                   gamma_initializer='random_uniform',
                                                   moving_mean_initializer='random_uniform',
                                                   moving_variance_initializer='random_uniform')(conv1_op, training=False)
        conv2_op = tf.keras.layers.Conv2D(32, (3, 3),
                                          kernel_initializer=tf.random_uniform_initializer(-1, 1),
                                          bias_initializer='random_uniform')(bn_op)
        bn_op2 = tf.keras.layers.BatchNormalization(fused=True, beta_initializer='random_uniform',
                                                    gamma_initializer='random_uniform',
                                                    moving_mean_initializer='random_uniform',
                                                    moving_variance_initializer='random_uniform')(conv2_op, training=False)
        relu_1 = tf.nn.relu(bn_op2)
        conv6_op = tf.keras.layers.Conv2D(32, (3, 3))(relu_1)

        _ = tf.nn.relu(conv6_op)

        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        sess.run(init)

        output_op = sess.graph.get_operation_by_name('Relu_1')
        input_op_name = "inputs"

        bias_corr_input = BiasCorrectionParams(batch_size=1, num_quant_samples=10,
                                               num_bias_correct_samples=10,
                                               input_op_names=[input_op_name],
                                               output_op_names=[output_op.name])
        quant_params = QuantParams(use_cuda=False)

        np.random.seed(0)

        input_tensor = sess.graph.get_tensor_by_name('inputs:0')
        shape = input_tensor.shape
        dataset = np.random.rand(1, shape[1], shape[2], shape[3])

        # store conv bns info
        conv_bn_dict = BiasCorrection.find_all_convs_bn_with_activation(sess,
                                                                        [input_op_name],
                                                                        [output_op.name])

        # perform CLE
        new_sess = equalize_model(sess, input_op_name, output_op.name)
        conv_with_bn_op = new_sess.graph.get_operation_by_name('conv2d_1/Conv2D')
        old_bias_as_numpy = BiasUtils.get_bias_as_numpy_data(new_sess, conv_with_bn_op)

        # perform bias correction and check analytical is performed.
        with unittest.mock.patch('aimet_tensorflow.bias_correction.iter_first_x') as iter_first_x:
            iter_first_x.return_value = [dataset]
            with unittest.mock.patch('aimet_tensorflow.bias_correction.BiasCorrection.analytical_bias_correction_per_layer',
                                     return_value = sess) as mocked_analytical_bias_correction_per_layer:
                updated_sess = BiasCorrection.correct_bias(new_sess, bias_corr_input, quant_params,
                                                           dataset, conv_bn_dict=conv_bn_dict,
                                                           perform_only_empirical_bias_corr=False)

        self.assertEqual(mocked_analytical_bias_correction_per_layer.call_count, 3)

        sess.close()
        new_sess.close()
