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
""" This file contains unit tests for testing winnowing for tf models. """

# pylint: disable=too-many-lines
import unittest
import logging
import struct
from typing import List
import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from aimet_common.winnow.mask import Mask
from aimet_common.utils import AimetLogger
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.examples.test_models import keras_model, single_residual, concat_model, pad_model, \
    depthwise_conv2d_model, keras_model_functional, dropout_keras_model, dropout_slim_model, tf_slim_basic_model, \
    upsample_model, multiple_input_model, model_with_postprocessing_nodes, minimum_maximum_model, \
    model_with_upsample_already_present, model_with_multiple_downsamples, model_with_upsample2d, \
    model_with_leaky_relu, keras_model_functional_with_non_fused_batchnorms, model_to_test_downstream_masks,\
    single_residual_for_tf2, upsample_model_for_tf2, keras_model_functional_for_tf2,\
    keras_model_functional_with_non_fused_batchnorms_for_tf2
from aimet_tensorflow.winnow.mask_propagation_winnower import MaskPropagationWinnower
import aimet_tensorflow.winnow.winnow as winnow
from aimet_tensorflow.utils.graph_saver import save_and_load_graph

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Test, logging.DEBUG)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Winnow, logging.DEBUG)


class TestTfModuleReducer(unittest.TestCase):
    """ Unit test cases for testing TensorFlowWinnower's module reducer. """

    @pytest.mark.tf1
    def test_reducing_tf_slim_model(self):
        """ Test mask propagation on a conv module in tf slim model """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        x = tf.compat.v1.placeholder(tf.float32, [1, 32, 32, 3])
        _ = tf_slim_basic_model(x)
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        # check that update_ops list is not empty
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        self.assertEqual(4, len(update_ops))

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("Conv_1/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("Conv_2/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("Conv_3/Conv2D")
        input_channels_to_winnow = [2, 4, 6]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["Placeholder"]
        output_op_names = ['tf_slim_model/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        conv3_relu = new_sess.graph.get_operation_by_name('Conv_3/Relu')
        self.assertEqual(conv3_relu.inputs[0].op.name, 'reduced_Conv_3/BiasAdd')
        reduced_conv2_op = new_sess.graph.get_operation_by_name('reduced_Conv_2/Conv2D')
        self.assertEqual(reduced_conv2_op.inputs[0].name, 'reduced_BatchNorm_1/cond/Merge:0')
        self.assertEqual(reduced_conv2_op.inputs[0].shape.as_list()[-1], 13)
        reduced_conv1_op = new_sess.graph.get_operation_by_name('reduced_Conv_1/Conv2D')
        self.assertEqual(reduced_conv1_op.inputs[0].name, 'reduced_BatchNorm/FusedBatchNormV3:0')

        # Check first batch norm's first input is from reduced conv2d, and that its is_training attribute is True
        reduced_batch_norm = new_sess.graph.get_operation_by_name('reduced_BatchNorm/FusedBatchNormV3')
        self.assertTrue(reduced_batch_norm.inputs[0].op.name, 'reduced_Conv/BiasAdd')
        self.assertEqual(reduced_batch_norm.get_attr('is_training'), True)
        # Check second batch norm uses an is training placeholder
        reduced_batch_norm_1 = new_sess.graph.get_operation_by_name('reduced_BatchNorm_1/cond/'
                                                                    'FusedBatchNormV3')
        is_training_placeholder = new_sess.graph.get_tensor_by_name('is_training:0')
        self.assertEqual(reduced_batch_norm_1.inputs[0].op.type, 'Switch')
        self.assertEqual(reduced_batch_norm_1.inputs[0].op.inputs[1].op.inputs[0], is_training_placeholder)
        # Check third batch norm's first input is from reduced scope 1 conv2d, and that its is_training_attribute is
        # False
        reduced_batch_norm_2 = new_sess.graph.get_operation_by_name('reduced_BatchNorm_2/FusedBatchNormV3')

        # Check that old and new epsilon and momentum values match
        orig_batch_norm = new_sess.graph.get_operation_by_name('BatchNorm/FusedBatchNormV3')
        new_batch_norm = new_sess.graph.get_operation_by_name('reduced_BatchNorm/FusedBatchNormV3')
        self.assertEqual(orig_batch_norm.get_attr('epsilon'), new_batch_norm.get_attr('epsilon'))
        orig_batch_norm_1 = new_sess.graph.get_operation_by_name('BatchNorm_1/cond/FusedBatchNormV3_1')
        new_batch_norm_1 = new_sess.graph.get_operation_by_name('reduced_BatchNorm_1/cond/FusedBatchNormV3_1')
        self.assertEqual(orig_batch_norm_1.get_attr('epsilon'), new_batch_norm_1.get_attr('epsilon'))
        orig_batch_norm_2 = new_sess.graph.get_operation_by_name('BatchNorm_2/FusedBatchNormV3')
        new_batch_norm_2 = new_sess.graph.get_operation_by_name('reduced_BatchNorm_2/FusedBatchNormV3')
        self.assertEqual(orig_batch_norm_2.get_attr('epsilon'), new_batch_norm_2.get_attr('epsilon'))

        orig_batch_norm_momentum = new_sess.graph.get_operation_by_name('BatchNorm/Const_3')
        new_batch_norm_momentum = new_sess.graph.get_operation_by_name('reduced_BatchNorm/Const_2')
        self.assertEqual(orig_batch_norm_momentum.get_attr('value').float_val[0],
                         new_batch_norm_momentum.get_attr('value').float_val[0])
        orig_batch_norm_1_momentum = new_sess.graph.get_operation_by_name('BatchNorm_1/cond_1/Const')
        new_batch_norm_1_momentum = new_sess.graph.get_operation_by_name('reduced_BatchNorm_1/cond_1/Const')
        self.assertEqual(orig_batch_norm_1_momentum.get_attr('value').float_val[0],
                         new_batch_norm_1_momentum.get_attr('value').float_val[0])

        self.assertTrue(reduced_batch_norm_2.inputs[0].op.name, 'reduced_Conv_2/Relu')
        self.assertEqual(reduced_batch_norm_2.get_attr('is_training'), False)

        self.assertEqual(10, len(ordered_modules_list))

        # check that update_ops list is empty
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        self.assertEqual(0, len(update_ops))

        # check that update_ops is still empty after saving and reloading graph
        with new_sess.graph.as_default():
            init = tf.compat.v1.global_variables_initializer()
            new_sess.run(init)
        new_sess_2 = save_and_load_graph('.', new_sess)
        with new_sess_2.graph.as_default():
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        self.assertEqual(0, len(update_ops))

        new_sess_2.close()
        new_sess.close()
        sess.close()

    @pytest.mark.tf1
    def test_reducing_with_downsample(self):
        """ Test reducing a single_residual model with downsampling layers """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = single_residual()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["input_1"]
        output_op_names = ['Relu_2']

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_2/Conv2D")
        input_channels_to_winnow_2 = [7, 12, 13, 14]
        module_mask_pair = (tf_op, input_channels_to_winnow_2)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        with new_sess.graph.as_default():
            reduced_conv2d_1_input = new_sess.graph.get_operation_by_name("reduced_conv2d_1/Conv2D").inputs[0]
            reduced_conv2d_2_input = new_sess.graph.get_operation_by_name("reduced_conv2d_2/Conv2D").inputs[0]
            reduced_relu_output = new_sess.graph.get_tensor_by_name("reduced_Relu:0")
            self.assertTrue("GatherV2" in reduced_conv2d_1_input.name)
            self.assertTrue("GatherV2" in reduced_conv2d_2_input.name)
            self.assertEqual(reduced_conv2d_1_input.shape.as_list()[-1], 13)
            self.assertEqual(reduced_conv2d_2_input.shape.as_list()[-1], 12)
            self.assertEqual(reduced_relu_output.shape.as_list()[-1], 15)
        self.assertEqual(6, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    @pytest.mark.tf2
    def test_reducing_with_downsample_for_tf2(self):
        """ Test reducing a single_residual model with downsampling layers """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = single_residual_for_tf2()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["input_1"]
        output_op_names = ['Relu_2']

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_2/Conv2D")
        input_channels_to_winnow_2 = [7, 12, 13, 14]
        module_mask_pair = (tf_op, input_channels_to_winnow_2)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        with new_sess.graph.as_default():
            reduced_conv2d_1_input = new_sess.graph.get_operation_by_name("reduced_conv2d_1/Conv2D").inputs[0]
            reduced_conv2d_2_input = new_sess.graph.get_operation_by_name("reduced_conv2d_2/Conv2D").inputs[0]
            reduced_relu_output = new_sess.graph.get_tensor_by_name("reduced_Relu:0")
            self.assertTrue("GatherV2" in reduced_conv2d_1_input.name)
            self.assertTrue("GatherV2" in reduced_conv2d_2_input.name)
            self.assertEqual(reduced_conv2d_1_input.shape.as_list()[-1], 13)
            self.assertEqual(reduced_conv2d_2_input.shape.as_list()[-1], 12)
            self.assertEqual(reduced_relu_output.shape.as_list()[-1], 15)
        self.assertEqual(6, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    @pytest.mark.tf1
    def test_reducing_inserting_downsample_upsample(self):
        """ Test reducing a single_residual model with inserting downsampling and upsampling layers """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()

        _ = upsample_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["input_1"]
        output_op_names = ['upsample_model/Softmax']

        module_zero_channels_list = []
        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_3/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        stack_output = new_sess.graph.get_tensor_by_name("upsample/stack:0")
        reduced_batch_normalization_1_output = new_sess.graph.get_tensor_by_name("reduced_batch_normalization_1/"
                                                                                 "cond/Merge:0")
        relu_1_output = new_sess.graph.get_tensor_by_name("Relu_1:0")
        gather_output = new_sess.graph.get_tensor_by_name("downsample/GatherV2:0")
        self.assertEqual([None, 7, 7, 5], reduced_batch_normalization_1_output.shape.as_list())
        self.assertEqual([None, 7, 7, 8], stack_output.shape.as_list())
        self.assertEqual([None, 7, 7, 8], relu_1_output.shape.as_list())
        self.assertEqual([None, 7, 7, 5], gather_output.shape.as_list())
        self.assertEqual(3, len(ordered_modules_list))

        # Test winnowing the graph again with the upsample and downsample in the graph
        module_zero_channels_list = []
        tf_op = new_sess.graph.get_operation_by_name("reduced_conv2d_3/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        new_sess_2, ordered_modules_list = winnow.winnow_tf_model(new_sess, input_op_names, output_op_names,
                                                                  module_zero_channels_list,
                                                                  reshape=True, in_place=True, verbose=True)

        # Since downsample is null connectivity, relu output should remain 8 even while downsample output is winnowed
        # further.
        relu_1_output = new_sess_2.graph.get_tensor_by_name("Relu_1:0")
        gather_output = new_sess_2.graph.get_tensor_by_name("downsample_1/GatherV2:0")
        self.assertEqual([None, 7, 7, 8], relu_1_output.shape.as_list())
        self.assertEqual([None, 7, 7, 2], gather_output.shape.as_list())
        self.assertEqual(1, len(ordered_modules_list))

        new_sess.close()
        new_sess_2.close()
        sess.close()

    @pytest.mark.tf2
    def test_reducing_inserting_downsample_upsample_for_tf2(self):
        """ Test reducing a single_residual model with inserting downsampling and upsampling layers """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()

        _ = upsample_model_for_tf2()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["input_1"]
        output_op_names = ['upsample_model/Softmax']

        module_zero_channels_list = []
        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_3/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        all_ops = new_sess.graph.get_operations()
        for op in all_ops:
            print(op.name)

        stack_output = new_sess.graph.get_tensor_by_name("upsample/stack:0")
        reduced_batch_normalization_1_output = new_sess.graph.get_tensor_by_name("reduced_batch_normalization_1/FusedBatchNormV3:0")
        relu_1_output = new_sess.graph.get_tensor_by_name("Relu_1:0")
        gather_output = new_sess.graph.get_tensor_by_name("downsample/GatherV2:0")
        self.assertEqual([None, 7, 7, 5], reduced_batch_normalization_1_output.shape.as_list())
        self.assertEqual([None, 7, 7, 8], stack_output.shape.as_list())
        self.assertEqual([None, 7, 7, 8], relu_1_output.shape.as_list())
        self.assertEqual([None, 7, 7, 5], gather_output.shape.as_list())
        self.assertEqual(3, len(ordered_modules_list))

        # Test winnowing the graph again with the upsample and downsample in the graph
        module_zero_channels_list = []
        tf_op = new_sess.graph.get_operation_by_name("reduced_conv2d_3/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        new_sess_2, ordered_modules_list = winnow.winnow_tf_model(new_sess, input_op_names, output_op_names,
                                                                  module_zero_channels_list,
                                                                  reshape=True, in_place=True, verbose=True)

        # Since downsample is null connectivity, relu output should remain 8 even while downsample output is winnowed
        # further.
        relu_1_output = new_sess_2.graph.get_tensor_by_name("Relu_1:0")
        gather_output = new_sess_2.graph.get_tensor_by_name("downsample_1/GatherV2:0")
        self.assertEqual([None, 7, 7, 8], relu_1_output.shape.as_list())
        self.assertEqual([None, 7, 7, 2], gather_output.shape.as_list())
        self.assertEqual(1, len(ordered_modules_list))

        new_sess.close()
        new_sess_2.close()
        sess.close()

    def test_reducing_with_concat(self):
        """ Test reducing a model with concat """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = concat_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["input_1"]
        output_op_names = ['concat_model/Softmax']

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_3/Conv2D")
        input_channels_to_winnow = [2, 3, 6, 7, 17]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_4/Conv2D")
        input_channels_to_winnow_1 = [2, 3, 6, 7, 8, 17]
        module_mask_pair = (tf_op, input_channels_to_winnow_1)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        with new_sess.graph.as_default():
            conv2d_3_input = new_sess.graph.get_operation_by_name("reduced_conv2d_3/Conv2D").inputs[0]
            conv2d_4_input = new_sess.graph.get_operation_by_name("reduced_conv2d_4/Conv2D").inputs[0]
            concat_output = new_sess.graph.get_tensor_by_name("concatenate/concat:0")
            concat_conv2d_1_input = new_sess.graph.get_operation_by_name("concatenate/concat").inputs[0]
            concat_conv2d_input = new_sess.graph.get_operation_by_name("concatenate/concat").inputs[1]
            concat_conv2d_2_input = new_sess.graph.get_operation_by_name("concatenate/concat").inputs[2]
            conv2d_1_output = new_sess.graph.get_tensor_by_name("reduced_conv2d_1/BiasAdd:0")
            conv2d_output = new_sess.graph.get_tensor_by_name("reduced_conv2d/BiasAdd:0")
            conv2d_2_output = new_sess.graph.get_tensor_by_name("reduced_conv2d_2/BiasAdd:0")

            # Validate tensor channel sizes are as expected
            self.assertEqual(13, conv2d_3_input.shape.as_list()[-1])
            self.assertEqual(12, conv2d_4_input.shape.as_list()[-1])
            self.assertEqual(18, concat_output.shape.as_list()[-1])
            self.assertEqual(6, concat_conv2d_1_input.shape.as_list()[-1])
            self.assertEqual(5, concat_conv2d_input.shape.as_list()[-1])
            self.assertEqual(7, concat_conv2d_2_input.shape.as_list()[-1])
            self.assertEqual(4, conv2d_1_output.shape.as_list()[-1])
            self.assertEqual(3, conv2d_output.shape.as_list()[-1])
            self.assertEqual(6, conv2d_2_output.shape.as_list()[-1])
        self.assertEqual(5, len(ordered_modules_list))

        # Test that original versions of winnowed ops have been detached from main graph
        new_conn_graph = ConnectedGraph(new_sess.graph, input_op_names, output_op_names)
        self.assertEqual(27, len(new_conn_graph.get_all_ops().keys()))
        self.assertTrue(new_conn_graph.get_op_from_module_name('conv2d_3/Conv2D') is None)
        self.assertTrue(new_conn_graph.get_op_from_module_name('conv2d_4/Conv2D') is None)

        new_sess.close()
        sess.close()

    def test_reducing_pad_in_module_reducer(self):
        """ Test calling module reducer reduce modules for conv op """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = pad_model()
        # _ = tf.compat.v1.summary.FileWriter('./pad_model', tf.compat.v1.get_default_graph())
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        input_op_names = ["input_1"]
        output_op_names = ['pad_model/Softmax']
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)
        self.assertEqual(3, len(ordered_modules_list))
        new_sess.close()
        sess.close()

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = pad_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_2/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        with self.assertRaises(NotImplementedError):
            _, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                             module_zero_channels_list,
                                                             reshape=True, in_place=True, verbose=True)
        sess.close()

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = pad_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_3/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)
        reduced_padv2 = new_sess.graph.get_operation_by_name("reduced_PadV2")
        self.assertTrue(len(reduced_padv2.inputs) > 1)
        orig_const_val = sess.graph.get_operation_by_name("PadV2").inputs[2].eval(session=sess)
        new_const_val = new_sess.graph.get_operation_by_name("reduced_PadV2").inputs[2].eval(session=new_sess)
        self.assertEqual(orig_const_val, new_const_val)
        self.assertEqual(3, len(ordered_modules_list))
        new_sess.close()
        sess.close()

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = pad_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_4/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)
        old_mode = sess.graph.get_operation_by_name("MirrorPad").get_attr('mode')
        new_mode = sess.graph.get_operation_by_name("reduced_MirrorPad").get_attr('mode')
        self.assertEqual(old_mode, new_mode)
        self.assertEqual(3, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    @unittest.skip("For some reason, with TF 1.15, regularization does not show up in the convolution op")
    def test_reducing_conv_with_l2_loss(self):
        """ Test that reducing conv ops with l2 regularization is able to keep the regularization parameter """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = keras_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        input_op_names = ["conv2d_input"]
        output_op_names = ['keras_model/Softmax']
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, _ = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                             module_zero_channels_list, reshape=True, in_place=True,
                                             verbose=True)

        scale_op = new_sess.graph.get_operation_by_name("reduced_conv2d_1/kernel/"
                                                        "Regularizer/l2_regularizer/scale")
        self.assertEqual(.5, scale_op.get_attr('value').float_val[0])
        new_sess.close()
        sess.close()

    def test_reducing_depthwise_conv2d(self):
        """ Test reducing depthwise conv2d """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = depthwise_conv2d_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("separable_conv2d/separable_conv2d")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [0, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["input_1"]
        output_op_names = ['depthwise_conv2d_model/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)
        reduced_depthwise = new_sess.graph.get_operation_by_name("reduced_depthwise_conv2d/depthwise")
        reduced_separable_depthwise = new_sess.graph.get_operation_by_name("reduced_separable_conv2d/separable_conv2d/"
                                                                           "depthwise")
        self.assertEqual(7, reduced_depthwise.outputs[0].shape.as_list()[-1])
        self.assertEqual(7, reduced_depthwise.inputs[0].shape.as_list()[-1])
        self.assertEqual(13, reduced_separable_depthwise.outputs[0].shape.as_list()[-1])
        self.assertEqual(13, reduced_separable_depthwise.inputs[0].shape.as_list()[-1])
        self.assertEqual(5, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    @pytest.mark.tf1
    def test_reducing_with_dropout_and_identity_keras(self):
        """ Test reducing a keras model with dropout and identity modules """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = dropout_keras_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["input_1"]
        output_op_names = ['dropout_keras_model/Softmax']

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [2, 3, 4]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        reduced_identity = new_sess.graph.get_tensor_by_name("reduced_Identity:0")
        self.assertEqual(13, reduced_identity.shape.as_list()[-1])
        self.assertEqual(reduced_identity.op.inputs[0].name, 'reduced_dropout/cond/Merge:0')
        old_dropout_greater_equal_op = new_sess.graph.get_operation_by_name('dropout/cond/dropout/GreaterEqual')
        reduced_dropout_greater_equal_op = new_sess.graph.get_operation_by_name('reduced_dropout/cond/dropout/'
                                                                                'GreaterEqual')
        old_rate = old_dropout_greater_equal_op.inputs[1].op.get_attr('value').float_val[0]
        rate = reduced_dropout_greater_equal_op.inputs[1].op.get_attr('value').float_val[0]
        self.assertTrue(np.allclose(old_rate, rate))
        self.assertEqual(4, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    @pytest.mark.tf1
    def test_reducing_with_dropout_and_identity_slim(self):
        """ Test reducing a keras model with dropout and identity modules """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = dropout_slim_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["input_1"]
        output_op_names = ['dropout_slim_model/Softmax']

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("Conv_1/Conv2D")
        input_channels_to_winnow = [2, 3, 4]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        reduced_identity = new_sess.graph.get_tensor_by_name("reduced_Identity:0")
        self.assertEqual(13, reduced_identity.shape.as_list()[-1])
        self.assertEqual(reduced_identity.op.inputs[0].name, 'reduced_Dropout/dropout/mul_1:0')
        old_dropout_greater_equal_op = new_sess.graph.get_operation_by_name('Dropout/dropout_1/GreaterEqual')
        reduced_dropout_greater_equal_op = new_sess.graph.get_operation_by_name('reduced_Dropout/dropout/'
                                                                                'GreaterEqual')
        old_rate = old_dropout_greater_equal_op.inputs[1].op.get_attr('value').float_val[0]
        rate = reduced_dropout_greater_equal_op.inputs[1].op.get_attr('value').float_val[0]
        self.assertTrue(np.allclose(old_rate, rate))

        self.assertEqual(5, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    @pytest.mark.tf1
    def test_reducing_keras_fused_bn_training_true_and_false(self):
        """ Test for reducing keras type fused bn ops, both for training true and false """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = keras_model_functional()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("scope_1/conv2d_2/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("scope_1/conv2d_1/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("scope_1/conv2d_3/Conv2D")
        input_channels_to_winnow = [2, 4, 6]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["input_1"]
        output_op_names = ['keras_model_functional/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        reduced_conv2d_2_tanh_op = new_sess.graph.get_operation_by_name('reduced_scope_1/conv2d_2/Tanh')
        self.assertEqual(reduced_conv2d_2_tanh_op.inputs[0].op.name, 'reduced_scope_1/conv2d_2/BiasAdd')
        reduced_conv2d_2_op = new_sess.graph.get_operation_by_name('reduced_scope_1/conv2d_2/Conv2D')
        self.assertEqual(reduced_conv2d_2_op.inputs[0].name, 'reduced_scope_1/batch_normalization_1/cond/Merge:0')
        self.assertEqual(reduced_conv2d_2_op.inputs[0].shape.as_list()[-1], 13)
        reduced_conv2d_1_op = new_sess.graph.get_operation_by_name('reduced_scope_1/conv2d_1/Conv2D')
        self.assertEqual(reduced_conv2d_1_op.inputs[0].name, 'reduced_batch_normalization/FusedBatchNormV3:0')

        # Check first batch norm's first input is from reduced conv2d, and that its is_training attribute is True
        reduced_batch_norm = new_sess.graph.get_operation_by_name('reduced_batch_normalization/FusedBatchNormV3')
        self.assertTrue(reduced_batch_norm.inputs[0].op.name, 'reduced_conv2d/BiasAdd')
        self.assertEqual(reduced_batch_norm.get_attr('is_training'), True)
        # Check second batch norm uses an is training placeholder
        reduced_batch_norm_1 = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_1/cond/'
                                                                    'FusedBatchNormV3')
        is_training_placeholder = new_sess.graph.get_tensor_by_name('is_training:0')
        self.assertEqual(reduced_batch_norm_1.inputs[0].op.type, 'Switch')
        self.assertEqual(reduced_batch_norm_1.inputs[0].op.inputs[1].op.inputs[0], is_training_placeholder)
        # Check third batch norm's first input is from reduced scope 1 conv2d, and that its is_training_attribute is
        # False
        reduced_batch_norm_2 = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_2/'
                                                                    'FusedBatchNormV3')
        self.assertTrue(reduced_batch_norm_2.inputs[0].op.name, 'reduced_scope_1/conv2d_2/Tanh')
        self.assertEqual(reduced_batch_norm_2.get_attr('is_training'), False)

        # Check that old and new epsilon and momentum values match
        orig_batch_norm = new_sess.graph.get_operation_by_name('batch_normalization/FusedBatchNormV3')
        new_batch_norm = new_sess.graph.get_operation_by_name('reduced_batch_normalization/FusedBatchNormV3')
        self.assertEqual(orig_batch_norm.get_attr('epsilon'), new_batch_norm.get_attr('epsilon'))
        orig_batch_norm_1 = new_sess.graph.get_operation_by_name('scope_1/batch_normalization_1/cond/'
                                                                 'FusedBatchNormV3_1')
        new_batch_norm_1 = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_1/'
                                                                'cond/FusedBatchNormV3_1')
        self.assertEqual(orig_batch_norm_1.get_attr('epsilon'), new_batch_norm_1.get_attr('epsilon'))
        orig_batch_norm_2 = new_sess.graph.get_operation_by_name('scope_1/batch_normalization_2/FusedBatchNormV3')
        new_batch_norm_2 = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_2/'
                                                                'FusedBatchNormV3')
        self.assertEqual(orig_batch_norm_2.get_attr('epsilon'), new_batch_norm_2.get_attr('epsilon'))

        orig_batch_norm_momentum = new_sess.graph.get_operation_by_name('batch_normalization/Const_2')
        new_batch_norm_momentum = new_sess.graph.get_operation_by_name('reduced_batch_normalization/Const_2')
        self.assertEqual(orig_batch_norm_momentum.get_attr('value').float_val[0],
                         new_batch_norm_momentum.get_attr('value').float_val[0])
        orig_batch_norm_1_momentum = new_sess.graph.get_operation_by_name('scope_1/batch_normalization_1/cond_1/Const')
        new_batch_norm_1_momentum = new_sess.graph.get_operation_by_name('reduced_scope_1/'
                                                                         'batch_normalization_1/cond_1/Const')
        self.assertEqual(orig_batch_norm_1_momentum.get_attr('value').float_val[0],
                         new_batch_norm_1_momentum.get_attr('value').float_val[0])

        self.assertEqual(9, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    @pytest.mark.tf2
    def test_reducing_keras_fused_bn_training_true_and_false_for_tf2(self):
        """ Test for reducing keras type fused bn ops, both for training true and false """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = keras_model_functional_for_tf2()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("scope_1/conv2d_2/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("scope_1/conv2d_1/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("scope_1/conv2d_3/Conv2D")
        input_channels_to_winnow = [2, 4, 6]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["input_1"]
        output_op_names = ['keras_model_functional/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        reduced_conv2d_2_tanh_op = new_sess.graph.get_operation_by_name('reduced_scope_1/conv2d_2/Tanh')
        self.assertEqual(reduced_conv2d_2_tanh_op.inputs[0].op.name, 'reduced_scope_1/conv2d_2/BiasAdd')
        reduced_conv2d_2_op = new_sess.graph.get_operation_by_name('reduced_scope_1/conv2d_2/Conv2D')
        self.assertEqual(reduced_conv2d_2_op.inputs[0].name, 'reduced_scope_1/batch_normalization_1/FusedBatchNormV3:0')
        self.assertEqual(reduced_conv2d_2_op.inputs[0].shape.as_list()[-1], 13)
        reduced_conv2d_1_op = new_sess.graph.get_operation_by_name('reduced_scope_1/conv2d_1/Conv2D')
        self.assertEqual(reduced_conv2d_1_op.inputs[0].name, 'reduced_batch_normalization/FusedBatchNormV3:0')

        # Check first batch norm's first input is from reduced conv2d, and that its is_training attribute is True
        reduced_batch_norm = new_sess.graph.get_operation_by_name('reduced_batch_normalization/FusedBatchNormV3')
        self.assertTrue(reduced_batch_norm.inputs[0].op.name, 'reduced_conv2d/BiasAdd')
        self.assertEqual(reduced_batch_norm.get_attr('is_training'), True)
        # Check second batch norm uses an is training placeholder
        reduced_batch_norm_1 = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_1/FusedBatchNormV3')
        self.assertEqual(reduced_batch_norm_1.get_attr('is_training'), False)
        # Check third batch norm's first input is from reduced scope 1 conv2d, and that its is_training_attribute is
        # False
        reduced_batch_norm_2 = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_2/'
                                                                    'FusedBatchNormV3')
        self.assertTrue(reduced_batch_norm_2.inputs[0].op.name, 'reduced_scope_1/conv2d_2/Tanh')
        self.assertEqual(reduced_batch_norm_2.get_attr('is_training'), False)

        # Check that old and new epsilon and momentum values match
        orig_batch_norm = new_sess.graph.get_operation_by_name('batch_normalization/FusedBatchNormV3')
        new_batch_norm = new_sess.graph.get_operation_by_name('reduced_batch_normalization/FusedBatchNormV3')
        self.assertEqual(orig_batch_norm.get_attr('epsilon'), new_batch_norm.get_attr('epsilon'))
        orig_batch_norm_1 = new_sess.graph.get_operation_by_name('scope_1/batch_normalization_1/FusedBatchNormV3')
        new_batch_norm_1 = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_1/FusedBatchNormV3')
        self.assertEqual(orig_batch_norm_1.get_attr('epsilon'), new_batch_norm_1.get_attr('epsilon'))
        orig_batch_norm_2 = new_sess.graph.get_operation_by_name('scope_1/batch_normalization_2/FusedBatchNormV3')
        new_batch_norm_2 = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_2/'
                                                                'FusedBatchNormV3')
        self.assertEqual(orig_batch_norm_2.get_attr('epsilon'), new_batch_norm_2.get_attr('epsilon'))

        orig_batch_norm_momentum = new_sess.graph.get_operation_by_name('batch_normalization/Const')
        new_batch_norm_momentum = new_sess.graph.get_operation_by_name('reduced_batch_normalization/Const')
        self.assertEqual(orig_batch_norm_momentum.get_attr('value').float_val[0],
                         new_batch_norm_momentum.get_attr('value').float_val[0])

        self.assertEqual(9, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    @pytest.mark.tf1
    def test_reducing_keras_non_fused_bn_training_true_and_false(self):
        """ Test for reducing keras type non fused bn ops, both for training true and false """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = keras_model_functional_with_non_fused_batchnorms()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("scope_1/conv2d_2/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("scope_1/conv2d_1/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("scope_1/conv2d_3/Conv2D")
        input_channels_to_winnow = [2, 4, 6]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["input_1"]
        output_op_names = ['keras_model_functional_with_non_fused_batchnorms/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        reduced_conv2d_1_tanh_op = new_sess.graph.get_operation_by_name('reduced_scope_1/conv2d_2/Tanh')
        self.assertEqual(reduced_conv2d_1_tanh_op.inputs[0].op.name, 'reduced_scope_1/conv2d_2/BiasAdd')
        reduced_conv2d_1_op = new_sess.graph.get_operation_by_name('reduced_scope_1/conv2d_2/Conv2D')
        self.assertEqual(reduced_conv2d_1_op.inputs[0].name, 'reduced_scope_1/batch_normalization_1/batchnorm/add_1:0')
        self.assertEqual(reduced_conv2d_1_op.inputs[0].shape.as_list()[-1], 13)
        reduced_conv2d_op = new_sess.graph.get_operation_by_name('reduced_scope_1/conv2d_1/Conv2D')
        self.assertEqual(reduced_conv2d_op.inputs[0].name, 'reduced_batch_normalization/batchnorm/add_1:0')

        # Check that old and new epsilon and momentum values match
        orig_batch_norm_epsilon = new_sess.graph.get_operation_by_name('batch_normalization/batchnorm/add/y')
        new_batch_norm_epsilon = new_sess.graph.get_operation_by_name('reduced_batch_normalization/batchnorm/add/y')
        self.assertEqual(orig_batch_norm_epsilon.get_attr('value').float_val[0],
                         new_batch_norm_epsilon.get_attr('value').float_val[0])
        orig_batch_norm_1_epsilon = new_sess.graph.get_operation_by_name('scope_1/batch_normalization_1/batchnorm/add/'
                                                                         'y')
        new_batch_norm_1_epsilon = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_1/'
                                                                        'batchnorm/add/y')
        self.assertEqual(orig_batch_norm_1_epsilon.get_attr('value').float_val[0],
                         new_batch_norm_1_epsilon.get_attr('value').float_val[0])
        orig_batch_norm_2_epsilon = new_sess.graph.get_operation_by_name('scope_1/batch_normalization_2/'
                                                                         'batchnorm/add/y')
        new_batch_norm_2_epsilon = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_2/'
                                                                        'batchnorm/add/y')
        self.assertEqual(orig_batch_norm_2_epsilon.get_attr('value').float_val[0],
                         new_batch_norm_2_epsilon.get_attr('value').float_val[0])

        orig_batch_norm_momentum = new_sess.graph.get_operation_by_name('batch_normalization/AssignMovingAvg_1/decay')
        new_batch_norm_momentum = new_sess.graph.get_operation_by_name('reduced_batch_normalization/'
                                                                       'AssignMovingAvg_1/decay')
        self.assertEqual(orig_batch_norm_momentum.get_attr('value').float_val[0],
                         new_batch_norm_momentum.get_attr('value').float_val[0])
        orig_batch_norm_1_momentum = new_sess.graph.get_operation_by_name('scope_1/batch_normalization_1/cond_3/'
                                                                          'AssignMovingAvg/decay')
        new_batch_norm_1_momentum = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_1/cond_3/'
                                                                         'AssignMovingAvg/decay')
        self.assertEqual(orig_batch_norm_1_momentum.get_attr('value').float_val[0],
                         new_batch_norm_1_momentum.get_attr('value').float_val[0])

        self.assertEqual(9, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    @pytest.mark.tf2
    def test_reducing_keras_non_fused_bn_training_true_and_false_for_tf2(self):
        """ Test for reducing keras type non fused bn ops, both for training true and false """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = keras_model_functional_with_non_fused_batchnorms_for_tf2()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("scope_1/conv2d_2/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("scope_1/conv2d_1/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("scope_1/conv2d_3/Conv2D")
        input_channels_to_winnow = [2, 4, 6]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["input_1"]
        output_op_names = ['keras_model_functional_with_non_fused_batchnorms/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        reduced_conv2d_1_tanh_op = new_sess.graph.get_operation_by_name('reduced_scope_1/conv2d_2/Tanh')
        self.assertEqual(reduced_conv2d_1_tanh_op.inputs[0].op.name, 'reduced_scope_1/conv2d_2/BiasAdd')
        reduced_conv2d_1_op = new_sess.graph.get_operation_by_name('reduced_scope_1/conv2d_2/Conv2D')
        self.assertEqual(reduced_conv2d_1_op.inputs[0].name, 'reduced_scope_1/batch_normalization_1/batchnorm/add_1:0')
        self.assertEqual(reduced_conv2d_1_op.inputs[0].shape.as_list()[-1], 13)
        reduced_conv2d_op = new_sess.graph.get_operation_by_name('reduced_scope_1/conv2d_1/Conv2D')
        self.assertEqual(reduced_conv2d_op.inputs[0].name, 'reduced_batch_normalization/batchnorm/add_1:0')

        # Check that old and new epsilon and momentum values match
        orig_batch_norm_epsilon = new_sess.graph.get_operation_by_name('batch_normalization/batchnorm/add/y')
        new_batch_norm_epsilon = new_sess.graph.get_operation_by_name('reduced_batch_normalization/batchnorm/add/y')
        self.assertEqual(orig_batch_norm_epsilon.get_attr('value').float_val[0],
                         new_batch_norm_epsilon.get_attr('value').float_val[0])
        orig_batch_norm_1_epsilon = new_sess.graph.get_operation_by_name('scope_1/batch_normalization_1/batchnorm/add/'
                                                                         'y')
        new_batch_norm_1_epsilon = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_1/'
                                                                        'batchnorm/add/y')
        self.assertEqual(orig_batch_norm_1_epsilon.get_attr('value').float_val[0],
                         new_batch_norm_1_epsilon.get_attr('value').float_val[0])
        orig_batch_norm_2_epsilon = new_sess.graph.get_operation_by_name('scope_1/batch_normalization_2/'
                                                                         'batchnorm/add/y')
        new_batch_norm_2_epsilon = new_sess.graph.get_operation_by_name('reduced_scope_1/batch_normalization_2/'
                                                                        'batchnorm/add/y')
        self.assertEqual(orig_batch_norm_2_epsilon.get_attr('value').float_val[0],
                         new_batch_norm_2_epsilon.get_attr('value').float_val[0])

        orig_batch_norm_momentum = new_sess.graph.get_operation_by_name('batch_normalization/AssignMovingAvg_1/decay')
        new_batch_norm_momentum = new_sess.graph.get_operation_by_name('reduced_batch_normalization/'
                                                                       'AssignMovingAvg_1/decay')
        self.assertEqual(orig_batch_norm_momentum.get_attr('value').float_val[0],
                         new_batch_norm_momentum.get_attr('value').float_val[0])

        self.assertEqual(9, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    def test_reducing_multiple_input_model(self):
        """ Test for reducing a model with multiple inputs"""

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = multiple_input_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["input1", "input2"]
        output_op_names = ['multiple_input_model/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        reduced_conv2d_output = new_sess.graph.get_tensor_by_name("reduced_conv1a/BiasAdd:0")
        self.assertEqual(5, reduced_conv2d_output.shape.as_list()[-1])
        reduced_conv2d_1_output = new_sess.graph.get_tensor_by_name("reduced_conv1b/BiasAdd:0")
        self.assertEqual(5, reduced_conv2d_1_output.shape.as_list()[-1])
        self.assertEqual(4, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    def test_reducing_minimum_maximum_ops(self):
        """ Test for reducing a model with minimum and maximum ops """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = minimum_maximum_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ['input_1']
        output_op_names = ['minimum_maximum_model/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        old_minimum_op = sess.graph.get_operation_by_name('Minimum')
        old_minimum_rate = old_minimum_op.inputs[1].op.get_attr('value').float_val[0]
        reduced_minimum_tensor = new_sess.graph.get_tensor_by_name("reduced_Minimum:0")
        new_minimum_rate = reduced_minimum_tensor.op.inputs[1].op.get_attr('value').float_val[0]
        self.assertEqual(29, reduced_minimum_tensor.shape.as_list()[-1])
        self.assertEqual(old_minimum_rate, new_minimum_rate)

        old_maximum_op = sess.graph.get_operation_by_name('Maximum')
        old_maximum_rate = old_maximum_op.inputs[1].op.get_attr('value').float_val[0]
        reduced_maximum_tensor = new_sess.graph.get_tensor_by_name("reduced_Maximum:0")
        new_maximum_rate = reduced_maximum_tensor.op.inputs[1].op.get_attr('value').float_val[0]
        self.assertEqual(29, reduced_maximum_tensor.shape.as_list()[-1])
        self.assertEqual(old_maximum_rate, new_maximum_rate)

        self.assertEqual(5, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    def test_reducing_upsample(self):
        """ Test for reducing a model with upsample """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = model_with_upsample_already_present()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ['input_1']
        output_op_names = ['model_with_upsample_already_present/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        # Check that upsample input and output shapes remain the same
        # Check that downsample was inserted after upsample
        stack_tensor = new_sess.graph.get_tensor_by_name('upsample/stack:0')
        downsample_tensor = new_sess.graph.get_tensor_by_name('downsample/GatherV2:0')
        downsample_indices = downsample_tensor.op.inputs[1].op.get_attr('value').tensor_content
        downsample_indices = struct.unpack('9i', downsample_indices)
        conv2d_tensor = new_sess.graph.get_tensor_by_name('conv2d/Conv2D:0')
        self.assertEqual((0, 4, 5, 6, 7, 8, 9, 10, 11), downsample_indices)
        self.assertEqual(12, stack_tensor.shape.as_list()[-1])
        self.assertEqual(9, downsample_tensor.shape.as_list()[-1])
        self.assertEqual(8, conv2d_tensor.shape.as_list()[-1])
        self.assertEqual(2, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    def test_reducing_downsample(self):
        """ Test for reducing a model with multiple downsample nodes """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = model_with_multiple_downsamples()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        conv2d = tf.compat.v1.get_default_graph().get_operation_by_name("downsample/conv2d/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (conv2d, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        conv2d_1 = tf.compat.v1.get_default_graph().get_operation_by_name("downsample_1/conv2d_1/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (conv2d_1, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        conv2d_2 = tf.compat.v1.get_default_graph().get_operation_by_name("downsample_1/conv2d_2/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (conv2d_2, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ['input_1']
        output_op_names = ['multiple_downsamples/Softmax']
        new_sess, _ = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                             module_zero_channels_list,
                                             reshape=True, in_place=True, verbose=True)
        sess.close()
        new_sess.close()
        self.assertEqual(0, 0)

    def test_reducing_upsample2d(self):
        """ Test for reducing a model with upsample2D op """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = model_with_upsample2d()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        conv2d = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (conv2d, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        const_op = sess.graph.get_operation_by_name('up_sampling2d/Const')
        tensor_content_length = const_op.get_attr('value').tensor_shape.dim[0].size
        unpack_string = str(tensor_content_length) + 'i'
        orig_upsample_size = struct.unpack(unpack_string, const_op.get_attr('value').tensor_content)

        input_op_names = ['input_1']
        output_op_names = ['model_with_upsample2d/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        self.assertEqual(3, len(ordered_modules_list))
        # Check that correct size was used
        const_op = new_sess.graph.get_operation_by_name('reduced_up_sampling2d/Const')
        tensor_content_length = const_op.get_attr('value').tensor_shape.dim[0].size
        unpack_string = str(tensor_content_length) + 'i'
        reduced_upsample_size = struct.unpack(unpack_string, const_op.get_attr('value').tensor_content)
        self.assertEqual(orig_upsample_size, reduced_upsample_size)

        sess.close()
        new_sess.close()

    def test_reducing_leakyrelu(self):
        """ Test for reducing a model with leaky_relu op """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = model_with_leaky_relu()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        conv2d = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [1, 2, 3]
        module_mask_pair = (conv2d, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        orig_alpha = sess.graph.get_operation_by_name('LeakyRelu').get_attr('alpha')

        input_op_names = ['input_1']
        output_op_names = ['model_with_leaky_relu/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)

        self.assertEqual(3, len(ordered_modules_list))
        # Check that correct alpha was used
        reduced_alpha = new_sess.graph.get_operation_by_name('reduced_LeakyRelu').get_attr('alpha')
        self.assertEqual(orig_alpha, reduced_alpha)

        sess.close()
        new_sess.close()


class TestTfWinnower(unittest.TestCase):
    """ Class for testing winnower module on tensorflow graphs """

    @pytest.mark.tf1
    def test_mask_propagation_on_keras_model(self):
        """ Test mask propagation on a conv module in keras_model """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = keras_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        input_op_names = ["conv2d_input"]
        output_op_names = ['keras_model/Softmax']
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        mask_winnower = MaskPropagationWinnower(sess, input_op_names, output_op_names,
                                                module_zero_channels_list, reshape=True,
                                                in_place=True, verbose=True)
        mask_winnower._propagate_masks()

        first_conv2d_opname = "conv2d/Conv2D"
        middle_batchnorm_opname = "batch_normalization"
        second_conv2d_opname = "conv2d_1/Conv2D"
        ops_dict = mask_winnower._conn_graph.get_all_ops()
        first_conv2d_mask = mask_winnower._mask_propagator.op_to_mask_dict[ops_dict[first_conv2d_opname]]
        self.assertEqual(3, sum(first_conv2d_mask.input_channel_masks[0]))
        self._check_mask_indices(input_channels_to_winnow, "output", first_conv2d_mask)
        middle_batchnorm_mask = mask_winnower._mask_propagator.op_to_mask_dict[ops_dict[middle_batchnorm_opname]]
        self._check_mask_indices(input_channels_to_winnow, "input", middle_batchnorm_mask)
        self._check_mask_indices(input_channels_to_winnow, "output", middle_batchnorm_mask)
        second_conv2d_mask = mask_winnower._mask_propagator.op_to_mask_dict[ops_dict[second_conv2d_opname]]
        self._check_mask_indices(input_channels_to_winnow, "input", second_conv2d_mask)
        self.assertEqual(4, sum(second_conv2d_mask.output_channel_masks[0]))
        sess.close()

    @pytest.mark.tf1
    def test_mask_propagation_on_single_residual_model(self):
        """ Test mask propagation on a conv module in keras_model """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = single_residual()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["input_1"]
        output_op_names = ['Relu_2']

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_3/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_2/Conv2D")
        input_channels_to_winnow_2 = [13, 15]
        module_mask_pair = (tf_op, input_channels_to_winnow_2)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow_3 = [13, 15]
        module_mask_pair = (tf_op, input_channels_to_winnow_3)
        module_zero_channels_list.append(module_mask_pair)

        mask_winnower = MaskPropagationWinnower(sess, input_op_names, output_op_names, module_zero_channels_list,
                                                reshape=True, in_place=True, verbose=True)
        mask_winnower._propagate_masks()
        ops_dict = mask_winnower._conn_graph.get_all_ops()
        self._check_mask_indices(input_channels_to_winnow, "input",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["conv2d_3/Conv2D"]])
        self._check_mask_indices(input_channels_to_winnow_2, "output",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["batch_normalization"]])
        self._check_mask_indices(input_channels_to_winnow_2, "input",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["batch_normalization"]])
        self._check_mask_indices(input_channels_to_winnow, "output",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["conv2d_2/Conv2D"]])
        self._check_mask_indices(input_channels_to_winnow_2, "output",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["Relu"]])
        self._check_mask_indices(input_channels_to_winnow_2, "input",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["Relu"]])
        self._check_mask_indices(input_channels_to_winnow_2, "output",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["conv2d/Conv2D"]])
        sess.close()

    @pytest.mark.tf2
    def test_mask_propagation_on_single_residual_model_for_tf2(self):
        """ Test mask propagation on a conv module in keras_model """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = single_residual_for_tf2()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["input_1"]
        output_op_names = ['Relu_2']

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_3/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_2/Conv2D")
        input_channels_to_winnow_2 = [13, 15]
        module_mask_pair = (tf_op, input_channels_to_winnow_2)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow_3 = [13, 15]
        module_mask_pair = (tf_op, input_channels_to_winnow_3)
        module_zero_channels_list.append(module_mask_pair)

        mask_winnower = MaskPropagationWinnower(sess, input_op_names, output_op_names, module_zero_channels_list,
                                                reshape=True, in_place=True, verbose=True)
        mask_winnower._propagate_masks()
        ops_dict = mask_winnower._conn_graph.get_all_ops()
        self._check_mask_indices(input_channels_to_winnow, "input",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["conv2d_3/Conv2D"]])
        self._check_mask_indices(input_channels_to_winnow_2, "output",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["batch_normalization"]])
        self._check_mask_indices(input_channels_to_winnow_2, "input",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["batch_normalization"]])
        self._check_mask_indices(input_channels_to_winnow, "output",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["conv2d_2/Conv2D"]])
        self._check_mask_indices(input_channels_to_winnow_2, "output",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["Relu"]])
        self._check_mask_indices(input_channels_to_winnow_2, "input",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["Relu"]])
        self._check_mask_indices(input_channels_to_winnow_2, "output",
                                 mask_winnower._mask_propagator.op_to_mask_dict[ops_dict["conv2d/Conv2D"]])
        sess.close()

    def test_mask_propagation_with_maxpool_as_last_layer(self):
        """ Test mask propagation on a model with a direct connectivity op as the last layer. """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        inputs = tf.keras.Input(shape=(64, 32, 3,))
        x = tf.keras.layers.Conv2D(64, (3, 3))(inputs)
        x = tf.keras.layers.Conv2D(16, (3, 3))(x)
        _ = tf.keras.layers.MaxPool2D()(x)

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [1]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["input_1"]
        output_op_names = ['max_pooling2d/MaxPool']
        mask_winnower = MaskPropagationWinnower(sess, input_op_names, output_op_names, module_zero_channels_list,
                                                reshape=True, in_place=True, verbose=True)
        mask_winnower._propagate_masks()
        self.assertEqual(0, 0)
        sess.close()

    def test_mask_propagation_with_conv_as_last_layer(self):
        """ Test mask propagation on a model with a conv op as the last layer. """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        inputs = tf.keras.Input(shape=(8, 8, 3,))
        x = tf.keras.layers.Conv2D(4, (2, 2))(inputs)
        _ = tf.keras.layers.Conv2D(2, (1, 1))(x)

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [1]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["input_1"]
        output_op_names = ['conv2d_1/BiasAdd']
        mask_winnower = MaskPropagationWinnower(sess, input_op_names, output_op_names, module_zero_channels_list,
                                                reshape=True, in_place=True, verbose=True)
        mask_winnower._propagate_masks()
        self.assertEqual(0, 0)
        sess.close()

    def test_mask_propagation_with_dense_as_last_layer(self):
        """ Test mask propagation on a model with a dense op as the last layer. """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        inputs = tf.keras.Input(shape=(8, 8, 3,))
        x = tf.keras.layers.Conv2D(4, (2, 2))(inputs)
        x = tf.keras.layers.Conv2D(2, (1, 1))(x)
        x = tf.keras.layers.Flatten()(x)
        _ = tf.keras.layers.Dense(2)(x)

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_1/Conv2D")
        input_channels_to_winnow = [1]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["input_1"]
        output_op_names = ['dense/BiasAdd']
        mask_winnower = MaskPropagationWinnower(sess, input_op_names, output_op_names, module_zero_channels_list,
                                                reshape=True, in_place=True, verbose=True)
        mask_winnower._propagate_masks()
        self.assertEqual(0, 0)
        sess.close()

    def test_mask_propagation_with_concat(self):
        """ Test mask propagation on a model with concat layer. """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = concat_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_3/Conv2D")
        input_channels_to_winnow = [2, 3, 6, 7, 17]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_4/Conv2D")
        input_channels_to_winnow_1 = [2, 3, 6, 7, 8, 17]
        module_mask_pair = (tf_op, input_channels_to_winnow_1)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["input_1"]
        output_op_names = ['concat_model/Softmax']
        mask_winnower = MaskPropagationWinnower(sess, input_op_names, output_op_names, module_zero_channels_list,
                                                reshape=True, in_place=True, verbose=True)
        mask_winnower._propagate_masks()
        modified_op_list = mask_winnower._mask_propagator.get_ops_with_non_default_ip_op_masks()
        self.assertEqual(6, len(modified_op_list))

        conv2d_1_op = mask_winnower._conn_graph.get_all_ops()["conv2d_1/Conv2D"]
        conv2d_1_op_mask = mask_winnower._mask_propagator.op_to_mask_dict[conv2d_1_op]
        self.assertEqual([1, 1, 0, 0, 1, 1], conv2d_1_op_mask.output_channel_masks[0])
        conv2d_op = mask_winnower._conn_graph.get_all_ops()["conv2d/Conv2D"]
        conv2d_op_mask = mask_winnower._mask_propagator.op_to_mask_dict[conv2d_op]
        self.assertEqual([0, 0, 1, 1, 1], conv2d_op_mask.output_channel_masks[0])
        conv2d_2_op = mask_winnower._conn_graph.get_all_ops()["conv2d_2/Conv2D"]
        conv2d_2_op_mask = mask_winnower._mask_propagator.op_to_mask_dict[conv2d_2_op]
        self.assertEqual([1, 1, 1, 1, 1, 1, 0], conv2d_2_op_mask.output_channel_masks[0])
        conv2d_3_op = mask_winnower._conn_graph.get_all_ops()["conv2d_3/Conv2D"]
        conv2d_3_op_mask = mask_winnower._mask_propagator.op_to_mask_dict[conv2d_3_op]
        self.assertEqual([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         conv2d_3_op_mask.input_channel_masks[0])
        conv2d_4_op = mask_winnower._conn_graph.get_all_ops()["conv2d_4/Conv2D"]
        conv2d_4_op_mask = mask_winnower._mask_propagator.op_to_mask_dict[conv2d_4_op]
        self.assertEqual([1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         conv2d_4_op_mask.input_channel_masks[0])
        sess.close()

    def test_mask_propagation_for_add_with_split_parent(self):
        """ Test mask propagation on a model with add that has a split parent """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()

        _ = upsample_model()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["input_1"]
        output_op_names = ['upsample_model/Softmax']

        # This conv2d is directly below an add that has split as one of its parents.  Thus we expect this mask to
        # propagate only through the parent that is not a split.  Additionally, due to special handling for add ops at
        # the end of mask propagation, we expect add's input and output masks to be all ones.
        module_zero_channels_list = []
        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_3/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        mask_winnower = MaskPropagationWinnower(sess, input_op_names, output_op_names, module_zero_channels_list,
                                                reshape=True, in_place=True, verbose=True)
        mask_winnower._propagate_masks()

        conv2d_2_op = mask_winnower._conn_graph.get_all_ops()["conv2d_2/Conv2D"]
        conv2d_2_op_mask = mask_winnower._mask_propagator.op_to_mask_dict[conv2d_2_op]
        self.assertEqual([1, 1, 1, 0, 1, 0, 1, 0], conv2d_2_op_mask.output_channel_masks[0])

        add_op = mask_winnower._conn_graph.get_all_ops()['Add']
        add_mask = mask_winnower._mask_propagator.op_to_mask_dict[add_op]
        self.assertEqual(8, sum(add_mask.input_channel_masks[0]))
        self.assertEqual(8, sum(add_mask.input_channel_masks[1]))
        self.assertEqual(8, sum(add_mask.output_channel_masks[0]))

        sess.close()

    def test_mask_propagation_for_add_with_non_split_parents(self):
        """ Test mask propagation on a model with add that does not have a split parent """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()

        _ = single_residual()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["input_1"]
        output_op_names = ['conv2d_4/Conv2D']

        # This conv2d is not directly below an add that has split as one of its parents.  Thus we expect this mask to
        # propagate through both parents of the add.  At the end of mask propagation, we expect add's input and output
        # masks to have channels 3, 5, and 7 marked for winnowing.
        module_zero_channels_list = []
        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_4/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        mask_winnower = MaskPropagationWinnower(sess, input_op_names, output_op_names, module_zero_channels_list,
                                                reshape=True, in_place=True, verbose=True)
        mask_winnower._propagate_masks()

        conv2d_1_op = mask_winnower._conn_graph.get_all_ops()["conv2d_1/Conv2D"]
        conv2d_1_op_mask = mask_winnower._mask_propagator.op_to_mask_dict[conv2d_1_op]
        self.assertEqual([1, 1, 1, 0, 1, 0, 1, 0], conv2d_1_op_mask.output_channel_masks[0])
        conv2d_3_op = mask_winnower._conn_graph.get_all_ops()["conv2d_3/Conv2D"]
        conv2d_3_op_mask = mask_winnower._mask_propagator.op_to_mask_dict[conv2d_3_op]
        self.assertEqual([1, 1, 1, 0, 1, 0, 1, 0], conv2d_3_op_mask.output_channel_masks[0])

        add_op = mask_winnower._conn_graph.get_all_ops()['Add']
        add_mask = mask_winnower._mask_propagator.op_to_mask_dict[add_op]
        self.assertEqual([1, 1, 1, 0, 1, 0, 1, 0], add_mask.input_channel_masks[0])
        self.assertEqual([1, 1, 1, 0, 1, 0, 1, 0], add_mask.input_channel_masks[1])
        self.assertEqual([1, 1, 1, 0, 1, 0, 1, 0], add_mask.output_channel_masks[0])

    def test_mask_propagation_set_downstream_masks(self):
        """ Test setting downstream masks """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()

        _ = model_to_test_downstream_masks()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["input_1"]
        output_op_names = ['model_to_test_downstream_masks/Softmax']

        # This conv2d is not directly below an add that has split as one of its parents.  Thus we expect this mask to
        # propagate through both parents of the add.  At the end of mask propagation, we expect add's input and output
        # masks to have channels 3, 5, and 7 marked for winnowing.
        module_zero_channels_list = []
        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_2/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_3/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        mask_winnower = MaskPropagationWinnower(sess, input_op_names, output_op_names, module_zero_channels_list,
                                                reshape=True, in_place=True, verbose=True)
        mask_winnower._propagate_masks()

        relu_op = mask_winnower._conn_graph.get_all_ops()["Relu"]
        relu_op_mask = mask_winnower._mask_propagator.op_to_mask_dict[relu_op]
        self.assertEqual(8, sum(relu_op_mask.output_channel_masks[0]))

        relu_1_op = mask_winnower._conn_graph.get_all_ops()["Relu_1"]
        relu_1_op_mask = mask_winnower._mask_propagator.op_to_mask_dict[relu_1_op]
        self.assertEqual(8, sum(relu_1_op_mask.output_channel_masks[0]))

        sess.close()

    def test_create_masks_with_postprocessing_ops(self):
        """ Test that create_masks() is able to handle models with postprocessing nodes """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []
        model_with_postprocessing_nodes()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        # _ = tf.compat.v1.summary.FileWriter('./model_with_postprocessing_nodes', tf.compat.v1.get_default_graph())
        input_op_names = ["input_1"]
        output_op_names = ['top1-acc', 'top5-acc']
        mask_winnower = MaskPropagationWinnower(sess, input_op_names, output_op_names, module_zero_channels_list,
                                                reshape=True, in_place=True, verbose=True)
        flatten_op = mask_winnower._conn_graph.get_all_ops()['flatten/Reshape']
        self.assertTrue(flatten_op not in mask_winnower._mask_propagator.op_to_mask_dict.keys())
        self.assertEqual(3, len(mask_winnower._mask_propagator.op_to_mask_dict))

    def _check_mask_indices(self, winnowed_channels: List, channel_type: str, op_mask: Mask):
        if channel_type == "input":
            for channel in winnowed_channels:
                self.assertEqual(0, op_mask.input_channel_masks[0][channel])
        elif channel_type == "output":
            for channel in winnowed_channels:
                self.assertEqual(0, op_mask.output_channel_masks[0][channel])
