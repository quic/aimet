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
""" This file contains nightly tests for testing winnowing for tf models. """

import pytest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import unittest
import logging
from packaging import version
import tensorflow as tf
if not version.parse(tf.version.VERSION) >= version.parse("2.00"):
    from tensorflow.contrib.slim.nets import vgg        # pylint: disable=no-name-in-module
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
import numpy as np

from aimet_common.utils import AimetLogger
from aimet_tensorflow.utils.graph import update_keras_bn_ops_trainable_flag
from aimet_tensorflow.utils.graph_saver import save_and_load_graph
import aimet_tensorflow.winnow.winnow as winnow
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars

tf.compat.v1.disable_eager_execution()
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Test, logging.DEBUG)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.ConnectedGraph, logging.DEBUG)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Winnow, logging.DEBUG)


class TestTfWinnower(unittest.TestCase):
    """ Class for testing winnower module on tensorflow graphs """

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    def test_reducing_vgg16(self):
        """ This test winnows a VGG16 model"""

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = VGG16(weights=None)
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("block5_conv1/Conv2D")
        input_channels_to_winnow = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op_2 = tf.compat.v1.get_default_graph().get_operation_by_name("block3_conv1/Conv2D")
        input_channels_to_winnow_2 = [11, 13, 15, 17]
        module_mask_pair_2 = (tf_op_2, input_channels_to_winnow_2)
        module_zero_channels_list.append(module_mask_pair_2)

        tf_op_3 = tf.compat.v1.get_default_graph().get_operation_by_name("block2_conv2/Conv2D")
        input_channels_to_winnow_3 = [1, 2, 3, 4, 5]
        module_mask_pair_3 = (tf_op_3, input_channels_to_winnow_3)
        module_zero_channels_list.append(module_mask_pair_3)

        tf_op_4 = tf.compat.v1.get_default_graph().get_operation_by_name("block2_conv1/Conv2D")
        input_channels_to_winnow_4 = [20, 21, 22, 23]
        module_mask_pair_4 = (tf_op_4, input_channels_to_winnow_4)
        module_zero_channels_list.append(module_mask_pair_4)

        input_op_names = ["input_1"]
        output_op_names = ['predictions/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)
        # Save and reload modified graph to allow changes to take effect
        new_sess = save_and_load_graph('./saver', new_sess)

        # uncomment the following to generate tensorboard viewable file
        # _ = tf.compat.v1.summary.FileWriter('./reduced_graph', new_sess.graph)

        # Check certain weight indices to ensure that weights were reduced correctly
        b4c3_kernel = new_sess.graph.get_tensor_by_name("block4_conv3/kernel/Read/"
                                                        "ReadVariableOp:0").eval(session=new_sess)
        red_b4c3_kernel = new_sess.graph.get_tensor_by_name("reduced_block4_conv3/kernel/"
                                                            "Read/ReadVariableOp:0").eval(session=new_sess)
        self.assertEqual(red_b4c3_kernel.shape, (3, 3, 512, 509))
        self.assertEqual(red_b4c3_kernel[0][0][0][2], b4c3_kernel[0][0][0][2])
        self.assertEqual(np.sum(red_b4c3_kernel[0][0][0][5:]), np.sum(b4c3_kernel[0][0][0][8:]))

        # Test that evaluating the new session uses the newly reduced modules.
        # Do so by first evaluating a tensor in a module coming after a set of reduced modules.
        # Zero out weights and biases of one of the original unreduced modules preceding the tensor.
        # Reevaluate the tensor and expect to see no change, since the original unreduced module should not be used
        # anymore.
        # Then zero out weights and biases of one of the newly reduced modules.
        # Finally reevaluate the same tensor as before.  This time, we expect to see the result be zero.

        with new_sess.graph.as_default():
            inp = tf.random.uniform(shape=(1, 224, 224, 3))
            inp_array = inp.eval(session=new_sess)
            model_input = new_sess.graph.get_tensor_by_name("input_1:0")

            # run through entire model to check no error is produced
            model_output = new_sess.graph.get_tensor_by_name("predictions/Softmax:0")
            _ = new_sess.run(model_output, feed_dict={model_input: inp_array})

        self.assertEqual(13, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    def test_reducing_resnet_50(self):
        """ Test module reduction in resnet_50 """
        tf.compat.v1.reset_default_graph()

        module_zero_channels_list = []

        model = ResNet50(weights=None)
        _ = update_keras_bn_ops_trainable_flag(model, False, "./t")
        sess = tf.compat.v1.keras.backend.get_session()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2_block1_1_conv/Conv2D")
        input_channels_to_winnow_1 = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow_1)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2_block1_0_conv/Conv2D")
        input_channels_to_winnow_2 = [3, 5, 7, 8]
        module_mask_pair = (tf_op, input_channels_to_winnow_2)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv3_block1_1_conv/Conv2D")
        input_channels_to_winnow_3 = [3, 5, 7]
        module_mask_pair = (tf_op, input_channels_to_winnow_3)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv3_block1_0_conv/Conv2D")
        input_channels_to_winnow_4 = [3, 5, 7, 8]
        module_mask_pair = (tf_op, input_channels_to_winnow_4)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["input_1"]
        output_op_names = ['probs/Softmax']
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            output_op_names = ['predictions/Softmax']

        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)
        # Save and reload modified graph to allow changes to take effect
        # Need to initialize uninitialized variables first since only newly winnowed conv ops are initialized during
        # winnow_tf_model, and all other newly winnowed ops are not.
        with new_sess.graph.as_default():
            initialize_uninitialized_vars(new_sess)
        new_sess = save_and_load_graph('./saver', new_sess)

        # _ = tf.compat.v1.summary.FileWriter('./reduced_graph', new_sess.graph)
        with new_sess.graph.as_default():
            inp = tf.random.uniform(shape=(1, 224, 224, 3))
            inp_array = inp.eval(session=new_sess)
        model_input = new_sess.graph.get_tensor_by_name("input_1:0")

        model_output = new_sess.graph.get_tensor_by_name(output_op_names[0]+':0')

        # check that reduced tensor shapes are as expected
        reduced_conv3_block1_1_input = new_sess.graph.get_operation_by_name("reduced_conv3_block1_1_conv/"
                                                                             "Conv2D").inputs[0]
        reduced_conv3_block1_0_input = new_sess.graph.get_operation_by_name("reduced_conv3_block1_0_conv/"
                                                                            "Conv2D").inputs[0]
        reduced_conv2_block3_3_output = new_sess.graph.get_tensor_by_name("reduced_conv2_block3_3_conv/"
                                                                           "Conv2D:0")
        reduced_conv2_block1_1_input = new_sess.graph.get_operation_by_name("reduced_conv2_block1_1_conv/"
                                                                             "Conv2D").inputs[0]
        reduced_conv2_block1_0_input = new_sess.graph.get_operation_by_name("reduced_conv2_block1_0_conv/"
                                                                            "Conv2D").inputs[0]
        reduced_conv1_output = new_sess.graph.get_tensor_by_name("reduced_conv1_conv/Conv2D:0")
        self.assertEqual(253, reduced_conv3_block1_1_input.shape.as_list()[-1])
        self.assertEqual(252, reduced_conv3_block1_0_input.shape.as_list()[-1])
        self.assertEqual(253, reduced_conv2_block3_3_output.shape.as_list()[-1])
        self.assertEqual(61, reduced_conv2_block1_1_input.shape.as_list()[-1])
        self.assertEqual(60, reduced_conv2_block1_0_input.shape.as_list()[-1])
        self.assertEqual(61, reduced_conv1_output.shape.as_list()[-1])

        # run through entire model to check no error is produced
        _ = new_sess.run(model_output, feed_dict={model_input: inp_array})
        self.assertEqual(11, len(ordered_modules_list))
        new_sess.close()
        sess.close()

    @pytest.mark.tf1
    def test_reducing_inceptionV3(self):
        """ Test module reduction in inceptionV3 """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        _ = InceptionV3(weights=None)
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_12/Conv2D")
        input_channels_to_winnow = [0, 1, 64, 128, 224]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_13/Conv2D")
        input_channels_to_winnow_1 = [0, 64, 65, 66, 128, 224]
        module_mask_pair = (tf_op, input_channels_to_winnow_1)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_15/Conv2D")
        input_channels_to_winnow_2 = [0, 64, 128, 129, 130, 131, 224]
        module_mask_pair = (tf_op, input_channels_to_winnow_2)
        module_zero_channels_list.append(module_mask_pair)

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("conv2d_18/Conv2D")
        input_channels_to_winnow_3 = [0, 64, 128, 224, 225, 226, 227, 228]
        module_mask_pair = (tf_op, input_channels_to_winnow_3)
        module_zero_channels_list.append(module_mask_pair)

        input_op_names = ["input_1"]
        output_op_names = ['predictions/Softmax']
        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list,
                                                                reshape=True, in_place=True, verbose=True)
        # Save and reload modified graph to allow changes to take effect
        # Need to initialize uninitialized variables first since only newly winnowed conv ops are initialized during
        # winnow_tf_model, and all other newly winnowed ops are not.
        with new_sess.graph.as_default():
            initialize_uninitialized_vars(new_sess)
        new_sess = save_and_load_graph('./saver', new_sess)

        # _ = tf.compat.v1.summary.FileWriter('./reduced_graph', new_sess.graph)

        with new_sess.graph.as_default():
            inp = tf.random.uniform(shape=(1, 299, 299, 3))
            inp_array = inp.eval(session=new_sess)
            model_input = new_sess.graph.get_tensor_by_name("input_1:0")
            model_output = new_sess.graph.get_tensor_by_name("predictions/Softmax:0")

            # check that reduced tensor shapes are as expected
            reduced_conv2d_12_input = new_sess.graph.get_operation_by_name("reduced_conv2d_12/Conv2D").inputs[0]
            reduced_conv2d_13_input = new_sess.graph.get_operation_by_name("reduced_conv2d_13/Conv2D").inputs[0]
            reduced_conv2d_15_input = new_sess.graph.get_operation_by_name("reduced_conv2d_15/Conv2D").inputs[0]
            reduced_conv2d_18_input = new_sess.graph.get_operation_by_name("reduced_conv2d_18/Conv2D").inputs[0]
            reduced_conv2d_5_output = new_sess.graph.get_tensor_by_name("reduced_conv2d_5/Conv2D:0")
            reduced_conv2d_7_output = new_sess.graph.get_tensor_by_name("reduced_conv2d_7/Conv2D:0")
            reduced_conv2d_10_output = new_sess.graph.get_tensor_by_name("reduced_conv2d_10/Conv2D:0")
            reduced_conv2d_11_output = new_sess.graph.get_tensor_by_name("reduced_conv2d_11/Conv2D:0")
            self.assertEqual(251, reduced_conv2d_12_input.shape.as_list()[-1])
            self.assertEqual(250, reduced_conv2d_13_input.shape.as_list()[-1])
            self.assertEqual(249, reduced_conv2d_15_input.shape.as_list()[-1])
            self.assertEqual(248, reduced_conv2d_18_input.shape.as_list()[-1])
            self.assertEqual(63, reduced_conv2d_5_output.shape.as_list()[-1])
            self.assertEqual(63, reduced_conv2d_7_output.shape.as_list()[-1])
            self.assertEqual(95, reduced_conv2d_10_output.shape.as_list()[-1])
            self.assertEqual(31, reduced_conv2d_11_output.shape.as_list()[-1])
            self.assertEqual(17, len(ordered_modules_list))

            # run through entire model to check no error is produced
            _ = new_sess.run(model_output, feed_dict={model_input: inp_array})
        new_sess.close()
        sess.close()

    @pytest.mark.tf1
    def test_reducing_vgg16_slim(self):
        """ Test reducing vgg16 slim model """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        module_zero_channels_list = []

        inp = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3])
        _ = vgg.vgg_16(inp)
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        input_op_names = ["Placeholder"]
        output_op_names = ['vgg_16/fc8/squeezed']

        tf_op = tf.compat.v1.get_default_graph().get_operation_by_name("vgg_16/fc7/Conv2D")
        input_channels_to_winnow = [2, 3, 4]
        module_mask_pair = (tf_op, input_channels_to_winnow)
        module_zero_channels_list.append(module_mask_pair)

        new_sess, ordered_modules_list = winnow.winnow_tf_model(sess, input_op_names, output_op_names,
                                                                module_zero_channels_list, reshape=True,
                                                                in_place=True, verbose=True)
        # Save and reload modified graph to allow changes to take effect
        new_sess = save_and_load_graph('./saver', new_sess)

        # _ = tf.compat.v1.summary.FileWriter('./reduced_graph', new_sess.graph)

        with new_sess.graph.as_default():
            inp = tf.random.uniform(shape=(1, 224, 224, 3))
            inp_array = inp.eval(session=new_sess)
            model_input = new_sess.graph.get_tensor_by_name("Placeholder:0")
            model_output = new_sess.graph.get_tensor_by_name("vgg_16/fc8/squeezed:0")

            # run through entire model to check no error is produced
            _ = new_sess.run(model_output, feed_dict={model_input: inp_array})
        self.assertEqual(4, len(ordered_modules_list))
        new_sess.close()
        sess.close()
