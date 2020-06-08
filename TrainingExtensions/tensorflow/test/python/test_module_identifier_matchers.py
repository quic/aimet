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
""" This file contains unit tests for testing  Module Identifier Matchers. """

import unittest
import logging
import tensorflow as tf

from aimet_common.utils import AimetLogger
import aimet_tensorflow.common.module_identifier_matchers as mod_matchers
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.examples.test_models import keras_model_functional, depthwise_conv2d_model


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Test, logging.DEBUG)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.ConnectedGraph, logging.DEBUG)


class TestModuleIdentifierMatchers(unittest.TestCase):
    """ Test Module Identifier Matcher functions """

    def test_match_conv2d(self):
        """ Test match_conv2d_dense_type_ops() for Conv2D """

        tf.reset_default_graph()
        with tf.device('/cpu:0'):
            model = keras_model_functional()
            model.summary()

        sess = tf.Session()
        initialize_uninitialized_vars(sess)

        conv2d_op = sess.graph.get_operation_by_name('conv2d/Conv2D')

        op_info = mod_matchers.ModuleIdentifierOpInfo(module_name=conv2d_op.name,
                                                      op_type=conv2d_op.type,
                                                      tf_op=conv2d_op)

        op_to_module_dict = dict()
        op_to_module_dict[conv2d_op] = op_info
        true_or_false = mod_matchers.match_conv2d_dense_type_ops(op_to_module_dict, op_info)
        self.assertTrue(true_or_false)

    def test_match_matmul(self):
        """ Test match_conv2d_dense_type_ops() for MatMul"""

        tf.reset_default_graph()
        with tf.device('/cpu:0'):
            model = keras_model_functional()
            model.summary()

        sess = tf.Session()
        initialize_uninitialized_vars(sess)

        matmul_op = sess.graph.get_operation_by_name('keras_model_functional/MatMul')

        op_info = mod_matchers.ModuleIdentifierOpInfo(module_name=matmul_op.name,
                                                      op_type=matmul_op.type,
                                                      tf_op=matmul_op)
        self.assertTrue(op_info.op_type == 'MatMul')

        op_to_module_dict = dict()
        op_to_module_dict[matmul_op] = op_info
        true_or_false = mod_matchers.match_conv2d_dense_type_ops(op_to_module_dict, op_info)
        self.assertTrue(true_or_false)

        # The function match_conv2d_dense_type_ops() changes op_type from 'MatMul' to'Dense'
        self.assertTrue(op_info.op_type == 'Dense')

    def test_match_depthwise_conv2d(self):
        """ Test match_conv2d_dense_type_ops() for DepthwiseConv2dNative """

        tf.reset_default_graph()
        with tf.device('/cpu:0'):
            _ = depthwise_conv2d_model()

        sess = tf.Session()
        initialize_uninitialized_vars(sess)

        depth_op = sess.graph.get_operation_by_name('depthwise_conv2d/depthwise')

        op_info = mod_matchers.ModuleIdentifierOpInfo(module_name=depth_op.name,
                                                      op_type=depth_op.type,
                                                      tf_op=depth_op)

        op_to_module_dict = dict()
        op_to_module_dict[depth_op] = op_info
        true_or_false = mod_matchers.match_conv2d_dense_type_ops(op_to_module_dict, op_info)
        self.assertTrue(true_or_false)
