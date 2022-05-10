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
""" This file contains unit tests for testing ModuleIdentifier modules. """

import unittest
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pytest
import tensorflow as tf
from aimet_common.utils import AimetLogger
from aimet_tensorflow.common.module_identifier import StructureModuleIdentifier
from aimet_tensorflow.examples.test_models import keras_model, keras_model_functional, tf_slim_basic_model,\
    keras_model_functional_for_tf2

tf.compat.v1.disable_eager_execution()
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Test, logging.DEBUG)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.ConnectedGraph, logging.DEBUG)


class TestStructureModuleIdentifier(unittest.TestCase):
    """ Test StructureModuleIdentifier module """

    @pytest.mark.tf1
    def test_get_op_info(self):
        """ Test get_op_info() in StructureModuleIdentifier """
        my_op_type_set = set()
        current_module_set = set()

        tf.compat.v1.reset_default_graph()
        _ = keras_model()

        module_identifier = StructureModuleIdentifier(tf.compat.v1.get_default_graph(), ["conv2d_input"],
                                                      set(tf.compat.v1.get_default_graph().get_operations()))
        for op_info in module_identifier.op_to_module_dict.values():
            my_op_type_set.add(op_info.op_type)
            current_module_set.add(op_info.module_name)

        # Only identifies 2 conv2d, 2 fusedbatchnorm, flatten, and dense
        self.assertEqual(6, len(current_module_set))
        self.assertEqual(4, len(my_op_type_set))

    @pytest.mark.tf2
    def test_get_op_info_for_tf2(self):
        """ Test get_op_info() in StructureModuleIdentifier """
        my_op_type_set = set()
        current_module_set = set()

        tf.compat.v1.reset_default_graph()
        _ = keras_model_functional_for_tf2()

        module_identifier = StructureModuleIdentifier(tf.compat.v1.get_default_graph(), ["conv2d_input"],
                                                      set(tf.compat.v1.get_default_graph().get_operations()))
        for op_info in module_identifier.op_to_module_dict.values():
            print(op_info.module_name)
            my_op_type_set.add(op_info.op_type)
            current_module_set.add(op_info.module_name)

        # Only identifies 4 conv2d, 3 fusedbatchnorm, flatten, and dense
        self.assertEqual(9, len(current_module_set))
        self.assertEqual(4, len(my_op_type_set))

    @pytest.mark.tf1
    def test_fused_batch_norm_matcher_keras(self):
        """ Test fused batch norm matchers """

        tf.compat.v1.reset_default_graph()
        _ = keras_model_functional()

        module_identifier = StructureModuleIdentifier(tf.compat.v1.get_default_graph(), ["input_1"],
                                                      set(tf.compat.v1.get_default_graph().get_operations()))
        bn_op = tf.compat.v1.get_default_graph().get_operation_by_name('batch_normalization/FusedBatchNormV3')
        self.assertTrue(bn_op in module_identifier.op_to_module_dict.keys())
        self.assertEqual(module_identifier.op_to_module_dict[bn_op].module_name, 'batch_normalization')
        switch_op = tf.compat.v1.get_default_graph().get_operation_by_name('scope_1/batch_normalization_1/cond/'
                                                                 'FusedBatchNormV3/Switch')
        self.assertEqual(module_identifier.op_to_module_dict[switch_op].module_name, 'scope_1/batch_normalization_1')

    @pytest.mark.tf2
    def test_fused_batch_norm_matcher_keras_for_tf2(self):
        """ Test fused batch norm matchers """
        tf.compat.v1.reset_default_graph()
        _ = keras_model_functional_for_tf2()

        module_identifier = StructureModuleIdentifier(tf.compat.v1.get_default_graph(), ["input_1"],
                                                      set(tf.compat.v1.get_default_graph().get_operations()))
        bn_op = tf.compat.v1.get_default_graph().get_operation_by_name('batch_normalization/FusedBatchNormV3')
        self.assertTrue(bn_op in module_identifier.op_to_module_dict.keys())
        self.assertEqual(module_identifier.op_to_module_dict[bn_op].module_name, 'batch_normalization')

    @pytest.mark.tf1
    def test_fused_batch_norm_matcher_slim(self):
        """ Test fused batch norm matchers """

        tf.compat.v1.reset_default_graph()
        x = tf.compat.v1.placeholder(tf.float32, [1, 32, 32, 3])
        _ = tf_slim_basic_model(x)
        module_identifier = StructureModuleIdentifier(tf.compat.v1.get_default_graph(), ["Placeholder"],
                                                      set(tf.compat.v1.get_default_graph().get_operations()))
        mul_op = tf.compat.v1.get_default_graph().get_operation_by_name('BatchNorm/FusedBatchNormV3')
        self.assertEqual(module_identifier.op_to_module_dict[mul_op].module_name, 'BatchNorm')
        bn_1_merge_op = tf.compat.v1.get_default_graph().get_operation_by_name('BatchNorm_1/cond/Merge')
        self.assertEqual(module_identifier.op_to_module_dict[bn_1_merge_op].module_name, 'BatchNorm_1')
        bn_2_op = tf.compat.v1.get_default_graph().get_operation_by_name('BatchNorm_2/FusedBatchNormV3')
        self.assertTrue(bn_2_op in module_identifier.op_to_module_dict.keys())
        self.assertEqual(module_identifier.op_to_module_dict[bn_2_op].module_name, 'BatchNorm_2')
