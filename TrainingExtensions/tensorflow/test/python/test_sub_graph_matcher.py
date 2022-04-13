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
""" This file contains unit tests for testing  Sub Graph  functions. """

import os

import pytest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import unittest
import logging
import tensorflow as tf
from packaging import version
from aimet_tensorflow.quantize import graph_matcher

from aimet_common.utils import AimetLogger
from aimet_tensorflow.common.sub_graph_matcher_op_templates import op_type_templates
from aimet_tensorflow.common.sub_graph_matcher import create_subgraph_for_op_default,\
    create_op_type_patterns_from_subgraph
from aimet_tensorflow.examples.test_models import keras_model_functional

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Test, logging.DEBUG)
AimetLogger.set_area_logger_level(AimetLogger.LogAreas.ConnectedGraph, logging.DEBUG)
tf.compat.v1.disable_eager_execution()

subgraph_constructors = op_type_templates


class TestSubGraph(unittest.TestCase):
    """ Test Sub Graph functions """

    @pytest.mark.tf1
    def test_instance_norm_model(self):
        input_shape = subgraph_constructors['InstanceNormalization']['input_shape']
        constructor_string = subgraph_constructors['InstanceNormalization']['constructor']
        instance_norm_subgraph = create_subgraph_for_op_default(input_shape, constructor_string)

        logger.debug("The OPs created by create_subgraph_for_op_default()")
        for op in instance_norm_subgraph.get_operations():
            logger.debug("OP: %s", op.name)

        op_type_patterns = create_op_type_patterns_from_subgraph(instance_norm_subgraph, additional_starting_ops=[])
        for pat in op_type_patterns:
            logger.debug(pat._op_type)

        self.assertEqual(op_type_patterns[-1]._op_type, 'AddV2')

    def test_conv2d_no_bias_subgraph(self):
        """ test sub graph for conv2D Op without bias """

        input_shape = subgraph_constructors['Conv2D']['input_shape']
        constructor_string = subgraph_constructors['Conv2D']['constructor']
        conv_subgraph = create_subgraph_for_op_default(input_shape, constructor_string)

        logger.debug("The OPs created by create_subgraph_for_op_default()")
        for op in conv_subgraph.get_operations():
            logger.debug("OP: %s", op.name)

        op_type_patterns = create_op_type_patterns_from_subgraph(conv_subgraph, additional_starting_ops=[])
        for pat in op_type_patterns:
            logger.debug(pat._op_type)

        # The pattern list consists of successive OpTypePattern() objects starting from the layer's input Op to
        # the output Op. Each OpTypePattern() becomes an input to the next OpTypePattern(). In the case of a Conv2D
        # without Bias, the last element in the pattern list is Conv2D
        self.assertEqual(op_type_patterns[-1]._op_type, 'Conv2D')

    def test_conv2d_with_bias_subgraph(self):
        """ test sub graph for conv2D Op with bias """

        input_shape = subgraph_constructors['Conv2D_with_bias']['input_shape']
        constructor_string = subgraph_constructors['Conv2D_with_bias']['constructor']
        conv_subgraph = create_subgraph_for_op_default(input_shape, constructor_string)

        logger.debug("The OPs created by create_subgraph_for_op_default()")
        for op in conv_subgraph.get_operations():
            logger.debug("OP: %s", op.name)

        op_type_patterns = create_op_type_patterns_from_subgraph(conv_subgraph, additional_starting_ops=[])
        for pat in op_type_patterns:
            logger.debug(pat._op_type)

        # The pattern list consists of successive OpTypePattern() objects starting from the layer's input Op to
        # the output Op. Each OpTypePattern() becomes an input to the next OpTypePattern(). In the case of a Conv2D
        # with Bias, the last element in the pattern list is BiasAdd.
        self.assertEqual(op_type_patterns[-1]._op_type, 'BiasAdd')

    @pytest.mark.tf1
    def test_fused_batchnorm_training_tensor_subgraph(self):
        """ test sub graph for FusedBatchNorm training Tensor"""

        input_shape = subgraph_constructors['BN_keras_with_training_tensor']['input_shape']
        constructor_string = subgraph_constructors['BN_keras_with_training_tensor']['constructor']
        bn_subgraph = create_subgraph_for_op_default(input_shape, constructor_string)

        logger.debug("Step: I The OPs created by create_subgraph_for_op_default()")
        for op in bn_subgraph.get_operations():
            logger.debug("OP 1: %s", op.name)

        op_type_patterns = create_op_type_patterns_from_subgraph(bn_subgraph, additional_starting_ops=[])
        logger.debug("Step: II The OPs created by create_op_type_pattens_from_subgraph() %d", len(op_type_patterns))
        for pat in op_type_patterns:
            logger.debug("OP 2: %s", pat._op_type)

        # The pattern list consists of successive OpTypePattern() objects starting from the layer's input Op to
        # the output Op. Each OpTypePattern() becomes an input to the next OpTypePattern().
        self.assertEqual(op_type_patterns[-1]._op_type, 'Merge')

    def test_fused_batchnorm_with_training_False_subgraph(self):
        """ test sub graph for FusedBatchNorm training False"""

        input_shape = subgraph_constructors['BN_keras_with_training_False']['input_shape']
        constructor_string = subgraph_constructors['BN_keras_with_training_False']['constructor']
        bn_subgraph = create_subgraph_for_op_default(input_shape, constructor_string)

        logger.debug("Step: I The OPs created by create_subgraph_for_op_default()")
        for op in bn_subgraph.get_operations():
            logger.debug("OP 1: %s", op.name)

        op_type_patterns = create_op_type_patterns_from_subgraph(bn_subgraph, additional_starting_ops=[])
        logger.debug("Step: II The OPs created by create_op_type_pattens_from_subgraph() %d", len(op_type_patterns))
        for pat in op_type_patterns:
            logger.debug("OP 2: %s", pat._op_type)

        # The pattern list consists of successive OpTypePattern() objects starting from the layer's input Op to
        # the output Op. Each OpTypePattern() becomes an input to the next OpTypePattern().
        self.assertEqual(op_type_patterns[-1]._op_type, 'FusedBatchNormV3')

    def test_dense_subgraph(self):
        """ test sub graph for dense """

        input_shape = subgraph_constructors['Dense']['input_shape']
        constructor_string = subgraph_constructors['Dense']['constructor']
        dense_subgraph = create_subgraph_for_op_default(input_shape, constructor_string)

        logger.debug("Step: I The OPs created by create_subgraph_for Dense Op()")
        for op in dense_subgraph.get_operations():
            logger.debug("OP: %s", op.name)

        dense_patterns = create_op_type_patterns_from_subgraph(dense_subgraph, additional_starting_ops=[])
        logger.debug("Length of Dense pattern: %d", len(dense_patterns))
        for pat in dense_patterns:
            logger.debug(pat._op_type)

        # The pattern list consists of successive OpTypePattern() objects starting from the layer's input Op to
        # the output Op. Each OpTypePattern() becomes an input to the next OpTypePattern(). In the case of a Dense
        # without activation, the last element in the pattern list is BiasAdd.
        self.assertEqual(dense_patterns[-1]._op_type, 'BiasAdd')

    @pytest.mark.tf1
    def test_conv_subgraph_with_a_model(self):
        """ Detect Conv2D, Conv2D with Bias and FusedBatchNorm subgraphs in the session graph. """

        patterns_to_match = []
        op_to_pattern_dict = {}
        for pattern_name, subgraph_constructor in subgraph_constructors.items():
            input_shape = subgraph_constructor['input_shape']
            constructor_string = subgraph_constructor['constructor']
            supported_tf_versions = subgraph_constructor['supported_tf_versions']
            if version.parse(tf.version.VERSION).major in supported_tf_versions:
                continue
            logger.debug(pattern_name)
            subgraph = create_subgraph_for_op_default(input_shape, constructor_string)
            patterns = create_op_type_patterns_from_subgraph(subgraph, additional_starting_ops=[])
            patterns_to_match.append(patterns[-1])
            op_to_pattern_dict[pattern_name] = patterns[-1]
            logger.debug("Length of %s pattern: %d", pattern_name, len(patterns))

        # OneOfPattern for Conv@D, Conv2D with Bias and FusedBatchNorm
        all_patterns = graph_matcher.OneofPattern(patterns_to_match)
        layer_matcher = graph_matcher.GraphMatcher(all_patterns)

        # Use the keras_model_functional.
        sess = tf.compat.v1.Session(graph=tf.Graph())
        with sess.graph.as_default():
            with tf.device('/cpu:0'):
                model = keras_model_functional()
                model.summary()
            init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        # Uncomment to use Tensorboard
        # _ = tf.compat.v1.summary.FileWriter('./subgraph', sess.graph)

        # Graph Match
        matched_op_set = set()  # Set to keep track of Ops that have been detected already.
        match_counter = 0
        # layer_matcher = graph_matcher.GraphMatcher(conv_bias_patterns[-1])
        for match_result in layer_matcher.match_graph(sess.graph):
            if match_result:
                match_counter += 1

                # Conv2D Ops could be with or without Bias.
                # As the first step, detect all the Conv2D Ops with Bias.
                conv_bias_op = match_result.get_op(op_to_pattern_dict['Conv2D_with_bias'])
                if conv_bias_op:
                    if conv_bias_op.inputs[0]._op not in matched_op_set:
                        logger.debug("Conv Op with bias: %s, %d", conv_bias_op.name, match_counter)
                        matched_op_set.add(conv_bias_op.inputs[0]._op)

                # Since the Conv Op with Bias is already added to the matched_op_set,
                # Conv Ops with Bias won't be duplicated by the following match.
                conv_op = match_result.get_op(op_to_pattern_dict['Conv2D'])
                if conv_op:
                    if conv_op not in matched_op_set:
                        logger.debug("Conv Op no bias: %s, %d", conv_op.name, match_counter)
                        matched_op_set.add(conv_op)

                bn_1_op = match_result.get_op(op_to_pattern_dict['BN_keras_with_training_tensor'])
                if bn_1_op:
                    if bn_1_op.inputs[0]._op not in matched_op_set:
                        matched_op_set.add(bn_1_op.inputs[0]._op)
                        logger.debug("FusedBatchNorm 1 Op: %s, %d", bn_1_op.inputs[0]._op.name, match_counter)
                    if bn_1_op.inputs[1]._op not in matched_op_set:
                        matched_op_set.add(bn_1_op.inputs[1]._op)
                        logger.debug("FusedBatchNorm 1 Op: %s, %d", bn_1_op.inputs[1]._op.name, match_counter)

                bn_2_op = match_result.get_op(op_to_pattern_dict['BN_keras_with_training_False'])
                if bn_2_op:
                    logger.debug("FusedBatchNorm 2 Op: %s", bn_2_op.name)

                bn_3_op = match_result.get_op(op_to_pattern_dict['BN_keras_with_training_True'])
                if bn_3_op:
                    logger.debug("FusedBatchNorm 3 Op: %s", bn_3_op.name)

                flatten_op = match_result.get_op(op_to_pattern_dict['Flatten'])
                if flatten_op:
                    logger.debug("Flatten Op: %s, %d", flatten_op.name, match_counter)
                    matched_op_set.add(flatten_op)

                dense_op = match_result.get_op(op_to_pattern_dict['Dense'])
                if dense_op:
                    logger.debug("dense Op: %s, %d", dense_op.name, match_counter)
                    matched_op_set.add(dense_op)

        logger.debug(len(matched_op_set))

    def test_transposed_conv2d_no_bias_subgraph(self):
        """ test sub graph for conv2D Op without bias """
        tf.compat.v1.reset_default_graph()
        input_shape = subgraph_constructors['Conv2DTranspose']['input_shape']
        constructor_string = subgraph_constructors['Conv2DTranspose']['constructor']
        transposed_conv_subgraph = create_subgraph_for_op_default(input_shape, constructor_string)
        logger.debug("The OPs created by create_subgraph_for_op_default()")
        for op in transposed_conv_subgraph.get_operations():
            logger.debug("OP: %s", op.name)

        op_type_patterns = create_op_type_patterns_from_subgraph(transposed_conv_subgraph, additional_starting_ops=[])
        for pat in op_type_patterns:
            logger.debug(pat._op_type)

        self.assertEqual(op_type_patterns[-1]._op_type, 'Conv2DBackpropInput')

    def test_transposed_conv2d_bias_subgraph(self):
        """ test sub graph for conv2D Op without bias """
        tf.compat.v1.reset_default_graph()
        input_shape = subgraph_constructors['Conv2DTranspose_with_bias']['input_shape']
        constructor_string = subgraph_constructors['Conv2DTranspose_with_bias']['constructor']
        transposed_conv_subgraph = create_subgraph_for_op_default(input_shape, constructor_string)
        logger.debug("The OPs created by create_subgraph_for_op_default()")
        for op in transposed_conv_subgraph.get_operations():
            logger.debug("OP: %s", op.name)

        op_type_patterns = create_op_type_patterns_from_subgraph(transposed_conv_subgraph, additional_starting_ops=[])
        for pat in op_type_patterns:
            logger.debug(pat._op_type)

        # The pattern list consists of successive OpTypePattern() objects starting from the layer's input Op to
        # the output Op. Each OpTypePattern() becomes an input to the next OpTypePattern(). In the case of a Conv2D
        # with Bias, the last element in the pattern list is BiasAdd.
        self.assertEqual(op_type_patterns[-1]._op_type, 'BiasAdd')