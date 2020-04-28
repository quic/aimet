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
""" This file contains functions associated with matching the sub graph of Ops in the Session graph """


# pylint: disable=no-name-in-module
# pylint: disable=no-member
# Including above pylint disables since pylint complains about certain module members not found, when they actually
# are there.
import re
from typing import List, Dict, Set
from collections import OrderedDict
import tensorflow as tf
from tensorflow_core.contrib import slim # pylint: disable=unused-import
from tensorflow_core.contrib.quantize.python import graph_matcher
from aimet_tensorflow.common.module_identifier_matchers import ModuleIdentifierOpInfo
from aimet_tensorflow.common.operation import TfApi
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)

# Dictionary mapping names of ops to a tuple of input shape into the op, and the constructor for the op.
# Note that 'inputs' is the name of an input op that is instantiated with shape of the input shape.
# 'Constants' is the name of a constant op that is instantiated with shape of the input shape.
subgraph_constructors = {
    'Conv2D': ((1, 10, 10, 3), "tf.keras.layers.Conv2D(10, (1, 1), use_bias=False)(constants)"),
    'Conv2D_with_bias': ((1, 10, 10, 3), "tf.keras.layers.Conv2D(10, (1, 1), use_bias=True)(constants)"),
    'Dense': ([1, 10], "tf.keras.layers.Dense(10, activation=None)(constants)"),
    'BN_2': ((10, 10, 3,), "tf.keras.layers.BatchNormalization()(inputs, training=False)"),
    'BN_3': ((10, 10, 3,), "tf.keras.layers.BatchNormalization()(inputs, training=True)"),
    'BN_0': ((10, 10, 3,), "tf.keras.layers.BatchNormalization()(inputs)"),
    'BN_4': ((10, 10, 3,), "slim.batch_norm(inputs, is_training=True)"),
    'BN_1': ((10, 10, 3,), "slim.batch_norm(inputs, is_training=False)"),
    'BN_5': ((10, 10, 3,), "slim.batch_norm(inputs, is_training=is_training)"),
    'Softmax': ((1, 10), "slim.softmax(constants)"),
    'Softmax_with_unknown_shape': ((10,), "slim.softmax(inputs)"),
    'Dropout_0': ((1, 10, 10, 3), "tf.keras.layers.Dropout(rate=.4)(constants)"),
    'Dropout_1': ((1, 10, 10, 3), "slim.dropout(constants, keep_prob=.6)"),
    'Dropout_2': ((1, 10, 10, 3), "slim.dropout(constants, keep_prob=.6, is_training=True)"),
    'Flatten': ((10, 10, 3,), "tf.keras.layers.Flatten()(inputs)")
}


class Node:
    """

    A data class that holds an Op and its input Ops.
    Used for creating OpTypePattern() for an Op.
    """
    def __init__(self, op, inputs: List):
        self._op = op
        self._inputs = inputs

    @property
    def op(self) -> tf.Operation:
        """
        Return the Op
        :return: The Op
        """
        return self._op

    @property
    def inputs(self) -> List[tf.Operation]:
        """
        Retrun the inputs which are Ops
        :return:
        """
        return self._inputs

    def __repr__(self):
        """ Printable representation of the object. """
        return self._op.name + ' (%s)' % [inp_op.name for inp_op in self._inputs]


class SubGraphMatcher:
    """

    The SubGraphMatcher class encapsulates the functionality associated with individual Op level subgraphs.
    It creates OpTypePattern for those Ops in a model that have multiple associated internal Ops in the Session Graph.
    It uses these OpTypePattern objects to detect Ops in the Session Graph. It holds the detected Ops and their
    associated internal Ops. This association is ued when the ConnectedGraph is constructed for a model.
    """

    def __init__(self, graph: tf.Graph):
        """
        Initialize the SubGraphMatcher.

        :param graph: Session Graph associated with the model.
        """

        self._graph = graph

        # The  self._pattern_subgraph is a Dictionary of Dictionary that is applicable to all models and
        # NOT specific to a particular model. The outer Dictionary's key is the Op Type. Examples of "Op Type" are
        # 'Conv-2D', 'Dense" 'BN-1' and 'BN-2'. 'BN-1', 'BN-2' represent two of the multiple different variations of
        # the BatchNormalization Op. The inner Dictionary's keys are 'pattern' and 'subgraph'. The inner Dictionary
        # holds the OpTypePattern and the linear sequence of Op for each Op Type.
        self._pattern_subgraph = OrderedDict()

        # The self._op_subgraph is a Dictionary that is specific to a model under consideration.
        # For each Op in a specific model, it holds the subgraph which is a list of associated Ops.
        self._op_subgraph = OrderedDict()

        self.detect_ops_in_graph()

    # The functions below access protected members of TF classes.
    # pylint: disable=protected-access

    def detect_ops_in_graph(self):
        """
        Create OpTypePattern objects for individual Ops. Use the OpTypePattern objects to detect Ops in a
        specific Session Graph. Keep the detected Ops and their associated internal Ops.

        :return:
        """

        self.create_patterns_for_ops()
        all_op_patterns_list = [op_dict['pattern'] for op_dict in list(self._pattern_subgraph.values())]
        one_of_pattern_for_all_ops = graph_matcher.OneofPattern(all_op_patterns_list)
        layer_matcher = graph_matcher.GraphMatcher(one_of_pattern_for_all_ops)

        # Graph Match
        matched_op_set = set()  # Set to keep track of Ops that have been detected already.
        for match_result in layer_matcher.match_graph(self._graph):
            if match_result:

                # Detect Conv Ops
                conv_op_pattern = self._pattern_subgraph['Conv2D']['pattern']
                conv_op = match_result.get_op(conv_op_pattern)
                if conv_op:
                    if conv_op not in matched_op_set:
                        matched_op_set.add(conv_op)
                        self.update_internal_ops_for_the_detected_op(conv_op, match_result)

                # Detect BN Ops
                self.detect_bn_ops(match_result, matched_op_set)

                # Detect Flatten Ops
                flatten_op_pattern = self._pattern_subgraph['Flatten']['pattern']
                flatten_op = match_result.get_op(flatten_op_pattern)
                if flatten_op:
                    matched_op_set.add(flatten_op)
                    self.update_internal_ops_for_the_detected_op(flatten_op, match_result)

                # Detect Dense Ops
                dense_op_pattern = self._pattern_subgraph['Dense']['pattern']
                dense_op = match_result.get_op(dense_op_pattern)
                if dense_op:
                    if dense_op.inputs[0]._op not in matched_op_set:
                        matched_op_set.add(dense_op.inputs[0]._op)
                        self.update_internal_ops_for_the_detected_op(dense_op.inputs[0]._op, match_result)

                # Detect Softmax Ops
                softmax_op_pattern = self._pattern_subgraph['Softmax']['pattern']
                softmax_op = match_result.get_op(softmax_op_pattern)
                if softmax_op:
                    matched_op_set.add(softmax_op)
                    self.update_internal_ops_for_the_detected_op(softmax_op, match_result)

                self.detect_dropout_ops(match_result, matched_op_set)

    def detect_dropout_ops(self, match_result: graph_matcher.MatchResult, matched_op_set: Set):
        """
         Check the matched result for one of the many types of Dropout OPs.

        :param match_result: MatchResult object returned by TensorFlow GraphMatcher
        :param matched_op_set: Set of already detected Ops.
        :return:
        """

        # Detect Keras Dropout pattern
        dropout_0_pattern = self._pattern_subgraph['Dropout_0']['pattern']
        dropout_0_op = match_result.get_op(dropout_0_pattern)
        if dropout_0_op:
            my_dict = match_result._pattern_to_op_tensor.values()
            random_uniform_op_list = [md_op for md_op, _ in my_dict if md_op.type == 'RandomUniform']
            matched_op_set.add(random_uniform_op_list[0])
            self.update_internal_ops_for_the_detected_op(random_uniform_op_list[0], match_result)

        # Detect Slim Dropout pattern
        dropout_1_pattern = self._pattern_subgraph['Dropout_1']['pattern']
        dropout_1_op = match_result.get_op(dropout_1_pattern)
        if dropout_1_op:
            my_dict = match_result._pattern_to_op_tensor.values()
            random_uniform_op_list = [md_op for md_op, _ in my_dict if md_op.type == 'RandomUniform']
            matched_op_set.add(random_uniform_op_list[0])
            self.update_internal_ops_for_the_detected_op(random_uniform_op_list[0], match_result)
            input_mul_op = dropout_1_op.inputs[0].op
            matched_op_set.add(input_mul_op)
            self.update_internal_ops_for_the_detected_op(input_mul_op, match_result)

    def detect_bn_ops(self, match_result: graph_matcher.MatchResult, matched_op_set: Set):
        """
        Check the matched result for one of the many types of Batch Normalization OPs.

        :param match_result: MatchResult object returned by TensorFlow GraphMatcher
        :param matched_op_set: Set of already detected Ops.
        :return:
        """

        bn_4_op_pattern = self._pattern_subgraph['BN_4']['pattern']
        bn_4_op = match_result.get_op(bn_4_op_pattern)
        if bn_4_op:
            matched_op_set.add(bn_4_op)
            self.update_internal_ops_for_the_detected_op(bn_4_op, match_result)

        bn_1_op_pattern = self._pattern_subgraph['BN_1']['pattern']
        bn_1_op = match_result.get_op(bn_1_op_pattern)
        if bn_1_op:
            matched_op_set.add(bn_1_op)
            self.update_internal_ops_for_the_detected_op(bn_1_op, match_result)

        bn_5_op_pattern = self._pattern_subgraph['BN_5']['pattern']
        bn_5_op = match_result.get_op(bn_5_op_pattern)
        if bn_5_op:
            if bn_5_op.inputs[0]._op not in matched_op_set:
                matched_op_set.add(bn_5_op.inputs[0]._op)
                self.update_internal_ops_for_the_detected_op(bn_5_op.inputs[0]._op, match_result)
            if bn_5_op.inputs[1]._op not in matched_op_set:
                matched_op_set.add(bn_5_op.inputs[1]._op)
                self.update_internal_ops_for_the_detected_op(bn_5_op.inputs[1]._op, match_result)

        bn_2_op_pattern = self._pattern_subgraph['BN_2']['pattern']
        bn_2_op = match_result.get_op(bn_2_op_pattern)
        if bn_2_op:
            matched_op_set.add(bn_2_op)
            self.update_internal_ops_for_the_detected_op(bn_2_op, match_result)

        bn_3_op_pattern = self._pattern_subgraph['BN_3']['pattern']
        bn_3_op = match_result.get_op(bn_3_op_pattern)
        if bn_3_op:
            matched_op_set.add(bn_3_op)
            self.update_internal_ops_for_the_detected_op(bn_3_op, match_result)

        bn_0_op_pattern = self._pattern_subgraph['BN_0']['pattern']
        bn_0_op = match_result.get_op(bn_0_op_pattern)
        if bn_0_op:
            if bn_0_op.inputs[0]._op not in matched_op_set:
                matched_op_set.add(bn_0_op.inputs[0]._op)
                self.update_internal_ops_for_the_detected_op(bn_0_op.inputs[0]._op, match_result)
            if bn_0_op.inputs[1]._op not in matched_op_set:
                matched_op_set.add(bn_0_op.inputs[1]._op)
                self.update_internal_ops_for_the_detected_op(bn_0_op.inputs[1]._op, match_result)

    def update_internal_ops_for_the_detected_op(self, op: tf.Operation, match_result: graph_matcher.MatchResult):
        """
        For the given Op, obtain all the associated internal Ops and update the Op-Subgraph dictionary.

        :param op: The Op for which the associated Ops must be obtained.
        :param match_result: The match result associated with the Op.
        :return:
        """

        internal_ops_list = []  # Place holder for the list of internal Ops associated with the detected Op.

        # The patter_to_op_tensor is a dictionary of Ops and Tensors encountered for a pattern while matching.
        op_tensor_dict = match_result._pattern_to_op_tensor.values()
        ops_list = [internal_op for internal_op, _ in op_tensor_dict]

        # The Ops_list also contains input Ops. Since only the internal ops associated with detected Op is needed,
        # skip the input Ops. This is done by making sure that the input Op's Parent Op is not in the ops_list.
        for int_op in ops_list:
            if int_op.inputs:
                parent_op = int_op.inputs[0].op
                if parent_op in ops_list:
                    internal_ops_list.append(int_op)

        self._op_subgraph[op] = internal_ops_list

    # pylint: enable=protected-access

    def create_patterns_for_ops(self):
        """
        Create OpTypePattern for all the required Ops and store them in Pattern-Subgraph dictionary.
        """
        for op_type, (input_shape, constructor_string) in subgraph_constructors.items():
            subgraph = create_subgraph_for_op(input_shape, constructor_string)
            patterns = create_op_type_pattens_from_subgraph(subgraph)
            self._pattern_subgraph[op_type] = {'pattern': patterns[-1], 'subgraph': subgraph}

    def match_op(self, op: tf.Operation):
        """
        Check if the given Op is in the  list of already detected Ops for the session graph.
        If found, return the list of  internal Ops associated with the Op.

        :param op: Op under consideration.
        :return: If the Op is in the list of detected Ops, return Turue and a list of associated Ops. If not found,
                 return False and an empty list.
        """

        if op in self._op_subgraph:
            return True, self._op_subgraph[op]

        return False, []

    def match_conv2d_dense_type_ops(self, op_to_module_dict: Dict[tf.Operation, ModuleIdentifierOpInfo],
                                    op_info: ModuleIdentifierOpInfo) -> bool:
        """
        Matcher for Conv2d and Dense type ops
        :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
        same module will be mapped to the same ModuleIdentifierOpInfo object.
        :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
        belong to
        :return: True if a valid match was made, False otherwise
        """

        # Begin at either the conv2d op or the matmul op
        op = op_info.tf_op

        result, op_sub_graph = self.match_op(op)
        logger.debug("match_conv2d_dense_type_ops() result: %s, Ops: %s", result, op_sub_graph)
        # Works for Conv2D but Doesn't work for DepthwiseConv2D. debugging.
        # By setting to True while debugging. This function is the same as the module id matcher.
        result = True

        if result:

            if op.type == 'MatMul':
                op_info.op_type = 'Dense'
            op_info.module_name = op.name
            op_to_module_dict[op] = op_info
            if len(op.outputs) > 1:
                logger.error('Not expecting Conv2D to ever have more than one output tensor')
                assert False
            if len(op.outputs[0].consumers()) > 1:
                # Hit end of Conv2D if output of current op goes to more than one child op
                return True
            if not op.outputs[0].consumers():
                # Conv op does not lead to any op. This can happen if this Conv op was winnowed, and this is a dangling
                # conv op with no bias. Still represent this as an Op in the Connected Graph.
                return True
            if op.outputs[0].consumers()[0].type == 'BiasAdd':
                op_to_module_dict[op.outputs[0].consumers()[0]] = op_info
                return True

        logger.debug("Unable to match Conv2D/Dense Op: %s", op.name)
        return False

    def match_fused_batchnorm(self, op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
        """
        Check if the FusedBatchNormV3 Op was detected as part of the session graph.
        If matched, fill in Op related information.

        :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
        same module will be mapped to the same ModuleIdentifierOpInfo object.
        :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
        belong to
        :return: True if a valid match was made, False otherwise
        """
        # Begin at op of type FusedBatchNorm, and try to match pattern 1 (uses placeholder tensor for switching between
        # training and non training) to op_to_module_dict
        op = op_info.tf_op
        is_training = op.get_attr('is_training')

        result, op_sub_graph = self.match_op(op)
        logger.debug("match_fused_batchnorm() result: %s, Ops: %s", result, op_sub_graph)

        if result:

            # This fusedbatchnorm uses a placeholder tensor for determining whether it is in training mode or not
            # op_info.add_attribute('training', training_tensor.name)
            # FusedBatchNorms of this type always end with /cond/FusedBatchNorm_1 in the name
            # Everything preceding the cond is the scope name
            match_name_1 = re.match('(.+)/cond/FusedBatchNormV3_1', op.name)
            if match_name_1:
                op_info.module_name = match_name_1.group(1)
                fill_batch_norm_pattern1_info(op_info, op_sub_graph)
            else:
                match_name_2 = re.match('(.+)/FusedBatchNormV3', op.name)
                if match_name_2:
                    op_info.module_name = match_name_2.group(1)
                    op_info.add_attribute('training', is_training)

            # Add the Op first.
            op_to_module_dict.update({op: op_info})

            # Add the associated Ops.
            for sub_graph_op in op_sub_graph:
                op_to_module_dict.update({sub_graph_op: op_info})

            return True

        logger.debug("Unable to match BatchNormalization Op: %s", op.name)
        return False

    def match_flatten(self, op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
        """
         Check if the Flatten Op was detected as part of the session graph. If matched, fill in Op related information.

        :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
        same module will be mapped to the same ModuleIdentifierOpInfo object.
        :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
        belong to
        :return: True if a valid match was made, False otherwise
        """

        op = op_info.tf_op
        op_info.op_type = "Flatten"
        op_info.tf_api = TfApi.slim  # Assume this is a Slim Op type. Check and if true, change to Keras Op type.

        result, op_sub_graph = self.match_op(op)
        logger.debug("match_flatten() result: %s, Ops: %s", result, op_sub_graph)

        if result:

            try:
                # Add the Op first.
                op_to_module_dict[op] = op_info

                # Add the associated Ops.
                for sub_graph_op in op_sub_graph:
                    op_to_module_dict.update({sub_graph_op: op_info})

                pack_op = op.inputs[1].op
                strided_slice_op = pack_op.inputs[0].op
                shape_op = strided_slice_op.inputs[0].op

                if shape_op.inputs:
                    op_info.tf_api = TfApi.keras
                    op_to_module_dict[pack_op] = op_info
                    op_to_module_dict[strided_slice_op] = op_info
                    op_to_module_dict[shape_op] = op_info

                return True
            except:     # pylint: disable=bare-except
                return False

        logger.debug("Unable to match Flatten Op: %s", op.name)
        return False

    def match_softmax(self, op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
        """
        Check if the Softmax Op was detected as part of the session graph. If matched, fill in Op related information.

        :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
        same module will be mapped to the same ModuleIdentifierOpInfo object.
        :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
        belong to
        :return: True if a valid match was made, False otherwise
        """

        # Begin at op of type softmax and try to match to tf slim softmax pattern
        op = op_info.tf_op
        op_info.tf_api = TfApi.slim

        result, op_sub_graph = self.match_op(op)
        logger.debug("match_softmax() result: %s, Ops: %s", result, op_sub_graph)

        if result:
            reshape = op.inputs[0].op
            if reshape.type != "Reshape":
                return False
            reshape_1 = op.outputs[0].consumers()[0]
            op_to_module_dict.update({op: op_info,
                                      reshape: op_info,
                                      reshape_1: op_info})
            if len(reshape_1.inputs) == 2 and reshape_1.inputs[1].op.type == "Shape":
                op_to_module_dict.update({reshape_1.inputs[1].op: op_info})

            return True
        logger.debug("Unable to match Softmax Op: %s", op.name)
        return False

    def match_dropout_1(self, op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
        """
        Check if the Dropout Op matching Dropout pattern 1 was detected as part of the session graph.
        If matched, fill in Op related information.

        :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
        same module will be mapped to the same ModuleIdentifierOpInfo object.
        :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
        belong to
        :return: True if a valid match was made, False otherwise
        """

        # Begin at op of type RandomUniform and try to match to dropout pattern 1 (keras pattern)
        op = op_info.tf_op

        result, op_sub_graph = self.match_op(op)
        logger.debug("match_dropout_1 result: %s, Ops: %s", result, op_sub_graph)

        if result:
            # Add ops to the op to module dict
            op_info.op_type = "Dropout"
            greater_equal_op = [op for op in op_sub_graph if op.type == 'GreaterEqual']
            op_info.add_attribute('rate_tensor', greater_equal_op[0].inputs[1])
            merge_op = [op for op in op_sub_graph if op.type == 'Merge']
            if merge_op:
                match_name = re.match("(.+)/cond", merge_op[0].name)
                if match_name:
                    op_info.module_name = match_name.group(1)
            else:
                match_name = re.search("(.+)/random_uniform/RandomUniform", op.name)
                if match_name:
                    op_info.module_name = match_name.group(1)

            # Add the associated Ops.
            for sub_graph_op in op_sub_graph:
                op_to_module_dict.update({sub_graph_op: op_info})

            return True
        logger.debug("Unable to match dropout_1  Op: %s", op.name)
        return False

    def match_dropout_2(self, op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
        """
        Check if the Dropout Op matching Dropout pattern 2 was detected as part of the session graph.
        If matched, fill in Op related information.

        :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
        same module will be mapped to the same ModuleIdentifierOpInfo object.
        :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
        belong to
        :return: True if a valid match was made, False otherwise
        """

        op = op_info.tf_op

        result, op_sub_graph = self.match_op(op)
        logger.debug("match_dropout_2 result: %s, Ops: %s", result, op_sub_graph)

        if result:
            # Add ops to the op to module dict
            op_info.op_type = "Dropout"
            op_info.tf_api = TfApi.slim
            greater_equal_op = [op for op in op_sub_graph if op.type == 'GreaterEqual']
            op_info.add_attribute('rate_tensor', greater_equal_op[0].inputs[1])
            random_uniform_op_list = [op for op in op_sub_graph if op.type == 'RandomUniform']
            match_name = re.search("(.+)/random_uniform/RandomUniform", random_uniform_op_list[0].name)
            if match_name:
                op_info.module_name = match_name.group(1)
            mul = op.outputs[0].consumers()[0]
            op_to_module_dict.update({op: op_info,
                                      mul: op_info})
            return True

        logger.debug("Unable to match dropout_2  Op: %s", op.name)
        return False


def create_op_type_pattens_from_subgraph(subgraph: tf.Graph) -> List[graph_matcher.OpTypePattern]:
    """
    Create and return a list of TensorFlow OpTypePattern objects for the given subgraph.
    The OpTypepatterns() are created in sequence from the input to the output of the subgraph.
    The last OpTypepattern() object in the returned list is for the Op under consideration.

    :param subgraph: The subgraph of an Op for which OpTypePattern is created.
    :return: List of OpTypePattern()
    """

    starting_op_names = ['aimet_input', 'aimet_constant']
    ending_op_names = ['aimet_identity']
    ops_from_ending_ops = set()
    node_list = []

    # For each ending op, do a Depth First Search (DFS) upwards and add all parent ops to "ops_from_ending_ops" set
    for name in ending_op_names:
        op = subgraph.get_operation_by_name(name)
        queue = [op]
        while queue:
            curr_op = queue.pop()
            ops_from_ending_ops.add(curr_op)
            input_ops = []
            for inp in curr_op.inputs:
                if inp.op not in ops_from_ending_ops:
                    queue.append(inp.op)
                if curr_op not in ending_op_names:
                    input_ops.append(inp)
            if curr_op.name not in ending_op_names and curr_op.name not in starting_op_names:
                add_node_to_list(curr_op, input_ops, node_list)

    # DFS is done bottom up.
    #   Reason:
    #       If we do top down DFS, it becomes necessary to indicate a starting Op other than well known 'aimet_input'
    #       For a Conv2D, for top down DFS, if only 'aimet_input' is given as starting Op for DFS, the kernel
    #       input sub-graph for the Conv2D is missed.
    #       This is not an issue for bottom up DFS since bottom up DFS looks at all inputs.
    # For building OpTypePattern() sequence, the dependent OpTypePattern() must be build first before using that
    # OpTypePattern() as an input in the next OpTypePattern()
    # For this purpose, the pattern list is reversed.
    node_list.reverse()
    sub_patterns = get_op_type_patterns(node_list)

    return sub_patterns


def get_op_type_patterns(node_list: List[Node]) -> List[graph_matcher.OpTypePattern]:
    """
    From the list of Nodes, create the OpTypePattern()
    :param node_list: List of Nodes that are specific to an Op.
    :return: the list of OpTypePattern() objects that are specific to an Op.
    """

    sub_patterns = []  # A list that holds all the OpTypePattern objects created for a specific Op
    for i, node in enumerate(node_list):
        node_op_type = node.op.type
        if node.inputs:
            # The list of input ops is used to create the OpTypePattern for the current Op.
            input_ops_list = get_op_type_patterns_for_input_ops(node, i, sub_patterns)
            sub_patterns.append(graph_matcher.OpTypePattern(str(node_op_type), name=node.op.name,
                                                            inputs=input_ops_list))
        else:
            sub_patterns.append(graph_matcher.OpTypePattern(str(node_op_type), name=node.op.name))

    return sub_patterns


def create_subgraph_for_op(input_shape: tuple, op_string: str) -> tf.Graph:
    """
    Create and return the TensorFlow session graph for a single Op.
    A well known input named "aimet_input" and a well known output named "aimet_identity" are used
    along with the Op for the purposes of traversing the graph for the Op.

    :param input_shape: Input shape to be used for the input to the Op.
    :param op_string: The string that contains the TensorFlow syntax for the Op
    :param bn_training_flag: True of False. Applies only to BatchNormalization Ops.
    :return: The subgraph for the Op.
    """
    sess = tf.Session(graph=tf.Graph())
    with sess.graph.as_default():
        with tf.device('/cpu:0'):
            # Use inputs when the batch size can be unknown. Otherwise use constant for an input with known shape.
            # Use is_training when the op requires a boolean tensor to be passed in to toggle training mode.
            # pylint: disable=unused-variable
            inputs = tf.keras.Input(shape=input_shape, name='aimet_input')
            constants = tf.constant(1, shape=input_shape, dtype=tf.float32, name='aimet_constant')
            is_training = tf.placeholder_with_default(tf.constant(True), shape=(), name='is_training')
            x = eval(op_string)  # pylint: disable=eval-used
            x = tf.identity(x, name='aimet_identity')
        init = tf.global_variables_initializer()
    sess.run(init)

    # Uncomment the following line to use TensorBoard.
    # _ = tf.summary.FileWriter('./subgraph', sess.graph)

    return sess.graph


def get_op_type_patterns_for_input_ops(node: Node, node_list_index: int,
                                       sub_patterns: List[graph_matcher.OpTypePattern]) \
        -> List[graph_matcher.OpTypePattern]:
    """
    For Ops with multiple inputs, return the list of OpTypePatterns corresponding to the Op's input Ops.

    :param node: The Node that is holding an Op and its input Ops
    :param node_list_index: The Node's index in the node_list
    :param sub_patterns A list where created OpTypePatten objects are added.
    :return: List of OpTypePatterns that correspond to the input Ops
    """

    inp_op_type_patterns_list = []
    for _, inp in enumerate(node.inputs):
        # if inp.op.type == 'Placeholder':
        if inp.op.type in ['Placeholder', 'Const']:
            # This sub-graph for the Op was created to always with an input of tf.Keras.Input Type = Placeholder) and
            # an output Op of tf.identity(Type = Identity). A give Op under consideration would receive it's input
            # from any other Op preceding it. For OpType pattern(), this is represented as a '*'
            inp_op_type_patterns_list.append('*')
        else:
            # When the Op has multiple inputs, check all the inputs and get the Node index in the seq_list
            # plain_index = find_index_of_node_in_node_list(inp.op.type, seq_list, seq_list_index)
            op_index = find_input_op_index_in_list_of_op_type_patterns(inp.op, node_list_index, sub_patterns)

            if op_index:
                inp_op_type_patterns_list.append(sub_patterns[op_index])
            else:
                # None means that we are dealing with an input for which a OpTypePattern() has not been created.
                inp_op_type_patterns_list.append('*')

    return inp_op_type_patterns_list


def add_node_to_list(op: tf.Operation, input_ops: List[tf.Operation], node_list: List):
    """
    Create a Node object and add it to the node list.

    :param op: Op to be added to the list
    :param input_ops: List of input Ops of the Op
    :param node_list: Node list to which Node objects are added
    :return:
    """
    node = Node(op, input_ops)
    node_list.append(node)


def find_input_op_index_in_list_of_op_type_patterns(op: tf.Operation, starting_index: int,
                                                    sub_patterns: List[graph_matcher.OpTypePattern]):
    """
    For every Node in the list of Nodes, an OpTypePattern() is created. Starting from the input of the model to the
    output, when creating the OpTypePattern() for an Op, the OpTypePattern for the Op's inputs would have been created
    already. This function finds the index of the input OpTypePattern() for a given "input Op" of an Op.

    :param op: The Op for which the the input Op's index in the list of OpTypePatterns(sub_patterns) must be found
    :param starting_index: The index of the Op
    :param sub_patterns: List of OpTypePatterns that have been already created.
    :return:
    """

    if not sub_patterns:
        # No OpTypePattern objects have been created yet.
        return None

    if starting_index == 0:
        # starting_index is the index of the Op in sub_patterns.
        # If it is 0, there is no previously created sub_patterns to consider.
        return None

    m = starting_index - 1  # Since starting_index is for the node, consider the sub_pattern just before it. Hence -1.
    while m != 0:
        pattern = sub_patterns[m]
        if op.type == pattern._op_type and op.name == pattern._name:  # pylint: disable=protected-access
            return m
        m = m - 1


def fill_batch_norm_pattern1_info(op_info: ModuleIdentifierOpInfo, op_sub_graph: List[tf.Operation]):
    """
    Fill in additional information associated with FusedBatchNorm of pattern 1.

    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
                    belong to
    :param op_sub_graph:  List of Ops associated with the Op getting matched.
    :return:
    """

    pred_id_op = [pred_op for pred_op in op_sub_graph if 'pred_id' in pred_op.name]
    if pred_id_op:
        if pred_id_op[0].inputs:
            training_tensor = pred_id_op[0].inputs[0]
            op_info.add_attribute('training', training_tensor.name)
