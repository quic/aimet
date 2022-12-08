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
# pylint: disable=unused-import
# Including above pylint disables since pylint complains about certain module members not found, when they actually
# are there.
import re
from typing import List, Dict, Set, Union
from collections import OrderedDict
import tensorflow as tf
from packaging import version

from aimet_common.utils import AimetLogger
from aimet_tensorflow.common.sub_graph_matcher_op_templates import op_type_templates
from aimet_tensorflow.quantize import graph_matcher
from aimet_tensorflow.utils.common import get_valid_ops

if version.parse(tf.version.VERSION) < version.parse("2.0"):
    from tensorflow_core.contrib import slim # pylint: disable=import-error
else:
    import transformers.activations_tf

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)


# ------------------------------------------------------------------------------------------ #
# Code below is for import time generation of OpTypePattern and sub-graphs corresponding
# to reference_op_templates.
# These are used by SubGraphMatcher to perform module detection that helps generate an
# intermediate representation of TF graph.
#  -- Begin--
# ------------------------------------------------------------------------------------------- #

def store_op_type_pattern_info(subgraph: tf.Graph, additional_starting_ops: List,
                               pattern_op_type: str, info_dict: Dict):
    """
    utility to dynamically create op type patterns based on template types in sub_graph_op_templates
    and update pattern to op type dict.
    :param subgraph: subgraph with op pattern
    :param additional_starting_ops: Additional starting points for identifying valid ops to match with.
    Valid ops are defined as ops which can be traversed with both a dfs any input op as well as dfs backwards from
    any output op.Additional starting ops can be used when simply using default input and output ops gives a
    pattern easily matched by individual ops that are not actually of the desired matched type
    (BN_non_fused_keras_with_training_False would be matched with only a mul -> add, for example)
    :param op_type: string name for op pattern represented by sub-graph
    :return:
    """

    patterns = create_op_type_patterns_from_subgraph(subgraph, additional_starting_ops)
    reference_op_pattern_info_dict[pattern_op_type] = {'pattern': patterns[-1], 'subgraph': subgraph,
                                                       'length': len(patterns),
                                                       'op_type': info_dict['op_type'],
                                                       'module_regex': info_dict['module_regex'],
                                                       'associated_op_regex': info_dict['associated_op_regex']}
    for pattern in patterns:
        pattern_to_op_type[pattern] = pattern_op_type


def create_patterns_for_ops():
    """
    Create OpTypePattern for all the required Ops and store them in Pattern-Subgraph dictionary.
    """

    for pattern_op_type, info_dict in op_type_templates.items():
        input_shape = info_dict['input_shape']
        constructor_string = info_dict['constructor']
        additional_starting_ops = info_dict.get('additional_starting_ops', [])
        op_type = info_dict['op_type']
        supported_tf_versions = info_dict['supported_tf_versions']
        if version.parse(tf.version.VERSION).major not in supported_tf_versions:
            continue
        # generate two  sub-graphs patterns for recurrent ops one with keras input and
        # the other with placeholder type
        subgraph_keras_input = create_subgraph_for_op_default(input_shape, constructor_string)
        store_op_type_pattern_info(subgraph_keras_input, additional_starting_ops, pattern_op_type, info_dict)

        # we need a unique key for pattern with placeholder type input
        # skip BNs for placeholder input types
        if op_type not in ['FusedBatchNormV3', 'BatchNorm']:
            pattern_op_type = pattern_op_type+'_placeholder'
            subgraph_placeholder = create_subgraph_for_op_with_placeholder(input_shape, constructor_string,
                                                                           op_type)
            store_op_type_pattern_info(subgraph_placeholder, additional_starting_ops, pattern_op_type, info_dict)


def get_module_name(module_regex_list: List[str], ops_list: List[tf.Operation]) -> str:
    """
    Extract module name for the matched ops by matching with a given regex pattern.
    :param module_regex_list: List of regex patterns to match with
    :param ops_list: List of matched ops
    :return: String representing the module name for the set of matched ops. If no name is successfully matched, return
    the name of the first op in the list.
    """
    for module_regex in module_regex_list:
        for op in ops_list:
            match_name = re.match(module_regex, op.name)
            if match_name:
                return match_name.group(1)
    logger.warning('Unable to identify module name, using name of first op as module name: %s', ops_list[0].name)
    return ops_list[0].name


def get_associated_op(associated_op_regex_list: List[str], ops_list: List[tf.Operation]) -> Union[None, tf.Operation]:
    """
    Identify the op to associate with the module representing the set of matched ops. Use a regex pattern to match a
    particular op.
    :param associated_op_regex_list: Regex pattern to match with
    :param ops_list: List of matched ops
    :return: Tf op that was matched by the regex. If no op is matched, use first op in ops_list.
    """
    for associated_op_regex in associated_op_regex_list:
        for op in ops_list:
            match_name = re.search(associated_op_regex, op.name)
            if match_name:
                return op
    logger.warning('Unable to identify associated op of module, setting first op in ops_list as associated op.')
    # Ops list is in reverse order with the first index being the last op in the sequence.
    return ops_list[0]


def get_internal_ops_for_pattern(match_result: graph_matcher.MatchResult) -> List[tf.Operation]:
    """
    Get all the ops corresponding to the matched pattern.
    :param match_result: Match result from graph matcher
    :return: List of tf ops corresponding to the matched pattern
    """
    internal_ops_list = []  # Place holder for the list of internal Ops associated with the detected Op.

    # The patter_to_op_tensor is a dictionary of Ops and Tensors encountered for a pattern while matching.
    # pylint: disable=protected-access
    op_tensor_dict = match_result._pattern_to_op_tensor.values()
    ops_list = [internal_op for internal_op, _ in op_tensor_dict]

    # The Ops_list also contains input Ops. Since only the internal ops associated with detected Op is needed,
    # skip the input Ops. This is done by making sure that the input Op's Parent Op is not in the ops_list.
    for int_op in ops_list:
        if int_op.inputs:
            parent_op = int_op.inputs[0].op
            if parent_op in ops_list:
                internal_ops_list.append(int_op)
    return internal_ops_list


def create_op_type_patterns_from_subgraph(subgraph: tf.Graph, additional_starting_ops: List[str]) -> \
        List[graph_matcher.OpTypePattern]:
    """
    Create and return a list of TensorFlow OpTypePattern objects for the given subgraph.
    The OpTypepatterns() are created in sequence from the input to the output of the subgraph.
    The last OpTypepattern() object in the returned list is for the Op under consideration.

    :param subgraph: The subgraph of an Op for which OpTypePattern is created.
    :param additional_starting_ops: Additional starting points for identifying valid ops to match with.  Valid ops are
    defined as ops which can be traversed with both a dfs any input op as well as dfs backwards from any output op.
    Additional starting ops can be used when simply using default input and output ops gives a pattern easily matched by
    individual ops that are not actually of the desired matched type (BN_non_fused_keras_with_training_False would be
    matched with only a mul -> add, for example)
    :return: List of OpTypePattern()
    """

    starting_op_names = ['aimet_input', 'aimet_constant', 'is_training'] + additional_starting_ops
    ending_op_names = ['aimet_identity']
    ops_from_ending_ops = set()
    op_list = []
    valid_ops = get_valid_ops(subgraph, starting_op_names=starting_op_names, ending_op_names=ending_op_names)

    # DFS is done bottom up.
    #   Reason:
    #       If we do top down DFS, it becomes necessary to indicate a starting Op other than well known 'aimet_input'
    #       For a Conv2D, for top down DFS, if only 'aimet_input' is given as starting Op for DFS, the kernel
    #       input sub-graph for the Conv2D is missed.
    #       This is not an issue for bottom up DFS since bottom up DFS looks at all inputs.
    # For building OpTypePattern() sequence, the dependent OpTypePattern() must be build first before using that
    # OpTypePattern() as an input in the next OpTypePattern()
    def dfs_upwards(curr_op):
        """ Function to perform DFS upwards starting at curr_op """
        if curr_op in ops_from_ending_ops or curr_op.name in starting_op_names:
            # Do not process curr_op if we have seen it before, or if it is one of the starting ops
            return
        ops_from_ending_ops.add(curr_op)
        # List to hold inputs to curr_op
        input_ops = []
        for inp in curr_op.inputs:
            input_ops.append(inp.op)
            dfs_upwards(inp.op)
        if curr_op.name not in ending_op_names and curr_op in valid_ops:
            op_list.append(curr_op)

    for name in ending_op_names:
        op = subgraph.get_operation_by_name(name)
        dfs_upwards(op)

    sub_patterns = get_op_type_patterns(op_list)

    return sub_patterns


def get_op_type_patterns(op_list: List[tf.Operation]) -> List[graph_matcher.OpTypePattern]:
    """
    From the list of ops, create the OpTypePattern()
    :param op_list: List of ops to create patterns for
    :return: the list of OpTypePattern() objects that are specific to an Op.
    """

    sub_patterns = []  # A list that holds all the OpTypePattern objects created for a specific Op
    for i, op in enumerate(op_list):
        if op.inputs:
            # The list of input ops is used to create the OpTypePattern for the current Op.
            input_ops_list = get_op_type_patterns_for_input_ops(op, i, sub_patterns)
            sub_patterns.append(graph_matcher.OpTypePattern(str(op.type), name=op.name,
                                                            inputs=input_ops_list))
        else:
            sub_patterns.append(graph_matcher.OpTypePattern(str(op.type), name=op.name))

    return sub_patterns


def create_subgraph_for_op_with_placeholder(input_shape: tuple, op_string: str, op_type: str) -> tf.Graph:
    """
    Create and return the TensorFlow session graph for a single Op either with placeholder input type
    A well known input named "aimet_input" and a well known output named "aimet_identity" are used
    along with the Op for the purposes of traversing the graph for the Op.
    :param input_shape: Input shape to be used for the input to the Op.
    :param op_string: The string that contains the TensorFlow syntax for the Op
    :param op_type: string description of op type
    :return: The subgraph for the Op.
    """

    sub_graph = tf.Graph()
    with sub_graph.as_default():
        with tf.device('/cpu:0'):
            # Use inputs when the batch size can be unknown. Otherwise use constant for an input with known shape.
            # Use is_training when the op requires a boolean tensor to be passed in to toggle training mode.
            # pylint: disable=unused-variable
            # convert tuple of input_shape  provided for Op into a list of shape for placeholder type

            input_shape_list = list(input_shape)

            # Also, if we have only one dimension input shape, we need to make it a 2D shape for placeholder input
            # unlike other modules, below op types expect 3D shape for placeholder input
            if len(input_shape) == 1 or op_type in ['SimpleRNN', 'LSTM', 'Upsample2D', 'GlobalMaxPool2D',
                                                    'Conv2DTranspose']:
                input_shape_list.insert(0, 1)
                input_shape = tuple(input_shape_list)
                inputs = tf.compat.v1.placeholder(tf.float32, input_shape, name='aimet_input')
            else:
                inputs = tf.compat.v1.placeholder(tf.float32, input_shape, name='aimet_input')

            constants = tf.compat.v1.placeholder(tf.float32, input_shape, name='aimet_constant')
            is_training = tf.compat.v1.placeholder_with_default(tf.constant(True), shape=(), name='is_training')
            x = eval(op_string)  # pylint: disable=eval-used
            x = tf.identity(x, name='aimet_identity')

    return sub_graph


def create_subgraph_for_op_default(input_shape: tuple, op_string: str) -> tf.Graph:
    """
    Create and return the TensorFlow session graph for a single Op with keras or constant input types.
    A well known input named "aimet_input" and a well known output named "aimet_identity" are used
    along with the Op for the purposes of traversing the graph for the Op.

    :param input_shape: Input shape to be used for the input to the Op.
    :param op_string: The string that contains the TensorFlow syntax for the Op
    :param bn_training_flag: True of False. Applies only to BatchNormalization Ops.
    :return: The subgraph for the Op.
    """

    sub_graph = tf.Graph()
    with sub_graph.as_default():
        with tf.device('/cpu:0'):
            # Use inputs when the batch size can be unknown. Otherwise use constant for an input with known shape.
            # Use is_training when the op requires a boolean tensor to be passed in to toggle training mode.
            # pylint: disable=unused-variable
            inputs = tf.keras.Input(shape=input_shape, name='aimet_input')
            constants = tf.constant(1, shape=input_shape, dtype=tf.float32, name='aimet_constant')
            is_training = tf.compat.v1.placeholder_with_default(tf.constant(True), shape=(), name='is_training')
            x = eval(op_string)  # pylint: disable=eval-used
            x = tf.identity(x, name='aimet_identity')

    return sub_graph


def get_op_type_patterns_for_input_ops(op: tf.Operation, op_list_index: int,
                                       sub_patterns: List[graph_matcher.OpTypePattern]) \
        -> List[graph_matcher.OpTypePattern]:
    """
    For Ops with multiple inputs, return the list of OpTypePatterns corresponding to the Op's input Ops.

    :param op: Tf operation to get pattern for
    :param op_list_index: The op's index in the op_list
    :param sub_patterns A list where created OpTypePatten objects are added.
    :return: List of OpTypePatterns that correspond to the input Ops
    """

    inp_op_type_patterns_list = []
    for _, inp in enumerate(op.inputs):
        if inp.op.type in ['Placeholder', 'Const']:
            # This sub-graph for the Op was created to always with an input of tf.Keras.Input Type = Placeholder) and
            # an output Op of tf.identity(Type = Identity). A give Op under consideration would receive it's input
            # from any other Op preceding it. For OpType pattern(), this is represented as a '*'
            inp_op_type_patterns_list.append('*')
        else:
            # When the Op has multiple inputs, check all the inputs and get the op index in the seq_list
            op_index = find_input_op_index_in_list_of_op_type_patterns(inp.op, op_list_index, sub_patterns)

            if op_index is not None:
                inp_op_type_patterns_list.append(sub_patterns[op_index])
            else:
                # None means that we are dealing with an input for which a OpTypePattern() has not been created.
                inp_op_type_patterns_list.append('*')

    return inp_op_type_patterns_list


def find_input_op_index_in_list_of_op_type_patterns(op: tf.Operation, starting_index: int,
                                                    sub_patterns: List[graph_matcher.OpTypePattern]) \
        -> Union[int, None]:
    """
    For every op, an OpTypePattern() is created. Starting from the input of the model to the
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

    m = starting_index - 1  # Since starting_index is for the op, consider the sub_pattern just before it. Hence -1.
    while m >= 0:
        pattern = sub_patterns[m]
        if op.type == pattern._op_type and op.name == pattern._name:  # pylint: disable=protected-access
            return m
        m = m - 1


class ModuleIdentifierOpInfo:
    """ Class for summarizing information regarding a tf operation """
    def __init__(self, module_name, op_type, tf_op, pattern_type: str = None, internal_ops: List[tf.Operation] = None):
        """
        Initialize the ModuleIdentifierOpInfo class.

        :param module_name: Module name associated with the module
        :param op_type: Op type associated with the module (this will be the type shown in ConnectedGraph)
        :param tf_op: Main op associated with the module (may be different than the op corresponding to this object
            in the op_to_module_dict)
        :param pattern_type: Pattern used to generate the graph that matched the op corresponding to the
            ModuleIdentifierOpInfo object. Only used in subgraph matcher
        :param internal_ops : List of internal tf operations associated with the module
        """
        self._module_name = module_name
        self._op_type = op_type
        self._tf_op = tf_op
        # Pattern type is only used in subgraph matcher, and contains info about which pattern (Conv2D_keras,
        # Conv2D_keras_with_bias, etc. was used to match the op for this op info class.
        self._pattern_type = pattern_type
        self._attributes = {}
        self._internal_ops = internal_ops

    @property
    def module_name(self):
        """ Returns the module name corresponding to this operation. """
        return self._module_name

    @module_name.setter
    def module_name(self, module_name):
        """ Sets the module name of an Operation. """
        self._module_name = module_name

    @property
    def op_type(self):
        """ Returns the op type of the module corresponding to this operation. """
        return self._op_type

    @op_type.setter
    def op_type(self, op_type):
        """ Sets the op type """
        self._op_type = op_type

    @property
    def tf_op(self):
        """ Returns the tf op for the module corresponding to this operation. """
        return self._tf_op

    @property
    def pattern_type(self):
        """ Returns the pattern type corresponding to this operation. """
        return self._pattern_type

    @property
    def internal_ops(self):
        """ Returns the internal ops for the module corresponding to this operation. """
        return self._internal_ops

    def add_attribute(self, attribute_name: str, attribute):
        """ Set an attribute of the module identifier op info """
        self._attributes[attribute_name] = attribute

    def get_attributes(self):
        """ Return the attributes dictionary """
        return self._attributes


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


# The reference_op_pattern_info_dict is a nested Dictionary that is applicable to all models and
# NOT specific to a particular model. The Dictionary's outer key is the Op Pattern Type identifier string
# (derived from template dictionary keys,reference_op_templates).
# Examples are
# 'Conv-2D', 'Dense" 'BN-1' and 'BN-2'. 'BN-1', 'BN-2' represent two of the multiple different variations of
# the BatchNormalization Op.
# The values are key-value pairs of info related to pattern like required by module identifier:
# 'pattern', 'subgraph', 'op_type', 'module_regex' so on.
# 'pattern' is OpTypePattern and 'subgraph' is the linear sequence of Op for each Op Type.
reference_op_pattern_info_dict = OrderedDict()

# Dictionary below is used to optimize reverse look-up on matched pattern during module detection
# maps pattern to it's op type
pattern_to_op_type = {}

# create patterns / sub-graphs for module detection corresponding to op templates types
create_patterns_for_ops()

# ---------------------------------------------------------------------------------------------------------- #
# import time generation of TF OpTypePattern and sub-graphs used for module identification
# -- End --
# ---------------------------------------------------------------------------------------------------------- #


class SubGraphMatcher:
    """

    The SubGraphMatcher class encapsulates the functionality associated with individual Op level subgraphs.
    It creates OpTypePattern for those Ops in a model that have multiple associated internal Ops in the Session Graph.
    It uses these OpTypePattern objects to detect Ops in the Session Graph. It holds the detected Ops and their
    associated internal Ops. This association is ued when the ConnectedGraph is constructed for a model.
    """

    def __init__(self, graph: tf.Graph, op_to_module_dict: Dict[tf.Operation, ModuleIdentifierOpInfo],
                 valid_ops: Set[tf.Operation]):
        """
        Initialize the SubGraphMatcher.

        :param graph: Session Graph associated with the model
        :param op_to_module_dict: Dictionary mapping op to module op info, to be filled in by SubGraphMatcher
        :param valid_ops: Ops that will be considered during module detection
        """

        self._graph = graph
        self._valid_ops = valid_ops
        self.detect_ops_in_graph(op_to_module_dict)

    # The functions below access protected members of TF classes.
    # pylint: disable=protected-access
    def detect_ops_in_graph(self, op_to_module_dict: Dict[tf.Operation, ModuleIdentifierOpInfo]):
        """
        Use the OpTypePattern objects to detect Ops in a
        specific Session Graph. Keep the detected Ops and their associated internal Ops.

        :param op_to_module_dict: Dictionary mapping op to module op info, to be filled in by SubGraphMatcher
        """

        all_op_patterns_list = [op_dict['pattern'] for op_dict in list(reference_op_pattern_info_dict.values())]
        for pattern in all_op_patterns_list:
            layer_matcher = graph_matcher.GraphMatcher(pattern)

            # Graph Match
            for match_result in layer_matcher.match_graph(self._graph):
                matched_patterns = list(match_result._pattern_to_op_tensor.keys())
                op = match_result.get_op(matched_patterns[0])
                # For ops like FusedBatchNorm, there are multiple output ops of the model which may be matched (Merge,
                # Merge_1, Merge_2. In these cases, Merge is the one that should be matched because if either of the
                # other two are matched, Merge will not make it into the op_to_module_dict.
                if op not in self._valid_ops:
                    continue
                template_pattern_type = pattern_to_op_type[matched_patterns[0]]
                if op in op_to_module_dict:
                    # op was already matched with a different pattern previously. Compare lengths of the previous
                    # pattern with current pattern, and replace the previous op type with the current op type if more
                    # ops were matched.
                    # This can happen if one pattern is a subset of another (Conv2D without bias vs Conv2D with bias for
                    # example. If the same op is matched with both patterns, we will pick Conv2D with bias to be the one
                    # to use.
                    op_info = op_to_module_dict[op]
                    if reference_op_pattern_info_dict[op_info.pattern_type]['length'] >= \
                            reference_op_pattern_info_dict[template_pattern_type]['length']:
                        # op was already matched with a larger pattern set
                        continue

                ops_list = [op for op in get_internal_ops_for_pattern(match_result) if op in self._valid_ops]
                # ops_list should not be empty since there was an earlier check that the current match_result has ops
                # in self._valid_ops.
                if not ops_list:
                    logger.error('Valid matched ops list should not be empty')
                    raise AssertionError('Valid matched ops list should not be empty')
                # Check if any ops in ops_list were already matched with a larger pattern. If so, no need to change
                # existing entries in op_to_module_dict.
                if not SubGraphMatcher._is_subset_of_already_matched_op(template_pattern_type, ops_list,
                                                                        op_to_module_dict):
                    op_type = reference_op_pattern_info_dict[template_pattern_type]['op_type']
                    module_name = get_module_name(reference_op_pattern_info_dict[template_pattern_type]['module_regex'],
                                                  ops_list)
                    associated_op = get_associated_op(reference_op_pattern_info_dict[template_pattern_type]
                                                      ['associated_op_regex'], ops_list)
                    op_info = ModuleIdentifierOpInfo(module_name, op_type, associated_op,
                                                     pattern_type=template_pattern_type,
                                                     internal_ops=ops_list)
                    for op in ops_list:
                        op_to_module_dict[op] = op_info

    @staticmethod
    def _is_subset_of_already_matched_op(current_pattern: str, ops_list: List[tf.Operation],
                                         op_to_module_dict: Dict[tf.Operation, ModuleIdentifierOpInfo]):
        """
        For each op in ops_list, check if it has already been associated with a module. If so, check if the length
        of the pattern for the module is longer or shorter than the pattern for the currently matched type.
        :param current_pattern: Currently matched pattern
        :param ops_list: List of ops that are currently matched together
        :param op_to_module_dict: Dictionary mapping previously matched ops to ModuleIdentifierOpInfo objects
        :return: True if the currently matched ops are a subset of a previously matched set of ops.
        """
        for op in ops_list:
            if op in op_to_module_dict:
                op_info = op_to_module_dict[op]
                if reference_op_pattern_info_dict[op_info.pattern_type]['length'] > \
                        reference_op_pattern_info_dict[current_pattern]['length']:
                    # op was already matched with a larger pattern set
                    # Below assertion is to make sure that all ops in ops_list are op_to_module_dict already
                    for an_op in ops_list:
                        assert an_op in op_to_module_dict
                    return True
        return False
