# /usr/bin/env python2.7
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Set of core utilities shared between quantization and svd code """

import re

import tensorflow as tf

from aimet_tensorflow.utils import constants
from aimet_tensorflow.common import op_defs
from aimet_common.utils import AimetLogger

_BIAS_TYPES = ['Add', 'BiasAdd']

# Ops to skip quantization on, eg backprop, etc
_SKIPPED_PREFIXES = ('gradients/', 'RMSProp/', 'Adagrad/', 'Const_', 'HistogramSummary', 'ScalarSummary', 'save/', 'truncated_normal', 'Adam')

# Valid activation ops for quantization end points.
_ACTIVATION_OP_SUFFIXES = ['/Relu6', '/Relu', '/Identity']

# Regular expression for recognizing nodes that are part of batch norm group.
_BATCHNORM_RE = re.compile(r'^(.*)/BatchNorm/batchnorm')

_OP_MAP = op_defs.default_op_map


class OpQuery:
    """
    Class for query a graph's operations and related data.
    """

    def __init__(self, graph, op_map=None, ops_to_ignore=None, strict=True):
        """
        Constructor
        :param graph: The graph to search
        :param op_map: The map of operations used to identify op sequences as "one op".
        The default op_map used is defined in op_deps.py. Please refer to
        that format for passing a custom op_map.
        :param ops_to_ignore: List of ops to ignore
        :param strict: If strict mode is set to True queries will only return the last ops
        at the end of well known "op layers" as defined by the op_map. When False,
        queries will return ops at the end of well known layers and, in addition,
        all ops which are not "known".

        Eg If you have a list of ops in a graph like: Conv2D, BiasAdd, WeirdOp
        Strict mode will return ["BiasAdd"] since it knows that Conv2D+BiasAdd are
        one logical "layer". When strict mode is disabled it will return ["BiasAdd", "WeirdOp"]
        :param debug: Whether to enable debug messages or not.
        """

        self._log = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)
        self._graph = graph
        self._strict = strict

        if op_map:
            self._op_map = op_map
        else:
            self._op_map = _OP_MAP

        if ops_to_ignore:
            self._ops_to_ignore = ops_to_ignore
        else:
            self._ops_to_ignore = []

        self._trained_vars = graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

    @staticmethod
    def _is_op_with_weights(op):
        """
        Checks if a given op has weights
        :param op: TF op
        :return: True, if op has weights, False otherwise
        """
        return (op.type in constants.OP_WEIGHT_TYPES and
                not op.name.startswith(_SKIPPED_PREFIXES))

    @classmethod
    def get_weights_for_op(cls, op):
        """
        Get the weight tensor for a given op
        :param op: TF op
        :return: Weight tensor for the op
        """
        weights = None
        if cls._is_op_with_weights(op):
            weights = op.inputs[constants.OP_WEIGHT_INDICES[op.type]]
        return weights

    @staticmethod
    def get_bias_for_op(op):
        """
        Get bias tensor for the given op
        :param op: TF op
        :return: Bias tensor for the op
        """
        bias = None
        if op.type in _BIAS_TYPES:
            bias = op.inputs[constants.OP_WEIGHT_INDICES[op.type]]
        return bias

    def get_weight_ops(self, ops=None, skip_bias_op=False):
        """
        Get all ops that contain weights. If a list of ops is passed search only ops
        from this list. Return the sequenced list of weight ops always with Conv/FC
        first, followed by the bias op, if present.
        :param ops: List of ops to use (optional)
        :param ops: If bias op has to be skipped (optional)
        :return:
        """
        if not ops:
            ops = self._graph.get_operations()

        ops_with_weights = []
        for op in ops:
            if self._is_op_with_weights(op):
                self._log.debug('Found op w/weights: %s', op.name)
                ops_with_weights.append(op)

                if not skip_bias_op and self._is_op_with_weights(op):
                    for consumer in op.outputs[0].consumers():
                        # Ignore Reshape as it can be placed between MatMul and BiasAdd on Dense layer of Transformer
                        if consumer.type in ['Reshape'] and len(consumer.outputs[0].consumers()) == 1:
                            consumer = consumer.outputs[0].consumers()[0]
                        if consumer.type in _BIAS_TYPES:
                            self._log.debug('Found op w/bias: %s', consumer.name+'('+consumer.type+')')
                            ops_with_weights.append(consumer)

        reduced_list = [x for x in ops_with_weights if not x.name.startswith(tuple(self._ops_to_ignore))]
        return reduced_list

    @staticmethod
    def get_weight_inputs(ops):
        """
        Given a list of ops, returns a corresponding list of the weight indexes for their inputs
        :param ops: List of TF ops
        :return:
        """
        indices = list()
        for op in ops:
            if op.type not in constants.OP_WEIGHT_INDICES:
                raise ValueError('Op type: '+op.type+' does not contain weights!')
            indices.append(constants.OP_WEIGHT_INDICES[op.type])
        return indices

    def _match_ops(self, current_op, candidate_op_list, matched_ops, visited_ops):
        """
        Recursive function that helps traverse a network and find matching ops
        :param current_op: Current op to traverse downstream from
        :param candidate_op_list: Current list of candidate ops that may result in a match
        :param matched_ops: List of already found matched_ops
        :param visited_ops: List of all ops that have been visited (to cut short duplicate traversals)
        :return:
        """
        if any(x in current_op.name for x in _SKIPPED_PREFIXES):
            return matched_ops

        self._log.debug('Processing op: %s (%s) w/current list=%s', current_op.name, current_op.type, candidate_op_list)

        candidate_op_list.append(current_op)
        match_len, max_len = op_defs.check_match(candidate_op_list, op_map=self._op_map)
        self._log.debug('Got match_len: %s and max_len: %s', str(match_len), str(max_len))

        if match_len != 0 and match_len == max_len:
            # Matched the maximum sequence possible
            matched_ops.append(current_op)
            op_type_list = [list_op.type for list_op in candidate_op_list]
            self._log.info('Found op match w/new op: %s and sequence: %s', current_op.name, str(op_type_list))
            candidate_op_list = []
        elif match_len == 0:
            # A list length > 1 means the current op_list was a match but not the newly added op. Save the previous last
            # op from the list
            if len(candidate_op_list) > 1:
                # Check if indeed the previous op_list is a match
                if op_defs.does_sequence_match(candidate_op_list[:-1], op_map=self._op_map):
                    matched_op = candidate_op_list[-2]
                    matched_ops.append(matched_op)
                    op_type_list = [list_op.type for list_op in candidate_op_list[:-1]]
                    self._log.info('Found op match: %s and sequence: %s', matched_op.name, str(op_type_list))

                # Test to see if the current op is a match by itself
                candidate_op_list = []
                matched_ops = self._match_ops(current_op, candidate_op_list, matched_ops, visited_ops)
                return matched_ops

            # No match, reset the list
            candidate_op_list = []

        # There was some match, but not the max match possible. Continue drilling through the
        # outputs to the next ops
        for tensor in current_op.outputs:
            for consumer in tensor.consumers():
                if consumer not in visited_ops:
                    visited_ops.add(consumer)
                    self._log.info('Adding to visited_logs: %s', consumer.name)
                    matched_ops = self._match_ops(consumer, candidate_op_list, matched_ops, visited_ops)

        return matched_ops

    def get_known_ops(self, inputs):
        """
        Given a set of inputs, find all the ops in the network that are from the "well known" op collections
        defined in the OpQuery's op_map
        :param inputs: List of input ops
        :return: List of all consumer ops in the network which are well-known
        """
        if not inputs:
            raise ValueError('No input op names provided!')

        input_ops = [self._graph.get_operation_by_name(name) for name in inputs]

        matched_ops = [[] for _ in range(len(input_ops))]
        for op_index, op in enumerate(input_ops):
            self._log.info('Matching ops starting from: %s', op.name)
            matched_ops[op_index] = self._match_ops(op, [], [], set())
            self._log.info('Found %i known op groups', len(matched_ops[op_index]))

        # Filter dups and merge the newly matched ops
        # Todo: Is there a more pythonic and faster way to detect duplicates?
        unique_ops = {}
        for ops in matched_ops:
            for op in ops:
                unique_ops[op.name] = op

        # Filter out the entries that should be ignored
        list_of_unique_ops = list(unique_ops.values())
        reduced_list = [x for x in list_of_unique_ops if not x.name.startswith(tuple(self._ops_to_ignore))]

        return reduced_list
