# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Module identifier class """

from abc import ABC, abstractmethod
from typing import List, Set
import tensorflow as tf
from aimet_common.utils import AimetLogger
from aimet_tensorflow.common import module_identifier_matchers
from aimet_tensorflow.common import sub_graph_matcher
from aimet_tensorflow.common.module_identifier_matchers import ModuleIdentifierOpInfo

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)


class ModuleIdentifier(ABC):
    """ Module identifier class for identifying submodules from groups of tf Operations """

    def __init__(self, graph: tf.Graph):
        """ Initializer for ModuleIdentifier """
        self._graph = graph

    @abstractmethod
    def _identify_modules(self):
        """ Parse tf graph to extract modules from operations """

    @abstractmethod
    def get_op_info(self, op: tf.Operation) -> ModuleIdentifierOpInfo:
        """
        Given a tf op in the graph, return OpInfo class containing:
        - opname: Name that op/product graph should use to represent whatever module
        this tf op belongs to
        - type: Module type that should be stored in the op/product graph for this module
        - module: Module name that this tf op belongs to (should be unique between separate modules)
        """


class StructureModuleIdentifier(ModuleIdentifier):
    """ Module identifier using graph structures """

    def __init__(self, graph: tf.Graph, starting_op_names: List[str], valid_ops: Set[tf.Operation]):
        """ Initializer for ModuleIdentifier
        :param graph: Tensorflow graph to represent using connected graph.
        :param starting_op_names: Names of the starting ops of the model.
        :param valid_ops: Set of tf operations that are valid
        """

        super().__init__(graph)
        self.op_to_module_dict = dict()
        self._num_products_made = 0
        self.starting_op_names = starting_op_names
        self._valid_ops = valid_ops
        self.processed_ops = set()
        self._sub_graph_matcher = sub_graph_matcher.SubGraphMatcher(self._graph)
        self._identify_modules()

    def _identify_modules(self):
        """ Parse tf graph to extract modules from operations """
        queue = []
        for starting_op_name in self.starting_op_names:
            queue.append(self._graph.get_operation_by_name(starting_op_name))
        while queue:
            current_op = queue.pop()
            self.processed_ops.add(current_op)
            if current_op not in self.op_to_module_dict.keys():
                self._add_ops_in_module(current_op, current_op.type)
            for product in current_op.outputs:
                for consumer in product.consumers():
                    if consumer not in self.processed_ops and consumer in self._valid_ops:
                        queue.append(consumer)

    def get_op_info(self, op: tf.Operation) -> ModuleIdentifierOpInfo:
        """
        Given a tf op in the graph, return OpInfo class containing:
        - opname: Name that op/product graph should use to represent whatever module
        this tf op belongs to
        - type: Module type that should be stored in the op/product graph for this module
        - module: Module name that this tf op belongs to (should be unique between separate modules)
        """
        default_op_info = ModuleIdentifierOpInfo(module_name=op.name,
                                                 op_type=op.type,
                                                 tf_op=op)

        op_info = self.op_to_module_dict.get(op, default_op_info)

        return op_info

    def _add_ops_in_module_using_sub_graph_matcher(self, op: tf.Operation, op_type: str):
        """ Find and add all ops belonging to the same module as op to op_to_module_dict (if possible) """

        op_info = ModuleIdentifierOpInfo(module_name=op.name,
                                         op_type=op.type,
                                         tf_op=op)

        # Each value in switcher is a list of functions which attempt to match known module patterns around the current
        # op.  For a certain type of op, we proceed through each function in the corresponding list until one function
        # returns True (means a module pattern match succeeded)
        switcher = {
            "Conv2D": [self._sub_graph_matcher.match_conv2d_dense_type_ops],
            "DepthwiseConv2dNative": [self._sub_graph_matcher.match_conv2d_dense_type_ops],
            "FusedBatchNormV3": [self._sub_graph_matcher.match_fused_batchnorm],
            "MatMul": [self._sub_graph_matcher.match_conv2d_dense_type_ops],
            "Reshape": [self._sub_graph_matcher.match_flatten],
            "RandomUniform": [self._sub_graph_matcher.match_dropout_1],
            "Mul": [self._sub_graph_matcher.match_dropout_2],
            "Softmax": [self._sub_graph_matcher.match_softmax],
            "Unpack": [module_identifier_matchers.match_upsample],
            "GatherV2": [module_identifier_matchers.match_downsample]
        }

        op_handlers = switcher.get(op_type, [module_identifier_matchers.handle_default])
        for handler in op_handlers:
            if handler(self.op_to_module_dict, op_info):       # match found, no need to try to match more functions
                break

    def _add_ops_in_module(self, op: tf.Operation, op_type: str):
        """ Find and add all ops belonging to the same module as op to op_to_module_dict (if possible) """

        op_info = ModuleIdentifierOpInfo(module_name=op.name,
                                         op_type=op.type,
                                         tf_op=op)

        # Each value in switcher is a list of functions which attempt to match known module patterns around the current
        # op.  For a certain type of op, we proceed through each function in the corresponding list until one function
        # returns True (means a module pattern match succeeded)
        switcher = {
            "Conv2D": [module_identifier_matchers.match_conv2d_dense_type_ops],
            "DepthwiseConv2dNative": [module_identifier_matchers.match_conv2d_dense_type_ops],
            "FusedBatchNormV3": [module_identifier_matchers.match_fusedbatchnorm_pattern_1,
                                 module_identifier_matchers.match_fusedbatchnorm_pattern_2,
                                 module_identifier_matchers.match_fusedbatchnorm_pattern_3],
            "MatMul": [module_identifier_matchers.match_conv2d_dense_type_ops],
            "Reshape": [module_identifier_matchers.match_flatten_ops],
            "RandomUniform": [module_identifier_matchers.match_dropout_pattern_1,
                              module_identifier_matchers.match_dropout_pattern_2],
            "Mul": [module_identifier_matchers.match_dropout_pattern_3,
                    module_identifier_matchers.match_leaky_relu],
            "Softmax": [module_identifier_matchers.match_softmax],
            "Unpack": [module_identifier_matchers.match_upsample],
            "GatherV2": [module_identifier_matchers.match_downsample],
            "Shape": [module_identifier_matchers.match_upsample2d],
            "Max": [module_identifier_matchers.match_global_max_pool2d]
        }

        op_handlers = switcher.get(op_type, [module_identifier_matchers.handle_default])
        for handler in op_handlers:
            if handler(self.op_to_module_dict, op_info):       # match found, no need to try to match more functions
                break
