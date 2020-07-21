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
from aimet_tensorflow.common import sub_graph_matcher
from aimet_tensorflow.common.sub_graph_matcher import ModuleIdentifierOpInfo

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)


class ModuleIdentifier(ABC):
    """ Module identifier class for identifying submodules from groups of tf Operations """

    def __init__(self, graph: tf.Graph):
        """ Initializer for ModuleIdentifier """
        self._graph = graph

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
        self._sub_graph_matcher = sub_graph_matcher.SubGraphMatcher(self._graph, self.op_to_module_dict,
                                                                    self._valid_ops)

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
