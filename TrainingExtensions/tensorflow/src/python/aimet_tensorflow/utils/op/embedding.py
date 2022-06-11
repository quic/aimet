# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" utilities for embedding op """

from typing import List, Dict
import tensorflow as tf

from aimet_common.graph_searcher import GraphSearcher
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.utils import AimetLogger
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.defs import ParameterInfo

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def get_embedding_params_using_patterns(conn_graph: ConnectedGraph) -> Dict[str, ParameterInfo]:
    """
    Find embedding parameters to quantize using patterns
    :param conn_graph: Connected graph of the model
    :return: Dictionary with name of parameters to quantize as keys and information about parameters as values
    """
    params_to_quantize = {}

    pattern_handlers = [BertEmbeddingPatternHandler()]
    patterns_with_callbacks = []
    for pattern_handler in pattern_handlers:
        patterns_with_callbacks.extend(pattern_handler.get_pattern_types())

    graph_searcher = GraphSearcher(conn_graph, patterns_with_callbacks)
    graph_searcher.find_all_patterns_in_graph_apply_actions()

    for pattern_handler in pattern_handlers:
        params_to_quantize.update(pattern_handler.get_parameters())

    return params_to_quantize


class BertEmbeddingPatternHandler:
    """ class with apis related to TF weight tensor of embedding op """
    def __init__(self):
        # List of tuple containing embedding layer pattern and index of associated_op
        patterns = [(['ResourceGather', 'Identity', 'branch', 'AddV2', 'AddV2', 'LayerNorm', 'Identity'], 4),
                    (['ResourceGather', 'Identity', 'Tile', 'AddV2', 'AddV2', 'LayerNorm', 'Identity'], 4),
                    (['ResourceGather', 'Identity', 'branch', 'Shape', 'StridedSlice', 'Pack', 'Tile', 'AddV2',
                      'AddV2', 'LayerNorm', 'Identity'], 8),
                    (['ResourceGather', 'Identity', 'AddV2', 'LayerNorm', 'Identity'], 2)]

        self.pattern_types = {PatternType(pattern=pattern, action=self): index for (pattern, index) in patterns}
        self.associated_ops = set()

    def __call__(self, *args, **kwargs):
        pattern_type, op_subset = args

        assert pattern_type in self.pattern_types
        index = self.pattern_types[pattern_type]
        add_1 = op_subset[index].output_op_node
        assert add_1.type == 'AddV2'
        self.associated_ops.add(add_1)

    def get_pattern_types(self) -> List[PatternType]:
        """
        Return pattern_types' keys as a list
        :return: List of PatternType
        """
        return list(self.pattern_types)

    def get_parameters(self) -> Dict[str, ParameterInfo]:
        """
        Iterate all the embedding layer (associated_ops) found and return dictionary of parameters associated with it
        :return: Dict with parameter name as key and ParameterInfo as value
        """
        parameter_info_list = {}

        for associated_op in self.associated_ops:
            word_tensor = self._get_word_tensor(associated_op)
            position_tensor = self._get_position_tensor(associated_op)
            token_tensor = self._get_token_tensor(associated_op)

            for param_tensor in [word_tensor, position_tensor, token_tensor]:
                op_with_param = None
                for consumer in param_tensor.consumers():
                    if not consumer.name.startswith('gradients/'):
                        assert op_with_param is None
                        op_with_param = consumer
                assert op_with_param is not None
                parameter_info_list[param_tensor.op.name] = ParameterInfo('weight', [op_with_param.name])

        return parameter_info_list

    @staticmethod
    def _get_word_tensor(embedding_op: tf.Operation) -> tf.Tensor:
        """
        Get word embedding op from embedding op
        :param embedding_op: associated op of embedding layer
        :return: word embedding tensor
        """
        assert embedding_op.type == 'AddV2'
        add = embedding_op.inputs[0].op
        assert add.type == 'AddV2'
        identity = add.inputs[0].op
        assert identity.type == 'Identity'
        gather = identity.inputs[0].op
        assert gather.type == 'ResourceGather'

        return gather.outputs[0]

    @staticmethod
    def _get_position_tensor(embedding_op: tf.Operation) -> tf.Tensor:
        """
        Get position embedding op from embedding op
        :param embedding_op: associated op of embedding layer
        :return: position embedding tensor
        """
        assert embedding_op.type == 'AddV2'
        add = embedding_op.inputs[0].op
        assert add.type == 'AddV2'
        tile = add.inputs[1].op
        assert tile.type == 'Tile'
        identity_1 = tile.inputs[0].op
        assert identity_1.type == 'Identity'
        gather_1 = identity_1.inputs[0].op
        assert gather_1.type == 'ResourceGather'

        # gather_1 is not connected to valid_ops, which makes quantization difficult
        # Quantize the consumer instead - which is identity_1
        return identity_1.outputs[0]

    @staticmethod
    def _get_token_tensor(embedding_op: tf.Operation) -> tf.Tensor:
        """
        Get token embedding op from embedding op
        :param embedding_op: associated op of embedding layer
        :return: token embedding tensor
        """
        assert embedding_op.type == 'AddV2'
        identity_2 = embedding_op.inputs[1].op
        assert identity_2.type == 'Identity'
        gather_2 = identity_2.inputs[0].op
        assert gather_2.type == 'ResourceGather'

        return gather_2.outputs[0]
