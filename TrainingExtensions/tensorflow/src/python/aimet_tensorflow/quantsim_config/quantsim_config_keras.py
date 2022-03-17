# /usr/bin/env python3.6
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
""" Utilities for parsing and applying quantsim configurations from json config file """
from typing import List, Tuple, Dict

from tensorflow.keras import layers

from aimet_common.connected_graph.operation import Op
from aimet_common.quantsim_config.json_config_importer import ConfigType, SupergroupType, OpTypeType, ParamType, \
    DefaultsType
from aimet_common.quantsim_config.quantsim_config import QuantSimConfigurator as AimetCommonQuantSimConfigurator, \
    get_all_ops_in_neighborhood
from aimet_tensorflow.keras.connectedgraph import ConnectedGraph

LayerAffectedQuantizerTupleType = Tuple[List[Tuple[layers.Layer, str]], List[Tuple[layers.Layer, str]],
                                        List[Tuple[layers.Layer, str]], List[Tuple[layers.Layer, str]]]


def _get_affected_tensor_quantizers_by_true_setting(op: Op, direction: str) -> List[Tuple[layers.Layer, str]]:
    """
    Get a list of tensor quantizers that would be affected if the quantization of target direction of op is enabled

    :param op: Op to enable target quantization (input or output) for
    :param direction: Target direction which will be enabled
    :return: List of tuples containing layer and direction that would be affected
    """
    return [(op.get_module(), direction)]


def _get_affected_tensor_quantizers_by_false_setting(op: Op, direction: str) -> List[Tuple[layers.Layer, str]]:
    """
    Get a list of tensor quantizers that would be affected if the quantization of target direction of op is disabled

    :param op: Op to disable target quantization (input or output) for
    :param direction: Target direction which will be disabled
    :return: List of tuples containing layer and direction that would be affected
    """
    affected_tensor_quantizers_by_false_setting = []
    neighboring_ops = get_all_ops_in_neighborhood(op, direction)
    for neighbor_op in neighboring_ops:
        if neighbor_op.type == "Split":
            continue

        if neighboring_ops[neighbor_op] == "input":
            affected_tensor_quantizers_by_false_setting.append((neighbor_op.get_module(), "input"))
        else:
            affected_tensor_quantizers_by_false_setting.append((neighbor_op.get_module(), "output"))

    return affected_tensor_quantizers_by_false_setting


class QuantSimConfigurator(AimetCommonQuantSimConfigurator):
    """ Class for parsing and applying quantsim configurations from json config file """

    def __init__(self, connected_graph: ConnectedGraph, config_file: str):
        super(QuantSimConfigurator, self).__init__(config_file)
        self._connected_graph = connected_graph
        self._layer_to_tensor_quantizers_dict = self._create_layer_to_tensor_quantizers_dict()

    def _create_layer_to_tensor_quantizers_dict(self) -> Dict[layers.Layer, LayerAffectedQuantizerTupleType]:
        """
        Create affected tensor quantizers information by layer dictionary
        - List of tensor quantizers to change if op's input quantizer setting is set to True
        - List of tensor quantizers to change if op's output quantizer setting is set to True
        - List of tensor quantizers to change if op's input quantizer setting is set to False
        - List of tensor quantizers to change if op's output quantizer setting is set to False
        :return: Dictionary mapping layer to tuple of lists of affected layer quantization information tuples
        """
        layer_to_tensor_quantizers_dict = {}
        for op in self._connected_graph.ordered_ops:
            affected_quantizers_when_input_enabled = _get_affected_tensor_quantizers_by_true_setting(op, "input")
            affected_quantizers_when_output_enabled = _get_affected_tensor_quantizers_by_true_setting(op, "output")
            affected_quantizers_when_input_disabled = _get_affected_tensor_quantizers_by_false_setting(op, "input")
            affected_quantizers_when_output_disabled = _get_affected_tensor_quantizers_by_false_setting(op, "output")

            layer_to_tensor_quantizers_dict[op.get_module()] = (affected_quantizers_when_input_enabled,
                                                                affected_quantizers_when_output_enabled,
                                                                affected_quantizers_when_input_disabled,
                                                                affected_quantizers_when_output_disabled)

        return layer_to_tensor_quantizers_dict

    def _set_default_configs(self, default_configs: DefaultsType):
        pass

    def _set_param_configs(self, param_configs: ParamType):
        pass

    def _set_op_type_configs(self, op_configs: OpTypeType):
        pass

    def _set_supergroup_configs(self, supergroups_configs: List[SupergroupType]):
        pass

    def _set_model_input_configs(self, model_input_configs: ConfigType):
        pass

    def _set_model_output_configs(self, model_output_configs: ConfigType):
        pass
