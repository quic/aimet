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
""" Utilities for parsing and applying quantsim configurations from json config file """

from abc import ABC, abstractmethod
from typing import List
from aimet_common.connected_graph.operation import Op
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.quantsim_config.json_config_importer import JsonConfigImporter, ConfigDictKeys, DefaultsType, \
    ParamType, OpTypeType, SupergroupType, ConfigType
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class SupergroupConfigCallback(ABC):
    """ Class acting as a callback for when supergroups are found """
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, _, op_list: List[Op]):
        """ Callback logic """


class OnnxConnectedGraphTypeMapper:
    """
    Class maintaining dictionaries for two way mapping from onnx types to connected graph types
    """
    def __init__(self, type_pairs: List[List[List[str]]]):
        self._onnx_to_conn_graph_dict = {}
        self._conn_graph_to_onnx_dict = {}
        for onnx_types, pytorch_types in type_pairs:
            for onnx_type in onnx_types:
                self._onnx_to_conn_graph_dict[onnx_type] = pytorch_types
            for pytorch_type in pytorch_types:
                self._conn_graph_to_onnx_dict[pytorch_type] = onnx_types

    def get_conn_graph_type_from_onnx_type(self, onnx_type: str):
        """
        Return connected graph type corresponding to onnx type
        :param onnx_type: Onnx type to find corresponding connected graph type
        :return: Connected graph type corresponding to onnx_type
        """
        return self._onnx_to_conn_graph_dict.get(onnx_type)

    def get_onnx_type_from_conn_graph_type(self, conn_graph_type: str):
        """
        Return onnx type corresponding to connected graph type
        :param conn_graph_type: Connected graph type to find corresponding onnx type
        :return: Onnx type corresponding to conn_graph_type
        """
        return self._conn_graph_to_onnx_dict.get(conn_graph_type)


class QuantSimConfigurator(ABC):
    """ Class for parsing and applying quantsim configurations from json config file """
    def __init__(self, config_file: str):
        self._quantsim_configs = JsonConfigImporter.import_json_config_file(config_file)

    def _set_quantsim_configs(self):
        """
        Apply quantsim configurations to the given model
        """
        self._set_default_configs(self._quantsim_configs[ConfigDictKeys.DEFAULTS])
        self._set_param_configs(self._quantsim_configs[ConfigDictKeys.PARAMS])
        self._set_op_type_configs(self._quantsim_configs[ConfigDictKeys.OP_TYPE])
        self._set_supergroup_configs(self._quantsim_configs[ConfigDictKeys.SUPERGROUPS])
        self._set_model_input_configs(self._quantsim_configs[ConfigDictKeys.MODEL_INPUT])
        self._set_model_output_configs(self._quantsim_configs[ConfigDictKeys.MODEL_OUTPUT])

    @abstractmethod
    def _set_default_configs(self, default_configs: DefaultsType):
        """
        Set default configurations for op and param quantizers in model (first level of specificity in configuration
        file)
        :param default_configs: Default configurations for quantizers
        """

    @abstractmethod
    def _set_param_configs(self, param_configs: ParamType):
        """
        Set configurations for all params of specific types (second level of specificity in configuration file)
        :param param_configs: Dictionary containing configurations for parameters of certain types
        """

    @abstractmethod
    def _set_op_type_configs(self, op_configs: OpTypeType):
        """
        Set configurations for all ops of specific types (third level of specificity in configuration file)
        :param op_configs: Dictionary containing configurations for ops of certain types
        """

    @classmethod
    def _build_supergroup_patterns(cls, supergroup_config: SupergroupType, callback: SupergroupConfigCallback,
                                   onnx_conn_graph_type_mapper: OnnxConnectedGraphTypeMapper) \
            -> List[PatternType]:
        """
        Create a list holding pattern types corresponding to sequences specified in the supergroup config
        :param supergroup_config: Quantsim wrapper configurations for supergroup ops
        :return: List of PatternTypes holding supergroup ops and callback for when the supergroup is found
        """
        op_list = supergroup_config[ConfigDictKeys.OP_LIST]
        list_of_permutations = _build_list_of_permutations(op_list, onnx_conn_graph_type_mapper)
        list_of_patterns = []
        for permutation in list_of_permutations:
            list_of_patterns.append(PatternType(pattern=permutation, action=callback))
        return list_of_patterns

    @abstractmethod
    def _set_supergroup_configs(self, supergroups_configs: List[SupergroupType]):
        """
        Set supergroup specific configurations (fourth level of specificity in configuration file)
        :param supergroups_configs: Configurations for supergroups
        """

    @abstractmethod
    def _set_model_input_configs(self, model_input_configs: ConfigType):
        """
        Set model input specific configurations (fifth level of specificity in configuration file)
        :param model_input_configs: Configuration for model inputs
        """

    @abstractmethod
    def _set_model_output_configs(self, model_output_configs: ConfigType):
        """
        Set model output specific configurations (sixth level of specificity in configuration file)
        :param model_output_configs: Configuration for model outputs
        """


def _build_list_of_permutations(op_list: List[str], onnx_conn_graph_type_mapper: OnnxConnectedGraphTypeMapper) \
        -> List[List[str]]:
    """
    Given a list of onnx op types, where each onnx op type could potentially map to multiple connected graph types,
    create a list of all permutations of lists of connected graph types that would satisfy the same ordering as the
    original onnx op type list.
    For example, for an onnx op type "o1" that maps to two connected graph types "c1_1" and
    "c1_2", and an onnx op type "o2" that maps to two connected graph types "c2_1" and "c2_2", all permutations of
    ["o1", "o2"] would lead to ["c1_1", "c2_1"], ["c1_1", "c2_2"], ["c1_2", "c2_1"], and ["c1_2", "c2_2"].
    :param op_list: List of onnx op types
    :param onnx_conn_graph_type_mapper: Class that provides utilities for mapping onnx op types to connected graph types
    :return: List of permutations of connected graph op types satisfying the ordering specified by op_list onnx types
    """
    # base case, return list of lists of connected graph ops corresponding to the only op in the list
    if len(op_list) == 1:
        permutations_of_op_list = []
        conn_graph_types_of_current_op = onnx_conn_graph_type_mapper.get_conn_graph_type_from_onnx_type(op_list[0])
        for op in conn_graph_types_of_current_op:
            permutations_of_op_list.append([op])
        return permutations_of_op_list

    permutations_of_op_list = []
    permutations_of_succeeding_ops = _build_list_of_permutations(op_list[1:], onnx_conn_graph_type_mapper)
    conn_graph_types_of_current_op = onnx_conn_graph_type_mapper.get_conn_graph_type_from_onnx_type(op_list[0])
    for op in conn_graph_types_of_current_op:
        for permutation in permutations_of_succeeding_ops:
            new_permutation = [op] + permutation
            permutations_of_op_list.append(new_permutation)
    return permutations_of_op_list


def get_setting_type(setting_name: str) -> str:
    """
    Return a string corresponding to the type of setting that is specified by setting_name.
    :param setting_name: Name of the setting to change
    :return: String corresponding to the type of setting that is specified by setting_name.
    """
    if setting_name in [ConfigDictKeys.IS_INPUT_QUANTIZED, ConfigDictKeys.IS_OUTPUT_QUANTIZED]:
        return ConfigDictKeys.IS_QUANTIZED
    if setting_name == ConfigDictKeys.IS_SYMMETRIC:
        return ConfigDictKeys.IS_SYMMETRIC
    logger.error('Unrecognized quantizer setter name %s', setting_name)
    raise AssertionError
