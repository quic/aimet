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

from typing import List, Dict, Tuple, Set, Union
import tensorflow as tf
from aimet_common.quantsim_config.json_config_importer import DefaultsType, OpType, ParamType, OpTypeType, \
    SupergroupType, ConfigType, ConfigDictKeys
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops, get_all_output_ops
from aimet_common.graph_searcher import GraphSearcher
from aimet_common.quantsim_config.quantsim_config import QuantSimConfigurator as AimetCommonQuantSimConfigurator
from aimet_common.quantsim_config.quantsim_config import SupergroupConfigCallback as AimetCommonSupergroupConfigCallback
from aimet_common.quantsim_config.quantsim_config import get_setting_type, OnnxConnectedGraphTypeMapper
from aimet_common.utils import AimetLogger
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.common.operation import Op
from aimet_tensorflow.utils.common import update_variables_with_values, onnx_tf_conn_graph_type_pairs

import libpymo as pymo


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
QuantizerListType = Tuple[List[tf.Operation], List[tf.Operation]]

MAP_TF_PARAM_NAME_TO_QUANTSIM_NAME = {
    "bias": "bias",
    "weight": "weight"
}

# Type for param set dictionary, mapping a param string type to a set of quantize ops for that param type
ParamSetDictType = Dict[str, Set[tf.Operation]]

# Type for one value of the op to quant ops dict, containing a list of two elements: a param set dict, and a single tf
# operation
ParamSetAndOutputOpListType = List[Union[ParamSetDictType, tf.Operation]]

# Type for op to quant ops dict
OpToQuantOpsDictType = Dict[Op, ParamSetAndOutputOpListType]


class SupergroupConfigCallback(AimetCommonSupergroupConfigCallback):
    """ Class acting as a callback for when supergroups are found """

    def __init__(self, sess: tf.Session, op_to_quant_ops_dict: OpToQuantOpsDictType):
        super().__init__()
        self._sess = sess
        self._op_to_quant_ops_dict = op_to_quant_ops_dict

    def __call__(self, _, op_list: List[Op]):
        # Turn off output quantizations for all ops except for the last op
        # Assumes op list is at least of length two
        vars_with_value = {}
        for op in op_list[:-1]:
            _, output_quantize_op = self._op_to_quant_ops_dict[op]
            vars_with_value[output_quantize_op.name + '_op_mode'] = int(pymo.TensorQuantizerOpMode.passThrough)
        update_variables_with_values(self._sess, vars_with_value)


class QuantSimConfigurator(AimetCommonQuantSimConfigurator):
    """ Class for parsing and applying
    quantsim configurations from json config file """
    def __init__(self, sess: tf.Session, conn_graph: ConnectedGraph, op_to_quant_ops_dict: OpToQuantOpsDictType,
                 config_file: str):
        super().__init__(config_file)

        self._sess = sess
        self._conn_graph = conn_graph
        self._op_to_quant_ops_dict = op_to_quant_ops_dict
        self._op_to_quantizer_lists_dict = self._get_op_to_quantizer_lists_dict()
        self._onnx_conn_graph_name_mapper = OnnxConnectedGraphTypeMapper(onnx_tf_conn_graph_type_pairs)

        # Set all quantizers to pass through mode first
        self._disable_all_quantizers()
        self._set_quantsim_configs()

    def _get_op_to_quantizer_lists_dict(self) -> Dict[Op, QuantizerListType]:
        """
        For every connected graph op that has a corresponding output quantizer, associate it with a tuple containing 2
        lists:
            - List of quantize ops to change if op's input quantizer setting is set
            - List of quantize ops to change if op's output quantizer setting is set
        :return: Dictionary mapping connected graph ops to a tuple of lists of qc_quantize_ops.
        """
        op_to_two_quantizer_lists_dict = {}
        for op in self._op_to_quant_ops_dict.keys():
            input_quant_op_list = self._get_quant_ops_for_input_setting(op)
            output_quant_op_list = self._get_quant_ops_for_output_setting(op)
            op_to_two_quantizer_lists_dict[op] = (input_quant_op_list, output_quant_op_list)
        return op_to_two_quantizer_lists_dict

    def _get_quant_ops_for_input_setting(self, op: Op):
        """
        Get a list of quantize ops to modify when setting input quantization for the op.
        :param op: Op to set input quantization settings for.
        :return: List of quantize ops to modify
        """
        input_true_quant_ops = []
        queue = [op]
        while queue:
            curr_op = queue.pop()
            for inp_op in curr_op.input_ops:
                if inp_op in self._op_to_quant_ops_dict:
                    _, output_quantize_op = self._op_to_quant_ops_dict[inp_op]
                    input_true_quant_ops.append(output_quantize_op)
                else:
                    queue.append(inp_op)
        return input_true_quant_ops

    def _get_quant_ops_for_output_setting(self, op: Op):
        """
        Get a list of quantize ops to modify when setting output quantization for the op.
        :param op: Op to set output quantization settings for.
        :return: List of quantize ops to modify
        """
        _, output_quantize_op = self._op_to_quant_ops_dict[op]
        output_true_quant_ops = [output_quantize_op]
        return output_true_quant_ops

    def _disable_all_quantizers(self):
        """
        Set all quantize ops to passThrough and use_symmetric_encodings False
        """
        vars_with_value = {}
        for param_dict, output_quantizer in self._op_to_quant_ops_dict.values():
            for param_quantize_op_set in param_dict.values():
                for param_quantize_op in param_quantize_op_set:
                    vars_with_value[param_quantize_op.name + '_op_mode'] = int(pymo.TensorQuantizerOpMode.passThrough)
            vars_with_value[output_quantizer.name + '_op_mode'] = int(pymo.TensorQuantizerOpMode.passThrough)

        update_variables_with_values(self._sess, vars_with_value)

    def _set_default_configs(self, default_configs: DefaultsType):
        """
        Set default configurations for op and param quantizers in model (first level of specificity in configuration
        file)
        :param default_configs: Default configurations for quantizers
        """
        self._set_default_configs_for_ops(default_configs[ConfigDictKeys.OPS])
        self._set_default_configs_for_params(default_configs[ConfigDictKeys.PARAMS])

    def _set_default_configs_for_ops(self, default_op_configs: ConfigType):
        """
        Set default configurations for all ops in the model.
        :param default_op_configs: Default configurations for ops
        """
        # Modified quantize ops is a dictionary that keeps track of the quantize ops that are modified by this function.
        # It is initialized as empty, and each time self._set_config_for_op is called, an op will be added.
        # If ever a quantize op's attribute has already been modified this round and a contradicting setting is
        # specified, an assertion will be thrown.
        modified_quantize_ops = {}
        input_ops = get_all_input_ops(self._conn_graph)
        for op, input_output_quantize_ops in self._op_to_quantizer_lists_dict.items():
            if op in input_ops:
                # Don't set the input op setting, since it should be off by default even if all ops within the model
                # are to be quantized.  If model input is to be quantized, model_input should be set in config file
                # instead.
                continue
            self._set_config_for_op(input_output_quantize_ops, default_op_configs, modified_quantize_ops)

    def _set_default_configs_for_params(self, default_param_configs: ConfigType):
        """
        Set default configurations for all params in the model.
        :param default_param_configs: Default configurations for params
        """
        for param_quantize_ops_dict, _ in self._op_to_quant_ops_dict.values():
            for param_quantize_op_set in param_quantize_ops_dict.values():
                for param_quantize_op in param_quantize_op_set:
                    self._set_config_for_param(param_quantize_op, default_param_configs)

    def _set_config_for_op(self, input_output_quantize_ops: QuantizerListType, op_config: OpType,
                           modified_quantize_ops: Dict[tf.Operation, Set], op: tf.Operation = None):
        """
        Set configurations for a specific op
        :param input_output_quantize_ops: Tuple containing 2 lists:
            - List of quantize ops to change if op's input quantizer setting is set
            - List of quantize ops to change if op's output quantizer setting is set
        :param op_config: Configuration for the op
        :param modified_quantize_ops: Dictionary of quantize ops mapping to set of settings that have been changed for
            that quantize op already.
        :param op: Quantize op to set config of
        """
        if ConfigDictKeys.IS_INPUT_QUANTIZED in op_config:
            self._modify_activation_quantize_op(input_output_quantize_ops, ConfigDictKeys.IS_INPUT_QUANTIZED,
                                                op_config[ConfigDictKeys.IS_INPUT_QUANTIZED], modified_quantize_ops)
        if ConfigDictKeys.IS_OUTPUT_QUANTIZED in op_config:
            self._modify_activation_quantize_op(input_output_quantize_ops, ConfigDictKeys.IS_OUTPUT_QUANTIZED,
                                                op_config[ConfigDictKeys.IS_OUTPUT_QUANTIZED], modified_quantize_ops)
        if ConfigDictKeys.IS_SYMMETRIC in op_config:
            self._modify_activation_quantize_op(input_output_quantize_ops, ConfigDictKeys.IS_SYMMETRIC,
                                                op_config[ConfigDictKeys.IS_SYMMETRIC], modified_quantize_ops)

        # Will only see this in the op_type section, not default
        if ConfigDictKeys.PARAMS in op_config:
            if op is None:
                logger.error('No module provided to set params for')
                raise AssertionError
            param_quantize_ops_dict, _ = self._op_to_quant_ops_dict[op]
            for param_name in param_quantize_ops_dict.keys():
                quantsim_param_name = MAP_TF_PARAM_NAME_TO_QUANTSIM_NAME.get(param_name, None)
                if quantsim_param_name is not None and quantsim_param_name in op_config[ConfigDictKeys.PARAMS]:
                    param_config = op_config[ConfigDictKeys.PARAMS][quantsim_param_name]
                    for param_quantize_op in param_quantize_ops_dict[param_name]:
                        self._set_config_for_param(param_quantize_op, param_config)

    def _set_config_for_param(self, param_quantize_op: tf.Operation, param_config: ConfigType):
        """
        Set configurations for a specific param quantize op
        :param param_quantize_op: Quantize op to set configurations for
        :param param_config: Configuration for the quantize op
        """
        vars_with_value = {}
        if ConfigDictKeys.IS_QUANTIZED in param_config:
            if param_config[ConfigDictKeys.IS_QUANTIZED]:
                vars_with_value[param_quantize_op.name + '_op_mode'] = \
                    int(pymo.TensorQuantizerOpMode.oneShotQuantizeDequantize)
            else:
                vars_with_value[param_quantize_op.name + '_op_mode'] = \
                    int(pymo.TensorQuantizerOpMode.passThrough)
        if ConfigDictKeys.IS_SYMMETRIC in param_config:
            vars_with_value[param_quantize_op.name + '_use_symmetric_encoding'] = \
                param_config[ConfigDictKeys.IS_SYMMETRIC]
        update_variables_with_values(self._sess, vars_with_value)

    def _set_param_configs(self, param_configs: ParamType):
        """
        Set configurations for all params of specific types (second level of specificity in configuration file)
        :param param_configs: Dictionary containing configurations for parameters of certain types
        """
        for param_quantize_ops_dict, _ in self._op_to_quant_ops_dict.values():
            for param_name in param_quantize_ops_dict.keys():
                quantsim_param_name = MAP_TF_PARAM_NAME_TO_QUANTSIM_NAME.get(param_name, None)
                if quantsim_param_name is not None and quantsim_param_name in param_configs:
                    param_config = param_configs[quantsim_param_name]
                    for param_quantize_op in param_quantize_ops_dict[param_name]:
                        self._set_config_for_param(param_quantize_op, param_config)

    def _set_op_type_configs(self, op_configs: OpTypeType):
        """
        Set configurations for all ops of specific types (third level of specificity in configuration file)
        :param op_configs: Dictionary containing configurations for ops of certain types
        """
        modified_quantize_ops = {}
        # Set op type configs for named modules
        for op, input_output_quantize_ops in self._op_to_quantizer_lists_dict.items():
            onnx_types = self._onnx_conn_graph_name_mapper.get_onnx_type_from_conn_graph_type(op.type)
            if not onnx_types:
                continue
            for onnx_type in onnx_types:
                if onnx_type in op_configs:
                    op_config = op_configs[onnx_type]
                    self._set_config_for_op(input_output_quantize_ops, op_config, modified_quantize_ops, op)

    def _set_supergroup_configs(self, supergroups_configs: List[SupergroupType]):
        """
        Set supergroup specific configurations (fourth level of specificity in configuration file)
        :param supergroups_configs: Configurations for supergroups
        """
        patterns_with_callbacks = []
        for supergroup_config in supergroups_configs:
            callback = SupergroupConfigCallback(self._sess, self._op_to_quant_ops_dict)
            patterns = self._build_supergroup_patterns(supergroup_config, callback, self._onnx_conn_graph_name_mapper)
            for pattern in patterns:
                patterns_with_callbacks.append(pattern)

        if patterns_with_callbacks:
            graph_searcher = GraphSearcher(self._conn_graph, patterns_with_callbacks)
            graph_searcher.find_all_patterns_in_graph_apply_actions()

    def _set_model_input_configs(self, model_input_configs: ConfigType):
        """
        Set model input specific configurations (fifth level of specificity in configuration file)
        :param model_input_configs: Configuration for model inputs
        """

        # In TF, turning model input quantization on actually means turning on the output quantizer of the input
        # placeholder op.  Thus we replace the IS_INPUT_QUANTIZED field with IS_OUTPUT_QUANTIZED instead.
        if ConfigDictKeys.IS_INPUT_QUANTIZED in model_input_configs and \
                model_input_configs[ConfigDictKeys.IS_INPUT_QUANTIZED]:
            del model_input_configs[ConfigDictKeys.IS_INPUT_QUANTIZED]
            model_input_configs[ConfigDictKeys.IS_OUTPUT_QUANTIZED] = True
            input_ops = get_all_input_ops(self._conn_graph)
            for op in input_ops:
                if op in self._op_to_quantizer_lists_dict:
                    modified_quantize_ops = {}
                    self._set_config_for_op(self._op_to_quantizer_lists_dict[op], model_input_configs,
                                            modified_quantize_ops)

    def _set_model_output_configs(self, model_output_configs: ConfigType):
        """
        Set model output specific configurations (sixth level of specificity in configuration file)
        :param model_output_configs: Configuration for model outputs
        """
        output_ops = get_all_output_ops(self._conn_graph)
        for op in output_ops:
            if op in self._op_to_quantizer_lists_dict:
                modified_quantize_ops = {}
                self._set_config_for_op(self._op_to_quantizer_lists_dict[op], model_output_configs,
                                        modified_quantize_ops)

    def _modify_activation_quantize_op(self, input_output_quantize_ops: QuantizerListType, setting_name: str,
                                       quantizer_setting: bool, modified_quantize_ops: Dict[tf.Operation, Set]):
        """
        Modify the appropriate quantize ops for the given quantizer setting.  If a quantize op has already been
        modified, compare the old setting with the new setting and assert if the settings conflict.
        :param input_output_quantize_ops: Tuple containing 2 lists:
            - List of quantize ops to change if op's input quantizer setting is set
            - List of quantize ops to change if op's output quantizer setting is set
        :param setting_name: String representing the setting to be modified
        :param quantizer_setting: Boolean representing the new setting value
        :param modified_quantize_ops: Dictionary of quantize ops mapping to set of settings that have been changed for
            that quantize op already.
        """
        setting_type = get_setting_type(setting_name)

        quantize_ops_to_modify = _get_quantize_ops_to_modify(input_output_quantize_ops, setting_name)
        for quantize_op in quantize_ops_to_modify:
            if quantize_op in modified_quantize_ops and \
                    setting_type in modified_quantize_ops[quantize_op]:
                # Tensor quantizer's setting has already been modified
                if setting_name in [ConfigDictKeys.IS_INPUT_QUANTIZED, ConfigDictKeys.IS_OUTPUT_QUANTIZED]:
                    op_mode_tensor = self._sess.graph.get_tensor_by_name(quantize_op.name + '_op_mode:0')
                    # current_setting will be True if op mode is not passThrough (is enabled), False otherwise
                    current_setting = (self._sess.run(op_mode_tensor) != int(pymo.TensorQuantizerOpMode.passThrough))
                else:
                    op_mode_tensor = self._sess.graph.get_tensor_by_name(quantize_op.name +
                                                                         '_use_symmetric_encoding:0')
                    current_setting = self._sess.run(op_mode_tensor)
                if current_setting != quantizer_setting:
                    logger.error('Conflicting tensor quantizer settings for symmetric encodings')
                    raise AssertionError
            else:
                vars_with_value = {}
                if setting_name in [ConfigDictKeys.IS_INPUT_QUANTIZED, ConfigDictKeys.IS_OUTPUT_QUANTIZED]:
                    if not quantizer_setting:
                        setting = pymo.TensorQuantizerOpMode.passThrough
                    else:
                        setting = pymo.TensorQuantizerOpMode.updateStats
                    vars_with_value[quantize_op.name + '_op_mode'] = int(setting)
                else:
                    vars_with_value[quantize_op.name + '_use_symmetric_encoding'] = quantizer_setting
                update_variables_with_values(self._sess, vars_with_value)
                if quantize_op not in modified_quantize_ops:
                    modified_quantize_ops[quantize_op] = {setting_type}
                else:
                    modified_quantize_ops[quantize_op].add(setting_type)


def _get_quantize_ops_to_modify(input_output_quantize_ops: QuantizerListType, setting_name: str) -> List[tf.Operation]:
    """
    Given the tuple containing lists of quantize ops to modify under different situations, identify a list of quantize
    ops to modify given the specific setting name and quantizer setting.
    :param input_output_quantize_ops: Tuple containing 2 lists:
        - List of quantize ops to change if op's input quantizer setting is set
        - List of quantize ops to change if op's output quantizer setting is set
    :param setting_name: String representing the setting to be modified
    :return: List of quantize ops to modify given the specific setting name and quantizer setting.
    """
    input_list, output_list = input_output_quantize_ops

    if setting_name == ConfigDictKeys.IS_INPUT_QUANTIZED:
        return input_list
    if setting_name == ConfigDictKeys.IS_OUTPUT_QUANTIZED:
        return output_list
    if setting_name == ConfigDictKeys.IS_SYMMETRIC:
        # Will modify all input and output quantizers in the False case
        return input_list + output_list
    logger.error('Encountered unrecognized case for setting name %s', setting_name)
    raise AssertionError
