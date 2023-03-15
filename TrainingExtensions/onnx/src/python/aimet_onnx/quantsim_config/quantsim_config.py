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
""" Utilities for parsing and applying quantsim configurations from json config file """

from typing import List, Dict

from onnx import onnx_pb
from aimet_common.defs import QuantizationDataType
from aimet_common.graph_searcher import GraphSearcher
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops, get_all_output_ops
from aimet_common.quantsim_config.json_config_importer import ConfigDictKeys, ConfigType, OpType, ParamType, OpTypeType, \
    SupergroupType
from aimet_common.quantsim_config.quantsim_config import QuantSimConfigurator as AimetCommonQuantSimConfigurator, \
    get_setting_type, SupergroupConfigCallback as AimetCommonSupergroupConfigCallback
from aimet_common.utils import AimetLogger
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.utils import get_product_name_from_quantized_name
from aimet_onnx.qc_quantize_op import OpMode, QcQuantizeOp


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class OpToQuantizers:
    """
    Maps an op to input, output and parameter QcQuantizeOps
    """
    def __init__(self):
        self.input_quantizers = []
        self.output_quantizers = []
        self.parameter_quantizers = []


class SupergroupConfigCallback(AimetCommonSupergroupConfigCallback):
    """ Class acting as a callback for when supergroups are found """

    def __init__(self, model: onnx_pb.ModelProto, op_to_quantizers: Dict):
        super().__init__()
        self._model = model
        self._op_to_quantizers = op_to_quantizers

    def __call__(self, _, op_list: List[str]):
        # Turn off output quantizaters for all ops except for the last op
        # Assumes op list is at least of length two
        for op in op_list[:-1]:
            output_quantizers = self._op_to_quantizers[op.dotted_name].output_quantizers
            for output_quantizer in output_quantizers:
                output_quantizer.enabled = False


class QuantSimConfigurator(AimetCommonQuantSimConfigurator):
    """ Class for parsing and applying
    quantsim configurations from json config file """

    def __init__(self, model: onnx_pb.ModelProto, conn_graph: ConnectedGraph, config_file: str, quantsim_output_bw: int,
                 quantsim_param_bw: int, quantsim_data_type: QuantizationDataType = QuantizationDataType.int):
        super().__init__(config_file, quantsim_data_type, quantsim_output_bw, quantsim_param_bw)

        self._model = model
        self._conn_graph = conn_graph
        self._quant_ops_dict = {}
        self._param_names = {}
        self._activation_names = {}
        self._op_to_quantizer_lists_dict = None
        self._op_to_quantizers = {}

    def configure_quantizers(self, quant_ops_dict: Dict,
                             param_names: List[str],
                             activation_names: List[str]):
        """
        Configures quantizers based on config file
        """
        self._quant_ops_dict = quant_ops_dict
        self._param_names = param_names
        self._activation_names = activation_names

        self._op_to_quantizers = self._map_quantizers_to_ops()

        # Disable all quantizers
        self._disable_all_quantizers()
        self._set_quantsim_configs()
        self._override_param_bw_dtype(self._param_names, self._default_data_type, self._default_param_bw)
        self._override_act_bw_dtype(self._activation_names, self._default_data_type, self._default_output_bw)

    def _map_quantizers_to_ops(self) -> Dict[str, OpToQuantizers]:
        """
        Creates a dict where key is the name of the op and value is OpToQuantizers which comprises of input quantizers
        and output quantizers of an op
        """
        op_to_quantizers = {}
        for name, op in self._conn_graph.get_all_ops().items():
            if 'branch' in name:
                continue
            node = op.get_module()
            op_to_quantizers[node.name] = OpToQuantizers()
            for input_product in node.input:
                self._populate_input_and_param_quantizer(op_to_quantizers[node.name], input_product)
            for output_product in node.output:
                self._populate_output_quantizer(op_to_quantizers[node.name], output_product)

        return op_to_quantizers

    def _populate_input_and_param_quantizer(self, op_to_quantizers: OpToQuantizers, input_product: str):
        """
        Populate input and param quantizer for an op
        """
        product_name = get_product_name_from_quantized_name(input_product)
        # In case where there is no quantizer product_name will turn out to be None
        if not product_name:
            return
        if product_name in self._activation_names:
            op_to_quantizers.input_quantizers.append(self._quant_ops_dict[product_name])
        else:
            op_to_quantizers.parameter_quantizers.append((product_name, self._quant_ops_dict[product_name]))

    def _populate_output_quantizer(self, op_to_quantizers: OpToQuantizers, output_product: str):
        """
        Populates output quantizer for an op
        """
        if output_product in self._activation_names:
            op_to_quantizers.output_quantizers.append(self._quant_ops_dict[output_product])

    def _disable_all_quantizers(self):
        """
        Disable all qc_quantize ops
        """
        for param_name in self._param_names:
            self._quant_ops_dict[param_name].enabled = False

        for activation_name in self._activation_names:
            self._quant_ops_dict[activation_name].enabled = False

    def _override_param_bw_dtype(self, quantizer_data, data_type: QuantizationDataType, bitwidth: int):
        """
        overrides data type and bitwidth default config for param quantizers of given data
        :param quantizer_data: object containing which param override will be applied to
        :param bitwidth: bitwidth
        :param data_type: data type as QuantizationDataType
        :return:
        """
        for param_name in quantizer_data:
            if param_name in self._quant_ops_dict:
                self._quant_ops_dict[param_name].data_type = data_type
                self._quant_ops_dict[param_name].bitwidth = bitwidth

    def _override_act_bw_dtype(self, quantizer_data, data_type: QuantizationDataType, bitwidth: int):
        """
        overrides data type and bitwidth default config for activation quantizers of given data
        :param quantizer_data: object containing which activation override will be applied to
        :param bitwidth: bitwidth
        :param data_type: data type as QuantizationDataType
        :return:
        """
        for act_name in quantizer_data:
            if act_name in self._quant_ops_dict:
                self._quant_ops_dict[act_name].data_type = data_type
                self._quant_ops_dict[act_name].bitwidth = bitwidth

    def _override_default_act_bw_dtype(self, data_type: QuantizationDataType, bitwidth: int):
        """
        overrides data type and bw default config for input/output quantizers.
        :param data_type: data type as QuantizationDataType
        :param bitwidth: bitwidth to be configured
        :return:
        """
        raise NotImplementedError

    def _override_default_param_bw_dtype(self, data_type: QuantizationDataType, bitwidth: int):
        """
        overrides data type and bitwidth default config for param quantizers
        :param bitwidth: bitwidth
        :param data_type: data type as QuantizationDataType
        :return:
        """
        raise NotImplementedError

    def _set_param_configs(self, param_configs: ParamType):
        """
        Set configurations for all params of specific types (second level of specificity in configuration file)
        :param param_configs: Dictionary containing configurations for parameters of certain types
        """
        for op_name, op_to_quantizer in self._op_to_quantizers.items():
            param_quantizers = op_to_quantizer.parameter_quantizers
            for param_name, param_quantizer in param_quantizers:
                quantsim_param_type = self._get_param_type(op_name, param_name)
                if quantsim_param_type is not None and quantsim_param_type in param_configs:
                    param_config = param_configs[quantsim_param_type]
                    self._set_config_for_param(param_quantizer, param_config)

    def _set_op_type_configs(self, op_configs: OpTypeType):
        """
        Set configurations for all ops of specific types (third level of specificity in configuration file)
        :param op_configs: Dictionary containing configurations for ops of certain types
        """
        modified_quantize_ops = {}
        for op_name, op_to_quantizer in self._op_to_quantizers.items():
            op = self._conn_graph.get_all_ops()[op_name]
            if op.type in op_configs:
                op_config = op_configs[op.type]
                self._set_config_for_op(op_name, op_to_quantizer, op_config, modified_quantize_ops)

    def _set_supergroup_configs(self, supergroups_configs: List[SupergroupType]):
        """
        Set supergroup specific configurations (fourth level of specificity in configuration file)
        :param supergroups_configs: Configurations for supergroups
        """
        patterns_with_callbacks = []
        for supergroup_config in supergroups_configs:
            callback = SupergroupConfigCallback(self._model, self._op_to_quantizers)
            op_list = supergroup_config[ConfigDictKeys.OP_LIST]

            # Op list consists of patterns to be searched for, we pass a list of op_list to be compatible with build_list
            patterns = self._build_list_of_pattern([op_list], callback)
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
        input_ops = get_all_input_ops(self._conn_graph)
        for input_op in input_ops:
            op_name = input_op.name
            if op_name in self._op_to_quantizers:
                modified_quantize_ops = {}
                self._set_config_for_op(op_name, self._op_to_quantizers[op_name],
                                        model_input_configs, modified_quantize_ops)

    def _set_model_output_configs(self, model_output_configs: ConfigType):
        """
        Set model output specific configurations (sixth level of specificity in configuration file)
        :param model_output_configs: Configuration for model outputs
        """
        output_ops = get_all_output_ops(self._conn_graph)

        for output_op in output_ops:
            op_name = output_op.name
            if op_name in self._op_to_quantizers:
                modified_quantize_ops = {}
                self._set_config_for_op(op_name, self._op_to_quantizers[op_name],
                                        model_output_configs, modified_quantize_ops)

    def _set_default_configs(self, default_configs):
        """
        Set default configurations for op and param quantizers in model (first level of specificity in configuration
        file)
        :param default_configs: Default configurations for quantizers
        """
        self._set_default_configs_for_ops(default_configs[ConfigDictKeys.OPS])
        self._set_default_configs_for_params(default_configs[ConfigDictKeys.PARAMS])
        if ConfigDictKeys.STRICT_SYMMETRIC in default_configs:
            self._set_strict_symmetric(default_configs[ConfigDictKeys.STRICT_SYMMETRIC])
        if ConfigDictKeys.UNSIGNED_SYMMETRIC in default_configs:
            self._set_unsigned_symmetric(default_configs[ConfigDictKeys.UNSIGNED_SYMMETRIC])

    def _set_default_configs_for_params(self, default_param_configs: ConfigType):
        """
        Set default configurations for all params in the model.
        :param default_param_configs: Default configurations for params
        """
        for param_name in self._param_names:
            param_quantizer = self._quant_ops_dict[param_name]
            self._set_config_for_param(param_quantizer, default_param_configs)

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
        for op_name, op_to_quantizer in self._op_to_quantizers.items():
            self._set_config_for_op(op_name, op_to_quantizer, default_op_configs, modified_quantize_ops)

        for model_input in self._model.model.graph.input:
            self._quant_ops_dict[model_input.name].enabled = False

    def _set_config_for_op(self, op_name, op_to_quantizer: OpToQuantizers, op_config: OpType,
                           modified_quantize_ops: Dict):
        """
        Set configurations for a specific op
        :param op_name: name of the op
        :param op_to_quantizer: OpToQuantizers class containing input, output and param quantizer to a node
        :param op_config: Configuration for the op
        :param modified_quantize_ops: Dictionary of quantize ops mapping to set of settings that have been changed for
            that quantize op already.
        """
        if ConfigDictKeys.IS_INPUT_QUANTIZED in op_config:
            self._modify_activation_quantize_op(op_to_quantizer.input_quantizers, ConfigDictKeys.IS_INPUT_QUANTIZED,
                                                op_config[ConfigDictKeys.IS_INPUT_QUANTIZED], modified_quantize_ops)
        if ConfigDictKeys.IS_OUTPUT_QUANTIZED in op_config:
            self._modify_activation_quantize_op(op_to_quantizer.output_quantizers, ConfigDictKeys.IS_OUTPUT_QUANTIZED,
                                                op_config[ConfigDictKeys.IS_OUTPUT_QUANTIZED], modified_quantize_ops)
        if ConfigDictKeys.IS_SYMMETRIC in op_config:
            self._modify_activation_quantize_op(op_to_quantizer.input_quantizers + op_to_quantizer.output_quantizers,
                                                ConfigDictKeys.IS_SYMMETRIC, op_config[ConfigDictKeys.IS_SYMMETRIC],
                                                modified_quantize_ops)

        # Will only see this in the op_type section, not default
        if ConfigDictKeys.PARAMS in op_config:
            param_quantizers = op_to_quantizer.parameter_quantizers
            for param_name, param_quantizer in param_quantizers:
                quantsim_param_type = self._get_param_type(op_name, param_name)
                if quantsim_param_type is not None and quantsim_param_type in op_config[ConfigDictKeys.PARAMS]:
                    param_config = op_config[ConfigDictKeys.PARAMS][quantsim_param_type]
                    self._set_config_for_param(param_quantizer, param_config)

    def _get_param_type(self, op_name: str, param_name: str) -> str:
        """ Returns the type of param, weight/ bias """
        conn_graph_op = self._conn_graph.get_all_ops()[op_name]
        _, param_type = conn_graph_op.parameters[param_name]
        return param_type

    @staticmethod
    def _modify_activation_quantize_op(quantize_ops_to_modify: List[QcQuantizeOp], setting_name: str,
                                       quantizer_setting: bool, modified_quantize_ops: Dict):
        """
        Modify the appropriate quantize ops for the given quantizer setting.  If a quantize op has already been
        modified, compare the old setting with the new setting and assert if the settings conflict.
        :param quantize_ops_to_modify: List of quantizers to modify
        :param setting_name: String representing the setting to be modified
        :param quantizer_setting: Boolean representing the new setting value
        :param modified_quantize_ops: Dictionary of quantize ops mapping to set of settings that have been changed for
            that quantize op already.
        """
        # pylint: disable=too-many-branches
        setting_type = get_setting_type(setting_name)

        for quantizer in quantize_ops_to_modify:
            if quantizer in modified_quantize_ops and \
                    setting_type in modified_quantize_ops[quantizer]:
                # Tensor quantizer's setting has already been modified
                if setting_name in [ConfigDictKeys.IS_INPUT_QUANTIZED, ConfigDictKeys.IS_OUTPUT_QUANTIZED]:
                    current_setting = quantizer.enabled
                else:
                    current_setting = quantizer.use_symmetric_encodings
                if current_setting != quantizer_setting:
                    logger.error('Conflicting tensor quantizer settings for symmetric encodings')
                    raise AssertionError('Conflicting tensor quantizer settings for symmetric encodings')
            else:
                if setting_name in [ConfigDictKeys.IS_INPUT_QUANTIZED, ConfigDictKeys.IS_OUTPUT_QUANTIZED]:
                    if not quantizer_setting:
                        quantizer.enabled = False
                    else:
                        quantizer.enabled = True
                        quantizer.op_mode = OpMode.updateStats
                else:
                    quantizer.use_symmetric_encodings = quantizer_setting
                if quantizer not in modified_quantize_ops:
                    modified_quantize_ops[quantizer] = {setting_type}
                else:
                    modified_quantize_ops[quantizer].add(setting_type)

    @staticmethod
    def _set_config_for_param(param_quantizer: QcQuantizeOp, param_config: ConfigType):
        """
        Set configurations for a specific param quantize op
        :param param_quantizer: Quantize op to set configurations for
        :param param_config: Configuration for the quantize op
        """
        if ConfigDictKeys.IS_QUANTIZED in param_config:
            param_quantizer.enabled = param_config[ConfigDictKeys.IS_QUANTIZED]
        if ConfigDictKeys.IS_SYMMETRIC in param_config:
            param_quantizer.use_symmetric_encodings = param_config[ConfigDictKeys.IS_SYMMETRIC]

    def _set_strict_symmetric(self, use_strict_symmetric: bool):
        """
        Set strict symmetric configuration for all quantizers in the model.
        :param use_strict_symmetric: True or False setting for using strict symmetric mode
        """
        for quantizer in self._quant_ops_dict.values():
            quantizer.use_strict_symmetric = use_strict_symmetric

    def _set_unsigned_symmetric(self, use_unsigned_symmetric: bool):
        """
        Set unsigned symmetric configuration for all quantizers in the model.
        :param use_unsigned_symmetric: True or False setting for using unsigned symmetric mode
        """
        for quantizer in self._quant_ops_dict.values():
            quantizer.use_unsigned_symmetric = use_unsigned_symmetric

    def _generate_and_apply_op_instance_specific_config(self):
        """
        Generate op instance specific configurations - currently supported_kernels and per_channel_quantization fields
        This function uses op specific supported_kernels (if absent use defaults), op specific per_channel_quantization
        fields (if absent use default per_channel_quantization) and generate op instance specific config
        :return: {op_instance_name, op_specific_config}
        """
        raise NotImplementedError
