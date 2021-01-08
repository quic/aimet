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

from typing import Dict, List, Tuple, Set
import torch

from aimet_common.utils import AimetLogger
from aimet_common.graph_searcher import GraphSearcher
from aimet_common.connected_graph.operation import Op
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops, get_all_output_ops
from aimet_common.quantsim_config.json_config_importer import ConfigDictKeys, ConfigType, SupergroupType, OpType, \
    ParamType, DefaultsType, OpTypeType, ConfigDictType
from aimet_common.quantsim_config.quantsim_config import QuantSimConfigurator as AimetCommonQuantSimConfigurator
from aimet_common.quantsim_config.quantsim_config import SupergroupConfigCallback as AimetCommonSupergroupConfigCallback
from aimet_common.quantsim_config.quantsim_config import get_setting_type, OnnxConnectedGraphTypeMapper
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch.tensor_quantizer import TensorQuantizer
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.onnx_utils import map_torch_types_to_onnx, onnx_pytorch_conn_graph_type_pairs

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

MAP_PYTORCH_PARAM_NAME_TO_QUANTSIM_NAME = {
    "bias": "bias",
    "weight": "weight"
}

ELEMENTWISE_OP_TYPES = ['add', 'mul', 'cat', 'div']

TensorQuantizersTupleType = Tuple[List[TensorQuantizer], List[TensorQuantizer], List[TensorQuantizer],
                                  List[TensorQuantizer]]


class SupergroupConfigCallback(AimetCommonSupergroupConfigCallback):
    """ Class acting as a callback for when supergroups are found """

    def __init__(self, module_to_quantsim_wrapper_dict: Dict[torch.nn.Module, QcQuantizeWrapper]):
        super().__init__()
        self._module_to_quantsim_wrapper_dict = module_to_quantsim_wrapper_dict

    def __call__(self, _, op_list: List[Op]):
        # Turn off input and output quantizations for all ops (only turn off output quantization for first op, and only
        # turn off input quantization for last op)
        # Assumes op list is at least of length two
        for index, op in enumerate(op_list):
            if op.type in ELEMENTWISE_OP_TYPES:
                # if op is an elementwise op, it does not have wrappers.  Thus nothing needs to be done, since previous
                # and subsequent ops with wrappers will be set correctly anyway.
                continue
            else:
                assert op.get_module() is not None
            if index == 0:
                # turn off only output quantization of first op in the list
                first_quantsim_wrapper = self._module_to_quantsim_wrapper_dict[op_list[0].get_module()]
                # pylint: disable=protected-access
                first_quantsim_wrapper.output_quantizers[0].enabled = False
            elif index == len(op_list) - 1:
                # turn off only input quantization of last op in the list
                last_quantsim_wrapper = self._module_to_quantsim_wrapper_dict[op_list[-1].get_module()]
                last_quantsim_wrapper.input_quantizer.enabled = False
            else:
                quantsim_wrapper = self._module_to_quantsim_wrapper_dict[op.get_module()]
                quantsim_wrapper.input_quantizer.enabled = False
                quantsim_wrapper.output_quantizers[0].enabled = False


class QuantSimConfigurator(AimetCommonQuantSimConfigurator):
    """ Class for parsing and applying quantsim configurations from json config file """
    def __init__(self, model, connected_graph: ConnectedGraph, config_file: str):
        super().__init__(config_file)

        _report_unsupported_ops(self._quantsim_configs)
        self._conn_graph = connected_graph
        self._onnx_conn_graph_name_mapper = OnnxConnectedGraphTypeMapper(onnx_pytorch_conn_graph_type_pairs)
        self._module_to_quantsim_wrapper_dict = _create_module_to_quantsim_wrapper_dict(model)
        self._named_modules_to_tensor_quantizers_dict = self._create_named_modules_to_tensor_quantizers_dict()
        self._elementwise_op_to_tensor_quantizers_dict = self._create_elementwise_op_to_tensor_quantizers_dict()

        self._disable_all_quantizers()
        self._set_quantsim_configs()

    def _create_named_modules_to_tensor_quantizers_dict(self) -> Dict[torch.nn.Module, TensorQuantizersTupleType]:
        """
        For every named module in the graph, associate it with a tuple containing 4 lists:
        - List of tensor quantizers to change if op's input quantizer setting is set to True
        - List of tensor quantizers to change if op's output quantizer setting is set to True
        - List of tensor quantizers to change if op's input quantizer setting is set to False
        - List of tensor quantizers to change if op's output quantizer setting is set to False
        :return: Dictionary mapping op to tuple of lists of tensor quantizers to change
        """
        module_to_tensor_quantizers_dict = {}
        for module, quantsim_wrapper in self._module_to_quantsim_wrapper_dict.items():
            # Attempt to find op in connected graph corresponding to this module
            op = [op for op in self._conn_graph.get_all_ops().values() if op.get_module() == module]
            if op:
                input_true_list = self._get_tensor_quantizers_for_input_true_setting(op[0])
                output_true_list = self._get_tensor_quantizers_for_output_true_setting(op[0])
                input_false_list = self._get_tensor_quantizers_for_input_false_setting(op[0])
                output_false_list = self._get_tensor_quantizers_for_output_false_setting(op[0])
            else:
                input_true_list = [quantsim_wrapper.input_quantizer]
                output_true_list = [quantsim_wrapper.output_quantizers[0]]
                input_false_list = [quantsim_wrapper.input_quantizer]
                output_false_list = [quantsim_wrapper.output_quantizers[0]]
            module_to_tensor_quantizers_dict[module] = (input_true_list, output_true_list, input_false_list,
                                                        output_false_list)
        return module_to_tensor_quantizers_dict

    def _create_elementwise_op_to_tensor_quantizers_dict(self) -> Dict[Op, TensorQuantizersTupleType]:
        """
        For every elementwise op in the graph, associate it with a tuple containing 4 lists:
        - List of tensor quantizers to change if op's input quantizer setting is set to True
        - List of tensor quantizers to change if op's output quantizer setting is set to True
        - List of tensor quantizers to change if op's input quantizer setting is set to False
        - List of tensor quantizers to change if op's output quantizer setting is set to False
        :return: Dictionary mapping op to tuple of lists of tensor quantizers to change
        """
        module_to_tensor_quantizers_dict = {}
        # Extract only ops in the model which correspond to elementwise ops
        elementwise_ops = [op for op in self._conn_graph.get_all_ops().values() if op.type in ELEMENTWISE_OP_TYPES]
        for op in elementwise_ops:
            input_true_list = self._get_tensor_quantizers_for_input_true_setting(op)
            output_true_list = self._get_tensor_quantizers_for_output_true_setting(op)
            input_false_list = self._get_tensor_quantizers_for_input_false_setting(op)
            output_false_list = self._get_tensor_quantizers_for_output_false_setting(op)

            module_to_tensor_quantizers_dict[op] = (input_true_list, output_true_list, input_false_list,
                                                    output_false_list)
        return module_to_tensor_quantizers_dict

    def _get_tensor_quantizers_for_input_true_setting(self, op: Op) -> List[TensorQuantizer]:
        """
        Get a list of tensor quantizers that would be affected if the given op is specified to have input quantization
        enabled.
        :param op: Op to enable input quantization for
        :return: List of tensor quantizers that would be affected if the given op is specified to have input
        quantization enabled.
        """
        tensor_quantizers_for_input_true = []
        if op.get_module() is not None:
            qc_quantize_wrapper = self._module_to_quantsim_wrapper_dict[op.get_module()]
            tensor_quantizers_for_input_true.append(qc_quantize_wrapper.input_quantizer)
        else:
            queue = [op]
            while queue:
                current_op = queue.pop()
                if current_op.inputs:
                    input_ops = [inp.producer for inp in current_op.inputs if not inp.is_model_input]
                    for input_op in input_ops:
                        if input_op.get_module() is not None and input_op.get_module() in \
                                self._module_to_quantsim_wrapper_dict:
                            qc_quantize_wrapper = self._module_to_quantsim_wrapper_dict[input_op.get_module()]
                            tensor_quantizers_for_input_true.append(qc_quantize_wrapper.output_quantizers[0])
                        elif input_op.type == 'Split':
                            queue.append(input_op)
        return tensor_quantizers_for_input_true

    def _get_tensor_quantizers_for_output_true_setting(self, op: Op) -> List[TensorQuantizer]:
        """
        Get a list of tensor quantizers that would be affected if the given op is specified to have output quantization
        enabled.
        :param op: Op to enable output quantization for
        :return: List of tensor quantizers that would be affected if the given op is specified to have output
        quantization enabled.
        """
        tensor_quantizers_for_output_true = []
        if op.get_module() is not None:
            qc_quantize_wrapper = self._module_to_quantsim_wrapper_dict[op.get_module()]
            tensor_quantizers_for_output_true.append(qc_quantize_wrapper.output_quantizers[0])
        else:
            queue = [op]
            while queue:
                current_op = queue.pop()
                if current_op.output:
                    output_ops = [consumer for consumer in current_op.output.consumers]
                    for output_op in output_ops:
                        if output_op.get_module() is not None and output_op.get_module() in \
                                self._module_to_quantsim_wrapper_dict:
                            qc_quantize_wrapper = self._module_to_quantsim_wrapper_dict[output_op.get_module()]
                            tensor_quantizers_for_output_true.append(qc_quantize_wrapper.input_quantizer)
                        elif output_op.type == 'Split':
                            queue.append(output_op)
        return tensor_quantizers_for_output_true

    def _get_tensor_quantizers_for_input_false_setting(self, op: Op) -> List[TensorQuantizer]:
        """
        Get a list of tensor quantizers that would be affected if the given op is specified to have input quantization
        disabled.
        :param op: Op to disable input quantization for
        :return: List of tensor quantizers that would be affected if the given op is specified to have input
        quantization disabled.
        """
        tensor_quantizers_for_input_false = []
        neighboring_input_ops = _get_all_ops_in_neighborhood(op, 'input')
        for input_op in neighboring_input_ops:
            if input_op.type != 'Split' and input_op.get_module() is not None and input_op.get_module() in \
                    self._module_to_quantsim_wrapper_dict:
                qc_quantize_wrapper = self._module_to_quantsim_wrapper_dict[input_op.get_module()]
                if neighboring_input_ops[input_op] == 'input':
                    tensor_quantizers_for_input_false.append(qc_quantize_wrapper.input_quantizer)
                else:
                    tensor_quantizers_for_input_false.append(qc_quantize_wrapper.output_quantizers[0])
        return tensor_quantizers_for_input_false

    def _get_tensor_quantizers_for_output_false_setting(self, op: Op) -> List[TensorQuantizer]:
        """
        Get a list of tensor quantizers that would be affected if the given op is specified to have output quantization
        disabled.
        :param op: Op to disable output quantization for
        :return: List of tensor quantizers that would be affected if the given op is specified to have output
        quantization disabled.
        """
        tensor_quantizers_for_output_false = []
        neighboring_output_ops = _get_all_ops_in_neighborhood(op, 'output')
        for output_op in neighboring_output_ops:
            if output_op.type != 'Split' and output_op.get_module() is not None and output_op.get_module() in \
                    self._module_to_quantsim_wrapper_dict:
                qc_quantize_wrapper = self._module_to_quantsim_wrapper_dict[output_op.get_module()]
                if neighboring_output_ops[output_op] == 'input':
                    tensor_quantizers_for_output_false.append(qc_quantize_wrapper.input_quantizer)
                else:
                    tensor_quantizers_for_output_false.append(qc_quantize_wrapper.output_quantizers[0])
        return tensor_quantizers_for_output_false

    def _disable_all_quantizers(self):
        """
        Set all tensor quantizers to enabled False and use_symmetric_encodings False
        """
        for quantsim_wrapper in self._module_to_quantsim_wrapper_dict.values():
            quantsim_wrapper.input_quantizer.enabled = False
            quantsim_wrapper.input_quantizer.use_symmetric_encodings = False
            quantsim_wrapper.output_quantizers[0].enabled = False
            quantsim_wrapper.output_quantizers[0].use_symmetric_encodings = False
            for param_quantizer in quantsim_wrapper.param_quantizers.values():
                param_quantizer.enabled = False
                param_quantizer.use_symmetric_encodings = False

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
        # Set configs for all ops
        modified_tensor_quantizers = {}
        # Set configs for all named modules
        for input_output_tensor_quantizers in self._named_modules_to_tensor_quantizers_dict.values():
            self._set_config_for_module(input_output_tensor_quantizers, default_op_configs, modified_tensor_quantizers)
        # Set configs for all elementwise ops
        for input_output_tensor_quantizers in self._elementwise_op_to_tensor_quantizers_dict.values():
            self._set_config_for_module(input_output_tensor_quantizers, default_op_configs, modified_tensor_quantizers)

    def _set_default_configs_for_params(self, default_param_configs: ConfigType):
        """
        Set default configurations for all params in the model.
        :param default_param_configs: Default configurations for params
        """
        # Set configs for all params
        for quantsim_wrapper in self._module_to_quantsim_wrapper_dict.values():
            if quantsim_wrapper.param_quantizers:
                for param_quantizer in quantsim_wrapper.param_quantizers.values():
                    _set_config_for_param(param_quantizer, default_param_configs)

    def _set_param_configs(self, param_configs: ParamType):
        """
        Set configurations for all params of specific types (second level of specificity in configuration file)
        :param param_configs: Dictionary containing configurations for parameters of certain types
        """
        for quantsim_wrapper in self._module_to_quantsim_wrapper_dict.values():
            for param_name in quantsim_wrapper.param_quantizers.keys():
                quantsim_param_name = MAP_PYTORCH_PARAM_NAME_TO_QUANTSIM_NAME.get(param_name, None)
                if quantsim_param_name is not None and quantsim_param_name in param_configs:
                    param_config = param_configs[quantsim_param_name]
                    _set_config_for_param(quantsim_wrapper.param_quantizers[param_name], param_config)

    def _set_op_type_configs(self, op_configs: OpTypeType):
        """
        Set configurations for all ops of specific types (third level of specificity in configuration file)
        :param op_configs: Dictionary containing configurations for ops of certain types
        """
        modified_tensor_quantizers = {}
        # Set op type configs for named modules
        for module, input_output_tensor_quantizers in self._named_modules_to_tensor_quantizers_dict.items():
            onnx_types = map_torch_types_to_onnx.get(type(module))
            if not onnx_types:
                continue
            for onnx_type in onnx_types:
                if onnx_type in op_configs:
                    op_config = op_configs[onnx_type]
                    self._set_config_for_module(input_output_tensor_quantizers, op_config, modified_tensor_quantizers,
                                                module)
        # Set op type configs for elementwise ops
        for op, input_output_tensor_quantizers in self._elementwise_op_to_tensor_quantizers_dict.items():
            onnx_types = self._onnx_conn_graph_name_mapper.get_onnx_type_from_conn_graph_type(op.type)
            if not onnx_types:
                continue
            for onnx_type in onnx_types:
                if onnx_type in op_configs:
                    op_config = op_configs[onnx_type]
                    self._set_config_for_module(input_output_tensor_quantizers, op_config, modified_tensor_quantizers)

    def _set_config_for_module(self, input_output_tensor_quantizers: TensorQuantizersTupleType, op_config: OpType,
                               modified_tensor_quantizers: Dict[TensorQuantizer, Set], module: torch.nn.Module = None):
        """
        Set configurations for a specific op
        :param input_output_tensor_quantizers: Tuple of 4 lists containing the following:
            - List of tensor quantizers to change if op's input quantizer setting is set to True
            - List of tensor quantizers to change if op's output quantizer setting is set to True
            - List of tensor quantizers to change if op's input quantizer setting is set to False
            - List of tensor quantizers to change if op's output quantizer setting is set to False
        :param op_config: Configuration for the op
        :param modified_tensor_quantizers: Dictionary of tensor quantizers mapping to set of settings that have been
            changed for that tensor quantizer already.
        :param module: Module to set config of (will be None for elementwise ops)
        """
        if ConfigDictKeys.IS_INPUT_QUANTIZED in op_config:
            _modify_tensor_quantizers(input_output_tensor_quantizers, ConfigDictKeys.IS_INPUT_QUANTIZED,
                                      op_config[ConfigDictKeys.IS_INPUT_QUANTIZED], modified_tensor_quantizers)
        if ConfigDictKeys.IS_OUTPUT_QUANTIZED in op_config:
            _modify_tensor_quantizers(input_output_tensor_quantizers, ConfigDictKeys.IS_OUTPUT_QUANTIZED,
                                      op_config[ConfigDictKeys.IS_OUTPUT_QUANTIZED], modified_tensor_quantizers)
        if ConfigDictKeys.IS_SYMMETRIC in op_config:
            _modify_tensor_quantizers(input_output_tensor_quantizers, ConfigDictKeys.IS_SYMMETRIC,
                                      op_config[ConfigDictKeys.IS_SYMMETRIC], modified_tensor_quantizers)

        # Will only see this in the op_type section, not default
        if ConfigDictKeys.PARAMS in op_config:
            if module is None:
                logger.error('No module provided to set params for')
                raise AssertionError
            quantsim_wrapper = self._module_to_quantsim_wrapper_dict[module]
            for param_name in quantsim_wrapper.param_quantizers.keys():
                quantsim_param_name = MAP_PYTORCH_PARAM_NAME_TO_QUANTSIM_NAME.get(param_name, None)
                if quantsim_param_name is not None and quantsim_param_name in op_config[ConfigDictKeys.PARAMS]:
                    param_config = op_config[ConfigDictKeys.PARAMS][quantsim_param_name]
                    _set_config_for_param(quantsim_wrapper.param_quantizers[param_name], param_config)

    def _set_supergroup_configs(self, supergroups_configs: List[SupergroupType]):
        """
        Set supergroup specific configurations (fourth level of specificity in configuration file)
        :param supergroups_configs: Configurations for supergroups
        """
        patterns_with_callbacks = []
        for supergroup_config in supergroups_configs:
            callback = SupergroupConfigCallback(self._module_to_quantsim_wrapper_dict)
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
        input_ops = get_all_input_ops(self._conn_graph)
        for op in input_ops:
            if op.get_module() in self._named_modules_to_tensor_quantizers_dict:
                modified_tensor_quantizers = {}
                self._set_config_for_module(self._named_modules_to_tensor_quantizers_dict[op.get_module()],
                                            model_input_configs, modified_tensor_quantizers)

    def _set_model_output_configs(self, model_output_configs: ConfigType):
        """
        Set model output specific configurations (sixth level of specificity in configuration file)
        :param model_output_configs: Configuration for model outputs
        """
        output_ops = get_all_output_ops(self._conn_graph)
        for op in output_ops:
            if op.get_module() in self._named_modules_to_tensor_quantizers_dict:
                modified_tensor_quantizers = {}
                self._set_config_for_module(self._named_modules_to_tensor_quantizers_dict[op.get_module()],
                                            model_output_configs, modified_tensor_quantizers)


def _create_module_to_quantsim_wrapper_dict(model: torch.nn.Module) -> Dict[torch.nn.Module, QcQuantizeWrapper]:
    """
    Create a dictionary mapping modules in the model to corresponding quantsim wrappers
    :param model: Pytorch model with quantsim wrappers in place
    :return: Dictionary mapping modules in the model to corresponding quantsim wrappers
    """
    module_to_quantsim_wrapper_dict = {}
    for _, module in model.named_modules():
        if isinstance(module, QcQuantizeWrapper):
            module_to_quantsim_wrapper_dict[module._module_to_wrap] = module      # pylint: disable=protected-access
    return module_to_quantsim_wrapper_dict


def _get_all_ops_in_neighborhood(op: Op, direction: str, neighborhood=None):
    """
    Given an op and a direction, populate neighborhood dictionary with all ops adjacent to that op, and which direction
    they are adjacent in.  If a neighboring op has other connections in the same direction as the op we began with, ops
    connecting to the neighboring op in those other connections will also be part of the same neighborhood.
    :param op: Op to find neighboring ops from
    :param direction: Direction to search for neighboring ops (will be 'input' or 'output')
    :param neighborhood: Dictionary mapping neighboring ops to the direction which they connect to op.
    """
    if neighborhood is None:
        neighborhood = {}
    neighborhood[op] = direction
    if direction == 'input' and op.inputs:
        input_products = [inp for inp in op.inputs if inp.is_inter_module()]
        input_ops = [inp.producer for inp in input_products]
        for input_op in input_ops:
            if input_op not in neighborhood:
                neighborhood[input_op] = 'output'
                if input_op.type == 'Split':
                    _get_all_ops_in_neighborhood(input_op, 'input', neighborhood)
                    _get_all_ops_in_neighborhood(input_op, 'output', neighborhood)
                else:
                    _get_all_ops_in_neighborhood(input_op, 'output', neighborhood)
    elif op.output:
        output_ops = [consumer for consumer in op.output.consumers]
        for output_op in output_ops:
            if output_op not in neighborhood:
                neighborhood[output_op] = 'input'
                if output_op.type == 'Split':
                    _get_all_ops_in_neighborhood(output_op, 'output', neighborhood)
                else:
                    _get_all_ops_in_neighborhood(output_op, 'input', neighborhood)
    return neighborhood


def _set_config_for_param(param_quantizer: TensorQuantizer, param_config: ConfigType):
    """
    Set configurations for a specific param tensor quantizer
    :param param_quantizer: Tensor quantizer to set configurations for
    :param param_config: Configuration for the tensor quantizer
    """
    if ConfigDictKeys.IS_QUANTIZED in param_config:
        param_quantizer.enabled = param_config[ConfigDictKeys.IS_QUANTIZED]
    if ConfigDictKeys.IS_SYMMETRIC in param_config:
        param_quantizer.use_symmetric_encodings = \
            param_config[ConfigDictKeys.IS_SYMMETRIC]


def _modify_tensor_quantizers(input_output_tensor_quantizers: TensorQuantizersTupleType, setting_name: str,
                              quantizer_setting: bool, modified_tensor_quantizers: Dict[TensorQuantizer, Set]):
    """
    Modify the appropriate tensor quantizers for the given quantizer setting.  If a tensor quantizer has already been
    modified, compare the old setting with the new setting and assert if the settings conflict.
    :param input_output_tensor_quantizers: Tuple of 4 lists containing the following:
        - List of tensor quantizers to change if op's input quantizer setting is set to True
        - List of tensor quantizers to change if op's output quantizer setting is set to True
        - List of tensor quantizers to change if op's input quantizer setting is set to False
        - List of tensor quantizers to change if op's output quantizer setting is set to False
    :param setting_name: String representing the setting to be modified
    :param quantizer_setting: Boolean representing the new setting value
    :param modified_tensor_quantizers: Dictionary of tensor quantizers mapping to set of settings that have been changed
        for that tensor quantizer already.
    """
    setting_type = get_setting_type(setting_name)

    tensor_quantizers_to_modify = _get_tensor_quantizers_to_modify(input_output_tensor_quantizers, setting_name,
                                                                   quantizer_setting)
    for tensor_quantizer in tensor_quantizers_to_modify:
        if tensor_quantizer in modified_tensor_quantizers and \
                setting_type in modified_tensor_quantizers[tensor_quantizer]:
            # Tensor quantizer's setting has already been modified
            if setting_name in [ConfigDictKeys.IS_INPUT_QUANTIZED, ConfigDictKeys.IS_OUTPUT_QUANTIZED]:
                current_setting = tensor_quantizer.enabled
            else:
                current_setting = tensor_quantizer.use_symmetric_encodings
            if current_setting != quantizer_setting:
                logger.error('Conflicting tensor quantizer settings for symmetric encodings')
                raise AssertionError
        else:
            if setting_name in [ConfigDictKeys.IS_INPUT_QUANTIZED, ConfigDictKeys.IS_OUTPUT_QUANTIZED]:
                tensor_quantizer.enabled = quantizer_setting
            else:
                tensor_quantizer.use_symmetric_encodings = quantizer_setting
            if tensor_quantizer not in modified_tensor_quantizers:
                modified_tensor_quantizers[tensor_quantizer] = {setting_type}
            else:
                modified_tensor_quantizers[tensor_quantizer].add(setting_type)


def _get_tensor_quantizers_to_modify(input_output_tensor_quantizers: TensorQuantizersTupleType, setting_name: str,
                                     quantizer_setting: bool) -> List[TensorQuantizer]:
    """
    Given the tuple containing lists of tensor quantizers to modify under different situations, identify a list of
    tensor quantizers to modify given the specific setting name and quantizer setting.
    :param input_output_tensor_quantizers: Tuple of 4 lists containing the following:
        - List of tensor quantizers to change if op's input quantizer setting is set to True
        - List of tensor quantizers to change if op's output quantizer setting is set to True
        - List of tensor quantizers to change if op's input quantizer setting is set to False
        - List of tensor quantizers to change if op's output quantizer setting is set to False
    :param setting_name: String representing the setting to be modified
    :param quantizer_setting: Boolean representing the new setting value
    :return: List of tensor quantizers to modify given the specific setting name and quantizer setting.
    """
    input_true_list, output_true_list, input_false_list, output_false_list = input_output_tensor_quantizers

    if setting_name == ConfigDictKeys.IS_INPUT_QUANTIZED and quantizer_setting:
        return input_true_list
    if setting_name == ConfigDictKeys.IS_OUTPUT_QUANTIZED and quantizer_setting:
        return output_true_list
    if setting_name == ConfigDictKeys.IS_INPUT_QUANTIZED and not quantizer_setting:
        return input_false_list
    if setting_name == ConfigDictKeys.IS_OUTPUT_QUANTIZED and not quantizer_setting:
        return output_false_list
    if setting_name == ConfigDictKeys.IS_SYMMETRIC:
        # Will modify all input and output quantizers in the False case
        return input_false_list + output_false_list
    logger.error('Encountered unrecognized case for setting name %s, setting value %s', setting_name, quantizer_setting)
    raise AssertionError


def _report_unsupported_ops(quantsim_config: ConfigDictType):
    """ Log unsupported op types found in the config """
    type_mapper = OnnxConnectedGraphTypeMapper(onnx_pytorch_conn_graph_type_pairs)

    # Look for unsupported ops in op_type section
    op_type_configs = quantsim_config[ConfigDictKeys.OP_TYPE]
    for op in op_type_configs.keys():
        found_op = False
        for op_list in map_torch_types_to_onnx.values():
            if op in op_list:
                found_op = True
                break
        # Need the below for elementwise ops which will use the type_mapper instead of map_torch_types_to_onnx
        found_op = found_op or (type_mapper.get_conn_graph_type_from_onnx_type(op) is not None)
        if not found_op:
            logger.info('Unsupported op type %s', op)

    # Look for unsupported ops in supergroups section
    supergroups = quantsim_config[ConfigDictKeys.SUPERGROUPS]
    for supergroup in supergroups:
        for op in supergroup[ConfigDictKeys.OP_LIST]:
            if type_mapper.get_conn_graph_type_from_onnx_type(op) is None:
                logger.error('Unsupported op type %s', op)
                # Raising an error here since an unrecognized op will cause supergroup graph matching to fail
                raise AssertionError
