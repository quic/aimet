#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Utilities for backend aware quantization """

# pylint: disable=import-error, no-name-in-module
import copy
import logging
from typing import List, Dict
from dataclasses import dataclass
import json
import torch
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch import onnx_utils
from aimet_torch.translation_mapping import op_to_weight_index_map, backend_datatype_to_aimet_map,\
    aimet_op_to_backend_op_name_map
from aimet_common.defs import QuantizationDataType
from aimet_common.quantsim_config.json_config_importer import ConfigDictKeys
from aimet_common.libpymo import ModelOpDefParser


logger = logging.getLogger()
default_index_to_set_bitwidth_dtype = 0

@dataclass
class SupportedBackendInfo:
    """
    Class which holds backend constraint information
    """

    activation_constraints: List[Dict]
    weights_constraints: List[Dict]

@dataclass
class QuantsimInfo:
    """
    Class which holds info. about quantsim object
    """

    activation_bitwidth: int
    param_bitwidth: int
    data_type: QuantizationDataType

def merge_constraints_from_xmls(op_name: str, supported_backend_info: SupportedBackendInfo, merged_op_and_supported_backend_info_map: Dict[str, List]):

    """
    Merges backend constraints in supported_backend_info to merged_op_and_supported_backend_info_map

    :param op_name: op_name in model
    :param supported_backend_info: backend constraints to be merged in merged_op_and_supported_backend_info_map
    :param merged_op_and_supported_backend_info_map: Map which holds info. about op to it's backend constraints
    """
    if op_name in merged_op_and_supported_backend_info_map:
        existing_weight_constraints = merged_op_and_supported_backend_info_map[op_name].weights_constraints
        existing_activation_constraints = merged_op_and_supported_backend_info_map[op_name].activation_constraints

        merged_weight_constraints = existing_weight_constraints + [x for x in supported_backend_info.weights_constraints
                                                                   if x not in existing_weight_constraints]
        merged_activation_constraints = existing_activation_constraints + [x for x in supported_backend_info.activation_constraints
                                                                           if x not in existing_activation_constraints]
        merged_op_and_supported_backend_info_map[op_name].weights_constraints = merged_weight_constraints
        merged_op_and_supported_backend_info_map[op_name].activation_constraints = merged_activation_constraints

    else:
        merged_op_and_supported_backend_info_map[op_name] = supported_backend_info

def get_activation_constraints(parser: ModelOpDefParser, op_name_in_opdef: str) -> List[Dict]:
    """
    Returns activation constraints for the op using parser
    :param parser: ModelOpDefParser object
    :param op_name_in_opdef: Op name whose activation constraints needs to be fetched
    :return: List of activation constraints for the op in {bitwidth, dtype} Dict form
    """
    datatype_constraints_size = parser.get_size(op_name_in_opdef)

    output_datatype_constraints_size = datatype_constraints_size['output_size']

    activation_constraints = []
    for output_index in range(output_datatype_constraints_size):
        try:
            datatype_constraints = parser.get_output_datatype(op_name_in_opdef, output_index)
            for datatype in datatype_constraints:
                if backend_datatype_to_aimet_map[datatype] not in activation_constraints:
                    activation_constraints.append(backend_datatype_to_aimet_map[datatype])
        # pylint: disable=bare-except
        except:
            #Parser API will throw appropriate error message if not able to get output datattypes
            pass

    return activation_constraints

def get_weight_constraints(parser: ModelOpDefParser, op_name_in_opdef: str) -> List[Dict]:
    """
    Returns weight constraints for the op using parser
    :param parser: ModelOpDefParser object
    :param op_name_in_opdef: Op name whose weight constraints needs to be fetched
    :return: List of weight constraints for the op in {bitwidth, dtype} Dict form
    """
    weight_constraints = []
    if op_name_in_opdef in op_to_weight_index_map.keys():
        try:
            datatype_constraints = parser.get_input_datatype(op_name_in_opdef, op_to_weight_index_map[op_name_in_opdef])
            for datatype in datatype_constraints:
                if backend_datatype_to_aimet_map[datatype] not in weight_constraints:
                    weight_constraints.append(backend_datatype_to_aimet_map[datatype])
        # pylint: disable=bare-except
        except:
            #Parser API will throw appropriate error message if not able to get input datatypes
            pass
    return weight_constraints

def get_backend_info(op_names: List[str], master_opdef_path: str, backend_opdef_paths: List[str]) -> Dict[str, SupportedBackendInfo]:
    """
    Returns backend constraints

    :param op_names: List of op names
    :param master_opdef_path: Master Op. Def. file path
    :param backend_opdef_paths: List of Backend Op. Def. file path
    :return: Dict of op_name and it's constraints
    """
    op_names_according_to_backend = copy.deepcopy(op_names)
    merged_opname_supported_backend_info_map = {}

    for i, op_name in enumerate(op_names):
        if op_name in aimet_op_to_backend_op_name_map.keys():
            op_names_according_to_backend[i] = aimet_op_to_backend_op_name_map[op_name]

    for backend_opdef_path in backend_opdef_paths:
        parser = ModelOpDefParser(master_opdef_path, backend_opdef_path, op_names_according_to_backend)

        op_and_supported_backend_info_map = {}

        for i, op_name in enumerate(op_names):
            op_name_in_opdef = op_names_according_to_backend[i]

            if op_name_in_opdef not in op_and_supported_backend_info_map:

                activation_constraints = get_activation_constraints(parser, op_name_in_opdef)
                weight_constraints = get_weight_constraints(parser, op_name_in_opdef)

                supported_backend_info = SupportedBackendInfo(activation_constraints, weight_constraints)
                op_and_supported_backend_info_map[op_name_in_opdef] = supported_backend_info

                merge_constraints_from_xmls(op_name_in_opdef, supported_backend_info, merged_opname_supported_backend_info_map)

    return merged_opname_supported_backend_info_map

def get_supported_kernel_in_dict_format(act_constraint: Dict, weight_constraint: Dict) -> Dict:
    """
    Returns supported kernel in dict format

    :param act_constraint: Activation constraint
    :param weight_constraint: Weight constraint
    :return: supported kernel in dict format
    """
    supported_kernel_in_dict_format = {'activation':{'bitwidth': act_constraint['bitwidth'], 'dtype':act_constraint['dtype']},
                                       'param':{'bitwidth': weight_constraint['bitwidth'], 'dtype':weight_constraint['dtype']}}
    return supported_kernel_in_dict_format

def set_and_return_supported_kernels(module: torch.nn.Module, backend_act_constraints: List[Dict], backend_weight_constraints: List[Dict], module_type: str) -> List[Dict]:
    """
    Set supported kernels for module

    :param module: torch.nn.Module
    :param backend_act_constraints: Activation constraints for op
    :param backend_weight_constraints: Weight constraints for op
    :param module_type: Module name to display in logger

    :return supported kernels for op
    """
    supported_kernels = []
    supported_kernels_in_dict_format = []
    for act_constraint in backend_act_constraints:
        for weight_constraint in backend_weight_constraints:
            supported_kernel = ((act_constraint['bitwidth'], act_constraint['dtype']),
                                (weight_constraint['bitwidth'], weight_constraint['dtype']))
            if supported_kernel not in supported_kernels:
                supported_kernels.append(supported_kernel)
                supported_kernels_in_dict_format.append(get_supported_kernel_in_dict_format(act_constraint, weight_constraint))

    module.supported_kernels = supported_kernels
    logger.info("Setting supported kernels of %s to %s", module_type, str(supported_kernels))
    return supported_kernels_in_dict_format

def set_datatype_bitwidth_for_weights(module: torch.nn.Module, backend_weight_constraints: List[Dict], module_type: str):
    """
    Set datatype, bitwidth for module having weights according to constraints

    :param module: torch.nn.Module
    :param backend_weight_constraints: Weight constraints for op
    :param module_type: Module name to display in logger
    """
    weight_bitwidth = module.param_quantizers['weight'].bitwidth
    weight_dtype = module.param_quantizers['weight'].data_type
    default_dtype_bitwidth_match = False
    supported_dtype_for_op_weight = backend_weight_constraints[default_index_to_set_bitwidth_dtype]['dtype']
    supported_bitwidth_for_op_weight = backend_weight_constraints[default_index_to_set_bitwidth_dtype]['bitwidth']

    for weight_constraint in backend_weight_constraints:
        if weight_bitwidth == weight_constraint['bitwidth'] and weight_dtype == weight_constraint['dtype']:
            default_dtype_bitwidth_match = True
            break

    if not default_dtype_bitwidth_match:
        module.param_quantizers['weight'].data_type = supported_dtype_for_op_weight
        module.param_quantizers['weight'].bitwidth = supported_bitwidth_for_op_weight
        logger.info("Setting datatype and bitwidth of %s weights to %s and %s according to backend constraints", module_type,
                    str(supported_dtype_for_op_weight), str(supported_bitwidth_for_op_weight))

def set_datatype_bitwidth_for_activations(module: torch.nn.Module, backend_act_constraints: List[Dict], module_type: str):
    """
    Set datatype, bitwidth for module's activations having according to constraints

    :param module: torch.nn.Module
    :param backend_act_constraints: Output activation constraints for op
    :param module_type: Module name to display in logger
    """
    dtype_to_set_for_activation = backend_act_constraints[default_index_to_set_bitwidth_dtype]['dtype']
    bitwidth_to_set_for_activation = backend_act_constraints[default_index_to_set_bitwidth_dtype]['bitwidth']
    for output_quantizer in module.output_quantizers:
        act_bitwidth = output_quantizer.bitwidth
        act_dtype = output_quantizer.data_type
        default_dtype_bitwidth_match = False

        for act_constraint in backend_act_constraints:
            if act_bitwidth == act_constraint['bitwidth'] and act_dtype == act_constraint['dtype']:
                default_dtype_bitwidth_match = True
                break

        if not default_dtype_bitwidth_match:
            output_quantizer.data_type = dtype_to_set_for_activation
            output_quantizer.bitwidth = bitwidth_to_set_for_activation
            logger.info("Setting datatype and bitwidth of %s output activations to %s and %s according to backend constraints.", module_type,
                        str(dtype_to_set_for_activation), str(bitwidth_to_set_for_activation))

    for input_quantizer in module.input_quantizers:
        if input_quantizer.enabled and not default_dtype_bitwidth_match:
            input_quantizer.data_type = dtype_to_set_for_activation
            input_quantizer.bitwidth = bitwidth_to_set_for_activation
            logger.info("Setting datatype and bitwidth of %s input activations to %s and %s according to backend constraints.", module_type,
                        str(dtype_to_set_for_activation), str(bitwidth_to_set_for_activation))

def set_supported_kernel_for_op(module: torch.nn.Module, op_to_supported_kernels: Dict, supported_kernels_for_op: List[Dict]):
    """
    Sets supported kernels for the op
    :param module: Module
    :param op_to_supported_kernels: Dict of op to it's supported kernels
    :param supported_kernels_for_op: SUpported kernels for the op
    """

    # pylint: disable=protected-access
    if type(module._module_to_wrap) in onnx_utils.map_torch_types_to_onnx.keys(): # pylint: disable=unidiomatic-typecheck
        onnx_types = onnx_utils.map_torch_types_to_onnx.get(type(module._module_to_wrap))
        for op in onnx_types:
            if op not in op_to_supported_kernels.keys():
                op_to_supported_kernels[op] = supported_kernels_for_op

# pylint: disable=too-many-locals
def populate_backend_info(model: torch.nn.Module, module_types: List[str], master_opdef_file_path: str,
                          backend_opdef_file_paths: List[str], quantsim_info: QuantsimInfo) -> Dict[str, List]:
    """
    Driver function to get and set backend constraints for model

    :param model: Model
    :param module_types: List of module names for whom backend constraints are retrieved and set accordingly
    :param master_opdef_file_path: Master Op. Def. file path
    :param backend_opdef_file_paths: List of Backend Op. Def. file path
    :param quantsim_info: Quantization info for model

    :return Dict of op to it's supported kernels
    """
    supported_kernels = get_backend_info(module_types, master_opdef_file_path, backend_opdef_file_paths)

    default_act_kernel = [{'bitwidth': quantsim_info.activation_bitwidth, 'dtype' : quantsim_info.data_type}]
    default_weight_kernel = [{'bitwidth' : quantsim_info.param_bitwidth, 'dtype' : quantsim_info.data_type}]
    op_to_supported_kernels = {}
    op_to_supported_kernels['defaults'] = [get_supported_kernel_in_dict_format(default_act_kernel[0], default_weight_kernel[0])]

    for module in model.modules():
        if isinstance(module, QcQuantizeWrapper):
            # pylint: disable=protected-access
            module_type = module._module_to_wrap.__class__.__name__
            if module_type in supported_kernels.keys():
                backend_supported_info = supported_kernels[module_type]
                is_weight_constraint_present = False
                is_act_constraint_present = False
                backend_weight_constraints = default_weight_kernel
                backend_act_constraints = default_act_kernel

                if backend_supported_info.weights_constraints:
                    is_weight_constraint_present = True
                    backend_weight_constraints = backend_supported_info.weights_constraints

                if backend_supported_info.activation_constraints:
                    is_act_constraint_present = True
                    backend_act_constraints = backend_supported_info.activation_constraints

                #set module's supported kernels
                if is_weight_constraint_present or is_act_constraint_present:
                    supported_kernels_for_op = set_and_return_supported_kernels(module, backend_act_constraints, backend_weight_constraints, module_type)
                    set_supported_kernel_for_op(module, op_to_supported_kernels, supported_kernels_for_op)

                #set bitwidth and dtype of module's weights according to supported_kernel
                if 'weight' in module.param_quantizers and is_weight_constraint_present:
                    set_datatype_bitwidth_for_weights(module, backend_weight_constraints, module_type)

                #set bitwidth and dtype of module's activation quantizers according to supported_kernel
                if  is_act_constraint_present:
                    set_datatype_bitwidth_for_activations(module, backend_act_constraints, module_type)

    return op_to_supported_kernels

def get_constraint_accrording_to_json_config(constraint: Dict) -> Dict:
    """
    Returns supported kernel constraint according to JSON file
    :param constraint: Activation or Param constraint
    :return: Dict format of suported kernel
    """
    constraint_according_json = {}
    constraint_according_json[ConfigDictKeys.BITWIDTH] = constraint[ConfigDictKeys.BITWIDTH]
    if constraint[ConfigDictKeys.DTYPE] == QuantizationDataType.int:
        constraint_according_json[ConfigDictKeys.DTYPE] = "int"
    elif constraint[ConfigDictKeys.DTYPE] == QuantizationDataType.float:
        constraint_according_json[ConfigDictKeys.DTYPE] = "float"
    return constraint_according_json

def get_supported_kernels_from_backend_info(supported_backend_info: SupportedBackendInfo) -> List[Dict]:
    """
    Returns supported for JSON config file from backend constraints

    :param supported_backend_info: Object which stores
    :return: List of supported kernels
    """
    supported_kernels = []
    for activation_constraint in supported_backend_info.activation_constraints:
        if supported_backend_info.weights_constraints:
            for weight_constraint in supported_backend_info.weights_constraints:

                if weight_constraint[ConfigDictKeys.DTYPE] == activation_constraint[ConfigDictKeys.DTYPE]:
                    json_act_constraint = get_constraint_accrording_to_json_config(activation_constraint)
                    json_param_constraint = get_constraint_accrording_to_json_config(weight_constraint)

                    supported_kernel = {ConfigDictKeys.ACTIVATION: json_act_constraint,
                                        ConfigDictKeys.PARAM: json_param_constraint}

                    if supported_kernel not in supported_kernels:
                        supported_kernels.append(supported_kernel)

        else:
            json_act_constraint = get_constraint_accrording_to_json_config(activation_constraint)
            supported_kernel = {ConfigDictKeys.ACTIVATION: json_act_constraint}

            if supported_kernel not in supported_kernels:
                supported_kernels.append(supported_kernel)

    return supported_kernels

def populate_supported_kernels_in_json_config(master_opdef_file_path: str,
                                              backend_opdef_file_paths: List[str],
                                              json_config_file_path: str):
    """
    Populate supported kernels per op basis in JSON config file

    :param master_opdef_file_path: Master opdef file path
    :param backend_opdef_file_paths: Backend opdef file paths
    :param json_config_file_path: Config file in which supported kernels will be populated
    """
    supported_kernels_dict = get_backend_info(list(aimet_op_to_backend_op_name_map.keys()), master_opdef_file_path, backend_opdef_file_paths)

    op_types_with_no_constraints = [key for key in supported_kernels_dict
                                    if not supported_kernels_dict[key].activation_constraints and
                                    not supported_kernels_dict[key].weights_constraints]

    for op_type in op_types_with_no_constraints:
        del supported_kernels_dict[op_type]

    with open(json_config_file_path) as file:
        quantsim_config = json.load(file)

    for backend_op_type in supported_kernels_dict:
        supported_kernels = get_supported_kernels_from_backend_info(supported_kernels_dict[backend_op_type])
        if backend_op_type not in quantsim_config[ConfigDictKeys.OP_TYPE]:
            quantsim_config[ConfigDictKeys.OP_TYPE][backend_op_type] = {}

        quantsim_config[ConfigDictKeys.OP_TYPE][backend_op_type][ConfigDictKeys.SUPPORTED_KERNELS] = supported_kernels

    with open(json_config_file_path, 'w') as file:
        json.dump(quantsim_config, file, indent=4)
