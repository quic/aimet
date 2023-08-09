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

import torch
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch import onnx_utils
from aimet_torch.pro.utils.translation_mappping import op_to_weight_index_map, qnn_datatype_to_aimet_map, \
    aimet_op_to_qnn_op_name_map
from aimet_common.defs import QuantizationDataType
from aimet_common.libpymopro import ModelOpDefParser


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

# pylint: disable=too-many-locals
def get_backend_info(op_names: List[str], master_opdef_path: str, backend_opdef_path: str) -> Dict[str, SupportedBackendInfo]:
    """
    Returns backend constraints

    :param op_names: List of op names
    :param master_opdef_path: Master Op. Def. file path
    :param backend_opdef_path: Backend Op. Def. file path
    :return: Dict of op_name and it's constraints
    """
    op_names_according_to_backend = copy.deepcopy(op_names)
    for i, op_name in enumerate(op_names):
        if op_name in aimet_op_to_qnn_op_name_map.keys():
            op_names_according_to_backend[i] = aimet_op_to_qnn_op_name_map[op_name]

    parser = ModelOpDefParser(master_opdef_path, backend_opdef_path, op_names_according_to_backend)

    op_and_supported_backend_info_map = {}

    for i, op_name in enumerate(op_names):
        op_name_in_opdef = op_names_according_to_backend[i]

        if op_name not in op_and_supported_backend_info_map:
            datatype_constraints_size = parser.get_size(op_name_in_opdef)

            output_datatype_constraints_size = datatype_constraints_size['output_size']

            activation_constraints = []
            for output_index in range(output_datatype_constraints_size):
                try:
                    datatype_constraints = parser.get_output_datatype(op_name_in_opdef, output_index)
                    for datatype in datatype_constraints:
                        activation_constraints.append(qnn_datatype_to_aimet_map[datatype])
                # pylint: disable=bare-except
                except:
                    #Parser API will throw appropriate error message if not able to get output datattypes
                    pass

            weight_constraints = []
            if op_name_in_opdef in op_to_weight_index_map.keys():
                try:
                    datatype_constraints = parser.get_input_datatype(op_name_in_opdef, op_to_weight_index_map[op_name_in_opdef])
                    for datatype in datatype_constraints:
                        weight_constraints.append(qnn_datatype_to_aimet_map[datatype])
                # pylint: disable=bare-except
                except:
                    #Parser API will throw appropriate error message if not able to get input datatypes
                    pass

            supported_backend_info = SupportedBackendInfo(activation_constraints, weight_constraints)
            op_and_supported_backend_info_map[op_name] = supported_backend_info

    return op_and_supported_backend_info_map

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
        if input_quantizer.enabled:
            input_quantizer.data_type = dtype_to_set_for_activation
            input_quantizer.bitwidth = bitwidth_to_set_for_activation
            logger.info("Setting datatype and bitwidth of %s input activations to %s and %s according to backend constraints.", module_type,
                        str(dtype_to_set_for_activation), str(bitwidth_to_set_for_activation))

def populate_backend_info(model: torch.nn.Module, module_types: List[str], master_opdef_file_path: str,
                          backend_opdef_file_path: str, quantsim_info: QuantsimInfo) -> Dict[str, List]:
    """
    Driver function to get and set backend constraints for model

    :param model: Model
    :param module_types: List of module names for whom backend constraints are retrieved and set accordingly
    :param master_opdef_file_path: Master Op. Def. file path
    :param backend_opdef_file_path: Backend Op. Def. file path
    :param quantsim_info: Quantization info for model

    :return Dict of op to it's supported kernels
    """
    supported_kernels = get_backend_info(module_types, master_opdef_file_path, backend_opdef_file_path)

    default_act_kernel = [{'bitwidth': quantsim_info.activation_bitwidth, 'dtype' : quantsim_info.data_type}]
    default_weight_kernel = [{'bitwidth' : quantsim_info.param_bitwidth, 'dtype' : quantsim_info.data_type}]
    op_to_supported_kernels = {}
    op_to_supported_kernels['defaults'] = [get_supported_kernel_in_dict_format(default_act_kernel[0], default_weight_kernel[0])]

    # pylint:disable=too-many-nested-blocks
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
                    # pylint: disable=protected-access
                    if type(module._module_to_wrap) in onnx_utils.map_torch_types_to_onnx.keys(): # pylint: disable=unidiomatic-typecheck
                        onnx_types = onnx_utils.map_torch_types_to_onnx.get(type(module._module_to_wrap))
                        for op in onnx_types:
                            if op not in op_to_supported_kernels.keys():
                                op_to_supported_kernels[op] = supported_kernels_for_op

                #set bitwidth and dtype of module's weights according to supported_kernel
                if 'weight' in module.param_quantizers and is_weight_constraint_present:
                    set_datatype_bitwidth_for_weights(module, backend_weight_constraints, module_type)

                #set bitwidth and dtype of module's activation quantizers according to supported_kernel
                if  is_act_constraint_present:
                    set_datatype_bitwidth_for_activations(module, backend_act_constraints, module_type)

    return op_to_supported_kernels
