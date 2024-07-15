# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Utilities for mixed precision feature """

from typing import List, Dict
from collections import defaultdict
import tensorflow as tf

from aimet_common.amp.utils import CANDIDATE_WITH_DTYPE, get_effective_bitwidth
from aimet_common.cost_calculator import CostCalculator
from aimet_tensorflow.keras.layer_database import LayerDatabase
from aimet_tensorflow.keras.amp.quantizer_groups import QuantizerGroup

OUTPUT_STR = 'output'
WEIGHT_STR = 'weight'
INPUT_STR = 'input'


def find_bit_ops_reduction(quantizer_group: QuantizerGroup, mac_dict: Dict, read_var_op_parent_op_dict: Dict[str, str],
                           max_candidate: CANDIDATE_WITH_DTYPE, candidate: CANDIDATE_WITH_DTYPE) -> int:
    """
    Find bit ops reduction when Bitwidth changes from max_candidate to candidate
    :param quantizer_group: Quantizer group for which we want to find bit ops reduction
    :param mac_dict: Dictionary mapping modules to mac counts
    :param max_candidate: Default starting bitwidth
    :param candidate: Current BW candidate
    :return: Bit ops reduction
    """
    # pylint: disable=too-many-locals
    quantizer_group_dict = create_quantizer_module_dict(quantizer_group)
    bit_ops_reduction = 0
    (act_bw_max, act_dtype_max), (param_bw_max, param_dtype_max) = max_candidate
    (act_bw, act_dtype), (param_bw, param_dtype) = candidate

    act_bw_max = get_effective_bitwidth(act_dtype_max, act_bw_max)
    param_bw_max = get_effective_bitwidth(param_dtype_max, param_bw_max)
    act_bw = get_effective_bitwidth(act_dtype, act_bw)
    param_bw = get_effective_bitwidth(param_dtype, param_bw)

    if OUTPUT_STR in quantizer_group_dict and WEIGHT_STR in quantizer_group_dict:
        for quant_name in quantizer_group_dict[WEIGHT_STR]:
            op_name = read_var_op_parent_op_dict[quant_name]
            if op_name in mac_dict:
                bit_ops_reduction = bit_ops_reduction - mac_dict[op_name] * act_bw * param_bw + \
                                    mac_dict[op_name] * act_bw_max * param_bw_max
    elif WEIGHT_STR in quantizer_group_dict:
        for quant_name in quantizer_group_dict[WEIGHT_STR]:
            op_name = read_var_op_parent_op_dict[quant_name]
            if op_name in mac_dict:
                bit_ops_reduction = bit_ops_reduction - mac_dict[op_name] * act_bw_max * param_bw + \
                                    mac_dict[op_name] * act_bw_max * param_bw_max
    if INPUT_STR in quantizer_group_dict  and WEIGHT_STR in quantizer_group_dict:
        for quant_name in quantizer_group_dict[WEIGHT_STR]:
            op_name = read_var_op_parent_op_dict[quant_name]
            if op_name in mac_dict:
                bit_ops_reduction = bit_ops_reduction - mac_dict[op_name] * act_bw * param_bw_max + \
                                    mac_dict[op_name] * act_bw_max * param_bw_max
    return bit_ops_reduction


def create_mac_dict(model: tf.keras.Model) -> Dict[str, int]:
    """
    Create a dictionary mapping compressible modules (Conv and Linear) to mac counts
    :param model: Torch model to evaluate
    """
    layer_db = LayerDatabase(model)
    mac_dict = {}
    for compressible_layer in layer_db.get_compressible_layers().values():
        module_name = compressible_layer.name
        mac_dict[module_name] = CostCalculator.compute_layer_cost(compressible_layer).mac
    return mac_dict


def create_quantizer_module_dict(quantizer_group: QuantizerGroup) -> Dict:
    """
    Creates quantizer module dictionary from a quantizer group
    :param quantizer_group:
    :return: Dict of quantizer
    """
    quantizer_module_dict = defaultdict(list)
    for module_name in quantizer_group.input_quantizers:
        quantizer_module_dict[INPUT_STR].append(module_name)
    for module_name in quantizer_group.output_quantizers:
        quantizer_module_dict[OUTPUT_STR].append(module_name)
    if quantizer_group.parameter_quantizers:
        for module_name in quantizer_group.parameter_quantizers:
            quantizer_module_dict[WEIGHT_STR].append(module_name)

    return quantizer_module_dict


def calculate_running_bit_ops(mac_dict: Dict[str, int],
                              quantizer_group: QuantizerGroup,
                              read_var_op_parent_op_dict: Dict[str, str],
                              module_bitwidth_dict: Dict[str, List[int]],
                              max_candidate: CANDIDATE_WITH_DTYPE,
                              new_candidate: CANDIDATE_WITH_DTYPE,
                              running_bit_ops: int) -> int:
    """
    Returns new running bit ops given previous running bit ops value and the current quantizer to change bitwidth of
    :param mac_dict: Dictionary mapping modules to mac count of the module (only Conv and Linear modules are present in
    the dictionary)
    :param quantizer_group: TensorQuantizer to change bitwidth of
    :param module_bitwidth_dict: Dictionary mapping modules to tuples of quantizer types and most recently used
    bitwidths for each quantizer type (only tracks input and weight quantizers)
    :param max_candidate: Maximum bitwidth and data type for the TensorQuantizer
    :param new_candidate: New bitwidth and data type for the TensorQuantizer
    :param running_bit_ops: previous running bit ops count
    """
    # pylint: disable=too-many-locals
    def _calculate_bit_ops(quantizer_type: str, running_bit_ops: int, module_name: str) -> int:
        """
        Helper function to compute bit ops for weight or inputs feeding into an op
        """
        if module_name in mac_dict:
            if module_name in module_bitwidth_dict.keys():
                # If an input or weight quantizer of the module was quantized previously, there will be an entry in
                # the dictionary.  Then, we simply need to update the corresponding index (input or weight) with
                # the new bitwidth.

                # Subtract the previous bitops count for this module (will add on the new bitops count for this
                # module later, taking new bitwidth value into account)
                running_bit_ops -= (module_bitwidth_dict[module_name][input_index] *
                                    module_bitwidth_dict[module_name][weight_index] *
                                    mac_dict[module_name])
                if quantizer_type == 'input':
                    module_bitwidth_dict[module_name][input_index] = activation_bw_new
                else:
                    module_bitwidth_dict[module_name][weight_index] = param_bw_new
            else:
                # New entry in module_bitwidth_dict needs to be made.  One index will be default_bitwidth, and the
                # other index will be the new bitwidth we are currently using.

                # Subtract the previous bitops count for this module (will add on the new bitops count for this
                # module later, taking new bitwidth value into account)
                running_bit_ops -= (activation_bw_max * param_bw_max) * mac_dict[module_name]
                if quantizer_type == 'input':
                    module_bitwidth_dict[module_name] = [activation_bw_new, param_bw_max]
                else:
                    module_bitwidth_dict[module_name] = [activation_bw_max, param_bw_new]
            # Add new bitops count to the running bit ops value, taking into account the updated bitwidths for
            # input and weight quantizers.
            running_bit_ops += (module_bitwidth_dict[module_name][input_index] *
                                module_bitwidth_dict[module_name][weight_index] *
                                mac_dict[module_name])
        return running_bit_ops

    input_index = 0
    weight_index = 1

    quantizer_module_dict = create_quantizer_module_dict(quantizer_group)

    (activation_bw_max, act_dtype_max), (param_bw_max, param_dtype_max) = max_candidate
    (activation_bw_new, act_dtype), (param_bw_new, param_dtype) = new_candidate

    activation_bw_max = get_effective_bitwidth(act_dtype_max, activation_bw_max)
    param_bw_max = get_effective_bitwidth(param_dtype_max, param_bw_max)
    activation_bw_new = get_effective_bitwidth(act_dtype, activation_bw_new)
    param_bw_new = get_effective_bitwidth(param_dtype, param_bw_new)

    if 'output' in quantizer_module_dict and 'weight' in quantizer_module_dict:
        for module_name in quantizer_module_dict[WEIGHT_STR]:
            # Output of previous op is the input to this op
            parent_op_name = read_var_op_parent_op_dict[module_name]
            running_bit_ops = _calculate_bit_ops('input', running_bit_ops, parent_op_name)

    if 'input' in quantizer_module_dict:
        for module_name in quantizer_module_dict[WEIGHT_STR]:
            parent_op_name = read_var_op_parent_op_dict[module_name]
            running_bit_ops = _calculate_bit_ops('input', running_bit_ops, parent_op_name)

    if 'weight' in quantizer_module_dict:
        for module_name in quantizer_module_dict[WEIGHT_STR]:
            parent_op_name = read_var_op_parent_op_dict[module_name]
            running_bit_ops = _calculate_bit_ops('weight', running_bit_ops, parent_op_name)

    return running_bit_ops


def find_read_var_op_parent_op_dict(model: tf.keras.Model) -> Dict[str, str]:
    """
    Find mapping of op (only Conv and Linear ops) names to their corresponding read variable op names
    :param graph: Tensorflow graph
    :return: Dictionary mapping read variable op name -> parent op name
    """
    read_var_op_parent_op_dict = {}

    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.BatchNormalization, tf.keras.layers.Normalization, tf.keras.layers.LayerNormalization, tf.keras.layers.UnitNormalization)):
            for param in layer.weights:
                param_name = param.name.split(':')[0]
                read_var_op_parent_op_dict[param_name] = layer.name
    return read_var_op_parent_op_dict
