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
""" Commonly used utilities for configuring quantsim """

from typing import overload, Callable, List, Optional, Tuple, Type
import torch
from aimet_common.utils import AimetLogger
from aimet_torch.v2.quantsim.quantsim import QuantizationSimModel
from aimet_torch.v2.quantization.affine import QuantizeDequantize, GroupedBlockQuantizeDequantize
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

def _parse_arg_for_condition(arg):
    """ Transform the given arg into a corresponding condition expression """
    if isinstance(arg, List) and isinstance(arg[0], type) and issubclass(arg[0], torch.nn.Module):
        module_type = arg
        condition = lambda module: isinstance(module, tuple(module_type))
    elif isinstance(arg, List):
        if not isinstance(arg[0], torch.nn.Module):
            raise RuntimeError('List given as arg must contain either torch.nn.Module types or specific '
                               'torch.nn.Module objects.')
        qmodules = arg
        condition = lambda module: module in qmodules
    else:
        condition = arg
    return condition


def _get_weight_quantizer_shape_from_block_size(quant_layer: torch.nn.Module, block_size: Tuple[int, ...]) \
        -> Optional[List[int]]:
    """ Given a block size, get the corresponding weight quantizer shape """
    weight_shape = quant_layer.weight.shape
    assert len(block_size) == len(weight_shape)
    quantizer_shape = []
    for idx, shape in enumerate(weight_shape):
        if block_size[idx] != -1:
            if not shape % block_size[idx] == 0:
                return None

            quantizer_shape.append(shape // block_size[idx])
        else:
            quantizer_shape.append(1)
    return quantizer_shape


@overload
def set_activation_quantizers_to_float(sim: QuantizationSimModel, module_type: List[Type[torch.nn.Module]],
                                       exponent_bits: int = None, mantissa_bits: int = None,
                                       dtype: torch.dtype = None):
    """ Set activation quantizers of the given module type to float """


@overload
def set_activation_quantizers_to_float(sim: QuantizationSimModel, qmodules: List[torch.nn.Module],
                                       exponent_bits: int = None, mantissa_bits: int = None, dtype: torch.dtype = None):
    """ Set activation quantizers of the given qmodules to float """


@overload
def set_activation_quantizers_to_float(sim: QuantizationSimModel, condition: Callable[[torch.nn.Module], bool],
                                       exponent_bits: int = None, mantissa_bits: int = None, dtype: torch.dtype = None):
    """ Set activation quantizers of all the modules that satisfy the given condition to float. """


def set_activation_quantizers_to_float(sim: QuantizationSimModel, arg, exponent_bits: int = None,
                                       mantissa_bits: int = None, dtype: torch.dtype = None):
    """
    Set activation quantizers of modules to float.

    :param sim: Quantsim to set activation quantizers for
    :param arg: Argument determining which modules to set. This can consist of either:
        1. A list of torch.nn.Module types, in which case all modules whose type is in the list will be set
        2. A list of torch.nn.Modules, in which case all modules in the list will be set
        3. A callable function which takes a torch.nn.Module as input and returns True if the module is to be set, False
           otherwise
    :param exponent_bits: Number of exponent bits to simulate
    :param mantissa_bits: Number of mantissa bits to simulate
    :param dtype: torch.dtype to simulate. This argument is mutually exclusive with exponent_bits and mantissa_bits.
    """

    condition = _parse_arg_for_condition(arg)
    _set_activation_quantizers_to_float(sim, condition, exponent_bits, mantissa_bits, dtype)


def _set_activation_quantizers_to_float(sim: QuantizationSimModel, condition: Callable[[torch.nn.Module], bool],
                                        exponent_bits: int = None, mantissa_bits: int = None,
                                        dtype: torch.dtype = None):
    """ Set activation quantizers of all the modules that satisfy the given condition to float. """
    for _, quant_layer in sim.named_qmodules():
        if condition(quant_layer):
            for idx, quantizer in enumerate(quant_layer.input_quantizers):
                if quantizer is not None:
                    quant_layer.input_quantizers[idx] = FloatQuantizeDequantize(exponent_bits, mantissa_bits, dtype)

            for idx, quantizer in enumerate(quant_layer.output_quantizers):
                if quantizer is not None:
                    quant_layer.output_quantizers[idx] = FloatQuantizeDequantize(exponent_bits, mantissa_bits, dtype)


@overload
def set_blockwise_quantization_for_weights(sim: QuantizationSimModel, module_type: List[Type[torch.nn.Module]],
                                           bitwidth: int, symmetric: bool, block_size: Tuple[int, ...]):
    """ Set weight parameter quantizers of the given module type to blockwise """


@overload
def set_blockwise_quantization_for_weights(sim: QuantizationSimModel, qmodules: List[torch.nn.Module], bitwidth: int,
                                           symmetric: bool, block_size: Tuple[int, ...]):
    """ Set weight parameter quantizers of the given modules to blockwise """


@overload
def set_blockwise_quantization_for_weights(sim: QuantizationSimModel, condition: Callable[[torch.nn.Module], bool],
                                           bitwidth: int, symmetric: bool, block_size: Tuple[int, ...]):
    """ Set weight parameter quantizers of modules that satisfy the given condition to blockwise """


def set_blockwise_quantization_for_weights(sim: QuantizationSimModel, arg, bitwidth: int, symmetric: bool,
                                           block_size: Tuple[int, ...]):
    """
    Set weight parameter quantizers of modules to blockwise.

    :param sim: Quantsim to set activation quantizers for
    :param arg: Argument determining which modules to set. This can consist of either:
        1. A list of torch.nn.Module types, in which case all modules whose type is in the list will be set
        2. A list of torch.nn.Modules, in which case all modules in the list will be set
        3. A callable function which takes a torch.nn.Module as input and returns True if the module is to be set, False
           otherwise
    :param bitwidth: Bitwidth for affine quantization
    :param symmetric: True if affine quantization is symmetric, False otherwise
    :param block_size: Block size for affine quantization
    """
    condition = _parse_arg_for_condition(arg)
    _set_blockwise_quantization_for_weights(sim, condition, bitwidth, symmetric, block_size)


def _get_layers_to_quantizer_shapes_for_block_size(sim, condition, block_size):
    layer_to_quantizer_shape_dict = {}
    invalid_layers_for_block_size = []
    for name, quant_layer in sim.named_qmodules():
        if condition(quant_layer) and 'weight' in quant_layer.param_quantizers and \
                quant_layer.param_quantizers['weight'] is not None:
            assert hasattr(quant_layer, 'weight')
            layer_to_quantizer_shape_dict[quant_layer] = \
                _get_weight_quantizer_shape_from_block_size(quant_layer, block_size)
            if layer_to_quantizer_shape_dict[quant_layer] is None:
                invalid_layers_for_block_size.append((name, quant_layer.weight.shape))

    if invalid_layers_for_block_size:
        for name, shape in invalid_layers_for_block_size:
            error_str = f"Quant layer {name} has shape {shape} which does not align with block_size {block_size}. " \
                        f"Each dimension's shape must divide evenly with the corresponding block size."
            logger.error(error_str)
            raise RuntimeError('Quant layers found whose weights do not align with block size.')

    return layer_to_quantizer_shape_dict


def _set_blockwise_quantization_for_weights(sim: QuantizationSimModel, condition: Callable[[torch.nn.Module], bool],
                                            bitwidth: int, symmetric: bool, block_size: Tuple[int, ...]):
    """ Set weight parameter quantizers of modules that satisfy the given condition to blockwise """
    layer_to_quantizer_shape_dict = _get_layers_to_quantizer_shapes_for_block_size(sim, condition, block_size)
    for layer, quantizer_shape in layer_to_quantizer_shape_dict.items():
        layer.param_quantizers['weight'] = QuantizeDequantize(quantizer_shape, bitwidth, symmetric, None, block_size)


@overload
def set_grouped_blockwise_quantization_for_weights(sim: QuantizationSimModel, module_type: List[Type[torch.nn.Module]],
                                                   bitwidth: int, symmetric: bool, decompressed_bw: int,
                                                   block_size: Tuple[int, ...], block_grouping):
    """ Set weight parameter quantizers of the given module type to grouped blockwise """


@overload
def set_grouped_blockwise_quantization_for_weights(sim: QuantizationSimModel, qmodules: List[torch.nn.Module],
                                                   bitwidth: int, symmetric: bool, decompressed_bw: int,
                                                   block_size: Tuple[int, ...], block_grouping: Tuple[int, ...]):
    """ Set weight parameter quantizers of the given modules to grouped blockwise """


@overload
def set_grouped_blockwise_quantization_for_weights(sim: QuantizationSimModel,
                                                   condition: Callable[[torch.nn.Module], bool],
                                                   bitwidth: int, symmetric: bool, decompressed_bw: int,
                                                   block_size: Tuple[int, ...], block_grouping: Tuple[int, ...]):
    """ Set weight parameter quantizers of modules that satisfy the given condition to grouped blockwise """


def set_grouped_blockwise_quantization_for_weights(sim: QuantizationSimModel, arg,
                                                   bitwidth: int, symmetric: bool, decompressed_bw: int,
                                                   block_size: Tuple[int, ...], block_grouping: Tuple[int, ...]):
    """
    Set weight parameter quantizers of modules to grouped blockwise.

    :param sim: Quantsim to set activation quantizers for
    :param arg: Argument determining which modules to set. This can consist of either:
        1. A list of torch.nn.Module types, in which case all modules whose type is in the list will be set
        2. A list of torch.nn.Modules, in which case all modules in the list will be set
        3. A callable function which takes a torch.nn.Module as input and returns True if the module is to be set, False
           otherwise
    :param bitwidth: Bitwidth for affine quantization
    :param symmetric: True if affine quantization is symmetric, False otherwise
    :param decompressed_bw: Decompressed bw for grouped block quantization
    :param block_size: Block size for affine quantization
    :param block_grouping: Block grouping for grouped block quantization
    """
    condition = _parse_arg_for_condition(arg)
    _set_grouped_blockwise_quantization_for_weights(sim, condition, bitwidth, symmetric, decompressed_bw, block_size,
                                                    block_grouping)


def _set_grouped_blockwise_quantization_for_weights(sim: QuantizationSimModel,
                                                    condition: Callable[[torch.nn.Module], bool],
                                                    bitwidth: int, symmetric: bool, decompressed_bw: int,
                                                    block_size: Tuple[int, ...], block_grouping: Tuple[int, ...]):
    """ Set weight parameter quantizers of modules that satisfy the given condition to grouped blockwise """
    layer_to_quantizer_shape_dict = _get_layers_to_quantizer_shapes_for_block_size(sim, condition, block_size)
    for layer, quantizer_shape in layer_to_quantizer_shape_dict.items():
        layer.param_quantizers['weight'] = GroupedBlockQuantizeDequantize(quantizer_shape, bitwidth, symmetric,
                                                                          decompressed_bw, None, block_size,
                                                                          block_grouping)
