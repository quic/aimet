# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Custom PyTorch Op for quantizing weights and activations """

import abc
from enum import Enum
from typing import Union, Dict
import torch
from torch import nn

import libpymo
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_torch import utils
from aimet_torch.tensor_quantizer import StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer
from aimet_torch.tensor_quantizer import QuantizationDataType
import aimet_torch.quantsim_straight_through_grad as ste

MAP_ROUND_MODE_TO_PYMO = {'nearest':     libpymo.RoundingMode.ROUND_NEAREST,
                          'stochastic':  libpymo.RoundingMode.ROUND_STOCHASTIC}

MAP_QUANT_SCHEME_TO_PYMO = {QuantScheme.post_training_tf_enhanced: libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                            QuantScheme.post_training_tf: libpymo.QuantizationMode.QUANTIZATION_TF}

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class QcQuantizeOpMode(Enum):
    """
    Mode for the Quantization Ops
    """
    PASSTHROUGH = 1
    ANALYSIS = 2
    ACTIVE = 3


def tensor_quantizer_factory(bitwidth: int, round_mode: str, quant_scheme: Union[QuantScheme, libpymo.QuantizationMode],
                             use_symmetric_encodings: bool, enabled_by_default: bool):
    """
    Instantiates TensorQuantizer depending on the quant_scheme
    :param bitwidth: Quantization bitwidth
    :param round_mode: Rounding mode (e.g. Nearest)
    :param quant_scheme: Quantization scheme (e.g. Range Learning)
    :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
    :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
    :return: An instance of StaticGridPerTensorQuantizer
    """
    assert quant_scheme in [libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED, libpymo.QuantizationMode.QUANTIZATION_TF]

    tensor_quantizer = StaticGridPerTensorQuantizer(bitwidth, round_mode, quant_scheme, use_symmetric_encodings,
                                                    enabled_by_default, QuantizationDataType.int)
    return tensor_quantizer


def module_has_weights(module):
    """
    Check if the module has a parameter called "weight"
    :param module: Module
    :return: True, if module has a parameter called "weight", False otherwise
    """
    for name, _ in module.named_parameters():
        if name == "weight":
            return True

    return False


class QcQuantizeStandAloneBase(nn.Module):
    """
    Base class for the quantization custom ops
    """

    def __init__(self, activation_bw, round_mode, quant_scheme, is_symmetric, data_type):
        """
        Constructor
        :param activation_bw: Quantization bitwidth for activations
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. TF Enhanced)
        :param is_symmetric: Symmetric or asymmetric quantization
        """
        super(QcQuantizeStandAloneBase, self).__init__()
        self.output_quantizers = [tensor_quantizer_factory(activation_bw, round_mode,
                                                           quant_scheme,
                                                           is_symmetric,
                                                           enabled_by_default=True)]
        # Temporary to satisfy dependent code
        # Todo: remove this once all dependent code has been cleaned up
        self.output_quantizer = self.output_quantizers[0]

        self._mode = QcQuantizeOpMode.ANALYSIS
        self._data_type = data_type # TODO remove this variable after tensor_quantizer_factory is using data_type

    @abc.abstractmethod
    def forward(self, *inputs):
        """
        Forward-pass routine. This quantizes the weights before delegating to the wrapped module and
        then quantizes the output before returning the same
        :param inputs: Inputs passed to the module in the forward pass
        :return: Quantized output from the wrapped module
        """

    def set_output_bw(self, output_bw: int):
        """
        Sets (overrides) the output bitwidth for a particular layer
        :param output_bw: Bitwidth from (4-32)
        :return: None
        """
        self.output_quantizers[0].bitwidth = output_bw

    def set_mode(self, mode):
        """
        Sets a working mode for the custom op
        :param mode:
        :return:
        """
        self._mode = mode

    def _quantize_activation(self, tensor_quantizers, tensors_to_quantize):
        """
        Forward-pass routine. This quantizes the weights before delegating to the wrapped module and
        then quantizes the output before returning the same
        :param tensor_quantizers: Tensor quantizers to use for updating stats or quantizing
        :param tensors_to_quantize: Inputs passed to the module in the forward pass
        :return: Quantized output from the wrapped module
        """

        outputs = []
        for index, input_tensor in enumerate(tensors_to_quantize):

            if self._mode is QcQuantizeOpMode.ANALYSIS:

                tensor_quantizers[index].update_encoding_stats(input_tensor)
                output = input_tensor

            elif self._mode is QcQuantizeOpMode.ACTIVE:
                # if we are not in training, then only nearest rounding should be used
                # else we should use whatever the user desires (i.e.. stochastic rounding is a valid option)
                if self.training:
                    round_mode = tensor_quantizers[index].round_mode
                else:
                    round_mode = libpymo.RoundingMode.ROUND_NEAREST
                output = tensor_quantizers[index].quantize_dequantize(input_tensor, round_mode, self, 'output')

            else:
                output = input_tensor

            outputs.append(output)

        # Flatten if there is only one output - which is by far the most common case
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class QcQuantizeWrapper(nn.Module):
    """
    Base class for the quantization custom ops
    """

    # pylint: disable=too-many-arguments
    def __init__(self, module_to_wrap: nn.Module, weight_bw: int, activation_bw: int, round_mode, quant_scheme,
                 is_output_quantized=True, is_symmetric=False, num_inputs=1, num_outputs=1,
                 data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Constructor
        :param module_to_wrap: Module that will be wrapped with this custom op
        :param weight_bw: Quantization bitwidth for weights
        :param activation_bw: Quantization bitwidth for activations
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. TF Enhanced)
        :param is_output_quantized: True if output tensor quantizer is enabled.  False otherwise.
        :param is_symmetric: True if symmetric encoding is used.  False otherwise.
        :param num_inputs: Number of inputs for this module
        :param num_outputs: Number of outputs for this module
        """
        super(QcQuantizeWrapper, self).__init__()

        self.output_quantizers = [tensor_quantizer_factory(activation_bw, round_mode,
                                                           quant_scheme,
                                                           is_symmetric,
                                                           enabled_by_default=is_output_quantized)
                                  for _ in range(num_outputs)]

        # Temporary to satisfy dependent code
        # Todo: remove this once all dependent code has been cleaned up
        self.output_quantizer = self.output_quantizers[0]

        self._mode = QcQuantizeOpMode.ANALYSIS
        self._module_to_wrap = module_to_wrap
        self._data_type = data_type

        # Create quantizer for each parameter and compute encodings
        self.param_quantizers = {}
        for name, _ in module_to_wrap.named_parameters():
            _logger.debug("Adding quantizer for parameter: %s", name)
            self.param_quantizers[name] = tensor_quantizer_factory(weight_bw, round_mode,
                                                                   quant_scheme,
                                                                   is_symmetric,
                                                                   enabled_by_default=True)

        # Create quantizer for layer input
        self.input_quantizers = [tensor_quantizer_factory(activation_bw, round_mode,
                                                          quant_scheme,
                                                          is_symmetric,
                                                          enabled_by_default=False)
                                 for _ in range(num_inputs)]
        self.input_quantizer = self.input_quantizers[0]

    @abc.abstractmethod
    def forward(self, *inputs):
        """
        Forward-pass routine. This quantizes the weights before delegating to the wrapped module and
        then quantizes the output before returning the same
        :param inputs: Inputs passed to the module in the forward pass
        :return: Quantized output from the wrapped module
        """

    def set_output_bw(self, output_bw: int):
        """
        Sets (overrides) the output bitwidth for a particular layer
        :param output_bw: Bitwidth from (4-32)
        :return: None
        """
        self.output_quantizers[0].bitwidth = output_bw

    def set_mode(self, mode):
        """
        Sets a working mode for the custom op
        :param mode: Mode for the Quantization Ops. Can be ANALYSIS or ACTIVE
        """
        self._mode = mode

    def reset_encodings(self):
        """
        Reset encoding stats and set encodings to None for all quantizers
        """
        for quantizer in self.input_quantizers:
            quantizer.reset_encoding_stats()

        for quantizer in self.output_quantizers:
            quantizer.reset_encoding_stats()

        for param_quantizer in self.param_quantizers.values():
            param_quantizer.reset_encoding_stats()

    def enable_per_channel_quantization(self):
        """
        Changes all parameter quantizers (if any) to per-channel mode
        Todo: This needs to change to an abstract method in the future. The purpose to add this method right now
        is to enable per-channel quantization for both only supported wrappers. Supported for static-grid and not
        supported for learned-grid
        """


class StaticGridQuantWrapper(QcQuantizeWrapper):
    """ A custom PyTorch module that derives from QcQuantizeWrapper and quantizes modules """

    # pylint: disable=too-many-arguments
    def __init__(self, module_to_wrap: nn.Module, weight_bw: int, activation_bw: int, round_mode, quant_scheme,
                 is_output_quantized=True, is_symmetric=False, num_inputs=1, num_outputs=1,
                 data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Constructor
        :param module_to_wrap: Module that will be wrapped with this custom op
        :param weight_bw: Quantization bitwidth for weights
        :param activation_bw: Quantization bitwidth for activations
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. TF Enhanced)
        :param is_output_quantized: True if output tensor quantizer is enabled.  False otherwise.
        :param is_symmetric: True if symmetric encoding is used.  False otherwise.
        :param num_inputs: Number of inputs for this module
        :param num_outputs: Number of outputs for this module
        """
        # Translate round mode and quant scheme into pymo types prior to initializing super()
        round_mode = MAP_ROUND_MODE_TO_PYMO[round_mode]
        quant_scheme = MAP_QUANT_SCHEME_TO_PYMO[quant_scheme]

        super(StaticGridQuantWrapper, self).__init__(module_to_wrap, weight_bw, activation_bw, round_mode, quant_scheme,
                                                     is_output_quantized, is_symmetric, num_inputs,
                                                     num_outputs, data_type)

    def forward(self, *inputs):
        """
        Forward-pass routine. This quantizes the weights before delegating to the wrapped module and
        then quantizes the output before returning the same
        :param inputs: Inputs passed to the module in the forward pass
        :return: Quantized output from the wrapped module
        """

        # Quantize the inputs
        quantized_inputs = self._quantize_activation(self.input_quantizers, inputs)
        if isinstance(quantized_inputs, torch.Tensor):
            quantized_inputs = [quantized_inputs]

        # Quantize the parameters
        shadow_params = self._quantize_dequantize_params()

        # Save quantized parameters tensors for backward pass and perform custom bakward pass for gating parameters grad
        # during backward pass
        quantized_inputs = SteGatingFuncForParameters.apply(self, *quantized_inputs)
        # Call the forward of the wrapped module
        wrapped_output = self._module_to_wrap(*quantized_inputs)

        self._restore_shadow_params(shadow_params)

        # Quantize the outputs

        if not self.output_quantizers[0].enabled:
            output = wrapped_output
        else:
            if isinstance(wrapped_output, torch.Tensor):
                wrapped_output = [wrapped_output]
            output = self._quantize_activation(self.output_quantizers, wrapped_output)

        return output

    def _restore_shadow_params(self, shadow_params):

        # Restore the parameters
        for name, param in self._module_to_wrap.named_parameters():
            param.data.zero_()
            param.data.add_(shadow_params[name].data)

    def _quantize_dequantize_params(self):
        """
        Quantizes and dequantizes a parameter
        """

        shadow_params = {}

        # Quantize the parameters, if present
        for name, param in self._module_to_wrap.named_parameters():

            # Store current weight for use later on
            shadow_params[name] = param.detach().clone()

            param_quantizer = self.param_quantizers[name]
            if param_quantizer.enabled:

                # If we are in training mode with quant-sim nodes, then we want to calculate encodings for the
                # parameters in every pass
                if self._module_to_wrap.training or param_quantizer.encoding is None:
                    param_quantizer.reset_encoding_stats()
                    param_quantizer.update_encoding_stats(param.data)
                    param_quantizer.compute_encoding()

                # if we are not in training, then only nearest rounding should be used
                # else we should use whatever the user desires (i.e.. stochastic rounding is a valid option)
                if self.training:
                    round_mode = param_quantizer.round_mode
                else:
                    round_mode = libpymo.RoundingMode.ROUND_NEAREST
                param.data = param_quantizer.quantize_dequantize(param.data, round_mode)

        return shadow_params

    def compute_weight_encodings(self):
        """
        Compute quantized model weight encoding.
        :return: weight_encoding value (libpymo.TfEncoding type)
        """

        if 'weight' in self.param_quantizers:
            return self.param_quantizers['weight'].encoding

        return None

    def compute_encoding(self):
        """
        Compute the quantization encoding for this layer
        """
        for quantizer in self.input_quantizers:
            quantizer.compute_encoding()

        for quantizer in self.output_quantizers:
            quantizer.compute_encoding()

    def _quantize_activation(self, tensor_quantizers, tensors_to_quantize):
        """
        Forward-pass routine. This quantizes the weights before delegating to the wrapped module and
        then quantizes the output before returning the same
        :param tensor_quantizers: Tensor quantizers to use for updating stats or quantizing
        :param tensors_to_quantize: Inputs passed to the module in the forward pass
        :return: Quantized output from the wrapped module
        """

        outputs = []
        for index, input_tensor in enumerate(tensors_to_quantize):
            if not isinstance(input_tensor, torch.Tensor):
                _logger.error('Expecting quantize activation input of type torch.Tensor but got %s', type(input_tensor))
                raise AssertionError
            if input_tensor.dtype in utils.torch_integer_dtypes:
                # Do not quantize integer tensors
                outputs.append(input_tensor)
                continue

            assert len(tensor_quantizers) > index, \
                f"Not enough tensor quantizers ({len(tensor_quantizers)}) allocated"

            if self._mode is QcQuantizeOpMode.ANALYSIS:

                tensor_quantizers[index].update_encoding_stats(input_tensor)
                output = input_tensor

            elif self._mode is QcQuantizeOpMode.ACTIVE:
                # if we are not in training, then only nearest rounding should be used
                # else we should use whatever the user desires (i.e.. stochastic rounding is a valid option)
                if self.training:
                    round_mode = tensor_quantizers[index].round_mode
                else:
                    round_mode = libpymo.RoundingMode.ROUND_NEAREST
                output = tensor_quantizers[index].quantize_dequantize(input_tensor, round_mode)

            else:
                output = input_tensor

            outputs.append(output)

        # Flatten if there is only one output - which is by far the most common case
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def set_and_freeze_param_encoding(self, module_name: str, param_encodings: Dict):
        """
        Set and freeze encoding for parameter from encodings dictionary
        :param module_name: name of module
        :param param_encodings: parameter encodings dictionary
        """
        for orig_param_name, param_quantizer in self.param_quantizers.items():
            param_name = module_name + '.' + orig_param_name
            if param_name in param_encodings:
                encoding_dict = param_encodings[param_name][0]
                encoding, is_symmetric = utils.create_encoding_from_dict(encoding_dict)
                param_quantizer.encoding = encoding

                param_quantizer.bitwidth = encoding.bw
                param_quantizer.use_symmetric_encodings = is_symmetric

                param_quantizer.freeze_encoding()
                _logger.info("Setting and freezing quantization encodings for parameter: %s", param_name)

    def enable_per_channel_quantization(self):
        """
        Changes all parameter quantizers (if any) to per-channel mode
        """
        new_param_quant_dict = {}
        for param_name, param in self._module_to_wrap.named_parameters():
            param_quantizer = self.param_quantizers[param_name]
            channel_axis = 0
            if isinstance(self._module_to_wrap, (torch.nn.ConvTranspose1d,
                                                 torch.nn.ConvTranspose2d,
                                                 torch.nn.ConvTranspose3d)):
                if len(param.shape) > 1:
                    channel_axis = 1

            per_channel_quantizer = StaticGridPerChannelQuantizer(param_quantizer.bitwidth, param_quantizer.round_mode,
                                                                  param_quantizer.quant_scheme,
                                                                  param_quantizer.use_symmetric_encodings,
                                                                  num_channels=param.shape[channel_axis],
                                                                  enabled_by_default=param_quantizer.enabled,
                                                                  ch_axis=channel_axis)

            new_param_quant_dict[param_name] = per_channel_quantizer
        self.param_quantizers = new_param_quant_dict


# Temporarily added for backwards compatibility
QcPostTrainingWrapper = StaticGridQuantWrapper


class QcQuantizeStandalone(QcQuantizeStandAloneBase):
    """ A custom PyTorch module that derives from QcQuantizeStandAloneBase and quantizes inputs """

    def forward(self, *inputs):
        """
        Forward-pass routine. This quantizes the weights before delegating to the wrapped module and
        then quantizes the output before returning the same
        :param inputs: Inputs passed to the module in the forward pass
        :return: Quantized output from the wrapped module
        """

        output = self._quantize_activation(self.output_quantizers, list(inputs))

        return output

    def compute_encoding(self):
        """
        Compute the quantization encoding for this op
        :return: None
        """
        self.output_quantizers[0].compute_encoding()


class SteGatingFuncForParameters(torch.autograd.Function):
    """
    Custom gradient function for STE
    """
    # pylint:disable = arguments-differ
    @staticmethod
    def forward(ctx, quant_wrapper_ref, *quantized_input):
        """
        Quantize-dequantize the tensor, using the saved encoding for this tensor
        :param tensor: Tensor to quantize-dequantize
        :param tensor_quantizer: Reference to the tensor quantizer
        :param round_mode: Rounding mode
        :param wrapper_ref: Reference to quantization wrapper
        :param tensor_name: Name of tensor to be quantized-dequantized
        :return: Resulting tensor
        """

        ctx.quantization_wrapper_ref = quant_wrapper_ref
        return quantized_input

    @staticmethod
    def backward(ctx, *output_grad):
        quant_wrapper_ref = ctx.quantization_wrapper_ref

        # pylint:disable = protected-access
        for name, param in quant_wrapper_ref._module_to_wrap.named_parameters():
            if quant_wrapper_ref.param_quantizers[name].enabled and param.grad is not None:
                param_quantizer = quant_wrapper_ref.param_quantizers[name]

                if isinstance(param_quantizer.encoding, list):
                    # Stack the encodings
                    max_encodings = [enc.max for enc in param_quantizer.encoding]
                    min_encodings = [enc.min for enc in param_quantizer.encoding]
                    param.grad = ste.compute_dloss_by_dx(param, param.grad, min_encodings, max_encodings,
                                                         param_quantizer._ch_axis)
                else:
                    param.grad = ste.compute_dloss_by_dx(param, param.grad, param_quantizer.encoding.min,
                                                         param_quantizer.encoding.max)
        return (None, *output_grad)
