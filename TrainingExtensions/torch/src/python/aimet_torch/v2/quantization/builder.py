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
""" Wrapper and quantizer builder class for supporting both v1 and v2 blocks """

import itertools
from typing import List, Optional, Tuple
import torch
from torch import nn

from aimet_common.defs import QuantScheme, QuantizationDataType, MAP_ROUND_MODE_TO_PYMO
from aimet_common.utils import AimetLogger, log_with_error_and_assert_if_false
from aimet_torch.v2.quantization.float import FloatQuantizeDequantize
from aimet_torch.utils import get_v1_quant_scheme_for_initialization
from aimet_torch.qc_quantize_op import QcQuantizeOpMode, QcQuantizeWrapper, StaticGridQuantWrapper, tensor_quantizer_factory
from aimet_torch.tensor_quantizer import TensorQuantizer, StaticGridPerChannelQuantizer
from aimet_torch.v2.nn import FakeQuantizationMixin
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer, PercentileEncodingAnalyzer, \
    SqnrEncodingAnalyzer
import aimet_torch.fp_quantization as v1_fp_quantization


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class LazyQuantizeWrapper(torch.nn.Module):
    """
    Wrapper builder class for supporting both v1 and v2 blocks
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(self, module_to_wrap: torch.nn.Module, weight_bw: int, activation_bw: int, rounding_mode,
                 quant_scheme: QuantScheme, is_output_quantized=True, is_symmetric=False, num_inputs=1, num_outputs=1,
                 data_type: QuantizationDataType = QuantizationDataType.int):
        super().__init__()
        if data_type == QuantizationDataType.float and weight_bw not in [8, 16]:
            raise ValueError('weight_bw in [8, 16] is the only supported configuration with floating point data type')

        if data_type == QuantizationDataType.float and activation_bw not in [8, 16]:
            raise ValueError('activation_bw in [8, 16] is the only supported configuration with floating point data type')

        # Save those parameters for v1 quant wrapper initialization
        self._weight_bw = weight_bw
        self._activation_bw = activation_bw
        self._rounding_mode = rounding_mode
        self._quant_scheme = quant_scheme
        self._is_output_quantized = is_output_quantized
        self._is_symmetric = is_symmetric
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._data_type = data_type
        self._module_to_wrap = module_to_wrap
        self._mode = QcQuantizeOpMode.ANALYSIS

        # Create quantizer for layer output
        self.output_quantizers = [LazyQuantizer(activation_bw,
                                                rounding_mode,
                                                quant_scheme,
                                                is_symmetric,
                                                enabled_by_default=is_output_quantized,
                                                data_type=data_type)
                                  for _ in range(num_outputs)]

        # Create quantizer for each parameter and compute encodings
        self.param_quantizers = {}
        for name, param in module_to_wrap.named_parameters():
            logger.debug("Adding quantizer for parameter: %s", name)
            self.param_quantizers[name] = LazyParamQuantizer(weight_bw,
                                                             rounding_mode,
                                                             quant_scheme,
                                                             is_symmetric,
                                                             enabled_by_default=True,
                                                             param=param,
                                                             data_type=data_type)

        # Create quantizer for layer input
        self.input_quantizers = [LazyQuantizer(activation_bw,
                                               rounding_mode,
                                               quant_scheme,
                                               is_symmetric,
                                               enabled_by_default=False,
                                               data_type=data_type)
                                 for _ in range(num_inputs)]

        self.supported_kernels = {}

    def enable_per_channel_quantization(self):
        """
        Changes all parameter quantizers (if any) to per-channel mode.
        """
        for param_name, param in self._module_to_wrap.named_parameters():
            param_quantizer = self.param_quantizers[param_name]
            channel_axis = 0
            if isinstance(self._module_to_wrap, (torch.nn.ConvTranspose1d,
                                                 torch.nn.ConvTranspose2d,
                                                 torch.nn.ConvTranspose3d)):
                if len(param.shape) > 1:
                    channel_axis = 1

            # pylint: disable = protected-access
            param_quantizer.enable_per_channel_quantization(channel_axis)

    def _update_quant_param_requires_grad(self, quantized_module: FakeQuantizationMixin):
        """
        Update requres_grad value of quantizers in quantized_module.

        :param quantized_module: module containing quantizers whose requires_grad need to be updated
        """
        if self._quant_scheme in (QuantScheme.post_training_tf_enhanced, QuantScheme.post_training_tf, \
                                  QuantScheme.post_training_percentile):
            for quantizer in itertools.chain(quantized_module.input_quantizers,
                                             quantized_module.output_quantizers,
                                             quantized_module.param_quantizers.values()):
                if quantizer is not None:
                    for _, param in quantizer.named_parameters():
                        param.requires_grad = False

    def _apply_quant_param_value_constraints(self, quantized_module: FakeQuantizationMixin):
        """
        Update min and max of quantizers if their values are specified in config

        :param quantized_module: module containing quantizers whose params need to be updated
        """
        param_quantizer_dict = quantized_module.param_quantizers
        param_quantizers = []
        param_quantizer_info_list = []
        for key in param_quantizer_dict:
            param_quantizers.append(param_quantizer_dict[key])
            param_quantizer_info_list.append(self.param_quantizers[key])

        quantizer_list = quantized_module.input_quantizers + quantized_module.output_quantizers + param_quantizers
        quantizer_info_list = self.input_quantizers + self.output_quantizers + param_quantizer_info_list

        for quantizer, quantizer_info in zip(quantizer_list, quantizer_info_list):
            if quantizer is not None and quantizer_info.encoding_min_max_fixed_vals:
                quantizer.min = torch.nn.Parameter(quantizer_info.encoding_min_max_fixed_vals[0] * torch.ones((1,)))
                quantizer.max = torch.nn.Parameter(quantizer_info.encoding_min_max_fixed_vals[1] * torch.ones((1,)))

    def realize_v1_wrapper(self) -> QcQuantizeWrapper:
        """
        Realizes v1 quant wrapper using collected information

        :return: v1 quant wrapper with specified properties
        """
        quant_scheme_for_initialization = get_v1_quant_scheme_for_initialization(self._quant_scheme)

        quantized_module = StaticGridQuantWrapper(self._module_to_wrap, self._weight_bw, self._activation_bw,
                                                  self._rounding_mode, quant_scheme_for_initialization,
                                                  self._is_output_quantized, self._is_symmetric, self._num_inputs,
                                                  self._num_outputs, self._data_type)

        quantized_module.input_quantizers = [quant_builder.get_v1_quantizer() for quant_builder in self.input_quantizers]
        quantized_module.output_quantizers = [quant_builder.get_v1_quantizer() for quant_builder in self.output_quantizers]
        quantized_module.param_quantizers = {param_name: quant_builder.get_v1_quantizer() \
                      for (param_name, quant_builder) in self.param_quantizers.items()}
        quantized_module.supported_kernels = self.supported_kernels

        return quantized_module

    def realize_v2_wrapper(self) -> FakeQuantizationMixin:
        """
        Realizes v2 quant wrapper using collected information

        :return: v2 quant wrapper with specified properties
        """
        if type(self._module_to_wrap) in QuantizationMixin.cls_to_qcls: # pylint: disable=unidiomatic-typecheck
            quantized_module = QuantizationMixin.from_module(self._module_to_wrap)
        else:
            quantized_module = FakeQuantizationMixin.from_module(self._module_to_wrap)

        def set_recursive(module_list, i, quantizer):
            """
            Set quantizer recursively.
            AIMET V1 handles nested input/output tensors with single input quantizers,
            whereas V2 quantized module allows having nested input/output quantizers.
            (For reference, see the class definition of FakeQuantizedLSTM in fake_quant.py)

            To implement V1 behavior, we set the nested input/output quantizers to
            share the same single quantizer, for example as below:
              - self.input_quantizers = [q1, q2, q3]
              - quant_module.input_quantizers = [None, [None, None], None]
                (before set_recursive)
              - quant_module.input_quantizers = [q1, [q2, q2], q3]
                (after set_recursive)
            """
            if module_list[i] is None:
                module_list[i] = quantizer
            elif isinstance(module_list[i], nn.ModuleList):
                for j in range(len(module_list[i])):
                    set_recursive(module_list[i], j, quantizer)
            else:
                raise RuntimeError

        for i, _ in list(enumerate(quantized_module.input_quantizers)):
            quantizer = self.input_quantizers[i].realize()
            set_recursive(quantized_module.input_quantizers, i, quantizer)

        for i, _ in list(enumerate(quantized_module.output_quantizers)):
            quantizer = self.output_quantizers[i].realize()
            set_recursive(quantized_module.output_quantizers, i, quantizer)

        for param_name, quant_builder in self.param_quantizers.items():
            quantized_module.param_quantizers[param_name] = quant_builder.realize()

        self._apply_quant_param_value_constraints(quantized_module)
        self._update_quant_param_requires_grad(quantized_module)
        quantized_module.supported_kernels = self.supported_kernels

        return quantized_module

    @staticmethod
    def forward(_):
        """
        Dummy forward-pass routine for implementing abstract function.
        """
        raise RuntimeError("forward function of LazyQuantizeWrapper should not be called before it is realized")


class LazyQuantizer:
    """
    Quantizer builder class for supporting both v1 and v2 blocks
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, bitwidth: int, round_mode, quant_scheme: QuantScheme,
                 use_symmetric_encodings: bool, enabled_by_default: bool,
                 data_type: QuantizationDataType = QuantizationDataType.int):
        self.round_mode = MAP_ROUND_MODE_TO_PYMO[round_mode]
        self.quant_scheme = quant_scheme
        self.use_symmetric_encodings = use_symmetric_encodings
        self.use_strict_symmetric = False
        self.use_unsigned_symmetric = False
        self.is_unsigned_symmetric = False
        self.bitwidth = bitwidth
        self.enabled = enabled_by_default
        self.data_type = data_type
        self.is_const = False
        self._encoding_min_max_fixed_vals = None

    @property
    def encoding_min_max_fixed_vals(self) -> Optional[Tuple[float, float]]:
        """ Accessor to self._encoding_min_max_fixed_vals """
        return self._encoding_min_max_fixed_vals

    @encoding_min_max_fixed_vals.setter
    def encoding_min_max_fixed_vals(self, min_max_vals: Tuple[float, float]):
        """ self._encoding_min_max_fixed_vals setter """
        log_with_error_and_assert_if_false(isinstance(min_max_vals, tuple), logger, 'Min max vals must be a tuple')
        log_with_error_and_assert_if_false(len(min_max_vals) == 2, logger, 'Min max vals must be a tuple of two '
                                                                           'values')
        log_with_error_and_assert_if_false(min_max_vals[0] < min_max_vals[1], logger,
                                           'Min value ' + str(min_max_vals[0]) + ' is not less than max val ' +
                                           str(min_max_vals[1]))
        if self.quant_scheme != QuantScheme.post_training_tf:
            self.quant_scheme = QuantScheme.post_training_tf
        self._encoding_min_max_fixed_vals = min_max_vals

    def _validate_quantizer_properties(self):
        """
        Checks quantizer properties before creating quantizer.
        """
        if self.use_symmetric_encodings:
            assert not self.use_strict_symmetric, "Strict symmetric is not supported in quantsim v1.5"
            assert not self.use_unsigned_symmetric, "Unsigned symmetric is not supported in quantsim v1.5"
            assert not self.is_unsigned_symmetric, "Unsigned symmetric is not supported in quantsim v1.5"

    def _get_v2_encoding_analyzer(self, shape):
        """
        Converts v1 quant scheme into v2 quant scheme.

        :return: corresponding v2 quant scheme
        """
        if self.quant_scheme in (QuantScheme.post_training_tf, QuantScheme.training_range_learning_with_tf_init):
            return MinMaxEncodingAnalyzer(shape)
        if self.quant_scheme == QuantScheme.post_training_percentile:
            return PercentileEncodingAnalyzer(shape)
        if self.quant_scheme in (QuantScheme.post_training_tf_enhanced,
                                 QuantScheme.training_range_learning_with_tf_enhanced_init):
            return SqnrEncodingAnalyzer(shape)
        raise NotImplementedError(f"Quant scheme {self.quant_scheme} in old quantsim is not supported yet in quantsim v1.5")

    @staticmethod
    def _get_param_shape() -> List[int]:
        """
        Returns param shape for quantization parameter.
        Can be overriden in child class.

        :return: param shape for quantization parameter
        """
        return [1]

    def realize(self) -> Optional[QuantizeDequantize]:
        """
        Returns spec for v2 quantizer initialization using collected information.

        :return: spec for v2 quantizer initialization
        """
        if not self.enabled:
            return None

        self._validate_quantizer_properties()

        quantizer_param_shape = self._get_param_shape()

        if self.data_type == QuantizationDataType.int:
            encoding_analyzer = self._get_v2_encoding_analyzer(quantizer_param_shape)
            quantizer = QuantizeDequantize(quantizer_param_shape, self.bitwidth,
                                           self.use_symmetric_encodings, encoding_analyzer)
        else:
            if self.bitwidth == 16:
                quantizer = FloatQuantizeDequantize(dtype=torch.float16)
            else:
                assert self.bitwidth == 8
                mantissa_bits = v1_fp_quantization.NUM_MANTISSA_BITS
                exponent_bits = 7 - mantissa_bits
                encoding_analyzer = self._get_v2_encoding_analyzer(quantizer_param_shape)
                quantizer = FloatQuantizeDequantize(exponent_bits, mantissa_bits,
                                                    encoding_analyzer=encoding_analyzer)
            # Float quantizers are not trainable in V1 quantsim
            for param in quantizer.parameters():
                param.requires_grad = False

        return quantizer

    def _set_internal_quantizer_properties(self, quantizer: TensorQuantizer):
        """
        Sets internal quantizer properties of v1 quantizer
        using collected information.

        :param quantizer: quantizer to update its internal properties
        """
        if self.encoding_min_max_fixed_vals is not None:
            quantizer.encoding_min_max_fixed_vals = self.encoding_min_max_fixed_vals
        quantizer.is_unsigned_symmetric = self.is_unsigned_symmetric
        quantizer.use_unsigned_symmetric = self.use_unsigned_symmetric
        quantizer.use_strict_symmetric = self.use_strict_symmetric
        quantizer.is_const = self.is_const

    def get_v1_quantizer(self) -> TensorQuantizer:
        """
        Returns v1 quantizer using collected information.

        :return: v1 quantizer with specified properties
        """
        quant_scheme_for_initialization = get_v1_quant_scheme_for_initialization(self.quant_scheme)

        quantizer = tensor_quantizer_factory(self.bitwidth, self.round_mode, quant_scheme_for_initialization,
                                             self.use_symmetric_encodings, self.enabled, self.data_type)

        self._set_internal_quantizer_properties(quantizer)

        return quantizer


#pylint: disable=W0223
class LazyParamQuantizer(LazyQuantizer):
    """
    Quantizer builder class for supporting both v1 and v2 blocks
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, bitwidth: int, round_mode, quant_scheme: QuantScheme,
                 use_symmetric_encodings: bool, enabled_by_default: bool,
                 param: torch.nn.Parameter,
                 data_type: QuantizationDataType = QuantizationDataType.int):
        super().__init__(bitwidth, round_mode, quant_scheme, use_symmetric_encodings, enabled_by_default, data_type)
        self.param_shape = param.shape
        self.channel_axis = None

    def enable_per_channel_quantization(self, channel_axis: int):
        """
        Set channel axis

        :param channel_axis: channel axis to quantize
        """
        self.channel_axis = channel_axis

    def _validate_quantizer_properties(self):
        """
        Checks quantizer properties before creating quantizer.
        """
        super()._validate_quantizer_properties()

        if self.channel_axis:
            assert 0 <= self.channel_axis < len(self.param_shape), \
                f"Channel axis {self.channel_axis} is out of bound of param shape {self.param_shape}"

    def _get_param_shape(self) -> List[int]:
        """
        Returns param shape for quantization parameter.
        Can be overriden in child class.

        :return: param shape for quantization parameter
        """
        if self.channel_axis is not None:
            channel_axis = self.channel_axis if self.channel_axis else 0

            quantizer_param_shape = [1] * len(self.param_shape)
            quantizer_param_shape[channel_axis] = self.param_shape[channel_axis]

            return quantizer_param_shape

        return super()._get_param_shape()

    def get_v1_quantizer(self) -> TensorQuantizer:
        """
        Returns v1 quantizer using collected information.

        :return: v1 quantizer with specified properties
        """
        if self.channel_axis is not None:
            quant_scheme_for_initialization = get_v1_quant_scheme_for_initialization(self.quant_scheme)
            channel_axis = self.channel_axis if self.channel_axis else 0
            num_channels = self.param_shape[channel_axis]

            quantizer = StaticGridPerChannelQuantizer(self.bitwidth, self.round_mode, quant_scheme_for_initialization,
                                                      self.use_symmetric_encodings, num_channels,
                                                      self.enabled, channel_axis, self.data_type)

            self._set_internal_quantizer_properties(quantizer)
        else:
            quantizer = super().get_v1_quantizer()

        return quantizer
