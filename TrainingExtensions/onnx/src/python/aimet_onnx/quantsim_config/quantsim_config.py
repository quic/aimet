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

from aimet_common.quantsim_config.quantsim_config import QuantSimConfigurator as AimetCommonQuantSimConfigurator
from aimet_common.utils import AimetLogger
from aimet_onnx.meta.connectedgraph import ConnectedGraph


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


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

    def configure_quantizers(self, quant_ops_dict: Dict,
                             param_names: List[str],
                             activation_names: List[str]):
        """
        Configures quantizers based on config file
        """
        self._quant_ops_dict = quant_ops_dict
        self._param_names = param_names
        self._activation_names = activation_names

        # Disable all quantizers
        self._disable_all_quantizers()

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
        raise NotImplementedError

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

    def _set_default_configs(self, default_configs):
        """
        Set default configurations for op and param quantizers in model (first level of specificity in configuration
        file)
        :param default_configs: Default configurations for quantizers
        """

    def _set_param_configs(self, param_configs):
        """
        Set configurations for all params of specific types (second level of specificity in configuration file)
        :param param_configs: Dictionary containing configurations for parameters of certain types
        """

    def _set_op_type_configs(self, op_configs):
        """
        Set configurations for all ops of specific types (third level of specificity in configuration file)
        :param op_configs: Dictionary containing configurations for ops of certain types
        """

    def _set_supergroup_configs(self, supergroups_configs):
        """
        Set supergroup specific configurations (fourth level of specificity in configuration file)
        :param supergroups_configs: Configurations for supergroups
        """

    def _set_model_input_configs(self, model_input_configs):
        """
        Set model input specific configurations (fifth level of specificity in configuration file)
        :param model_input_configs: Configuration for model inputs
        """

    def _set_model_output_configs(self, model_output_configs):
        """
        Set model output specific configurations (sixth level of specificity in configuration file)
        :param model_output_configs: Configuration for model outputs
        """

    def _generate_and_apply_op_instance_specific_config(self):
        """
        Generate op instance specific configurations - currently supported_kernels and per_channel_quantization fields
        This function uses op specific supported_kernels (if absent use defaults), op specific per_channel_quantization
        fields (if absent use default per_channel_quantization) and generate op instance specific config
        :return: {op_instance_name, op_specific_config}
        """
