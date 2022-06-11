# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Implementation for simulating models running on Quantized hardware """
# pylint: disable=too-many-lines
import os
import io
import copy
import pickle
from typing import Tuple, List, Union, Dict, Callable
from collections.abc import Iterable
import json
import torch
import onnx

import aimet_common
import aimet_common.libpymo as libpymo
from aimet_common.utils import AimetLogger, save_json_yaml
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_common.quantsim import encoding_version, validate_quantsim_inputs, calculate_delta_offset
from aimet_common.quant_utils import get_conv_accum_bounds

from aimet_torch.quantsim_config.quantsim_config import QuantSimConfigurator
from aimet_torch.qc_quantize_op import QcQuantizeStandAloneBase, QcQuantizeWrapper, QcQuantizeOpMode, \
    StaticGridQuantWrapper, LearnedGridQuantWrapper, QUANTIZER_TYPE_INPUT, QUANTIZER_TYPE_OUTPUT
from aimet_torch.tensor_quantizer import StaticGridTensorQuantizer
from aimet_torch import torchscript_utils, utils, transformer_utils
from aimet_torch.onnx_utils import OnnxSaver, OnnxExportApiArgs, CustomMarker
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.qc_quantize_recurrent import QcQuantizeRecurrent


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# Types of modules which cannot be quantized
unquantizable_modules = (QcQuantizeWrapper, QcQuantizeStandAloneBase, QcQuantizeRecurrent, torch.nn.Identity)

# If a torch module type is in this dictionary, call the corresponding quantized module constructor instead of wrapping
# it with QcQuantizeWrapper.
qc_quantize_modules_dict = {
    torch.nn.RNN: QcQuantizeRecurrent,
    torch.nn.LSTM: QcQuantizeRecurrent,
    torch.nn.GRU: QcQuantizeRecurrent
}

# Length of the string '._module_to_wrap'
MODULE_TO_WRAP_STRING_REVERSE_INDEX = -16

MAP_PYMO_TO_ROUND_MODE = {libpymo.RoundingMode.ROUND_NEAREST: 'nearest',
                          libpymo.RoundingMode.ROUND_STOCHASTIC: 'stochastic'}


class QuantParams:
    """
    Data type to hold quantization related params.
    """

    def __init__(self,
                 weight_bw: int = 8,
                 act_bw: int = 8,
                 round_mode: str = 'nearest',
                 quant_scheme: Union[QuantScheme, str] = QuantScheme.post_training_tf_enhanced,
                 config_file: str = None):
        """
        Constructor

        :param weight_bw: Weight bitwidth (4-31) to use for quantizing layer weights. Default = 8
        :param act_bw: Activation bitwidth(4-31) to use for quantizing layer activations. Default = 8
        :param round_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
        :param quant_scheme: Quantization scheme. Supported options are 'tf_enhanced' or 'tf' or using Quant Scheme Enum
                             QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
        :param config_file: Path to Configuration file for model quantizers
        """

        self.weight_bw = weight_bw
        self.act_bw = act_bw
        self.round_mode = round_mode
        self.quant_scheme = quant_scheme
        self.config_file = config_file


class QuantizationSimModel:
    """
    Implements mechanism to add quantization simulations ops to a model. This allows for off-target simulation of
    inference accuracy. Also allows the model to be fine-tuned to counter the effects of quantization.
    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes
    def __init__(self, model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple],
                 quant_scheme: Union[str, QuantScheme] = QuantScheme.post_training_tf_enhanced,
                 rounding_mode: str = 'nearest', default_output_bw: int = 8, default_param_bw: int = 8,
                 in_place: bool = False, config_file: str = None,
                 default_data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Constructor

        :param model: Model to add simulation ops to
        :param dummy_input: Dummy input to the model. Used to parse model graph. If the model has more than one input,
                            pass a tuple. User is expected to place the tensors on the appropriate device.
        :param quant_scheme: Quantization scheme. The Quantization scheme is used to compute the Quantization encodings.
                             There are multiple schemes available. Please refer the QuantScheme enum definition.
        :param rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing all layer inputs and outputs
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing all layer parameters
        :param in_place: If True, then the given 'model' is modified in-place to add quant-sim nodes.
                Only suggested use of this option is when the user wants to avoid creating a copy of the model
        :param config_file: Path to Configuration file for model quantizers
        :param default_data_type: Default data type to use for quantizing all layer inputs, outputs and parameters.
                                 Possible options are QuantizationDataType.int and QuantizationDataType.float.
                                 Note that the mode default_data_type=QuantizationDataType.float is only supported with
                                 default_output_bw=16 and default_param_bw=16
        """
        # Perform sanity checks on inputs
        validate_quantsim_inputs(quant_scheme, rounding_mode, default_output_bw, default_param_bw,
                                 default_data_type)

        # save some parameters
        if in_place:
            self.model = model
        else:
            self.model = copy.deepcopy(model)

        try:
            self.connected_graph = ConnectedGraph(self.model, dummy_input)
        except (torch.jit.TracingCheckError, AssertionError):
            self.connected_graph = None

        if isinstance(quant_scheme, str):
            if quant_scheme == 'tf':
                quant_scheme = QuantScheme.post_training_tf
            elif quant_scheme == 'tf_enhanced':
                quant_scheme = QuantScheme.post_training_tf_enhanced
            elif quant_scheme == 'percentile':
                quant_scheme = QuantScheme.post_training_percentile
        self._quant_scheme = quant_scheme
        self._rounding_mode = rounding_mode
        self._default_output_bw = default_output_bw
        self._default_param_bw = default_param_bw
        self._is_conditional = False
        self._module_marker_map = {}
        self._percentile_value = 100 # default percentile value

        # Add quantization layers
        num_inout_tensors = utils.find_num_inout_tensors_per_module(self.model, dummy_input)

        self._add_quantization_wrappers(self.model, num_inout_tensors, default_data_type)

        # Disable bias quantization
        self.exclude_param_from_quantization("bias")

        # override specific quantizers to tf mode in transformer model
        self._override_quant_config_for_transformer_layers()

        self._quantsim_configurator = self.configure_quantization_ops(config_file, default_output_bw, default_param_bw,
                                                                      default_data_type)
        self._supported_kernels = self._quantsim_configurator.get_supported_kernels()

    def get_supported_kernels(self) -> Dict:
        """
        Return _supported_kernels parsed from the config file
        :return: Dictionary containing supported_kernels
        """
        return self._supported_kernels

    def __str__(self):
        """
        Pretty-printed output indicating where in the model, quantizers have been activated
        :return:
        """

        def pp_quantizer(stream, quantizer, prefix_string):
            if quantizer.enabled:
                stream.write(f'  {prefix_string}: bw={quantizer.bitwidth}, '
                             f'encoding-present={bool(quantizer.encoding)}\n')

                if quantizer.encoding:
                    stream.write(f'    {quantizer}')
            else:
                stream.write(f'  {prefix_string}: Not quantized\n')

            stream.write('  -------\n')

        stream = io.StringIO(newline='\n')
        stream.write("-------------------------\n")
        stream.write("Quantized Model Report\n")
        stream.write("-------------------------\n")

        wrappers = [(name, module) for name, module in self.model.named_modules()
                    if isinstance(module, QcQuantizeWrapper)]

        for name, wrapper in wrappers:
            stream.write('----------------------------------------------------------\n')
            stream.write('Layer: {}\n'.format(name))

            # Inputs
            for index, quantizer in enumerate(wrapper.input_quantizers):
                pp_quantizer(stream, quantizer, prefix_string=f"Input[{index}]")

            # Params
            for param_name, quantizer in wrapper.param_quantizers.items():
                pp_quantizer(stream, quantizer, prefix_string=f"Param[{param_name}]")

            # Outputs
            for index, quantizer in enumerate(wrapper.output_quantizers):
                pp_quantizer(stream, quantizer, prefix_string=f"Output[{index}]")

        return stream.getvalue()

    def compute_encodings(self, forward_pass_callback, forward_pass_callback_args):
        """
        Computes encodings for all quantization sim nodes in the model. It is also used to find initial encodings for
        Range Learning

        :param forward_pass_callback: A callback function that simply runs forward passes on the model. This callback
            function should use representative data for the forward pass, so the calculated encodings work for all
            data samples. This callback internally chooses the number of data samples it wants to use for calculating
            encodings.
        :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
            the user to determine the type of this parameter. E.g. could be simply an integer representing the number
            of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
            If set to None, forward_pass_callback will be invoked with no parameters.
        :return: None

        """

        quantized_layers = self._get_qc_quantized_layers(self.model)

        for _, layer in quantized_layers:
            # Clear stats and encodings if they are present
            layer.reset_encodings()

            # And set the mode to analysis
            layer.set_mode(QcQuantizeOpMode.ANALYSIS)

        for _, layer in quantized_layers:
            # call only when quant scheme is percentile
            if self._quant_scheme == QuantScheme.post_training_percentile:
                layer.set_percentile_value(self._percentile_value)

        # Run forward iterations so we can collect statistics to compute the appropriate encodings
        with utils.in_eval_mode(self.model), torch.no_grad():
            _ = forward_pass_callback(self.model, forward_pass_callback_args)

        # Get the computed per-layer encodings and log them
        for name, layer in quantized_layers:
            layer.compute_encoding()

            # Before we return we set the mode to active - meaning ready for quantize/de-quantize
            # for layers with valid_encoding, otherwise we set to pass through
            if isinstance(layer, QcQuantizeRecurrent):
                self.set_mode_for_recurrent_module(layer, name)

            else:
                # By default we want to set the Quantization wrappers to ACTIVE mode
                layer.set_mode(QcQuantizeOpMode.ACTIVE)

        self.replace_wrappers_for_quantize_dequantize()

        self._clamp_transformer_attention_mask_encoding()

    @classmethod
    def set_mode_for_recurrent_module(cls, layer: QcQuantizeRecurrent, name: str):
        """
        Sets Recurrent module to active or pass through mode based on quantizer state

        :param layer:  Qc Quantizer layer for recurrent module
        :param name:  layer name
        :return: True if the encoding is invalid

        """
        for quantizer_name, output_quantizer in layer.output_quantizers.items():
            if output_quantizer.enabled:
                if output_quantizer.encoding:
                    encoding = output_quantizer.encoding
                    logger.debug("Encoding for %s-%s: min=%f, max=%f, offset=%f. delta=%f, bw=%f",
                                 name, quantizer_name, encoding.min, encoding.max,
                                 encoding.delta, encoding.offset, encoding.bw)

        for quantizer_name, input_quantizer in layer.input_quantizers.items():
            if input_quantizer.enabled:
                if input_quantizer.encoding:
                    encoding = input_quantizer.encoding
                    logger.debug("Encoding for %s-%s: min=%f, max=%f, offset=%f. delta=%f, bw=%f",
                                 name, quantizer_name, encoding.min, encoding.max,
                                 encoding.delta, encoding.offset, encoding.bw)

        layer.set_mode(QcQuantizeOpMode.ACTIVE)

    def set_percentile_value(self, percentile_value: float):
        """
        Set the percentile value to be used while computing encodings
        """
        if percentile_value < 90 or percentile_value > 100:
            raise ValueError("Percentile value must be in range [90, 100]")
        self._percentile_value = percentile_value

    def export(self, path: str, filename_prefix: str, dummy_input: Union[torch.Tensor, Tuple],
               onnx_export_args: Union[OnnxExportApiArgs, None] = OnnxExportApiArgs(),
               propagate_encodings: bool = False):
        """
        This method exports out the quant-sim model so it is ready to be run on-target.

        Specifically, the following are saved

        1. The sim-model is exported to a regular PyTorch model without any simulation ops
        2. The quantization encodings are exported to a separate JSON-formatted file that can
           then be imported by the on-target runtime (if desired)
        3. Optionally, An equivalent model in ONNX format is exported. In addition, nodes in the ONNX model are named
           the same as the corresponding PyTorch module names. This helps with matching ONNX node to their quant
           encoding from #2.

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param dummy_input: Dummy input to the model. Used to parse model graph. It is required for the dummy_input to
                be placed on CPU.
        :param onnx_export_args: optional export argument with onnx specific overrides if not provide export via
                torchscript graph
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops.
        :return: None

        """
        # save the quantized model and encodings
        model_filename = filename_prefix + '.pth'
        model_path = os.path.join(path, model_filename)

        # Create a version of the model without any quantization ops
        model_to_export = copy.deepcopy(self.model).cpu()
        all_modules_in_model_to_export = [module for module in model_to_export.modules()]
        self._remove_quantization_wrappers(model_to_export, all_modules_in_model_to_export)

        torch.save(model_to_export, model_path)

        if onnx_export_args is None:
            self.export_torch_script_model_and_encodings(path, filename_prefix, model_to_export, self.model,
                                                         dummy_input)
        elif isinstance(onnx_export_args, OnnxExportApiArgs):
            self.export_onnx_model_and_encodings(path, filename_prefix, model_to_export, self.model,
                                                 dummy_input, onnx_export_args, propagate_encodings,
                                                 self._module_marker_map, self._is_conditional)
        else:

            raise ValueError(f'unsupported opt_args type={type(onnx_export_args)}')

    @staticmethod
    def export_torch_script_model_and_encodings(path: str, filename_prefix: str,
                                                original_model: torch.nn.Module,
                                                sim_model: torch.nn.Module,
                                                dummy_input: Union[torch.Tensor, Tuple]):
        """
        This method exports  a onnx mode and the corresponding encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param original_model: model without the quantsim wrappers
        :param sim_model: model with the quantsim wrappers
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :return: None
        """
        with utils.in_eval_mode(original_model), torch.no_grad():
            trace = torch.jit.trace(original_model, dummy_input)
            ts_path = os.path.join(path, filename_prefix + '.torchscript.pth')
            trace.save(ts_path)

            # reload the trace from the saved trace file
            trace = torch.jit.load(ts_path)
            torch_script_node_io_tensor_map, valid_param_set = \
                torchscript_utils.get_node_to_io_tensor_names_map(original_model, trace, dummy_input)

        # Export encodings
        QuantizationSimModel._export_encodings_to_files(sim_model, path, filename_prefix,
                                                        torch_script_node_io_tensor_map, valid_param_set,
                                                        propagate_encodings=False)

    @staticmethod
    def export_onnx_model_and_encodings(path: str, filename_prefix: str, original_model: torch.nn.Module,
                                        sim_model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple],
                                        onnx_export_args: OnnxExportApiArgs, propagate_encodings: bool,
                                        module_marker_map: Dict[torch.nn.Module, torch.Tensor] = None,
                                        is_conditional: bool = False):
        """
        This method exports a onnx model and the corresponding encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param original_model: model without the quantsim wrappers
        :param sim_model: model with the quantsim wrappers
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :param module_marker_map: Maps module names to traced custom markers (only used for conditional models)
        :param is_conditional: True if model is conditional, False otherwise
        :param onnx_export_args: Additional onnx export args including export api overrides
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops.
        :return: None

        """
        if module_marker_map is None:
            module_marker_map = {}
        if onnx_export_args is None:
            onnx_export_args = OnnxExportApiArgs()
        # Save model to onnx
        onnx_path = os.path.join(path, filename_prefix + '.onnx')

        utils.replace_modules_of_type1_with_type2(original_model, torch.nn.Dropout2d, torch.nn.Identity)
        utils.replace_modules_of_type1_with_type2(original_model, torch.nn.Dropout, torch.nn.Identity)
        utils.replace_modules_of_type1_with_type2(original_model, torch.nn.Dropout3d, torch.nn.Identity)

        OnnxSaver.set_node_names(onnx_path, original_model, dummy_input, is_conditional, module_marker_map,
                                 onnx_export_args)

        onnx_model = onnx.load(onnx_path)
        onnx_node_to_io_tensor_map, valid_param_set = OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)

        # Export encodings
        QuantizationSimModel._export_encodings_to_files(sim_model, path, filename_prefix,
                                                        onnx_node_to_io_tensor_map, valid_param_set,
                                                        propagate_encodings)

    def exclude_layers_from_quantization(self, layers_to_exclude: List[torch.nn.Module]):
        """
        Excludes certain layers from being quantized-dequantized by the simulator
        :param layers_to_exclude: List of torch layers to exclude
        :return: None
        """
        self._remove_quantization_wrappers(self.model, layers_to_exclude)

    def exclude_param_from_quantization(self, param_name_to_exclude: str):
        """
        Excludes all parameters matching 'param_name' from quantization
        :param param_name_to_exclude: Name of the parameter to exclude
        :return: None
        """
        for module in self.model.modules():
            if isinstance(module, (QcQuantizeWrapper, QcQuantizeRecurrent)):
                if param_name_to_exclude in module.param_quantizers:
                    module.param_quantizers[param_name_to_exclude].enabled = False

    def _replace_quantization_wrapper(self, model, device):
        """
        Recursively remove quantization wrappers from all appropriate modules starting with a given module
        :param model: model for which PostTrainingWrapper gets replaced with Trainable wrapped module
        :param device: device on which model is present
        :return: None
        """
        for module_name, module_ref in model.named_children():

            if isinstance(module_ref, StaticGridQuantWrapper):
                # Create a Trainable wrapper and copy properties of PostTrainingWrapper to the Trainable wrapper
                quantized_module = self._construct_and_initialize_trainable_wrapper(module_ref, device)
                setattr(model, module_name, quantized_module)

            # Recursively call children modules if present
            if not utils.is_leaf_module(module_ref):
                self._replace_quantization_wrapper(module_ref, device)

    def _construct_and_initialize_trainable_wrapper(self, post_training_module: StaticGridQuantWrapper,
                                                    device) -> StaticGridQuantWrapper:
        """
        Copies tensor properties (use_symmetric_encodings, enabled, encodings) from post_training_module wrapper to
        trainable_module wrapper
        G:param post_training_module: StaticGridQuantWrapper wrapped module
        :param device: device on which model is present
        :return: trainable_module: QcTrainable wrapper module
        """
        # Creating a StaticGridQuantWrapper module
        # pylint: disable=protected-access
        module = post_training_module._module_to_wrap

        num_inputs = len(post_training_module.input_quantizers)
        num_outputs = len(post_training_module.output_quantizers)

        trainable_module = LearnedGridQuantWrapper(module, self._default_param_bw,
                                                   self._default_output_bw, self._rounding_mode, self._quant_scheme,
                                                   device=device, num_inputs=num_inputs, num_outputs=num_outputs,
                                                   data_type=QuantizationDataType.int)
        for index, post_training_output_quantizer in enumerate(post_training_module.output_quantizers):
            # Setting user set parameters for output
            trainable_module.output_quantizers[index].use_symmetric_encodings = post_training_output_quantizer. \
                use_symmetric_encodings
            trainable_module.output_quantizers[index].enabled = post_training_output_quantizer.enabled
            # Initializing encodings for trainable wrapper
            trainable_module.output_quantizers[index].encoding = post_training_output_quantizer.encoding

        for index, post_training_input_quantizer in enumerate(post_training_module.input_quantizers):
            # Setting user set parameters for input
            trainable_module.input_quantizers[index].use_symmetric_encodings = post_training_input_quantizer. \
                use_symmetric_encodings
            trainable_module.input_quantizers[index].enabled = post_training_input_quantizer.enabled
            # Initializing encodings for trainable wrapper
            trainable_module.input_quantizers[index].encoding = post_training_input_quantizer.encoding

        # Setting user set parameters for input
        for name, _ in module.named_parameters():
            trainable_module.param_quantizers[name].use_symmetric_encodings = \
                post_training_module.param_quantizers[name].use_symmetric_encodings
            trainable_module.param_quantizers[name].enabled = \
                post_training_module.param_quantizers[name].enabled
            trainable_module.param_quantizers[name].encoding = \
                post_training_module.param_quantizers[name].encoding

        return trainable_module

    def replace_wrappers_for_quantize_dequantize(self):
        """
        Replaces StaticGridWrapper with LearnedGridWrapper
        """
        if self._quant_scheme == QuantScheme.training_range_learning_with_tf_init or self._quant_scheme == \
                QuantScheme.training_range_learning_with_tf_enhanced_init:
            device = utils.get_device(self.model)

            self._replace_quantization_wrapper(self.model, device)

    def _override_quant_config_for_transformer_layers(self):
        """Looks for specfic ops in a transformer and overrides the quantizer to tf mode
        """
        # pylint: disable=protected-access
        attention_with_mask_add_quantizer_dict = transformer_utils.get_attention_with_mask_add_quantizer_dict(self.model)

        for attention_head, (mask_add_quantizer_wrapper, mask_add_name) in attention_with_mask_add_quantizer_dict.items():

            assert isinstance(mask_add_quantizer_wrapper, StaticGridQuantWrapper)

            # clamping needs to be done only if data type is int
            if mask_add_quantizer_wrapper.output_quantizer.data_type == QuantizationDataType.int:

                module_to_quantize = mask_add_quantizer_wrapper._module_to_wrap

                quantizer_wrapper_type = qc_quantize_modules_dict.get(type(module_to_quantize), StaticGridQuantWrapper)

                # Add a quantizer set to tf mode and bw to 16 and copy over remaining attributes
                # we need 16 bit to retain the max representation for this quantizer.
                quantized_module = quantizer_wrapper_type(module_to_quantize, 16, 16,
                                                          MAP_PYMO_TO_ROUND_MODE[mask_add_quantizer_wrapper.output_quantizer.round_mode],
                                                          QuantScheme.post_training_tf,
                                                          num_inputs=len(mask_add_quantizer_wrapper.input_quantizers),
                                                          num_outputs=len(mask_add_quantizer_wrapper.output_quantizers),
                                                          data_type=mask_add_quantizer_wrapper.output_quantizer.data_type)

                setattr(attention_head, mask_add_name, quantized_module)

    def _clamp_transformer_attention_mask_encoding(self):
        """
        clamps the quantizer encoding min associated with mask adder
        op within a attention head.
        :return:
        """
        # pylint: disable=protected-access
        attention_with_mask_add_quantizer_dict = transformer_utils.get_attention_with_mask_add_quantizer_dict(self.model)

        for (mask_add_quantizer_wrapper, _) in attention_with_mask_add_quantizer_dict.values():
            # we check if quantizer is enabled and data type is set to int before clamping
            # clamping is not necessary for FP16 mode.
            assert isinstance(mask_add_quantizer_wrapper, StaticGridQuantWrapper)
            if mask_add_quantizer_wrapper.output_quantizer.enabled and \
                    mask_add_quantizer_wrapper.output_quantizer.data_type == QuantizationDataType.int:
                for output_quantizer in mask_add_quantizer_wrapper.output_quantizers:
                    # get the min/max from accumulated stats associated with this quantizer
                    encoding = output_quantizer.encoding
                    output_quantizer.encoding.min = max(encoding.min,
                                                        transformer_utils.MASK_OVERRIDE_VALUE)
                    output_quantizer.encoding.max = encoding.max

                    # recompute grid params as we clamped min and updated max above
                    # with bitwidth as dictated by default config
                    clamped_encoding = aimet_common.quantsim.recompute_grid_params(
                        output_quantizer.encoding,
                        self._default_output_bw,
                        output_quantizer.use_symmetric_encodings)

                    # update encoding of this quantizer
                    output_quantizer.encoding = clamped_encoding
                    mask_add_quantizer_wrapper.output_quantizer.freeze_encoding()
            else:
                logger.debug("Skipping clamp on %s. Quantizer is disabled or not int type",
                             mask_add_quantizer_wrapper)

    @staticmethod
    def _validate_quantsim_inputs(quant_scheme: Union[str, QuantScheme], rounding_mode: str, default_output_bw: int,
                                  default_param_bw: int, data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Perform sanity checks on inputs to QuantSim

        NOTE: This method will be deprecated.
              Call aimet_common.quantsim.validate_quantsim_inputs directly instead.

        :param quant_scheme: Quantization scheme. Supported options are 'tf_enhanced' or 'tf' or using Quant Scheme Enum
                             QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
        :param rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters
        :param data_type: Data type of the quantized values (int or float).
        """
        validate_quantsim_inputs(quant_scheme,
                                 rounding_mode,
                                 default_output_bw,
                                 default_param_bw,
                                 data_type)

    @staticmethod
    def _find_next_downstream_modules(op):
        downstream_modules = []
        for succeeding_op in list(op.output.consumers):
            if succeeding_op.get_module():
                downstream_modules.append(succeeding_op.get_module())

            elif succeeding_op.type == 'Split':
                downstream_modules += QuantizationSimModel._find_next_downstream_modules(succeeding_op)

        return downstream_modules

    @staticmethod
    def _export_encodings_to_files(model: torch.nn.Module, path: str, filename_prefix: str, op_to_io_tensor_map: Dict,
                                   valid_param_set: set, propagate_encodings: bool):
        """
        Save the quantized model weight encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: filename to store exported weight encodings in json format
        :param op_to_io_tensor_map: Dictionary of layer to I/O tensor mapping from onnx or torch script model
        :param valid_param_set: a set of valid param input names in model
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops.
        """

        # Create a dictionary to export to JSON
        activation_encodings_onnx = {}
        activation_encodings_torch = {}
        param_encodings = {}

        for layer_name, layer in QuantizationSimModel._get_qc_quantized_layers(model):
            QuantizationSimModel._update_encoding_dicts_for_layer(layer, layer_name, activation_encodings_onnx,
                                                                  activation_encodings_torch,
                                                                  param_encodings, op_to_io_tensor_map,
                                                                  valid_param_set, propagate_encodings)

        encodings_dict_onnx = {'version': encoding_version,
                               'activation_encodings': activation_encodings_onnx,
                               'param_encodings': param_encodings}

        encodings_dict_pytorch = {'version': encoding_version,
                                  'activation_encodings': activation_encodings_torch,
                                  'param_encodings': param_encodings}

        # export weight encodings to output json file
        encoding_file_path = os.path.join(path, filename_prefix + '.encodings')
        encoding_file_path_pytorch = os.path.join(path, filename_prefix + '_torch' + '.encodings')
        save_json_yaml(encoding_file_path, encodings_dict_onnx)
        save_json_yaml(encoding_file_path_pytorch, encodings_dict_pytorch)

    @staticmethod
    def _update_param_encodings_dict_for_layer(layer: torch.nn.Module, layer_name: str, param_encodings: Dict,
                                               valid_param_set: set):
        """
        :param layer: layer as torch.nn.Module
        :param layer_name : Name of the layer
        :param param_encodings: dictionary of param encodings
        :param valid_param_set: a set of valid param input names in model
        """

        for orig_param_name, param_quantizer in layer.param_quantizers.items():
            param_name = layer_name + '.' + orig_param_name

            if not param_quantizer.enabled:
                continue
            elif param_name not in valid_param_set:
                logger.error('Param tensor {%s} not found in valid param set', param_name)
                continue
            elif isinstance(param_quantizer.encoding, Iterable):
                param_encodings[param_name] = []
                for encoding in param_quantizer.encoding:
                    enc_dict = QuantizationSimModel._create_encoding_dict(encoding,
                                                                          param_quantizer, propagate_encodings=False)
                    param_encodings[param_name].append(enc_dict)
            else:
                enc_dict = QuantizationSimModel._create_encoding_dict(param_quantizer.encoding, param_quantizer,
                                                                      propagate_encodings=False)
                param_encodings[param_name] = [enc_dict]

    @staticmethod
    def _update_encoding_dicts_for_layer(layer: torch.nn.Module, layer_name: str, activation_encodings_onnx: Dict,
                                         activation_encodings_torch: Dict, param_encodings: Dict,
                                         op_to_io_tensor_map: Dict, valid_param_set: set, propagate_encodings: bool):
        """
        Add given layer param and activation encodings to respective dictionaries to be used for exporting encodings
        :param layer: layer as torch.nn.Module
        :param layer_name: Name of the layer
        :param activation_encodings_onnx: dictionary of activation encodings which maps onnx attribute to encodings
        :param activation_encodings_torch: dictionary of activation encodings which maps pytorch names to encodings
        :param param_encodings: dictionary of param encodings
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :param valid_param_set: a set of valid param input names in model
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops.
        """

        if layer_name not in op_to_io_tensor_map:
            logger.info("layer with name {%s} not found in model, not an issue; "
                        "skip and continue ", layer_name)
        else:
            if isinstance(layer, QcQuantizeWrapper):

                # --------------------------------------
                # Update encodings for Input activations
                # --------------------------------------
                QuantizationSimModel._update_encoding_dict_for_input_activations(layer, layer_name, op_to_io_tensor_map,
                                                                                 activation_encodings_onnx,
                                                                                 activation_encodings_torch)
                # ---------------------------------------
                # Update encodings for output activations
                # ---------------------------------------
                QuantizationSimModel._update_encoding_dict_for_output_activations(layer, layer_name,
                                                                                  op_to_io_tensor_map,
                                                                                  activation_encodings_onnx,
                                                                                  activation_encodings_torch,
                                                                                  propagate_encodings)
                # ---------------------------
                # Update encodings for Params
                # ---------------------------
                QuantizationSimModel._update_param_encodings_dict_for_layer(layer, layer_name, param_encodings,
                                                                            valid_param_set)

            if isinstance(layer, QcQuantizeRecurrent):
                # Update encodings for Recurrent layers
                QuantizationSimModel._update_encoding_dict_for_recurrent_layers(layer, layer_name, op_to_io_tensor_map,
                                                                                activation_encodings_onnx,
                                                                                param_encodings, propagate_encodings)

    @staticmethod
    def find_last_op_name_for_layer(layer_name: str, op_to_io_tensor_map: Dict) -> Tuple[str, List[str]]:
        """
        This function returns the last ONNX op and the list of ONNX Ops that were mapped from a PyTorch Op.
        What is referred to as the last op here is an ONNX op that has been temporarily named as
        <derived op name>.end in the function _set_onnx_node_names() in the file onnx_utils.py

        When a PyTorch Op is mapped to ONNX Op, there are many scenarios to consider.

        Scenario #1:
        There is a one-to-one mapping between a PyTorch Op and the corresponding ONNX Op.
        In this scenario, there won't be any ONNX Op that is marked as an .end Op
        This function returns the single ONNX Op as the last_op_name and the ONNX Op name associated with
        the PyTorch Op.

        Scenario #2:
        In this case there is exactly one ONNX Op that is marked as an .end Op.
        This function returns the last_op_name as and returns all the ONNX Op names associated with the PyTorch Op

        Scenario #3
        In the case of a PyTorch Op that returns multiple outputs, there are two possible scenarios.
        a) There are more than one ONNX Op that is marked as an .end Op
        b) There is NO ONNX Op that is marked as an .end Op
        For both these scenarios, this function returns the last_op_name as None and returns all the ONNX Op names
        associated with the PyTorch Op.

        :param layer_name: Name of the PyTorch layer
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :return: tuple(last op name, all op names)
        """
        op_names = [key for key in op_to_io_tensor_map if key.startswith(layer_name)]
        end_op_names = [op_name for op_name in op_names if op_name.endswith('.end')]

        if len(op_names) == 1:  # Scenario #1
            logger.debug('Scenario #1: For layer name %s, the op_names are %s and the end Ops are %s', layer_name,
                         op_names, end_op_names)
            last_op_name = op_names[0]
            return last_op_name, op_names

        if len(end_op_names) == 1:  # Scenario #2
            last_op_name = end_op_names[0]
            logger.debug('Scenario #2: For layer name %s, the op_names are %s and the end Ops are %s', layer_name,
                         op_names, end_op_names)
        else:  # Scenario #3
            last_op_name = None
            logger.debug('Scenario #3: For layer name %s, the op_names are %s and the end Ops are %s', layer_name,
                         op_names, end_op_names)
        return last_op_name, op_names

    @staticmethod
    def _update_encoding_dict_for_output_activations(layer: torch.nn.Module, layer_name: str, op_to_io_tensor_map: Dict,
                                                     activation_encodings_onnx: Dict, activation_encodings_torch: Dict,
                                                     propagate_encodings: bool):

        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-nested-blocks

        last_op_name, op_names = QuantizationSimModel.find_last_op_name_for_layer(layer_name,
                                                                                  op_to_io_tensor_map)

        if last_op_name is None:
            # This is the scenario where the number of .end Ops is more than 1 or 0
            # Refer scenario #3 defined in find_last_op_name_for_layer()
            num_onnx_ops = len(op_names)
            num_pytorch_output_quantizers = len(layer.output_quantizers)

            if num_onnx_ops == num_pytorch_output_quantizers:
                for index, (op_name, out_quantizer) in enumerate(zip(op_names, layer.output_quantizers)):
                    if out_quantizer.enabled:
                        onnx_output_tensor = op_to_io_tensor_map[op_name].outputs[0]
                        enc = QuantizationSimModel._create_encoding_dict(out_quantizer.encoding,
                                                                         out_quantizer,
                                                                         propagate_encodings=propagate_encodings)
                        activation_encodings_onnx[onnx_output_tensor] = [enc]
            else:
                logger.warning("\nFor layer_name: %s, the number of ONNX OPs: %s, doesn't match with the "
                               "number of PyTorch Output Quantizers: %s. Encodings are not generated.",
                               layer_name, num_onnx_ops, num_pytorch_output_quantizers)
        else:
            # There is exactly 1 last_op_name
            if not propagate_encodings:
                op_names = [last_op_name]

            for op_name in op_names:
                if op_to_io_tensor_map[op_name].outputs:
                    output_tensors = op_to_io_tensor_map[op_name].outputs
                    if len(output_tensors) != len(layer.output_quantizers):
                        logger.error("For ONNX node: %s, encodings are not generated. "
                                     "Number of output quantizers: %d available for layer: %s "
                                     "doesn't match with number of output tensors: %d for ONNX node: %s",
                                     op_name, len(layer.output_quantizers), layer_name, len(output_tensors), op_name)

                    for index, output_tensor in enumerate(output_tensors):
                        propagate_flag = propagate_encodings and op_name != last_op_name

                        quantizer = layer.output_quantizers[0]
                        if propagate_flag is False:
                            quantizer = layer.output_quantizers[index]

                        if quantizer.enabled:
                            enc = QuantizationSimModel._create_encoding_dict(quantizer.encoding,
                                                                             quantizer,
                                                                             propagate_encodings=propagate_flag)
                            activation_encodings_onnx[output_tensor] = [enc]

                            # Check if layer exists in the pytorch encoding dictionary
                            if layer_name not in activation_encodings_torch:
                                activation_encodings_torch[layer_name] = {}
                            if QUANTIZER_TYPE_OUTPUT not in activation_encodings_torch[layer_name]:
                                activation_encodings_torch[layer_name][QUANTIZER_TYPE_OUTPUT] = {}
                            activation_encodings_torch[layer_name][QUANTIZER_TYPE_OUTPUT][index] = enc


    @staticmethod
    def _update_encoding_dict_for_input_activations(layer: torch.nn.Module, layer_name: str, op_to_io_tensor_map: Dict,
                                                    activation_encodings_onnx: Dict, activation_encodings_torch: Dict):

        param_inputs = [layer_name + '.' + param_name for param_name in layer.param_quantizers]
        input_tensors = [t for t in op_to_io_tensor_map[layer_name].inputs if t not in param_inputs]
        for index, input_tensor in enumerate(input_tensors):
            if (index < len(layer.input_quantizers)) and layer.input_quantizers[index].enabled:
                encoding = QuantizationSimModel._create_encoding_dict(layer.input_quantizers[index].encoding,
                                                                      layer.input_quantizers[index],
                                                                      propagate_encodings=False)
                activation_encodings_onnx[input_tensor] = [encoding]
                # Check if layer exists in the pytorch encoding dictionary
                if layer_name not in activation_encodings_torch:
                    activation_encodings_torch[layer_name] = {}
                if QUANTIZER_TYPE_INPUT not in activation_encodings_torch[layer_name]:
                    activation_encodings_torch[layer_name][QUANTIZER_TYPE_INPUT] = {}
                # Store encodings for a particular index so that they can be used to check if a quantizer was
                # enabled or not
                activation_encodings_torch[layer_name][QUANTIZER_TYPE_INPUT][index] = encoding

    @staticmethod
    def _update_encoding_dict_for_recurrent_layers(layer: torch.nn.Module, layer_name: str, op_to_io_tensor_map: Dict,
                                                   activation_encodings_onnx: Dict, param_encodings: Dict,
                                                   propagate_encodings: bool):
        """

        :param layer:
        :param layer_name:
        :param op_to_io_tensor_map:
        :param activation_encodings_onnx:
        :param param_encodings:
        :param propagate_encodings:
        :return:
        """

        # pylint: disable=too-many-nested-blocks
        # pylint: disable=too-many-locals

        onnx_activations_to_quantizers, onnx_params_to_quantizers = \
            layer.get_activation_param_quantizers_for_onnx_tensors(op_to_io_tensor_map[layer_name +
                                                                                       '#root_node'])
        # ------------------
        # Activations
        # ------------------
        quantizer = None
        for tensor, quantizer in onnx_activations_to_quantizers.items():
            encoding = QuantizationSimModel._create_encoding_dict(quantizer.encoding, quantizer,
                                                                  propagate_encodings=False)
            activation_encodings_onnx[tensor] = [encoding]

        if propagate_encodings and quantizer:
            _, op_names = QuantizationSimModel.find_last_op_name_for_layer(layer_name, op_to_io_tensor_map)
            for op_name in op_names:
                io_tensor_list = op_to_io_tensor_map[op_name]
                if not isinstance(io_tensor_list, list):
                    io_tensor_list = [io_tensor_list]

                for io_tensors in io_tensor_list:

                    if io_tensors.outputs:
                        for output_tensor in io_tensors.outputs:
                            if output_tensor in onnx_activations_to_quantizers:
                                continue
                            encoding = QuantizationSimModel._create_encoding_dict(quantizer.encoding, quantizer,
                                                                                  True)

                            activation_encodings_onnx[output_tensor] = [encoding]

        # ------------------
        # Params
        # ------------------
        for tensor, quantizer in onnx_params_to_quantizers.items():
            encoding = QuantizationSimModel._create_encoding_dict(quantizer.encoding, quantizer,
                                                                  propagate_encodings=False)
            param_encodings[tensor] = [encoding]

    @staticmethod
    def _get_qc_quantized_layers(model) -> List[Tuple[str, QcQuantizeWrapper]]:
        quantized_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (QcQuantizeWrapper, QcQuantizeRecurrent)):
                quantized_layers.append((name, module))
        return quantized_layers

    @staticmethod
    def _is_quantizable_module(module_ref):
        """ Function to check if a module is eligible for quantization.
            If the module is NOT an PyTorch module type or if the module was already
            Quantized or if the module is in the layers_to_ignore list, don't quantize.
        """

        if isinstance(module_ref, unquantizable_modules):
            logger.debug("Module %s not quantizable", module_ref)
            return False

        logger.debug("Module %s is quantizable", module_ref)
        return True

    def _create_quantizer_module(self, module_to_quantize: torch.nn.Module, num_inout_tensors: Dict,
                                 data_type: QuantizationDataType) -> torch.nn.Module:
        """Instantiates wrapper based on quant scheme
        """
        assert self._quant_scheme in [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced,
                                      QuantScheme.training_range_learning_with_tf_enhanced_init,
                                      QuantScheme.training_range_learning_with_tf_init,
                                      QuantScheme.post_training_percentile]

        # We lookup the number of input and output tensors already determined
        # Special case, we are adding a wrapper for a module not in the forward pass: Use default of 1, 1
        num_in_tensors, num_out_tensors = num_inout_tensors.get(module_to_quantize, (1, 1))

        # Set quantizer to be a module replacer if it is in qc_quantize_modules_dict, otherwise set as
        # StaticGridQuantWrapper.
        quantizer_wrapper_type = qc_quantize_modules_dict.get(type(module_to_quantize), StaticGridQuantWrapper)

        if self._quant_scheme in [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced,
                                  QuantScheme.post_training_percentile]:
            quant_scheme_for_initialization = self._quant_scheme

        elif self._quant_scheme == QuantScheme.training_range_learning_with_tf_init:
            quant_scheme_for_initialization = QuantScheme.post_training_tf

        elif self._quant_scheme == QuantScheme.training_range_learning_with_tf_enhanced_init:
            quant_scheme_for_initialization = QuantScheme.post_training_tf_enhanced

        quantized_module = quantizer_wrapper_type(module_to_quantize, self._default_param_bw, self._default_output_bw,
                                                  self._rounding_mode, quant_scheme_for_initialization, num_inputs=num_in_tensors,
                                                  num_outputs=num_out_tensors, data_type=data_type)

        return quantized_module

    def _add_quantization_wrappers(self, module, num_inout_tensors, default_data_type: QuantizationDataType):
        """Recursively add quantization wrappers to all appropriate modules starting with module
        """
        for module_name, module_ref in module.named_children():
            logger.debug("nn.Module found : %s", module_ref)

            # check if the module already quantized then ignore
            if not self._is_quantizable_module(module_ref):
                continue

            # check if the module is leaf or not
            if utils.is_leaf_module(module_ref):

                # Create a new QcQuantize wrapper module
                quantized_module = self._create_quantizer_module(module_ref, num_inout_tensors, default_data_type)

                setattr(module, module_name, quantized_module)

            # recursively call children modules
            else:
                self._add_quantization_wrappers(module_ref, num_inout_tensors, default_data_type)

    @staticmethod
    def _create_encoding_dict(encoding: libpymo.TfEncoding, quantizer, propagate_encodings: bool) -> Union[Dict, None]:
        """
        Create encoding dictionary from encoding object
        :param encoding: Encoding of the quantizer
        :param quantizer: Tensor Quantizer
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops.
        :return: Encoding Dictionary
        """
        data_type, bitwidth = quantizer.data_type, quantizer.bitwidth

        if data_type == QuantizationDataType.float:
            enc_dict = {'bitwidth': bitwidth, 'dtype': "float"}
        else:
            if encoding:
                encoding_min, encoding_max, bw, scale, offset = encoding.min, encoding.max, encoding.bw, \
                                                                encoding.delta, encoding.offset
                is_symmetric = quantizer.use_symmetric_encodings
                if not isinstance(quantizer, StaticGridTensorQuantizer):
                    scale, offset = calculate_delta_offset(encoding_min, encoding_max, bitwidth)

                if propagate_encodings:
                    # Shortened encodings will be filled into a layer that only exists due to expansion of PyTorch ops
                    # into multiple ONNX ops so that it's necessarily to use the same bitwidth and type
                    enc_dict = {'bitwidth': bw, 'dtype': "int"}
                else:
                    enc_dict = {'min': encoding_min, 'max': encoding_max, 'scale': scale, 'offset': int(offset),
                                'bitwidth': bw, 'is_symmetric': str(is_symmetric), 'dtype': "int"}
            else:
                enc_dict = None
        return enc_dict

    @classmethod
    def _remove_quantization_wrappers(cls, starting_module, list_of_modules_to_exclude):
        """
        Recursively remove quantization wrappers from all appropriate modules starting with a given module
        :param starting_module: Module to recursive search downstream from
        :param list_of_modules_to_exclude: List of torch modules to remove quantization wrappers from (if present)
        :return: None
        """
        for module_name, module_ref in starting_module.named_children():

            # If modules is in the exclude list, remove the wrapper
            if module_ref in list_of_modules_to_exclude:

                if isinstance(module_ref, QcQuantizeWrapper):
                    # Remove the wrapper, gets auto-deleted
                    # pylint: disable=protected-access
                    setattr(starting_module, module_name, module_ref._module_to_wrap)

                elif isinstance(module_ref, QcQuantizeStandAloneBase):
                    setattr(starting_module, module_name, torch.nn.Identity())

                elif isinstance(module_ref, QcQuantizeRecurrent):
                    module_ref.update_params()
                    setattr(starting_module, module_name, module_ref.module_to_quantize)

            # Recursively call children modules if present
            if not utils.is_leaf_module(module_ref):
                cls._remove_quantization_wrappers(module_ref, list_of_modules_to_exclude)

    def _add_inputs_hook(self, hooks):
        module_to_name_map = {}
        for name, module in self.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # pylint: disable=protected-access
                module_to_name_map[module._module_to_wrap] = name

        # Add any leaf modules that are not wrapped by QcQuantizeWrapper (like Identity)
        for name, module in self.model.named_modules():
            if utils.is_leaf_module(module) and module not in module_to_name_map.keys():
                module_to_name_map[module] = name

        def inputs_hook(module_ref, inputs, _):
            # Need to remove hook here, otherwise the jit trace of CustomMarker with module ref will error since the
            # hook will be recursively hit.
            hooks[module_ref].remove()
            del hooks[module_ref]
            module_name = module_to_name_map[module_ref]
            marker_layer = torch.jit.trace(CustomMarker(module_ref, module_name),
                                           inputs)
            self._module_marker_map[module_name] = marker_layer

        for name, module in self.model.named_modules():
            if name not in self._module_marker_map and utils.is_leaf_module(module):
                hooks[module] = module.register_forward_hook(inputs_hook)

    def _validate_module_marker_map(self):
        """
        Check to make sure all leaf modules have traced Custom Markers associated with them.
        """
        all_leaf_modules = set()
        missing_inputs_entries = []
        for name, module in self.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                all_leaf_modules.add(name)

        # Add any modules that are not wrapped by QcQuantizeWrappers (like Identity)
        for name, module in self.model.named_modules():
            if utils.is_leaf_module(module) and '_module_to_wrap' not in name:
                all_leaf_modules.add(name)

        for leaf_module in all_leaf_modules:
            if leaf_module not in self._module_marker_map.keys():
                missing_inputs_entries.append(leaf_module)

        if missing_inputs_entries:
            logger.info('In order to export a conditional model, all leaf modules need to be run with some input so '
                        'torch trace can be done.')
            logger.info('The following modules were not run during compute encodings:')
            logger.info(missing_inputs_entries)
            logger.info('Please use the sim.run_modules_for_traced_custom_marker(<module list>, dummy_input) api to '
                        'pass dummy inputs to these modules.')
            logger.info('Modules which can take the same dummy input can be '
                        'grouped as a list. For groups of modules with different input shapes, please call '
                        'sim.run_modules_for_traced_custom_markers() for each group.')
            logger.info('Exiting quantsim export early.')
            return False
        return True

    def _export_conditional(self, path: str, filename_prefix: str, dummy_input: Union[torch.Tensor, Tuple],
                            forward_pass_callback: Callable, forward_pass_callback_args,
                            onnx_export_args: Union[OnnxExportApiArgs, None] = OnnxExportApiArgs(),
                            propagate_encodings: bool = False):
        """
        Export function for conditional models. Performs another round of forward passes to create and store traced
        CustomMarker info for each leaf module to be later used when scripting the model for export.
        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param dummy_input: Dummy input to the model. Used to parse model graph. It is required for the dummy_input to
                be placed on CPU.
        :param forward_pass_callback: A callback function that simply runs forward passes on the model. This callback
            function should use representative data for the forward pass, so the calculated encodings work for all
            data samples. This callback internally chooses the number of data samples it wants to use for calculating
            encodings. The callback should exercise all paths of the conditional model.
        :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
            the user to determine the type of this parameter. E.g. could be simply an integer representing the number
            of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
            If set to None, forward_pass_callback will be invoked with no parameters.
        :param onnx_export_args: onnx specific export arguments
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops.
        :return: None
        """
        self._is_conditional = True
        if onnx_export_args is None:
            onnx_export_args = OnnxExportApiArgs()

        # If model is conditional, we need to create traced CustomMarkers to be used later during export. Create hooks
        # here for creating a traced CustomMarker for each leaf module during the forward pass callback.
        hooks = {}
        if self._is_conditional:
            self._add_inputs_hook(hooks)

        with utils.in_eval_mode(self.model), torch.no_grad():
            _ = forward_pass_callback(self.model, forward_pass_callback_args)

        # Any hooks that were hit during forward pass callback would have removed themselves. Remove the remaining
        # hooks that were not run.
        for h in hooks.values():
            h.remove()

        # Check that all paths were exercised
        if not self._validate_module_marker_map():
            return
        self.export(path, filename_prefix, dummy_input, onnx_export_args, propagate_encodings)

    def configure_quantization_ops(self, config_file: str, default_output_bw: int, default_param_bw: int,
                                   default_data_type: QuantizationDataType) -> QuantSimConfigurator:
        """
        Configure inserted quantize ops using config file and fill in all the supported kernels
        :param config_file: Configuration file to use
        :param default_output_bw: default bitwidth for activations
        :param default_param_bw: default bitwidth for params
        :param default_data_type: default data type
        :return: QuantSimConfigurator object
        """
        if self.connected_graph is None:
            logger.error('A connected graph failed to be built.\n'
                         'Unable to proceed with automatically configuring quantization ops using the config file.\n'
                         'Please configure quantization ops manually by redefining '
                         'QuantizationSimModel.configure_quantization_ops()')
            raise AssertionError
        return QuantSimConfigurator(self.model, self.connected_graph, config_file, default_output_bw,
                                    default_param_bw, default_data_type)

    def set_and_freeze_param_encodings(self, encoding_path: str):
        """
        Set and freeze parameter encodings from encodings JSON file
        :param encoding_path: path from where to load parameter encodings file
        """
        # Load parameter encodings file
        with open(encoding_path) as json_file:
            param_encodings = json.load(json_file)

        for name, quant_module in self.model.named_modules():
            if isinstance(quant_module, QcQuantizeWrapper):
                quant_module.set_param_encoding(name, param_encodings)
                quant_module.freeze_param_encoding(name, param_encodings)

    def quant_wrappers(self):
        """
        Generator for yielding all quantization wrappers
        """
        for module in self.model.modules():
            if isinstance(module, (QcQuantizeWrapper, QcQuantizeRecurrent)):
                yield module

    def run_modules_for_traced_custom_marker(self, module_list: List[torch.nn.Module], dummy_input):
        """
        Given a list of modules to run and dummy input for the module, create a traced CustomMarker for each module
        and store it in the module_marker map. The same dummy input will be used for all modules.
        :param module_list: List of modules to create traced CustomMarkers for
        :param dummy_input: Dummy input for all modules
        """
        module_to_name_map = {}
        for name, module in self.model.named_modules():
            if utils.is_leaf_module(module):
                if '._module_to_wrap' in name:
                    module_to_name_map[module] = name[:MODULE_TO_WRAP_STRING_REVERSE_INDEX]
                else:
                    module_to_name_map[module] = name

        for module in module_list:
            if isinstance(module, QcQuantizeWrapper):
                module = getattr(module, '_module_to_wrap')
            # Only perform init and trace if the given module is a leaf module, and we have not recorded it before
            if module in module_to_name_map and module_to_name_map[module] not in self._module_marker_map:
                with utils.in_eval_mode(module), torch.no_grad():
                    marker_layer = torch.jit.trace(CustomMarker(module, module_to_name_map[module]),
                                                   dummy_input)
                self._module_marker_map[module_to_name_map[module]] = marker_layer


def save_checkpoint(quant_sim_model: QuantizationSimModel, file_path: str):
    """
    This API provides a way for the user to save a checkpoint of the quantized model which can
    be loaded at a later point to continue fine-tuning e.g.
    See also load_checkpoint()

    :param quant_sim_model: QuantizationSimModel to save checkpoint for
    :param file_path: Path to the file where you want to save the checkpoint
    :return: None
    """
    with open(file_path, 'wb') as file:
        pickle.dump(quant_sim_model, file)


def load_checkpoint(file_path: str) -> QuantizationSimModel:
    """
    Load the quantized model

    :param file_path: Path to the file where you want to save the checkpoint
    :return: A new instance of the QuantizationSimModel created after loading the checkpoint
    """
    with open(file_path, 'rb') as file:
        sim = pickle.load(file)
        return sim


def check_accumulator_overflow(model: torch.nn.Module, quant_bw: int, accum_bw: int):
    """
    Checks for any potential for accumulator overflow across all the layers of the given model
    :param model: Model
    :param quant_bw: Bitwidth the layers are quantized at
    :param accum_bw: Bitwidth of the accumulator
    :return: Name of the layer with the most accumulator range used and range used
    """

    most_accum_range_used = 0
    most_accum_range_used_layer = None

    for layer_name, layer in model.named_modules():

        if isinstance(layer, torch.nn.Conv2d):
            was_accum_range_exceeded, accum_range_used = get_conv_accum_bounds(layer.weight.detach().numpy(),
                                                                               quant_bw, accum_bw)
            if accum_range_used > most_accum_range_used:
                most_accum_range_used = accum_range_used
                most_accum_range_used_layer = layer_name

            if was_accum_range_exceeded:
                logger.info('Possible accumulator overflow for layer: %s', layer_name)

    if most_accum_range_used < 1:
        logger.info('No overflow detected. Layer %s had the most accumulator range used: %f%%',
                    most_accum_range_used_layer, most_accum_range_used * 100)
    else:
        logger.info('Overflow detected. Layer %s had the most accumulator range used: %f%%',
                    most_accum_range_used_layer, most_accum_range_used * 100)

    return most_accum_range_used_layer, most_accum_range_used


def load_encodings_to_sim(quant_sim_model: QuantizationSimModel, pytorch_encoding_path: str):
    """
    Loads the saved encodings to quant sim model
    :param quant_sim_model: quantized model. Note: The model configuration should be the same as when encodings were exported
    :param pytorch_encoding_path:
    :return:
    """
    # Load encodings file
    with open(pytorch_encoding_path) as json_file:
        encodings = json.load(json_file)

    # Set param encodings
    for name, quant_module in quant_sim_model.model.named_modules():
        if isinstance(quant_module, QcQuantizeWrapper):
            quant_module.set_param_encoding(name, encodings['param_encodings'])
            quant_module.set_mode(QcQuantizeOpMode.ACTIVE)
            if name in encodings['activation_encodings']:
                quant_module.set_activation_encoding(name, encodings['activation_encodings'])

    quant_sim_model.replace_wrappers_for_quantize_dequantize()
