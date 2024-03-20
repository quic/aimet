# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import contextlib
import os
import io
import copy
import pickle
from typing import Tuple, List, Union, Dict, Callable, Optional, Any, runtime_checkable, Protocol
from collections import OrderedDict, defaultdict
import json
import torch
import onnx
from packaging import version

import aimet_common
import aimet_common.libpymo as libpymo

from aimet_common.connected_graph.connectedgraph_utils import CG_SPLIT
from aimet_common.utils import AimetLogger, save_json_yaml, log_with_error_and_assert_if_false
from aimet_common.defs import QuantScheme, QuantizationDataType, SupportedKernelsAction, QuantDtypeBwInfo
from aimet_common.quantsim import encoding_version, validate_quantsim_inputs, extract_global_quantizer_args
from aimet_common.quant_utils import get_conv_accum_bounds

from aimet_torch.quantsim_config.quantsim_config import QuantSimConfigurator
from aimet_torch.qc_quantize_op import QcQuantizeStandAloneBase, QcQuantizeWrapper, QcQuantizeOpMode, \
    StaticGridQuantWrapper, LearnedGridQuantWrapper, NativeTorchQuantWrapper, QUANTIZER_TYPE_INPUT, QUANTIZER_TYPE_OUTPUT
from aimet_torch.tensor_quantizer import initialize_learned_grid_quantizer_attributes
from aimet_torch.qc_quantize_op import get_encoding_by_quantizer as _get_encoding_by_quantizer
from aimet_torch import torchscript_utils, utils, transformer_utils, onnx_utils
from aimet_torch.onnx_utils import OnnxSaver, OnnxExportApiArgs, CustomMarker, get_pytorch_name_from_onnx_name
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.qc_quantize_recurrent import QcQuantizeRecurrent
from aimet_torch.v2.quantization.builder import LazyQuantizeWrapper
from aimet_torch.v2.nn import BaseQuantizationMixin


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# Types of modules which cannot be quantized
unquantizable_modules = (
    QcQuantizeWrapper,
    QcQuantizeStandAloneBase,
    QcQuantizeRecurrent,
    BaseQuantizationMixin,
    torch.nn.Identity,
)

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

SUPPORTED_KERNELS_ACTION = SupportedKernelsAction.warn_on_error


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


@runtime_checkable
class ExportableQuantModule(Protocol):
    """
    Defines the minimum interface requirements for exporting encodings from a module.
    """

    def export_input_encodings(self) -> List[List[Dict]]:
        """
        Returns a list of input encodings, each represented as a List of Dicts
        """

    def export_output_encodings(self) -> List[List[Dict]]:
        """
        Returns a list of output encodings, each represented as a List of Dicts
        """

    def export_param_encodings(self) -> Dict[str, List[Dict]]:
        """
        Returns a dict of {param name: param encodings}, with each encoding represented as a List of Dicts
        """

    def import_input_encodings(self, encodings: Dict[str, Dict], ignore_when_quantizer_disabled=False,
                               disable_quantizer_without_encoding=True, freeze=False):
        """
        Import input encodings represented in below format:
        {
            '0': dict,
            '1': dict,
            ...
        }
        """

    def import_output_encodings(self, encodings: Dict[str, Dict], ignore_when_quantizer_disabled=False,
                                disable_quantizer_without_encoding=True, freeze=False):
        """
        Import output encodings represented in below format:
        {
            '0': dict,
            '1': dict,
            ...
        }
        """

    def import_param_encodings(self, encodings: Dict[str, List[Dict]], ignore_when_quantizer_disabled=False,
                               disable_quantizer_without_encoding=True, freeze=False):
        """
        Import parameter encodings represented in below format:
        {
            'param_name_0': [dict, dict, ...],
            'param_name_1': [dict, dict, ...],
            ...
        }
        """

    def get_original_module(self) -> torch.nn.Module:
        """
        Returns the floating point version of quantized module
        """


class QuantizationSimModel:
    """
    Implements mechanism to add quantization simulations ops to a model. This allows for off-target simulation of
    inference accuracy. Also allows the model to be fine-tuned to counter the effects of quantization.
    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes, too-many-locals, too-many-public-methods
    def __init__(self, model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple],
                 quant_scheme: Union[str, QuantScheme] = QuantScheme.post_training_tf_enhanced,
                 rounding_mode: str = 'nearest', default_output_bw: int = 8, default_param_bw: int = 8,
                 in_place: bool = False, config_file: str = None,
                 default_data_type: QuantizationDataType = QuantizationDataType.int):

        """
        Constructor for QuantizationSimModel.

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
        self._config_file = config_file
        self._is_conditional = False
        self._module_marker_map = {}
        self._percentile_value = 100 # default percentile value
        self._excluded_layer_names = []

        # Add quantization layers
        num_inout_tensors = utils.find_num_inout_tensors_per_module(self.model, dummy_input)
        inout_tensors_dtypes_for_cast_ops = utils.get_inout_tensors_dtypes_for_cast_modules(self.model, dummy_input)

        self._add_quantization_wrappers(self.model, num_inout_tensors, default_data_type)
        self._set_tensor_quantizers_for_consts()

        # Disable bias quantization
        self.exclude_param_from_quantization("bias")

        # override specific quantizers to tf mode in transformer model
        self._override_quant_config_for_transformer_layers()

        quantsim_configurator = self.configure_quantization_ops(config_file, default_output_bw, default_param_bw,
                                                                default_data_type)

        self.quant_args = extract_global_quantizer_args(quant_scheme, quantsim_configurator)

        self._enable_output_quantizers_for_specific_cast_ops(inout_tensors_dtypes_for_cast_ops)

        # pylint: disable=protected-access
        self._hw_version = quantsim_configurator._get_hw_version()
        self._supported_kernels = quantsim_configurator.get_supported_kernels()
        self._validate_supported_kernels_for_quantizers(SUPPORTED_KERNELS_ACTION)

        # Initialize real wrappers using collected information
        self._realize_quant_wrappers_in_model(self.model)

    def _realize_quant_wrappers_in_model(self, model: torch.nn.Module):
        """
        Prepare QuantSim for compute encodings. Resets encodings for each quantizable layer and sets mode to Analysis.
        Realize quant wrappers using collected information in LazyQuantWrapper.

        :param model: model containing modules wrapped with LazyQuantWrapper
        """
        for module_name, module_ref in model.named_children():
            if isinstance(module_ref, LazyQuantizeWrapper):
                quantized_module = self._realize_quant_wrapper(module_ref)
                setattr(model, module_name, quantized_module)

            elif not utils.is_leaf_module(module_ref):
                self._realize_quant_wrappers_in_model(module_ref)

    @staticmethod
    def _realize_quant_wrapper(module: torch.nn.Module) -> QcQuantizeWrapper:
        return module.realize_v1_wrapper()

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

        def print_quantizer_state(stream, quantizer, prefix_string):
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

        for layer_name, layer in self._get_qc_quantized_layers(self.model):
            stream.write('----------------------------------------------------------\n')
            stream.write('Layer: {}\n'.format(layer_name))

            # Inputs
            if isinstance(layer.input_quantizers, dict):
                for name, quantizer in layer.input_quantizers.items():
                    print_quantizer_state(stream, quantizer, prefix_string=f"Input[{name}]")
            else:
                for index, quantizer in enumerate(layer.input_quantizers):
                    print_quantizer_state(stream, quantizer, prefix_string=f"Input[{index}]")

            # Params
            for param_name, quantizer in layer.param_quantizers.items():
                print_quantizer_state(stream, quantizer, prefix_string=f"Param[{param_name}]")

            # Outputs
            if isinstance(layer.output_quantizers, dict):
                for name, quantizer in layer.output_quantizers.items():
                    print_quantizer_state(stream, quantizer, prefix_string=f"Output[{name}]")
            else:
                for index, quantizer in enumerate(layer.output_quantizers):
                    print_quantizer_state(stream, quantizer, prefix_string=f"Output[{index}]")

        return stream.getvalue()

    @staticmethod
    def prepare_sim_for_compute_encodings(sim: 'QuantizationSimModel'):
        """
        Prepare QuantSim for compute encodings. Resets encodings for each quantizable layer and sets mode to Analysis.

        :param sim: QuantSim to prepare
        """
        # pylint: disable=protected-access
        quantized_layers = sim._get_qc_quantized_layers(sim.model)

        for _, layer in quantized_layers:
            # Clear stats and encodings if they are present
            layer.reset_encodings()

            # And set the mode to analysis
            layer.set_mode(QcQuantizeOpMode.ANALYSIS)

        for _, layer in quantized_layers:
            # call only when quant scheme is percentile
            if sim._quant_scheme == QuantScheme.post_training_percentile:
                layer.set_percentile_value(sim._percentile_value)

    @staticmethod
    def compute_layer_encodings_for_sim(sim: 'QuantizationSimModel'):
        """
        Compute encodings for each quantizable layer in sim after forward pass has been called.

        :param sim: QuantSim to compute encodings for
        """
        # pylint: disable=protected-access
        quantized_layers = sim._get_qc_quantized_layers(sim.model)
        # Get the computed per-layer encodings and log them
        for name, layer in quantized_layers:
            layer.compute_encoding()

            # Before we return we set the mode to active - meaning ready for quantize/de-quantize
            # for layers with valid_encoding, otherwise we set to pass through
            if isinstance(layer, QcQuantizeRecurrent):
                sim.set_mode_for_recurrent_module(layer, name)
            else:
                # By default we want to set the Quantization wrappers to ACTIVE mode
                layer.set_mode(QcQuantizeOpMode.ACTIVE)

        sim.replace_wrappers_for_quantize_dequantize()

        sim._clamp_transformer_attention_mask_encoding()

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

        QuantizationSimModel.prepare_sim_for_compute_encodings(self)

        # Run forward iterations so we can collect statistics to compute the appropriate encodings
        with utils.in_eval_mode(self.model), torch.no_grad():
            _ = forward_pass_callback(self.model, forward_pass_callback_args)

        QuantizationSimModel.compute_layer_encodings_for_sim(self)

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
               onnx_export_args: Optional[Union[OnnxExportApiArgs, Dict]] = None, propagate_encodings: bool = False,
               export_to_torchscript: bool = False, use_embedded_encodings: bool = False):
        """
        This method exports out the quant-sim model so it is ready to be run on-target.

        Specifically, the following are saved:

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
        :param onnx_export_args: Optional export argument with onnx specific overrides provided as a dictionary or
            OnnxExportApiArgs object. If not provided, defaults to "opset_version" = None, "input_names" = None,
            "output_names" = None, and for torch version < 1.10.0, "enable_onnx_checker" = False.
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops. Defaults to False.
        :param export_to_torchscript: If True, export to torchscript. Export to onnx otherwise. Defaults to False.
        :param use_embedded_encodings: If True, another onnx model embedded with fakequant nodes will be exported
        """
        # save the quantized model and encodings
        model_filename = filename_prefix + '.pth'
        model_path = os.path.join(path, model_filename)

        # Create a version of the model without any quantization ops
        model_to_export = QuantizationSimModel.get_original_model(self.model)

        torch.save(model_to_export, model_path)

        if onnx_export_args is None:
            onnx_export_args = {'opset_version': None,
                                'input_names': None,
                                'output_names': None}
            if version.parse(torch.__version__) < version.parse("1.10.0") and isinstance(onnx_export_args, dict):
                onnx_export_args['enable_onnx_checker'] = False
        log_with_error_and_assert_if_false(isinstance(onnx_export_args, (OnnxExportApiArgs, dict)),
                                           logger,
                                           f'unsupported opt_args type={type(onnx_export_args)}')

        if use_embedded_encodings:
            QuantizationSimModel.save_model_with_embedded_quantization_nodes(self.model, path, filename_prefix, dummy_input,
                                                                             onnx_export_args, export_to_torchscript, self._is_conditional)
        else:
            if export_to_torchscript:
                self.export_torch_script_model_and_encodings(path, filename_prefix, model_to_export, self.model,
                                                             dummy_input, self._excluded_layer_names)
            else:
                self.export_onnx_model_and_encodings(path, filename_prefix, model_to_export, self.model,
                                                     dummy_input, onnx_export_args, propagate_encodings,
                                                     self._module_marker_map, self._is_conditional,
                                                     self._excluded_layer_names, quantizer_args=self.quant_args)

    @staticmethod
    def export_torch_script_model_and_encodings(path: str, filename_prefix: str,
                                                original_model: torch.nn.Module,
                                                sim_model: torch.nn.Module,
                                                dummy_input: Union[torch.Tensor, Tuple],
                                                excluded_layer_names: List = None):
        """
        This method exports a torchscript mode and the corresponding encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param original_model: model without the quantsim wrappers
        :param sim_model: model with the quantsim wrappers
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :param excluded_layer_names: List of names of layers that have been excluded from quantization.
        :return: None
        """
        # Create torchscript model and obtain node to i/o tensor name map
        ts_path = os.path.join(path, filename_prefix + '.torchscript.pth')
        with utils.in_eval_mode(original_model), torch.no_grad():
            torchscript_utils.create_torch_script_model(ts_path, original_model, dummy_input)

            trace = torch.jit.load(ts_path)
            torch_script_node_io_tensor_map, valid_param_set = \
                torchscript_utils.get_node_to_io_tensor_names_map(original_model, trace, dummy_input)

        # Export encodings
        QuantizationSimModel._export_encodings_to_files(sim_model, path, filename_prefix,
                                                        torch_script_node_io_tensor_map, valid_param_set,
                                                        excluded_layer_names, propagate_encodings=False)

    @staticmethod
    def export_onnx_model_and_encodings(path: str, filename_prefix: str, original_model: torch.nn.Module,
                                        sim_model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple],
                                        onnx_export_args: Union[OnnxExportApiArgs, dict], propagate_encodings: bool,
                                        module_marker_map: Dict[torch.nn.Module, torch.Tensor] = None,
                                        is_conditional: bool = False, excluded_layer_names: List = None,
                                        quantizer_args: Dict = None):
        """
        This method exports a onnx model and the corresponding encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param original_model: model without the quantsim wrappers
        :param sim_model: model with the quantsim wrappers
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :param onnx_export_args: Additional onnx export args including export api overrides
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
               multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
               ops.
        :param module_marker_map: Maps module names to traced custom markers (only used for conditional models)
        :param is_conditional: True if model is conditional, False otherwise
        :param excluded_layer_names: List of names of layers that have been excluded from quantization.
        :return: None

        """
        # pylint: disable=too-many-locals
        onnx_path = os.path.join(path, filename_prefix + '.onnx')
        if version.parse(torch.__version__) >= version.parse("1.13.0") and onnx_utils.EXPORT_TO_ONNX_DIRECT:
            logger.debug('Exporting quantsim using torch.onnx.export directly')
            original_model.cpu()
            if isinstance(onnx_export_args, OnnxExportApiArgs):
                kwargs = onnx_export_args.kwargs
            else:
                kwargs = onnx_export_args
            torch.onnx.export(original_model, dummy_input, onnx_path, **kwargs)
        else:
            # Create onnx model and obtain node to i/o tensor name map
            OnnxSaver.create_onnx_model_with_pytorch_layer_names(onnx_path, original_model, dummy_input, is_conditional,
                                                                 module_marker_map, onnx_export_args)

        onnx_model = onnx.load(onnx_path)
        onnx_node_to_io_tensor_map, valid_param_set = OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)

        # Export encodings
        QuantizationSimModel._export_encodings_to_files(sim_model, path, filename_prefix,
                                                        onnx_node_to_io_tensor_map, valid_param_set,
                                                        excluded_layer_names, propagate_encodings,
                                                        quantizer_args=quantizer_args)

    def save_encodings_to_json(self, path: str, filename_prefix: str):
        """
        Save encodings in the model to json.

        :param path: Path to save file
        :param filename_prefix: Filename to use for saved file
        """
        activation_encodings, param_encodings = self.get_activation_param_encodings()
        encodings_dict = {'activation_encodings': activation_encodings, 'param_encodings': param_encodings}
        with open(os.path.join(path, filename_prefix + '.json'), 'w') as encoding_json:
            json.dump(encodings_dict, encoding_json, sort_keys=True, indent=4)

    def get_activation_param_encodings(self):
        """
        Get activation and param encodings from sim.model.

        :return: Tuple of activation and param encodings dictionaries mapping torch module names to encodings
        """
        activation_encodings = OrderedDict()
        param_encodings = OrderedDict()

        for module_name, module in self.model.named_modules():
            if not isinstance(module, ExportableQuantModule):
                continue

            activation_encodings[module_name] = defaultdict(OrderedDict)

            for i, encoding in enumerate(module.export_input_encodings()):
                if not encoding:
                    continue
                if len(encoding) == 1:
                    encoding = encoding[0]
                activation_encodings[module_name]['input'][i] = encoding

            for i, encoding in enumerate(module.export_output_encodings()):
                if not encoding:
                    continue
                if len(encoding) == 1:
                    encoding = encoding[0]
                activation_encodings[module_name]['output'][i] = encoding

            if not activation_encodings[module_name]:
                del activation_encodings[module_name]

            for param_name, encoding in module.export_param_encodings().items():
                if not encoding:
                    continue
                param_encodings[f'{module_name}.{param_name}'] = encoding

        return activation_encodings, param_encodings

    def exclude_layers_from_quantization(self, layers_to_exclude: List[torch.nn.Module]):
        """
        Excludes certain layers from being quantized-dequantized by the simulator
        :param layers_to_exclude: List of torch layers to exclude
        :return: None
        """
        # Save the excluded layer names. Do not save the modules since the wrapper removal depends on
        # reference count to automatically remove the layers.
        module_to_name_dict = utils.get_module_to_name_dict(self.model)
        quant_layers_to_exclude = []
        quant_cls = (QcQuantizeWrapper,
                     QcQuantizeRecurrent,
                     LazyQuantizeWrapper,
                     BaseQuantizationMixin)
        for layer in layers_to_exclude:
            for module in layer.modules():
                if isinstance(module, quant_cls):
                    quant_layers_to_exclude.append(module)
                    excluded_module_name = module_to_name_dict.get(module)
                    self._excluded_layer_names.append(excluded_module_name)

        self._remove_quantization_wrappers(self.model, quant_layers_to_exclude)

    def exclude_param_from_quantization(self, param_name_to_exclude: str):
        """
        Excludes all parameters matching 'param_name' from quantization
        :param param_name_to_exclude: Name of the parameter to exclude
        :return: None
        """
        for module in self.model.modules():
            if isinstance(module, (QcQuantizeWrapper, QcQuantizeRecurrent, LazyQuantizeWrapper)):
                if param_name_to_exclude in module.param_quantizers:
                    module.param_quantizers[param_name_to_exclude].enabled = False
            if isinstance(module, BaseQuantizationMixin):
                if param_name_to_exclude in module.param_quantizers:
                    module.param_quantizers[param_name_to_exclude] = None

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

            elif isinstance(module_ref, QcQuantizeRecurrent):
                # Set Recurrent layer for training mode
                module_ref.construct_and_initialize_trainable_quantizers(self._quant_scheme)

            # Recursively call children modules if present
            if not utils.is_leaf_module(module_ref):
                self._replace_quantization_wrapper(module_ref, device)

    def _construct_and_initialize_trainable_wrapper(self, post_training_module: StaticGridQuantWrapper,
                                                    device: torch.device) -> LearnedGridQuantWrapper:
        """
        Copies following tensor quantizer attributes from StaticGridQuantWrapper to LearnedGridQuantWrapper
        to avoid any mismatch.
            - enabled
            - bitwidth
            - encoding
            - use_symmetric_encodings
            - use_strict_symmetric
            - use_unsigned_symmetric

        :param post_training_module: StaticGridQuantWrapper wrapped module
        :param device: device on which model is present
        :return: trainable_module: QcTrainable wrapper module
        """

        # pylint: disable=protected-access
        module = post_training_module._module_to_wrap

        num_inputs = len(post_training_module.input_quantizers)
        num_outputs = len(post_training_module.output_quantizers)

        # Creating a LearnedGridQuantWrapper module
        trainable_module = LearnedGridQuantWrapper(module, self._default_param_bw,
                                                   self._default_output_bw, self._rounding_mode, self._quant_scheme,
                                                   device=device, num_inputs=num_inputs, num_outputs=num_outputs,
                                                   data_type=QuantizationDataType.int)
        # Copy user settable attributes for outputs
        for index, quantizer in enumerate(post_training_module.output_quantizers):
            initialize_learned_grid_quantizer_attributes(trainable_module.output_quantizers[index], quantizer)
            if trainable_module.output_quantizers[index].encoding_min_max_fixed_vals is not None:
                trainable_module.output_quantizers[index].freeze_encoding()
        # Copy user settable attributes for inputs
        for index, quantizer in enumerate(post_training_module.input_quantizers):
            initialize_learned_grid_quantizer_attributes(trainable_module.input_quantizers[index], quantizer)
            if trainable_module.input_quantizers[index].encoding_min_max_fixed_vals is not None:
                trainable_module.input_quantizers[index].freeze_encoding()
        # Copy user settable attributes for params
        for name, quantizer in post_training_module.param_quantizers.items():
            learned_grid_quantizer = trainable_module.param_quantizers[name]
            initialize_learned_grid_quantizer_attributes(learned_grid_quantizer, quantizer)
            if learned_grid_quantizer.encoding_min_max_fixed_vals is not None:
                learned_grid_quantizer.freeze_encoding()

        return trainable_module

    def replace_wrappers_for_quantize_dequantize(self):
        """
        Replaces StaticGridWrapper with LearnedGridWrapper
        """
        if self._quant_scheme == QuantScheme.training_range_learning_with_tf_init or self._quant_scheme == \
                QuantScheme.training_range_learning_with_tf_enhanced_init:
            try:
                device = utils.get_device(self.model)
            except StopIteration:
                # Model doesn't have any parameter.
                # Set device to cpu by default.
                device = torch.device('cpu')

            self._replace_quantization_wrapper(self.model, device)

    def _override_quant_config_for_transformer_layers(self):
        """
        Looks for specific ops in a transformer and overrides the quantizer to tf mode
        """
        # pylint: disable=protected-access
        attention_with_mask_add_quantizer_dict = transformer_utils.get_attention_with_mask_add_quantizer_dict(self.model)

        for attention_head, (mask_add_quantizer_wrapper, mask_add_name) in attention_with_mask_add_quantizer_dict.items():

            assert isinstance(mask_add_quantizer_wrapper, (StaticGridQuantWrapper, LazyQuantizeWrapper))

            # clamping needs to be done only if data type is int
            if mask_add_quantizer_wrapper.output_quantizers and \
                    mask_add_quantizer_wrapper.output_quantizers[0].data_type == QuantizationDataType.int:

                module_to_quantize = mask_add_quantizer_wrapper._module_to_wrap

                quantizer_wrapper_type = qc_quantize_modules_dict.get(type(module_to_quantize), LazyQuantizeWrapper)

                # Add a quantizer set to tf mode and bw to 16 and copy over remaining attributes
                # we need 16 bit to retain the max representation for this quantizer.
                quantized_module = quantizer_wrapper_type(module_to_quantize, 16, 16,
                                                          MAP_PYMO_TO_ROUND_MODE[mask_add_quantizer_wrapper.output_quantizers[0].round_mode],
                                                          QuantScheme.post_training_tf,
                                                          num_inputs=len(mask_add_quantizer_wrapper.input_quantizers),
                                                          num_outputs=len(mask_add_quantizer_wrapper.output_quantizers),
                                                          data_type=mask_add_quantizer_wrapper.output_quantizers[0].data_type)

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
            if mask_add_quantizer_wrapper.output_quantizers and mask_add_quantizer_wrapper.output_quantizers[0].enabled \
                    and mask_add_quantizer_wrapper.output_quantizers[0].data_type == QuantizationDataType.int:
                for output_quantizer in mask_add_quantizer_wrapper.output_quantizers:
                    # get the min/max from accumulated stats associated with this quantizer
                    if output_quantizer.is_encoding_frozen:
                        continue
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
                    output_quantizer.freeze_encoding()
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

            elif succeeding_op.type == CG_SPLIT:
                downstream_modules += QuantizationSimModel._find_next_downstream_modules(succeeding_op)

        return downstream_modules

    @staticmethod
    def _export_encodings_to_files(sim_model: torch.nn.Module, path: str, filename_prefix: str,
                                   op_to_io_tensor_map: Dict, valid_param_set: set, excluded_layer_names,
                                   propagate_encodings: bool, quantizer_args: Dict = None):
        """
        Save the quantized model weight encodings

        :param sim_model: Quantsim model to export encodings for
        :param path: path where to store model pth and encodings
        :param filename_prefix: filename to store exported weight encodings in json format
        :param op_to_io_tensor_map: Dictionary of layer to I/O tensor mapping from onnx or torch script model
        :param valid_param_set: a set of valid param input names in model
        :param excluded_layer_names: List of names of layers that have been excluded from quantization.
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in
                multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of
                ops.
        :param quantizer_args
        """

        # pylint: disable=too-many-locals

        # Create a dictionary to export to JSON
        activation_encodings_onnx = {}
        activation_encodings_torch = {}
        param_encodings = {}
        layers_to_onnx_op_names = QuantizationSimModel._get_layers_in_io_tensor_map(op_to_io_tensor_map)
        tensor_to_consumer_map = QuantizationSimModel._get_tensor_to_consumer_map(op_to_io_tensor_map)
        layer_names_not_found = []

        for layer_name, layer in sim_model.named_modules():
            if not isinstance(layer, (ExportableQuantModule, QcQuantizeRecurrent)):
                continue
            if not has_valid_encodings(layer):
                continue
            # TODO: specifically call out dropout layers here since they are specifically switched out during export.
            # These ops should eventually be reworked as part of math invariant ops to ignore quantization altogether.
            # pylint: disable=protected-access
            if isinstance(layer, ExportableQuantModule) and isinstance(layer.get_original_module(), utils.DROPOUT_TYPES):
                continue

            if layer_name not in layers_to_onnx_op_names.keys():
                layer_names_not_found.append(layer_name)
            else:
                QuantizationSimModel._update_encoding_dicts_for_layer(layer, layer_name, activation_encodings_onnx,
                                                                      activation_encodings_torch,
                                                                      param_encodings, op_to_io_tensor_map,
                                                                      valid_param_set, propagate_encodings,
                                                                      tensor_to_consumer_map, layers_to_onnx_op_names)

        if layer_names_not_found:
            logger.warning("The following layers were not found in the exported onnx model. Encodings for these layers"
                           " will not appear in the exported encodings file:\n"
                           "%s\n"
                           "This can be due to several reasons:\n"
                           "\t- The layer is set to quantize with float datatype, but was not exercised in compute "
                           "encodings. Not an issue if the layer is not meant to be run.\n"
                           "\t- The layer has valid encodings but was not seen while exporting to onnx using the dummy "
                           "input provided in sim.export(). Ensure that the dummy input covers all layers.",
                           layer_names_not_found)
        encodings_dict_onnx = {'version': encoding_version,
                               'activation_encodings': activation_encodings_onnx,
                               'param_encodings': param_encodings,
                               'excluded_layers': excluded_layer_names}

        encodings_dict_pytorch = {'version': encoding_version,
                                  'activation_encodings': activation_encodings_torch,
                                  'param_encodings': param_encodings,
                                  'excluded_layers': excluded_layer_names}

        if quantizer_args:
            encodings_dict_pytorch.update({'quantizer_args': quantizer_args})
            encodings_dict_onnx.update({'quantizer_args': quantizer_args})

        logger.info("Layers excluded from quantization: %s", excluded_layer_names)

        # export weight encodings to output json file
        encoding_file_path = os.path.join(path, filename_prefix + '.encodings')
        encoding_file_path_pytorch = os.path.join(path, filename_prefix + '_torch' + '.encodings')
        save_json_yaml(encoding_file_path, encodings_dict_onnx)
        save_json_yaml(encoding_file_path_pytorch, encodings_dict_pytorch)

    @staticmethod
    def _get_tensor_to_consumer_map(op_to_io_tensor_map: Dict[str, Dict]) -> Dict[str, str]:
        """
        Get a dictionary mapping tensor names to names of ops consuming that tensor.

        :param op_to_io_tensor_map: Dictionary mapping op names to IO Tensors
        :return: Dictionary mapping tensor names to names of ops consuming that tensor
        """
        tensor_to_consumer_map = {}
        if version.parse(torch.__version__) >= version.parse("1.13.0") and onnx_utils.EXPORT_TO_ONNX_DIRECT:
            for op_name, io_tensors in op_to_io_tensor_map.items():
                for inp in io_tensors.inputs:
                    if inp not in tensor_to_consumer_map:
                        tensor_to_consumer_map[inp] = [op_name]
                    else:
                        tensor_to_consumer_map[inp].append(op_name)
                for output in io_tensors.outputs:
                    if output not in tensor_to_consumer_map:
                        tensor_to_consumer_map[output] = []
        return tensor_to_consumer_map

    @staticmethod
    def _get_layers_in_io_tensor_map(op_to_io_tensor_map: Dict) -> Dict[str, str]:
        """
        extract root(layer) names of onnx op names in tensor map
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :return: a set containing layer names present in io tensor map.
        """
        layers_to_onnx_op_names = {}
        if version.parse(torch.__version__) < version.parse("1.13.0") or not onnx_utils.EXPORT_TO_ONNX_DIRECT:
            for name in op_to_io_tensor_map.keys():
                modified_name = name
                if modified_name.endswith('.end'):
                    modified_name = modified_name[:-4]
                if name in layers_to_onnx_op_names.keys():
                    layers_to_onnx_op_names[modified_name.split('#')[0]].append(name)
                else:
                    layers_to_onnx_op_names[modified_name.split('#')[0]] = [name]
        else:
            for name in op_to_io_tensor_map.keys():
                pytorch_name = get_pytorch_name_from_onnx_name(name)
                if pytorch_name in layers_to_onnx_op_names.keys():
                    layers_to_onnx_op_names[pytorch_name].append(name)
                else:
                    layers_to_onnx_op_names[pytorch_name] = [name]
        return layers_to_onnx_op_names

    @staticmethod
    def _update_param_encodings_dict_for_layer(layer: ExportableQuantModule, layer_name: str, param_encodings: Dict,
                                               valid_param_set: set):
        """
        :param layer: layer as torch.nn.Module
        :param layer_name : Name of the layer
        :param param_encodings: dictionary of param encodings
        :param valid_param_set: a set of valid param input names in model
        """

        for orig_param_name, param_encoding in layer.export_param_encodings().items():
            param_name = layer_name + '.' + orig_param_name
            if param_encoding is None:
                continue
            elif param_name not in valid_param_set:
                logger.error('Param tensor {%s} not found in valid param set', param_name)
                continue
            param_encodings[param_name] = param_encoding

    @staticmethod
    def _update_encoding_dicts_for_layer(layer: ExportableQuantModule, layer_name: str, activation_encodings_onnx: Dict,
                                         activation_encodings_torch: Dict, param_encodings: Dict,
                                         op_to_io_tensor_map: Dict, valid_param_set: set, propagate_encodings: bool,
                                         tensor_to_consumer_map: Dict[str, str],
                                         layers_to_onnx_op_names: Dict[str, str]):
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
        :param tensor_to_consumer_map: Dictionary mapping tensor names to op names which consume the tensor
        :param layers_to_onnx_op_names: Dictionary mapping PyTorch layer names to names of corresponding ONNX ops
        """

        if isinstance(layer, ExportableQuantModule):

            # --------------------------------------
            # Update encodings for Input activations
            # --------------------------------------
            QuantizationSimModel._update_encoding_dict_for_input_activations(layer, layer_name, op_to_io_tensor_map,
                                                                             activation_encodings_onnx,
                                                                             activation_encodings_torch,
                                                                             layers_to_onnx_op_names)
            # ---------------------------------------
            # Update encodings for output activations
            # ---------------------------------------
            QuantizationSimModel._update_encoding_dict_for_output_activations(layer, layer_name,
                                                                              op_to_io_tensor_map,
                                                                              activation_encodings_onnx,
                                                                              activation_encodings_torch,
                                                                              propagate_encodings,
                                                                              tensor_to_consumer_map,
                                                                              layers_to_onnx_op_names)
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
    def find_op_names_for_layer(layer_name: str, op_to_io_tensor_map: Dict,
                                tensor_to_consumer_map: Optional[Dict[str, str]],
                                layers_to_onnx_op_names: Optional[Dict[str, str]]) -> Tuple[List[str], List[str]]:
        """
        This function returns the last ONNX op and the list of ONNX Ops that were mapped from a PyTorch Op.

        :param layer_name: Name of the PyTorch layer
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :param tensor_to_consumer_map: Dictionary mapping tensor names to op names which consume the tensor
        :param layers_to_onnx_op_names: Dictionary mapping PyTorch layer names to names of corresponding ONNX ops
        :return: tuple(end op names, all op names)
        """
        if version.parse(torch.__version__) < version.parse("1.13.0") or not onnx_utils.EXPORT_TO_ONNX_DIRECT:
            op_names = [key for key in op_to_io_tensor_map if (key.startswith(layer_name) and layer_name+'#' in key)
                        or key == layer_name]
            if len(op_names) == 1:
                return op_names, op_names

            end_op_names = [op_name for op_name in op_names if op_name.endswith('.end')]
            return end_op_names, op_names

        assert tensor_to_consumer_map is not None
        assert layers_to_onnx_op_names is not None
        # Get all ops which correspond to the current PyTorch layer being processed.
        op_names = layers_to_onnx_op_names.get(layer_name, [])
        op_name_set = set(op_names)

        end_op_names = []
        end_op_names_set = set()
        for op_name in op_names:
            # Loop through outputs of each op and check whether the output leads to an op not in
            for output in op_to_io_tensor_map[op_name].outputs:
                assert output in tensor_to_consumer_map.keys()
                if not tensor_to_consumer_map[output]:
                    if op_name not in end_op_names_set:
                        # output has no consumers, and can either be a model output or an unused op output.
                        # List it as an end_op_name all the same.
                        end_op_names.append(op_name)
                        end_op_names_set.add(op_name)
                else:
                    for consumer in tensor_to_consumer_map[output]:
                        if consumer not in op_name_set and op_name not in end_op_names_set:
                            end_op_names.append(op_name)
                            end_op_names_set.add(op_name)

        return end_op_names, op_names

    @staticmethod
    def _update_encoding_dict_for_output_activations(layer: ExportableQuantModule, layer_name: str, op_to_io_tensor_map: Dict,
                                                     activation_encodings_onnx: Dict, activation_encodings_torch: Dict,
                                                     propagate_encodings: bool, tensor_to_consumer_map: Dict[str, str],
                                                     layers_to_onnx_op_names: Dict[str, str]):
        # pylint: disable=too-many-locals
        output_tensors, propagate_tensors = QuantizationSimModel._get_layer_activation_tensors(layer_name,
                                                                                               op_to_io_tensor_map,
                                                                                               tensor_to_consumer_map,
                                                                                               layers_to_onnx_op_names)
        output_encodings = layer.export_output_encodings()

        if len(output_tensors) != len(output_encodings):
            logger.warning("number of output quantizers: %d available for layer: %s "
                           "doesn't match with number of output tensors: %d", len(output_encodings), layer_name,
                           len(output_tensors))

        for index, (output_tensor, encoding) in enumerate(zip(output_tensors, output_encodings)):

            if encoding is not None:
                activation_encodings_onnx[output_tensor] = encoding
                if layer_name not in activation_encodings_torch:
                    activation_encodings_torch[layer_name] = {}
                if QUANTIZER_TYPE_OUTPUT not in activation_encodings_torch[layer_name]:
                    activation_encodings_torch[layer_name][QUANTIZER_TYPE_OUTPUT] = {}
                activation_encodings_torch[layer_name][QUANTIZER_TYPE_OUTPUT][index] = encoding[0]

        if propagate_encodings:
            valid_encodings = [enc for enc in output_encodings if enc is not None]
            if valid_encodings:
                encoding = valid_encodings[0]
                for activation_tensor in propagate_tensors:
                    activation_encodings_onnx[activation_tensor] = utils.get_propagated_encoding_dict(encoding)


    @staticmethod
    def _update_encoding_dict_for_input_activations(layer: ExportableQuantModule, layer_name: str, op_to_io_tensor_map: Dict,
                                                    activation_encodings_onnx: Dict, activation_encodings_torch: Dict,
                                                    layers_to_onnx_op_names: Dict[str, str]):
        input_encodings = layer.export_input_encodings()
        # skip layer if it has no input encodings.
        if all(encoding is None for encoding in input_encodings):
            return

        input_tensors = QuantizationSimModel._get_layer_input_tensors(layer, layer_name, op_to_io_tensor_map,
                                                                      layers_to_onnx_op_names)

        if len(input_tensors) != len(input_encodings):
            logger.warning("number of input quantizers: %d available for layer: %s "
                           "doesn't match with number of input tensors: %d", len(input_encodings), layer_name,
                           len(input_tensors))

        for index, (input_tensor, encoding) in enumerate(zip(input_tensors, input_encodings)):
            if encoding is not None:
                activation_encodings_onnx[input_tensor] = encoding
                # Check if layer exists in the pytorch encoding dictionary
                if layer_name not in activation_encodings_torch:
                    activation_encodings_torch[layer_name] = {}
                if QUANTIZER_TYPE_INPUT not in activation_encodings_torch[layer_name]:
                    activation_encodings_torch[layer_name][QUANTIZER_TYPE_INPUT] = {}
                # Store encodings for a particular index so that they can be used to check if a quantizer was
                # enabled or not
                activation_encodings_torch[layer_name][QUANTIZER_TYPE_INPUT][index] = encoding[0]

    @staticmethod
    def _get_layer_input_tensors(layer: torch.nn.Module, layer_name: str, op_to_io_tensor_map: Dict,
                                 layers_to_onnx_op_names: Dict[str, str] = None) -> List[str]:
        """
        This function returns the list of input tensor names mapped from a PyTorch Op.

        :param layer: layer as torch.nn.Module
        :param layer_name: Name of the PyTorch layer
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :param layers_to_onnx_op_names: Dictionary mapping PyTorch layer names to names of corresponding ONNX ops
        :return: list of input tensor names.
        """

        param_inputs = [layer_name + '.' + param_name for param_name, _ in layer.named_parameters()]
        if version.parse(torch.__version__) < version.parse("1.13.0") or not onnx_utils.EXPORT_TO_ONNX_DIRECT:
            start_op_names = [key for key in op_to_io_tensor_map
                              if (key.startswith(layer_name) and '#0' in key) or key == layer_name]
        else:
            assert layers_to_onnx_op_names is not None
            op_names = layers_to_onnx_op_names.get(layer_name, [])
            op_name_set = set(op_names)
            start_op_names = set()
            for op_name in op_names:
                # For each op's inputs, if the input comes from an op not associated with this layer, add it to
                # start_op_names.
                for inp in op_to_io_tensor_map[op_name].inputs:
                    if inp not in op_name_set:
                        start_op_names.add(op_name)

        input_tensors = []
        input_tensors_set = set()
        for name in start_op_names:
            for input_tensor in op_to_io_tensor_map[name].inputs:
                if input_tensor not in param_inputs and input_tensor not in input_tensors_set:
                    input_tensors.append(input_tensor)
                    input_tensors_set.add(input_tensor)

        return input_tensors

    @classmethod
    def _get_layer_activation_tensors(cls, layer_name: str, op_to_io_tensor_map: Dict,
                                      tensor_to_consumer_map: Dict[str, str] = None,
                                      layers_to_onnx_op_names: Dict[str, str] = None) -> Tuple[List[str], List[str]]:
        """
        This function returns the list of output tensor and intermediate tensor names mapped from a PyTorch Op.

        :param layer_name: Name of the PyTorch layer
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :param tensor_to_consumer_map: Dictionary mapping tensor names to op names which consume the tensor
        :param layers_to_onnx_op_names: Dictionary mapping PyTorch layer names to names of corresponding ONNX ops
        :return: tuple containing list of output tensor names and list of intermediate tensors
        """
        end_op_names, op_names = cls.find_op_names_for_layer(layer_name, op_to_io_tensor_map, tensor_to_consumer_map,
                                                             layers_to_onnx_op_names)

        if len(end_op_names) > 1:
            output_op_map_str = cls._get_output_map_str(end_op_names, layer_name, op_to_io_tensor_map)
            logger.info("layer_name: %s, has multiple output onnx ops: %s", layer_name, output_op_map_str)

        output_tensors = []
        intermediate_tensors = []
        for name in op_names:
            if name in end_op_names:
                output_tensors.extend(op_to_io_tensor_map[name].outputs)
            else:
                intermediate_tensors.extend(op_to_io_tensor_map[name].outputs)

        return output_tensors, intermediate_tensors

    @staticmethod
    def _get_output_map_str(end_op_names, layer_name, op_to_io_tensor_map) -> str:
        """
        This function returns formatted list of output ops tensor mapping

        :param end_op_names: list of output onnx ops
        :param layer_name: Name of the PyTorch layer
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :return: formatted string with output ops and their corresponding output count.
        """
        num_output_ops = len(end_op_names)
        op_map_str = ','.join([f'{name.replace(layer_name, "")}:{len(op_to_io_tensor_map[name].outputs)}'
                               for name in end_op_names[:5]])
        if num_output_ops > 5:
            op_map_str += ', ..'
        return f'{num_output_ops},[{op_map_str}]'

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
            quantizer_encoding = _get_encoding_by_quantizer(quantizer)
            encoding = QuantizationSimModel._create_encoding_dict(quantizer_encoding, quantizer,
                                                                  propagate_encodings=False)
            activation_encodings_onnx[tensor] = [encoding]

        if propagate_encodings and quantizer:
            _, op_names = QuantizationSimModel.find_op_names_for_layer(layer_name, op_to_io_tensor_map, None, None)
            for op_name in op_names:
                io_tensor_list = op_to_io_tensor_map[op_name]
                if not isinstance(io_tensor_list, list):
                    io_tensor_list = [io_tensor_list]

                for io_tensors in io_tensor_list:

                    if io_tensors.outputs:
                        for output_tensor in io_tensors.outputs:
                            if output_tensor in onnx_activations_to_quantizers:
                                continue
                            quantizer_encoding = _get_encoding_by_quantizer(quantizer)
                            encoding = QuantizationSimModel._create_encoding_dict(quantizer_encoding, quantizer,
                                                                                  True)

                            activation_encodings_onnx[output_tensor] = [encoding]

        # ------------------
        # Params
        # ------------------
        for tensor, quantizer in onnx_params_to_quantizers.items():
            quantizer_encoding = _get_encoding_by_quantizer(quantizer)
            encoding = QuantizationSimModel._create_encoding_dict(quantizer_encoding, quantizer,
                                                                  propagate_encodings=False)
            param_encodings[tensor] = [encoding]

    @staticmethod
    def _get_qc_quantized_layers(model) -> List[Tuple[str, QcQuantizeWrapper]]:
        quantized_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (QcQuantizeWrapper, QcQuantizeRecurrent, LazyQuantizeWrapper, BaseQuantizationMixin)):
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
        quantizer_wrapper_type = qc_quantize_modules_dict.get(type(module_to_quantize), LazyQuantizeWrapper)

        if quantizer_wrapper_type == LazyQuantizeWrapper:
            quant_scheme_for_initialization = self._quant_scheme
        else:
            quant_scheme_for_initialization = utils.get_v1_quant_scheme_for_initialization(self._quant_scheme)

        # TODO add quant_scheme_for_initialization for FP8 case
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

    def _set_tensor_quantizers_for_consts(self):
        """
        Identify and set is_const for tensor quantizers which correspond to constant inputs in the model.
        """

        if self.connected_graph is not None:
            for _, qc_quantize_wrapper in self.quant_wrappers():
                if isinstance(qc_quantize_wrapper, (QcQuantizeWrapper, LazyQuantizeWrapper)):
                    # Only handling QcQuantWrappers and not QcQuantizeRecurrents
                    # pylint: disable=protected-access
                    conn_graph_op = self.connected_graph._module_to_op_dict.get(qc_quantize_wrapper._module_to_wrap)
                    if conn_graph_op is not None:
                        for (input_quantizer, inp) in zip(qc_quantize_wrapper.input_quantizers, conn_graph_op.inputs):
                            input_quantizer.is_const = inp.is_const

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
        return utils.create_encoding_dict(encoding, quantizer, propagate_encodings)

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

                if isinstance(module_ref, ExportableQuantModule):
                    # Remove the wrapper, gets auto-deleted
                    # pylint: disable=protected-access
                    setattr(starting_module, module_name, module_ref.get_original_module())

                elif isinstance(module_ref, QcQuantizeStandAloneBase):
                    setattr(starting_module, module_name, torch.nn.Identity())

                elif isinstance(module_ref, QcQuantizeRecurrent):
                    module_ref.update_params()
                    setattr(starting_module, module_name, module_ref.module_to_quantize)

            # Recursively call children modules if present
            if not utils.is_leaf_module(module_ref):
                cls._remove_quantization_wrappers(module_ref, list_of_modules_to_exclude)

    @staticmethod
    def get_original_model(model: torch.nn.Module):
        """
        This function returns the model with all quantization wrappers removed.
        :return: Model without quantization wrappers.
        """
        original_model = copy.deepcopy(model)
        all_modules_in_original_model = [module for module in original_model.modules()]
        QuantizationSimModel._remove_quantization_wrappers(original_model, all_modules_in_original_model)
        return original_model

    def _get_leaf_module_to_name_map(self):
        """
        Returns a mapping from leaf modules to module name, where any ExportableQuantModule is considered a leaf module,
        and is therefore not further recursed (since we do not want to retrieve all internal quantizers/modules).
        """
        def recursively_populate_map(starting_module, module_map, start_str):
            for name, module in starting_module.named_children():
                if isinstance(module, ExportableQuantModule) or utils.is_leaf_module(module):
                    module_map[module] = start_str + name
                else:
                    recursively_populate_map(module, module_map, start_str + name + ".")
        module_to_name_map = {}
        recursively_populate_map(self.model, module_to_name_map, "")
        return module_to_name_map

    def _add_inputs_hook(self, hooks):
        module_to_name_map = self._get_leaf_module_to_name_map()

        def inputs_hook(module_ref, inputs, _):
            # Need to remove hook here, otherwise the jit trace of CustomMarker with module ref will error since the
            # hook will be recursively hit.
            hooks[module_ref].remove()
            del hooks[module_ref]
            module_name = module_to_name_map[module_ref]
            if isinstance(module_ref, ExportableQuantModule):
                module_ref = module_ref.get_original_module()
            marker_layer = torch.jit.trace(CustomMarker(module_ref, module_name, 'True'),
                                           inputs)
            self._module_marker_map[module_name] = marker_layer

        for name, module in self.model.named_modules():
            if name in module_to_name_map.values():
                hooks[module] = module.register_forward_hook(inputs_hook)

    def _validate_module_marker_map(self):
        """
        Check to make sure all leaf modules have traced Custom Markers associated with them.
        """
        all_leaf_modules = self._get_leaf_module_to_name_map().values()
        missing_inputs_entries = []

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
            error_msg = ('A connected graph failed to be built.\n'
                         'Unable to proceed with automatically configuring quantization ops using the config file.\n'
                         'Please configure quantization ops manually by redefining '
                         'QuantizationSimModel.configure_quantization_ops()')
            logger.error(error_msg)
            raise AssertionError(error_msg)
        return QuantSimConfigurator(self.model, self.connected_graph, config_file, default_output_bw,
                                    default_param_bw, default_data_type)

    def load_and_freeze_encodings(self, encoding_path: str, ignore_when_quantizer_disabled: bool = False):
        """
        Functionality to set encodings (both activation and parameter) as per the given encodings JSON file and
        freeze them.
        .. note:
            The encodings JSON file should be the {prefix}_torch.encodings json exported during sim.export()

        :param encoding_path: JSON file path from where to load the encodings.
        :param ignore_when_quantizer_disabled: ignore raising RuntimeError while setting encodings,
            when quantizers are disabled.
        """
        with open(encoding_path, mode='r') as json_file:
            encodings_dict = json.load(json_file)

        self._set_param_encodings(encodings_dict['param_encodings'],
                                  freeze=True,
                                  ignore_when_quantizer_disabled=ignore_when_quantizer_disabled,
                                  disable_quantizer_without_encoding=False)

        self._set_activation_encodings(encodings_dict['activation_encodings'],
                                       freeze=True,
                                       ignore_when_quantizer_disabled=ignore_when_quantizer_disabled,
                                       disable_quantizer_without_encoding=False)


    def _set_param_encodings(self,
                             encoding_dict: Dict,
                             freeze: bool,
                             ignore_when_quantizer_disabled: bool,
                             disable_quantizer_without_encoding: bool):
        for name, quant_module in self.model.named_modules():
            if isinstance(quant_module, ExportableQuantModule):
                param_encoding = {
                    param_name: encoding_dict[f'{name}.{param_name}']
                    for param_name, _ in quant_module.param_quantizers.items()
                    if f'{name}.{param_name}' in encoding_dict
                }
                quant_module.import_param_encodings(param_encoding,
                                                    freeze=freeze,
                                                    ignore_when_quantizer_disabled=ignore_when_quantizer_disabled,
                                                    disable_quantizer_without_encoding=disable_quantizer_without_encoding)

    def _set_activation_encodings(self,
                                  activation_encoding_dict: dict,
                                  freeze: bool,
                                  ignore_when_quantizer_disabled: bool,
                                  disable_quantizer_without_encoding: bool):
        for module_name, module in self.model.named_modules():
            if not isinstance(module, ExportableQuantModule):
                continue

            try:
                input_encoding = activation_encoding_dict[module_name]['input']
            except KeyError:
                input_encoding = {}

            module.import_input_encodings(input_encoding,
                                          freeze=freeze,
                                          ignore_when_quantizer_disabled=ignore_when_quantizer_disabled,
                                          disable_quantizer_without_encoding=disable_quantizer_without_encoding)

            try:
                output_encoding = activation_encoding_dict[module_name]['output']
            except KeyError:
                output_encoding = {}

            module.import_output_encodings(output_encoding,
                                           freeze=freeze,
                                           ignore_when_quantizer_disabled=ignore_when_quantizer_disabled,
                                           disable_quantizer_without_encoding=disable_quantizer_without_encoding)


    def set_and_freeze_param_encodings(self, encoding_path: str):
        """
        Set and freeze parameter encodings from encodings JSON file.
        .. note:
            The loaded json file should contain ONLY weight encodings. This is different from the json file used in
            `load_and_freeze_encodings`, which contains both weight and activation dictionaries.

        :param encoding_path: path from where to load parameter encodings file
        """
        # Load parameter encodings file
        with open(encoding_path) as json_file:
            param_encodings = json.load(json_file)
            if 'param_encodings' in param_encodings:
                param_encodings = param_encodings['param_encodings']
            else:
                logger.warning("An older AdaRound exported encoding file type has been detected! "
                               "Please regenerate it using the AdaRound export function from the latest "
                               "AIMET (version 1.32 or higher) if necessary. "
                               "Support for this encoding file will be deprecated in AIMET version 1.33.0.")

        self._set_param_encodings(param_encodings,
                                  freeze=True,
                                  ignore_when_quantizer_disabled=False,
                                  disable_quantizer_without_encoding=True)

    def quant_wrappers(self):
        """
        Generator for yielding all quantization wrappers and their names
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (QcQuantizeWrapper, QcQuantizeRecurrent, LazyQuantizeWrapper, BaseQuantizationMixin)):
                yield name, module

    def run_modules_for_traced_custom_marker(self, module_list: List[torch.nn.Module], dummy_input):
        """
        Given a list of modules to run and dummy input for the module, create a traced CustomMarker for each module
        and store it in the module_marker map. The same dummy input will be used for all modules.

        :param module_list: List of modules to create traced CustomMarkers for
        :param dummy_input: Dummy input for all modules
        """

        module_to_name_map = self._get_leaf_module_to_name_map()

        for module in module_list:
            # Only perform init and trace if the given module is a leaf module, and we have not recorded it before
            if module in module_to_name_map and module_to_name_map[module] not in self._module_marker_map:
                name = module_to_name_map[module]
                module = module.get_original_module() if isinstance(module, ExportableQuantModule) else module
                with utils.in_eval_mode(module), torch.no_grad():
                    marker_layer = torch.jit.trace(CustomMarker(module, name, True), dummy_input)
                    self._module_marker_map[name] = marker_layer

    def _validate_supported_kernels_for_quantizers(self, action: SupportedKernelsAction):
        """
        Validate supported kernels for all the Quantizers in the QuantSimModel
        :param action: The action to be performed when incorrect candidate is set in a quantizer
        """

        def apply_act_param_rules(curr_candidate: QuantDtypeBwInfo, allowed_supported_kernels: List[QuantDtypeBwInfo], module_name):
            """
            helper function to validate both activation and param against the supported_kernels passed
            :param curr_candidate: candidate of interest
            :param allowed_supported_kernels: List of supported kernels for the given module
            :param module_name: name of the module
            """
            if action != SupportedKernelsAction.allow_error:
                for k in allowed_supported_kernels:
                    if curr_candidate == k:
                        return

                if action == SupportedKernelsAction.warn_on_error:
                    logger.warning("candidate:%s is not under the supported_kernels for the module %s", curr_candidate,
                                   module_name)

                if action == SupportedKernelsAction.assert_on_error:
                    error_msg = f'candidate: {curr_candidate} is not under the supported_kernels for the module {module_name}'
                    raise RuntimeError(error_msg)

        def apply_act_rules(act: Tuple[int, QuantizationDataType], allowed_supported_kernels: List[QuantDtypeBwInfo], module_name):
            """
            helper function to validate both activation only against the supported_kernels passed
            :param act: act of the candidate to be validated
            :param allowed_supported_kernels: List of supported kernels for the given module
            :param module_name: name of the module
            """
            if action != SupportedKernelsAction.allow_error:
                for k in allowed_supported_kernels:
                    if k.is_same_activation(act[1], act[0]):
                        return

                if action == SupportedKernelsAction.warn_on_error:
                    logger.warning("activation:%s is not under the supported_kernels for the module %s", act, module_name)

                if action == SupportedKernelsAction.assert_on_error:
                    error_msg = f'activation: {act} is not under the supported_kernels for the module {module_name}'
                    raise RuntimeError(error_msg)

        # retrieve all the act and param quantizer candidates, and validate them against supported_kernels
        for name, module in self.model.named_modules():
            if isinstance(module, (QcQuantizeWrapper, LazyQuantizeWrapper)) and module.supported_kernels:
                supported_kernels = []
                for supported_kernel in module.supported_kernels:
                    # ((activation bitwidth, activation data type), (param bitwidth, param data type))
                    # TODO modify this once reformat_supported_kernels generates of type QuantDtypeBwInfo
                    if isinstance(supported_kernel[1], tuple):
                        supported_kernels.append(
                            QuantDtypeBwInfo(supported_kernel[0][1], supported_kernel[0][0],
                                             supported_kernel[1][1], supported_kernel[1][0]))
                    else:
                        supported_kernels.append(
                            QuantDtypeBwInfo(supported_kernel[1], supported_kernel[0]))
                act_candidates = []
                param_candidate = ()
                for quantizer in module.input_quantizers + module.output_quantizers:
                    act_candidates.append((quantizer.bitwidth, quantizer.data_type))

                if 'weight' in module.param_quantizers:
                    param_candidate = (module.param_quantizers['weight'].bitwidth,
                                       module.param_quantizers['weight'].data_type)

                if param_candidate:
                    # we need to check weights against all the activations
                    for act_candidate in set(act_candidates):
                        apply_act_param_rules(QuantDtypeBwInfo(act_candidate[1], act_candidate[0], param_candidate[1],
                                                               param_candidate[0]), supported_kernels, name)
                else:
                    for candidate in set(act_candidates):
                        apply_act_rules(candidate, supported_kernels, name)

    @staticmethod
    def _replace_quantization_wrapper_with_native_torch_quantization_nodes(quant_sim_model, device: torch.device):
        """
        Recursively remove quantization wrappers from all appropriate modules starting with a given module
        :param quant_sim_model: model for which QcQuantizeWrapper gets replaced with wrapped module using
        native torch quantization nodes
        :param device: device on which model is present
        :return:
        """
        # Recursively replace quantization wrappers to native torch quantization nodes
        for module_name, module_ref in quant_sim_model.named_children():
            # Create a native torch quantization node
            if isinstance(module_ref, QcQuantizeWrapper):
                embedded_module = NativeTorchQuantWrapper(module_ref, '_module_to_wrap', device)
                setattr(quant_sim_model, module_name, embedded_module)

            elif isinstance(module_ref, QcQuantizeRecurrent):
                logger.error('Do not support save model embedded native torch quantization nodes using QcQuantizeRecurrent.')
                raise AssertionError

            # Recursively call children modules if present
            if not utils.is_leaf_module(module_ref):
                QuantizationSimModel._replace_quantization_wrapper_with_native_torch_quantization_nodes(module_ref, device)

    @staticmethod
    def save_model_with_embedded_quantization_nodes(sim_model, path: str, filename_prefix: str, dummy_input: Union[torch.Tensor, Tuple],
                                                    onnx_export_args: Optional[Union[OnnxExportApiArgs, Dict]] = None,
                                                    export_to_torchscript: bool = False, is_conditional: bool = False):
        """
        Export model embedded with native torch quantization nodes. These nodes will be exported
        as default onnx or torch script quantized nodes.
        :param sim_model: model with the quantsim wrappers
        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param dummy_input: Dummy input to the model. Used to parse model graph
        :param onnx_export_args: optional export argument with onnx specific overrides if not provide export via
                torchscript graph. Int16 can only be exported by torchscript
        :param export_to_torchscript: If True, export to torchscript. Export to onnx otherwise. Defaults to False.
        :param is_conditional: True if model is conditional, False otherwise
        :return:
        """
        def _validate_torchquantizer(quant_sim_model):
            # To avoid non 8 bit TorchQuantizer are exported to ONNX
            for _, module in quant_sim_model.named_modules():
                if isinstance(module, NativeTorchQuantWrapper):
                    quantizers = module.input_quantizers + module.output_quantizers
                    if 'weight' in module.param_quantizers:
                        quantizers += [module.param_quantizers['weight']]
                    if 'bias' in module.param_quantizers:
                        quantizers += [module.param_quantizers['bias']]

                    for quantizer in quantizers:
                        if quantizer.enabled and quantizer.data_type == QuantizationDataType.int and quantizer.bitwidth != 8:
                            raise ValueError('Only 8 bit quantizers are supported by exporting to ONNX model.'
                                             'Please enable export_to_torchscript if you want to export non 8 bit quantizers.')

        model_filename = filename_prefix + '_embedded' + '.onnx'
        model_path = os.path.join(path, model_filename)
        quant_sim_model = copy.deepcopy(sim_model)

        device = utils.get_device(quant_sim_model)
        if isinstance(dummy_input, torch.Tensor):
            dummy_input = dummy_input.to(device)
        else:
            dummy_input = tuple([input.to(device) for input in dummy_input])
        QuantizationSimModel._replace_quantization_wrapper_with_native_torch_quantization_nodes(quant_sim_model, device)

        if export_to_torchscript:
            with utils.in_eval_mode(quant_sim_model), torch.no_grad():
                trace = torch.jit.trace(quant_sim_model, dummy_input)
                ts_path = os.path.join(path, filename_prefix + '_embedded' + '.torchscript.pth')
                trace.save(ts_path)
        else:
            _validate_torchquantizer(quant_sim_model)
            OnnxSaver._export_model_to_onnx(quant_sim_model, dummy_input, model_path, is_conditional, onnx_export_args) # pylint: disable=protected-access

    def _enable_output_quantizers_for_specific_cast_ops(self, inout_tensors_dtypes: Dict[torch.nn.Module, Tuple[torch.dtype, torch.dtype]]):
        """
        Enable output quantizer for Cast Ops where datatype of input tensor is int/bool
        and data type of output tensor is float.
        """
        # pylint: disable=protected-access
        model_prefix = self.connected_graph._model_name + '.'
        torch_int_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.bool, torch.uint8}
        torch_float_dtypes = {torch.float16, torch.float32, torch.float64}

        for module, inout_dtypes in inout_tensors_dtypes.items():
            input_tensor_dtype = inout_dtypes[0]
            output_tensor_dtype = inout_dtypes[1]
            # pylint: disable=protected-access
            module_name = self.connected_graph._module_to_name[module].split(model_prefix)[-1]

            if input_tensor_dtype != output_tensor_dtype and input_tensor_dtype in torch_int_dtypes and output_tensor_dtype in torch_float_dtypes:
                logger.info("Enabling output quantizer for module %s", module_name)
                wrapped_module = getattr(self.model, module_name)
                for output_quantizer in wrapped_module.output_quantizers:
                    setattr(output_quantizer, 'enabled', True)


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
    Loads the saved encodings to quant sim model. The encoding filename to load should end in _torch.encodings,
    generated as part of quantsim export.

    :param quant_sim_model: Quantized model to load encodings for. Note: The model configuration should be the same as
        when encodings were exported.
    :param pytorch_encoding_path: Path of the encodings file to load.
    """
    # pylint: disable=too-many-locals, too-many-branches
    # Load encodings file
    with open(pytorch_encoding_path) as json_file:
        encodings = json.load(json_file)

    param_encodings = encodings['param_encodings']
    activation_encodings = encodings['activation_encodings']

    for module_name, module in quant_sim_model.model.named_modules():
        if not isinstance(module, ExportableQuantModule):
            continue

        if isinstance(module, QcQuantizeWrapper):
            module.set_mode(QcQuantizeOpMode.ACTIVE)

        param_encoding = {
            param_name: param_encodings[f'{module_name}.{param_name}']
            for param_name, _ in module.param_quantizers.items()
            if f'{module_name}.{param_name}' in param_encodings
        }
        module.import_param_encodings(param_encoding)

        try:
            input_encoding = activation_encodings[module_name]['input']
        except KeyError:
            input_encoding = {}

        module.import_input_encodings(input_encoding)

        try:
            output_encoding = activation_encodings[module_name]['output']
        except KeyError:
            output_encoding = {}

        module.import_output_encodings(output_encoding)

    def is_enabled(quantizer):
        if quantizer is None:
            return False
        if hasattr(quantizer, 'enabled'):
            return quantizer.enabled
        return True

    def is_initialized(quantizer):
        if hasattr(quantizer, 'is_initialized'):
            return quantizer.is_initialized()
        return quantizer.encoding is not None

    for name, layer in quant_sim_model.quant_wrappers():
        if isinstance(layer, ExportableQuantModule):
            input_quantizers = layer.input_quantizers
            output_quantizers = layer.output_quantizers
        else:
            input_quantizers = list(layer.input_quantizers.values())
            output_quantizers = list(layer.output_quantizers.values())

        for idx, quantizer in enumerate(input_quantizers):
            if is_enabled(quantizer) and not is_initialized(quantizer):
                logger.debug('No encoding loaded for input quantizer %s of layer %s', idx, name)

        for idx, (param_name, quantizer) in enumerate(layer.param_quantizers.items()):
            if is_enabled(quantizer) and not is_initialized(quantizer):
                logger.debug('No encoding loaded for param quantizer %s of layer %s', param_name, name)
        for idx, quantizer in enumerate(output_quantizers):
            if is_enabled(quantizer) and not is_initialized(quantizer):
                logger.debug('No encoding loaded for output quantizer %s of layer %s', idx, name)

    if isinstance(quant_sim_model, QuantizationSimModel):
        # Only for V1 quantsim
        quant_sim_model.replace_wrappers_for_quantize_dequantize()


def has_valid_encodings(qc_quantize_op: ExportableQuantModule) -> bool:
    """
    Utility for determining whether a given qc_quantize_op has any valid encodings.

    :param qc_quantize_op: Qc quantize op to evaluate
    :return: True if any input, param, or output quantizers have valid encodings, False otherwise
    """
    if not isinstance(qc_quantize_op, (ExportableQuantModule, QcQuantizeRecurrent)):
        logger.error("has_valid_encodings only supported for QcQuantizeWrapper and QcQuantizeRecurrent "
                     "modules")
        assert isinstance(qc_quantize_op, (ExportableQuantModule, QcQuantizeRecurrent))
    if isinstance(qc_quantize_op, ExportableQuantModule):
        all_encodings = qc_quantize_op.export_output_encodings() + qc_quantize_op.export_input_encodings() + \
                        list(qc_quantize_op.export_param_encodings().values())
        return any([encoding is not None for encoding in all_encodings])
    input_quantizers = list(qc_quantize_op.input_quantizers.values())
    output_quantizers = list(qc_quantize_op.output_quantizers.values())

    for quantizer in input_quantizers + output_quantizers + list(qc_quantize_op.param_quantizers.values()):
        if quantizer.enabled and (quantizer.encoding is not None or quantizer.data_type is QuantizationDataType.float):
            return True

    return False


def compute_encodings_for_sims(sim_list: List[QuantizationSimModel], forward_pass_callback: Callable,
                               forward_pass_callback_args: Any):
    """
    Compute encodings for a list of QuantSims.

    :param sim_list: List of QuantSims to compute encodings for.
    :param forward_pass_callback: A callback function that simply runs forward passes on the models. This callback
        function should use representative data for the forward pass, so the calculated encodings work for all
        data samples. This callback internally chooses the number of data samples it wants to use for calculating
        encodings.
        The callback expects exactly two inputs:
            - List of models which are involved in the forward pass. The models are taken directly from calling
            sim.model for each sim in sim_list, passed in the same order in which the sims appear in sim_list.
            - Forward pass callback args
    :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
        the user to determine the type of this parameter. E.g. could be simply an integer representing the number
        of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
        If set to None, forward_pass_callback will be invoked with no parameters.
    """
    ctx_managers = [torch.no_grad()]
    for sim in sim_list:
        ctx_managers.append(utils.in_eval_mode(sim.model))
        QuantizationSimModel.prepare_sim_for_compute_encodings(sim)

    with contextlib.ExitStack() as stack:
        for mgr in ctx_managers:
            stack.enter_context(mgr)
        _ = forward_pass_callback([sim.model for sim in sim_list], forward_pass_callback_args)

    for sim in sim_list:
        QuantizationSimModel.compute_layer_encodings_for_sim(sim)
