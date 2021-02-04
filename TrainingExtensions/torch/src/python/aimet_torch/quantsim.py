# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
import os
import io
import copy
import pickle
from typing import Tuple, List, Union, Dict
import json
import torch
import onnx

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_common.quantsim import calculate_delta_offset
from aimet_torch.quantsim_config.quantsim_config import QuantSimConfigurator
from aimet_torch.qc_quantize_op import QcQuantizeStandAloneBase, QcQuantizeWrapper, QcQuantizeOpMode, \
    QcPostTrainingWrapper
from aimet_torch import torchscript_utils
from aimet_torch.tensor_quantizer import TensorQuantizer
from aimet_torch.batch_norm_fold import PassThroughOp
from aimet_torch import utils
from aimet_torch import onnx_utils
from aimet_torch.meta.connectedgraph_utils import create_connected_graph_with_input_shapes
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.qc_quantize_recurrent import QcQuantizeRecurrent
import libpymo

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


# Types of modules which cannot be quantized
unquantizable_modules = (QcQuantizeWrapper, QcQuantizeStandAloneBase, QcQuantizeRecurrent, PassThroughOp,
                         torch.nn.Identity)

# If a torch module type is in this dictionary, call the corresponding quantized module constructor instead of wrapping
# it with QcQuantizeWrapper.
qc_quantize_modules_dict = {
    torch.nn.RNN: QcQuantizeRecurrent,
    torch.nn.LSTM: QcQuantizeRecurrent,
    torch.nn.GRU: QcQuantizeRecurrent
}


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

    # pylint: disable=too-many-arguments
    def __init__(self, model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]] = None,
                 quant_scheme: Union[str, QuantScheme] = QuantScheme.post_training_tf_enhanced,
                 rounding_mode: str = 'nearest', default_output_bw: int = 8, default_param_bw: int = 8,
                 in_place: bool = False, config_file: str = None,
                 dummy_input: Union[torch.Tensor, Tuple] = None):
        """
        Constructor

        :param model: Model to add simulation ops to
        :param input_shapes: List of input shapes to the model
        :param quant_scheme: Quantization scheme. Supported options are 'tf_enhanced' or 'tf' or using Quant Scheme Enum
                             QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
        :param rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters
        :param in_place: If True, then the given 'model' is modified in-place to add quant-sim nodes.
                Only suggested use of this option is when the user wants to avoid creating a copy of the model
        :param config_file: Path to Configuration file for model quantizers
        :param dummy_input: Dummy input to the model. Used to parse model graph. If the model has more than one input,
                            pass a tuple. User is expected to place the tensors on the appropriate device.
        """
        # Perform sanity checks on inputs
        QuantizationSimModel._validate_quantsim_inputs(quant_scheme, rounding_mode, default_output_bw, default_param_bw)
        # save some parameters
        if in_place:
            self.model = model
        else:
            self.model = copy.deepcopy(model)

        try:
            if dummy_input is not None:
                connected_graph = ConnectedGraph(self.model, dummy_input)
            else:
                if input_shapes is None:
                    raise AssertionError('Must provide either input shapes or a dummy input for export')
                connected_graph = create_connected_graph_with_input_shapes(self.model, input_shapes)

        except (torch.jit.TracingCheckError, AssertionError):
            connected_graph = None

        if isinstance(quant_scheme, str):
            if quant_scheme == 'tf':
                quant_scheme = QuantScheme.post_training_tf
            elif quant_scheme == 'tf_enhanced':
                quant_scheme = QuantScheme.post_training_tf_enhanced
        self._quant_scheme = quant_scheme
        self._rounding_mode = rounding_mode
        self._default_output_bw = default_output_bw
        self._default_param_bw = default_param_bw

        # Add quantization layers
        self._add_quantization_wrappers(self.model)

        # Disable bias quantization
        self.exclude_param_from_quantization("bias")

        self.configure_quantization_ops(connected_graph, config_file)

    def __str__(self):
        """
        Pretty-printed output indicating where in the model, quantizers have been activated
        :return:
        """
        stream = io.StringIO(newline='\n')
        stream.write("-------------------------\n")
        stream.write("Quantized Model Report\n")
        stream.write("-------------------------\n")

        wrappers = [(name, module) for name, module in self.model.named_modules()
                    if isinstance(module, QcQuantizeWrapper)]

        for name, wrapper in wrappers:
            stream.write('Layer: {}\n'.format(name))

            # Inputs
            if wrapper.input_quantizer.enabled:
                stream.write('    Input: bw={}, encoding-present={}\n'.format(wrapper.input_quantizer.bitwidth,
                                                                              bool(wrapper.input_quantizer.encoding)))
            else:
                stream.write('    Input: Unquantized\n')

            # Params
            stream.write('    Params:\n')
            for param_name, quantizer in wrapper.param_quantizers.items():
                if quantizer.enabled:
                    stream.write('        {}: bw={}, encoding-present={}\n'.format(param_name,
                                                                                   quantizer.bitwidth,
                                                                                   bool(quantizer.encoding)))
                else:
                    stream.write('        {}: Unquantized\n'.format(param_name))

            # Outputs
            if wrapper.output_quantizers[0].enabled:
                stream.write('    Output: bw={}, encoding-present={}\n'.format(wrapper.output_quantizers[0].bitwidth,
                                                                               bool(wrapper.output_quantizers[0].encoding)))
            else:
                stream.write('    Output: Unquantized\n')

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

        # Run forward iterations so we can collect statistics to compute the appropriate encodings
        self.model.eval()
        with torch.no_grad():
            _ = forward_pass_callback(self.model, forward_pass_callback_args)

        layers_with_invalid_encodings = []
        # Get the computed per-layer encodings and log them
        for name, layer in quantized_layers:
            layer.compute_encoding()

            # Before we return we set the mode to active - meaning ready for quantize/de-quantize
            # for layers with valid_encoding, otherwise we set to pass through
            if isinstance(layer, QcQuantizeRecurrent):
                has_invalid_encoding = self.set_mode_for_recurrent_module(layer, name)
                if has_invalid_encoding is True:
                    layers_with_invalid_encodings.append(name)

            else:
                # By default we want to set the Quantization wrappers to ACTIVE mode
                layer.set_mode(QcQuantizeOpMode.ACTIVE)

                # Unless we have invalid encodings
                if (layer.output_quantizers[0].enabled and not layer.output_quantizers[0].encoding) or \
                        (layer.input_quantizer.enabled and not layer.input_quantizer.encoding):
                    layers_with_invalid_encodings.append(name)
                    layer.set_mode(QcQuantizeOpMode.PASSTHROUGH)

        if layers_with_invalid_encodings:
            logger.info('The following modules (name, input|output quantizer) did not have valid encodings and have '
                        'been set to passThrough mode: %s', layers_with_invalid_encodings)
            logger.info('This can be due to the quantizers not having been evaluated during the forward pass in '
                        'compute encodings. Evaluation is required to collect statistics needed to compute valid '
                        'encodings.\n'
                        'As a result, the quantizers have been set to passThrough mode, meaning no quantization '
                        'noise will be simulated for these modules if they are evaluated in the future.\n'
                        'If this is not desired, amend the forward pass to evaluate these modules, and recompute '
                        'encodings.')

        self._replace_wrappers_for_quantize_dequantize()

    @classmethod
    def set_mode_for_recurrent_module(cls, layer: QcQuantizeRecurrent, name: str):
        """
        Sets Recurrent module to active or pass through mode based on quantizer state

        :param layer:  Qc Quantizer layer for recurrent module
        :param name:  layer name
        :return: True if the encoding is invalid

        """
        is_quantizer_enabled = False
        has_invalid_encoding = False
        for quantizer_name, output_quantizer in layer.output_quantizers.items():
            if output_quantizer.enabled:
                if output_quantizer.encoding:
                    encoding = output_quantizer.encoding
                    logger.debug("Encoding for %s-%s: min=%f, max=%f, offset=%f. delta=%f, bw=%f",
                                 name, quantizer_name, encoding.min, encoding.max,
                                 encoding.delta, encoding.offset, encoding.bw)
                    is_quantizer_enabled = True
                else:
                    has_invalid_encoding = True
        for quantizer_name, input_quantizer in layer.input_quantizers.items():
            if input_quantizer.enabled:
                if input_quantizer.encoding:
                    encoding = input_quantizer.encoding
                    logger.debug("Encoding for %s-%s: min=%f, max=%f, offset=%f. delta=%f, bw=%f",
                                 name, quantizer_name, encoding.min, encoding.max,
                                 encoding.delta, encoding.offset, encoding.bw)
                    is_quantizer_enabled = True
                else:
                    has_invalid_encoding = True

        if is_quantizer_enabled is False or has_invalid_encoding is True:
            layer.set_mode(QcQuantizeOpMode.PASSTHROUGH)
        else:
            layer.set_mode(QcQuantizeOpMode.ACTIVE)
        return has_invalid_encoding

    def export(self, path: str, filename_prefix: str, input_shape: Union[Tuple, List[Tuple]],
               set_onnx_layer_names: bool = True, dummy_input: Union[torch.Tensor, Tuple] = None,
               use_torch_script_graph: bool = False, ):
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
        :param input_shape: shape of the model input as a tuple. If the model takes more than one input, specify this as
               a list of shapes.
        :param set_onnx_layer_names: If ONNX layer names should be set while exporting the model. Default is True
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :param use_torch_script_graph: if exporting of encoding should use torch script tensor names. Default is False
        :return: None

        """
        # save the quantized model and encodings
        model_filename = filename_prefix + '.pth'
        model_path = os.path.join(path, model_filename)

        # Create a version of the model without any quantization ops
        model_to_export = copy.deepcopy(self.model)
        all_modules_in_model_to_export = [module for module in model_to_export.modules()]
        self._remove_quantization_wrappers(model_to_export, all_modules_in_model_to_export)
        torch.save(model_to_export, model_path)

        if not dummy_input:
            if input_shape is None:
                raise AssertionError('Must provide either input shape or a dummy input for export')
            dummy_input = tuple(utils.create_rand_tensors_given_shapes(input_shape,
                                                                       device=utils.get_device(model_to_export)))
        if use_torch_script_graph:
            self.export_torch_script_model_and_encodings(path, filename_prefix, model_to_export, dummy_input)
        else:
            self.export_onnx_model_and_encodings(path, filename_prefix, model_to_export,
                                                 dummy_input, set_onnx_layer_names)

    def export_torch_script_model_and_encodings(self, path: str, filename_prefix: str, model: torch.nn.Module,
                                                dummy_input: Union[torch.Tensor, Tuple]):
        """
        This method exports  a onnx mode and the corrsponding encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param model: model without the quantsim ops
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :return: None
        """
        with torch.no_grad():
            trace = torch.jit.trace(model, dummy_input)
            ts_path = os.path.join(path, filename_prefix + '.torchscript.pth')
            trace.save(ts_path)

            # reload the trace from the saved trace file
            trace = torch.jit.load(ts_path)
            torch_script_node_io_tensor_map, valid_param_set = \
                torchscript_utils.get_node_to_io_tensor_names_map(model, trace, dummy_input)

        # Export encodings
        self._export_encodings_to_json(path, filename_prefix, torch_script_node_io_tensor_map, valid_param_set)

    def export_onnx_model_and_encodings(self, path: str, filename_prefix: str, model: torch.nn.Module,
                                        dummy_input: Union[torch.Tensor, Tuple], set_onnx_layer_names):
        """
        This method exports a onnx model and the corresponding encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param model: model without the quantsim ops
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :param set_onnx_layer_names: If ONNX layer names should be set while exporting the model
        :return: None

        """
        # Save model to onnx
        onnx_path = os.path.join(path, filename_prefix + '.onnx')
        torch.onnx.export(model, dummy_input, onnx_path)
        #  Set the onnx layer names
        if set_onnx_layer_names:
            onnx_utils.OnnxSaver.set_node_names(onnx_path, model, dummy_input)
        onnx_model = onnx.load(onnx_path)
        onnx_node_to_io_tensor_map, valid_param_set = \
            onnx_utils.OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)

        # Export encodings
        self._export_encodings_to_json(path, filename_prefix, onnx_node_to_io_tensor_map, valid_param_set)

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

    def _replace_wrappers_for_quantize_dequantize(self):
        pass

    @staticmethod
    def _validate_quantsim_inputs(quant_scheme: Union[str, QuantScheme], rounding_mode: str, default_output_bw: int,
                                  default_param_bw: int):
        """
        Perform sanity checks on inputs to QuantSim
        :param quant_scheme: Quantization scheme. Supported options are 'tf_enhanced' or 'tf' or using Quant Scheme Enum
                             QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
        :param rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs and outputs
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters
        :return:
        """
        # sanity checks
        if quant_scheme not in ('tf_enhanced', 'tf') and not isinstance(quant_scheme, QuantScheme):
            raise ValueError('Parameter quantization mode is not a valid selection. Valid selections are tf, '
                             'tf_enhanced, QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced')

        if rounding_mode not in ('nearest', 'stochastic'):
            raise ValueError('Parameter round mode is not a valid selection. Valid selections are nearest or '
                             'stochastic')

        if default_param_bw < 4 or default_param_bw > 32:
            raise ValueError('Default bitwidth for parameters must be between 4 and 32, not '+str(default_param_bw))

        if default_output_bw < 4 or default_output_bw > 32:
            raise ValueError('Activation bitwidth must be between 4 and 32, not '+str(default_output_bw))

    @staticmethod
    def _find_next_downstream_modules(op):
        downstream_modules = []
        for succeeding_op in list(op.output.consumers):
            if succeeding_op.get_module():
                downstream_modules.append(succeeding_op.get_module())

            elif succeeding_op.type == 'Split':
                downstream_modules += QuantizationSimModel._find_next_downstream_modules(succeeding_op)

        return downstream_modules

    def _export_encodings_to_json(self, path: str, filename_prefix: str, op_to_io_tensor_map: Dict,
                                  valid_param_set: set):
        """
        Save the quantized model weight encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: filename to store exported weight encodings in json format
        :param op_to_io_tensor_map: Dictionary of layer to I/O tensor mapping from onnx or torch script model
        :param valid_param_set: a set of valid param input names in model
        :return: None
        """

        # Create a dictionary to export to JSON
        activation_encodings = {}
        param_encodings = {}
        quantized_layers = self._get_qc_quantized_layers(self.model)

        for layer_name, layer in quantized_layers:
            self._update_encoding_dicts_for_layer(layer, layer_name, activation_encodings,
                                                  param_encodings, op_to_io_tensor_map,
                                                  valid_param_set)

        encodings_dict = {'activation_encodings': activation_encodings,
                          'param_encodings': param_encodings}

        # export weight encodings to output json file
        encoding_file_path = os.path.join(path, filename_prefix + '.encodings')
        with open(encoding_file_path, 'w') as encoding_fp:
            json.dump(encodings_dict, encoding_fp, sort_keys=True, indent=4)

    def _update_param_encodings_dict_for_layer(self, layer: torch.nn.Module, layer_name: str,
                                               param_encodings: Dict, valid_param_set: set):
        """

        :param layer: layer as torch.nn.Module
        :param layer_name : Name of the layer
        :param param_encodings: dictionary of param encodings
        :param valid_param_set: a set of valid param input names in model
        :return:
        """

        disabled_param_quantizers = []
        for orig_param_name, param_quantizer in layer.param_quantizers.items():
            param_name = layer_name + '.' + orig_param_name
            if param_quantizer.enabled:
                if param_name in valid_param_set:
                    tensor_encoding = self._create_encoding_dict_for_quantizer(param_quantizer)
                    param_encodings[param_name] = [tensor_encoding]
                else:
                    logger.error('Param tensor {%s} not found in valid param set', param_name)
            else:
                disabled_param_quantizers.append(orig_param_name)

        # retrieve the appropriate param generator
        if isinstance(layer, QcQuantizeWrapper):
            # pylint: disable=protected-access
            named_parameters = layer._module_to_wrap.named_parameters()
        else:
            named_parameters = layer.named_parameters(recurse=False)

        for name, param in named_parameters:
            # if the param quantizer was disabled generate encoding assuming bitwidth of 32
            if name in disabled_param_quantizers:
                param_name = layer_name + '.' + name
                tensor_encoding = self.compute_encoding_for_given_bitwidth(layer.param_quantizers[name], param, 32)
                param_encodings[param_name] = [tensor_encoding]

    @staticmethod
    def compute_encoding_for_given_bitwidth(quantizer: TensorQuantizer, tensor, bitwidth) -> Dict:
        """
        Utility function to compute encoding for a given bitwidth
        :param quantizer: quantizer to use for configuration
        :param tensor: tensor to quantize
        :param bitwidth: bitwidth for generating the encoding
        """
        encoding_analyzer = libpymo.EncodingAnalyzerForPython(quantizer.quant_scheme)
        encoding_analyzer.updateStats(tensor.cpu().detach().numpy(), False)
        encoding, is_encoding_valid = \
            encoding_analyzer.computeEncoding(bitwidth, quantizer.use_symmetric_encodings)
        if is_encoding_valid:
            return {'min': encoding.min,
                    'max': encoding.max,
                    'scale': encoding.delta,
                    'offset': encoding.offset,
                    'bitwidth': encoding.bw,
                    'is_symmetric': str(quantizer.use_symmetric_encodings)}

        return {}

    def _update_encoding_dicts_for_layer(self, layer: torch.nn.Module, layer_name: str, activation_encodings: Dict,
                                         param_encodings: Dict, op_to_io_tensor_map: Dict,
                                         valid_param_set: set):

        """
        Add given layer param and activation encodings to respective dictionaries to be used for exporting encodings
        :param layer: layer as torch.nn.Module
        :param layer_name: Name of the layer
        :param activation_encodings: dictionary of activation encodings
        :param param_encodings: dictionary of param encodings
        :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
        :param valid_param_set: a set of valid param input names in model
        :return:
        """

        # pylint: disable=too-many-nested-blocks
        # pylint: disable=too-many-branches
        if layer_name not in op_to_io_tensor_map:
            logger.info("layer with name {%s} not found in model, not an issue; "
                        "skip and continue ", layer_name)
        else:
            if isinstance(layer, QcQuantizeWrapper):
                # get activation quantizers
                if layer.input_quantizer.enabled:
                    param_inputs = [layer_name + '.' + param_name for param_name in layer.param_quantizers]
                    if op_to_io_tensor_map[layer_name].inputs:
                        for input_tensor in op_to_io_tensor_map[layer_name].inputs:
                            if input_tensor not in param_inputs:
                                tensor_encoding = self._create_encoding_dict_for_quantizer(layer.input_quantizer)
                                activation_encodings[input_tensor] = [tensor_encoding]
                if layer.output_quantizers[0].enabled:
                    if op_to_io_tensor_map[layer_name].outputs:
                        for output_tensor in op_to_io_tensor_map[layer_name].outputs:
                            tensor_encoding = self._create_encoding_dict_for_quantizer(layer.output_quantizers[0])
                            activation_encodings[output_tensor] = [tensor_encoding]

                # get param quantizers
                self._update_param_encodings_dict_for_layer(layer, layer_name, param_encodings, valid_param_set)

            if isinstance(layer, QcQuantizeRecurrent):
                onnx_activations_to_quantizers, onnx_params_to_quantizers = \
                    layer.get_activation_param_quantizers_for_onnx_tensors(op_to_io_tensor_map[layer_name])
                for tensor, quantizer in onnx_activations_to_quantizers.items():
                    tensor_encoding = self._create_encoding_dict_for_quantizer(quantizer)
                    activation_encodings[tensor] = [tensor_encoding]
                for tensor, quantizer in onnx_params_to_quantizers.items():
                    tensor_encoding = self._create_encoding_dict_for_quantizer(quantizer)
                    param_encodings[tensor] = [tensor_encoding]

    @staticmethod
    def _create_encoding_dict_for_quantizer(quantizer: TensorQuantizer) -> Dict:
        if quantizer.encoding:
            encoding_min, encoding_max, bw = quantizer.encoding.min, quantizer.encoding.max, quantizer.encoding.bw
            scale, offset = calculate_delta_offset(encoding_min, encoding_max, bw)
            return {'min': encoding_min,
                    'max': encoding_max,
                    'scale': scale,
                    'offset': offset,
                    'bitwidth': bw,
                    'is_symmetric': str(quantizer.use_symmetric_encodings)}
        return None

    @staticmethod
    def _get_qc_quantized_layers(model) -> List[Tuple[str, QcQuantizeWrapper]]:
        quantized_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (QcQuantizeWrapper, QcQuantizeRecurrent)):
                quantized_layers.append((name, module))
        return quantized_layers

    @staticmethod
    def _is_leaf_module(module):
        """Utility function to determine if the given module is a leaf module - that is, does not have children modules
        :return:
            True if the module is a leaf, False otherwise
        """
        module_list = list(module.modules())
        return bool(len(module_list) == 1)

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

    def _create_quantizer_module(self, module_to_quantize: torch.nn.Module) -> torch.nn.Module:
        """Instantiates wrapper based on quant scheme
        """
        assert self._quant_scheme in [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced]

        # Set quantizer to be a module replacer if it is in qc_quantize_modules_dict, otherwise set as
        # QcPostTrainingWrapper.
        quantizer = qc_quantize_modules_dict.get(type(module_to_quantize), QcPostTrainingWrapper)
        quantized_module = quantizer(module_to_quantize, self._default_param_bw, self._default_output_bw,
                                     self._rounding_mode, self._quant_scheme)

        return quantized_module

    def _add_quantization_wrappers(self, module):
        """Recursively add quantization wrappers to all appropriate modules starting with module
        """
        for module_name, module_ref in module.named_children():

            logger.debug("nn.Module found : %s", module_ref)

            # check if the module already quantized then ignore
            if not self._is_quantizable_module(module_ref):
                continue

            # check if the module is leaf or not
            if self._is_leaf_module(module_ref):

                # Create a new QcQuantize wrapper module
                quantized_module = self._create_quantizer_module(module_ref)

                setattr(module, module_name, quantized_module)

            # recursively call children modules
            else:
                self._add_quantization_wrappers(module_ref)

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
                    setattr(starting_module, module_name, PassThroughOp())

                elif isinstance(module_ref, QcQuantizeRecurrent):
                    module_ref.update_params()
                    setattr(starting_module, module_name, module_ref.module_to_quantize)

            # Recursively call children modules if present
            if not cls._is_leaf_module(module_ref):
                cls._remove_quantization_wrappers(module_ref, list_of_modules_to_exclude)

    def configure_quantization_ops(self, connected_graph: Union[None, ConnectedGraph], config_file: str):
        """
        Configure inserted quantize ops using config file
        :param connected_graph: Connected graph representation of the model
        :param config_file: Configuration file to use
        """
        if connected_graph is None:
            logger.error('A connected graph failed to be built.\n'
                         'Unable to proceed with automatically configuring quantization ops using the config file.\n'
                         'Please configure quantization ops manually by redefining '
                         'QuantizationSimModel.configure_quantization_ops()')
            raise AssertionError
        QuantSimConfigurator(self.model, connected_graph, config_file)

    def set_and_freeze_param_encodings(self, encoding_path: str):
        """
        Set and freeze parameter encodings from encodings JSON file
        :param encoding_path: path from where to load parameter encodings file
        """
        # Load parameter encodings file
        with open(encoding_path) as json_file:
            param_encodings = json.load(json_file)

        for name, quant_module in self.model.named_modules():
            if isinstance(quant_module, QcPostTrainingWrapper):
                quant_module.set_and_freeze_param_encoding(name, param_encodings)


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
