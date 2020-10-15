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
from aimet_torch.tensor_quantizer import TensorQuantizer
from aimet_torch.batch_norm_fold import PassThroughOp
from aimet_torch import utils
from aimet_torch import onnx_utils
from aimet_torch.meta.connectedgraph_utils import create_connected_graph
from aimet_torch.meta.connectedgraph import ConnectedGraph

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


# Types of modules which cannot be quantized
unquantizable_modules = (QcQuantizeWrapper, QcQuantizeStandAloneBase, PassThroughOp)


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
    def __init__(self, model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]],
                 quant_scheme: Union[str, QuantScheme] = QuantScheme.post_training_tf_enhanced,
                 rounding_mode: str = 'nearest', default_output_bw: int = 8, default_param_bw: int = 8,
                 in_place: bool = False, config_file: str = None):
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

        # save some parameters
        if in_place:
            self.model = model
        else:
            self.model = copy.deepcopy(model)

        try:
            connected_graph = create_connected_graph(self.model, input_shapes)
        except (torch.jit.TracingCheckError, AssertionError):
            logger.warning('Error in tracing while creating the connected graph.\n'
                           'The connected graph passed into self.configure_quantization_ops() will be None.\n'
                           'If this function has been overridden to not depend on connected graph, this warning can be '
                           'ignored.')
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
            if wrapper.output_quantizer.enabled:
                stream.write('    Output: bw={}, encoding-present={}\n'.format(wrapper.output_quantizer.bitwidth,
                                                                               bool(wrapper.output_quantizer.encoding)))
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
            if layer.output_quantizer.enabled and layer.output_quantizer.encoding:
                layer.set_mode(QcQuantizeOpMode.ACTIVE)
                encoding = layer.output_quantizer.encoding
                logger.debug("Encoding for %s: min=%f, max=%f, offset=%f. delta=%f, bw=%f",
                             name, encoding.min, encoding.max, encoding.delta, encoding.offset, encoding.bw)
            elif layer.output_quantizer.enabled:
                layers_with_invalid_encodings.append((name, 'output'))
                layer.set_mode(QcQuantizeOpMode.PASSTHROUGH)

            if layer.input_quantizer.enabled and layer.input_quantizer.encoding:
                layer.set_mode(QcQuantizeOpMode.ACTIVE)
                encoding = layer.input_quantizer.encoding
                logger.debug("Encoding for %s: min=%f, max=%f, offset=%f. delta=%f, bw=%f",
                             name, encoding.min, encoding.max, encoding.delta, encoding.offset, encoding.bw)
            elif layer.input_quantizer.enabled:
                layers_with_invalid_encodings.append((name, 'input'))
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

    def export(self, path: str, filename_prefix: str, input_shape: Union[Tuple, List[Tuple]],
               set_onnx_layer_names: bool = True):
        """
        This method exports out the quant-sim model so it is ready to be run on-target.

        Specifically, the following are saved

        1. The sim-model is exported to a regular PyTorch model without any simulation ops
        2. The quantization encodings are exported to a separate JSON-formatted file that can
           then be imported by the on-target runtime (if desired)
        3. An equivalent model in ONNX format is exported. In addition, nodes in the ONNX model are named
           the same as the corresponding PyTorch module names. This helps with matching ONNX node to their quant
           encoding from #2.

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param input_shape: shape of the model input as a tuple. If the model takes more than one input, specify this as
               a list of shapes.
        :param set_onnx_layer_names: If ONNX layer names should be set while exporting the model. Default is True
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

        # Save model to onnx
        onnx_path = os.path.join(path, filename_prefix + '.onnx')
        dummy_input = utils.create_rand_tensors_given_shapes(input_shape)
        torch.onnx.export(model_to_export, tuple(dummy_input), onnx_path)

        #  Set the onnx layer names
        if set_onnx_layer_names:
            onnx_utils.OnnxSaver.set_node_names(onnx_path, model_to_export, input_shape)
        onnx_model = onnx.load(onnx_path)
        onnx_node_to_io_tensor_map, valid_param_set = onnx_utils.OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx_model)

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
            if isinstance(module, QcQuantizeWrapper):
                if param_name_to_exclude in module.param_quantizers:
                    module.param_quantizers[param_name_to_exclude].enabled = False

    def create_super_nodes_of_layers_and_activation_functions(self, input_shape: Tuple):
        """
        Creates a super-node from a quantization perspective of a layer followed by an activation.
        For example, a conv and a relu
        :param input_shape: Shapes of the input to the model
        :return: None
        """

        random_inputs = utils.create_rand_tensors_given_shapes(input_shape)
        graph = ConnectedGraph(self.model, random_inputs)

        conv_modules = [(name, module) for name, module in self.model.named_modules(prefix=type(self.model).__name__)
                        if isinstance(module, QcQuantizeWrapper) and
                        isinstance(module._module_to_wrap, torch.nn.Conv2d)]  # pylint: disable=protected-access

        conv_ops = [(graph.get_op_from_module_name(conv_name + '._module_to_wrap'), module)
                    for conv_name, module in conv_modules]

        for op, quantized_wrapper in conv_ops:
            if self._is_following_module_relu(op):
                quantized_wrapper.output_quantizer.enabled = False

    def handle_element_wise_ops(self, input_shape: Tuple):
        """
        Special handling for element-wise ops
        :param input_shape: Shape of the input to the model
        :return: None
        """
        random_inputs = utils.create_rand_tensors_given_shapes(input_shape)
        graph = ConnectedGraph(self.model, random_inputs)

        element_wise_ops = []
        # Find all add ops
        for op in graph.get_all_ops().values():
            if op.type in ['add', 'mul', 'div', 'cat']:
                element_wise_ops.append(op)

        modules_downstream_from_elementwise_ops = []
        for op in element_wise_ops:
            modules_downstream_from_elementwise_ops += self._find_next_downstream_modules(op)

        # pylint: disable=protected-access
        for module in self.model.modules():
            if isinstance(module, QcQuantizeWrapper) and \
                    (module._module_to_wrap in modules_downstream_from_elementwise_ops):
                module.input_quantizer.enabled = True

    def _replace_wrappers_for_quantize_dequantize(self):
        pass

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
    def _is_following_module_relu(op):
        succeeding_modules = list(op.output.consumers)

        # Cannot fold into more than one downstream ops
        if len(succeeding_modules) > 1:
            return False

        # No downstream op to fold into
        if not succeeding_modules:
            return False

        if succeeding_modules[0].type == 'relu':
            return True

        # Downstream op not the right type
        return False

    def _export_encodings_to_json(self, path: str, filename_prefix: str, onnx_node_to_io_tensor_map: Dict,
                                  valid_param_set: set):
        """
        Save the quantized model weight encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: filename to store exported weight encodings in json format
        :param onnx_node_to_io_tensor_map: Dictionary of layer to I/O tensor mapping from onnx model
        :param valid_param_set: a set of valid param input names in model
        :return: None
        """

        # Create a dictionary to export to JSON
        activation_encodings = {}
        param_encodings = {}
        quantized_layers = self._get_qc_quantized_layers(self.model)

        for layer_name, layer in quantized_layers:
            self._update_encoding_dicts_for_layer(layer, layer_name, activation_encodings,
                                                  param_encodings, onnx_node_to_io_tensor_map,
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

        for orig_param_name, param_quantizer in layer.param_quantizers.items():
            param_name = layer_name + '.' + orig_param_name
            if param_quantizer.enabled:
                if param_name in valid_param_set:
                    tensor_encoding = self._create_encoding_dict_for_quantizer(param_quantizer)
                    param_encodings[param_name] = [tensor_encoding]
                else:
                    logger.error('Param tensor {%s} not found in onnx valid param set', param_name)

    def _update_encoding_dicts_for_layer(self, layer: torch.nn.Module, layer_name: str, activation_encodings: Dict,
                                         param_encodings: Dict, onnx_node_to_io_tensor_map: Dict,
                                         valid_param_set: set):

        """
        Add given layer param and activation encodings to respective dictionaries to be used for exporting encodings
        :param layer: layer as torch.nn.Module
        :param layer_name: Name of the layer
        :param activation_encodings: dictionary of activation encodings
        :param param_encodings: dictionary of param encodings
        :param onnx_node_to_io_tensor_map: ONNX map of layer name to it's input/output tensors
        :param valid_param_set: a set of valid param input names in model
        :return:
        """

        if layer_name not in onnx_node_to_io_tensor_map:
            logger.info("layer with name {%s} not found in onnx model, not an issue; "
                        "skip and continue ", layer_name)

        if isinstance(layer, QcQuantizeWrapper) and layer_name in onnx_node_to_io_tensor_map:
            # get activation quantizers
            if layer.input_quantizer.enabled:
                param_inputs = [layer_name + '.' + param_name for param_name in layer.param_quantizers]
                if onnx_node_to_io_tensor_map[layer_name].inputs:
                    for input_tensor in onnx_node_to_io_tensor_map[layer_name].inputs:
                        if input_tensor not in param_inputs:
                            tensor_encoding = self._create_encoding_dict_for_quantizer(layer.input_quantizer)
                            activation_encodings[input_tensor] = [tensor_encoding]
            if layer.output_quantizer.enabled:
                if onnx_node_to_io_tensor_map[layer_name].outputs:
                    for output_tensor in onnx_node_to_io_tensor_map[layer_name].outputs:
                        tensor_encoding = self._create_encoding_dict_for_quantizer(layer.output_quantizer)
                        activation_encodings[output_tensor] = [tensor_encoding]

            # get param quantizers
            self._update_param_encodings_dict_for_layer(layer, layer_name, param_encodings, valid_param_set)


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
            if isinstance(module, QcQuantizeWrapper):
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

    def _create_quantizer_module(self, module_to_wrap: torch.nn.Module) -> torch.nn.Module:
        """Instantiates wrapper based on quant scheme
        """
        assert self._quant_scheme in [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced]

        quantized_module = QcPostTrainingWrapper(module_to_wrap, self._default_param_bw, self._default_output_bw,
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
                         'Unable to proceed with configuring the quantization ops using the config file.\n'
                         'Please configure quantization ops manually by redefining the configure_quantization_ops() '
                         'function.')
            raise AssertionError
        QuantSimConfigurator(self.model, connected_graph, config_file)


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
    :return: A new instance of the QUantizationSimModel created after loading the checkpoint
    """
    with open(file_path, 'rb') as file:
        sim = pickle.load(file)
        return sim
