# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

from dataclasses import dataclass
import os
from typing import Dict, List, Union, Tuple, Optional
import json
import numpy as np
import onnx

from onnx import helper
from onnxsim import simplify
import onnxruntime as ort
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from packaging import version

from aimet_common import libpymo
from aimet_common import libquant_info
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_common.quantsim import encoding_version, extract_global_quantizer_args
from aimet_common.utils import save_json_yaml, AimetLogger
from aimet_onnx import utils
from aimet_onnx.meta.operations import Op
from aimet_onnx.meta.utils import get_op_given_param_name, get_param_shape_using_connected_graph
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.qc_quantize_op import QcQuantizeOp, OpMode, TensorQuantizerParams
from aimet_onnx.quantsim_config.quantsim_config import QuantSimConfigurator
from aimet_onnx.utils import make_dummy_input, add_hook_to_get_activation, remove_activation_hooks

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import ModelProto
else:
    from onnx.onnx_pb import ModelProto

WORKING_DIR = '/tmp/quantsim/'

op_types_to_ignore = ["branch", "Flatten", "Gather", "Reshape", "Shape", "Unsqueeze", "Squeeze", "Split",
                      "Compress", "Tile", "Transpose", "Identity"]

allowed_op_type_for_per_channel = ['Conv', 'Gemm', 'MatMul', 'ConvTranspose']

data_types_to_quantize = [np.float32]

@dataclass
class EncodingMismatchInfo:
    """
    Dataclass tracking information about mismatched quantizer vs. encoding settings.
    """
    quantizer_name: str
    enabled_mismatch: Optional[Tuple] = None
    dtype_mismatch: Optional[Tuple] = None
    bitwidth_mismatch: Optional[Tuple] = None
    is_symmetric_mismatch: Optional[Tuple] = None
    is_strict_symmetric_mismatch: Optional[Tuple] = None
    is_unsigned_symmetric_mismatch: Optional[Tuple] = None

    def has_mismatch(self) -> bool:
        """
        Returns True if there is a mismatched setting.

        :return: True if there is a mismatched setting, False otherwise
        """
        return (self.enabled_mismatch is not None or
                self.dtype_mismatch is not None or
                self.bitwidth_mismatch is not None or
                self.is_symmetric_mismatch is not None or
                self.is_strict_symmetric_mismatch is not None or
                self.is_unsigned_symmetric_mismatch is not None)


class QuantizationSimModel:
    """ Creates a QuantizationSimModel model by adding quantization simulations ops to a given model """

    # pylint: disable=too-many-arguments, too-many-locals, too-many-instance-attributes
    def __init__(self,
                 model: ModelProto,
                 dummy_input: Dict[str, np.ndarray] = None,
                 quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                 rounding_mode: str = 'nearest',
                 default_param_bw: int = 8,
                 default_activation_bw: int = 8,
                 use_symmetric_encodings: bool = False, use_cuda: bool = True,
                 device: int = 0, config_file: str = None,
                 default_data_type: QuantizationDataType = QuantizationDataType.int,
                 simplify_model: bool = True, user_onnx_libs: List[str] = None):
        """
        Constructor

        :param model: ONNX model or path to model
        :param dummy_input: Dummy input to the model. If None, will attempt to auto-generate a dummy input
        :param quant_scheme: Quantization scheme (e.g. QuantScheme.post_training_tf)
        :param rounding_mode: Rounding mode (e.g. nearest)
        :param default_param_bw: Quantization bitwidth for parameter
        :param default_activation_bw: Quantization bitwidth for activation
        :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        :param use_cuda: True if using CUDA to run quantization op. False otherwise.
        :param config_file: Path to Configuration file for model quantizers
        :param default_data_type: Default data type to use for quantizing all layer inputs, outputs and parameters.
                                 Possible options are QuantizationDataType.int and QuantizationDataType.float.
                                 Note that the mode default_data_type=QuantizationDataType.float is only supported with
                                 default_output_bw=16 and default_param_bw=16
        :param simplify_model: Default True, uses onnx simplifier to simplify model
        :param user_onnx_libs: List of paths to all compiled ONNX custom ops libraries
        """
        self.model = model
        if not isinstance(model, ONNXModel):
            self.model = ONNXModel(model)

        if simplify_model:
            try:
                self.model.model, _ = simplify(self.model.model)
            # pylint: disable=bare-except
            except:
                logger.info('ONNX Simplifier failed. Proceeding with unsimplified model.')

        if not dummy_input:
            dummy_input = make_dummy_input(self.model.model)
        self.qc_quantize_op_dict = {}
        self.connected_graph = ConnectedGraph(self.model)
        self._quant_scheme = quant_scheme
        self._rounding_mode = rounding_mode
        self._default_param_bw = default_param_bw
        self._default_activation_bw = default_activation_bw
        self._default_quantization_data_type = default_data_type
        self._use_symmetric_encodings = use_symmetric_encodings
        self._use_cuda = use_cuda
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            self._use_cuda = False
        if self._use_cuda:
            self._op_domain = "aimet.customop.cuda"
            self.providers = [('CUDAExecutionProvider', {'device_id': device, 'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider']
        else:
            self._op_domain = "aimet.customop.cpu"
            self.providers = ['CPUExecutionProvider']
        self._user_onnx_libs = user_onnx_libs
        self.param_names = []
        self.input_quantizers_name = []
        self.activation_names = []
        self.activation_dtypes = {}
        self._get_param_names()
        self._get_activations_to_quantize(dummy_input)
        self._add_quantization_nodes()
        self.session = QuantizationSimModel.build_session(self.model.model, self.providers, self._user_onnx_libs)

        quantsim_configurator = self._add_configuration_(config_file)

        self._supported_kernels = quantsim_configurator.get_supported_kernels()
        self._op_to_supported_kernel = quantsim_configurator.get_op_to_supported_kernels()

        self.quant_args = extract_global_quantizer_args(quant_scheme, quantsim_configurator)

    def get_supported_kernels(self) -> Dict:
        """
        Return _supported_kernels parsed from the config file
        :return: Dictionary containing supported_kernels
        """
        return self._supported_kernels

    def _add_configuration_(self, config_file: str):
        """
        Add configuration based on config file

        :param config_file: Path to Configuration file for model quantizers
        """
        quantsim_configurator = QuantSimConfigurator(self.model, self.connected_graph, config_file,
                                                     self._default_activation_bw, self._default_param_bw,
                                                     self._default_quantization_data_type)
        quantsim_configurator.configure_quantizers(self.qc_quantize_op_dict, self.param_names, self.activation_names,
                                                   self.input_quantizers_name)

        return quantsim_configurator

    def _get_param_names(self):
        """
        Get the names of params
        """
        valid_ops = self._get_ops_with_parameter()
        for op in valid_ops:
            for param_info in op.parameters.values():
                param, _ = param_info
                if param.name and param.name not in self.param_names:
                    self.param_names.append(param.name)

    def _get_ops_with_parameter(self) -> List[Op]:
        """
        Gets ops with parameters to add quantization nodes for

        :return: Connected graph ops
        """
        valid_ops = [op for op in self.connected_graph.get_all_ops().values() if op.type not in ['BatchNormalization']]
        return valid_ops

    def _get_activations_to_quantize(self, dummy_input: Dict[str, np.ndarray]):
        """
        Get the names of activations to quantize

        :param dummy_input: Sample input to be run through the model
        """
        self.fill_activation_dtypes(dummy_input)
        for node in self.model.nodes():
            if node.op_type not in op_types_to_ignore:
                for name in node.output:
                    if name not in self.activation_names and name not in self.param_names and \
                            self._is_op_quantizable(name):
                        self.activation_names.append(name)
            for input_name in node.input:
                if input_name not in self.activation_names and input_name not in self.param_names:
                    for tensors in self.model.model.graph.initializer:
                        if tensors.name == input_name and tensors.data_type == 1: # 1 corresponds to float, dictionary can be found by using onnx.TensorProto.DataType.items()
                            self.activation_names.append(tensors.name)
                            self.input_quantizers_name.append(tensors.name)

        # Model inputs
        for node in self.model.graph().input:
            name = node.name
            if name not in self.activation_names and name not in self.param_names and self._is_op_quantizable(name):
                self.activation_names.append(name)

        # Model outputs
        for node in self.model.graph().output:
            if node.name in self.activation_names:
                node.name += '_updated'

    def _is_op_quantizable(self, name: str) -> bool:
        """
        Checks whether the given activation should be quantized

        :param name: Name of the activation
        :return: True if the activation should be quantized
        """
        # Check if activation is used as an input to another node
        if name not in self.activation_dtypes.keys() or self.activation_dtypes[name] not in data_types_to_quantize:
            return False
        return True

    def fill_activation_dtypes(self, dummy_input: Dict[str, np.ndarray]):
        """
        Get the data type for each activation

        :param dummy_input: Sample input to run through the model
        """
        activations = utils.get_graph_intermediate_activations(self.model.graph())
        hooks = []
        for name in activations:
            hooks.append(add_hook_to_get_activation(self.model.model, name))
        sess = QuantizationSimModel.build_session(self.model.model, self.providers, self._user_onnx_libs)
        outputs = sess.run(None, dummy_input)
        for idx in range(len(self.model.graph().output)):
            act_name = self.model.graph().output[idx].name
            dtype = outputs[idx].dtype
            self.activation_dtypes[act_name] = dtype
        remove_activation_hooks(self.model.model, hooks)

    def _add_quantization_nodes(self):
        """
        Call insert functions for quantization nodes
        """
        self._insert_param_quantization_nodes()
        self._insert_activation_quantization_nodes()

    def _insert_param_quantization_nodes(self):
        """
        Insert quantization node for each param tensor
        """
        for name in self.param_names:
            self.model.replace_input_of_all_nodes(name, name + '_qdq')

            quant_info, tensor_quantizer_params = self._create_quant_info_object_for_param(name)
            custom_node = helper.make_node(
                op_type='QcQuantizeOp',
                inputs=[name],
                outputs=[name + '_qdq'],
                name='QcQuantizeOp_' + name,
                domain=self._op_domain,
                op_name=name,
                quant_info=libpymo.PtrToInt64(quant_info),
            )
            self.model.add_node(custom_node)
            self.qc_quantize_op_dict[name] = QcQuantizeOp(quant_info=quant_info,
                                                          quant_scheme=self._quant_scheme,
                                                          rounding_mode=self._rounding_mode,
                                                          encodings=None,
                                                          op_mode=OpMode.oneShotQuantizeDequantize,
                                                          bitwidth=self._default_param_bw,
                                                          use_symmetric_encodings=self._use_symmetric_encodings,
                                                          tensor_quantizer_params=tensor_quantizer_params)

    def _create_quant_info_object_for_param(self, param_name: str):
        """
        Creates quant info object for QcQuantizeOp and QDQ node

        :param param_name: Name of the parameter for which the quant info object will be created
        :return: quant info object
        """
        quant_info = libquant_info.QcQuantizeInfo()
        quant_info.usePerChannelMode = False
        tensor_quantizer_params = TensorQuantizerParams()
        op = get_op_given_param_name(self.connected_graph, param_name)
        param_shape = get_param_shape_using_connected_graph(self.connected_graph, param_name)
        if len(param_shape) == 1:
            tensor_quantizer_params.axis = 0
            tensor_quantizer_params.num_output_channels = param_shape[0]
        else:
            tensor_quantizer_params.axis = self._get_quantization_axis(op)
            tensor_quantizer_params.num_output_channels = param_shape[quant_info.channelAxis]

        return quant_info, tensor_quantizer_params

    @staticmethod
    def _get_quantization_axis(op: Op) -> int:
        """
        Gets quantization axis for Per channel quantization

        :param op: Connected graph op
        return: axis
        """
        if op.type in ['Conv']:
            return 0
        if op.type in ['ConvTranspose']:
            return 1
        if op.type in ['Gemm', 'MatMul'] and op.transposed_params:
            return 0
        if op.type in ['Gemm', 'MatMul']:
            return 1
        return -1

    def _insert_activation_quantization_nodes(self):
        """
        Insert quantization node for each activation tensor
        """
        for name in self.activation_names:
            self.model.replace_input_of_all_nodes(name, name + '_updated')
            quant_info = libquant_info.QcQuantizeInfo()
            custom_node = helper.make_node(
                op_type='QcQuantizeOp',
                inputs=[name],
                outputs=[name + '_updated'],
                name='QcQuantizeOp_' + name,
                domain=self._op_domain,
                op_name=name,
                quant_info=libpymo.PtrToInt64(quant_info)
            )
            self.model.add_node(custom_node)
            self.qc_quantize_op_dict[name] = QcQuantizeOp(quant_info=quant_info,
                                                          quant_scheme=self._quant_scheme,
                                                          rounding_mode=self._rounding_mode,
                                                          encodings=None,
                                                          op_mode=OpMode.updateStats,
                                                          bitwidth=self._default_activation_bw,
                                                          use_symmetric_encodings=self._use_symmetric_encodings
                                                          )

    @staticmethod
    def build_session(model, providers: List, user_onnx_libs: List[str] = None):
        """
        Build and return onnxruntime inference session

        :param model: onnx model
        :param providers: providers to execute onnxruntime
        :param user_onnx_libs: list of paths to user custom ONNX op libraries
        """
        sess_options = SessionOptions()
        shared_library = os.path.dirname(libquant_info.__file__)
        shared_library = os.path.join(shared_library, "libaimet_onnxrt_ops.so")
        sess_options.register_custom_ops_library(shared_library)
        if user_onnx_libs is not None:
            for lib in user_onnx_libs:
                sess_options.register_custom_ops_library(lib)
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
        session = InferenceSession(
            path_or_bytes=model.SerializeToString(),
            sess_options=sess_options,
            providers=providers,
        )
        return session

    def get_qc_quantize_op(self):
        """
        Return dict of qc quantize ops
        """
        return self.qc_quantize_op_dict

    def save_model_graph(self, filename_prefix: str):
        """
        Save model to given path

        :param filename_prefix: filename to save the onnx model
        """
        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR)
        self.model.save_model_to_file(os.path.join(WORKING_DIR, filename_prefix) + '.onnx')

    def compute_encodings(self, forward_pass_callback, forward_pass_callback_args):
        """
        Compute and return the encodings of each tensor quantizer

        :param forward_pass_callback: A callback function that simply runs forward passes on the model. This callback
            function should use representative data for the forward pass, so the calculated encodings work for all
            data samples. This callback internally chooses the number of data samples it wants to use for calculating
            encodings.
        :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
            the user to determine the type of this parameter. E.g. could be simply an integer representing the number
            of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
            If set to None, forward_pass_callback will be invoked with no parameters.
        """
        for op_name, qc_op in self.qc_quantize_op_dict.items():
            qc_op.reset_encoding_stats()
            if op_name in self.activation_names:
                qc_op.op_mode = OpMode.updateStats
            else:
                qc_op.op_mode = OpMode.oneShotQuantizeDequantize
                if qc_op.is_encoding_frozen():
                    qc_op.op_mode = OpMode.quantizeDequantize

        forward_pass_callback(self.session, forward_pass_callback_args)
        for op_name, qc_op in self.qc_quantize_op_dict.items():
            if qc_op.data_type == QuantizationDataType.int and not qc_op.is_encoding_frozen():
                qc_op.compute_encodings()
            qc_op.op_mode = OpMode.quantizeDequantize

    @staticmethod
    def _create_encoding_dict(encoding: libpymo.TfEncoding, qc_quantize_op: QcQuantizeOp) -> Union[Dict, None]:
        """
        Create encoding dictionary from encoding object
        :param encoding: Encoding of the quantizer
        :param qc_quantize_op: Quantizer
        :return: Encoding Dictionary
        """
        data_type, bitwidth = qc_quantize_op.data_type, qc_quantize_op.bitwidth

        if data_type == QuantizationDataType.float:
            enc_dict = {'bitwidth': bitwidth, 'dtype': "float"}
        else:
            if encoding:
                encoding_min, encoding_max, bw, scale, offset = encoding.min, encoding.max, encoding.bw, \
                                                                encoding.delta, encoding.offset
                is_symmetric = qc_quantize_op.use_symmetric_encodings

                enc_dict = {'min': encoding_min, 'max': encoding_max, 'scale': scale, 'offset': int(offset),
                            'bitwidth': bw, 'is_symmetric': str(is_symmetric), 'dtype': "int"}
            else:
                enc_dict = None
        return enc_dict

    def _export_encodings(self, encoding_file_path):
        """
        Export encodings to json and yaml file

        :param encoding_file_path: path to save the encoding files
        """

        def update_encoding_dict_entry(encoding_dict: Dict, op_name: str):
            qc_quantize_op = self.qc_quantize_op_dict[op_name]
            encoding_dict[op_name] = []
            for encoding in qc_quantize_op.encodings:
                encoding_dict[op_name].append(QuantizationSimModel._create_encoding_dict(encoding, qc_quantize_op))

        param_encodings = {}
        for name in self.param_names:
            if self.qc_quantize_op_dict[name].enabled:
                update_encoding_dict_entry(param_encodings, name)

        activation_encodings = {}
        for name in self.activation_names:
            if self.qc_quantize_op_dict[name].enabled:
                update_encoding_dict_entry(activation_encodings, name)

        encodings_dict = {'version': encoding_version,
                          'activation_encodings': activation_encodings,
                          'param_encodings': param_encodings,
                          'quantizer_args': self.quant_args}

        save_json_yaml(encoding_file_path, encodings_dict)

    def remove_quantization_nodes(self):
        """
        Remove quantization nodes
        """
        nodes_to_remove = []
        for node in self.model.nodes():
            if node.op_type == 'QcQuantizeOp':
                nodes_to_remove.append(node)
            else:
                for name in node.input:
                    self.model.replace_input_of_all_nodes(name, name.replace('_qdq', '').replace('_updated', ''))
        self.model.remove_nodes(nodes_to_remove)

        for node in self.model.graph().output:
            node.name = node.name.replace('_updated', '')

    def export(self, path: str, filename_prefix: str):
        """
        Compute encodings and export to files

        :param path: dir to save encoding files
        :param filename_prefix: filename to save encoding files
        """
        self._export_encodings(os.path.join(path, filename_prefix) + '.encodings')
        self.remove_quantization_nodes()
        self.model.save_model_to_file(os.path.join(path, filename_prefix) + '.onnx')

    def set_and_freeze_param_encodings(self, encoding_path: str):
        """
        Set and freeze parameter encodings from encodings JSON file

        :param encoding_path: path from where to load parameter encodings file
        """

        # Load encodings file
        with open(encoding_path) as json_file:
            encodings = json.load(json_file)

        for quantizer_name in encodings:
            if quantizer_name in self.qc_quantize_op_dict:
                libpymo_encodings = _create_libpymo_encodings(encodings[quantizer_name])
                is_symmetric, is_strict_symmetric, is_unsigned_symmetric = \
                    get_symmetric_properties(encodings[quantizer_name])
                data_type = QuantizationDataType.int if encodings[quantizer_name][0]['dtype'] == 'int' else \
                    QuantizationDataType.float
                self.qc_quantize_op_dict[quantizer_name].update_quantizer_and_load_encodings(libpymo_encodings,
                                                                                             is_symmetric,
                                                                                             is_strict_symmetric,
                                                                                             is_unsigned_symmetric,
                                                                                             data_type)
                self.qc_quantize_op_dict[quantizer_name].freeze_encodings()

    def get_all_quantizers(self) -> Tuple[List, List]:
        """
        Returns all QcQuantizeOps through which TensorQuantizer's attributes can be accessed.
        """
        param_quantizers = []
        activation_quantizers = []

        for param in self.param_names:
            param_quantizers.append(self.qc_quantize_op_dict[param])

        for activation in self.activation_names:
            activation_quantizers.append(self.qc_quantize_op_dict[activation])

        return param_quantizers, activation_quantizers


def load_encodings_to_sim(quant_sim_model: QuantizationSimModel, onnx_encoding_path: str, strict=True) -> \
        List[EncodingMismatchInfo]:
    """
    Loads the saved encodings to quant sim model. The encoding filename to load should end in .encodings,
    generated as part of quantsim export.

    :param quant_sim_model: Quantized model to load encodings for. Note: The model configuration should be the same as
        when encodings were exported.
    :param onnx_encoding_path: Path of the encodings file to load.
    :param strict: If set to True and encoding settings between encodings to load do not line up with Quantsim
        initialized settings, an assertion will be thrown. If set to False, quantizer settings will update to align with
        encodings to load.
    :return: List of EncodingMismatchInfo objects containing quantizer names and mismatched settings
    """
    mismatched_encodings = []

    # Load encodings file
    with open(onnx_encoding_path) as json_file:
        encodings = json.load(json_file)

    validate_encodings_to_load(encodings, quant_sim_model)

    # First pass through quantizers to check for mismatched encodings
    for quantizer_name, quantizer in quant_sim_model.qc_quantize_op_dict.items():
        if quantizer_name not in encodings['activation_encodings'] and \
                quantizer_name not in encodings['param_encodings']:
            mismatched_info = get_encoding_mismatch_info(quantizer_name, quantizer, None)
            if mismatched_info.has_mismatch():
                mismatched_encodings.append(mismatched_info)
            continue

        if quantizer_name in encodings['activation_encodings']:
            encodings_to_load = encodings['activation_encodings'][quantizer_name]
        else:
            encodings_to_load = encodings['param_encodings'][quantizer_name]

        mismatched_info = get_encoding_mismatch_info(quantizer_name, quantizer, encodings_to_load)
        if mismatched_info.has_mismatch():
            mismatched_encodings.append(mismatched_info)

    log_and_catch_mismatched_encodings(mismatched_encodings, strict)

    # Second pass through quantizers to set quantizer settings
    for quantizer_name, quantizer in quant_sim_model.qc_quantize_op_dict.items():
        if quantizer_name not in encodings['activation_encodings'] and \
                quantizer_name not in encodings['param_encodings']:
            quantizer.enabled = False
            continue

        if quantizer_name in encodings['activation_encodings']:
            encodings_to_load = encodings['activation_encodings'][quantizer_name]
        else:
            encodings_to_load = encodings['param_encodings'][quantizer_name]

        is_symmetric, is_strict_symmetric, is_unsigned_symmetric = \
            get_symmetric_properties(encodings_to_load)
        data_type = QuantizationDataType.int if encodings_to_load[0]['dtype'] == 'int' else \
                QuantizationDataType.float
        libpymo_encodings = _create_libpymo_encodings(encodings_to_load)
        quant_sim_model.qc_quantize_op_dict[quantizer_name].update_quantizer_and_load_encodings(
            libpymo_encodings, is_symmetric, is_strict_symmetric, is_unsigned_symmetric, data_type)

    return mismatched_encodings


def validate_encodings_to_load(encodings_to_load: Dict, quant_sim_model: QuantizationSimModel):
    """
    Validate that all names of encodings to load are found in the model.

    :param encodings_to_load: Encodings to load
    :param quant_sim_model: Quantsim model to check for encoding names.
    """
    # Check that all encoding names in the encodings to load are found in the model. This check only works for verifying
    # that names in encodings_to_load are valid. The reverse check will not work, since quantizers which are disabled
    # will not show up in encodings_to_load.
    encoding_names_not_found = []
    for quantizer_name in (list(encodings_to_load['activation_encodings'].keys()) +
                           list(encodings_to_load['param_encodings'].keys())):
        if quantizer_name not in quant_sim_model.qc_quantize_op_dict:
            encoding_names_not_found.append(quantizer_name)
    if encoding_names_not_found:
        logger.error('The following encoding names were present in the encodings to load but not found in the model: '
                     '%s', str(encoding_names_not_found))
        raise AssertionError('The following encoding names were present in the encodings to load but not found in the '
                             'model: ' + str(encoding_names_not_found))


def log_and_catch_mismatched_encodings(mismatched_encodings: List[EncodingMismatchInfo], strict: bool):
    """
    If mismatched_encodings is not empty, log details for each entry. If strict is True, raise an AssertionError.

    :param mismatched_encodings: List of mismatched quantizer names and encoding settings
    :param strict: If True, raise an AssertionError if there are mismatched settings
    """
    if mismatched_encodings:
        logging_strings = ['The following quantizers had settings not matching with provided encodings to load:']
        for mismatched_encoding_info in mismatched_encodings:
            logging_strings.append(mismatched_encoding_info.quantizer_name + ':')
            if mismatched_encoding_info.enabled_mismatch:
                logging_strings.append(f'\tenabled: {mismatched_encoding_info.enabled_mismatch[0]}, '
                                       f'loaded encoding enabled: '
                                       f'{mismatched_encoding_info.enabled_mismatch[1]}')

            if mismatched_encoding_info.dtype_mismatch:
                logging_strings.append(f'\tdtype: {mismatched_encoding_info.dtype_mismatch[0]}, '
                                       f'loaded encoding dtype: '
                                       f'{mismatched_encoding_info.dtype_mismatch[1]}')

            if mismatched_encoding_info.bitwidth_mismatch:
                logging_strings.append(f'\tbitwidth: '
                                       f'{mismatched_encoding_info.bitwidth_mismatch[0]}, loaded encoding bitwidth:'
                                       f'{mismatched_encoding_info.bitwidth_mismatch[1]}')

            if mismatched_encoding_info.is_symmetric_mismatch:
                logging_strings.append(f'\tsymmetric: '
                                       f'{mismatched_encoding_info.is_symmetric_mismatch[0]}, '
                                       f'loaded encoding symmetric: '
                                       f'{mismatched_encoding_info.is_symmetric_mismatch[1]}')

            if mismatched_encoding_info.is_strict_symmetric_mismatch:
                logging_strings.append(f'\tstrict symmetric: '
                                       f'{mismatched_encoding_info.is_strict_symmetric_mismatch[0]}, '
                                       f'loaded encoding strict symmetric: '
                                       f'{mismatched_encoding_info.is_strict_symmetric_mismatch[1]}')

            if mismatched_encoding_info.is_unsigned_symmetric_mismatch:
                logging_strings.append(f'\tunsigned symmetric: '
                                       f'{mismatched_encoding_info.is_unsigned_symmetric_mismatch[0]}, '
                                       f'loaded encoding unsigned symmetric: '
                                       f'{mismatched_encoding_info.is_unsigned_symmetric_mismatch[1]}')
        log_message = '\n'.join(logging_strings)
        if strict:
            logger.error(log_message)
            raise AssertionError(log_message)
        logger.info(log_message)


def _create_libpymo_encodings(encoding: Dict[str, Union[str, int, float]]) -> List[libpymo.TfEncoding]:
    """
    Given encoding dict, return a TfEncoding object with corresponding info.

    :param encoding: Encoding dict to create TfEncoding object with
    :return: TfEncoding object containing encoding dict info
    """
    libpymo_encodings = []
    for enc_val in encoding:
        enc = libpymo.TfEncoding()
        enc.bw = enc_val['bitwidth']
        enc.delta, enc.max, enc.min, enc.offset = 0.0, 0.0, 0.0, 0
        if enc_val['dtype'] == 'int':
            enc.delta, enc.max, enc.min, enc.offset = (enc_val['scale'], enc_val['max'], enc_val['min'],
                                                       enc_val['offset'])
        libpymo_encodings.append(enc)
    return libpymo_encodings


def get_symmetric_properties(encodings: List[Dict]) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    """
    Return symmetric properties of the given encodings. If encodings are float, return None for each.

    :param encodings: Encodings to get symmetric properties for
    :return: Tuple of is_symmetric, is_strict_symmetric, and is_unsigned symmetric properties
    """
    if encodings[0]['dtype'] == 'float':
        return None, None, None

    is_symmetric = encodings[0]['is_symmetric'] == 'True'

    is_strict_symmetric = False
    if is_symmetric and encodings[0]['offset'] == -2**(encodings[0]['bitwidth'] - 1) + 1:
        is_strict_symmetric = True

    # Note: Even if the original quantizer had is_unsigned_symmetric set to True, if any observed values were negative,
    # the resulting encodings will look signed. This logic can only perform a best effort check to return True only if
    # any encoding showed unsigned symmetric properties.
    is_unsigned_symmetric = False
    if is_symmetric:
        for encoding in encodings:
            if encoding['offset'] == 0:
                is_unsigned_symmetric = True
                break
    return is_symmetric, is_strict_symmetric, is_unsigned_symmetric

def get_encoding_mismatch_info(quantizer_name: str, quantizer: QcQuantizeOp,
                               encodings_to_load: Optional[List[Dict]]) -> EncodingMismatchInfo:
    """
    Check that quantizer settings align with the settings in encodings_to_load. If settings do not align, track the
    mismatching settings in a EncodingMismatchInfo object and add it to mismatched_encodings_info list.

    :param quantizer_name: Name of quantizer to check
    :param quantizer: Quantizer to check
    :param encodings_to_load: Encodings to check
    """
    encoding_mismatch_info = EncodingMismatchInfo(quantizer_name)

    # Match enabled state
    if quantizer.enabled and encodings_to_load is None:
        encoding_mismatch_info.enabled_mismatch = (quantizer.enabled, False)
    if not quantizer.enabled and encodings_to_load is not None:
        encoding_mismatch_info.enabled_mismatch = (quantizer.enabled, True)

    if encodings_to_load is not None:
        is_symmetric, is_strict_symmetric, is_unsigned_symmetric = get_symmetric_properties(encodings_to_load)

        if quantizer.bitwidth != encodings_to_load[0]['bitwidth']:
            encoding_mismatch_info.bitwidth_mismatch = (quantizer.bitwidth, encodings_to_load[0]['bitwidth'])
        if quantizer.data_type.name != encodings_to_load[0]['dtype']:
            encoding_mismatch_info.dtype_mismatch = (quantizer.data_type.name, encodings_to_load[0]['dtype'])
        if quantizer.use_symmetric_encodings != is_symmetric:
            encoding_mismatch_info.is_symmetric_mismatch = (quantizer.use_symmetric_encodings, is_symmetric)
        if quantizer.use_strict_symmetric != is_strict_symmetric:
            encoding_mismatch_info.is_strict_symmetric_mismatch = (quantizer.use_strict_symmetric, is_strict_symmetric)

        # Unsigned symmetric is a special case because even if the setting is true, the encodings may appear to be
        # signed symmetric if any observed tensor values were < 0.
        # In this case, only mark a mismatch if quantizer was set to signed symmetric but an unsigned symmetric
        # encoding was seen.
        if quantizer.use_unsigned_symmetric != is_unsigned_symmetric and not quantizer.use_unsigned_symmetric:
            encoding_mismatch_info.is_unsigned_symmetric_mismatch = (quantizer.use_unsigned_symmetric,
                                                                     is_unsigned_symmetric)

    return encoding_mismatch_info
