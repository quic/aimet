# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

from typing import List, Union, Dict, Callable, Any, Tuple
import os
import shutil
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from packaging import version

import aimet_common.libpymo as libpymo
import aimet_common.libaimet_tf_ops as qcops
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_common.quantsim import calculate_delta_offset, encoding_version, validate_quantsim_inputs, \
    recompute_grid_params, extract_global_quantizer_args
from aimet_common.quant_utils import get_conv_accum_bounds
from aimet_common.utils import AimetLogger, save_json_yaml
from aimet_tensorflow import graph_editor
from aimet_tensorflow.utils.common import update_variables_with_values, save_data_to_pickle_file, \
    load_data_from_pickle_file, get_valid_ops
from aimet_tensorflow import utils
from aimet_tensorflow.utils import transformer_utils
from aimet_tensorflow.utils.constants import QuantizeOpIndices
from aimet_tensorflow.utils.op.embedding import get_embedding_params_using_patterns
from aimet_tensorflow.utils.quantsim import create_op_to_quant_ops_dict, is_op_quantizable, \
    get_time_steps_tensor_from_rnn_inner_ops, create_encoding_from_dict, swap_last_two_dim
from aimet_tensorflow.utils.graph import updated_graph_flow_context_to_loop_context, set_graph_flow_context, \
    op_not_in_loop_control_flow_context
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.defs import ParameterInfo, AxisHandling
from aimet_tensorflow.quantizer_info import QuantizerInfo, QuantizerType, quant_scheme_to_libpymo
from aimet_tensorflow.quantsim_config.quantsim_config import QuantSimConfigurator
from aimet_tensorflow.quantsim_recurrent import _select_simple_rnn_internal_ops_to_quantize, \
    _select_lstm_internal_ops_to_quantize, SUPPORTED_RECURRENT_TYPES

# this is required to associate gradient with QcQuantize op
from aimet_tensorflow import quantsim_straight_through_grad      # pylint: disable=unused-import


# pylint: disable=too-many-lines

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
WORKING_DIR = '/tmp/quantsim/'


# Op types which we will not place quantize ops after
op_types_to_ignore = {'branch', 'Flatten', 'Shape', 'Identity', 'Reshape', 'Transpose', 'ResourceGather', 'Tile'}

# Connected graph types to ignore parameter quantization
param_quant_conn_op_ignore_list = {'FusedBatchNorm', 'FusedBatchNormV3', 'BatchNorm'}

DTYPES_QUANTIZE_NOT_REQUIRED = [tf.dtypes.int8, tf.dtypes.uint8, tf.dtypes.int16, tf.dtypes.uint16,
                                tf.dtypes.int32, tf.dtypes.uint32, tf.dtypes.int64, tf.dtypes.uint64,
                                tf.bool, tf.dtypes.string]

class PickleableQuantSimState:
    """
    State variables to be saved while pickling
    """
    def __init__(self, quant_scheme, rounding_mode, use_cuda,
                 param_quantizer_dict, activation_quantizer_dict):
        """
        class type to save pickle-able info pertaining to quantsim config
        :param quant_scheme: quant scheme
        :param rounding_mode: rounding mode
        :param use_cuda: flag to indicate usage of GPU
        :param param_quantizer_dict: param quantizers dictionary
        :param activation_quantizer_dict: activation quantizers dictionary
        """

        self.quant_scheme = quant_scheme
        self.rounding_mode = rounding_mode
        self.use_cuda = use_cuda
        self.param_quantizers = param_quantizer_dict
        self.activation_quantizers = activation_quantizer_dict


class QuantizationSimModel:

    """
    Creates a QuantSim model by adding quantization simulations ops to a given model.

    This enables

    #. off-target simulation of inference accuracy
    #. the model to be fine-tuned to counter the effects of quantization

    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(self, session: tf.compat.v1.Session, starting_op_names: List[str], output_op_names: List[str],
                 quant_scheme: Union[str, QuantScheme] = 'tf_enhanced', rounding_mode: str = 'nearest',
                 default_output_bw: int = 8, default_param_bw: int = 8, use_cuda: bool = True, config_file: str = None,
                 default_data_type: QuantizationDataType = QuantizationDataType.int):
        """
        :param session: The input model as session to add quantize ops to
        :param starting_op_names: List of starting op names of the model
        :param output_op_names: List of output op names of the model
        :param quant_scheme: Quantization Scheme, currently supported schemes are post_training_tf and
               post_training_tf_enhanced, defaults to post_training_tf_enhanced
        :param rounding_mode: The round scheme to used. One of: 'nearest' or 'stochastic', defaults to 'nearest'.
        :param default_output_bw: bitwidth to use for activation tensors, defaults to 8
        :param default_param_bw: bitwidth to use for parameter tensors, defaults to 8
        :param use_cuda: If True, places quantization ops on GPU. Defaults to True
        :param config_file: Path to a config file to use to specify rules for placing quant ops in the model
        :param default_data_type: Default data type to use for quantizing all layer parameters.
                                 Possible options are QuantizationDataType.int and QuantizationDataType.float.
                                 Note that the mode default_data_type=QuantizationDataType.float is only supported with
                                 default_output_bw=16 and default_param_bw=16

        :returns: An object which can be used to perform quantization on a tensorflow graph
        :raises: ValueError: An error occurred processing one of the input parameters.

        """
        # sanity checks
        validate_quantsim_inputs(quant_scheme,
                                 rounding_mode,
                                 default_output_bw,
                                 default_param_bw,
                                 default_data_type)

        self.session = session

        if isinstance(quant_scheme, str):
            quant_scheme_lookup = {'tf': QuantScheme.post_training_tf,
                                   'tf_enhanced': QuantScheme.post_training_tf_enhanced}
            quant_scheme = quant_scheme_lookup[quant_scheme]
        self._quant_scheme = quant_scheme
        self._rounding_mode = rounding_mode
        self._use_cuda = use_cuda
        self._param_quantizers = {}
        self._activation_quantizers = {}
        self._default_output_bw = default_output_bw
        self._default_param_bw = default_param_bw
        self._op_to_quant_ops_dict = {}
        self.connected_graph = ConnectedGraph(self.session.graph, starting_op_names, output_op_names)

        # We save a copy of the original model (to be used during export later)
        with self.session.graph.as_default():
            saver = tf.compat.v1.train.Saver()
        saver.save(self.session, save_path=WORKING_DIR+'orig_model_before_quantsim')
        self._quantsim_configurator = QuantSimConfigurator(session, self.connected_graph, config_file, default_output_bw,
                                                           default_param_bw, default_data_type)
        self._supported_kernels = self._quantsim_configurator.get_supported_kernels()
        self.per_channel_quantization_enabled = self._quantsim_configurator.per_channel_quantization_flag
        self._op_name_to_output_channels_axis_handling_dict = {}

        self.quant_args = extract_global_quantizer_args(quant_scheme, self._quantsim_configurator)

        with self.session.graph.as_default():
            self._add_and_configure_quant_nodes(starting_op_names, output_op_names, default_param_bw, default_output_bw,
                                                default_data_type)

        self._override_quant_config_for_transformer_mask_add()
        # Save and load the session so the graph changes can take effect
        self._save_and_load_sim_model()

    def __getstate__(self):
        # convert object to pickle-able state
        state = PickleableQuantSimState(self._quant_scheme, self._rounding_mode,
                                        self._use_cuda, self._param_quantizers,
                                        self._activation_quantizers)
        return state

    def __setstate__(self, state):
        self.session = None
        self._quant_scheme = state.quant_scheme
        self._rounding_mode = state.rounding_mode
        self._use_cuda = state.use_cuda
        self._param_quantizers = state.param_quantizers
        self._activation_quantizers = state.activation_quantizers

    def quantizer_config(self, quant_op_name: str) -> Union[QuantizerInfo, None]:
        """
        gets QuantizerInfo associated with given quantize op
        :param quant_op_name: Name of the Quantize op
        :return: QuantizerInfo associated with the Quant op
        """

        if quant_op_name in self._param_quantizers:
            return self._param_quantizers[quant_op_name]

        if quant_op_name in self._activation_quantizers:
            return self._activation_quantizers[quant_op_name]

        _logger.error('Could not find  Quantizer for given op {%s} ', quant_op_name)
        return None

    def get_supported_kernels(self) -> Dict:
        """
        Return _supported_kernels parsed from the config file
        :return: Dictionary containing supported_kernels
        """
        return self._supported_kernels

    def _get_op_variable_value(self, quant_op: tf.Operation, var_index: int):
        """
        Utility to load variable values from quant op
        :param quant_op: quantize op
        :param var_index: variable index to be read
        :return: variable value
        """

        op_var_tensor = quant_op.inputs[var_index]
        return self.session.run(op_var_tensor)

    def configure_quantization_ops(self, conn_graph: ConnectedGraph, ops_with_param_names: List[str], indices: List[int],
                                   params_to_quantize: Dict[str, ParameterInfo], activation_op_names: List[str]):
        """
        Configure inserted quantize ops using config file
        :param conn_graph: Connected graph of the model
        :param ops_with_param_names: List of ops for which param quantization ops were inserted for
        :param indices: List of input indices (one-to-one for each entry in ops)
        :param params_to_quantize: Dictionary of parameters to quantize
        :param activation_op_names: List of ops for which activation quantization ops were inserted for
        """
        if not conn_graph:
            error_msg = (f'Connected graph passed into configure_quantization_ops() is None. If manual insertion of '
                         f'quantization ops is being done, and get_ops_to_quantize_activations_for() has been '
                         f'overriden, please override configure_quantization_ops() as well.')
            _logger.error(error_msg)
            raise AssertionError(error_msg)
        self._op_to_quant_ops_dict = create_op_to_quant_ops_dict(self.session.graph, conn_graph, ops_with_param_names, indices,
                                                                 params_to_quantize, activation_op_names)
        self._quantsim_configurator.configure_quantizers(self._op_to_quant_ops_dict, self._param_quantizers,
                                                         self._activation_quantizers)

    def compute_encodings(self, forward_pass_callback: Callable[[tf.compat.v1.Session, Any], None],
                          forward_pass_callback_args):
        """
        Computes encodings for all quantization sim nodes in the model.
        This is also used to set initial encodings for Range Learning.

        :param forward_pass_callback: A callback function that is expected to runs forward passes on a session.
               This callback function should use representative data for the forward pass, so the calculated
               encodings work for all data samples. This callback internally chooses the number of data samples
               it wants to use for calculating encodings.

        :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
               the user to determine the type of this parameter. E.g. could be simply an integer representing the number
               of data samples to use. Or could be a tuple of parameters or an object representing something more
               complex.

        :return: None

        """

        self._compute_and_set_parameter_encodings()

        # At the beginning before we do forward pass we want to set parameters to quantize dequantize mode and once we
        # compute the encodings for activations we set it to the required op mode based on quant scheme & if per channel
        # quantization is enabled
        self._set_op_mode_parameters(libpymo.TensorQuantizerOpMode.quantizeDequantize, [])

        ops_with_invalid_encodings = []

        # Run data through the quantsim so we can compute activation encodings
        forward_pass_callback(self.session, forward_pass_callback_args)

        # For activations, calculate encodings and update min-max parameters
        for op_name, quantizer_info in self._activation_quantizers.items():
            # Calculate encodings
            if quantizer_info.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
                op_bitwidth, op_use_symmetric_encodings = quantizer_info.bitwidth, quantizer_info.use_symmetric_encoding
                encoding = quantizer_info.compute_encoding(op_bitwidth, op_use_symmetric_encodings)
                # encoding would be invalid for dtype=fp because there is no encoding computed in float mode through the
                # tensor_quantizer
                if quantizer_info.data_type == QuantizationDataType.float:
                    quantizer_info.set_op_mode(libpymo.TensorQuantizerOpMode.quantizeDequantize)
                else:
                    if quantizer_info.is_encoding_valid():
                        quantizer_info.set_encoding(encoding)
                        quantizer_info.set_op_mode(libpymo.TensorQuantizerOpMode.quantizeDequantize)
                    else:
                        quantizer_info.set_op_mode(libpymo.TensorQuantizerOpMode.passThrough)
                        ops_with_invalid_encodings.append(op_name)

        # For post-training mode, params will always be in one-shot mode
        op_mode = self._param_op_mode_after_analysis(self._quant_scheme)

        self._set_op_mode_parameters(op_mode, ops_with_invalid_encodings)

        if ops_with_invalid_encodings:
            _logger.info('The following quantizers did not have valid encodings and have been set to passThrough mode: '
                         '%s', ops_with_invalid_encodings)
            _logger.info('This can be due to the quantizers not having been evaluated during the forward pass in '
                         'compute encodings. Evaluation is required to collect statistics needed to compute valid '
                         'encodings.\n'
                         'As a result, the quantizers have been set to passThrough mode, meaning no quantization noise '
                         'will be simulated for these ops if they are evaluated in the future.\n'
                         'If this is not desired, amend the forward pass to evaluate tensors which require these ops '
                         'to be evaluated, and recompute encodings.')

        self._clamp_transformer_attention_mask_encoding()

    def get_enabled_parameter_quantizers(self):
        """
        For given quantsim model, get all enabled param quantizers.
        :return: List of enabled param quantizers.
        """
        enabled_param_quantizers = []
        for quantizer_info in self._param_quantizers.values():
            if quantizer_info.enabled:
                enabled_param_quantizers.append(quantizer_info)
        return enabled_param_quantizers

    def get_enabled_activation_quantizers(self):
        """
        For given quantsim model, get all enabled activation quantizers.
        :return: List of enabled activation quantizers.
        """
        enabled_activation_quantizers = []
        for quantizer_info in self._activation_quantizers.values():
            if quantizer_info.enabled:
                enabled_activation_quantizers.append(quantizer_info)
        return enabled_activation_quantizers

    def _set_op_mode_parameters(self, op_mode: libpymo.TensorQuantizerOpMode,
                                ops_with_invalid_encodings: List):
        """
        Sets op mode for parameters and if the encodings are invalid, then adds those ops to ops_with_invalid_encodings
        :param op_mode: libpymo.TensorQuantizerOpMode
        :param ops_with_invalid_encodings: list of ops that don't have vallid encodings
        """
        for op_name, quantizer_info in self._param_quantizers.items():
            if quantizer_info.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
                # encoding would be invalid for dtype=fp because there is no encoding computed in float mode through the
                # tensor_quantizer
                if quantizer_info.data_type == QuantizationDataType.float:
                    quantizer_info.set_op_mode(libpymo.TensorQuantizerOpMode.quantizeDequantize)
                else:
                    if quantizer_info.is_encoding_valid():
                        quantizer_info.set_op_mode(op_mode)
                    else:
                        quantizer_info.set_op_mode(libpymo.TensorQuantizerOpMode.passThrough)
                        ops_with_invalid_encodings.append(op_name)

    def export(self, path: str, filename_prefix: str, orig_sess: tf.compat.v1.Session = None):
        """
        This method exports out the quant-sim model so it is ready to be run on-target.

        Specifically, the following are saved

        1. The sim-model is exported to a regular tensorflow meta/checkpoint without any simulation ops

        2. The quantization encodings are exported to a separate JSON-formatted file that can
           then be imported by the on-target runtime (if desired)

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param orig_sess: optional param to pass in original session without quant nodes for export
        :return: None

        """
        # Recompute encodings before export for parameters
        self._compute_and_set_parameter_encodings()
        # save session without quant nodes
        if orig_sess is not None:
            with orig_sess.graph.as_default():
                saver = tf.compat.v1.train.Saver()
            saver.save(orig_sess, save_path=WORKING_DIR+'orig_model_before_quantsim')
        else:
            _logger.info('Original session is not provided, use orig_model_before_quantsim.meta to export')

        self._remove_quantization_nodes_and_save_graph(path, filename_prefix)
        self._export_encodings(os.path.join(path, filename_prefix) + '.encodings')

    def _compute_and_set_parameter_encodings(self):

        for quantizer_info in self._param_quantizers.values():

            if quantizer_info.enabled and quantizer_info.data_type == QuantizationDataType.int:
                # 0th input to our quant op is the tensor being quantized - in this case the parameter tensor
                weight_tensor = quantizer_info.get_variable_from_op(0)

                # Per-channel
                if isinstance(quantizer_info.tensor_quantizer, list):
                    for index, tensor_quantizer in enumerate(quantizer_info.tensor_quantizer):
                        if quantizer_info.axis_handling == AxisHandling.LAST_TWO_AXES:
                            last_two_axes_combined_shape = list(weight_tensor.shape[:-2]) + [-1]
                            channel_slice = weight_tensor.reshape(*last_two_axes_combined_shape)
                            channel_slice = channel_slice.take(index, channel_slice.ndim - 1)
                            tensor_quantizer.updateStats(channel_slice, False)
                        else:
                            channel_slice = weight_tensor.take(index, weight_tensor.ndim - 1)
                            tensor_quantizer.updateStats(channel_slice, False)

                # Per-tensor
                else:
                    tensor_quantizer = quantizer_info.tensor_quantizer
                    tensor_quantizer.updateStats(weight_tensor, False)

                encoding = quantizer_info.compute_encoding(quantizer_info.bitwidth,
                                                           quantizer_info.use_symmetric_encoding)

                quantizer_info.set_encoding(encoding)

    def _remove_quantization_nodes_and_save_graph(self, path: str, filename_prefix: str):
        """
        This function removes the quantization nodes from quantized graph and saves it
        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        """
        vars_to_save = []
        with self.session.graph.as_default():
            for var in tf.compat.v1.global_variables():
                if not var.name[:-2].endswith(('_quantized', '_quantized_op_mode', '_quantized_quant_ref',
                                               '_quantized_encoding_min', '_quantized_encoding_max',
                                               '_quantized_bit_width', '_quantized_use_symmetric_encoding',
                                               '_quantized_axis_handling', '_quantized_data_type')):
                    vars_to_save.append(var)

            saver = tf.compat.v1.train.Saver(vars_to_save)
            saver.save(self.session, save_path=os.path.join(path, filename_prefix))
            shutil.copyfile(WORKING_DIR + 'orig_model_before_quantsim.meta',
                            os.path.join(path, filename_prefix) + '.meta')

    def save_to_keras(self, temp_dir_path: str = "/tmp/") -> tf.compat.v1.Session:
        """
        This method exports out the quant-sim model so it is ready to be eval/trained using a Keras pipeline

        :param temp_dir_path: temporary directory to store intermediate files
        :return: Session to import into a Keras model

        """
        current_graph = self.session.graph
        with current_graph.as_default():
            ops = current_graph.get_operations()
            for op in ops:
                if op.type in ['QcQuantize', 'QcQuantizeRecurrentParam']:

                    # Read the config
                    # -----------------
                    quant_config = self.quantizer_config(op.name)
                    config_tuple = self.session.run([op.inputs[QuantizeOpIndices.op_mode],
                                                     op.inputs[QuantizeOpIndices.encoding_min],
                                                     op.inputs[QuantizeOpIndices.encoding_max],
                                                     op.inputs[QuantizeOpIndices.bit_width],
                                                     op.inputs[QuantizeOpIndices.use_symmetric_encoding]])
                    op_mode, encoding_min, encoding_max, bitwidth, is_symmetric = config_tuple

                    # Create the static op
                    # --------------------
                    if not self._use_cuda:
                        with tf.device('/cpu:0'):
                            static_op = qcops.qc_quantize_static(name=op.name+"_static", in_tensor=op.inputs[0],
                                                                 encoding_min=encoding_min, encoding_max=encoding_max,
                                                                 bitwidth=bitwidth, quant_scheme=quant_config.quant_scheme,
                                                                 op_mode=op_mode, is_symmetric=bool(is_symmetric))
                    else:
                        static_op = qcops.qc_quantize_static(name=op.name + "_static", in_tensor=op.inputs[0],
                                                             encoding_min=encoding_min, encoding_max=encoding_max,
                                                             bitwidth=bitwidth, quant_scheme=quant_config.quant_scheme,
                                                             op_mode=op_mode, is_symmetric=bool(is_symmetric))

                    # Replace in graph
                    # -----------------
                    graph_editor.reroute_ts(ts0=[static_op], ts1=[op.outputs[0]],
                                            can_modify=op.outputs[0].consumers())
                    graph_editor.detach_inputs(op)

        new_sess = utils.graph_saver.save_and_load_graph(temp_dir_path, self.session)
        return new_sess

    def save_model_with_embedded_quantization_nodes(self, checkpoint_path: str, encoding_path: str = None,
                                                    orig_sess: tf.compat.v1.Session = None):
        """
        This method is to export model embedded with native tensorflow quantization nodes
        :param checkpoint_path: path to save the checkpoint files
        :param encoding_path: optional param to pass the path from where to load parameter encodings file
        :param orig_sess: optional param to pass in original session without quant nodes
        """
        # Load encodings file
        encodings_dicts = {}
        if encoding_path and os.path.exists(encoding_path):
            with open(encoding_path) as json_file:
                encodings_dicts = json.load(json_file)
                encodings_dicts = dict(encodings_dicts["activation_encodings"], **encodings_dicts["param_encodings"])

        if orig_sess is None:
            _logger.info('Original session is not provided, use orig_model_before_quantsim.meta as default graph')
            orig_sess = utils.graph_saver.load_model_from_meta(meta_path=os.path.join(WORKING_DIR,
                                                                                      'orig_model_before_quantsim' + '.meta'))
        with orig_sess.graph.as_default():
            for op_name, quantizer_info in dict(self._param_quantizers, **self._activation_quantizers).items():
                tensor_name = self.session.graph.get_operation_by_name(op_name).inputs[0].name
                op = orig_sess.graph.get_tensor_by_name(tensor_name).op
                consumers = [consumer for consumer in op.outputs[0].consumers() if 'gradients' not in consumer.name]
                if tensor_name in encodings_dicts:
                    # Check for per channel quantization
                    if self.per_channel_quantization_enabled and len(encodings_dicts[tensor_name]) > 1:
                        encoding_min = [channel_dict['min'] for channel_dict in encodings_dicts[tensor_name]]
                        encoding_max = [channel_dict['max'] for channel_dict in encodings_dicts[tensor_name]]
                        encoding_bw = encodings_dicts[tensor_name][0]['bitwidth']
                    else:
                        encoding_max = encodings_dicts[tensor_name][0].get('max')
                        encoding_min = encodings_dicts[tensor_name][0].get('min')
                        encoding_bw = encodings_dicts[tensor_name][0].get('bitwidth')

                else:
                    if not quantizer_info.is_encoding_valid():
                        if  quantizer_info.data_type == QuantizationDataType.float and quantizer_info.get_op_mode() in\
                            [int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize),
                             int(libpymo.TensorQuantizerOpMode.quantizeDequantize)]:
                            # Cast input tensor to data_type and dequant it to fp32
                            with tf.device('' if self._use_cuda else '/cpu:0'):
                                tf_quantization_op = tf.cast(tf.cast(op.outputs[0], tf.float16), tf.float32)
                            # Replace in graph
                            # -----------------
                            graph_editor.reroute_ts(ts0=tf_quantization_op, ts1=[op.outputs[0]],
                                                    can_modify=consumers)
                        continue
                    _logger.info("Can't find %s in encodings file, encodings in QuantizationSimModel will be used",
                                 self._get_quantized_name(op.name))
                    encoding_min, encoding_max = self.read_min_max(self._get_quantized_name(op.name))
                    # if per channel quantization is enabled, then min and max are numpy arrays, and this function gates the array
                    encoding_bw = int(self._get_op_variable_value(self.session.graph.get_operation_by_name(op_name),
                                                                  QuantizeOpIndices.bit_width))

                _logger.info("Adding native tensorflow quantization op %s", self._get_quantized_name(op.name))
                # inser native tensorflow quantization nodes into graph
                with tf.device('' if self._use_cuda else '/cpu:0'):
                    if not isinstance(encoding_max, (list, np.ndarray)):
                        tf_quantization_op = \
                            tf.quantization.fake_quant_with_min_max_vars(op.outputs[0], min=encoding_min, max=encoding_max,
                                                                         num_bits=encoding_bw, narrow_range=False,
                                                                         name=self._get_quantized_name(op.name))
                    else:
                        tf_quantization_op = \
                            tf.quantization.fake_quant_with_min_max_vars_per_channel(op.outputs[0], min=np.array(encoding_min),
                                                                                     max=np.array(encoding_max), num_bits=encoding_bw,
                                                                                     narrow_range=False, name=self._get_quantized_name(op.name))

                # Replace in graph
                # -----------------
                graph_editor.reroute_ts(ts0=tf_quantization_op, ts1=[op.outputs[0]],
                                        can_modify=consumers)

            utils.graph_saver.save_model_to_meta(orig_sess, os.path.join(checkpoint_path + '_embedded_quant_nodes'))
            return utils.graph_saver.load_model_from_meta(meta_path=str(checkpoint_path + '_embedded_quant_nodes.meta'))

    def set_and_freeze_param_encodings(self, encoding_path: str):
        """
        Set and freeze parameter encodings from encodings JSON file
        :param encoding_path: path from where to load parameter encodings file
        """
        # Load parameter encodings file
        with open(encoding_path) as json_file:
            param_encodings = json.load(json_file)

        # op mode will be Quantize dequantize
        op_mode = libpymo.TensorQuantizerOpMode.quantizeDequantize

        for op_name, quantizer_info in self._param_quantizers.items():
            quant_op = self.session.graph.get_operation_by_name(op_name)
            tensor_name = quant_op.inputs[0].name
            if tensor_name in param_encodings:
                encoding_dict = param_encodings[tensor_name] if self.per_channel_quantization_enabled else \
                    param_encodings[tensor_name][0]
                encoding, is_symmetric = create_encoding_from_dict(encoding_dict)
                quantizer_info.use_symmetric_encoding = is_symmetric
                quantizer_info.set_and_freeze_encoding_and_op_mode(encoding, op_mode)
                _logger.info("Setting and freezing quantization encodings for parameter: %s", tensor_name)

    def _param_op_mode_after_analysis(self, quant_scheme) -> libpymo.TensorQuantizerOpMode:
        """
        Returns op mode to use for parameters after encodings have been computed
        :param quant_scheme: Quantization scheme to use
        :return:
        """
        if quant_scheme in [QuantScheme.training_range_learning_with_tf_init,
                            QuantScheme.training_range_learning_with_tf_enhanced_init]:
            op_mode = libpymo.TensorQuantizerOpMode.quantizeDequantize
        else:
            op_mode = libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize

        if self.per_channel_quantization_enabled:
            op_mode = libpymo.TensorQuantizerOpMode.quantizeDequantize

        return op_mode

    def get_min_max_var_dict(self)-> Dict:
        """
        Fetches all the min max variables in given Quantized graph.
        :return: dictionary of min/ max variable names to var mapping
        """
        variable_dict = {}
        with self.session.graph.as_default():
            for var in tf.compat.v1.global_variables():
                if var.name.endswith('_encoding_min:0') or var.name.endswith('_encoding_max:0'):
                    variable_dict[var.name] = var

        return variable_dict

    def read_min_max(self, quant_op_name: str, variable_dict: Dict = None)-> (float, float):
        """
        Reads min and max params from quantize op
        :param quant_op_name: quantize op name to read min and max variables from.
        :param variable_dict: dictionary of min/max variable names to variable mapping for given quantized graph, optional
        :return: min and max variable values from the given quant op.
        """
        if not variable_dict:
            # get a variable dict if one is not provided
            variable_dict = self.get_min_max_var_dict()

        min_var = variable_dict[quant_op_name + '_encoding_min:0']
        max_var = variable_dict[quant_op_name + '_encoding_max:0']
        return self.session.run([min_var, max_var])

    def _export_encodings(self, encoding_file_path: str):

        variable_dict = self.get_min_max_var_dict()

        def update_encoding_dict_entry_float(encoding_dict: Dict, op_name: str):
            quant_op = self.session.graph.get_operation_by_name(op_name)
            op_bitwidth = int(self._get_op_variable_value(quant_op, QuantizeOpIndices.bit_width))
            if op_bitwidth != 16:
                raise ValueError('dtype is set to float but bitwidth is not 16 for the layer:', op_name)

            tensor_name = quant_op.inputs[0].name
            encoding_dict[tensor_name] = [{'dtype': 'float',
                                           'bitwidth': op_bitwidth}]

        def update_encoding_dict_entry_int(encoding_dict: Dict, quant_op_name: str):
            quant_op = self.session.graph.get_operation_by_name(quant_op_name)
            min_val, max_val = self.read_min_max(quant_op_name, variable_dict)
            # if per channel quantization is enabled, then min and max are numpy arrays, and this function gates the array
            op_bitwidth = int(self._get_op_variable_value(quant_op, QuantizeOpIndices.bit_width))
            delta, offset = calculate_delta_offset(min_val, max_val, op_bitwidth,
                                                   use_symmetric_encodings=False, use_strict_symmetric=False)
            # Min and max will be numpy arrays, so to make them JSON serializable
            if self.per_channel_quantization_enabled and isinstance(min_val, np.ndarray):
                min_val = min_val.tolist()
                max_val = max_val.tolist()
            else:
                # Wrap single min/max value in a list to support list comprehension
                min_val = [min_val]
                max_val = [max_val]
                delta = [delta]
                offset = [offset]
            is_symmetric = str(self._get_op_variable_value(quant_op,
                                                           QuantizeOpIndices.use_symmetric_encoding))

            tensor_name = quant_op.inputs[0].name
            if quant_op.type in ['QcQuantizePerChannel'] and 'EagerPyFunc' in tensor_name:
                tensor_name = quant_op.inputs[0].op.inputs[0].name
            encoding_dict[tensor_name] = [{'min': min_val[idx],
                                           'max': max_val[idx],
                                           'scale': delta[idx],
                                           'offset': offset[idx],
                                           'bitwidth': op_bitwidth,
                                           'is_symmetric': is_symmetric,
                                           'dtype': 'int'} for idx in range(len(min_val))]

        param_encodings = {}
        for quant_op_name, quantizer_info in self._param_quantizers.items():
            if quantizer_info.data_type == QuantizationDataType.float:
                update_encoding_dict_entry_float(param_encodings, quant_op_name)
            else:
                if not quantizer_info.is_encoding_valid():
                    continue
                update_encoding_dict_entry_int(param_encodings, quant_op_name)

        activation_encodings = {}
        for quant_op_name, quantizer_info in self._activation_quantizers.items():
            if quantizer_info.data_type == QuantizationDataType.float:
                update_encoding_dict_entry_float(activation_encodings, quant_op_name)
            else:
                if not quantizer_info.is_encoding_valid():
                    continue
                update_encoding_dict_entry_int(activation_encodings, quant_op_name)

        encodings_dict = {'version': encoding_version,
                          'activation_encodings': activation_encodings,
                          'param_encodings': param_encodings,
                          'quantizer_args': self.quant_args}

        save_json_yaml(encoding_file_path, encodings_dict)

    def _save_and_load_sim_model(self):
        self.session = utils.graph_saver.save_and_load_graph(WORKING_DIR, self.session)
        update_tensor_quantizer_references(self.session, self._activation_quantizers)
        update_tensor_quantizer_references(self.session, self._param_quantizers)

    def _add_quant_nodes_recurrent(self, conn_graph: ConnectedGraph, default_param_bw: int, default_output_bw: int) \
            -> Tuple[List[str], List[int], List[str]]:
        """
        Utility to add quant nodes to recurrent module
        :param conn_graph: Connected graph of the model
        :param default_param_bw: default param bitwidth
        :param default_output_bw: default output bitwidth
        :return: Tuple[List[str], List[int], List[str]], param op names, input indices and activation op names
        """
        # pylint: disable=protected-access
        # pylint: disable=too-many-locals

        # Register custom handlers to select internal ops to quantize in a given recurrent module type
        switcher = {
            "SimpleRNN": _select_simple_rnn_internal_ops_to_quantize,
            "LSTM": _select_lstm_internal_ops_to_quantize
        }

        ops_with_param_names = []
        input_indices = []
        activation_op_names = []

        for op in conn_graph.get_all_ops().values():
            #  we can configure custom layer selectors per recurrent type or use default one
            if op.type in SUPPORTED_RECURRENT_TYPES:
                if version.parse(tf.version.VERSION) >= version.parse("2.00"):
                    raise AssertionError('Recurrent layers are not supported with TF2.x, instead use TF1.15.')
                internal_ops = op.internal_ops

                # select internal ops to quantize in this recurrent type
                select_internal_ops_to_quantize = switcher.get(op.type)
                module_ops_with_param_names, module_op_input_indices, module_activation_op_names = \
                    select_internal_ops_to_quantize(self.session.graph, internal_ops)

                # insert the quant nodes
                self._insert_param_quantization_ops_loop_context(module_ops_with_param_names, module_op_input_indices,
                                                                 default_param_bw, internal_ops)

                self._insert_activation_quantization_ops(module_activation_op_names, default_output_bw,
                                                         in_loop_context=True)

                # if there are multiple recurrent modules, we want a list containing all the param
                # and activation info
                if module_ops_with_param_names and module_op_input_indices:
                    ops_with_param_names.extend(module_ops_with_param_names)
                    input_indices.extend(module_op_input_indices)
                if module_activation_op_names:
                    activation_op_names.extend(module_activation_op_names)

        return ops_with_param_names, input_indices, activation_op_names

    def _add_and_configure_quant_nodes(self, starting_op_names: List[str], output_op_names: List[str],
                                       default_param_bw: int, default_output_bw: int,
                                       default_data_type: QuantizationDataType):
        """
        Utility to add quant nodes
        :param starting_op_names: List of starting op names of the model
        :param output_op_names: List of output op names of the model
        :param default_param_bw: default param bitwidth
        :param default_output_bw: default output bitwidth
        :param default_data_type: Default data type to use for quantizing all layer parameters
        """

        # Get list of ops with params to insert quantizers for, as well as the input indices to insert on.
        params_to_quantize = QuantizationSimModel._get_ops_to_quantize_params_for(self.session.graph,
                                                                                  self.connected_graph,
                                                                                  starting_op_names,
                                                                                  output_op_names)

        # Get list of activation ops to insert quantizers for
        activation_op_names = QuantizationSimModel._get_ops_to_quantize_activations_for(self.session.graph,
                                                                                        self.connected_graph)


        self._insert_param_quantization_ops(params_to_quantize, default_param_bw, data_type=default_data_type)
        self._insert_activation_quantization_ops(activation_op_names, default_output_bw, data_type=default_data_type)

        # this takes care of quant node insertion in loop context of recurrent layer, which makes a cell
        recurrent_ops_with_param_names, recurrent_input_indices, recurrent_activation_op_names = \
            self._add_quant_nodes_recurrent(self.connected_graph, default_param_bw, default_output_bw)

        if recurrent_activation_op_names:
            activation_op_names.extend(recurrent_activation_op_names)

        # Note: at this point, the session used to construct conn_graph is different than the current
        # self.session, however we still use the connected graph to traverse the graph structure.
        self.configure_quantization_ops(self.connected_graph, recurrent_ops_with_param_names, recurrent_input_indices,
                                        params_to_quantize, activation_op_names)

    @staticmethod
    def _get_quantized_name(op_name: str) -> str:
        """
        Small utility function to name a quantized parameter
        :param op_name: Name of the op being quantized
        :return: Returns an appropriate name for the quantized op
        """
        return op_name + '_quantized'

    @staticmethod
    def _get_unquantized_name(quant_op_name: str) -> str:
        """
        Small utility function to get the name of the op being quantized
        :param quant_op_name: Name of the quant op
        :return: Returns the name of the op being quantized
        """
        assert quant_op_name.endswith('_quantized')
        return quant_op_name[:-len('_quantized')]

    @staticmethod
    def _get_op_to_modify_with_param_in(op: tf.Operation, index: int) -> (tf.Operation, tf.Tensor):
        """
        utility to get op to modify along with param input
        :param op: TensorFlow operation
        :param index: input index to get param from
        :return: Tuple of TF operation and param in tensor
        """

        op_to_modify = None
        param_in = None
        # case of params being depth 2 input nodes to MatMul
        # via strided-slice or split op
        if op.inputs[index].op.type in ['StridedSlice', 'Split']:
            strided_slice_op = op.inputs[index].op
            for inp in strided_slice_op.inputs:
                if inp.op.type in ['ReadVariableOp']:
                    op_to_modify = strided_slice_op
                    param_in = inp
        else:
            # case of params being direct input nodes to MatMul
            op_to_modify = op
            param_in = op.inputs[index]

        return op_to_modify, param_in

    def _insert_param_quantization_ops(self, params_to_quantize: Dict[str, ParameterInfo], default_param_bw: int,
                                       data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Inserts quantization ops for individual parameters
        :param params_to_quantize: dictionary of parameters to quantize
        :param default_param_bw : default param bitwidth
        :return: None
        """
        # pylint: disable=too-many-locals
        for param_name, param_info in params_to_quantize.items():
            param_in = self.session.graph.get_operation_by_name(param_name).outputs[0]
            can_modify_ops = [self.session.graph.get_operation_by_name(consumer) \
                              for consumer in param_info.op_with_param_name]
            # Assume all ops that are consumers of the param are of the same type for axis handling purposes
            can_modify_op_type = can_modify_ops[0].type
            if param_in is not None:
                num_output_channels, quantization_axis_handling = \
                    QuantizationSimModel._get_number_of_output_channels_and_quantization_axis_handling(
                        param_in.get_shape().as_list(), can_modify_op_type)
                quant_op_name = self._get_quantized_name(param_name)

                self._op_name_to_output_channels_axis_handling_dict[quant_op_name] = [num_output_channels,
                                                                                      quantization_axis_handling]
                _logger.info("Adding weight quantization op %s", quant_op_name)
                op_mode = libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize

                # If per channel quantization is enabled we tranpose the weights of tranpose op and then
                # perform per channel quantization
                if can_modify_op_type in ['Conv2DTranspose', 'Conv2DBackpropInput'] and \
                        self.per_channel_quantization_enabled:

                    fout = tf.py_function(func=swap_last_two_dim, inp=[param_in], Tout=tf.float32)

                    q_op_out = self._insert_post_training_quant_op(fout, quant_op_name,
                                                                   op_mode, self._param_quantizers, QuantizerType.param,
                                                                   default_param_bw, data_type)

                    q_op_out = tf.py_function(func=swap_last_two_dim, inp=[q_op_out], Tout=tf.float32)
                else:
                    q_op_out = self._insert_post_training_quant_op(param_in, quant_op_name,
                                                                   op_mode, self._param_quantizers, QuantizerType.param,
                                                                   default_param_bw, data_type)

                nodes_modified_count = graph_editor.reroute_ts(tf_ops.convert_to_tensor(q_op_out), param_in,
                                                               can_modify=can_modify_ops)

                if nodes_modified_count != len(can_modify_ops):
                    raise ValueError(f'Issue quantizing {param_in.name}')

    def _insert_param_quantization_ops_loop_context(self, op_names: List[str], indices: List[int],
                                                    default_param_bw: int,
                                                    inner_ops: List[tf.Operation],
                                                    data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Inserts quantization ops for individual parameters
        :param op_names: List of ops whose parameters are being quantized
        :param indices: List of input indices (one-to-one for each entry in ops)
        :param default_param_bw : default param bitwidth
        :param inner_ops: list of tf.Operations inside a RNN op
        :param data_type: Default data type to use for quantizing all layer parameters
        :return: None
        """
        # pylint: disable=too-many-locals
        ops = [self.session.graph.get_operation_by_name(op_name) for op_name in op_names]
        assert len(ops) == len(indices)

        for op, index in zip(ops, indices):
            # Modify the weight/bias inputs to use the quantized inputs
            can_modify_op, param_in = QuantizationSimModel._get_op_to_modify_with_param_in(op, index)

            if param_in is not None:
                num_output_channels, quantization_axis_handling = \
                    QuantizationSimModel._get_number_of_output_channels_and_quantization_axis_handling(
                        can_modify_op.inputs[index].get_shape().as_list(), can_modify_op.type)
                quant_op_name = self._get_quantized_name(param_in.op.name)

                self._op_name_to_output_channels_axis_handling_dict[quant_op_name] = [num_output_channels,
                                                                                      quantization_axis_handling]
                _logger.info("Adding weight quantization op %s", quant_op_name)
                op_mode = libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize

                q_op_out = self._insert_param_quantizer_loop_context(inner_ops, param_in, quant_op_name,
                                                                     op_mode, self._param_quantizers,
                                                                     QuantizerType.param,
                                                                     default_param_bw, data_type)

                nodes_modified_count = graph_editor.reroute_ts(tf_ops.convert_to_tensor(q_op_out), param_in,
                                                               can_modify=can_modify_op)
                if nodes_modified_count != 1:
                    raise ValueError('Input ' + param_in.name + ' not quantized!')


    @staticmethod
    def _get_number_of_output_channels_and_quantization_axis_handling(weight_shape: List[int],
                                                                      consumer_op_type: str) -> \
        Tuple[int, AxisHandling]:
        """
        Gets number of output channels and quantization axis handling for an op for per channel quantization
        :param weight_shape: list containing tensor shape of weight
        :param consumer_op_type: type of op that consumes weight
        :return number of output channel and axis handling from weight_shape
        """
        # Initialize axis_handling and num_output_channels with values fitting most ops
        axis_handling = AxisHandling.LAST_AXIS
        num_output_channels = weight_shape[-1]
        if consumer_op_type in ['Conv2DTranspose', 'Conv2DBackpropInput']:
            num_output_channels = weight_shape[2]
        elif consumer_op_type == 'DepthwiseConv2dNative':
            num_output_channels *= weight_shape[-2]
            axis_handling = AxisHandling.LAST_TWO_AXES

        # If op is not any special op, fall through and return the unmodified values.
        return num_output_channels, axis_handling

    @staticmethod
    def _is_op_quantizable(op: tf.Operation) -> bool:
        """
        utility to check if the quantization can be supported for this op
        :param op: op as tf.Operation type
        :return: True if the op can be quantized, False otherwise
        """

        if op.outputs:
            if op.outputs[0].dtype not in DTYPES_QUANTIZE_NOT_REQUIRED:
                return True

        return False

    def _insert_activation_quantization_ops(self, valid_op_names: List[str], default_output_bw,
                                            in_loop_context: bool = False,
                                            data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Inserts quantization ops at the outputs of given ops
        :param valid_op_names: List of op names to insert activation quantizers for
        :param default_output_bw: default activation bitwidth
        :param in_loop_context: True, if the ops belong to a loop control flow context
        :param data_type: Default data type to use for quantizing all layer activations
        return:
        """
        for op_name in valid_op_names:
            quant_op_name = self._get_quantized_name(op_name)
            op = self.session.graph.get_operation_by_name(op_name)
            _logger.info("Adding activation quantization op %s", quant_op_name)

            consumers = [consumer for consumer in op.outputs[0].consumers() if 'gradients' not in consumer.name]

            if not QuantizationSimModel._is_op_quantizable(op):
                error_msg = f'Unsupported dtype {op.outputs[0].dtype} detected for op {op_name}.'
                _logger.error(error_msg)
                raise AssertionError(error_msg)

            if in_loop_context:
                q_op_out = self._insert_post_training_quant_op_in_loop_context(op.outputs[0], quant_op_name,
                                                                               libpymo.TensorQuantizerOpMode.updateStats,
                                                                               self._activation_quantizers,
                                                                               QuantizerType.activation,
                                                                               default_output_bw, data_type)
            else:
                q_op_out = self._insert_post_training_quant_op(op.outputs[0], quant_op_name,
                                                               libpymo.TensorQuantizerOpMode.updateStats,
                                                               self._activation_quantizers, QuantizerType.activation,
                                                               default_output_bw, data_type)

            # Re-route
            num_rerouted_outputs = graph_editor.reroute_ts(tf_ops.convert_to_tensor(q_op_out),
                                                           op.outputs[0], can_modify=consumers)
            if num_rerouted_outputs != len(consumers):
                raise ValueError('Failed to map ' + str(len(consumers)) + ' quantization output(s). Only mapped ' +
                                 str(num_rerouted_outputs))

    def _create_encoding_min_max_vars(self, q_op_name: str, quantizer_type: QuantizerType = None) -> (tf.Variable, tf.Variable):
        """
        creates encoding min and max variables for quant op.
        :param q_op_name: name of quantize op
        :param quantizer_type: Quantizer type param or activation
        :return: encoding min and max as tf.Variable type
        """

        is_trainable = False
        if self._quant_scheme in [QuantScheme.training_range_learning_with_tf_init,
                                  QuantScheme.training_range_learning_with_tf_enhanced_init]:
            is_trainable = True

        initial_min_val = 0.0
        initial_max_val = 0.0

        if quantizer_type == QuantizerType.param and self.per_channel_quantization_enabled:
            num_output_channels, _ = self._op_name_to_output_channels_axis_handling_dict[q_op_name]
            initial_min_val = [0.0] * num_output_channels
            initial_max_val = [0.0] * num_output_channels

        encoding_min_var = tf.Variable(initial_value=initial_min_val,
                                       name=q_op_name + '_encoding_min',
                                       trainable=is_trainable, dtype=tf.double)
        encoding_max_var = tf.Variable(initial_value=initial_max_val,
                                       name=q_op_name + '_encoding_max',
                                       trainable=is_trainable, dtype=tf.double)

        return encoding_min_var, encoding_max_var

    @staticmethod
    def _get_ops_to_quantize_params_for(graph: tf.Graph, conn_graph: ConnectedGraph, starting_op_names: List[str],
                                        output_op_names: List[str]) -> Dict[str, ParameterInfo]:
        """
        Get names of ops to insert param quantizers for, as well as corresponding indices
        :param graph: TensorFlow graph to get names of ops to quantize weights for
        :param conn_graph: Connected graph of the model
        :param starting_op_names: List of starting op names of the model
        :param output_op_names: List of output op names of the model
        :return: Dictionary with name of parameters to quantize as keys and information about parameters as values
        """
        if conn_graph is None:
            _logger.error("Connected graph is not passed as a parameter")
            raise AssertionError("Connected graph is not passed as a parameter")

        # Get available connected graphs
        valid_conns = [conn for conn in conn_graph.get_all_ops().values()
                       if conn.type not in param_quant_conn_op_ignore_list]

        valid_ops = get_valid_ops(graph, starting_op_names, output_op_names)

        # Get parameters of connected graphs
        params_to_quantize = {}
        for conn in valid_conns:
            for param_name, param_info in conn.parameters.items():
                for consumer_name in param_info.op_with_param_name:
                    consumer = graph.get_operation_by_name(consumer_name)
                    if op_not_in_loop_control_flow_context(graph, consumer) and consumer in valid_ops:
                        if param_name in params_to_quantize:
                            # Parameter can be a weight shared parameter, that was used for a different op that was
                            # processed earlier. In this case, there will already be a parameter info entry for this
                            # parameter, and we need to update the op_with_param_name list to include the current op.
                            params_to_quantize[param_name].op_with_param_name.extend(param_info.op_with_param_name)
                        else:
                            params_to_quantize[param_name] = param_info

        params_to_quantize.update(get_embedding_params_using_patterns(conn_graph))

        return params_to_quantize

    @staticmethod
    def _get_ops_to_quantize_activations_for(graph: tf.Graph, conn_graph: ConnectedGraph) -> List[str]:
        """
        Get names of ops to insert activation quantizers for
        :param graph: TensorFlow graph to get names of ops to quantize weights for
        :param conn_graph: Connected graph of the model
        :return: List of op names to insert activation quantize ops for
        """
        valid_ops = [op for op in conn_graph.get_all_ops().values() if op.type not in op_types_to_ignore]
        op_names_to_quantize = [conn_graph_op.output_op_node.name for conn_graph_op in valid_ops if
                                is_op_quantizable(conn_graph_op.output_op_node)
                                and op_not_in_loop_control_flow_context(graph, conn_graph_op.output_op_node)]

        return op_names_to_quantize

    def _insert_post_training_quant_op_in_loop_context(self, preceeding_tensor,
                                                       quant_op_name: str,
                                                       op_mode: libpymo.QuantizationMode,
                                                       quantizer_dict: Dict[str, QuantizerInfo],
                                                       quantizer_type: QuantizerType,
                                                       bit_width: int = 8,
                                                       data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Create and insert a post-training quant op after a given tensor in a loop control flow context.
        :param preceeding_tensor: Preceeding tensor to insert the quant op after
        :param quant_op_name: Name to give to the new quant op
        :param op_mode: Starting mode to configure for the new quant op
        :param quantizer_dict: dictionary of op and QuantizerInfo
        :param quantizer_type : indicate param or activation quantizer
        :param bit_width : bit-width to be used (output or param quantization bit-width), default set to 8
        :param data_type: data type to use for quantizing all layer parameters
        :return: None
        """

        # this handles cases such as conditional blocks that are defined in their own context
        context_bk = updated_graph_flow_context_to_loop_context(self.session.graph, preceeding_tensor)
        q_op_out = self._insert_post_training_quant_op(preceeding_tensor, quant_op_name, op_mode, quantizer_dict,
                                                       quantizer_type, bit_width, data_type)

        # revert the context back to graph level from op context
        set_graph_flow_context(self.session.graph, context_bk)

        return q_op_out

    def _insert_param_quantizer_loop_context(self, inner_ops, preceeding_tensor,
                                             quant_op_name: str,
                                             op_mode: libpymo.QuantizationMode,
                                             quantizer_dict: Dict[str, QuantizerInfo],
                                             quantizer_type: QuantizerType,
                                             bit_width: int = 8,
                                             data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Create and insert a post-training quant op after a given tensor in a loop control flow context.
        :param preceeding_tensor: Preceeding tensor to insert the quant op after
        :param quant_op_name: Name to give to the new quant op
        :param op_mode: Starting mode to configure for the new quant op
        :param quantizer_dict: dictionary of op and QuantizerInfo
        :param quantizer_type : indicate param or activation quantizer
        :param bit_width: bit-width to be used (output or param quantization bit-width), default set to 8.
        :param data_type: data type to use for quantizing all layer parameters
        :return: None
        """

        # this handles cases such as conditional blocks that are defined in their own context
        context_bk = updated_graph_flow_context_to_loop_context(self.session.graph, preceeding_tensor)
        q_op_out = self._insert_param_quantizer_recurrent(inner_ops, preceeding_tensor, quant_op_name, op_mode, quantizer_dict,
                                                          quantizer_type, bit_width, data_type)

        # revert the context back to graph level from op context
        set_graph_flow_context(self.session.graph, context_bk)

        return q_op_out

    # pylint: disable=too-many-locals
    def _create_and_init_quant_op_input_vars(self, quant_op_name: str, quantizer_dict: Dict[str, QuantizerInfo],
                                             quantizer_type, op_mode: libpymo.QuantizationMode, bit_width: int = 8,
                                             data_type: QuantizationDataType = QuantizationDataType.int):
        """
        creates input variables to Quantize op and initializes them
        :param quant_op_name: quantize op name
        :param quantizer_dict: dictionary of op and QuantizerInfo
        :param quantizer_type: indicate param or activation quantizer
        :param op_mode: Starting mode to configure for the new quant op
        :param bit_width: bit-width to be used (output or param quantization bit-width), default set to 8
        :param data_type: data type to use for quantizing all layer parameters
        :return: quant op input variables created
        """
        with self.session.graph.as_default():
            op_mode_var = tf.Variable(int(op_mode),
                                      name=quant_op_name + '_op_mode', trainable=False,
                                      dtype=tf.int32)

            bit_width = tf.Variable(initial_value=bit_width,
                                    name=quant_op_name + '_bit_width',
                                    trainable=False, dtype=tf.int8)

            # Note: Later, is_symmetric_encoding value is to be read from config file
            use_symmetric_encoding = tf.Variable(initial_value=False,
                                                 name=quant_op_name + '_use_symmetric_encoding',
                                                 trainable=False, dtype=tf.bool)
            axis_handling = AxisHandling.LAST_AXIS
            if quantizer_type == QuantizerType.param and self.per_channel_quantization_enabled:
                tensor_quantizer, tensor_quant_ref, encoding_min, encoding_max, axis_handling = \
                    self._create_per_channel_quantizers_and_encodings(quant_op_name)
            else:
                tensor_quantizer, tensor_quant_ref, \
                encoding_min, encoding_max = self._create_per_tensor_quantizers_and_encodings(quant_op_name)

            quantization_axis_handling = tf.Variable(initial_value=axis_handling.value,
                                                     name=quant_op_name + '_axis_handling',
                                                     trainable=False, dtype=tf.int32)

            is_int_data_type = tf.Variable(initial_value=(data_type == QuantizationDataType.int),
                                           name=quant_op_name + '_data_type', trainable=False, dtype=tf.bool)

            # Add to quantizer dict
            quantizer_info = QuantizerInfo(self.session, tensor_quantizer, quant_op_name, quantizer_type, axis_handling)
            quantizer_dict[quant_op_name] = quantizer_info

            self.session.run([op_mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                              encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer,
                              quantization_axis_handling.initializer, is_int_data_type.initializer])

        return op_mode_var, tensor_quant_ref, encoding_min, encoding_max, bit_width, use_symmetric_encoding, \
               quantization_axis_handling, is_int_data_type

    def _create_per_channel_quantizers_and_encodings(self, quant_op_name: str) -> \
            Tuple[List[libpymo.TensorQuantizer], tf.Variable, tf.Variable, tf.Variable, AxisHandling]:
        """
        Creates per channel quantizers and encoding min max variables
        :param quant_op_name: Name of quantization op with parameter to create per channel quantizers for
        :return: Tensor quantizers, variable with quantizer pointer, encoding min variable, encoding max variable, and
        axis handling enum
        """
        num_output_channels, axis_handling = self._op_name_to_output_channels_axis_handling_dict[quant_op_name]
        tensor_quantizer_int64 = [None] * num_output_channels
        tensor_quantizers = [None] * num_output_channels
        # Create a tensor_quantizer per channel
        for i in range(num_output_channels):
            tensor_quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                                       libpymo.RoundingMode.ROUND_NEAREST)

            tensor_quantizers[i] = tensor_quantizer
            val = libpymo.PtrToInt64(tensor_quantizer)
            tensor_quantizer_int64[i] = val

        tensor_quant_ref = tf.Variable(tensor_quantizer_int64, name=quant_op_name + '_quant_ref',
                                       trainable=False, dtype=tf.int64)

        encoding_min, encoding_max = self._create_encoding_min_max_vars(quant_op_name,
                                                                        quantizer_type=QuantizerType.param)

        return tensor_quantizers, tensor_quant_ref, encoding_min, encoding_max, axis_handling

    def _create_per_tensor_quantizers_and_encodings(self, quant_op_name: str):
        """
        Creates per tensor quantizers and encoding min max variables
        """
        tensor_quantizer = libpymo.TensorQuantizer(quant_scheme_to_libpymo[self._quant_scheme],
                                                   libpymo.RoundingMode.ROUND_NEAREST)
        tensor_quantizer_int64 = libpymo.PtrToInt64(tensor_quantizer)
        tensor_quant_ref = tf.Variable(tensor_quantizer_int64, name=quant_op_name + '_quant_ref',
                                       trainable=False, dtype=tf.int64)
        encoding_min, encoding_max = self._create_encoding_min_max_vars(quant_op_name)
        return tensor_quantizer, tensor_quant_ref, encoding_min, encoding_max

    def _insert_param_quantizer_recurrent(self, inner_ops, preceeding_tensor, quant_op_name: str,
                                          op_mode: libpymo.QuantizationMode,
                                          quantizer_dict: Dict[str, QuantizerInfo], quantizer_type: QuantizerType,
                                          bit_width: int = 8,
                                          data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Create and insert a post-training quant op after a given tensor
        :param preceeding_tensor: Preceeding tensor to insert the quant op after
        :param quant_op_name: Name to give to the new quant op
        :param op_mode: Starting mode to configure for the new quant op
        :param quantizer_dict: dictionary of op and QuantizerInfo
        :param quantizer_type : indicate param or activation quantizer
        :param bit_width : bit-width to be used (output or param quantization bit-width), default set to 8
        :param data_type: data type to use for quantizing all layer parameters
        :return: None
        """
        # pylint: disable=too-many-locals
        # Create variables for op_mode, tensor_quantizer_reference, encoding_min, encoding_max, bitwidth and
        # is_symmetric_encoding flag
        # (so we can change these in the future, if needed)

        op_mode_var, tensor_quant_ref, encoding_min, encoding_max, bit_width, use_symmetric_encoding, _, _ = \
            self._create_and_init_quant_op_input_vars(quant_op_name, quantizer_dict, quantizer_type, op_mode,
                                                      bit_width, data_type)

        # extract loop cond bool variable
        time_step_tensor = get_time_steps_tensor_from_rnn_inner_ops(inner_ops)

        # CPU device assignment for QcQuantize op
        q_op_out = self._create_and_place_recurrent_param_quantize_op(quant_op_name, preceeding_tensor,
                                                                      op_mode_var,
                                                                      tensor_quant_ref,
                                                                      encoding_min,
                                                                      encoding_max,
                                                                      bit_width,
                                                                      use_symmetric_encoding,
                                                                      time_step_tensor)

        return q_op_out

    def _insert_post_training_quant_op(self, preceeding_tensor, quant_op_name: str, op_mode: libpymo.QuantizationMode,
                                       quantizer_dict: Dict[str, QuantizerInfo], quantizer_type: QuantizerType,
                                       bit_width: int = 8, data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Create and insert a post-training quant op after a given tensor
        :param preceeding_tensor: Preceeding tensor to insert the quant op after
        :param quant_op_name: Name to give to the new quant op
        :param op_mode: Starting mode to configure for the new quant op
        :param quantizer_dict: dictionary of op and QuantizerInfo
        :param quantizer_type : indicate param or activation quantizer
        :param bit_width : bit-width to be used (output or param quantization bit-width), default set to 8.
        :param data_type: data type to use for quantizing the op
        :return: None
        """
        # pylint: disable=too-many-locals
        # Create variables for op_mode, tensor_quantizer_reference, encoding_min, encoding_max, bitwidth and
        # is_symmetric_encoding flag
        # (so we can change these in the future, if needed)

        op_mode_var, tensor_quant_ref, encoding_min, encoding_max, bit_width, use_symmetric_encoding, \
        quantization_axis_handling, is_int_data_type = self._create_and_init_quant_op_input_vars(quant_op_name,
                                                                                                 quantizer_dict,
                                                                                                 quantizer_type,
                                                                                                 op_mode,
                                                                                                 bit_width, data_type)

        # CPU device assignment for QcQuantize op
        q_op_out = self._create_and_place_quantize_op(quant_op_name, preceeding_tensor, op_mode_var, tensor_quant_ref,
                                                      encoding_min, encoding_max, bit_width, use_symmetric_encoding,
                                                      quantizer_type, quantization_axis_handling, is_int_data_type)

        return q_op_out

    def _create_and_place_quantize_op(self, quant_op_name: str, preceeding_tensor,
                                      op_mode_var: tf.Variable, tensor_quant_ref: tf.Variable,
                                      encoding_min: tf.Variable, encoding_max: tf.Variable, bit_width: tf.Variable,
                                      use_symmetric_encoding: tf.Variable, quantizer_type: QuantizerType,
                                      quantization_axis_handling: tf.Variable, is_int_data_type: tf.Variable):
        """
        Create a QcQuantize op and place it on CPU/CPU and with the right custom-gradient function registered
        """
        # pylint: disable=too-many-arguments

        def create_quantize_op():
            if self.per_channel_quantization_enabled and quantizer_type == QuantizerType.param:

                is_training = tf.keras.backend.learning_phase()

                op = qcops.qc_quantize_per_channel(name=quant_op_name, in_tensor=preceeding_tensor,
                                                   op_mode=op_mode_var,
                                                   tensor_quantizer_reference=tensor_quant_ref,
                                                   encoding_min=encoding_min,
                                                   encoding_max=encoding_max,
                                                   bit_width=bit_width,
                                                   is_int_data_type=is_int_data_type,
                                                   use_symmetric_encoding=use_symmetric_encoding,
                                                   axis_handling=quantization_axis_handling, is_training=is_training)
            else:
                op = qcops.qc_quantize(name=quant_op_name, in_tensor=preceeding_tensor,
                                       op_mode=op_mode_var,
                                       tensor_quantizer_reference=tensor_quant_ref,
                                       encoding_min=encoding_min, encoding_max=encoding_max,
                                       bit_width=bit_width,
                                       use_symmetric_encoding=use_symmetric_encoding,
                                       is_int_data_type=is_int_data_type)

            return op

        if not self._use_cuda:
            with tf.device('/cpu:0'):
                if self._quant_scheme in [QuantScheme.training_range_learning_with_tf_init,
                                          QuantScheme.training_range_learning_with_tf_enhanced_init]:
                    with self.session.graph.gradient_override_map(
                            {"QcQuantize": "QcQuantizeRangeLearningCustomGradient",
                             "QcQuantizePerChannel": "QcQuantizePerChannelRangeLearningCustomGradient"}):
                        q_op_out = create_quantize_op()
                else:
                    q_op_out = create_quantize_op()

        # GPU device assignment for QcQuantize op
        else:
            if self._quant_scheme in [QuantScheme.training_range_learning_with_tf_init,
                                      QuantScheme.training_range_learning_with_tf_enhanced_init]:
                with self.session.graph.gradient_override_map(
                        {"QcQuantize": "QcQuantizeRangeLearningCustomGradient",
                         "QcQuantizePerChannel": "QcQuantizePerChannelRangeLearningCustomGradient"}):
                    q_op_out = create_quantize_op()
            else:
                q_op_out = create_quantize_op()

        return q_op_out

    def _create_and_place_recurrent_param_quantize_op(self, quant_op_name: str, preceeding_tensor,
                                                      op_mode_var: tf.Variable, tensor_quant_ref: tf.Variable,
                                                      encoding_min: tf.Variable, encoding_max: tf.Variable,
                                                      bit_width: tf.Variable,
                                                      use_symmetric_encoding: tf.Variable, time_steps):
        def create_recurrent_param_quantize_op():
            op = qcops.qc_quantize_recurrent_param(name=quant_op_name, in_tensor=preceeding_tensor,
                                                   op_mode=op_mode_var, tensor_quantizer_reference=tensor_quant_ref,
                                                   encoding_min=encoding_min, encoding_max=encoding_max,
                                                   bit_width=bit_width, use_symmetric_encoding=use_symmetric_encoding,
                                                   time_steps=time_steps)
            return op

        if not self._use_cuda:
            with tf.device('/cpu:0'):
                q_op_out = create_recurrent_param_quantize_op()

        # GPU device assignment for QcQuantize op
        else:
            q_op_out = create_recurrent_param_quantize_op()

        return q_op_out

    @staticmethod
    def _is_op_transformer_mask(quant_op_name: str) -> bool:
        """
        Check if quant_op_name is transformer mask add op
        :param quant_op_name: op name to check
        :return: True if quant_op_name belongs to transformer mask add op
        """
        for supported_mask in transformer_utils.SUPPORTED_ATTENTION_MASK_OVERRIDE:
            if quant_op_name.endswith(supported_mask + '_quantized'):
                return True
        return False

    def _override_quant_config_for_transformer_mask_add(self):
        """
        Find transformer mask add op and change bitwidth to 16 and quant_scheme to tf
        """
        for quant_op_name, quantizer_info in self._activation_quantizers.items():
            if self._is_op_transformer_mask(quant_op_name) and quantizer_info.data_type == QuantizationDataType.int:
                quantizer_info.bitwidth = 16
                quantizer_info.quant_scheme = QuantScheme.post_training_tf

    def _clamp_transformer_attention_mask_encoding(self):
        """
        Clamp the quantizer encoding min associated with mask adder op within an attention head.
        """
        for quant_op_name, quantizer_info in self._activation_quantizers.items():
            if self._is_op_transformer_mask(quant_op_name) and quantizer_info.enabled \
                    and quantizer_info.data_type == QuantizationDataType.int:
                encoding = quantizer_info.get_encoding()
                encoding.min = max(encoding.min, transformer_utils.MASK_OVERRIDE_VALUE)

                clamped_encoding = recompute_grid_params(encoding, self._default_output_bw,
                                                         quantizer_info.use_symmetric_encoding)
                quantizer_info.bitwidth = self._default_output_bw
                quantizer_info.quant_scheme = self._quant_scheme
                quantizer_info.set_encoding(clamped_encoding)
                quantizer_info.freeze_encoding()


# load and save utilities
def update_tensor_quantizer_references(quant_sim_sess: tf.compat.v1.Session, quantizer_dict: Dict[str, QuantizerInfo]):
    """
    updates the param / activation quant ops in the passed-in session with new tensor quantizer references.
    :param quant_sim_sess: tensorflow session held by quantsim object
    :param quantizer_dict: dictionary with quant ops and associated quantizer info
    :return: None, updates passed-in session quant ops with new tensor quantizer references.
    """

    vars_with_value = {}
    for q_op_name in quantizer_dict:
        # also update the session held by tensor quantizer object
        quantizer_dict[q_op_name].session = quant_sim_sess
        # For per channel quantization of parameters
        tensor_quantizers = quantizer_dict[q_op_name].tensor_quantizer
        tensor_quantizer_ref = []
        if isinstance(tensor_quantizers, list):
            for tensor_quantizer in tensor_quantizers:
                ptr_to_int64_val = libpymo.PtrToInt64(tensor_quantizer)
                tensor_quantizer_ref.append(ptr_to_int64_val)
        else:
            ptr_to_int64_val = libpymo.PtrToInt64(tensor_quantizers)
            tensor_quantizer_ref.append(ptr_to_int64_val)
            tensor_quantizer_ref = tensor_quantizer_ref[0]
        vars_with_value[q_op_name + '_quant_ref'] = tensor_quantizer_ref

    update_variables_with_values(quant_sim_sess, vars_with_value)


def save_checkpoint(quantsim: QuantizationSimModel, meta_path: str, file_name_prefix: str):
    """
    Saves a checkpoint of the QuantSim model which can be loaded at a later point to continue fine-tuning.
    See also load_checkpoint().

    :param quantsim: QuantizationSimModel to be saved
    :param meta_path: path to save the meta file
    :param file_name_prefix: filename prefix string
    """
    if not os.path.exists(meta_path):
        os.mkdir(meta_path)

    save_path = os.path.join(meta_path, file_name_prefix)

    # save the model with quant ops
    utils.graph_saver.save_model_to_meta(quantsim.session, save_path)

    # save info in the quantsim object
    save_data_to_pickle_file(quantsim, meta_path, 'orig_quantsim_config')


def load_checkpoint(meta_path: str, file_name_prefix: str) -> QuantizationSimModel:
    """
    Loads QuantSim model from saved checkpoint and pickle files.

    :param meta_path: to load meta from
    :param file_name_prefix: filename prefix string
    :return: returns new QuantSim object
    """
    #pylint: disable=protected-access

    # load saved session with quant ops
    new_sess = utils.graph_saver.load_model_from_meta(meta_path=str(meta_path + '/' + file_name_prefix + '.meta'))

    # load quant sim model object with params from saved pickle data
    new_quant_sim = load_data_from_pickle_file(meta_path + '/orig_quantsim_config')

    # set session for the new quantsim object
    new_quant_sim.session = new_sess

    # update tensor references in the new quantsim object
    update_tensor_quantizer_references(new_sess, new_quant_sim._param_quantizers)
    update_tensor_quantizer_references(new_sess, new_quant_sim._activation_quantizers)

    return new_quant_sim


def check_accumulator_overflow(sess: tf.compat.v1.Session, quant_bw: int, accum_bw: int):
    """
    Checks for any potential for accumulator overflow across all the layers of the given model
    :param sess: Tensorflow session
    :param quant_bw: Bitwidth the layers are quantized at
    :param accum_bw: Bitwidth of the accumulator
    :return: Name of the layer with the most accumulator range used and range used
    """

    most_accum_range_used = 0
    most_accum_range_used_layer = None

    for op in sess.graph.get_operations():
        if op.type == 'Conv2D':
            weights = utils.op.conv.WeightTensorUtils.get_tensor_as_numpy_data(sess, op)
            weights = np.transpose(weights, (3, 2, 0, 1))       # Reshape from HWIO to OIHW
            was_accum_range_exceeded, accum_range_used = get_conv_accum_bounds(weights, quant_bw, accum_bw)
            if accum_range_used > most_accum_range_used:
                most_accum_range_used = accum_range_used
                most_accum_range_used_layer = op.name

            if was_accum_range_exceeded:
                _logger.info('Possible accumulator overflow for layer: %s', op.name)

    if most_accum_range_used < 1:
        _logger.info('No overflow detected. Layer %s had the most accumulator range used: %f%%',
                     most_accum_range_used_layer, most_accum_range_used * 100)
    else:
        _logger.info('Overflow detected. Layer %s had the most accumulator range used: %f%%',
                     most_accum_range_used_layer, most_accum_range_used * 100)

    return most_accum_range_used_layer, most_accum_range_used
