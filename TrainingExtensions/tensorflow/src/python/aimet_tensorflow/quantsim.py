# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorflow.contrib import graph_editor
from aimet_common.defs import QuantScheme
from aimet_common.quantsim import gate_min_max, calculate_delta_offset, encoding_version
from aimet_common.utils import AimetLogger, save_json_yaml
from aimet_tensorflow.common import core
from aimet_tensorflow.utils.common import update_variables_with_values, save_data_to_pickle_file, \
    load_data_from_pickle_file, get_valid_ops
from aimet_tensorflow.utils import graph_saver
from aimet_tensorflow.utils.constants import QuantizeOpIndices
from aimet_tensorflow.utils.quantsim import create_op_to_quant_ops_dict, is_op_quantizable, \
    get_time_steps_tensor_from_rnn_inner_ops, create_encoding_from_dict
from aimet_tensorflow.utils.graph import updated_graph_flow_context_to_loop_context, set_graph_flow_context, \
    op_not_in_loop_control_flow_context
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.quantizer_info import QuantizerInfo, QuantizerType, quant_scheme_to_libpymo
from aimet_tensorflow.quantsim_config.quantsim_config import QuantSimConfigurator
from aimet_tensorflow.quantsim_recurrent import _select_simple_rnn_internal_ops_to_quantize, \
    _select_lstm_internal_ops_to_quantize, SUPPORTED_RECURRENT_TYPES

# this is required to associate gradient with QcQuantize op
from aimet_tensorflow import quantsim_straight_through_grad      # pylint: disable=unused-import
import libpymo


# pylint: disable=too-many-lines

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
WORKING_DIR = '/tmp/quantsim/'


# Op types which we will not place quantize ops after
op_types_to_ignore = {'branch', 'Flatten', 'Shape'}

DTYPES_QUANTIZE_NOT_REQUIRED = [tf.dtypes.int8, tf.dtypes.uint8, tf.dtypes.int16, tf.dtypes.uint16,
                                tf.dtypes.int32, tf.dtypes.uint32, tf.dtypes.int64, tf.dtypes.uint64,
                                tf.bool, tf.dtypes.string]

def _load_ops():
    """
    Function which loads the quantization op library. In order to load a graph with
    custom quantization ops this must be called first as this provides tensorflow with
    the required op definitions.

    :return: Loaded library
    """
    return tf.load_op_library('libaimet_tf_ops.so')


# Load the aimet ops
qcops = _load_ops()


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
    def __init__(self, session: tf.compat.v1.Session, starting_op_names: List[str], output_op_names: List[str],
                 quant_scheme: Union[str, QuantScheme] = 'tf_enhanced', rounding_mode: str = 'nearest',
                 default_output_bw: int = 8, default_param_bw: int = 8, use_cuda: bool = True, config_file: str = None
                 ):
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

        :returns: An object which can be used to perform quantization on a tensorflow graph
        :raises: ValueError: An error occurred processing one of the input parameters.

        """
        # sanity checks
        if quant_scheme not in ['tf_enhanced', 'tf'] and not isinstance(quant_scheme, QuantScheme):
            raise ValueError('Parameter quantization scheme is not a valid selection. ')

        if rounding_mode not in ('nearest', 'stochastic'):
            raise ValueError('Parameter round mode is not a valid selection. Valid selections are nearest or '
                             'stochastic')

        if default_param_bw < 4 or default_param_bw > 32:
            raise ValueError('Default bitwidth for parameters must be between 4 and 32, not ' + str(default_param_bw))

        if default_output_bw < 4 or default_output_bw > 32:
            raise ValueError('Activation bitwidth must be between 4 and 32, not ' + str(default_output_bw))

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

        # We save a copy of the original model (to be used during export later)
        with self.session.graph.as_default():
            saver = tf.compat.v1.train.Saver()
        saver.save(self.session, save_path=WORKING_DIR+'orig_model_before_quantsim')

        self._add_and_configure_quant_nodes(starting_op_names, output_op_names, default_param_bw, default_output_bw,
                                            config_file)

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

    def quantizer_config(self, quant_op_name: str) -> QuantizerInfo:
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

    def _get_op_variable_value(self, quant_op: tf.Operation, var_index: int):
        """
        Utility to load variable values from quant op
        :param quant_op: quantize op
        :param var_index: variable index to be read
        :return: variable value
        """

        op_var_tensor = quant_op.inputs[var_index]
        return self.session.run(op_var_tensor)

    def configure_quantization_ops(self, conn_graph: ConnectedGraph, ops_with_param_names: List[str],
                                   indices: List[int], activation_op_names: List[str], config_file: str):
        """
        Configure inserted quantize ops using config file
        :param conn_graph: Connected graph of the model
        :param ops_with_param_names: List of ops for which param quantization ops were inserted for
        :param indices: List of input indices (one-to-one for each entry in ops)
        :param activation_op_names: List of ops for which activation quantization ops were inserted for
        :param config_file: Configuration file to use
        """
        if not conn_graph:
            _logger.error('Connected graph passed into configure_quantization_ops() is None. If manual insertion of '
                          'quantization ops is being done, and get_ops_to_quantize_activations_for() has been '
                          'overriden, please override configure_quantization_ops() as well.')
            raise AssertionError
        op_to_quant_ops_dict = create_op_to_quant_ops_dict(self.session.graph, conn_graph,
                                                           ops_with_param_names, indices,
                                                           activation_op_names)
        QuantSimConfigurator(self.session, conn_graph, op_to_quant_ops_dict, config_file)

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

        # Run data through the quantsim so we can compute activation encodings
        forward_pass_callback(self.session, forward_pass_callback_args)

        ops_with_invalid_encodings = []

        # For activations, calculate encodings and update min-max parameters
        for op_name, quantizer_info in self._activation_quantizers.items():
            # Calculate encodings
            if quantizer_info.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
                op_bitwidth, op_use_symmetric_encodings = quantizer_info.bitwidth, quantizer_info.use_symmetric_encoding
                encoding = quantizer_info.compute_encoding(op_bitwidth, op_use_symmetric_encodings)
                if quantizer_info.is_encoding_valid():
                    quantizer_info.set_encoding(encoding)
                    quantizer_info.set_op_mode(libpymo.TensorQuantizerOpMode.quantizeDequantize)
                else:
                    quantizer_info.set_op_mode(libpymo.TensorQuantizerOpMode.passThrough)
                    ops_with_invalid_encodings.append(op_name)

        # For post-training mode, params will always be in one-shot mode
        op_mode = QuantizationSimModel._param_op_mode_after_analysis(self._quant_scheme)

        for op_name, quantizer_info in self._param_quantizers.items():
            if quantizer_info.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
                op_bitwidth, op_use_symmetric_encodings = quantizer_info.bitwidth, quantizer_info.use_symmetric_encoding
                encoding = quantizer_info.compute_encoding(op_bitwidth, op_use_symmetric_encodings)
                if quantizer_info.is_encoding_valid():
                    quantizer_info.set_encoding(encoding)
                    quantizer_info.set_op_mode(op_mode)
                else:
                    quantizer_info.set_op_mode(libpymo.TensorQuantizerOpMode.passThrough)
                    ops_with_invalid_encodings.append(op_name)

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

        # save session without quant nodes
        if orig_sess is not None:
            with orig_sess.graph.as_default():
                saver = tf.compat.v1.train.Saver()
            saver.save(orig_sess, save_path=WORKING_DIR+'orig_model_before_quantsim')
        else:
            _logger.info('Original session is not provided, use orig_model_before_quantsim.meta to export')

        vars_to_save = []
        with self.session.graph.as_default():
            for var in tf.compat.v1.global_variables():
                if not var.name[:-2].endswith(('_quantized', '_quantized_op_mode', '_quantized_quant_ref',
                                               '_quantized_encoding_min', '_quantized_encoding_max',
                                               '_quantized_bit_width', '_quantized_use_symmetric_encoding')):
                    vars_to_save.append(var)

            saver = tf.compat.v1.train.Saver(vars_to_save)
            saver.save(self.session, save_path=os.path.join(path, filename_prefix))
            shutil.copyfile(WORKING_DIR+'orig_model_before_quantsim.meta',
                            os.path.join(path, filename_prefix) + '.meta')

        self._export_encodings(os.path.join(path, filename_prefix) + '.encodings')

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

        new_sess = graph_saver.save_and_load_graph(temp_dir_path, self.session)
        return new_sess

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
                encoding_dict = param_encodings[tensor_name][0]
                encoding, is_symmetric = create_encoding_from_dict(encoding_dict)
                quantizer_info.use_symmetric_encoding = is_symmetric
                quantizer_info.set_and_freeze_encoding_and_op_mode(encoding, op_mode)
                _logger.info("Setting and freezing quantization encodings for parameter: %s", tensor_name)

    @staticmethod
    def _param_op_mode_after_analysis(_) -> libpymo.TensorQuantizerOpMode:
        """
        Returns op mode to use for parameters after encodings have been computed
        :return:
        """
        return libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize

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

        def update_encoding_dict_entry(encoding_dict: Dict,
                                       quant_op_name: str):

            quant_op = self.session.graph.get_operation_by_name(quant_op_name)
            min_val, max_val = self.read_min_max(quant_op_name, variable_dict)

            min_val, max_val = gate_min_max(min_val, max_val)
            op_bitwidth = int(self._get_op_variable_value(quant_op, QuantizeOpIndices.bit_width))
            delta, offset = calculate_delta_offset(min_val, max_val, op_bitwidth)
            is_symmetric = str(self._get_op_variable_value(quant_op,
                                                           QuantizeOpIndices.use_symmetric_encoding))

            tensor_name = quant_op.inputs[0].name
            encoding_dict[tensor_name] = [{'min': min_val,
                                           'max': max_val,
                                           'scale': delta,
                                           'offset': offset,
                                           'bitwidth': op_bitwidth,
                                           'is_symmetric': is_symmetric}]

        param_encodings = {}
        for quant_op_name, quantizer_info in self._param_quantizers.items():
            if not quantizer_info.tensor_quantizer.isEncodingValid:
                continue
            update_encoding_dict_entry(param_encodings, quant_op_name)

        activation_encodings = {}
        for quant_op_name, quantizer_info in self._activation_quantizers.items():
            if not quantizer_info.tensor_quantizer.isEncodingValid:
                continue
            update_encoding_dict_entry(activation_encodings, quant_op_name)

        encodings_dict = {'version': encoding_version,
                          'activation_encodings': activation_encodings,
                          'param_encodings': param_encodings}

        save_json_yaml(encoding_file_path, encodings_dict)

    def _save_and_load_sim_model(self):
        self.session = graph_saver.save_and_load_graph(WORKING_DIR, self.session)
        update_tensor_quantizer_references(self.session, self._activation_quantizers)
        update_tensor_quantizer_references(self.session, self._param_quantizers)

    def _add_quant_nodes_recurrent(self, graph, starting_op_names: List[str], output_op_names: List[str],
                                   default_param_bw: int, default_output_bw: int) \
            -> Tuple[List[str], List[int], List[str]]:
        """
        Utility to add quant nodes to recurrent module
        :param graph: TensorFlow graph
        :param starting_op_names: List of starting op names of the model
        :param output_op_names: List of output op names of the model
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

        conn_graph = ConnectedGraph(graph, starting_op_names, output_op_names)
        for op in conn_graph.get_all_ops().values():
            #  we can configure custom layer selectors per recurrent type or use default one
            if op.type in SUPPORTED_RECURRENT_TYPES:

                internal_ops = op.internal_ops

                # select internal ops to quantize in this recurrent type
                select_internal_ops_to_quantize = switcher.get(op.type)
                module_ops_with_param_names, module_op_input_indices, module_activation_op_names = \
                    select_internal_ops_to_quantize(self.session.graph, internal_ops)

                # insert the quant nodes
                self._insert_param_quantization_ops(module_ops_with_param_names, module_op_input_indices,
                                                    default_param_bw, internal_ops, in_loop_context=True)

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
                                       default_param_bw: int, default_output_bw: int, config_file: str):
        """
        Utility to add quant nodes
        :param starting_op_names: List of starting op names of the model
        :param output_op_names: List of output op names of the model
        :param default_param_bw: default param bitwidth
        :param default_output_bw: default output bitwidth
        :param config_file: Configuration file to use
        """

        # Get list of ops with params to insert quantizers for, as well as the input indices to insert on.
        ops_with_param_names, input_indices = QuantizationSimModel._get_ops_to_quantize_params_for(self.session.graph,
                                                                                                   starting_op_names,
                                                                                                   output_op_names)

        # Get list of activation ops to insert quantizers for, and the connected graph used to obtain these ops
        activation_op_names, conn_graph = QuantizationSimModel._get_ops_to_quantize_activations_for(self.session.graph,
                                                                                                    starting_op_names,
                                                                                                    output_op_names)

        self._insert_param_quantization_ops(ops_with_param_names, input_indices, default_param_bw)
        self._insert_activation_quantization_ops(activation_op_names, default_output_bw)

        # this takes care of quant node insertion in loop context of recurrent layer, which makes a cell
        recurrent_ops_with_param_names, recurrent_input_indices, recurrent_activation_op_names = \
            self._add_quant_nodes_recurrent(self.session.graph, starting_op_names, output_op_names,
                                            default_param_bw, default_output_bw)

        if recurrent_ops_with_param_names and recurrent_input_indices:
            ops_with_param_names.extend(recurrent_ops_with_param_names)
            input_indices.extend(recurrent_input_indices)
        if recurrent_activation_op_names:
            activation_op_names.extend(recurrent_activation_op_names)

        # Note: at this point, the session used to construct conn_graph is different than the current
        # self.session, however we still use the connected graph to traverse the graph structure.
        self.configure_quantization_ops(conn_graph, ops_with_param_names, input_indices, activation_op_names,
                                        config_file)

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

    def _insert_param_quantization_ops(self, op_names: List[str], indices: List[int], default_param_bw: int,
                                       inner_ops: List[str] = None, in_loop_context: bool = False):
        """
        Inserts quantization ops for individual parameters
        :param ops: List of ops whose parameters are being quantized
        :param indices: List of input indices (one-to-one for each entry in ops)
        :param default_param_bw : default param bitwidth
        :param in_loop_context: True, if the ops belong to loop control flow context
        :return: None
        """
        ops = [self.session.graph.get_operation_by_name(op_name) for op_name in op_names]
        assert len(ops) == len(indices)

        for op, index in zip(ops, indices):
            # Modify the weight/bias inputs to use the quantized inputs
            can_modify_op, param_in = QuantizationSimModel._get_op_to_modify_with_param_in(op, index)

            if param_in is not None:
                quant_op_name = self._get_quantized_name(param_in.op.name)
                _logger.info("Adding weight quantization op %s", quant_op_name)
                op_mode = libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize

                if in_loop_context:
                    q_op_out = self._insert_param_quantizer_loop_context(inner_ops, param_in, quant_op_name,
                                                                         op_mode, self._param_quantizers,
                                                                         QuantizerType.param,
                                                                         default_param_bw)
                else:
                    q_op_out = self._insert_post_training_quant_op(param_in, quant_op_name,
                                                                   op_mode, self._param_quantizers, QuantizerType.param,
                                                                   default_param_bw)

                nodes_modified_count = graph_editor.reroute_ts(tf_ops.convert_to_tensor(q_op_out), param_in,
                                                               can_modify=can_modify_op)
                if nodes_modified_count != 1:
                    raise ValueError('Input ' + param_in.name + ' not quantized!')

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
                                            in_loop_context: bool = False):
        """
        Inserts quantization ops at the outputs of given ops
        :param valid_op_names: List of op names to insert activation quantizers for
        :param default_output_bw: default activation bitwidth
        :param in_loop_context: True, if the ops belong to a loop control flow context
        return:
        """
        for op_name in valid_op_names:
            quant_op_name = self._get_quantized_name(op_name)
            op = self.session.graph.get_operation_by_name(op_name)
            _logger.info("Adding activation quantization op %s", quant_op_name)

            consumers = [consumer for consumer in op.outputs[0].consumers() if 'gradients' not in consumer.name]

            if not QuantizationSimModel._is_op_quantizable(op):
                _logger.error('Unsupported dtype {%s} detected for op {%s}.', op.outputs[0].dtype, op_name)
                raise AssertionError

            if in_loop_context:
                q_op_out = self._insert_post_training_quant_op_in_loop_context(op.outputs[0], quant_op_name,
                                                                               libpymo.TensorQuantizerOpMode.updateStats,
                                                                               self._activation_quantizers,
                                                                               QuantizerType.activation,
                                                                               default_output_bw)
            else:
                q_op_out = self._insert_post_training_quant_op(op.outputs[0], quant_op_name,
                                                               libpymo.TensorQuantizerOpMode.updateStats,
                                                               self._activation_quantizers, QuantizerType.activation,
                                                               default_output_bw)

            # Re-route
            num_rerouted_outputs = graph_editor.reroute_ts(tf_ops.convert_to_tensor(q_op_out),
                                                           op.outputs[0], can_modify=consumers)
            if num_rerouted_outputs != len(consumers):
                raise ValueError('Failed to map ' + str(len(consumers)) + ' quantization output(s). Only mapped ' +
                                 str(num_rerouted_outputs))

    def _create_encoding_min_max_vars(self, q_op_name: str) -> (tf.Variable, tf.Variable):
        """
        creates encoding min and max variables for quant op.
        :param q_op_name: name of quantize op
        :return: encoding min and max as tf.Variable type
        """

        is_trainable = True
        if self._quant_scheme in [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced]:
            is_trainable = False

        encoding_min_var = tf.Variable(initial_value=0.0,
                                       name=q_op_name + '_encoding_min',
                                       trainable=is_trainable, dtype=tf.double)
        encoding_max_var = tf.Variable(initial_value=0.0,
                                       name=q_op_name + '_encoding_max',
                                       trainable=is_trainable, dtype=tf.double)

        return encoding_min_var, encoding_max_var

    @staticmethod
    def _get_ops_to_quantize_params_for(graph: tf.Graph, starting_op_names: List[str], output_op_names: List[str]) \
            -> Tuple[List[str], List[int]]:
        """
        Get names of ops to insert param quantizers for, as well as corresponding indices
        :param graph: TensorFlow graph to get names of ops to quantize weights for
        :param starting_op_names: List of starting op names of the model
        :param output_op_names: List of output op names of the model
        :return: Tuple consisting of list of op names with params to insert quantize ops for as well as list of indices
        of parameters for each op
        """
        # Get the op query module
        query = core.OpQuery(graph, ops_to_ignore=None)
        valid_ops = get_valid_ops(graph, starting_op_names, output_op_names)
        ops_with_param_names = [op.name for op in query.get_weight_ops() if op in valid_ops and
                                op_not_in_loop_control_flow_context(graph,
                                                                    graph.get_operation_by_name(op.name))]
        # op's control_flow_context() will be populated for ops within conditional blocks
        ops_with_params = [graph.get_operation_by_name(op_name) for op_name in ops_with_param_names]
        input_indices = query.get_weight_inputs(ops_with_params)
        if len(ops_with_param_names) != len(input_indices):
            _logger.error("Length of ops with params and input indices differ")
            raise AssertionError
        return ops_with_param_names, input_indices

    @staticmethod
    def _get_ops_to_quantize_activations_for(graph: tf.Graph, starting_op_names: List[str], output_op_names: List[str]) \
            -> Tuple[List[str], ConnectedGraph]:
        """
        Get names of ops to insert activation quantizers for
        :param graph: TensorFlow graph to get names of ops to quantize weights for
        :param starting_op_names: List of starting op names of the model
        :param output_op_names: List of output op names of the model
        :return: List of op names to insert activation quantize ops for, and the connected graph used to obtain these
        ops.
        """
        conn_graph = ConnectedGraph(graph, starting_op_names, output_op_names)
        valid_ops = [op for op in conn_graph.get_all_ops().values() if op.type not in op_types_to_ignore]
        op_names_to_quantize = [conn_graph_op.output_op_node.name for conn_graph_op in valid_ops if
                                is_op_quantizable(conn_graph_op.output_op_node)
                                and op_not_in_loop_control_flow_context(graph, conn_graph_op.output_op_node)]
        return op_names_to_quantize, conn_graph

    def _insert_post_training_quant_op_in_loop_context(self, preceeding_tensor,
                                                       quant_op_name: str,
                                                       op_mode: libpymo.QuantizationMode,
                                                       quantizer_dict: Dict[str, QuantizerInfo],
                                                       quantizer_type: QuantizerType,
                                                       bit_width: int = 8):
        """
        Create and insert a post-training quant op after a given tensor in a loop control flow context.
        :param preceeding_tensor: Preceeding tensor to insert the quant op after
        :param quant_op_name: Name to give to the new quant op
        :param op_mode: Starting mode to configure for the new quant op
        :param quantizer_dict: dictionary of op and QuantizerInfo
        :param quantizer_type : indicate param or activation quantizer
        :param bit_width : bit-width to be used (output or param quantization bit-width), default set to 8.
        :return: None
        """

        # this handles cases such as conditional blocks that are defined in their own context
        context_bk = updated_graph_flow_context_to_loop_context(self.session.graph, preceeding_tensor)
        q_op_out = self._insert_post_training_quant_op(preceeding_tensor, quant_op_name, op_mode, quantizer_dict,
                                                       quantizer_type, bit_width)

        # revert the context back to graph level from op context
        set_graph_flow_context(self.session.graph, context_bk)

        return q_op_out

    def _insert_param_quantizer_loop_context(self, inner_ops, preceeding_tensor,
                                             quant_op_name: str,
                                             op_mode: libpymo.QuantizationMode,
                                             quantizer_dict: Dict[str, QuantizerInfo],
                                             quantizer_type: QuantizerType,
                                             bit_width: int = 8):
        """
        Create and insert a post-training quant op after a given tensor in a loop control flow context.
        :param preceeding_tensor: Preceeding tensor to insert the quant op after
        :param quant_op_name: Name to give to the new quant op
        :param op_mode: Starting mode to configure for the new quant op
        :param quantizer_dict: dictionary of op and QuantizerInfo
        :param quantizer_type : indicate param or activation quantizer
        :param bit_width : bit-width to be used (output or param quantization bit-width), default set to 8.
        :return: None
        """

        # this handles cases such as conditional blocks that are defined in their own context
        context_bk = updated_graph_flow_context_to_loop_context(self.session.graph, preceeding_tensor)
        q_op_out = self._insert_param_quantizer_recurrent(inner_ops, preceeding_tensor, quant_op_name, op_mode, quantizer_dict,
                                                          quantizer_type, bit_width)

        # revert the context back to graph level from op context
        set_graph_flow_context(self.session.graph, context_bk)

        return q_op_out

    def _create_and_init_quant_op_input_vars(self, quant_op_name: str, quantizer_dict: Dict[str, QuantizerInfo],
                                             quantizer_type, op_mode: libpymo.QuantizationMode, bit_width: int = 8):
        """
        creates input variables to Quantize op and initializes them
        :param quant_op_name: quantize op name
        :param quantizer_dict: dictionary of op and QuantizerInfo
        :param quantizer_type: indicate param or activation quantizer
        :param op_mode: Starting mode to configure for the new quant op
        :param bit_width: bit-width to be used (output or param quantization bit-width), default set to 8.
        :return: quant op input variables created
        """
        with self.session.graph.as_default():
            op_mode_var = tf.Variable(int(op_mode),
                                      name=quant_op_name + '_op_mode', trainable=False,
                                      dtype=tf.int32)

            # Note: Last param of TensorQuantizer is a flag to indicate is_symmetric_encoding,
            # this value is to be read from config file
            tensor_quantizer = libpymo.TensorQuantizer(quant_scheme_to_libpymo[self._quant_scheme],
                                                       libpymo.RoundingMode.ROUND_NEAREST)
            tensor_quantizer_int64 = libpymo.PtrToInt64(tensor_quantizer)
            tensor_quant_ref = tf.Variable(tensor_quantizer_int64, name=quant_op_name + '_quant_ref',
                                           trainable=False, dtype=tf.int64)

            # Add to quantizer dict
            quantizer_info = QuantizerInfo(self.session, tensor_quantizer, quant_op_name, quantizer_type)
            quantizer_dict[quant_op_name] = quantizer_info

            bit_width = tf.Variable(initial_value=bit_width,
                                    name=quant_op_name + '_bit_width',
                                    trainable=False, dtype=tf.int8)

            encoding_min, encoding_max = self._create_encoding_min_max_vars(quant_op_name)

            # Note: Later, is_symmetric_encoding value is to be read from config file
            use_symmetric_encoding = tf.Variable(initial_value=False,
                                                 name=quant_op_name + '_use_symmetric_encoding',
                                                 trainable=False, dtype=tf.bool)

            self.session.run([op_mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                              encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer])

        return op_mode_var, tensor_quant_ref, encoding_min, encoding_max, bit_width, use_symmetric_encoding

    def _insert_param_quantizer_recurrent(self, inner_ops, preceeding_tensor, quant_op_name: str,
                                          op_mode: libpymo.QuantizationMode,
                                          quantizer_dict: Dict[str, QuantizerInfo], quantizer_type: QuantizerType,
                                          bit_width: int = 8):
        """
        Create and insert a post-training quant op after a given tensor
        :param preceeding_tensor: Preceeding tensor to insert the quant op after
        :param quant_op_name: Name to give to the new quant op
        :param op_mode: Starting mode to configure for the new quant op
        :param quantizer_dict: dictionary of op and QuantizerInfo
        :param quantizer_type : indicate param or activation quantizer
        :param bit_width : bit-width to be used (output or param quantization bit-width), default set to 8.
        :return: None
        """
        # pylint: disable=too-many-locals
        # Create variables for op_mode, tensor_quantizer_reference, encoding_min, encoding_max, bitwidth and
        # is_symmetric_encoding flag
        # (so we can change these in the future, if needed)

        op_mode_var, tensor_quant_ref, encoding_min, encoding_max, bit_width, use_symmetric_encoding = \
            self._create_and_init_quant_op_input_vars(quant_op_name, quantizer_dict, quantizer_type, op_mode,
                                                      bit_width)

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
                                       bit_width: int = 8):
        """
        Create and insert a post-training quant op after a given tensor
        :param preceeding_tensor: Preceeding tensor to insert the quant op after
        :param quant_op_name: Name to give to the new quant op
        :param op_mode: Starting mode to configure for the new quant op
        :param quantizer_dict: dictionary of op and QuantizerInfo
        :param quantizer_type : indicate param or activation quantizer
        :param bit_width : bit-width to be used (output or param quantization bit-width), default set to 8.
        :return: None
        """
        # pylint: disable=too-many-locals
        # Create variables for op_mode, tensor_quantizer_reference, encoding_min, encoding_max, bitwidth and
        # is_symmetric_encoding flag
        # (so we can change these in the future, if needed)

        op_mode_var, tensor_quant_ref, encoding_min, encoding_max, bit_width, use_symmetric_encoding = \
            self._create_and_init_quant_op_input_vars(quant_op_name, quantizer_dict, quantizer_type, op_mode, bit_width)

        # CPU device assignment for QcQuantize op
        q_op_out = self._create_and_place_quantize_op(quant_op_name, preceeding_tensor, op_mode_var, tensor_quant_ref,
                                                      encoding_min, encoding_max, bit_width, use_symmetric_encoding)

        return q_op_out

    def _create_and_place_quantize_op(self, quant_op_name: str, preceeding_tensor,
                                      op_mode_var: tf.Variable, tensor_quant_ref: tf.Variable,
                                      encoding_min: tf.Variable, encoding_max: tf.Variable, bit_width: tf.Variable,
                                      use_symmetric_encoding: tf.Variable):
        def create_quantize_op():
            op = qcops.qc_quantize(name=quant_op_name, in_tensor=preceeding_tensor,
                                   op_mode=op_mode_var, tensor_quantizer_reference=tensor_quant_ref,
                                   encoding_min=encoding_min, encoding_max=encoding_max,
                                   bit_width=bit_width, use_symmetric_encoding=use_symmetric_encoding
                                   )
            return op

        if not self._use_cuda:
            with tf.device('/cpu:0'):
                q_op_out = create_quantize_op()

        # GPU device assignment for QcQuantize op
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
        tensor_quantizer_ref = libpymo.PtrToInt64(quantizer_dict[q_op_name].tensor_quantizer)
        vars_with_value[q_op_name + '_quant_ref'] = tensor_quantizer_ref

    update_variables_with_values(quant_sim_sess, vars_with_value)


def save_checkpoint(quantsim: QuantizationSimModel, meta_path: str, file_name_prefix: str):
    """
    Saves a checkpoint of the QuantSim model which can be loaded at a later point to continue fine-tuning.
    See also load_checkpoint().

    :param quantsim: QuantizationSimModel to be saved
    :param meta_path: path to save the meta file
    :param file_name_prefix: filename prefix string
    :return: None

    """

    if not os.path.exists(meta_path):
        os.mkdir(meta_path)

    save_path = os.path.join(meta_path, file_name_prefix)

    # save the model with quant ops
    graph_saver.save_model_to_meta(quantsim.session, save_path)

    # save info in the quantsim object
    save_data_to_pickle_file(quantsim, meta_path, 'orig_quantsim_config')


def load_checkpoint(meta_path: str, file_name_prefix: str) -> QuantizationSimModel:
    """
    Loads QuantSim model from saved checkpoint and pickle files.

    :param meta_path path: to load meta from
    :param file_name_prefix: filename prefix string
    :return: returns new QuantSim object

    """

    #pylint: disable=protected-access

    # load saved session with quant ops
    new_sess = graph_saver.load_model_from_meta(meta_path=str(meta_path + '/' + file_name_prefix + '.meta'))

    # load quant sim model object with params from saved pickle data
    new_quant_sim = load_data_from_pickle_file(meta_path + '/orig_quantsim_config')

    # set session for the new quantsim object
    new_quant_sim.session = new_sess

    # update tensor references in the new quantsim object
    update_tensor_quantizer_references(new_sess, new_quant_sim._param_quantizers)
    update_tensor_quantizer_references(new_sess, new_quant_sim._activation_quantizers)

    return new_quant_sim
