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

from typing import List, Union, Dict, Callable, Any
import os
import shutil
import json
import math

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorflow.contrib import graph_editor

from aimet_common.defs import QuantScheme
from aimet_common.utils import AimetLogger
from aimet_tensorflow.utils.common import update_variables_with_values
from aimet_tensorflow.utils import graph_saver
from aimet_tensorflow.common import core
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.common.operation import Op
from aimet_tensorflow.quantsim_config.quantsim_config import QuantSimConfigurator

import libpymo as pymo

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
WORKING_DIR = '/tmp/quantsim/'


# by default we will have this registered for Qc Quantize op.
@tf_ops.RegisterGradient("QcQuantize")
def _qc_dummy_quantized_grad(op, grad):
    # pylint: disable=unused-argument
    """
    Dummy function to allow QcQuantize op to not do any gradient calculations
    :param op: quantize op
    :param grad: gradient
    :return: gradients computed per input
    """
    return grad, None, None, None, None, None, None


class QuantizerInfo:
    """
    Holds information about a given MO quantizer object
    """

    __slots__ = ['tensor_quantizer', 'quant_op_name']

    def __init__(self, tensor_quantizer: pymo.TensorQuantizer, quant_op_name: str):
        self.tensor_quantizer = tensor_quantizer
        self.quant_op_name = quant_op_name

    def get_op_mode(self, session: tf.Session) -> pymo.TensorQuantizerOpMode:
        """
        Get op mode for this quantizer
        :param session: Session in which the quant-op is added
        :return: Op mode
        """
        op = session.graph.get_operation_by_name(self.quant_op_name)
        op_mode_tensor = op.inputs[1]
        return session.run(op_mode_tensor)


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


class QuantizationSimModel:
    """
    |

    Creates a QuantSim model by adding quantization simulations ops to a given model.

    This enables

    #. off-target simulation of inference accuracy
    #. the model to be fine-tuned to counter the effects of quantization

    """
    # pylint: disable=too-many-arguments
    def __init__(self, session: tf.Session, starting_op_names: List[str], output_op_names: List[str],
                 quant_scheme: Union[str, QuantScheme] = 'tf_enhanced', rounding_mode: str = 'nearest',
                 default_output_bw: int = 8, default_param_bw: int = 8, use_cuda: bool = True, config_file: str = None):
        """
        :param session: The input model as session to add quantize ops to
        :param starting_op_names: Names of the input ops of the model graph
        :param output_op_names: Names of the output ops of the model graph
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
        self._default_output_bw = default_output_bw
        self._default_param_bw = default_param_bw
        self._use_cuda = use_cuda
        self._param_quantizers = {}
        self._activation_quantizers = {}
        self._op_to_quant_ops_dict = {}
        self._conn_graph = ConnectedGraph(self.session.graph, starting_op_names, output_op_names)

        # We save a copy of the original model (to be used during export later)
        with self.session.graph.as_default():
            saver = tf.train.Saver()
        saver.save(self.session, save_path=WORKING_DIR+'orig_model_before_quantsim')

        self._add_quant_nodes()

        # Save and load the session so the graph changes can take effect
        self._save_and_load_sim_model()

        # Use config file to set starting quantize op configurations
        # Note: at this point, the session used to construct self._conn_graph is different than the current
        # self.session, however we still use the connected graph to traverse the graph structure.
        if config_file:
            QuantSimConfigurator(self.session, self._conn_graph, self._op_to_quant_ops_dict, config_file)

    def _set_op_input_variables(self, op_name: str, encoding: pymo.TfEncoding, op_mode: pymo.TensorQuantizerOpMode):
        """
        Helper function that set op's input params.
        :param op_name: Name of the quantize op
        :param encoding: Encodings computed for given op
        :return: None , sets op's input variable values.
        """
        # Set the op mode
        with self.session.graph.as_default():
            vars_with_value = {}
            quant_op = self.session.graph.get_operation_by_name(op_name)
            vars_with_value[quant_op.name + '_encoding_min'] = encoding.min
            vars_with_value[quant_op.name + '_encoding_max'] = encoding.max
            vars_with_value[quant_op.name + '_op_mode'] = int(op_mode)
            update_variables_with_values(self.session, vars_with_value)

    def compute_encodings(self, forward_pass_callback: Callable[[tf.Session, Any], None],
                          forward_pass_callback_args):
        """
        Computes encodings for all quantization sim nodes in the model

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

        # For activations, calculate encodings and update min-max parameters
        for op_name, quantizer_info in self._activation_quantizers.items():

            # Calculate encodings
            if quantizer_info.get_op_mode(self.session) != int(pymo.TensorQuantizerOpMode.passThrough):
                encoding = quantizer_info.tensor_quantizer.computeEncoding()
                self._set_op_input_variables(op_name, encoding, pymo.TensorQuantizerOpMode.quantizeDequantize)

        # For post-training mode, params will always be in one-shot mode
        op_mode = QuantizationSimModel._param_op_mode_after_analysis(self._quant_scheme)

        for op_name, quantizer_info in self._param_quantizers.items():

            if quantizer_info.get_op_mode(self.session) != int(pymo.TensorQuantizerOpMode.passThrough):
                encoding = quantizer_info.tensor_quantizer.computeEncoding()
                self._set_op_input_variables(op_name, encoding, op_mode)

    def export(self, path: str, filename_prefix: str):
        """
        This method exports out the quant-sim model so it is ready to be run on-target.

        Specifically, the following are saved

        1. The sim-model is exported to a regular tensorflow meta/checkpoint without any simulation ops

        2. The quantization encodings are exported to a separate JSON-formatted file that can
           then be imported by the on-target runtime (if desired)

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :return: None

        """

        vars_to_save = []
        with self.session.graph.as_default():
            for var in tf.global_variables():
                if not var.name[:-2].endswith(('_quantized', '_quantized_op_mode', '_quantized_quant_ref',
                                               '_quantized_encoding_min', '_quantized_encoding_max',
                                               '_quantized_bit_width', '_quantized_use_symmetric_encoding')):
                    vars_to_save.append(var)

            saver = tf.train.Saver(vars_to_save)
            saver.save(self.session, save_path=os.path.join(path, filename_prefix))
            shutil.copyfile(WORKING_DIR+'orig_model_before_quantsim.meta',
                            os.path.join(path, filename_prefix) + '.meta')

        self._export_encodings(os.path.join(path, filename_prefix) + '.encodings')

    @staticmethod
    def _param_op_mode_after_analysis(_) -> pymo.TensorQuantizerOpMode:
        """
        Returns op mode to use for parameters after encodings have been computed
        :return:
        """
        return pymo.TensorQuantizerOpMode.oneShotQuantizeDequantize

    def _export_encodings(self, encoding_file_path: str):

        variable_dict = {}
        with self.session.graph.as_default():
            for var in tf.global_variables():
                if var.name.endswith('_encoding_min:0') or var.name.endswith('_encoding_max:0'):
                    variable_dict[var.name] = var

        def read_min_max(op_name: str):
            min_var = variable_dict[op_name + '_encoding_min:0']
            max_var = variable_dict[op_name + '_encoding_max:0']
            return self.session.run([min_var, max_var])

        def calculate_delta_offset(min_val: float, max_val: float, bw: int):
            delta = (max_val - min_val) / (2 ** bw - 1)
            if delta == 0:
                delta = 1e-5
            offset = math.floor(-min_val / delta)
            return delta, offset

        def update_encoding_dict_entry(encoding_dict: Dict,
                                       quant_op_name: str, quantizer_info: QuantizerInfo):
            min_val, max_val = read_min_max(quant_op_name)
            delta, offset = calculate_delta_offset(min_val, max_val, quantizer_info.tensor_quantizer.bitwidth)
            op_name = self._get_unquantized_name(quant_op_name)
            encoding_dict[op_name] = {'min': min_val,
                                      'max': max_val,
                                      'scale': delta,
                                      'offset': offset,
                                      'bitwidth': quantizer_info.tensor_quantizer.bitwidth}

        param_encodings = {}
        for quant_op_name, quantizer_info in self._param_quantizers.items():
            if not quantizer_info.tensor_quantizer.isEncodingValid:
                continue
            update_encoding_dict_entry(param_encodings, quant_op_name, quantizer_info)

        activation_encodings = {}
        for quant_op_name, quantizer_info in self._activation_quantizers.items():
            if not quantizer_info.tensor_quantizer.isEncodingValid:
                continue
            update_encoding_dict_entry(activation_encodings, quant_op_name, quantizer_info)

        encodings_dict = {'param_encodings': param_encodings,
                          'activation_encodings': activation_encodings}

        with open(encoding_file_path, 'w') as encoding_fp:
            json.dump(encodings_dict, encoding_fp, sort_keys=True, indent=4)

    def _save_and_load_sim_model(self):
        self.session = graph_saver.save_and_load_graph(WORKING_DIR, self.session)

    def _add_quant_nodes(self):
        """
        Add quantization ops to the model
        """
        # Get the op query module
        query = core.OpQuery(self.session.graph, ops_to_ignore=None)

        # Query all ops with weights and quantize the input weights
        weight_ops = query.get_weight_ops()
        input_indices = query.get_weight_inputs(weight_ops)

        self._insert_weight_quantization_ops(weight_ops, input_indices, self._conn_graph)
        self._insert_activation_quantization_ops(self._conn_graph)

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

    def _insert_weight_quantization_ops(self, ops: List[tf.Operation], indices: List[int], conn_graph: ConnectedGraph):
        """
        Inserts quantization ops for individual parameters
        :param ops: List of ops whose parameters are being quantized
        :param indices: List of input indices (one-to-one for each entry in ops)
        :param conn_graph: Connected graph to lookup Ops for ops with parameters
        :return: None
        """
        assert len(ops) == len(indices)

        for op, index in zip(ops, indices):

            # Identify connected graph op corresponding to the weight op
            conn_graph_op = conn_graph.get_op_from_module_name(op.name)
            if not conn_graph_op:
                # Connected graph does not contain the op, so it is not in the active path.  Skip adding a weight
                # quantize op for this op.
                continue

            # Modify the weight/bias inputs to use the quantized inputs
            param_in = op.inputs[index]

            quant_op_name = self._get_quantized_name(param_in.op.name)
            _logger.info("Adding weight quantization op %s", quant_op_name)

            if op.type == 'BiasAdd':
                op_mode = pymo.TensorQuantizerOpMode.passThrough
                param_type = 'bias'
            else:
                op_mode = pymo.TensorQuantizerOpMode.oneShotQuantizeDequantize
                param_type = 'weight'

            q_op_out = self._insert_post_training_quant_op(param_in, quant_op_name,
                                                           op_mode, self._param_quantizers, self._default_param_bw)

            nodes_modified_count = graph_editor.reroute_ts(tf_ops.convert_to_tensor(q_op_out), param_in, can_modify=op)
            if nodes_modified_count != 1:
                raise ValueError('Input ' + param_in.name + ' not quantized!')

            # Add a mapping to the connected graph op for the newly created weight quantizer op
            self._add_op_to_quant_ops_dict_entry(q_op_out, conn_graph_op, True, param_type)

    def _insert_activation_quantization_ops(self, conn_graph: ConnectedGraph):
        """
        Inserts quantization ops at the outputs of given ops
        :param conn_graph: Connected graph containing ops to place quantization ops after
        :return: None
        """
        # Op types which we will not place quantize ops after
        op_types_to_ignore = {'branch', 'Flatten'}
        # Get a list of valid connected graph Ops to insert quantize ops after
        valid_ops = [op for op in conn_graph.get_all_ops().values() if op.type not in op_types_to_ignore]
        for conn_graph_op in valid_ops:
            # Get the last op in the connected graph Op
            output_op = conn_graph_op.output_op_node
            quant_op_name = self._get_quantized_name(output_op.name)
            _logger.info("Adding activation quantization op %s", quant_op_name)

            consumers = [consumer for consumer in output_op.outputs[0].consumers() if 'gradients' not in consumer.name]

            q_op_out = self._insert_post_training_quant_op(output_op.outputs[0], quant_op_name,
                                                           pymo.TensorQuantizerOpMode.updateStats,
                                                           self._activation_quantizers, self._default_output_bw)

            # Re-route
            num_rerouted_outputs = graph_editor.reroute_ts(tf_ops.convert_to_tensor(q_op_out),
                                                           output_op.outputs[0], can_modify=consumers)
            if num_rerouted_outputs != len(consumers):
                raise ValueError('Failed to map ' + str(len(consumers)) + ' quantization output(s). Only mapped ' +
                                 str(num_rerouted_outputs))

            # Map connected graph op to output qc quantize op
            self._add_op_to_quant_ops_dict_entry(q_op_out, conn_graph_op, False)

    def _add_op_to_quant_ops_dict_entry(self, qc_quantize_tensor: tf.Operation, conn_graph_op: Op, is_param: bool,
                                        param_type: str = ''):
        """
        Add an entry to the op_to_quant_ops_dict
        :param qc_quantize_tensor: Output tensor of qc quantize op to add to the dictionary
        :param conn_graph_op: Connected graph Op associated with the qc quantize op
        :param is_param: True if the qc quantize op was created for a parameter, False otherwise
        :param param_type: Type of parameter (defaults to empty string, unused for activation quantizers)
        """
        if is_param:
            if conn_graph_op in self._op_to_quant_ops_dict:
                param_quant_op_dict, _ = self._op_to_quant_ops_dict[conn_graph_op]
                if param_type in param_quant_op_dict:
                    param_quant_op_dict[param_type].add(qc_quantize_tensor.op)
                else:
                    param_quant_op_dict[param_type] = {qc_quantize_tensor.op}
            else:
                param_quant_op_dict = {param_type: {qc_quantize_tensor.op}}
                self._op_to_quant_ops_dict[conn_graph_op] = [param_quant_op_dict, None]
        else:
            if conn_graph_op in self._op_to_quant_ops_dict:
                self._op_to_quant_ops_dict[conn_graph_op][1] = qc_quantize_tensor.op
            else:
                self._op_to_quant_ops_dict[conn_graph_op] = [dict(), qc_quantize_tensor.op]

    def _insert_post_training_quant_op(self, preceeding_tensor, quant_op_name: str, op_mode: pymo.QuantizationMode,
                                       quantizer_dict: Dict[str, QuantizerInfo], bit_width: int = 8):
        """
        Create and insert a post-training quant op after a given tensor
        :param preceeding_tensor: Preceeding tensor to insert the quant op after
        :param quant_op_name: Name to give to the new quant op
        :param op_mode: Starting mode to configure for the new quant op
        :param quantizer_dict: dictionary of op and QuantizerInfo
        :param bit_width : bit-width to be used (output or param quantization bit-width), default set to 8.
        :return: None
        """
        # pylint: disable=too-many-locals
        # Create variables for op_mode, tensor_quantizer_reference, encoding_min, encoding_max, bitwidth and
        # is_symmetric_encoding flag
        # (so we can change these in the future, if needed)

        _quant_scheme_to_pymo = {QuantScheme.post_training_tf: pymo.QuantizationMode.QUANTIZATION_TF,
                                 QuantScheme.post_training_tf_enhanced: pymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                 QuantScheme.training_range_learning: pymo.QuantizationMode.QUANTIZATION_RANGE_LEARNING}

        with self.session.graph.as_default():
            op_mode_var = tf.Variable(int(op_mode),
                                      name=quant_op_name + '_op_mode', trainable=False,
                                      dtype=tf.int32)

            # Note: Last param of TensorQuantizer is a flag to indicate is_symmetric_encoding,
            # this value is to be read from config file
            tensor_quantizer = pymo.TensorQuantizer(bit_width, _quant_scheme_to_pymo[self._quant_scheme],
                                                    pymo.RoundingMode.ROUND_NEAREST,
                                                    False)
            quantizer_info = QuantizerInfo(tensor_quantizer, quant_op_name)
            tensor_quantizer = pymo.PtrToInt64(tensor_quantizer)
            tensor_quant_ref = tf.Variable(tensor_quantizer, name=quant_op_name + '_quant_ref',
                                           trainable=False, dtype=tf.int64)

            encoding_min = tf.Variable(initial_value=0.0,
                                       name=quant_op_name + '_encoding_min',
                                       trainable=True, dtype=tf.double)
            encoding_max = tf.Variable(initial_value=0.0,
                                       name=quant_op_name + '_encoding_max',
                                       trainable=True, dtype=tf.double)

            bit_width = tf.Variable(initial_value=bit_width,
                                    name=quant_op_name + '_bit_width',
                                    trainable=False, dtype=tf.int8)

            # Note: Later, is_symmetric_encoding value is to be read from config file
            use_symmetric_encoding = tf.Variable(initial_value=False,
                                                 name=quant_op_name + '_use_symmetric_encoding',
                                                 trainable=False, dtype=tf.bool)

            self.session.run([op_mode_var.initializer, tensor_quant_ref.initializer, encoding_min.initializer,
                              encoding_max.initializer, bit_width.initializer, use_symmetric_encoding.initializer])

        # Add to quantizer dict
        quantizer_dict[quant_op_name] = quantizer_info

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
                                   bit_width=bit_width, use_symmetric_encoding=use_symmetric_encoding)
            return op

        if not self._use_cuda:
            with tf.device('/cpu:0'):
                q_op_out = create_quantize_op()

        # GPU device assignment for QcQuantize op
        else:
            q_op_out = create_quantize_op()

        return q_op_out
