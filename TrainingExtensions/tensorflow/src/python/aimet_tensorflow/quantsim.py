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
import io
from enum import Enum
import shutil
import json

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorflow.contrib import graph_editor

from aimet_common.defs import QuantScheme
from aimet_common.quantsim import gate_min_max, calculate_delta_offset
from aimet_common.utils import AimetLogger
from aimet_tensorflow.utils.common import update_variables_with_values, save_data_to_pickle_file, \
    load_data_from_pickle_file, get_valid_ops
from aimet_tensorflow.utils import graph_saver
from aimet_tensorflow.utils.constants import QuantizeOpIndices
from aimet_tensorflow.utils.quantsim import create_op_to_quant_ops_dict
from aimet_tensorflow.common import core
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.quantsim_config.quantsim_config import QuantSimConfigurator

# this is required to associate gradient with QcQuantize op
from aimet_tensorflow import quantsim_straight_through_grad      # pylint: disable=unused-import
import libpymo

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
WORKING_DIR = '/tmp/quantsim/'

DTYPES_QUANTIZE_NOT_REQUIRED = [tf.dtypes.int8, tf.dtypes.uint8, tf.dtypes.int16, tf.dtypes.uint16,
                                tf.dtypes.int32, tf.dtypes.uint32, tf.dtypes.int64, tf.dtypes.uint64,
                                tf.bool, tf.dtypes.string]

quant_scheme_to_libpymo = {QuantScheme.post_training_tf: libpymo.QuantizationMode.QUANTIZATION_TF,
                           QuantScheme.post_training_tf_enhanced:
                           libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                           QuantScheme.training_range_learning_with_tf_init:
                           libpymo.QuantizationMode.QUANTIZATION_TF,
                           QuantScheme.training_range_learning_with_tf_enhanced_init:
                           libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED}


class QuantizerType(Enum):
    """ Enum for quantize op types """
    param = 0
    activation = 1

# Op types which we will not place quantize ops after
op_types_to_ignore = {'branch', 'Flatten', 'Shape'}


class PickleableTensorQuantizerState:
    """
    State variables to be saved while pickling tensor quantizer
    """
    def __init__(self, quant_op_name, tensor_quantizer_ref, quantizer_type):
        """
        class type to save pickle-able info pertaining to tensor quantizer
        :param quant_op_name: name of the quantize op
        :param tensor_quantizer_ref: TensorQuantizer reference
        :param quantizer_type : param or activation quantizer
        """

        self.quant_op_name = quant_op_name
        self.quant_scheme = tensor_quantizer_ref.getQuantScheme()
        self.rounding_mode = tensor_quantizer_ref.roundingMode
        self.is_encoding_valid = tensor_quantizer_ref.isEncodingValid
        self.quantizer_type = quantizer_type


class QuantizerInfo:
    """
    Holds information about a given MO Quantizer object and active session
    """

    __slots__ = ['session', 'tensor_quantizer', 'quant_op_name', 'quantizer_type']

    def __init__(self, session: tf.Session, tensor_quantizer: libpymo.TensorQuantizer,
                 quant_op_name: str, quantizer_type: QuantizerType):
        self.session = session
        self.tensor_quantizer = tensor_quantizer
        self.quant_op_name = quant_op_name
        self.quantizer_type = quantizer_type

    def set_variable(self, var_name, value):
        """
        sets Quantize op variable with value passed
        :param var_name: Name of the variable to be updated
        :param value: value to be assigned to the variable
        :return:
        """

        with self.session.graph.as_default():
            vars_with_given_name = [var for var in tf.global_variables()
                                    if var.op.name == var_name]
        var_to_be_updated = vars_with_given_name[0]
        var_to_be_updated.load(value, self.session)

    def get_variable_from_op(self, var_index):
        """
        Reads variable from Quantize op
        :param var_index: Quantize op input param index corresponding to the variable to be read
        :return: variable value read from the Quantize op
        """
        quantize_op = self.session.graph.get_operation_by_name(self.quant_op_name)
        op_var_tensor = quantize_op.inputs[var_index]
        return self.session.run(op_var_tensor)

    @property
    def bitwidth(self) -> int:
        """
        Reads bitwidth from the Quantize op
        :return: returns the bitiwdth associated with Quantize op
        """
        # return the variable value from op
        return self.get_variable_from_op(QuantizeOpIndices.bit_width)

    @bitwidth.setter
    def bitwidth(self, bitwidth: int):
        """
        Sets the bitwidth in the Quantize op
        :param bitwidth: value to be assigned to bitwidth variable
        :return:
        """

        var_name = self.quant_op_name + '_bit_width'
        self.set_variable(var_name, bitwidth)
        self.tensor_quantizer.isEncodingValid = False

    @property
    def use_symmetric_encoding(self) -> bool:
        """
        Reads use_symmetric_encoding flag in the Quantize op
        :return: use_symmetric_encoding config as bool
        """

        return self.get_variable_from_op(QuantizeOpIndices.use_symmetric_encoding)

    @use_symmetric_encoding.setter
    def use_symmetric_encoding(self, use_symmetric_encoding: bool):
        """
        Sets the use_symmetric_encoding flag in the Quantize op
        :param use_symmetric_encoding: value to be assigned to use_symmetric_encoding flag
        :return:
        """

        var_name = self.quant_op_name + '_use_symmetric_encoding'
        self.set_variable(var_name, use_symmetric_encoding)
        self.tensor_quantizer.isEncodingValid = False

    @property
    def quant_scheme(self) -> libpymo.QuantizationMode:
        """
        Reads the quant_scheme associated with the Quantize op
        :return: quant_scheme as libpymo.QuantizationMode type
        """
        return self.tensor_quantizer.getQuantScheme()

    @quant_scheme.setter
    def quant_scheme(self, quant_scheme: libpymo.QuantizationMode):
        """
        Sets the quant_scheme associated with the Quantize op
        :param quant_scheme: value to be assigned to quant_scheme param in Quantizer
        :return:
        """

        self.tensor_quantizer.setQuantScheme(quant_scheme_to_libpymo[quant_scheme])

    @property
    def rounding_mode(self) -> libpymo.RoundingMode:
        """
        Reads rounding_mode associated with the Quantize op
        :return: rounding_mode value as libpymo.RoundingMode type
        """

        return self.tensor_quantizer.roundingMode

    @rounding_mode.setter
    def rounding_mode(self, rounding_mode: libpymo.RoundingMode):
        """
        Sets the rounding_mode associated with the Quantize op
        :param rounding_mode: value to be assigned to rounding_mode param in Quantizer
        :return:
        """
        self.tensor_quantizer.isEncodingValid = False
        self.tensor_quantizer.roundingMode = rounding_mode

    def get_op_mode(self) -> libpymo.TensorQuantizerOpMode:
        """
        Reads op mode variable from Quantize op
        :return: Op mode as pymo.TensorQuantizerOpMode type
        """
        op = self.session.graph.get_operation_by_name(self.quant_op_name)
        op_mode_tensor = op.inputs[QuantizeOpIndices.op_mode]
        return self.session.run(op_mode_tensor)

    @property
    def enabled(self) -> bool:
        """
        Reads Quantize op flag that indicates if op is enabled or disabled
        :return: bool
        """
        is_enabled = True
        # return the variable value from op
        if self.get_op_mode() == int(libpymo.TensorQuantizerOpMode.passThrough):
            is_enabled = False
        return is_enabled

    @enabled.setter
    def enabled(self, enabled: bool):
        """
         Enables or disables given Quantize op if enabled is False
        :param enabled: boolean flag to indicate enable or disable
        :return:
        """

        # if disable is requested on the op and this op was not already in "passThrough" mode,
        # we will disable the op by marking it as "passThrough"
        if not enabled and self.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
            op_mode = int(libpymo.TensorQuantizerOpMode.passThrough)
            # update the isEncodingValid state to False
            self.tensor_quantizer.isEncodingValid = False
        # if enable is requested and this op was previously disabled
        # we enable the op by setting the initial op_mode that depends on the Quantizer type
        elif enabled and self.get_op_mode() == int(libpymo.TensorQuantizerOpMode.passThrough):
            if self.quantizer_type is QuantizerType.param:
                op_mode = int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize)
            elif self.quantizer_type is QuantizerType.activation:
                op_mode = int(libpymo.TensorQuantizerOpMode.updateStats)
            # update the isEncodingValid state to False
            self.tensor_quantizer.isEncodingValid = False

        var_name = self.quant_op_name + '_op_mode'
        self.set_variable(var_name, op_mode)

    def __getstate__(self):
        # convert tensor quantizer state to pickle-able form
        state = PickleableTensorQuantizerState(self.quant_op_name,
                                               self.tensor_quantizer,
                                               self.quantizer_type)
        return state

    def __setstate__(self, state):
        self.session = None
        # Create the cpp tensor quantizer reference
        self.quant_op_name = state.quant_op_name
        self.quantizer_type = state.quantizer_type
        self.tensor_quantizer = libpymo.TensorQuantizer(state.quant_scheme,
                                                        state.rounding_mode)
        self.tensor_quantizer.isEncodingValid = state.is_encoding_valid

    def __str__(self):
        stream = io.StringIO(newline='\n')
        stream.write('Quantizer Info:\n')
        stream.write(' quantize_op_name:{}\n quantizer_type:{}\n bitwidth={}\n use_symmetric_encoding={}\n'
                     ' round_mode={}\n quant_scheme={}\n enabled:{}\n'.format(self.quant_op_name,
                                                                              self.quantizer_type,
                                                                              self.bitwidth,
                                                                              self.use_symmetric_encoding,
                                                                              self.rounding_mode,
                                                                              self.quant_scheme,
                                                                              self.enabled))

        return stream.getvalue()


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
    def __init__(self, session: tf.Session, starting_op_names: List[str], output_op_names: List[str],
                 quant_scheme: Union[str, QuantScheme] = 'tf_enhanced', rounding_mode: str = 'nearest',
                 default_output_bw: int = 8, default_param_bw: int = 8, use_cuda: bool = True, config_file: str = None):
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
            saver = tf.train.Saver()
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

    def _set_op_input_variables(self, op_name: str, encoding: libpymo.TfEncoding, op_mode: libpymo.TensorQuantizerOpMode):
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

    def _get_op_variable_value(self, quant_op_name: str, var_index: int):
        """
        Utility to load variable values from quant op
        :param quant_op_name: quantize op name
        :param var_index: variable index to be read
        :return: variable value
        """

        op = self.session.graph.get_operation_by_name(quant_op_name)
        op_var_tensor = op.inputs[var_index]
        return self.session.run(op_var_tensor)

    def _get_bitwidth_and_symmetric_flag(self, quant_op_name: str) -> (int, bool):
        """
        utility to read bitwidth and symmetric encoding flag values from given Quantize op
        :param quant_op_name: Quantize op name
        :return: bitwidth and symmetric encoding flag
        """

        op_bitwidth = self._get_op_variable_value(quant_op_name,
                                                  int(QuantizeOpIndices.bit_width))
        op_use_symmetric_encodings = self._get_op_variable_value(quant_op_name,
                                                                 int(QuantizeOpIndices.use_symmetric_encoding))

        return op_bitwidth, op_use_symmetric_encodings

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
        op_to_quant_ops_dict = create_op_to_quant_ops_dict(self.session.graph, conn_graph, ops_with_param_names,
                                                           indices, activation_op_names)
        QuantSimConfigurator(self.session, conn_graph, op_to_quant_ops_dict, config_file)

    def compute_encodings(self, forward_pass_callback: Callable[[tf.Session, Any], None],
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

        # For activations, calculate encodings and update min-max parameters
        for op_name, quantizer_info in self._activation_quantizers.items():

            # Calculate encodings
            if quantizer_info.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
                op_bitwidth, op_use_symmetric_encodings = self._get_bitwidth_and_symmetric_flag(
                    quantizer_info.quant_op_name)
                encoding = quantizer_info.tensor_quantizer.computeEncoding(op_bitwidth, op_use_symmetric_encodings)
                self._set_op_input_variables(op_name, encoding, libpymo.TensorQuantizerOpMode.quantizeDequantize)

        # For post-training mode, params will always be in one-shot mode
        op_mode = QuantizationSimModel._param_op_mode_after_analysis(self._quant_scheme)

        for op_name, quantizer_info in self._param_quantizers.items():

            if quantizer_info.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
                op_bitwidth, op_use_symmetric_encodings = self._get_bitwidth_and_symmetric_flag(
                    quantizer_info.quant_op_name)
                encoding = quantizer_info.tensor_quantizer.computeEncoding(op_bitwidth, op_use_symmetric_encodings)
                self._set_op_input_variables(op_name, encoding, op_mode)

    def export(self, path: str, filename_prefix: str, orig_sess: tf.Session = None):
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
                saver = tf.train.Saver()
            saver.save(orig_sess, save_path=WORKING_DIR+'orig_model_before_quantsim')
        else:
            _logger.info('Original session is not provided, use orig_model_before_quantsim.meta to export')

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
            for var in tf.global_variables():
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
            min_val, max_val = self.read_min_max(quant_op_name, variable_dict)
            min_val, max_val = gate_min_max(min_val, max_val)
            op_bitwidth = int(self._get_op_variable_value(quant_op_name, int(QuantizeOpIndices.bit_width)))
            delta, offset = calculate_delta_offset(min_val, max_val, op_bitwidth)
            quant_op = self.session.graph.get_operation_by_name(quant_op_name)
            tensor_name = quant_op.inputs[0].name
            encoding_dict[tensor_name] = [{'min': min_val,
                                           'max': max_val,
                                           'scale': delta,
                                           'offset': offset,
                                           'bitwidth': op_bitwidth}]

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

        encodings_dict = {'param_encodings': param_encodings,
                          'activation_encodings': activation_encodings}

        with open(encoding_file_path, 'w') as encoding_fp:
            json.dump(encodings_dict, encoding_fp, sort_keys=True, indent=4)

    def _save_and_load_sim_model(self):
        self.session = graph_saver.save_and_load_graph(WORKING_DIR, self.session)
        update_tensor_quantizer_references(self.session, self._activation_quantizers)
        update_tensor_quantizer_references(self.session, self._param_quantizers)

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
        ops_with_param_names, input_indices = QuantizationSimModel.get_ops_to_quantize_params_for(self.session.graph,
                                                                                                  starting_op_names,
                                                                                                  output_op_names)

        # Get list of activation ops to insert quantizers for, and the connected graph used to obtain these ops
        activation_op_names, conn_graph = QuantizationSimModel.get_ops_to_quantize_activations_for(self.session.graph,
                                                                                                   starting_op_names,
                                                                                                   output_op_names)

        self._insert_param_quantization_ops(ops_with_param_names, input_indices, default_param_bw)
        self._insert_activation_quantization_ops(activation_op_names, default_output_bw)

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
    def get_ops_to_quantize_params_for(graph: tf.Graph, starting_op_names: List[str], output_op_names: List[str]) \
            -> Tuple[List[str], List[int]]:
        """
        Get names of ops to insert param quantizers for, as well as corresponding indices
        :param graph: Tensorflow graph to get names of ops to quantize weights for
        :param starting_op_names: List of starting op names of the model
        :param output_op_names: List of output op names of the model
        :return: Tuple consisting of list of op names with params to insert quantize ops for as well as list of indices
        of parameters for each op
        """
        # Get the op query module
        query = core.OpQuery(graph, ops_to_ignore=None)
        valid_ops = get_valid_ops(graph, starting_op_names, output_op_names)
        ops_with_param_names = [op.name for op in query.get_weight_ops() if op in valid_ops]
        ops_with_params = [graph.get_operation_by_name(op_name) for op_name in ops_with_param_names]
        input_indices = query.get_weight_inputs(ops_with_params)
        if len(ops_with_param_names) != len(input_indices):
            _logger.error("Length of ops with params and input indices differ")
            raise AssertionError
        return ops_with_param_names, input_indices

    @staticmethod
    def get_ops_to_quantize_activations_for(graph: tf.Graph, starting_op_names: List[str], output_op_names: List[str])\
            -> Tuple[List[str], ConnectedGraph]:
        """
        Get names of ops to insert activation quantizers for
        :param graph: Tensorflow graph to get names of ops to quantize weights for
        :param starting_op_names: List of starting op names of the model
        :param output_op_names: List of output op names of the model
        :return: List of op names to insert activation quantize ops for, and the connected graph used to obtain these
        ops.
        """
        conn_graph = ConnectedGraph(graph, starting_op_names, output_op_names)
        valid_ops = [op for op in conn_graph.get_all_ops().values() if op.type not in op_types_to_ignore]
        op_names_to_quantize = [conn_graph_op.output_op_node.name for conn_graph_op in valid_ops if
                                QuantizationSimModel._is_op_quantizable(conn_graph_op.output_op_node)]
        return op_names_to_quantize, conn_graph

    def _insert_param_quantization_ops(self, op_names: List[str], indices: List[int], default_param_bw: int):
        """
        Inserts quantization ops for individual parameters
        :param ops: List of ops whose parameters are being quantized
        :param indices: List of input indices (one-to-one for each entry in ops)
        :param default_param_bw : default param bitwidth
        :return: None
        """
        ops = [self.session.graph.get_operation_by_name(op_name) for op_name in op_names]
        assert len(ops) == len(indices)

        for op, index in zip(ops, indices):
            # Modify the weight/bias inputs to use the quantized inputs
            param_in = op.inputs[index]

            quant_op_name = self._get_quantized_name(param_in.op.name)
            _logger.info("Adding weight quantization op %s", quant_op_name)
            op_mode = libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize

            q_op_out = self._insert_post_training_quant_op(param_in, quant_op_name,
                                                           op_mode, self._param_quantizers, QuantizerType.param,
                                                           default_param_bw)

            nodes_modified_count = graph_editor.reroute_ts(tf_ops.convert_to_tensor(q_op_out), param_in, can_modify=op)
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

    def _insert_activation_quantization_ops(self, valid_op_names: List[str], default_output_bw):
        """
        Inserts quantization ops at the outputs of given ops
        :param valid_op_names: List of op names to insert activation quantizers for
        :param default_output_bw: default activation bitwidth
        """
        for op_name in valid_op_names:
            quant_op_name = self._get_quantized_name(op_name)
            op = self.session.graph.get_operation_by_name(op_name)
            _logger.info("Adding activation quantization op %s", quant_op_name)

            consumers = [consumer for consumer in op.outputs[0].consumers() if 'gradients' not in consumer.name]

            if not QuantizationSimModel._is_op_quantizable(op):
                _logger.error('Unsupported dtype {%s} detected for op {%s}.', op.outputs[0].dtype, op_name)
                raise AssertionError

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

        with self.session.graph.as_default():
            op_mode_var = tf.Variable(int(op_mode),
                                      name=quant_op_name + '_op_mode', trainable=False,
                                      dtype=tf.int32)

            # Note: Last param of TensorQuantizer is a flag to indicate is_symmetric_encoding,
            # this value is to be read from config file
            tensor_quantizer = libpymo.TensorQuantizer(quant_scheme_to_libpymo[self._quant_scheme],
                                                       libpymo.RoundingMode.ROUND_NEAREST)
            quantizer_info = QuantizerInfo(self.session, tensor_quantizer, quant_op_name, quantizer_type)
            tensor_quantizer = libpymo.PtrToInt64(tensor_quantizer)
            tensor_quant_ref = tf.Variable(tensor_quantizer, name=quant_op_name + '_quant_ref',
                                           trainable=False, dtype=tf.int64)

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


# load and save utilities
def update_tensor_quantizer_references(quant_sim_sess: tf.Session, quantizer_dict: Dict[str, QuantizerInfo]):
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
