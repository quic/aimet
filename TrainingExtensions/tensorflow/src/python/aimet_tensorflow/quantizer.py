# /usr/bin/env python2.7
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Quantization feature for TensorFlow """

import os
import json
from typing import List

# Import tensorflow and associated helpers
import tensorflow as tf

from tensorflow.contrib import graph_editor as ge
from tensorflow.python.framework import ops as tf_ops

# Import aimet specific modules
import libpytrext
import libpymo
from aimet_common.utils import AimetLogger
from aimet_tensorflow.common import core
from aimet_tensorflow.common import graph_eval
from aimet_tensorflow.common import op_defs

_QUANT_MODES = {'tf_enhanced': libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                'tf': libpymo.QuantizationMode.QUANTIZATION_TF}

_ROUNDING_MODES = {'nearest': libpymo.RoundingMode.ROUND_NEAREST,
                   'stochastic': libpymo.RoundingMode.ROUND_STOCHASTIC}

_log = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


def _load_ops():
    """
    Function which loads the quantization op library. In order to load a graph with
    custom quantization ops this must be called first as this provides tensorflow with
    the required op definitions.

    :return: Loaded library
    """
    return tf.load_op_library('libaimet_tf_ops.so')


# Load the aimet ops
_qcops = _load_ops()


@tf_ops.RegisterGradient("QcQuantizeDeprecated")
def _qc_dummy_quantized_grad(op, grad, a, b):
    # pylint: disable=unused-argument
    """
    Dummy function to allow QcQuantize op to not do any gradient calculations
    :param op:
    :param grad:
    :param a:
    :param b:
    :return:
    """
    return grad, None


def _reset_session(sess):
    """
    Helper to reset a TF session
    :param sess: TF session
    :return: None
    """
    tf.reset_default_graph()
    sess.close()


def _load_graph(graph, meta_graph, checkpoint):
    """
    Load a TF graph given the meta and checkpoint files
    :param graph: Graph to load into
    :param meta_graph: Meta file
    :param checkpoint: Checkpoint file
    :return: Newly created TF session
    """
    _log.info('Loading graph: %s', meta_graph)
    sess = tf.Session(graph=graph)

    # Open the graph and restore the parameters
    saver = tf.train.import_meta_graph(meta_graph)
    saver.restore(sess, checkpoint)

    # Initialize any uninitialized variables
    graph_eval.initialize_uninitialized_vars(sess)

    return sess, saver


def _set_activation_encodings(sess, encodings, gpu):
    bitwidth = encodings['activation_bitwidth']
    op_encodings = encodings['encodings']
    _log.info('Got encodings for the following ops: %s', op_encodings.keys())
    temp_is_train_variable = tf.Variable(initial_value=False, trainable=False, name='training_in_progress', dtype=tf.bool)
    sess.run(temp_is_train_variable.initializer)
    for op_name, enc_data in op_encodings.items():
        _log.info("Setting %s activation encoding", op_name)

        # Replace new activation quantization op with a new op in "CONFIG_TYPE_GET_ENCODING" mode
        # CPU device assignment for QcQuantize op
        if not gpu:
            with tf.device('/cpu:0'):
                q_op_out = _qcops.qc_quantize_deprecated(op_name=op_name, training_in_progress=temp_is_train_variable,
                                                         config=int(libpytrext.config_type.CONFIG_TYPE_SET_ENCODING),
                                                         bitwidth=bitwidth, in_tensors=[[]], fixed_enc_mins=enc_data[0],
                                                         fixed_enc_maxs=enc_data[1])
        # GPU device assignment for QcQuantize op
        else:
            q_op_out = _qcops.qc_quantize_deprecated(op_name=op_name, training_in_progress=temp_is_train_variable,
                                                     config=int(libpytrext.config_type.CONFIG_TYPE_SET_ENCODING),
                                                     bitwidth=bitwidth, in_tensors=[[]], fixed_enc_mins=enc_data[0],
                                                     fixed_enc_maxs=enc_data[1])
        sess.run(q_op_out[0])


def load_quantized_graph(meta_graph, checkpoint, encodings, graph=None, gpu=True):
    """
    Function to call to setup the saved quantization encodings and model. When loading a quantized graph
    from saved files the quantizer must first be initialized with the quantization op names
    and the saved encodings.

    :param meta_graph: Path to meta file
    :param checkpoint: Path to checkpoint file
    :param encodings: Path to encodings file
    :param graph: Graph to load into
    :param gpu: If True, use GPU ops
    :return: Newly created TF session
    """
    comp_mode = libpymo.ComputationMode.COMP_MODE_GPU if gpu else libpymo.ComputationMode.COMP_MODE_CPU

    # Check to see if it's a file passed in and we need to process it, or if it's the
    # actual map data
    if isinstance(encodings, str):
        with open(encodings, 'r') as f:
            encodings = json.load(f)

    quant_mode = encodings['quant_mode']
    if quant_mode not in _QUANT_MODES:
        raise ValueError('Invalid quantization mode: '+quant_mode)
    quant_mode = _QUANT_MODES[quant_mode]

    libpytrext.ResetQuantizer()
    libpytrext.InitQuantizer(list(encodings.keys()), comp_mode, [], quant_mode)

    g = tf.Graph()
    with g.as_default():
        sess = tf.Session(graph=g)
        _set_activation_encodings(sess, encodings, gpu=gpu)

    # Use the provided graph, if it exists
    if not graph:
        graph = tf.Graph()
    with graph.as_default():
        sess, _ = _load_graph(graph, meta_graph, checkpoint)

    return sess


class Quantizer:
    """
    The Quantizer class enables quantization of a tensorflow model by analyzing data run through
    the network and calculating the optimal quantization encodings based on the provided algorithm
    and bit width.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, graph, checkpoint, output_file='./quantize/q_graph', quant_mode='tf_enhanced',
                 round_mode='nearest', op_map=None, ops_to_ignore=None, gpu=True, debug=False, skip_output=False,
                 skip_bias=True):
        """
        :param graph: The input meta graph to add quantization ops to
        :param checkpoint: The checkpoint file for the given graph
        :param output_file: The file path for saving the compressed tensorflow graph
        :param quant_mode: Indicates which quantization algorithm should be used, either
                'tf' or 'tf_enhanced'. Defaults to 'tf_enhanced'.
        :param round_mode: The round scheme to used. One of: 'nearest' or 'stochastic'. Default
                is 'nearest'.
        :param op_map: A map representing the op sequences to identify and quantize. See op_defs.py
                for an example of the formatting required.
        :param ops_to_ignore: A list of op names to ignore when selecting quantization ops
        :param gpu: Indicates on which hardware which the quantization algorithm should run. Currently
                defaults to CPU (False). To use GPU pass True (currently broken).
        :param debug: Indicates whether debug information should be printed or not. Defaults to False.
        :param skip_output: If output quantization is to be turned off. Default to False
        :param skip_bias: If bias quantization is to be turned off. Default to True
        :returns: An object which can be used to perform quantization on a tensorflow graph
        :raises: ValueError: An error occurred processing one of the input parameters.
        """

        # pylint: disable=too-many-arguments
        self._log = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
        self._debug = debug
        self._default_meta_graph = graph
        self._default_checkpoint = checkpoint
        self._output_file = output_file
        self._output_dir = os.path.dirname(output_file)
        self._skip_output = skip_output
        self._skip_bias = skip_bias
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        self._log.info('Saving quantized model as: %s', output_file)

        if op_map:
            self._op_map = op_map
        else:
            self._op_map = op_defs.default_op_map

        if not ops_to_ignore:
            ops_to_ignore = []
        self._ops_to_ignore = ops_to_ignore

        if quant_mode not in _QUANT_MODES:
            raise ValueError('Invalid quantization mode: '+quant_mode)
        self._quant_mode = _QUANT_MODES[quant_mode]
        self._quant_mode_str = quant_mode.upper()

        if round_mode not in _ROUNDING_MODES:
            raise ValueError('Invalid rounding mode: '+round_mode)
        self._round_mode_str = round_mode.upper()

        self._comp_mode = libpymo.ComputationMode.COMP_MODE_GPU if gpu else libpymo.ComputationMode.COMP_MODE_CPU
        self._gpu = gpu
        self._quant_act_ops = []
        self._activation_encodings = {'quant_mode': quant_mode,
                                      'encodings': {},
                                      'activation_bitwidth': 8}
        self._is_train_variable = None
        self._input_tensor_names = None

        # Todo: Need to look at these attributes and see how to handle them better
        # Very likely these attributes don't need to be object attributes
        self._saver = None
        self._bw_acts = None
        self._bw_params = None
        self._forward_callback = None
        self._sess = None
        self._iterations = None

    @staticmethod
    def _get_quantized_name(name):
        return name+'_quantized'

    @staticmethod
    def _get_prequantized_name(name):
        return name[:name.rfind("_quantized")]

    def _insert_weight_quantization_ops(self, ops, indices):
        if (not ops) or (len(ops) != len(indices)):
            raise ValueError('No weights to quantize!')

        self._is_train_variable = tf.Variable(initial_value=False, name='training_in_progress', dtype=tf.bool)
        self._sess.run(self._is_train_variable.initializer)

        for op, index in zip(ops, indices):

            # Modify the weight/bias inputs to use the quantized inputs
            param_in = op.inputs[index]
            self._log.debug('Quantizing input: %s for op: %s', param_in.name, op.name)
            # Rename using scope to be clearer what the op is quantizing. If no scope exists, use the default name
            w_op_name = os.path.split(param_in.name)[0]
            if not w_op_name:
                w_op_name = op.name
            w_op_name = self._get_quantized_name(w_op_name)
            self._log.info("Adding weight quantization op %s", w_op_name)
            # CPU device assignment for QcQuantize op
            if not self._gpu:
                with tf.device('/cpu:0'):
                    q_op_out = _qcops.qc_quantize_deprecated(name=w_op_name, op_name=w_op_name,
                                                             training_in_progress=self._is_train_variable,
                                                             config=int(libpytrext.config_type.CONFIG_TYPE_Q_DQ_PARAMS),
                                                             bitwidth=self._bw_params, in_tensors=[param_in],
                                                             fixed_enc_mins=[], fixed_enc_maxs=[],
                                                             quant_mode=self._quant_mode_str,
                                                             round_mode=self._round_mode_str, num_tensors=1)
            # GPU device assignment for QcQuantize op
            else:
                q_op_out = _qcops.qc_quantize_deprecated(name=w_op_name, op_name=w_op_name,
                                                         training_in_progress=self._is_train_variable,
                                                         config=int(libpytrext.config_type.CONFIG_TYPE_Q_DQ_PARAMS),
                                                         bitwidth=self._bw_params, in_tensors=[param_in],
                                                         fixed_enc_mins=[], fixed_enc_maxs=[],
                                                         quant_mode=self._quant_mode_str,
                                                         round_mode=self._round_mode_str, num_tensors=1)

            nodes_modified_count = ge.reroute_ts(tf_ops.convert_to_tensor(q_op_out[0][0]), param_in, can_modify=op)
            if nodes_modified_count != 1:
                raise ValueError('Input '+param_in.name+' not quantized!')

    def _insert_activation_quantization_ops(self, ops, collect_stats=True):

        # pylint: disable=too-many-locals
        encodings = self._activation_encodings['encodings']
        if encodings:
            self._log.info("Using fixed activation encodings")

        # Add all the activation quantization operations
        for op in ops:
            op_name = self._get_quantized_name(op.name)
            self._log.info("Adding quantization activation %s for %s", op_name, op.name)

            # When fixed encodings are set we aren't collecting stats, so use the collected encodings
            enc_mins, enc_maxs = [], []
            config = int(libpytrext.config_type.CONFIG_TYPE_UPDATE_STATS)
            if not collect_stats:
                if not encodings:
                    raise RuntimeError('No activation encodings recorded for activation quantization ops!')
                if op_name not in encodings:
                    raise RuntimeError("Can't find activation encoding for: "+op_name)
                config = int(libpytrext.config_type.CONFIG_TYPE_Q_DQ_ACTIVATIONS)
                encoding_tuple = encodings[op_name]
                enc_mins = encoding_tuple[0]
                enc_maxs = encoding_tuple[1]
                self._log.debug("Using min,max encodings: %s,%s", enc_mins, enc_maxs)

            # Add the new activation quantization op and reroute the outputs from the producer node to the
            # quantization op and the quantization outputs to the consumer(s)
            inputs = [output for output in op.outputs]
            num_tensors = len(inputs)
            consumers = []
            for inp in inputs:
                for consumer in inp.consumers():
                    if 'gradients' not in consumer.name:
                        consumers.append(consumer)

            # CPU device assignment for QcQuantize op
            if not self._gpu:
                with tf.device('/cpu:0'):
                    q_op_out = _qcops.qc_quantize_deprecated(name=op_name, op_name=op_name,
                                                             training_in_progress=self._is_train_variable,
                                                             config=config, bitwidth=self._bw_acts, in_tensors=inputs,
                                                             fixed_enc_mins=enc_mins, fixed_enc_maxs=enc_maxs,
                                                             quant_mode=self._quant_mode_str,
                                                             round_mode=self._round_mode_str, num_tensors=num_tensors)

            # GPU device assignment for QcQuantize op
            else:
                q_op_out = _qcops.qc_quantize_deprecated(name=op_name, op_name=op_name,
                                                         training_in_progress=self._is_train_variable,
                                                         config=config, bitwidth=self._bw_acts, in_tensors=inputs,
                                                         fixed_enc_mins=enc_mins, fixed_enc_maxs=enc_maxs,
                                                         quant_mode=self._quant_mode_str,
                                                         round_mode=self._round_mode_str, num_tensors=num_tensors)
            qc_outputs = [tf_ops.convert_to_tensor(q_op_out[i][0]) for i in range(len(inputs))]
            num_rerouted_outputs = ge.reroute_ts(qc_outputs, inputs, can_modify=consumers)
            if num_rerouted_outputs != len(consumers):
                raise ValueError('Failed to map ' + str(len(consumers)) + ' quantization output(s). Only mapped ' +
                                 str(num_rerouted_outputs))
            # Save the activation quantization op name for later
            if collect_stats:
                self._quant_act_ops.append(op_name)

    def _retrieve_activation_encodings(self, sess):
        """
        Retrieve activation encodings from PyMo library
        :param sess: TF session
        :return:
        """
        for op_name in self._quant_act_ops:
            self._log.info("Retrieving %s quantization activation encodings", op_name)
            prequantized_name = self._get_prequantized_name(op_name)
            prequantized_op = sess.graph.get_operation_by_name(prequantized_name)
            num_tensors = len(prequantized_op.outputs)
            # Replace new activation quantization op with a new op in "CONFIG_TYPE_GET_ENCODING" mode
            is_train_variable = self._sess.graph.get_tensor_by_name('training_in_progress:0')
            # CPU device assignment for QcQuantize op
            if not self._gpu:
                with tf.device('/cpu:0'):
                    q_op_out = _qcops.qc_quantize_deprecated(op_name=op_name, training_in_progress=is_train_variable,
                                                             config=int(libpytrext.config_type.CONFIG_TYPE_GET_ENCODING),
                                                             bitwidth=self._bw_acts, in_tensors=[[]],
                                                             fixed_enc_mins=[], fixed_enc_maxs=[],
                                                             num_tensors=num_tensors)
            # GPU device assignment for QcQuantize op
            else:
                q_op_out = _qcops.qc_quantize_deprecated(op_name=op_name, training_in_progress=is_train_variable,
                                                         config=int(libpytrext.config_type.CONFIG_TYPE_GET_ENCODING),
                                                         bitwidth=self._bw_acts, in_tensors=[[]],
                                                         fixed_enc_mins=[], fixed_enc_maxs=[], num_tensors=num_tensors)

            enc_mins = tf_ops.convert_to_tensor(q_op_out[1]).eval(session=sess).tolist()
            enc_maxs = tf_ops.convert_to_tensor(q_op_out[2]).eval(session=sess).tolist()
            encodings = self._activation_encodings['encodings']
            encodings[op_name] = (enc_mins, enc_maxs)
            self._log.info('Got %s encodings (mins,maxs): %s', op_name, encodings[op_name])

    def _prepare_graph_for_quantization(self, collect_stats=True):
        """
        Inserts the appropriate quantization ops and prequantizes the params depending upon the
        configuration parameters. Operations are inserted in the current default graph.
        Raises:
            RuntimeError: Thrown when there was an error inserting operations
        :param collect_stats: If True, stats are collected
        :return:
        """

        # Get the op query module
        query = core.OpQuery(self._sess.graph, op_map=self._op_map, ops_to_ignore=self._ops_to_ignore)

        # Query the known op groups and insert quantization nodes after the ops
        # Should we also be including quantization ops starting with labels? No for now...
        activation_ops = query.get_known_ops(inputs=self._input_tensor_names)

        # Query all ops with weights and quantize the input weights
        weight_ops = query.get_weight_ops(skip_bias_op=self._skip_bias)
        input_indices = query.get_weight_inputs(weight_ops)

        # Instantiate DlQuantization object
        quant_node_names = [self._get_quantized_name(op.name) for op in activation_ops]
        libpytrext.ResetQuantizer()
        libpytrext.InitQuantizer(quant_node_names, self._comp_mode, [], self._quant_mode)

        # Add quantization ops/data
        self._insert_weight_quantization_ops(weight_ops, input_indices)
        if not self._skip_output:
            self._insert_activation_quantization_ops(activation_ops, collect_stats)

    def _analyze_and_prepare_graph(self):
        """
        Runs the original graph and records the baseline performance. Inserts quantization ops and prequantizes
        the params depending upon the configuration parameters. Saves the graph with added quantization ops.

        Raises:
            RuntimeError: Thrown when there was an error evaluating/analyzing the graph or inserting operations
        :return: Path to file where the quantized graph is stored
        """
        # Get baseline model performance
        g = tf.Graph()
        with g.as_default():
            self._sess, self._saver = _load_graph(g, self._default_meta_graph, self._default_checkpoint)

            # Detect and insert quantization nodes for fixed point analysis
            self._prepare_graph_for_quantization()
            tmp_output = os.path.join(self._output_dir, 'tmp_quantized_graph')
            self.save_quantized_graph(tmp_output)

        _reset_session(self._sess)
        return tmp_output

    def _collect_quantization_stats(self, graph):
        """
        Collect quantization stats for provided graph
        :param graph: TF graph
        :return:
        """
        self._log.info('Collecting quantization stats')
        g = tf.Graph()
        with g.as_default():
            self._sess, self._saver = _load_graph(g, graph+'.meta', graph)
            self._forward_callback(self._sess, self._iterations)
            self._retrieve_activation_encodings(self._sess)
        _reset_session(self._sess)

    def _finalize_and_analyze_graph(self):
        """
        Create and save a final quantized graph
        :return: None
        """
        g = tf.Graph()
        with g.as_default():
            # Load and modify the original graph
            self._sess, self._saver = _load_graph(g, self._default_meta_graph, self._default_checkpoint)
            self._prepare_graph_for_quantization(collect_stats=False)
            self.save_quantized_graph(self._output_file)
        _reset_session(self._sess)

        # Load and analyze the final quantized graph
        g = tf.Graph()
        with g.as_default():
            self._sess = load_quantized_graph(self._output_file+'.meta', self._output_file,
                                              encodings=self._activation_encodings, graph=g, gpu=self._gpu)

    def save_quantized_graph(self, output_graph):
        """
        Saves the quantized graph and, if available, the output encodings used for training/inference.

        :param output_graph: The file path for saving the quantized tensorflow graph.
        :return: None
        :raises: RuntimeError: Either a quantized graph doesn't exist or there aren't permissions to write to
                the provided  location. A string will be passed which indicates the exact error.
        """
        self._log.info('Saving quantized graph: %s', output_graph)
        self._saver.save(self._sess, output_graph)
        _ = tf.summary.FileWriter(os.path.dirname(output_graph)+"/summary", self._sess.graph)

        # Check to see if encodings are saved and save the entire dictionary
        encodings = self._activation_encodings['encodings']
        if encodings:
            encoding_output = output_graph+'.encodings'
            self._log.info('Saving activation encodings to: %s', encoding_output)
            with open(encoding_output, 'w') as f:
                json.dump(self._activation_encodings, f, sort_keys=True, indent=4)

    def quantize_net(self, input_tensor_names: List[str], forward_callback, bw_params=8, bw_acts=8, iterations=100):
        """
        Quantizes the network based on the parameters set during Quantizer construction

        The quantizer performs all quantization steps automatically, however you are limited to specific
        network types which use a single input and label set as well as using the Tensorflow records format.
        The steps are:
        1. Update the network for quantization (insert quantization operations)
        2. Quantize the parameters (weights, biases, etc)
        3. Run data through the network collecting statistics for activation quantization data
        4. Generate the encodings for the network
        5. Generate a new quantized network that can be fine tuned.

        :param input_tensor_names: List of tensor names that are used to feed input data to the model
        :param forward_callback: Callback function that should run forward-passes on the model using representative
                data. The expected signature of this callback should be forward_callback(session, iterations)
        :param bw_params: The bit width to use for quantizing the parameters in the network. Default
                is 8 bits.
        :param bw_acts: The bit width to use for quantizing the activations in the network. Default
                is 8 bits.
        :param iterations: The number of iterations (data batches) to run through the network for analysis
        :return: The quantized graph object

        :raises: - ValueError: An invalid parameter was passed
                 - RuntimeError: An error occurred analyzing or compressing the network. The associated error
                   and other information will be returned with the error.
        """
        # pylint: disable=too-many-arguments

        # Cache and validate parameters
        self._input_tensor_names = input_tensor_names
        self._forward_callback = forward_callback

        if iterations <= 0:
            raise ValueError('Invalid iterations: '+str(iterations)+'. Number of iterations must be > 0')
        self._iterations = iterations

        if bw_params < 1:
            raise ValueError('Parameter bitwidth must be a valid value > 0, not '+str(bw_params))
        self._bw_params = bw_params

        if bw_acts < 1:
            raise ValueError('Activation bitwidth must be a valid value > 0, not '+str(bw_acts))
        self._bw_acts = bw_acts

        self._activation_encodings['activation_bitwidth'] = bw_acts

        # Get the baseline model performance and prep the graph for quantization
        tmp_output = self._analyze_and_prepare_graph()

        # Collect quantization statistics and save them
        self._collect_quantization_stats(tmp_output)

        # Finalize the quantized graph and analyze the performance
        self._finalize_and_analyze_graph()
        return self._sess
