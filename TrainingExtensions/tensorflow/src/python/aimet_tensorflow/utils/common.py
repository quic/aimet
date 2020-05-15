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
""" common TF utilities """
from typing import List, Union, Tuple, Dict, Set

import pickle
import os
import numpy as np
import tensorflow as tf
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


# List of associations between onnx types and tf connected graph types.
# Multiple onnx types may be associated with a tf connected graph type, and vice versa.
onnx_tf_conn_graph_type_pairs = [
    [["Conv"], ["Conv2D", "DepthwiseConv2dNative"]],
    [["BatchNormalization"], ["FusedBatchNormV3"]],
    [["MaxPool"], ["MaxPool"]],
    [["AveragePool"], ["AvgPool"]],
    [["Relu"], ["Relu", "Relu6"]],
    [["Gemm"], ["Dense"]],
    [["Add"], ["Add", "AddV2"]],
    [["Mean"], ["Mean"]],
    [["Pad"], ["Pad"]],
    [["Squeeze"], ["Squeeze"]],
]


def iter_first_x(dataset: tf.data.Dataset, num_batches: int):
    """
     Return a generator for the first num_batches batches in a given dataset
    :param dataset: tf.data.Dataset object
    :param num_batches: number of batches
    :return:
    """
    counter = 0

    # creating one shot iterator ops in same graph as dataset ops
    # TODO: this will keep adding iterator ops in the same graph every time this function is being called, need
    #  better solution (reinitializable iterator)
    # pylint: disable=protected-access
    with dataset._graph.as_default():

        iterator = dataset.make_one_shot_iterator()
        # create iterator graph element.  This does not actually get the data, but makes a tensor which must be
        # evaluated.
        next_element = iterator.get_next()

    # creating session with graph that has dataset and iterator ops
    with tf.Session(graph=dataset._graph) as sess:

        while counter < num_batches:
            try:
                batch_data = sess.run(next_element)

            except tf.errors.OutOfRangeError:
                # if no data remains in the dataset, instead of throwing out of range error,
                # treat it as if it were the end of the iteration
                logger.info('dataset ran out of elements, stopping after %s batches', counter)
                break

            yield batch_data
            counter += 1

    # close session
    sess.close()


def change_out_act_shape_to_channels_first(op: tf.Operation) -> List:
    """
    Convert TensorFlow Conv2d output shape 'channels_last' to 'channels_first'
    :return: shape [N, C, H, W]
    """
    data_format = op.get_attr('data_format')
    shape = op.outputs[0].get_shape().as_list()

    if str(data_format.decode("utf-8")) == "NHWC":
        shape = [shape[0], shape[3], shape[1], shape[2]]

    return shape


def get_succeeding_bias_op(op: tf.Operation) -> tf.Operation:
    """
    For given op, return following bias op if exists
    :param op: TF op
    :return: bias op if exists or None
    """

    bias = None
    for consumer in op.outputs[0].consumers():

        if consumer.type in ('Add', 'BiasAdd') and len(consumer.inputs[1].shape) == 1:
            bias = consumer

    return bias


def get_succeeding_bias_tensor(op: tf.Operation) -> tf.Tensor:
    """
    For given op, return following tensor if exists
    :param op: TF Op
    :return: bias Op if exists or None
    """

    bias = None
    for consumer in op.outputs[0].consumers():

        if consumer.type in ('Add', 'BiasAdd') and len(consumer.inputs[1].shape) == 1:
            bias = consumer.inputs[1]

    return bias


def create_input_feed_dict(graph: tf.Graph, input_op_names_list: List, input_data: Union[np.ndarray, List, Tuple],
                           training=False) -> Dict:
    """
    Creates feed dictionary [op_name] = data for session.run
    :param graph: tf.Graph
    :param input_op_names_list: list of input op names
    :param input_data: either single numpy array, list or tuple of numpy array
    :param training: True if graph is in training mode, false otherwise
    :return: feed_dict
    """

    feed_dict = {}

    # single input
    if isinstance(input_data, np.ndarray):
        input_data_list = [input_data]

    # list of multiple inputs
    elif isinstance(input_data, list):
        input_data_list = input_data

    # tuple of multiple inputs
    elif isinstance(input_data, tuple):
        input_data_list = list(input_data)

    else:
        raise ValueError('Session run return value should be either numpy array, list or tuple')

    if not len(input_op_names_list) == len(input_data_list):
        raise ValueError('There is mismatch between number of input op names and input data!')

    for inp_op_name, inp_data in zip(input_op_names_list, input_data_list):

        inp_tensor = graph.get_tensor_by_name(inp_op_name + ':0')
        feed_dict[inp_tensor] = inp_data

    # Identify and set all training tensors to True or False depending on training parameter
    for training_tensor in get_training_tensors(graph):
        feed_dict[training_tensor] = training

    return feed_dict


def get_padding(input_shape: tuple, output_shape: tuple, kernel_size: tuple, stride: tuple) -> tuple:
    """
    Get the equivalent padding for 'SAME' and 'VALID' type
    :param input_shape: input activation shape (height, width)
    :param output_shape: output activation shape (height, width)
    :param kernel_size: kernel size (k_h, k_w)
    :param stride: stride (height, width)
    :return: padding (height, width)
    """

    padding_height = max((output_shape[0] - 1) * stride[0] + kernel_size[0] - input_shape[0], 0)
    padding_width = max((output_shape[1] - 1) * stride[1] + kernel_size[1] - input_shape[1], 0)

    padding_height = padding_height // 2
    padding_width = padding_width // 2

    return padding_height, padding_width


def is_op_compressible(op: tf.Operation) -> bool:
    """
    Check if the given op is valid compressible op and not starts with _SKIPPED_PREFIXES
    :param op: TensorFlow Op
    :return: True if Op is valid, False otherwise
    """

    return op.type in ('Conv2D', 'MatMul') and not op.name.startswith('gradients/')


def get_ordered_ops(graph: tf.Graph, starting_op_names: List) -> List:
    """
    Function to get all the ops in graph based on occurrence by Depth First Traversal
    :param graph: tf.Graph
    :param starting_op_names: List of starting op names
    :return: ordered_ops: List of ops in order of occurrence
    """

    def add_children_ops_before_parent_op(current_op: tf.Operation, visited_ops: Set, ordered_ops: List):
        """
        Util function to add all the children ops in ordered_ops list before parent op using Depth First Traversal
        :param current_op: tf.Operation
        :param visited_ops: Set of ops visited so far (to cut short duplicate traversals)
        :param ordered_ops: List of ops in order of occurrence
        """
        # Add current op to visited_ops set
        visited_ops.add(current_op)

        # iterate all the output tensors of current op
        for output_tensor in current_op.outputs:
            # iterate all the consumer ops of output tensor
            for consumer_op in output_tensor.consumers():
                # add consumer op to visited_ops list if not added previously and recursively call
                if consumer_op not in visited_ops:
                    add_children_ops_before_parent_op(consumer_op, visited_ops, ordered_ops)

        # add to ordered_ops list only when all the children ops are traversed
        ordered_ops.append(current_op)

    #  Set of all ops that have been visited (to cut short duplicate traversals)
    visited_ops = set()

    # List of all ops in order of occurrence
    ordered_ops = []

    for starting_op_name in starting_op_names:

        starting_op = graph.get_operation_by_name(starting_op_name)

        add_children_ops_before_parent_op(starting_op, visited_ops, ordered_ops)

    # reverse the list because ops are in reverse order
    ordered_ops.reverse()

    return ordered_ops


def get_ordered_conv_linears(sess: tf.Session, input_op_names: List[str]) -> List[str]:
    """
    helper to select a list of candidate layers for bias correction
    :param sess: active tensorflow session as tf.Session type
    :param input_op_names: list of input op names
    :return: List of conv/linear layer names
    """

    # get ordered operations list in TF graph
    list_of_ordered_ops = get_ordered_ops(sess.graph, input_op_names)

    # look for conv layers
    ordered_convs = []
    for op in list_of_ordered_ops:
        if op.type in ['Conv2D', 'DepthwiseConv2dNative', 'MatMul'] and not op.name.startswith('gradients/'):
            ordered_convs.append(op)

    return ordered_convs


def get_valid_ops(graph: tf.Graph, starting_op_names: List[str], ending_op_names: List[str]) -> Set[tf.Operation]:
    """
    Get a set of valid ops.  Valid ops are ops which can be reached both by a DFS from a starting op, as well
    as upward DFS from an ending op.  If no ending ops are given, all ops reachable by DFS from starting ops are
    considered valid ops.
    DFS search both ways is needed since training ops will be seen by top down DFS but not bottom up, while parameters
    like weights and biases will be seen by bottom up search but not top down.  Taking the intersection allows us to
    leave these ops out.
    :param graph: Graph to search for valid ops
    :param starting_op_names: Ops to start top down DFS search from
    :param ending_op_names: Ending ops to start bottom up DFS search from
    """
    ops_from_starting_ops = set()
    ops_from_ending_ops = set()
    # For each starting op, do a DFS and add all child ops to ops_from_starting_ops set
    for name in starting_op_names:
        op = graph.get_operation_by_name(name)
        queue = [op]
        while queue:
            curr_op = queue.pop()
            ops_from_starting_ops.add(curr_op)
            for output in curr_op.outputs:
                for consumer in output.consumers():
                    if consumer not in ops_from_starting_ops:
                        queue.append(consumer)

    # For each ending op, do a DFS upwards and add all parent ops to ops_from_ending_ops set
    for name in ending_op_names:
        op = graph.get_operation_by_name(name)
        queue = [op]
        while queue:
            curr_op = queue.pop()
            ops_from_ending_ops.add(curr_op)
            for inp in curr_op.inputs:
                if inp.op not in ops_from_ending_ops:
                    queue.append(inp.op)
    # Only ops in the intersection of starting ops and ending ops sets are valid
    return ops_from_starting_ops.intersection(ops_from_ending_ops)


def get_training_tensors(graph: tf.Graph):
    """
    Return a list of tensors in the graph used to set training mode
    :param graph: Graph to search for training tensors in
    :return: List of tensors in the graph used to set training mode
    """
    training_tensors = set()
    for op in graph.get_operations():
        # Currently the only training tensors we know of are attached to FusedBatchNorm blocks
        if op.type == 'FusedBatchNormV3' and op.get_attr('is_training'):
            try:
                switch_op = op.inputs[0].op
                assert switch_op.type == 'Switch'
                pred_id_op = switch_op.inputs[1].op
                assert pred_id_op.type == 'Identity'
                training_tensor = pred_id_op.inputs[0]
                training_tensors.add(training_tensor)
            # pylint: disable=bare-except
            except:
                continue
    return training_tensors


def update_variables_with_values(sess: tf.Session, vars_with_values: Dict)->None:
    """
    update a given variable with the value provided
    :param sess: current tf Session
    :param vars_with_values: Dictionary of variable names and their values
    :return: None, assert if variable not found.
    """

    with sess.graph.as_default():
        for var_name in vars_with_values.keys():
            vars_with_given_name = [var for var in tf.global_variables()
                                    if var.op.name == var_name]

            # could not find variable
            if not vars_with_given_name:
                logger.error("Could not find any variable with name: %s", var_name)
                assert False
            else:
                var_to_be_updated = vars_with_given_name[0]
                var_to_be_updated.load(vars_with_values[var_name], sess)


def save_data_to_pickle_file(info_to_be_saved, output_path: str, output_file_name: str):
    """
    utility to save given data structure to a pickle file.
    :param info_to_be_saved: data to be saved as pickle file
    :param output_path: output path to save pickle file
    :param output_file_name: output pickle file name
    :return: None
    """

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_file_with_path = os.path.join(output_path, output_file_name)
    # save the dictionaries of quant op nodes added as pickle files
    outfile = open(output_file_with_path, 'wb')
    pickle.dump(info_to_be_saved, outfile)
    outfile.close()


def load_data_from_pickle_file(filename: str):
    """
    utility to load data from pickle file
    :param filename: name of the pickle file with path
    :return: data loaded from pickle file
    """

    infile = open(filename, 'rb')
    loaded_data = pickle.load(infile)
    infile.close()
    return loaded_data


def create_rand_tensors_given_shapes(input_shape: Union[Tuple, List[Tuple]]) -> List[np.ndarray]:
    """
    Given shapes of some tensors, create one or more random numpy tensors and return them as a list of tensors
    :param input_shape: Shapes of tensors to create
    :return: Created list of numpy tensors
    """
    if isinstance(input_shape, List):
        input_shapes = input_shape
    else:
        input_shapes = [input_shape]

    rand_tensors = []
    for shape in input_shapes:
        rand_tensors.append(np.random.rand(*shape))

    return rand_tensors
