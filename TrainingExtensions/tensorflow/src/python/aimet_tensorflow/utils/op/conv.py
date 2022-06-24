# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" utilities for conv op """

from typing import Tuple, List, Union
import numpy as np
import tensorflow as tf

from aimet_common.utils import AimetLogger
from aimet_tensorflow.utils.common import get_padding, create_input_feed_dict, create_rand_tensors_given_shapes, \
    get_valid_ops
from aimet_tensorflow import graph_editor
from aimet_tensorflow.utils.graph_saver import save_and_load_graph
from aimet_tensorflow.utils import constants

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


class WeightTensorUtils:
    """ class with generic apis related to TF weight tensor of conv op """

    @staticmethod
    def get_tensor_index_in_given_op(input_op: tf.Operation) -> int:
        """
        Returns the index to weight tensor in the op specified
        :param input_op: tf operation type
        :return: index of weight tensor in the inputs of the given op
        """
        if input_op.type not in constants.OP_WEIGHT_INDICES:
            raise ValueError('Op type: '+input_op.type+' does not contain weights!')
        return constants.OP_WEIGHT_INDICES[input_op.type]

    @staticmethod
    def get_tensor_shape(input_op: tf.Operation) -> List[int]:
        """
        Returns the shape of weight tensor in the op specified
        :param input_op: tf operation type
        :return: shape as list
        """
        weight_tensor_index = WeightTensorUtils.get_tensor_index_in_given_op(input_op)
        return input_op.inputs[weight_tensor_index].shape

    @staticmethod
    def get_read_op(input_op: tf.Operation) -> tf.Operation:
        """
        Returns the read op associated with the weight tensor in given op
        :param input_op: operation for which the read op on weight tensor is to be obtained
        :return: read op associated with weight tensor
        """
        wt_tensor_index = WeightTensorUtils.get_tensor_index_in_given_op(input_op)
        return input_op.inputs[wt_tensor_index].op

    @staticmethod
    def get_wt_tensor(op: tf.Operation) -> tf.Tensor:
        """
        get weight tensor in given op
        This is used by api used for updating weight tensor
        :param op: tf operation to extract weight tensor from
        :return : weight tensor sa tf.Tensor type
        """

        wt_tensor_index = WeightTensorUtils.get_tensor_index_in_given_op(op)
        wt_var_read_op = op.inputs[wt_tensor_index].op

        wt_tensor = wt_var_read_op.inputs[constants.OP_VAR_WEIGHT_INDEX]

        return wt_tensor


    @staticmethod
    def get_wt_as_read_var_tensor(op: tf.Operation) -> tf.Tensor:
        """
        get weight kernel in the op as a readVariableOp tensor
        we need to evaluate this to get weights (not get_wt_tensor)
        :param op: tf operation to extract weight tensor from
        :return : weight tensor as  ReadVariableOp tensor
        """

        wt_tensor_index = WeightTensorUtils.get_tensor_index_in_given_op(op)
        get_wt_as_read_var_tensor = op.inputs[wt_tensor_index]

        return get_wt_as_read_var_tensor

    @staticmethod
    def get_tensor_as_numpy_data(sess: tf.compat.v1.Session, op: tf.Operation) -> np.array:
        """
        return weight kernel in the op as numpy data
        :param sess: TensorFlow session
        :param op: tf operation to extract weight tensor from.
        :return : weight tensor as numpy array type, if found in the given op
        """

        wt_tensor = WeightTensorUtils.get_wt_as_read_var_tensor(op)
        numpy_data = sess.run(wt_tensor)
        return numpy_data

    @staticmethod
    def update_tensor_for_sim_op(sess: tf.compat.v1.Session, op: tf.Operation, tensor_as_numpy_array):
        """
        updated existing weight tensor variable in given op with new value.
        :param sess: active tf.compat.v1.Session
        :param op: op for which the weight tensor is to be updated
        :param tensor_as_numpy_array: new weight tensor as numpy array
        :return: None
        """
        # validate the shapes are same
        assert WeightTensorUtils.get_tensor_shape(op) == tensor_as_numpy_array.shape
        # update the weight tensor
        with sess.graph.as_default():
            wt_tensor_index = WeightTensorUtils.get_tensor_index_in_given_op(op)
            wt_var_read_op = op.inputs[wt_tensor_index].op.inputs[0].op
            wt_tensor = wt_var_read_op.inputs[constants.OP_VAR_WEIGHT_INDEX]
            assert wt_tensor is not None, ('Error, no weight tensor found for this op', op.name)
            wt_as_var = [var for var in tf.compat.v1.global_variables() if var.name == wt_tensor.name][0]
            wt_as_var.load(tensor_as_numpy_array, sess)


    @staticmethod
    def update_tensor_for_op(sess: tf.compat.v1.Session, op: tf.Operation, tensor_as_numpy_array):
        """
        updated existing weight tensor variable in given op with new value.
        :param sess: active tf.compat.v1.Session
        :param op: op for which the weight tensor is to be updated
        :param tensor_as_numpy_array: new weight tensor as numpy array
        :return: None
        """

        # validate the shapes are same
        assert WeightTensorUtils.get_tensor_shape(op) == tensor_as_numpy_array.shape

        # update the weight tensor
        with sess.graph.as_default():
            wt_tensor = WeightTensorUtils.get_wt_tensor(op)
            assert wt_tensor is not None, ('Error, no weight tensor found for this op', op.name)
            # use tensor name to lookup var type associated with it
            wt_as_var = [var for var in tf.compat.v1.global_variables() if var.name == wt_tensor.name][0]
            wt_as_var.load(tensor_as_numpy_array, sess)


class BiasUtils:
    """ util for operating on TF bias tensor"""

    @staticmethod
    def _get_bias_shape_from_weights(conv_op: tf.Operation) -> int:
        """
        helper function to get bias shape from weight shape of a given op.
        :param conv_op: conv op as tf.Operation
        :return: bias shape for given op
        """

        assert conv_op.type in ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']
        w_shape = WeightTensorUtils.get_tensor_shape(conv_op)
        b_index = constants.OP_WEIGHT_SHAPE_INDEX_FOR_BIAS[conv_op.type]
        return w_shape[b_index]

    @ staticmethod
    def insert_bias_add_op(sess: tf.compat.v1.Session, conv_op_out_tensor: tf.Tensor,
                           new_bias_tensor: tf.Variable, bias_name="bias_value") -> None:
        """
        Insert bias-add op to given conv op.
        :param sess: model as tf.compat.v1.Session
        :param conv_op_out_tensor: output of conv op that should feed into the new bias op as tf.Tensor
        :param new_bias_tensor:  bias tensor to be added as tf.Variable
        :param bias_name: name string for the bias op
        :return: None ,
        Note : Higher level api needs to perform a save and load to get updated session after usage of this api
        """

        assert conv_op_out_tensor is not None, 'Error, insert_bias_add_op() : conv op output tensor must be provided'
        with sess.graph.as_default():
            if conv_op_out_tensor.consumers():

                consumer_list = []
                for consumer in conv_op_out_tensor.consumers():
                    consumer_list.append(consumer)

                # create new Bias add op
                bias_add_op = tf.nn.bias_add(value=conv_op_out_tensor, bias=new_bias_tensor, name=bias_name)

                # use reroute to insert bias-add and swap current outputs of conv with bias-add op
                graph_editor.reroute_ts(bias_add_op, conv_op_out_tensor, can_modify=consumer_list)

                # initialize tensor once it's added
                sess.run(tf.compat.v1.variables_initializer([new_bias_tensor]))

    @staticmethod
    def initialize_model_with_bias(sess: tf.compat.v1.Session, input_op_names: List[str], output_op_names: List[str]) \
            -> tf.compat.v1.Session:
        """
        Initializes given model with bias.
        Adds zero bias to conv/linear layers without bias param, in given model.
        :param sess: model to be updated as tf.compat.v1.Session
        :return: updated session as tf.compat.v1.Session
        """

        assert sess is not None
        with sess.graph.as_default():
            ops = get_valid_ops(sess.graph, input_op_names, output_op_names)

            for op in ops:
                # skip gradient ops
                if not op.name.startswith('gradients/') and \
                        op.type in ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']:
                    # add bias if not present
                    if BiasUtils.is_bias_none(op):
                        # add bias param
                        bias_shape = BiasUtils._get_bias_shape_from_weights(op)
                        zero_bias = tf.Variable(initial_value=np.zeros(bias_shape), dtype=tf.float32)
                        BiasUtils._create_bias_add_op_and_insert(sess, op, zero_bias)

        new_sess = save_and_load_graph('./temp', sess)
        sess.close()

        return new_sess

    @staticmethod
    def _create_bias_add_op_and_insert(sess: tf.compat.v1.Session, conv_op: tf.Operation, new_bias_var: tf.Variable,
                                       bias_name="bias_value") -> None:
        """
        creates and adds a bias_add op to conv op
        :param sess: active tf.compat.v1.Session
        :param conv_op: Convolution op
        :param new_bias_var: bias variable
        :param bias_name: an optional string for bias name
        :return: None
        """

        assert conv_op.type in ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']

        with sess.graph.as_default():
            if conv_op.outputs:
                bias_index_in_op = BiasUtils.get_bias_index_in_given_op(conv_op)
                conv_op_out_tensor = conv_op.outputs[bias_index_in_op]
                sess.run(tf.compat.v1.variables_initializer([new_bias_var]))
                BiasUtils.insert_bias_add_op(sess, conv_op_out_tensor, new_bias_var,
                                             bias_name)

    @staticmethod
    def get_bias_index_in_given_op(input_op: tf.Operation) -> int:
        """
        Returns the index to bias tensor in the op specified
        :param input_op: tf operation type
        :return: index of bias tensor in the inputs of the given op
        """
        if input_op.type not in constants.OP_BIAS_INDICES:
            raise ValueError('Op type: ' + input_op.type + ' does not contain bias!')
        return constants.OP_BIAS_INDICES[input_op.type]

    @staticmethod
    def is_bias_none(input_op: tf.Operation):
        """
        checks for the presence of bias in a given op
        :param input_op: tf operation type
        :return: True if given op has NO bias, false otherwise
        """
        is_bias_none = True
        if not input_op.outputs:
            is_bias_none = False
        else:
            # Bias is consumers of output_0 of the conv op, look for it
            # output 0 is the bias tensor
            bias_index = BiasUtils.get_bias_index_in_given_op(input_op)
            for consumer in input_op.outputs[bias_index].consumers():
                # Ignore Reshape as it can be placed between MatMul and BiasAdd on Dense layer of Transformer
                if consumer.type in ['Reshape'] and len(consumer.outputs[0].consumers()) == 1:
                    consumer = consumer.outputs[0].consumers()[0]
                # check the input types of the add or bias_add
                if consumer.type == 'BiasAdd':
                    # check num tensors and op types coming into this bias add or add
                    assert len(consumer.inputs) == 2
                    # check if one of the inputs is ReadVariableOp type or Identity type
                    # when we add BiasAdd op to a conv op programmatically, the read op is 'Identity' type.
                    if consumer.inputs[constants.BIAS_ADD_CONSUMERS_INPUT_BIAS_READ_INDEX].op.type in \
                            ['ReadVariableOp', 'Identity', 'QcQuantize', 'QcQuantizePerChannel']:
                        is_bias_none = False
                    break
        return is_bias_none

    @staticmethod
    def get_shape(input_op: tf.Operation) -> tf.TensorShape:
        """
        Returns the shape of the bias in the op specified
        :param input_op: tf operation type (conv)
        :return: shape of bias as a List type
        """
        # infer bias shape from weight kernel shape
        return [BiasUtils._get_bias_shape_from_weights(input_op)]

    @staticmethod
    def get_bias_read_op(input_op: tf.Operation) -> tf.Operation:
        """
        Returns the read op associated with the bias in given op
        :param input_op: operation for which the read op on bias is to be obtained
        :return: read op associated with bias
        """

        bias_read_op = None
        bias_index = BiasUtils.get_bias_index_in_given_op(input_op)
        if not BiasUtils.is_bias_none(input_op):
            for consumer in input_op.outputs[bias_index].consumers():
                # Ignore Reshape as it can be placed between MatMul and BiasAdd on Dense layer of Transformer
                if consumer.type in ['Reshape'] and len(consumer.outputs[0].consumers()) == 1:
                    consumer = consumer.outputs[0].consumers()[0]
                if consumer.type in ['Add', 'BiasAdd']:
                    assert len(consumer.inputs) == 2
                    # pick the bias ReadVariableOp type
                    bias_read_op = consumer.inputs[constants.BIAS_ADD_CONSUMERS_INPUT_BIAS_READ_INDEX].op
        return bias_read_op

    @staticmethod
    def get_bias_tensor(op: tf.Operation):
        """
        return bias in the op as a tf tensor type
        bias_add op has two inputs : conv's output and a bias_read op
        :param op: tf operation to extract bias from.
        :return : bias as tf tensor type, if found in the given op
        """

        # bias tensor feeds into biasadd op through ReadVariableOp type
        # bias add inputs[1] is the bias tensor we want.
        bias_tensor = None
        bias_add = BiasUtils.get_bias_add_read_var_op_tensor(op)

        if bias_add is not None:
            assert len(bias_add.inputs) == 2
            # input to a bias add op is bias
            bias_tensor = bias_add.inputs[constants.BIAS_ADD_READ_VAR_OP_BIAS_TENSOR_INDEX]

        return bias_tensor

    @staticmethod
    def get_bias_add_read_var_op_tensor(input_op: tf.Operation) -> tf.Operation:
        """
        Returns the readVariableOp tensor associated with bias add of given op
        :param input_op: operation for which the bias add op is to be obtained
        :return: bias_tensor associated with bias in the given conv op
        """

        bias_add_op = None
        bias_index = BiasUtils.get_bias_index_in_given_op(input_op)
        if not BiasUtils.is_bias_none(input_op):
            for consumer in input_op.outputs[bias_index].consumers():
                # Ignore Reshape as it can be placed between MatMul and BiasAdd on Dense layer of Transformer
                if consumer.type in ['Reshape'] and len(consumer.outputs[0].consumers()) == 1:
                    consumer = consumer.outputs[0].consumers()[0]
                if consumer.type in ['Add', 'BiasAdd']:
                    bias_add_op = consumer
        return bias_add_op

    @staticmethod
    def get_bias_as_numpy_data(sess: tf.compat.v1.Session, op: tf.Operation) -> tf.Variable:
        """
        return bias in the op as a tf variable type
        :param sess: TensorFlow session
        :param op: tf operation to extract weight tensor from.
        :return : weight tensor as tf variable type, if found in the given op
        """

        # bias tensor feeds into bias-add op through ReadVariableOp type
        # bias add inputs[1] is the bias tensor we want to read
        bias_tensor = BiasUtils.get_bias_tensor(op)
        assert bias_tensor is not None
        numpy_data = sess.run(bias_tensor)
        return numpy_data

    @staticmethod
    def update_bias_for_quantized_op(sess: tf.compat.v1.Session, op: tf.Operation, bias_as_numpy_array,
                                     is_bias_none: bool = False):
        """
        update existing bias in given op with new bias value
        creates and adds new bias if it does not exist.
        Note :
        Caller needs to perform a load and save of the graph
        if this api is invoked for an op without existing bias.
        :param sess: TensorFlow session
        :param op:op for which the bias is to be updated
        :param bias_as_numpy_array: new bias as a numpy array
        :param is_bias_none: True if Bias is None
        :return: None
        """

        with sess.graph.as_default():
            if not is_bias_none:
                bias_tensor_as_read_var_op_input = BiasUtils.get_bias_tensor(op)
                assert len(bias_tensor_as_read_var_op_input.op.inputs) == 8
                bias_add = bias_tensor_as_read_var_op_input.op.inputs[constants.OP_BIAS_INDICES[op.type]]
                bias_tensor = bias_add.op.inputs[constants.OP_BIAS_INDICES[op.type]]
                assert BiasUtils.get_shape(op)[0] == bias_as_numpy_array.size
                # use tensor name to lookup var type associated with it
                assert bias_tensor is not None, ('Error, bias tensor lookup failed for op ', op.name)
                bias_as_var = [var for var in tf.compat.v1.global_variables() if var.name == bias_tensor.name][0]
                bias_as_var.load(bias_as_numpy_array, sess)

    @staticmethod
    def update_bias_for_sim_op(sess: tf.compat.v1.Session, op: tf.Operation, bias_as_numpy_array,
                               bias_name="bias_value"):
        """
        update existing bias in given op with new bias value
        creates and adds new bias if it does not exist.
        Note :
        Caller needs to perform a load and save of the graph
        if this api is invoked for an op without existing bias.
        :param sess: TensorFlow session
        :param op:op for which the bias is to be updated
        :param bias_as_numpy_array: new bias as a numpy array
        :param bias_name: optional name can be specified by user
        :return: None
        """
        with sess.graph.as_default():
            if not BiasUtils.is_bias_none(op):
                bias_quant_op = BiasUtils.get_bias_tensor(op)
                bias_tensor_as_read_var_quant_op_input = bias_quant_op.op.inputs[0]
                # assert len(bias_tensor_as_read_var_op_input.op.inputs) == 1
                bias_tensor = bias_tensor_as_read_var_quant_op_input.op.inputs[constants.OP_BIAS_INDICES[op.type]]
                assert BiasUtils.get_shape(op)[0] == bias_as_numpy_array.size
                # use tensor name to lookup var type associated with it
                assert bias_tensor is not None, ('Error, bias tensor lookup failed for op ', op.name)
                bias_as_var = [var for var in tf.compat.v1.global_variables() if var.name == bias_tensor.name][0]
                bias_as_var.load(bias_as_numpy_array, sess)
            else:
                # _create_bias_add_op_and_insert
                new_bias_var = tf.Variable(initial_value=bias_as_numpy_array, name=bias_name, dtype=tf.float32)
                BiasUtils._create_bias_add_op_and_insert(sess, op, new_bias_var, bias_name)

    @staticmethod
    def update_bias_for_op(sess: tf.compat.v1.Session, op: tf.Operation, bias_as_numpy_array,
                           bias_name="bias_value"):
        """
        update existing bias in given op with new bias value
        creates and adds new bias if it does not exist.
        Note :
        Caller needs to perform a load and save of the graph
        if this api is invoked for an op without existing bias.
        :param sess: TensorFlow session
        :param op:op for which the bias is to be updated
        :param bias_as_numpy_array: new bias as a numpy array
        :param bias_name: optional name can be specified by user
        :return: None
        """

        with sess.graph.as_default():
            if not BiasUtils.is_bias_none(op):
                bias_tensor_as_read_var_op_input = BiasUtils.get_bias_tensor(op)
                assert len(bias_tensor_as_read_var_op_input.op.inputs) == 1
                bias_tensor = bias_tensor_as_read_var_op_input.op.inputs[constants.OP_BIAS_INDICES[op.type]]
                assert BiasUtils.get_shape(op)[0] == bias_as_numpy_array.size
                # use tensor name to lookup var type associated with it
                assert bias_tensor is not None, ('Error, bias tensor lookup failed for op ', op.name)
                bias_as_var = [var for var in tf.compat.v1.global_variables() if var.name == bias_tensor.name][0]
                bias_as_var.load(bias_as_numpy_array, sess)
            else:
                # _create_bias_add_op_and_insert
                new_bias_var = tf.Variable(initial_value=bias_as_numpy_array, name=bias_name, dtype=tf.float32)
                BiasUtils._create_bias_add_op_and_insert(sess, op, new_bias_var, bias_name)


def get_conv2d_op_params(op: tf.Operation) -> (Tuple, Tuple, Tuple):
    """
    Get Conv2d op's parameters
    :param op: TensorFlow Op
    :return: (strides, padding, groups)
    """

    strides = op.get_attr('strides')
    data_format = op.get_attr('data_format')
    padding = op.get_attr('padding')

    if str(data_format.decode("utf-8")) == "NHWC":
        strides = (strides[1], strides[2])

    elif str(data_format.decode("utf-8")) == "NCHW":
        strides = (strides[2], strides[3])

    else:
        raise ValueError("unknown data format")

    # For Conv2D op groups should be 1
    groups = 1

    return strides, padding, groups


def get_strides_for_split_conv_ops(op: tf.Operation) -> (List, List):
    """
    :param op: TensorFlow Op
    :return: (conv_a_strides, conv_b_strides)
    """

    if not op.type == 'Conv2D':
        raise ValueError("Only Conv2d op can be split")

    strides = op.get_attr("strides")
    data_format = op.get_attr("data_format")

    if str(data_format.decode("utf-8")) == "NHWC":
        conv_a_strides = [strides[1], 1]
        conv_b_strides = [1, strides[2]]

    elif str(data_format.decode("utf-8")) == "NCHW":
        conv_a_strides = [strides[2], 1]
        conv_b_strides = [1, strides[3]]

    else:
        raise ValueError("Unknown data format!")

    return conv_a_strides, conv_b_strides


def get_weight_shape(op: tf.Operation) -> List:
    """
    Weight shape of an Op in Common format
    Common format
    Conv2D - [Noc, Nic, k_h, k_w]
    MatMul - [Noc, Nic]

    :param op: TensorFlow Op
    :return: shape
    """
    weight_index = WeightTensorUtils.get_tensor_index_in_given_op(input_op=op)
    weight_shape = op.inputs[weight_index].get_shape().as_list()

    # Conv2D weights are stored in the order [kh, kw, Nic, Noc] in TensorFlow
    # Re-order them to the form [Noc, Nic, kh, kw]
    if op.type == 'Conv2D':
        weight_shape = [weight_shape[3], weight_shape[2], weight_shape[0], weight_shape[1]]

    # FC weights are stored in order [Nic, Noc] in TensorFlow
    # Re-order them to the form  [Noc, Nic]
    elif op.type == 'MatMul':
        weight_shape = [weight_shape[1], weight_shape[0]]

    else:
        raise ValueError('op type not supported!')

    return weight_shape


def get_output_activation_shape(sess: tf.compat.v1.Session, op: tf.Operation, input_op_names: List[str],
                                input_shape: Union[Tuple, List[Tuple]]) -> List:
    """
     Output activation shape in the Common format [NCHW]
    :param sess: TensorFlow Session
    :param op: TensorFlow op
    :param input_op_names: list of input op names of model
    :param input_shape: tuple or list of tuple of input shape of model
    :return: output_shape in Common format [NCHW]
    """
    if op.type == 'MatMul':
        output_shape = get_matmul_activation_shape(op=op, input_activation=False)

    elif op.type == 'Conv2D':
        output_shape = get_conv2d_activation_shape(sess, op, input_op_names, input_shape, input_activation=False)

    else:
        raise ValueError("Op type is not supported!")

    return output_shape


def get_conv2d_activation_shape(sess: tf.compat.v1.Session, op: tf.Operation, input_op_names: List[str],
                                input_shape: Union[Tuple, List[Tuple]], input_activation: bool) -> List:
    """
    :param sess: TensorFlow Session
    :param op: TensorFlow op
    :param input_op_names: list of input op names of model
    :param input_shape: tuple or list of tuple of input shape of model
    :param input_activation: whether input / output activation shape
    :return: List of input / output activation shape in Common format [NCHW]
    """
    # use static shape for input / output activations
    if input_activation:
        activation_shape = op.inputs[0].get_shape().as_list()

    else:
        activation_shape = op.outputs[0].get_shape().as_list()

    data_format = op.get_attr('data_format')

    # convert input / output activation shape to Common format [NCHW], if channels_last
    if str(data_format.decode("utf-8")) == "NHWC":
        activation_shape = [activation_shape[0], activation_shape[3], activation_shape[1], activation_shape[2]]

    # if the static shape is undefined, then find dynamic shape of input / output activations
    if activation_shape[2] is None:

        # get input data
        input_data = create_rand_tensors_given_shapes(input_shape=input_shape)

        # create feed_dict
        feed_dict = create_input_feed_dict(graph=op.graph,
                                           input_op_names_list=input_op_names,
                                           input_data=input_data)

        if input_activation:
            # get the input activation shape by evaluating the input tensor
            input_tensor = op.inputs[0]
            activation_shape = input_tensor.eval(feed_dict=feed_dict, session=sess).shape
        else:
            # get the output activation shape by evaluating the output tensor
            output_tensor = op.outputs[0]
            activation_shape = output_tensor.eval(feed_dict=feed_dict, session=sess).shape

        # convert output activation shape to Common format [NCHW], if channels_last
        if str(data_format.decode("utf-8")) == "NHWC":
            activation_shape = [activation_shape[0], activation_shape[3], activation_shape[1], activation_shape[2]]

    return activation_shape


def get_matmul_activation_shape(op: tf.Operation, input_activation: bool) -> List:
    """
    :param op: TensorFlow Operation
    :param input_activation: whether input / output activation shape
    :return: List activation shape [N, out_channels, 1, 1]
    """
    assert op.type == 'MatMul'

    # use static shape for output/input activations of matmul
    if input_activation:
        activation_shape = op.inputs[0].get_shape().as_list()
        activation_shape.extend([1, 1])
        return activation_shape

    activation_shape = op.outputs[0].get_shape().as_list()
    activation_shape.extend([1, 1])
    return activation_shape


def get_layer_attributes(sess: tf.compat.v1.Session, op: tf.Operation, input_op_names: List[str],
                         input_shape: Union[Tuple, List[Tuple]]) -> (Tuple, Tuple, Tuple):
    """
    Get attributes (kernel_size, stride, padding) of tf.nn.Conv2d Op
    :param sess: TensorFLow Session
    :param op: TensorFLow Operation
    :param input_op_names: List of input op names of model
    :param input_shape: tuple or list of tuple of input shape of model
    :return: (kernel_size, stride, padding)
    """
    # pylint: disable=too-many-locals
    assert op.type == 'Conv2D'

    stride = op.get_attr('strides')
    data_format = op.get_attr('data_format')

    output_activation_shape = get_conv2d_activation_shape(sess=sess, op=op, input_op_names=input_op_names,
                                                          input_shape=input_shape, input_activation=False)

    input_activation_shape = get_conv2d_activation_shape(sess=sess, op=op, input_op_names=input_op_names,
                                                         input_shape=input_shape, input_activation=True)

    _, _, activation_h, activation_w = output_activation_shape
    output_shape = (activation_h, activation_w)

    _, _, activation_h, activation_w = input_activation_shape
    input_shape = (activation_h, activation_w)

    # 'channels_last' format
    if str(data_format.decode("utf-8")) == "NHWC":

        stride = (int(stride[1]), int(stride[2]))

    # 'channels_first' format
    elif str(data_format.decode("utf-8")) == "NCHW":

        stride = (int(stride[2]), int(stride[3]))

    else:
        raise ValueError("Unknown data format!")

    # Conv2d weight shape in TensorFlow  [kh, kw, Nic, Noc]
    weight_index = WeightTensorUtils.get_tensor_index_in_given_op(input_op=op)
    weight_shape = op.inputs[weight_index].shape
    kernel_size = (int(weight_shape[0]), int(weight_shape[1]))

    # get the padding for (height, width) dimension
    padding = get_padding(input_shape=input_shape, output_shape=output_shape, kernel_size=kernel_size, stride=stride)

    return kernel_size, stride, padding


def get_weight_tensor_with_shape(model: tf.compat.v1.Session, input_op: tf.Operation):
    """
     generic function to extract weight tensor of a given conv/linear op
    :param model: tf.compat.v1.Session type
    :param input_op: input op as tf.Operation type
    :return: weight and shape of tensor extracted from given op
    """

    with model.graph.as_default():

        weight_tensor = WeightTensorUtils.get_tensor_as_numpy_data(model, input_op)

        # Conv2d weight shape in TensorFlow  [kh, kw, Nic, Noc]
        # re order in the common shape  [Noc, Nic, kh, kw]
        shape = WeightTensorUtils.get_tensor_shape(input_op)
        wt_tensor = None

        if input_op.type == 'DepthwiseConv2dNative':
            # Depthwise conv layers in TF have outputs(Noc) set to 1.
            # we will use format [Nic, Noc, kh, kw] -
            # to be compatible with cpp backend.
            wt_tensor = np.transpose(weight_tensor, (2, 3, 0, 1))
            # [Nic, Noc, kh, kw]
            shape = np.array([shape[2], shape[3], shape[0], shape[1]])
        elif input_op.type == 'MatMul':
            shape = np.concatenate((np.array([1, 1]), shape))
            wt_tensor = np.transpose(weight_tensor, (1, 0))
            # [Noc, Nic, kh, kw]
            shape = np.array([shape[3], shape[2], shape[0], shape[1]])
        elif input_op.type == 'Conv2D':
            wt_tensor = np.transpose(weight_tensor, (3, 2, 0, 1))
            # [Noc, Nic, kh, kw]
            shape = np.array([shape[3], shape[2], shape[0], shape[1]])
        else:
            logger.error("_get_weight_tensor_transpose_reshape(): Operation type unsupported")

    return wt_tensor, shape
