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

""" TF Code to fold batch-norm layers """

from typing import List, Tuple, Union
import os
import numpy as np
import tensorflow as tf

import aimet_tensorflow
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.common.operation import OpWithMetaInfoType, Op
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils import graph_saver
from aimet_tensorflow.utils.op.conv import WeightTensorUtils, BiasUtils
from aimet_tensorflow.utils.op.fusedbatchnorm import BNUtils
from aimet_tensorflow.utils.graph_saver import save_and_load_graph
from aimet_tensorflow.utils.op.conv import get_weight_tensor_with_shape
from aimet_tensorflow.utils.common import get_ordered_conv_linears
from aimet_common.graph_searcher import GraphSearcher
from aimet_common.bias_correction import ConvBnPatternHandler
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.utils import AimetLogger
import aimet_common.libpymo as libpymo


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.CrosslayerEqualization)

# save required information for performing bn fold on candidate bns as
# <PairTypes> that includes :
# tf.Operation type  : op which bn needs to be folded into.
# OpWithMetaInfoType : bn op will store the input and output tensors along with tf.Operation
# bool : Flag indicating if bn op can be folded upstream or downstream.
PairType = Tuple[tf.Operation, Union[OpWithMetaInfoType, Op], bool]

def _conv_bn_select_custom_pattern_init():
    """
    initialize the patterns we want to use to pick layers for bn based bias correction
    :return: patterns and associated actions to be performed upon match
    """

    patterns_with_callbacks = []

    # the types we want to handle
    conv_layer_types = ['Conv2D', 'DepthwiseConv2dNative']
    preceeding_linear_op_types = ['Flatten', 'Reshape']

    # handler when pattern match
    layer_select_handler = ConvBnPatternHandler()

    # Linear layer combinations
    for preceeding_linear_op_type in preceeding_linear_op_types:
        # BN -> Linear
        patterns_with_callbacks.append(PatternType(pattern=['FusedBatchNormV3', preceeding_linear_op_type, 'Dense'],
                                                   action=layer_select_handler))

        patterns_with_callbacks.append(PatternType(pattern=['FusedBatchNorm', preceeding_linear_op_type, 'Dense'],
                                                   action=layer_select_handler))
        # note: we cannot perform linear -> BN on TF

    # conv layer combinations
    for conv in conv_layer_types:

        # BN -> Conv / Conv -> BN
        patterns_with_callbacks.append(PatternType(pattern=[conv, 'FusedBatchNormV3'],
                                                   action=layer_select_handler))

        patterns_with_callbacks.append(PatternType(pattern=['FusedBatchNormV3', conv],
                                                   action=layer_select_handler))

        patterns_with_callbacks.append(PatternType(pattern=[conv, 'FusedBatchNorm'],
                                                   action=layer_select_handler))

        patterns_with_callbacks.append(PatternType(pattern=['FusedBatchNorm', conv],
                                                   action=layer_select_handler))

    return patterns_with_callbacks, layer_select_handler


def _find_conv_bn_pairs(model, start_op_names: Union[List[str], str],
                        output_op_names: Union[List[str], str]):
    """
    uses searcher to choose convs/ linears with bn and activation info.
    :param model: tf.compat.v1.Session type
    :param start_op_names: list of strings with names of starting ops in the model
    :param output_op_names: List of output op names of the model, used to help ConnectedGraph determine valid ops
    (to ignore training ops for example).
    :return: dictionary of conv/linear layers with associated bn op / activation info
    """

    if isinstance(start_op_names, str):
        start_op_names = [start_op_names]

    if isinstance(output_op_names, str):
        output_op_names = [output_op_names]

    conn_graph = ConnectedGraph(model.graph, start_op_names, output_op_names)

    # create a list of patterns and corresponding handlers or actions to be applied for selecting
    # layers for bias correction.
    # layer_select_handler is an instance of custom handler created for bias correction.
    patterns_with_callback, layer_select_handler = _conv_bn_select_custom_pattern_init()

    # graph searcher looks for patterns and applies actions when matching patterns are found
    graph_searcher = GraphSearcher(conn_graph, patterns_with_callback)
    graph_searcher.find_all_patterns_in_graph_apply_actions()

    # use custom handler instance and fetch the selected layer info for bias correction
    convs_linears_bn_activation_info_dict = layer_select_handler.get_conv_linear_bn_info_dict()

    return convs_linears_bn_activation_info_dict


def find_all_batch_norms_to_fold(sess: tf.compat.v1.Session, start_op_names: Union[List[str], str],
                                 output_op_names: Union[List[str], str], return_bn_conn_op=False) -> List[PairType]:
    """
    uses searcher to choose layers for bias correction
    :param sess: tf.compat.v1.Session type
    :param start_op_names: list of strings with names of starting ops in the model
    :param output_op_names: List of output op names of the model, used to help ConnectedGraph determine valid ops
    :param return_bn_conn_op: Return bn op as connected graph op instead of tf tensor
    (to ignore training ops for example).  If None, all ops in the model are considered valid.

    :return: List of conv/linear layers with associated bn op / activation info
    """

    convs_linears_bn_activation_info_dict = _find_conv_bn_pairs(sess, start_op_names, output_op_names)

    # get all ordered convs/ linears and skip gradient ops
    ordered_conv_linears = get_ordered_conv_linears(sess, start_op_names, output_op_names)

    # get the in out tensor for bns found, we need this on TF to remove the bns after fold.
    bn_conv_linear_pairs = []

    # track BNs added for fold
    marked_bn_set = set()

    for conv_linear_op in ordered_conv_linears:
        if conv_linear_op in convs_linears_bn_activation_info_dict.keys():
            bn_info = convs_linears_bn_activation_info_dict[conv_linear_op]
            if bn_info.output_bn:
                if bn_info.output_bn not in marked_bn_set:
                    if return_bn_conn_op:
                        bn_conv_linear_pairs.append((conv_linear_op, bn_info.output_bn, True))
                    else:
                        bn_conv_linear_pairs.append((conv_linear_op, bn_info.output_bn.get_tf_op_with_io_tensor(), True))
                    marked_bn_set.add(bn_info.output_bn)
            elif bn_info.input_bn:
                if bn_info.input_bn not in marked_bn_set:
                    if return_bn_conn_op:
                        bn_conv_linear_pairs.append((conv_linear_op, bn_info.output_bn, True))
                    else:
                        bn_conv_linear_pairs.append((conv_linear_op, bn_info.input_bn.get_tf_op_with_io_tensor(), False))
                    marked_bn_set.add(bn_info.input_bn)

    return bn_conv_linear_pairs


def _get_bias_tensor(sess: tf.compat.v1.Session, conv: tf.Operation) -> libpymo.TensorParams():
    """
    Get bias tensor in given conv op.
    Packs bias in the format required for BN fold
    (libpymo.TensorParams()).
    :param sess: current session
    :param conv: conv op
    :return: return bias param in libpymo.TensorParams() format.
    """

    # Bias tensor
    bias_tensor = libpymo.TensorParams()
    with sess.graph.as_default():
        if not BiasUtils.is_bias_none(conv):
            bias_tensor.shape = BiasUtils.get_shape(conv)
            bias_tensor.data = BiasUtils.get_bias_as_numpy_data(sess, conv)

    return bias_tensor


def _get_weight_tensor_transpose_reshape(sess: tf.compat.v1.Session, conv: tf.Operation) -> libpymo.TensorParams():
    """
    Get weight tensor from conv op
    Converts to right format - performs transpose and reshape.
    Packs it to the format required for BN fold (libpymo.TensorParams()).
    :param sess: current session
    :param conv: conv op
    :return: return weight tensor in libpymo.TensorParams() format.
    """

    # Weight tensor libpymo format
    weight_tensor = libpymo.TensorParams()

    wt_tensor, shape = get_weight_tensor_with_shape(sess, conv)

    # linear array to be sent for bn fold
    weight_tensor.data = wt_tensor.reshape(-1)
    weight_tensor.shape = shape

    return weight_tensor


def _get_bn_params(sess: tf.compat.v1.Session, bn: tf.Operation) -> libpymo.BNParams():
    """
    helper to populate BN params from given BN op, required for fold
    :param sess: tf.compat.v1.Session type
    :param bn: BatchNorm or a FusedBatch Norm op
    :return: bn_params
    """
    with sess.graph.as_default():
        # create BNParams type and populate
        bn_params = libpymo.BNParams()
        bn_params.beta = BNUtils.get_beta_as_numpy_data(sess, bn).reshape(-1)
        bn_params.gamma = BNUtils.get_gamma_as_numpy_data(sess, bn).reshape(-1)
        bn_params.runningMean = BNUtils.get_moving_mean_as_numpy_data(sess, bn).reshape(-1)
        bn_params.runningVar = BNUtils.get_moving_variance_as_numpy_data(sess, bn).reshape(-1)
        if bn.type == 'Identity':
            # can't find a way to read epsilon if BN type is Identity
            epsilon = 1.0009999641624745e-05
        else:
            epsilon = BNUtils.get_epsilon(bn)

        var = BNUtils.get_moving_variance_as_numpy_data(sess, bn).reshape(-1)
        var_with_epsilon = var + epsilon
        sigma = np.sqrt(var_with_epsilon)
        # sigma = tf.sqrt(BNUtils.get_moving_variance_as_numpy_data(sess, bn).reshape(-1) + epsilon)
        bn_params.runningVar = sigma
    return bn_params

# pylint: disable=too-many-locals
def _fold_given_auto_selected_batch_norms(sess: tf.compat.v1.Session, layer_pairs: List[PairType]) -> tf.compat.v1.Session:
    """
    Fold a given set of batch_norm layers into conv layers

    :param sess: tf.compat.v1.Session
    :param layer_pairs: pair of conv and bn layers
    :return: new session with updated graph
    """

    with sess.graph.as_default():

        for pair in layer_pairs:

            conv_linear, batchnorm, is_batch_norm_second = pair

            assert conv_linear.type in ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']

            #  check flag
            is_bias_valid = False

            if not BiasUtils.is_bias_none(conv_linear):
                is_bias_valid = True

            bn_params = _get_bn_params(sess, batchnorm.op)
            weight_tensor = _get_weight_tensor_transpose_reshape(sess, conv_linear)
            bias_tensor = _get_bias_tensor(sess, conv_linear)

            bias = libpymo.fold(bn_params, weight_tensor, bias_tensor, is_bias_valid, is_batch_norm_second)

            # converting back to TF format [kh, kw, Nic, Noc] before updating weight tensor value
            if conv_linear.type == 'DepthwiseConv2dNative':
                # Depthwise conv layers in TF have outputs(Noc) set to 1.
                # we send in format [Nic, Noc, kh, kw]
                numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape).transpose((2, 3, 0, 1))
            elif conv_linear.type == 'MatMul':
                # o, i - convert to i , o
                numpy_weight_reshaped = np.reshape(weight_tensor.data,
                                                   [weight_tensor.shape[0], weight_tensor.shape[1]]).transpose(1, 0)
            else:
                # conv2D case
                # we sent in format [Noc, Nic, kh, kw]
                numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape).transpose((2, 3, 1, 0))

            WeightTensorUtils.update_tensor_for_op(sess, conv_linear, numpy_weight_reshaped)

            # remove bn op
            BNUtils.skip_bn_op(sess, batchnorm.op, batchnorm.in_tensor, batchnorm.out_tensor)

            # update bias tensor, even in case there was no existing bias add op in given conv2D op.
            bias_tensor_shape = [weight_tensor.shape[0]]
            numpy_bias_reshaped = np.reshape(bias, bias_tensor_shape)
            BiasUtils.update_bias_for_op(sess, conv_linear, numpy_bias_reshaped)

        # we edited the graph, so we should load and save for the metagraph associated with the session to be updated
        after_bn_fold_sess = save_and_load_graph('./temp_bn_fold', sess)

    return after_bn_fold_sess


def fold_given_batch_norms(sess: tf.compat.v1.Session, input_op_names: Union[str, List[str]],
                           output_op_names: Union[str, List[str]],
                           layer_pairs: List[Tuple[tf.Operation, tf.Operation, bool]]) -> tf.compat.v1.Session:

    """
    Api to fold custom set of bn layers in a model

    :param sess: active tensorflow session
    :param input_op_names: starting op in model or a list of starting ops in the model
    :param layer_pairs: List of tuple with conv and bn op layers as tf.Operation and
           a flag to indicate fold upstream or downstream
    :param output_op_names: List of output op names of the model, used to help ConnectedGraph determine valid ops
           (to ignore training ops for example).
    :return: updated_session after fold

    """

    # check for valid types
    if not isinstance(input_op_names, (str, List)):
        logger.error('start op names must be passed as a string or a List of strings')

    # if passed start op name is a single string, create a list
    if isinstance(input_op_names, str):
        input_op_names = [input_op_names]

    connected_graph = ConnectedGraph(sess.graph, input_op_names, output_op_names)

    conn_tf_n_op_map = {}
    for op in connected_graph.get_all_ops().values():
        if op.type in ['FusedBatchNormV3', 'FusedBatchNorm']:
            conn_tf_n_op_map[op.get_module()] = op

    layer_pairs_internal_format = []
    for layer_pair in layer_pairs:
        conv_op, bn_op, is_bn_op_second = layer_pair
        layer_pairs_internal_format.append((conv_op, conn_tf_n_op_map[bn_op].get_tf_op_with_io_tensor(), is_bn_op_second))

    # invoke internal api
    new_sess = _fold_given_auto_selected_batch_norms(sess, layer_pairs_internal_format)

    # save and load graph
    after_fold_sess = save_and_load_graph('./temp_graph', new_sess)

    return after_fold_sess


def fold_all_batch_norms(sess: tf.compat.v1.Session, input_op_names: Union[str, List[str]],
                         output_op_names: Union[str, List[str]])\
        -> Tuple[tf.compat.v1.Session, List[Tuple[tf.Operation, tf.Operation]]]:
    """
    Fold all batch_norm layers in a model into corresponding conv layers

    :param sess: active tf.compat.v1.Session
    :param input_op_names: Name of the starting op in the given graph or a list of names in case of multi-input model
    :param output_op_names: List of output op names of the model, used to help ConnectedGraph determine valid ops
           (to ignore training ops for example).  If None, all ops in the model are considered valid.
    :return: A new session with edited graph and a list of pairs of layers [(Conv/Linear, BN layer that got folded)]

    """

    # check for valid types
    if not isinstance(input_op_names, (str, List)):
        logger.error('start op names must be passed as a string or a List of strings')

    # if passed start op name is only a string - create a list for connected graph
    if isinstance(input_op_names, str):
        input_op_names = [input_op_names]

    # if passed output op name is only a string - create a list for connected graph
    if isinstance(output_op_names, str):
        output_op_names = [output_op_names]

    bn_conv_linear_pairs = find_all_batch_norms_to_fold(sess, input_op_names, output_op_names)

    after_fold_sess = _fold_given_auto_selected_batch_norms(sess, bn_conv_linear_pairs)

    # When returning the pairs, we want the second element of the pair to be the BN
    pairs_to_return = []

    # tf.Operation type conv , pair[1] nis of type OpWithMetaInfoType
    # bn op is stored as OpWithMetaInfoType, get the op from it.
    # pair[0] is always conv op and bn op is pair[1]
    for pair in bn_conv_linear_pairs:
        pairs_to_return.append((pair[0], pair[1].op))

    return after_fold_sess, pairs_to_return

def fold_all_batch_norms_to_scale(sim: QuantizationSimModel, input_op_names: Union[str, List[str]],
                                  output_op_names: Union[str, List[str]]):
    """
    Fold all batch_norm layers in a model into corresponding conv layers

    :param sim: tf quantized model
    :param input_op_names: Name of the starting op in the given graph or a list of names in case of multi-input model
    :param output_op_names: List of output op names of the model, used to help ConnectedGraph determine valid ops
           (to ignore training ops for example).  If None, all ops in the model are considered valid.
    :return: A new session with edited graph and a list of pairs of layers [(Conv/Linear, BN layer that got folded)]

    """
    # check for valid types
    if not isinstance(input_op_names, (str, List)):
        logger.error('start op names must be passed as a string or a List of strings')
    # if passed start op name is only a string - create a list for connected graph
    if isinstance(input_op_names, str):
        input_op_names = [input_op_names]
    # if passed output op name is only a string - create a list for connected graph
    if isinstance(output_op_names, str):
        output_op_names = [output_op_names]

    sim.export("/tmp/", "sim_model")
    new_sess = graph_saver.load_model_from_meta(meta_path=os.path.join("/tmp/", 'sim_model.meta'))
    bn_conv_linear_pairs = find_all_batch_norms_to_fold(new_sess, input_op_names, output_op_names)
    _fold_given_auto_selected_batch_norms_scale(sim, bn_conv_linear_pairs)

def _fold_given_auto_selected_batch_norms_scale(sim: QuantizationSimModel, layer_pairs: List[PairType]):
    """
     Fold a given set of batch_norm layers into conv layers
    :param sim: tf quantized model
    :param layer_pairs: layer_pairs: pair of conv and bn layers
    """

    sess = sim.session
    with sess.graph.as_default():
        for pair in layer_pairs:
            batchnorm_tf_op = sess.graph.get_operation_by_name(pair[1].op.name)
            bn_quantizer_name = batchnorm_tf_op.name + "_quantized"
            conv_linear_tf_op = sess.graph.get_operation_by_name(pair[0].name)
            assert batchnorm_tf_op.type in ['FusedBatchNormV3', 'Identity']
            #  check flag
            is_bias_valid = False
            if not BiasUtils.is_bias_none(conv_linear_tf_op):
                is_bias_valid = True
                conv_linear_quantizer_a_name = conv_linear_tf_op.outputs[0].consumers()[0].outputs[0].consumers()[
                    0].name
            else:
                conv_linear_quantizer_a_name = conv_linear_tf_op.outputs[0].consumers()[0].name

            conv_linear_quantizer_a = sim.quantizer_config(conv_linear_quantizer_a_name)
            assert isinstance(conv_linear_quantizer_a, aimet_tensorflow.quantizer_info.QuantizerInfo)

            # Disable quantizers activation of conv
            conv_linear_quantizer_a.set_op_mode(int(libpymo.TensorQuantizerOpMode.passThrough))

            # Disable quantizers of batchnorms
            bn_quantizer = sim.quantizer_config(bn_quantizer_name)
            bn_quantizer.set_op_mode(int(libpymo.TensorQuantizerOpMode.passThrough))

            bn_params = _get_bn_params(sess, batchnorm_tf_op)
            weight_tensor = _get_weight_tensor_transpose_reshape(sess, conv_linear_tf_op)
            bias_tensor = _get_bias_tensor(sess, conv_linear_tf_op)
            bias = libpymo.fold(bn_params, weight_tensor, bias_tensor, is_bias_valid, pair[2])
            # converting back to TF format [kh, kw, Nic, Noc] before updating weight tensor value
            if conv_linear_tf_op.type == 'DepthwiseConv2dNative':
                # Depthwise conv layers in TF have outputs(Noc) set to 1.
                # we send in format [Nic, Noc, kh, kw]
                numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape).transpose((2, 3, 0, 1))
            elif conv_linear_tf_op.type == 'MatMul':
                # o, i - convert to i , o
                numpy_weight_reshaped = np.reshape(weight_tensor.data,
                                                   [weight_tensor.shape[0], weight_tensor.shape[1]]).transpose(1, 0)
            else:
                # conv2D case
                # we sent in format [Noc, Nic, kh, kw]
                numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape).transpose((2, 3, 1, 0))
            WeightTensorUtils.update_tensor_for_sim_op(sess, conv_linear_tf_op, numpy_weight_reshaped)
            BiasUtils.update_bias_for_sim_op(sess, conv_linear_tf_op, np.reshape(bias, [weight_tensor.shape[0]]))
            _fold_pair_scale(sim, conv_linear_tf_op, bn_params)
            BNUtils.modify_bn_params_to_make_as_passthrough(sess, batchnorm_tf_op)
        # we edited the graph, so we should load and save for the metagraph associated with the session to be
        # updated
        after_fold_sess = save_and_load_graph('./temp_bn_fold', sess)
    sim.session = after_fold_sess



def _fold_pair_scale(sim: QuantizationSimModel, conv_linear_tf_op: tf.Operation, bn_params: libpymo.BNParams()):
    """
     Fold a batch_norm layer into conv_linear's scale
    :param sim: tf quantized model
    :param conv_linear_tf_op: conv layer or Linear layer
    :param bn_params: bn_params
    """
    conv_linear_quantizer_weights = sim.quantizer_config(conv_linear_tf_op.name + "/ReadVariableOp_quantized")
    if conv_linear_quantizer_weights:
        encodings = conv_linear_quantizer_weights.get_encoding()
        new_encodings = []
        for old_encoding, bn_gamma_to_runningvar_ratio in zip(encodings, np.array(bn_params.gamma) * (1.0 / np.array(bn_params.runningVar))):
            new_encoding = libpymo.TfEncoding()
            if bn_gamma_to_runningvar_ratio >= 0:
                new_encoding.max = old_encoding.max * bn_gamma_to_runningvar_ratio
                new_encoding.min = old_encoding.min * bn_gamma_to_runningvar_ratio
            else:
                new_encoding.max = old_encoding.min * bn_gamma_to_runningvar_ratio
                new_encoding.min = old_encoding.max * bn_gamma_to_runningvar_ratio
            new_encoding.delta = old_encoding.delta * abs(bn_gamma_to_runningvar_ratio)
            new_encoding.offset = new_encoding.min / new_encoding.delta
            new_encoding.bw = old_encoding.bw
            new_encodings.append(new_encoding)
        conv_linear_quantizer_weights.set_encoding(new_encodings)
