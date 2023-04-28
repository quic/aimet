# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

from typing import List, Tuple, Union, Set
import numpy as np
import tensorflow as tf

from aimet_common.graph_searcher import GraphSearcher
from aimet_common.bias_correction import ConvBnPatternHandler
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.utils import AimetLogger
import aimet_common.libpymo as libpymo

from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.common.operation import OpWithMetaInfoType, Op
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils.op.conv import WeightTensorUtils, BiasUtils
from aimet_tensorflow.utils.op.fusedbatchnorm import BNUtils
from aimet_tensorflow.utils.graph_saver import save_and_load_graph
from aimet_tensorflow.utils.op.conv import get_weight_tensor_with_shape
from aimet_tensorflow.utils.common import get_ordered_conv_linears, get_ordered_ops
from aimet_tensorflow.quantizer_info import QuantizerInfo

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.BatchNormFolding)

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


def _find_conv_bn_pairs(conn_graph: ConnectedGraph):
    """
    uses searcher to choose convs/ linears with bn and activation info.
    :param conn_graph: tf.compat.v1.Session type
    :return: dictionary of conv/linear layers with associated bn op / activation info
    """
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
                                 output_op_names: Union[List[str], str], return_bn_conn_op=False) -> Tuple[List[PairType], Set[tf.Operation]]:
    """
    uses searcher to choose layers for bias correction
    :param sess: tf.compat.v1.Session type
    :param start_op_names: list of strings with names of starting ops in the model
    :param output_op_names: List of output op names of the model, used to help ConnectedGraph determine valid ops
    :param return_bn_conn_op: Return bn op as connected graph op instead of tf tensor
    (to ignore training ops for example).  If None, all ops in the model are considered valid.

    :return: List of conv/linear layers with associated bn op / activation info
    """
    if isinstance(start_op_names, str):
        start_op_names = [start_op_names]

    if isinstance(output_op_names, str):
        output_op_names = [output_op_names]

    conn_graph = ConnectedGraph(sess.graph, start_op_names, output_op_names)
    bn_conv_linear_pairs, marked_bn_set = _find_all_batch_norms_to_fold(conn_graph, start_op_names, output_op_names,
                                                                        return_bn_conn_op)
    return bn_conv_linear_pairs, marked_bn_set


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

        if bn.type == 'Identity':
            # can't find a way to read epsilon if BN type is Identity
            epsilon = 0.001
        else:
            epsilon = BNUtils.get_epsilon(bn)

        var = BNUtils.get_moving_variance_as_numpy_data(sess, bn).reshape(-1)
        sigma = np.sqrt(var + epsilon)
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
            conv_linear, bn, fold_backward = pair
            assert conv_linear.type in ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']
            #  check flag
            is_bias_valid = False
            if not BiasUtils.is_bias_none(conv_linear):
                is_bias_valid = True

            bn_params = _get_bn_params(sess, bn.op)
            weight_tensor = _get_weight_tensor_transpose_reshape(sess, conv_linear)
            bias_tensor = _get_bias_tensor(sess, conv_linear)

            bias = libpymo.fold(bn_params, weight_tensor, bias_tensor, is_bias_valid, fold_backward)

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
            BNUtils.skip_bn_op(sess, bn.op, bn.in_tensor, bn.out_tensor)

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

    bn_conv_linear_pairs, bns_to_fold = find_all_batch_norms_to_fold(sess, input_op_names, output_op_names)

    after_fold_sess = _fold_given_auto_selected_batch_norms(sess, bn_conv_linear_pairs)

    # When returning the pairs, we want the second element of the pair to be the BN
    pairs_to_return = []

    # tf.Operation type conv , pair[1] nis of type OpWithMetaInfoType
    # bn op is stored as OpWithMetaInfoType, get the op from it.
    # pair[0] is always conv op and bn op is pair[1]
    for pair in bn_conv_linear_pairs:
        pairs_to_return.append((pair[0], pair[1].op))

    # Convert the standalone BNs which are not folded
    bn_converted = convert_standalone_batchnorms(after_fold_sess, input_op_names, output_op_names, bns_to_fold)
    if bn_converted:
        logger.info("%d BatchNorms' weights got converted", len(bn_converted))

        # we edited the graph, so we should load and save for the metagraph associated with the session to be updated
        after_fold_sess = save_and_load_graph('./temp_bn_fold', after_fold_sess)

    return after_fold_sess, pairs_to_return


def convert_standalone_batchnorms(sess, input_op_names: Union[str, List[str]],
                                  output_op_names: Union[str, List[str]], bns_folded: List) -> List[tf.Operation]:

    """
    Converts the weights of standalone batch norms remaining in the model after BN folding.

    :param sess: TF session in which the graph is loaded
    :param input_op_names: Name of the starting op in the given graph or a list of names in case of multi-input model
    :param output_op_names: List of output op names of the model, used to help ConnectedGraph determine valid ops
           (to ignore training ops for example).  If None, all ops in the model are considered valid.
    :param bns_folded: list of batch norms which got folded
    :return: list of BatchNorms whose weights is converted
    """

    list_of_ordered_ops = get_ordered_ops(sess.graph, input_op_names, output_op_names)

    converted_bns = []
    # look for bn layers which are not folded
    for op in list_of_ordered_ops:
        if op.type in ['FusedBatchNormV3', 'FusedBatchNorm', 'BatchNormalization'] and op not in bns_folded:
            convert_batchnorm_parameters(sess, op)
            converted_bns.append(op)
            logger.debug("%s weights got converted", op)
    return converted_bns


def convert_batchnorm_parameters(sess, op):
    """
    Convert the weights of BN such that it works as y = weights * x + bias

    :param sess: TF Session in which the graph is loaded
    :param op: bn_op which whose weights need to be converted
    """
    bn_params = _get_bn_params(sess, op)
    weight = np.array(bn_params.gamma) / np.array(bn_params.runningVar)
    bias = np.array(bn_params.beta) - np.array(bn_params.runningMean) * weight
    BNUtils.modify_bn_params_to_weight_bias_form(sess, op, weight, bias)


def fold_all_batch_norms_to_scale(sim: QuantizationSimModel,
                                  starting_op_names: List[str],
                                  output_op_names: List[str]):
    """
    Fold all batch_norm layers in a model into the quantization scale parameter
    of the corresponding conv layers

    :param sim: tf quantized model
    :param starting_op_names: List of starting op names of the model
    :param output_op_names: List of output op names of the model
    """
    assert sim.session is not None
    assert sim.connected_graph is not None

    connected_graph = sim.connected_graph
    bn_conv_linear_pairs, _ = _find_all_batch_norms_to_fold(connected_graph, starting_op_names, output_op_names)
    _fold_given_auto_selected_batch_norms_scale(sim, bn_conv_linear_pairs)


def _fold_given_auto_selected_batch_norms_scale(sim: QuantizationSimModel, layer_pairs: List[PairType]):
    """
     Fold a given set of batch_norm layers into conv layers.

     NOTE: Need to retrieve operation(s) by name since TensorFlow graph associated with Connected graph
     and sim.session are different (after save and load step).

    :param sim: QuantizationSimModel object.
    :param layer_pairs pairs of conv and bn layers.
    """
    sess = sim.session
    with sess.graph.as_default():
        for pair in layer_pairs:
            conv_linear, bn, fold_backward = pair

            bn_tf_op = sess.graph.get_operation_by_name(bn.op.name)
            assert bn_tf_op.type in ['FusedBatchNormV3', 'Identity'], "Only Fused BN is supported."
            bn_params = _get_bn_params(sess, bn_tf_op)

            conv_linear_tf_op = sess.graph.get_operation_by_name(conv_linear.name)
            is_bias_valid = False
            if not BiasUtils.is_bias_none(conv_linear_tf_op):
                is_bias_valid = True

            # _fold_to_weight() using FP32 weights and bias (if exists).
            weight_tensor = _get_weight_tensor_transpose_reshape(sess, conv_linear_tf_op)
            bias_tensor = _get_bias_tensor(sess, conv_linear_tf_op)
            bias = libpymo.fold(bn_params, weight_tensor, bias_tensor, is_bias_valid, fold_backward)
            # converting back to TF format [kh, kw, Nic, Noc] before updating weight tensor value
            if conv_linear_tf_op.type == 'DepthwiseConv2dNative':
                # Depthwise conv layers in TF have outputs(Noc) set to 1.
                # we send in format [Nic, Noc, kh, kw]
                numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape).transpose((2, 3, 0, 1))
            elif conv_linear_tf_op.type == 'MatMul':
                # o, i - convert to i, o
                numpy_weight_reshaped = np.reshape(weight_tensor.data,
                                                   [weight_tensor.shape[0], weight_tensor.shape[1]]).transpose((1, 0))
            else:
                # conv2D case
                # we sent in format [Noc, Nic, kh, kw]
                numpy_weight_reshaped = np.reshape(weight_tensor.data, weight_tensor.shape).transpose((2, 3, 1, 0))
            WeightTensorUtils.update_tensor_for_op(sess, conv_linear_tf_op, numpy_weight_reshaped)
            BiasUtils.update_bias_for_op(sess, conv_linear_tf_op, np.reshape(bias, [weight_tensor.shape[0]]))

            # fold to scale
            conv_linear_w_quantizer, conv_linear_a_quantizer, bn_a_quantizer = \
                _find_quantizers(sim, conv_linear_tf_op, bn_tf_op, is_bias_valid)
            _fold_pair_scale(conv_linear_w_quantizer, conv_linear_a_quantizer, bn_a_quantizer, bn_params)

            # remove bn op
            _delete_bn_from_model(sess, bn, is_bias_valid)

    # we edited the graph, so we should load and save for the metagraph associated with the session to be updated
    updated_sess = save_and_load_graph('./temp_bn_fold_to_scale', sess)
    sim.session = updated_sess


def _delete_bn_from_model(sess: tf.compat.v1.Session,
                          bn_op: OpWithMetaInfoType,
                          is_bias_valid: bool):
    """
    Delete BN and BN_quantized ops from the session.graph.
    If BN's previous conv doesn't have bias, is_bias_valid must
    be False. In that case, need to find the correct BN's input tensor.

    Note: supports only Fused BN op types (FusedBatchNormV3, Identity).

    :param sess: TensorFlow session.
    :param bn_op: BN op with meta info.
    :param is_bias_valid: False if BN's preceding Conv doesn't have bias, True otherwise.
    """
    bn_tf_op = sess.graph.get_operation_by_name(bn_op.op.name)
    bn_in_tensor = sess.graph.get_tensor_by_name(bn_op.in_tensor.name)
    bn_out_tensor = sess.graph.get_tensor_by_name(bn_op.out_tensor.name)

    # Find BNs correct input tensor.
    if not is_bias_valid:
        bn_in_tensor = bn_in_tensor.consumers()[0].outputs[0].consumers()[0].outputs[0]
        assert bn_in_tensor.op.type == 'BiasAdd', 'BNs preceding op must be of type BiasAdd.'
    else:
        bn_in_tensor = bn_in_tensor.consumers()[0].outputs[0]
        assert bn_in_tensor.op.type == 'QcQuantize', 'BNs preceding op must be of type QcQuantize.'

    # Find BNs correct output tensor.
    bn_out_tensor = bn_out_tensor.consumers()[0].outputs[0]
    assert bn_out_tensor.op.type == 'QcQuantize', 'BNs output op must be of type QcQuantize.'

    # Detach BN and following BN_quantized ops from the graph.
    BNUtils.skip_bn_op(sess, bn_tf_op, bn_in_tensor, bn_out_tensor)


def _fold_pair_scale(conv_linear_w_quantizer: QuantizerInfo,
                     conv_linear_a_quantizer: QuantizerInfo,
                     bn_a_quantizer: QuantizerInfo,
                     bn_params: libpymo.BNParams):
    """
     Fold a batch_norm layer into conv_linear's scale

    :param conv_linear_w_quantizer: conv or Linear op weight quantizer.
    :param conv_linear_a_quantizer: conv or Linear op activation quantizer
    :param bn_a_quantizer: BN op activation quantizer
    :param bn_params: bn_params
    """
    if all(quantizer is None for quantizer in [conv_linear_w_quantizer, conv_linear_a_quantizer, bn_a_quantizer]):
        raise RuntimeError

    encodings = conv_linear_w_quantizer.get_encoding()
    if encodings is None:
        raise RuntimeError

    gamma = np.array(bn_params.gamma)
    sigma = np.array(bn_params.runningVar)

    new_encodings = []
    for old_encoding, c in zip(encodings, gamma/sigma):
        new_encoding = libpymo.TfEncoding()
        new_encoding.delta = old_encoding.delta * abs(c)
        if c >= 0:
            new_encoding.max = old_encoding.max * c
            new_encoding.min = old_encoding.min * c
        else:
            new_encoding.max = old_encoding.min * c
            new_encoding.min = old_encoding.max * c
        new_encoding.offset = old_encoding.offset
        new_encoding.bw = old_encoding.bw
        new_encodings.append(new_encoding)

    conv_linear_w_quantizer.set_encoding(new_encodings)

    # Copy batchnorm's output quantizers to conv output quantizers
    conv_linear_a_quantizer.enabled = bn_a_quantizer.enabled

    if bn_a_quantizer.get_encoding() is not None:
        encoding = libpymo.TfEncoding()
        bn_encoding = bn_a_quantizer.get_encoding()
        encoding.delta = bn_encoding.delta
        encoding.max = bn_encoding.max
        encoding.min = bn_encoding.min
        encoding.offset = bn_encoding.offset
        encoding.bw = bn_encoding.bw
        conv_linear_a_quantizer.set_op_mode(int(libpymo.TensorQuantizerOpMode.quantizeDequantize))
        conv_linear_a_quantizer.set_encoding(encoding)

    bn_a_quantizer.enabled = False


def _find_all_batch_norms_to_fold(conn_graph: ConnectedGraph,
                                  start_op_names: List[str],
                                  output_op_names: List[str],
                                  return_bn_conn_op: bool = False) -> Tuple[List, Set]:
    """
    Find all possible batch norm layers that can be folded. And returns a list of pairs such that (bn, layer)
    means bn will be forward-folded into layer and (layer, bn) means bn will be backward-folded into layer

    :param conn_graph: Connected graph associated with the model.
    :param start_op_names: List of starting op names of the model
    :param output_op_names: List of output op names of the model
    :param return_bn_conn_op: Return bn op as connected graph op instead of tf tensor if True.
    :return: A list of (layer, bn) pairs and a list of (bn, layer) pairs,
             where `bn` can be folded into to `layer',
             A set of bn ops which can be folded.
    """
    conv_linear_bn_activation_info_dict = _find_conv_bn_pairs(conn_graph)

    # get all ordered conv/linear ops
    ordered_conv_linear_op = get_ordered_conv_linears(conn_graph.graph, start_op_names, output_op_names)

    # get the in out tensor for bns found, we need this on TF to remove the bns after fold.
    bn_conv_linear_pairs = []

    # track BNs added for fold
    bn_picked_for_folding = set()

    for conv_linear_op in ordered_conv_linear_op:
        if conv_linear_op in conv_linear_bn_activation_info_dict.keys():
            bn_info = conv_linear_bn_activation_info_dict[conv_linear_op]
            if bn_info.output_bn:
                if bn_info.output_bn not in bn_picked_for_folding:
                    fold_backward = True
                    if return_bn_conn_op:
                        bn_conv_linear_pairs.append((conv_linear_op, bn_info.output_bn, fold_backward))
                    else:
                        bn_conv_linear_pairs.append((conv_linear_op, bn_info.output_bn.get_tf_op_with_io_tensor(),
                                                     fold_backward))
                    bn_picked_for_folding.add(bn_info.output_bn)
            elif bn_info.input_bn:
                if bn_info.input_bn not in bn_picked_for_folding:
                    fold_backward = False
                    if return_bn_conn_op:
                        bn_conv_linear_pairs.append((conv_linear_op, bn_info.input_bn, fold_backward))
                    else:
                        bn_conv_linear_pairs.append((conv_linear_op, bn_info.input_bn.get_tf_op_with_io_tensor(),
                                                     fold_backward))
                    bn_picked_for_folding.add(bn_info.input_bn)
    return bn_conv_linear_pairs, bn_picked_for_folding


def _find_quantizers(sim: QuantizationSimModel,
                     conv_linear_tf_op: tf.Operation,
                     bn_tf_op: tf.Operation,
                     is_bias_valid: bool) -> Tuple[QuantizerInfo, QuantizerInfo, QuantizerInfo]:
    """
    Find quantizers.

    :param sim: QuantizationSimModel object
    :param conv_linear_tf_op: Conv/Linear tf operation.
    :param bn_tf_op: BN tf operation
    :param is_bias_valid: is bias valid.
    :return: conv/linear weight quantizer, conv/linear activation quantizer, bn activation quantizer.
    """
    if is_bias_valid:
        bias_add_op = conv_linear_tf_op.outputs[0].consumers()[0]
        assert bias_add_op.type == 'BiasAdd'
        conv_linear_a_quantizer_op = bias_add_op.outputs[0].consumers()[0]
        assert conv_linear_a_quantizer_op.type == 'QcQuantize'
        conv_linear_a_quantizer_name = conv_linear_a_quantizer_op.name
    else:
        conv_linear_a_quantizer_op = conv_linear_tf_op.outputs[0].consumers()[0]
        assert conv_linear_a_quantizer_op.type == 'QcQuantize'
        conv_linear_a_quantizer_name = conv_linear_a_quantizer_op.name

    bn_a_quantizer_name = bn_tf_op.name + "_quantized"
    conv_linear_w_quantizer_name = conv_linear_tf_op.inputs[1].op.inputs[0].op.name + "_quantized"

    conv_linear_w_quantizer = sim.quantizer_config(conv_linear_w_quantizer_name)
    conv_linear_a_quantizer = sim.quantizer_config(conv_linear_a_quantizer_name)
    bn_a_quantizer = sim.quantizer_config(bn_a_quantizer_name)

    return conv_linear_w_quantizer, conv_linear_a_quantizer, bn_a_quantizer
