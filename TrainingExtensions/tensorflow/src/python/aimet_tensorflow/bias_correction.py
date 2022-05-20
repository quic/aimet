# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Code to perform bias correction for layers """

from typing import List, Union, Tuple, Dict
import numpy as np
import tensorflow as tf

import aimet_common.libpymo as libpymo
from aimet_common.bias_correction import ConvBnInfoType
from aimet_common.defs import ActivationType, QuantScheme
from aimet_common.utils import AimetLogger
from aimet_common.graph_searcher import GraphSearcher
from aimet_common.bias_correction import ConvBnPatternHandler
from aimet_common.graph_pattern_matcher import PatternType

from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils.graph_saver import save_model_to_meta, save_and_load_graph, load_model_from_meta
from aimet_tensorflow.utils.common import create_input_feed_dict, iter_first_x, get_ordered_conv_linears
from aimet_tensorflow.utils.op.fusedbatchnorm import BNUtils
from aimet_tensorflow.utils.op.conv import get_weight_tensor_with_shape, BiasUtils
from aimet_tensorflow.common.connectedgraph import ConnectedGraph


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class QuantParams:
    """
    Quant Params to be passed in by user

    """

    def __init__(self,
                 quant_mode='tf_enhanced',
                 round_mode='nearest',
                 use_cuda=True,
                 ops_to_ignore=None):
        """
        Constructor

        :param quant_mode: Indicates which quantization algorithm should be used, either
                           'tf' or 'tf_enhanced'. Defaults to 'tf_enhanced'
        :param round_mode:  The round scheme to used. One of: 'nearest' or 'stochastic'. Default is 'nearest'.
        :param use_cuda: flag to indicate if GPU is to be used
        :param ops_to_ignore: ops to be ignored
        """
        self.quant_mode = quant_mode
        self.round_mode = round_mode
        self.ops_to_ignore = ops_to_ignore
        self.use_cuda = use_cuda


class BiasCorrectionParams:
    """
    Input for bias correction to be passed by the user

    :param batch_size: input batch size to be used
    :param num_quant_samples: samples to be used for quantization
    :param num_bias_correct_samples: samples to be used for bias correction
    :param input_op_names: list of input op names of the given model
    :param output_op_names: list of output op names of the given model

    """

    def __init__(self,
                 batch_size: int,
                 num_quant_samples: int,
                 num_bias_correct_samples: int,
                 input_op_names: List[str],
                 output_op_names: List[str]):

        self.batch_size = batch_size
        self.num_quant_samples = num_quant_samples
        self.num_bias_correct_samples = num_bias_correct_samples
        self.input_op_names = input_op_names
        self.output_op_names = output_op_names


class BiasCorrection:
    """
    class for bias correction in tensorflow
    """

    @staticmethod
    def _get_output_data(sess: tf.compat.v1.Session, input_op_names: List[str], output_op_name: str,
                         batch_data: Union[np.ndarray, Tuple[np.ndarray], List[np.ndarray]]) -> np.ndarray:
        """
        Function to get output values of a layer
        :param sess: tf.compat.v1.Session containing the layer to evaluate
        :param input_op_names: List of names of input ops to the session graph
        :param output_op_name: Name of the output layer to evaluate
        :param batch_data: Batch of data to feed into model input
        :return: Output of layer for all batches of images
        """

        feed_dict = create_input_feed_dict(sess.graph, input_op_names, batch_data)
        tf_op = sess.graph.get_operation_by_name(output_op_name)
        assert tf_op.outputs
        assert tf_op.outputs[0].consumers()
        assert tf_op.outputs[0].consumers()[0].outputs
        biasadd_tensor = tf_op.outputs[0].consumers()[0].outputs[0]     # Replace with a get BiasAdd utils later
        output_data = sess.run(biasadd_tensor, feed_dict=feed_dict)
        return output_data

    @staticmethod
    def _call_mo_correct_bias(corrected_model: tf.compat.v1.Session, layer_name: str,
                              bias_correction: libpymo.BiasCorrection,
                              bias_shape: int, is_bias_none: bool):
        """
         helper to perform bias correction using cpp backend
        :param corrected_model: active tensorflow session with corrected model as tf.compat.v1.Session
        :param layer_name: name of the layer to be bias corrected
        :param bias_correction: bias correction inputs
        :param bias_shape: shape of bias associated with the layer
        :param is_bias_none: True if bias for a layer is None
        :return: None, updates bias for the given layer
        """

        bias_tensor = libpymo.TensorParamBiasCorrection()

        layer_to_be_corrected = corrected_model.graph.get_operation_by_name(layer_name)

        with corrected_model.graph.as_default():
            assert(layer_to_be_corrected.type in ['Conv2D', 'DepthwiseConv2dNative', 'MatMul'])

            if is_bias_none:
                bias_tensor.data = np.zeros(bias_shape)
            else:
                # read bias from given op
                bias_tensor.data = BiasUtils.get_bias_as_numpy_data(corrected_model, layer_to_be_corrected)

            # perform bias correction
            bias_correction.correctBias(bias_tensor)

            # this api updates bias or adds bias add to layer if not present
            BiasUtils.update_bias_for_quantized_op(corrected_model, layer_to_be_corrected, np.array(bias_tensor.data),
                                                   is_bias_none)

    @staticmethod
    def _get_quantized_model(corrected_model: tf.compat.v1.Session, quant_params: QuantParams, input_op_names: List[str],
                             output_op_names: List[str], num_quant_samples: int, batch_size: int,
                             data_set: tf.data.Dataset) -> QuantizationSimModel:
        """
         api to get quantized session
        :param corrected_model: active tensorflow session with corrected model as tf.compat.v1.Session
        :param quant_params: quantization params from user
        :param input_op_names: names of the input nodes of the given model
        :param output_op_names: names of the output nodes of the given model
        :param num_quant_samples: number of dataset samples to use during quantization
        :param batch_size: batch size to use for dataset samples
        :return: quantized sim model
        """

        def bias_correction_callback(session: tf.compat.v1.Session, iterations: int):
            dataset_samples_quant_itr = iter_first_x(data_set, iterations)
            output_ops = []
            for output_op_name in output_op_names:
                output_ops.append(session.graph.get_operation_by_name(output_op_name))
            for data in dataset_samples_quant_itr:
                feed_dict = create_input_feed_dict(session.graph, input_op_names, data)
                for output_op in output_ops:
                    output_op.outputs[0].eval(session=session, feed_dict=feed_dict)

        save_model_to_meta(corrected_model, './bias_correction/temp')

        # Allocate the quantizer and quantize the network using the default 8 bit params/activations
        quantsim = QuantizationSimModel(corrected_model, input_op_names, output_op_names,
                                        quant_params.quant_mode, quant_params.round_mode)

        # Disable all output quantizers
        # pylint:disable = protected-access
        for quantize_op in quantsim._activation_quantizers:
            if quantsim._activation_quantizers[quantize_op].enabled:
                quantsim._activation_quantizers[quantize_op].enabled = False

        n_batches_quantization = int(np.ceil(num_quant_samples / batch_size))
        quantsim.compute_encodings(bias_correction_callback, forward_pass_callback_args=n_batches_quantization)

        return quantsim


    # pylint: disable=too-many-locals
    @staticmethod
    def bias_correction_per_layer(reference_model: tf.compat.v1.Session,
                                  corrected_model: tf.compat.v1.Session,
                                  bias_correct_params: BiasCorrectionParams,
                                  layer_name_to_be_corrected: str,
                                  data_set: tf.data.Dataset) -> tf.compat.v1.Session:
        """
         Helper function to perform empirical bias correction per layer.

        :param reference_model: active tensorflow session for reference model
        :param corrected_model: active tensorflow session for corrected model
        :param bias_correct_params: bias correction params
        :param layer_name_to_be_corrected: name of layer on which bias correction is to be performed
        :param quant_params: Quantization specific params from user
        :return: None, updates corrected model in-place.

        """

        ref_layer = reference_model.graph.get_operation_by_name(layer_name_to_be_corrected)

        bias_correction = libpymo.BiasCorrection()
        logger.info('Correcting layer %s', ref_layer.name)

        n_batches_bias_correction = int(np.ceil(bias_correct_params.num_bias_correct_samples /
                                                bias_correct_params.batch_size))

        reduced_dataset_iter = iter_first_x(data_set, n_batches_bias_correction)

        for batch_input in reduced_dataset_iter:
            # reference model without corrected nodes
            reference_output_batch = BiasCorrection._get_output_data(reference_model,
                                                                     bias_correct_params.input_op_names,
                                                                     ref_layer.name,
                                                                     batch_input)

            quantized_model_output_batch = BiasCorrection._get_output_data(corrected_model,
                                                                           bias_correct_params.input_op_names,
                                                                           ref_layer.name,
                                                                           batch_input)



            if ref_layer.type == 'MatMul':
                extended_shape = np.concatenate((reference_output_batch.shape, np.array([1, 1])))
                reference_output_batch = reference_output_batch.reshape(extended_shape)
                quantized_model_output_batch = quantized_model_output_batch.reshape(extended_shape)

            # we need to reshape from tensorflow shape NxHxWxC to NxCxHxW
            bias_correction.storePreActivationOutput(np.ascontiguousarray(reference_output_batch.transpose(0, 3, 1, 2)))
            bias_correction.storeQuantizedPreActivationOutput(np.ascontiguousarray(
                quantized_model_output_batch.transpose(0, 3, 1, 2)))

        bias_shape = None
        is_bias_none = False
        # get shape for bias if the layer does not have bias
        if BiasUtils.is_bias_none(ref_layer):
            is_bias_none = True
            if ref_layer.type == 'MatMul':
                bias_shape = reference_output_batch.shape[1]
            elif ref_layer.type in ['Conv2D', 'DepthwiseConv2dNative']:
                # for conv2d or depthwise conv2d
                bias_shape = reference_output_batch.shape[3]

        # bias is to be corrected in the corrected model graph
        BiasCorrection._call_mo_correct_bias(corrected_model, ref_layer.name, bias_correction, bias_shape,
                                             is_bias_none)

        logger.info('Completed empirical bias correction for layer  %s', ref_layer.name)

    @staticmethod
    def _get_quantized_weights(weight_tensor, quant_params):
        """
        helper function to get quantized dequantized weights
        :param weight_tensor: weight tensor
        :param quant_params: quantization params such as mode, rounding  etc
        :return: quantized de-quantized weight tensor
        """

        q_wt_tensor = weight_tensor

        quant_mode = libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED
        if quant_params.quant_mode == QuantScheme.post_training_tf or quant_params.quant_mode == 'tf':
            quant_mode = libpymo.QuantizationMode.QUANTIZATION_TF

        round_mode = libpymo.RoundingMode.ROUND_NEAREST
        if quant_params.round_mode == 'stochastic':
            round_mode = libpymo.RoundingMode.ROUND_STOCHASTIC

        bitwidth = 8

        # use tensorQuantizerForPython to get quantizeDequantize weights
        encoding_analyzer = libpymo.EncodingAnalyzerForPython(quant_mode)
        encoding_analyzer.updateStats(weight_tensor, quant_params.use_cuda)
        encoding, is_encoding_valid = encoding_analyzer.computeEncoding(bitwidth, False, False, False)

        if is_encoding_valid:
            tensor_quantizer = libpymo.TensorQuantizationSimForPython()
            q_wt_tensor = tensor_quantizer.quantizeDequantize(weight_tensor, encoding, round_mode, quant_params.use_cuda)

        return q_wt_tensor


    @staticmethod
    def _get_conv_linear_params(model, layer_to_be_corrected):
        """
        Extract weights and bias of given conv/linear layer
        :param model: tf.compat.v1.Session type
        :param layer_to_be_corrected: conv/linear layer as tf.Operation
        :return: bias, weight and quantized weights as TensorParamBiasCorrection types
        """

        bias_tensor = libpymo.TensorParamBiasCorrection()

        # get weight tensor
        weight_tensor, _ = get_weight_tensor_with_shape(model, layer_to_be_corrected)

        if weight_tensor is None:
            logger.error('Weight tensor extraction failed for layer {%s}', layer_to_be_corrected.name)

        bias_tensor.data = BiasUtils.get_bias_as_numpy_data(model, layer_to_be_corrected)
        bias_tensor.shape = BiasUtils.get_shape(layer_to_be_corrected)

        return bias_tensor, weight_tensor

    @staticmethod
    def _get_bn_params(model, bn_layer) -> libpymo.BnParamsBiasCorr():
        """
        get bn params for bn based bias correction
        :param model: tf.compat.v1.Session type
        :param bn_layer: tf.Operation type
        :return: bn params as libpymo.BnParamsBiasCorr() type
        """

        bn_params = libpymo.BnParamsBiasCorr()
        bn_params.beta = BNUtils.get_beta_as_numpy_data(model, bn_layer).reshape(-1)
        bn_params.gamma = BNUtils.get_gamma_as_numpy_data(model, bn_layer).reshape(-1)

        return bn_params

    @staticmethod
    def analytical_bias_correction_per_layer(corrected_model: tf.compat.v1.Session, layer: tf.Operation,
                                             preceeding_bn_layer_info: ConvBnInfoType, quant_params: QuantParams,
                                             is_first_conv: bool = False) -> tf.compat.v1.Session:
        """
        Perform bn based bias correction (analytical bc).

        :param corrected_model: active tensorflow session for corrected model
        :param layer: conv/linear layer to be corrected
        :param preceeding_bn_layer_info: corresponding preceeding bn/ activation info
        :param quant_params: Quantization specific params from user
        :param is_first_conv: flag to indicate if it's the first conv layer
        :return: None, updates corrected_model in place

        """

        layer = corrected_model.graph.get_operation_by_name(layer.name)
        # get bn param and quantized weights from conv for this layer
        bias_tensor, weight_tensor = BiasCorrection._get_conv_linear_params(corrected_model, layer)
        quantized_weight = BiasCorrection._get_quantized_weights(weight_tensor, quant_params)

        bn_params = libpymo.BnParamsBiasCorr()
        activation_type = libpymo.ActivationType.noActivation

        if preceeding_bn_layer_info:
            input_tf_bn_op_name = preceeding_bn_layer_info.input_bn.get_module().name
            bn_op = corrected_model.graph.get_operation_by_name(input_tf_bn_op_name)
            bn_params = BiasCorrection._get_bn_params(corrected_model, bn_op)
            if preceeding_bn_layer_info.in_activation_type == ActivationType.relu:
                activation_type = libpymo.ActivationType.relu
            elif preceeding_bn_layer_info.in_activation_type == ActivationType.relu6:
                activation_type = libpymo.ActivationType.relu6
            elif preceeding_bn_layer_info.in_activation_type == ActivationType.no_activation:
                activation_type = libpymo.ActivationType.noActivation
            else:
                assert(0, 'Unknown activation type', preceeding_bn_layer_info.in_activation_type)
        else:
            if is_first_conv:
                # for the first conv layer case, we use gamma = 1 and beta = 0
                shape = weight_tensor.shape[1]
                bn_params.gamma = np.ones(shape)
                bn_params.beta = np.zeros(shape)
            else:
                assert 0, "layer info is None and is not first conv layer"

        # need to invoke cpp api for bn based bias correction
        biasCorrection = libpymo.BnBasedBiasCorrection()

        biasCorrection.correctBias(bias_tensor, quantized_weight, weight_tensor, bn_params, activation_type)

        # this api updates bias or adds bias add to layer if not present
        layer = corrected_model.graph.get_operation_by_name(layer.name)
        BiasUtils.update_bias_for_quantized_op(corrected_model, layer, np.array(bias_tensor.data))
        logger.info('Completed analytical bias correction for layer %s', layer.name)

    @staticmethod
    def _conv_bn_select_custom_pattern_init():
        """
        initialize the patterns we want to use to pick layers for bn based bias correction.
        :return: patterns and associated actions to be performed upon match
        """

        patterns_with_callbacks = []

        # the types we want to handle
        conv_layer_types = ['Conv2D', 'DepthwiseConv2dNative']
        activation_types = ['Relu', 'Relu6']

        # add the patterns we are interested in along with a handler
        layer_select_handler = ConvBnPatternHandler()

        # conv layer combinations
        for conv in conv_layer_types:

            for activation in activation_types:
                patterns_with_callbacks.append(PatternType(pattern=['FusedBatchNormV3', activation, conv],
                                                           action=layer_select_handler))

            patterns_with_callbacks.append(PatternType(pattern=['FusedBatchNormV3', conv],
                                                       action=layer_select_handler))

        return patterns_with_callbacks, layer_select_handler

    @staticmethod
    def find_all_convs_bn_with_activation(model, start_op_names: Union[List[str], str],
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
        patterns_with_callback, layer_select_handler = BiasCorrection._conv_bn_select_custom_pattern_init()

        # graph searcher looks for patterns and applies actions when matching patterns are found
        graph_searcher = GraphSearcher(conn_graph, patterns_with_callback)
        graph_searcher.find_all_patterns_in_graph_apply_actions()

        # use custom handler instance and fetch the selected layer info for bias correction
        convs_bn_activation_info_dict = layer_select_handler.get_conv_linear_bn_info_dict()

        return convs_bn_activation_info_dict

    @staticmethod
    def refresh_op_ref(sess, conv_bn_dict):
        """
        Updates the conv op references saved in user passed in conv bn dictionary.

        :param reference_model: active tf.compat.v1.Session for the model.
        :param conv_bn_dict: Dict of conv and bn with activation info
        :return: dict of conv and bn with updated conv references

        """
        conv_linears_with_bn_dict = {}
        for conv in conv_bn_dict.keys():
            refreshed_conv = sess.graph.get_operation_by_name(conv.name)
            bn_activation_info = conv_bn_dict[conv]
            conv_linears_with_bn_dict[refreshed_conv] = bn_activation_info

        return conv_linears_with_bn_dict

    @staticmethod
    def correct_bias(reference_model: tf.compat.v1.Session, bias_correct_params: BiasCorrectionParams,
                     quant_params: QuantParams, data_set: tf.data.Dataset,
                     conv_bn_dict: Union[Dict[tf.Operation, ConvBnInfoType], None] = None,
                     perform_only_empirical_bias_corr: bool = True):
        """
         Top level function for bias correction

        :param reference_model: active tf.compat.v1.Session for the model to be corrected.
        :param bias_correct_params: input params for bias correction
        :param quant_params: QuantParams type with params for quantization simulation for bias correction.
        :param data_set: input data set
        :param conv_bn_dict: Dict of conv and bn with activation info. If None, the function looks for it.
                             This can be obtained on the model with bns and convs using
                             BiasCorrection.find_all_convs_bn_with_activation() api.
        :param perform_only_empirical_bias_corr: a flag to indicate only empirical bias correction is to be performed.
        :return: updated session with corrected bias for given ops

        """

        # one time initialization of all layers with bias param
        reference_model = BiasUtils.initialize_model_with_bias(reference_model, bias_correct_params.input_op_names,
                                                               bias_correct_params.output_op_names)

        # Create a copy of the model as reference model
        corrected_model = save_and_load_graph('./temp_meta_path', reference_model)

        # get all ordered convs/ linears and skip gradient ops
        ordered_conv_linears = get_ordered_conv_linears(reference_model, bias_correct_params.input_op_names,
                                                        bias_correct_params.output_op_names)

        # Get conv2D, depthwise with preceding BN ops info for analytical bias correction
        # if user has not passed any dictionary
        if conv_bn_dict is None:
            convs_bn_activation_info_dict = BiasCorrection.find_all_convs_bn_with_activation(reference_model,
                                                                                             bias_correct_params.input_op_names,
                                                                                             bias_correct_params.output_op_names)
        else:
            convs_bn_activation_info_dict = BiasCorrection.refresh_op_ref(reference_model, conv_bn_dict)

        # Quantize model
        quantsim = BiasCorrection._get_quantized_model(corrected_model, quant_params,
                                                       bias_correct_params.input_op_names,
                                                       bias_correct_params.output_op_names,
                                                       bias_correct_params.num_quant_samples,
                                                       bias_correct_params.batch_size,
                                                       data_set)

        # Perform analytical bias correction for first conv layer
        # we always perform empirical bias correction for linear layers
        if ordered_conv_linears:
            if not perform_only_empirical_bias_corr and ordered_conv_linears[0].type not in ['MatMul']:
                first_conv = ordered_conv_linears.pop(0)
                BiasCorrection.analytical_bias_correction_per_layer(quantsim.session,
                                                                    first_conv,
                                                                    None,
                                                                    quant_params,
                                                                    is_first_conv=True)

        # for each candidate layer in an ordered list of conv/lieanr ops
        # find the corresponding bn and activation info
        for layer in ordered_conv_linears:

            # if this layer is in selected patterns of convs with preceding BN op and
            # if empirical flag is false
            # perform analytical Bias correction
            if layer in convs_bn_activation_info_dict.keys() and not perform_only_empirical_bias_corr:

                preceding_bn_layer_info = convs_bn_activation_info_dict[layer]

                BiasCorrection.analytical_bias_correction_per_layer(quantsim.session,
                                                                    layer,
                                                                    preceding_bn_layer_info,
                                                                    quant_params)
            else:
                # stand-alone convs/ linears or when perform_only_empirical_bias_corr is set to True
                # perform empirical bias correction
                BiasCorrection.bias_correction_per_layer(reference_model,
                                                         quantsim.session,
                                                         bias_correct_params,
                                                         layer.name,
                                                         data_set)
        logger.info('Completed bias correction')
        # Remove quantization nodes and save bias correction model
        # pylint:disable = protected-access
        quantsim._remove_quantization_nodes_and_save_graph('./temp_meta_path', 'bias_corrected_model')
        corrected_model = load_model_from_meta(meta_path=str('./temp_meta_path' + '/' + 'bias_corrected_model' +
                                                             '.meta'))
        return corrected_model
