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
# pylint: disable=missing-docstring
""" These are TF code examples to be used when generating AIMET documentation via Sphinx """

# tensorflow
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50

# Cross layer Equalization related imports
from aimet_tensorflow.cross_layer_equalization import equalize_model
from aimet_tensorflow.cross_layer_equalization import GraphSearchUtils, CrossLayerScaling, HighBiasFold
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.batch_norm_fold import fold_given_batch_norms
from aimet_tensorflow.utils.graph_saver import save_and_load_graph
from aimet_tensorflow.utils.op.conv import BiasUtils

# Bias correction related imports
from aimet_tensorflow.bias_correction import BiasCorrectionParams, QuantParams, BiasCorrection


def cross_layer_equalization_auto_stepwise():
    """ Individual api calls to perform cross layer equalization one step at a time"""

    # load a model
    tf.keras.backend.clear_session()
    _ = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
    sess = tf.compat.v1.keras.backend.get_session()

    # get starting op name to invoke api for cle
    start_op_name = 'input_1'
    output_op_name = 'fc1000/Softmax'

    with sess.as_default():
        # replace any ReLU6 layers with ReLU
        graph_util = GraphSearchUtils(sess.graph, start_op_name, output_op_name)
        after_relu_replace_sess = graph_util.find_and_replace_relu6_with_relu(sess)

        # fold batchnorm layers
        after_bn_fold_sess, folded_pairs = fold_all_batch_norms(after_relu_replace_sess, start_op_name, output_op_name)

        # perform cross-layer scaling on applicable layer groups
        after_cls_sess, cls_set_info_list = CrossLayerScaling.scale_model(after_bn_fold_sess, start_op_name, output_op_name)

        # perform high bias fold
        # use the session after high bias fold returned for further evaluations on TF graph
        after_hbf_sess = HighBiasFold.bias_fold(after_cls_sess, folded_pairs, cls_set_info_list)
    sess.close()


def cross_layer_equalization_auto():
    """ perform auto cross layer equalization """

    # load a model
    tf.keras.backend.clear_session()
    _ = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
    sess = tf.compat.v1.keras.backend.get_session()

    # get starting op name to invoke api for cle
    input_op_name = 'input_1'
    output_op_name = 'fc1000/Softmax'

    # Equalize a model with Batchnorms
    # Performs BatchNorm fold, replacing Relu6 with Relu, Cross layer scaling and High bias fold
    # use the new session returned for further evaluations on TF graph
    with sess.as_default():
        new_session = equalize_model(sess, input_op_name, output_op_name)
    sess.close()


def get_layer_pairs_Resnet50_for_folding(sess: tf.compat.v1.Session):
    """
    Helper function to pick example conv-batchnorm layer pairs for folding.
    :param sess: tensorflow session as tf.compat.v1.Session
    :return: pairs of conv and batchnorm layers for batch norm folding in Resnet50 model.
    """

    # pick conv and bn op pairs
    conv_op_1 = sess.graph.get_operation_by_name('res2a_branch2a/Conv2D')
    bn_op_1 = sess.graph.get_operation_by_name('bn2a_branch2a/cond/FusedBatchNorm_1')

    conv_op_2 = sess.graph.get_operation_by_name('res2a_branch2b/Conv2D')
    bn_op_2 = sess.graph.get_operation_by_name('bn2a_branch2b/cond/FusedBatchNorm_1')

    conv_op_3 = sess.graph.get_operation_by_name('res2a_branch2c/Conv2D')
    bn_op_3 = sess.graph.get_operation_by_name('bn2a_branch2c/cond/FusedBatchNorm_1')

    # make a layer pair list with potential the conv op and bn_op pair along with a flag
    # to indicate if given bn op can be folded upstream or downstream.
    # example of two pairs of conv and bn op  shown below
    layer_pairs = [(conv_op_1, bn_op_1, True),
                   (conv_op_2, bn_op_2, True),
                   (conv_op_3, bn_op_3, True)]

    return layer_pairs


def get_consecutive_layer_list_from_resnet50_for_scaling(sess: tf.compat.v1.Session):
    """
    helper function to pick example consecutive layer list for scaling.
    :param sess: tf.compat.v1.Session
    :return: sample layers for scaling as consecutive_layer_list from Resnet50 model
    """
    conv1_op = sess.graph.get_operation_by_name('res2a_branch2a/Conv2D')
    conv1_depthwise_op = sess.graph.get_operation_by_name('res2a_branch2b/Conv2D')
    conv1_pointwise_op = sess.graph.get_operation_by_name('res2a_branch2c/Conv2D')

    # conv layers for scaling (after bn fold)
    consecutive_layer_list = [(conv1_op, conv1_depthwise_op, conv1_pointwise_op)]

    return consecutive_layer_list


def format_info_for_high_bias_fold(sess, layer_pairs, consecutive_layer_list, scaling_factor_list):
    """
     Helper function that formats data from cross layer scaling and bn fold for usage by high bias fold.
    :param sess: tf.compat.v1.Session type
    :param layer_pairs: info obtained after batchnorm fold.
    :param consecutive_layer_list: info obtained after cross layer scaling
    :param scaling_factor_list: scaling params corresponding to consecutive_layer_list
    :return: data formatted for high bias fold.
    """

    # convert info after batch norm fold and cross layer scaling for usage by high bias fold api
    folded_pairs = []
    for (conv_op, bn_op_with_meta, _fold_upstream_flag) in layer_pairs:
        folded_pairs.append((conv_op, bn_op_with_meta.op))

    # List that hold a boolean for if there were relu activations between layers of each cross layer scaling set
    is_relu_activation_in_cls_sets = []
    # Note the user is expected to fill in this list manually

    # Convert to a list of cls-set-info elements
    cls_set_info_list = CrossLayerScaling.create_cls_set_info_list(consecutive_layer_list,
                                                                   scaling_factor_list,
                                                                   is_relu_activation_in_cls_sets)

    # load and save the updated graph after scaling
    after_cls_sess = save_and_load_graph('./temp_cls', sess)

    return after_cls_sess, folded_pairs, cls_set_info_list


def cross_layer_equalization_manual():
    """ perform cross layer equalization using manual api"""

    # load a model
    tf.keras.backend.clear_session()
    _ = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
    sess = tf.compat.v1.keras.backend.get_session()

    with sess.as_default():
        # Batch Norm Fold
        # pick potential pairs of conv and bn ops for fold
        layer_pairs = get_layer_pairs_Resnet50_for_folding(sess)

        # fold given layer
        after_fold_sess = fold_given_batch_norms(sess=sess, input_op_names="input_1", output_op_names="fc1000/Softmax",
                                                 layer_pairs=layer_pairs)

        # replace any ReLU6 layers with ReLU
        graph_search = GraphSearchUtils(after_fold_sess.graph, "input_1", "fc1000/Softmax")
        after_relu_replace_sess = graph_search.find_and_replace_relu6_with_relu(after_fold_sess)

        # Cross Layer Scaling
        # Create a list of consecutive conv layers to be equalized
        consecutive_layer_list = get_consecutive_layer_list_from_resnet50_for_scaling(after_relu_replace_sess)

        # invoke api to perform scaling on given list of cls pairs
        scaling_factor_list = CrossLayerScaling.scale_cls_sets(after_relu_replace_sess, consecutive_layer_list)

        # get info from bn fold and cross layer scaling in format required for high bias fold
        after_cls_sess, folded_pairs, cls_set_info_list = format_info_for_high_bias_fold(after_relu_replace_sess,
                                                                                         layer_pairs,
                                                                                         consecutive_layer_list,
                                                                                         scaling_factor_list)

        # perform high-bias fold
        after_hbf_sess = HighBiasFold.bias_fold(after_cls_sess, folded_pairs, cls_set_info_list)
    sess.close()


def bias_correction_single_layer_empirical(dataset: tf.data.Dataset):
    """ perform bias correction on one layer """

    # load a model
    tf.keras.backend.clear_session()
    _ = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
    sess = tf.compat.v1.keras.backend.get_session()

    # input parameters for bias correction
    # populate required parameters in two data types QuantParams and BiasCorrectParams

    quant_params = QuantParams(quant_mode='tf_enhanced',
                               round_mode='nearest',
                               use_cuda=True,
                               ops_to_ignore=None)

    bias_correction_params = BiasCorrectionParams(batch_size=1,
                                                  num_quant_samples=10,
                                                  num_bias_correct_samples=10,
                                                  input_op_names=['input_1'],
                                                  output_op_names=['fc1000/Softmax'])

    with sess.as_default():
        # initialize model with zero bias
        sess = BiasUtils.initialize_model_with_bias(sess, bias_correction_params.input_op_names,
                                                    bias_correction_params.output_op_names)

        # pick a layer for bias correction
        example_conv_layer = sess.graph.get_operation_by_name('res2a_branch2a/Conv2D')

        # invoke bias correction of one layer
        BiasCorrection.bias_correction_per_layer(reference_model=sess,
                                                 corrected_model=sess,
                                                 bias_correct_params=bias_correction_params,
                                                 layer_name_to_be_corrected=example_conv_layer.name,
                                                 quant_params=quant_params,
                                                 data_set=dataset)
    sess.close()


def bias_correction_single_layer_analytical():
    """ perform analytical bias correction on one layer """

    # load a model
    tf.keras.backend.clear_session()
    _ = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
    sess = tf.compat.v1.keras.backend.get_session()

    # input parameters for bias correction
    # populate required parameters in two data types QuantParams and BiasCorrectParams

    quant_params = QuantParams(quant_mode='tf_enhanced',
                               round_mode='nearest',
                               use_cuda=True,
                               ops_to_ignore=None)

    with sess.as_default():
        # initialize model with zero bias
        sess = BiasUtils.initialize_model_with_bias(sess, ['input_1'], ['fc1000/Softmax'])

        # pick a layer for bias correction
        example_conv_layer = sess.graph.get_operation_by_name('res2a_branch2a/Conv2D')

        # get candidate conv bns in the model
        convs_bn_activation_info_dict = BiasCorrection.find_all_convs_bn_with_activation(sess,
                                                                                         ['input_1'],
                                                                                         ['fc1000/Softmax'])

        # make sure to pick example_conv_layer that has a bn op associated with it
        if example_conv_layer in convs_bn_activation_info_dict.keys():

            preceding_bn_layer_info = convs_bn_activation_info_dict[example_conv_layer]

            # invoke analytical bias correction on this layer
            BiasCorrection.analytical_bias_correction_per_layer(sess,
                                                                example_conv_layer,
                                                                preceding_bn_layer_info,
                                                                quant_params)
    sess.close()


def bias_correction_empirical(dataset: tf.data.Dataset):
    """
    Perform bias correction on a given model
    :param dataset: Data passed by user as tf.Dataset type.
    :return: None
    """

    # load a model
    tf.keras.backend.clear_session()
    _ = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
    sess = tf.compat.v1.keras.backend.get_session()

    # input parameters for bias correction
    # populate required parameters in two data types QuantParams and BiasCorrectParams

    quant_params = QuantParams(quant_mode='tf_enhanced',
                               round_mode='nearest',
                               use_cuda=True,
                               ops_to_ignore=None)

    bias_correction_params = BiasCorrectionParams(batch_size=1,
                                                  num_quant_samples=10,
                                                  num_bias_correct_samples=10,
                                                  input_op_names=['input_1'],
                                                  output_op_names=['fc1000/Softmax'])

    with sess.as_default():
        # run bias correction on the model
        _new_session = BiasCorrection.correct_bias(sess, bias_correction_params, quant_params, dataset)
    sess.close()


def bias_correction_empirical_analytical(dataset: tf.data.Dataset):
    """
    Perform bias correction on a given model (mix of empirical and analytical)
    :param dataset: Data passed by user as tf.Dataset type.
    :return: None
    """

    # load a model
    tf.keras.backend.clear_session()
    _ = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
    sess = tf.compat.v1.keras.backend.get_session()

    # input parameters for bias correction
    # populate required parameters in two data types QuantParams and BiasCorrectParams

    quant_params = QuantParams(quant_mode='tf_enhanced',
                               round_mode='nearest',
                               use_cuda=True,
                               ops_to_ignore=None)

    bias_correction_params = BiasCorrectionParams(batch_size=1,
                                                  num_quant_samples=10,
                                                  num_bias_correct_samples=10,
                                                  input_op_names=['input_1'],
                                                  output_op_names=['fc1000/Softmax'])

    with sess.as_default():
        # run empirical and analytical bias correction on the model
        _new_session = BiasCorrection.correct_bias(sess, bias_correction_params, quant_params,
                                                   dataset,
                                                   perform_only_empirical_bias_corr=False)
    sess.close()


def bias_correction_after_cle(dataset: tf.data.Dataset):
    """
    Perform bias correction on a given model (mix of empirical and analytical) after
    cross layer equalization.
    :param dataset: Data passed by user as tf.Dataset type.
    :return: None
    """

    # load a model
    tf.keras.backend.clear_session()
    _ = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
    sess = tf.compat.v1.keras.backend.get_session()

    # input parameters for bias correction
    # populate required parameters in two data types QuantParams and BiasCorrectParams

    quant_params = QuantParams(quant_mode='tf_enhanced',
                               round_mode='nearest',
                               use_cuda=True,
                               ops_to_ignore=None)

    bias_correction_params = BiasCorrectionParams(batch_size=1,
                                                  num_quant_samples=10,
                                                  num_bias_correct_samples=10,
                                                  input_op_names=['input_1'],
                                                  output_op_names=['fc1000/Softmax'])

    with sess.as_default():

        # store conv bns info before performing CLE
        conv_bn_dict = BiasCorrection.find_all_convs_bn_with_activation(sess,
                                                                        start_op_names=['input_1'],
                                                                        output_op_names=['fc1000/Softmax'])

        # perform CLE
        sess_after_cle = equalize_model(sess, start_op_names=['input_1'], output_op_names=['fc1000/Softmax'])

        # run empirical and analytical bias correction on the model
        _new_session = BiasCorrection.correct_bias(sess_after_cle, bias_correction_params, quant_params,
                                                   dataset,
                                                   conv_bn_dict=conv_bn_dict,
                                                   perform_only_empirical_bias_corr=False)
    sess.close()
