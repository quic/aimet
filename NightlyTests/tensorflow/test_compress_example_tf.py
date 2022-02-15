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
""" Acceptance tests for various compression techniques """

import pytest
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import unittest
import unittest.mock
import logging
import shutil
from decimal import Decimal
from packaging import version
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet import MobileNet

import aimet_common.defs
import aimet_tensorflow.utils.graph_saver
from aimet_common.utils import AimetLogger
import aimet_tensorflow.defs
from aimet_tensorflow.defs import ModuleCompRatioPair
from aimet_tensorflow.common import graph_eval
from aimet_tensorflow.compress import ModelCompressor
from aimet_tensorflow.common import tfrecord_generator
from aimet_tensorflow.common.tfrecord_generator import MnistParser
from aimet_tensorflow.utils.graph import update_keras_bn_ops_trainable_flag
from aimet_tensorflow.examples.test_models import model_with_three_convs

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
tf.compat.v1.disable_eager_execution()

mnist_model_path = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'mnist/models/')
mnist_tfrecords_path = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'mnist/data/')


def tiny_imagenet_parse(serialized_example):
    """
    Parser for TINY IMAGENET models, reads the tfRecords file
    :param serialized_example:
    :return: Input image
    """

    # This works for tf_slim model: resnet_50_v2 but NOT for Keras VGG16
    # Dense features in Example proto.
    feature_map = {
        'height': tf.compat.v1.FixedLenFeature((), tf.int64),
        'width': tf.compat.v1.FixedLenFeature((), tf.int64),
        'channel': tf.compat.v1.FixedLenFeature((), tf.int64),
        'label': tf.compat.v1.FixedLenFeature((), tf.int64),
        'image_raw': tf.compat.v1.FixedLenFeature((), tf.string),
        'location_raw': tf.compat.v1.FixedLenFeature((), tf.string)}

    features = tf.compat.v1.parse_single_example(serialized_example, feature_map)

    image_raw = tf.compat.v1.decode_raw(features["image_raw"], tf.uint8)
    image = tf.reshape(image_raw, [64, 64, 3])

    return image


def imagenet_parse(serialized_example):
    """
    Parser for IMAGENET models, reads the tfRecords file
    :param serialized_example:
    :return: Input image and labels
    """
    dim = 224

    features = tf.compat.v1.parse_single_example(serialized_example,
                                                 features={
                                                     'image/class/label': tf.FixedLenFeature([], tf.int64),
                                                     'image/encoded': tf.FixedLenFeature([], tf.string)})
    image_data = features['image/encoded']

    # Decode the jpeg
    with tf.name_scope('prep_image', [image_data], None):
        # decode and reshape to default 224x224
        # pylint: disable=no-member
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [dim, dim])

    return image


def evaluate(model: tf.compat.v1.Session, iterations: int, use_cuda: bool):

    """
    eval function for MNIST LeNet model
    :param model: tf.compat.v1.Session
    :param iterations: iterations
    :param use_cuda: use_cuda
    :return:
    """

    total_test_images = 10000
    batch_size = 64

    # iterate over entire test data set, when iterations is None
    # TODO : figure out way to end iterator when the data set is exhausted
    if iterations is None:
        iterations = int(total_test_images / batch_size)

    parser = MnistParser(data_inputs=['reshape_input'], validation_inputs=['labels'], batch_size=batch_size)

    # Allocate the generator you wish to use to provide the network with data
    generator = tfrecord_generator.TfRecordGenerator(tfrecords=[os.path.join(mnist_tfrecords_path, 'validation.tfrecords')],
                                                     parser=parser, num_gpus=1)

    # Create the tensor map for input and ground truth ops
    input_tensor_map = {}
    inputs = ['reshape_input', 'labels']

    for name in inputs:
        input_tensor_map[name] = model.graph.get_tensor_by_name(name + ':0')

    # get the evaluation tensor
    eval_tensor = model.graph.get_tensor_by_name('accuracy:0')

    avg_accuracy = 0
    current_iterations = 0

    for batch in generator:

        current_iterations += 1
        # Setup the feed dictionary
        feed_dict = {}

        for name, data in batch.items():
            feed_dict[input_tensor_map[name]] = data

        with model.as_default():
            accuracy = model.run(eval_tensor, feed_dict=feed_dict)

        avg_accuracy += accuracy

        if current_iterations >= iterations:
            break

    return avg_accuracy / current_iterations


class SvdAcceptanceTests(unittest.TestCase):

    def test_spatial_svd_compress_auto_with_finetuning(self):
        """
        End to end test with MNIST model following fine tuning
        :return:
        """
        tf.compat.v1.set_random_seed(10)
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.compat.v1.reset_default_graph()

        # load the meta file
        meta_path = os.path.join(mnist_model_path, 'mnist_save.meta')
        sess = aimet_tensorflow.utils.graph_saver.load_model_from_meta(meta_path)

        # ignore first Conv2D op
        conv2d = sess.graph.get_operation_by_name('conv1/Conv2D')
        modules_to_ignore = [conv2d]

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.5),
                                                                    num_comp_ratio_candidates=10,
                                                                    use_monotonic_fit=True,
                                                                    saved_eval_scores_dict=None)

        auto_params = aimet_tensorflow.defs.SpatialSvdParameters.AutoModeParams(greedy_select_params=greedy_params,
                                                                                modules_to_ignore=modules_to_ignore)

        params = aimet_tensorflow.defs.SpatialSvdParameters(input_op_names=['reshape_input'],
                                                            output_op_names=['dense_1/BiasAdd'],
                                                            mode=aimet_tensorflow.defs.SpatialSvdParameters.Mode.auto,
                                                            params=auto_params, multiplicity=8)
        input_shape = (1, 1, 28, 28)

        compr_model_sess, stats = ModelCompressor.compress_model(sess=sess,
                                                                 working_dir=None,
                                                                 eval_callback=evaluate, eval_iterations=5,
                                                                 input_shape=input_shape,
                                                                 compress_scheme=aimet_common.defs.CompressionScheme.
                                                                 spatial_svd,
                                                                 cost_metric=aimet_common.defs.CostMetric.mac,
                                                                 parameters=params)

        print(stats)

        self.assertEqual(evaluate(compr_model_sess, 1, True), float(stats.compressed_model_accuracy))

        all_ops = compr_model_sess.graph.get_operations()

        conv_ops = [op for op in all_ops if op.type == 'Conv2D']

        self.assertEqual(len(conv_ops), 4)
        self.assertTrue(math.isclose(float(stats.mac_compression_ratio), 0.5, abs_tol=0.1))

        # get the weights after fine tuning

        conv2d_1_a_op = compr_model_sess.graph.get_operation_by_name('conv2_a/Conv2D')
        conv2d_1_a_op_weights_before = conv2d_1_a_op.inputs[1].eval(session=compr_model_sess)

        # fine tune the model

        # get the input and validation place holders
        x = compr_model_sess.graph.get_tensor_by_name('reshape_input:0')
        y = compr_model_sess.graph.get_tensor_by_name('labels:0')
        cross_entropy = compr_model_sess.graph.get_tensor_by_name('xent:0')

        with compr_model_sess.graph.as_default():

            # new optimizer and back propagation Op
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3, name='Adam_new')
            train_step = optimizer.minimize(loss=cross_entropy, name='train_step_new')

            # initialize only uninitialized variables
            # only needed when fine tuning, because we are adding new optimizer
            graph_eval.initialize_uninitialized_vars(compr_model_sess)

        input_data = np.random.rand(32, 784)
        labels = np.random.randint(low=2, size=(32, 10))
        for i in range(1):
            _, loss_val = compr_model_sess.run([train_step, cross_entropy], feed_dict={x: input_data, y: labels})

        # get the weights after fine tuning

        conv2d_1_a_op = compr_model_sess.graph.get_operation_by_name('conv2_a/Conv2D')
        conv2d_1_a_op_weights_after = conv2d_1_a_op.inputs[1].eval(session=compr_model_sess)

        # weight should be different after one iteration
        self.assertFalse(np.allclose(conv2d_1_a_op_weights_before, conv2d_1_a_op_weights_after))

        # close original session
        sess.close()
        # close compressed model session
        compr_model_sess.close()

        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))

    def test_spatial_svd_compress_manual(self):
        """
        End to end manual mode spatial SVD using Resnet50 Keras model
        :return:
        """
        np.random.seed(1)
        AimetLogger.set_level_for_all_areas(logging.INFO)
        tf.compat.v1.reset_default_graph()
        # Grow GPU memory as needed at the cost of fragmentation.
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        model = ResNet50(weights=None, input_shape=(224, 224, 3))
        _ = update_keras_bn_ops_trainable_flag(model, False, "./t")
        sess = tf.compat.v1.keras.backend.get_session()

        output_op_names = ['probs/Softmax']
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            output_op_names = ['predictions/Softmax']

        with sess.graph.as_default():

            # predicted value of the model
            y_hat = sess.graph.get_tensor_by_name(output_op_names[0]+':0')
            # place holder for the labels
            y = tf.compat.v1.placeholder(tf.int64, shape=[None, 1000], name='labels')
            # prediction Op
            correct_prediction = tf.equal(tf.argmax(y_hat, axis=1), tf.argmax(y, axis=1))
            # accuracy Op
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        writer = tf.compat.v1.summary.FileWriter('./', sess.graph)
        # make sure the learning_phase flag is False (inference mode)
        learning_phase = sess.graph.get_tensor_by_name('keras_learning_phase/input:0')
        self.assertFalse(sess.run(learning_phase))

        input_shape = (1, 224, 224, 3)

        conv2_block1_2_conv = sess.graph.get_operation_by_name('conv2_block1_2_conv/Conv2D')
        conv3_block1_2_conv = sess.graph.get_operation_by_name('conv3_block1_2_conv/Conv2D')

        list_of_module_comp_ratio_pairs = [ModuleCompRatioPair(conv2_block1_2_conv, 0.5),
                                           ModuleCompRatioPair(conv3_block1_2_conv, 0.4)
                                           ]

        manual_params = aimet_tensorflow.defs.SpatialSvdParameters.ManualModeParams(list_of_module_comp_ratio_pairs=
                                                                                    list_of_module_comp_ratio_pairs)

        output_op_names = ['probs/Softmax']
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            output_op_names = ['predictions/Softmax']
        params = aimet_tensorflow.defs.SpatialSvdParameters(input_op_names=['input_1'],
                                                            output_op_names=output_op_names,
                                                            mode=aimet_tensorflow.defs.SpatialSvdParameters.Mode.manual,
                                                            params=manual_params, multiplicity=8)

        mocked_eval = unittest.mock.MagicMock()
        mocked_eval.side_effect = [87, 64]

        results = ModelCompressor.compress_model(sess=sess, working_dir=None, eval_callback=mocked_eval,
                                                 eval_iterations=5, input_shape=input_shape,
                                                 compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 trainer=None)

        compr_model_sess, stats = results
        print(stats)

        # split ops for res2a_branch2b_Conv2D op
        conv2_block1_2_conv_a = compr_model_sess.graph.get_operation_by_name('conv2_block1_2_conv_a/Conv2D')
        conv2_block1_2_conv_b = compr_model_sess.graph.get_operation_by_name('conv2_block1_2_conv_b/Conv2D')

        # split ops for res3a_branch2b_Conv2D op
        conv3_block1_2_conv_a = compr_model_sess.graph.get_operation_by_name('conv3_block1_2_conv_a/Conv2D')
        conv3_block1_2_conv_b = compr_model_sess.graph.get_operation_by_name('conv3_block1_2_conv_b/Conv2D')

        # res2a_branch2b_Conv2D
        self.assertEqual(compr_model_sess.run(conv2_block1_2_conv_a.inputs[1]).shape, (3, 1, 64, 48))
        self.assertEqual(compr_model_sess.run(conv2_block1_2_conv_b.inputs[1]).shape, (1, 3, 48, 64))

        self.assertEqual(compr_model_sess.run(conv3_block1_2_conv_a.inputs[1]).shape, (3, 1, 128, 80))
        self.assertEqual(compr_model_sess.run(conv3_block1_2_conv_b.inputs[1]).shape, (1, 3, 80, 128))

        # forward pass to the model with random data and labels
        input_data = np.random.rand(32, 224, 224, 3)
        labels = np.random.randint(low=2, size=(32, 1000))
        accuracy_tensor = compr_model_sess.graph.get_tensor_by_name('accuracy:0')
        input_tensor = compr_model_sess.graph.get_tensor_by_name('input_1:0')
        label_tensor = compr_model_sess.graph.get_tensor_by_name('labels:0')

        # make sure the learning_phase flag is False (inference mode)
        learning_phase = compr_model_sess.graph.get_tensor_by_name('keras_learning_phase/input:0')
        self.assertFalse(compr_model_sess.run(learning_phase))

        compr_model_sess.run(accuracy_tensor, feed_dict={input_tensor: input_data, label_tensor: labels})

        # close original session
        sess.close()
        # close compressed model session
        compr_model_sess.close()

        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))

    def test_spatial_svd_compress_auto(self):
        """
        End to end auto mode spatial SVD using Resnet50 Keras model
        :return:
        """
        np.random.seed(1)
        AimetLogger.set_level_for_all_areas(logging.INFO)
        tf.compat.v1.reset_default_graph()
        # Grow GPU memory as needed at the cost of fragmentation.
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        model = ResNet50(weights=None, input_shape=(224, 224, 3))
        _ = update_keras_bn_ops_trainable_flag(model, False, "./t")
        sess = tf.compat.v1.keras.backend.get_session()

        init = tf.compat.v1.global_variables_initializer()

        all_ops = sess.graph.get_operations()
        for op in all_ops:
            print(op.name)

        output_op_names = ['probs/Softmax']
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            output_op_names = ['predictions/Softmax']

        with sess.graph.as_default():

            # predicted value of the model
            y_hat = sess.graph.get_tensor_by_name(output_op_names[0]+":0")
            # place holder for the labels
            y = tf.compat.v1.placeholder(tf.int64, shape=[None, 1000], name='labels')
            # prediction Op
            correct_prediction = tf.equal(tf.argmax(y_hat, axis=1), tf.argmax(y, axis=1))
            # accuracy Op
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        sess.run(init)

        # make sure the learning_phase flag is False (inference mode)
        learning_phase = sess.graph.get_tensor_by_name('keras_learning_phase/input:0')
        self.assertFalse(sess.run(learning_phase))

        input_shape = (1, 224, 224, 3)

        # compressing only two layers
        modules_to_ignore = list()
        all_ops = sess.graph.get_operations()
        for op in all_ops:
            if op.type == 'Conv2D':
                modules_to_ignore.append(op)

        del modules_to_ignore[5:7]

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.7),
                                                                    num_comp_ratio_candidates=3,
                                                                    use_monotonic_fit=True,
                                                                    saved_eval_scores_dict=None)

        auto_params = aimet_tensorflow.defs.SpatialSvdParameters.AutoModeParams(greedy_select_params=greedy_params,
                                                                                modules_to_ignore=modules_to_ignore)

        params = aimet_tensorflow.defs.SpatialSvdParameters(input_op_names=['input_1'],
                                                            output_op_names=output_op_names,
                                                            mode=aimet_tensorflow.defs.SpatialSvdParameters.Mode.auto,
                                                            params=auto_params,
                                                            multiplicity=8)

        mocked_eval = unittest.mock.MagicMock()
        mocked_eval.side_effect = [50, 80,
                                   30, 76,
                                   87, 64]

        results = ModelCompressor.compress_model(sess=sess, working_dir=None, eval_callback=mocked_eval,
                                                 eval_iterations=5, input_shape=input_shape,
                                                 compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 trainer=None)

        compr_model_sess, stats = results
        print(stats)

        # forward pass to the model with random data and labels
        input_data = np.random.rand(32, 224, 224, 3)
        labels = np.random.randint(low=2, size=(32, 1000))
        accuracy_tensor = compr_model_sess.graph.get_tensor_by_name('accuracy:0')
        input_tensor = compr_model_sess.graph.get_tensor_by_name('input_1:0')
        label_tensor = compr_model_sess.graph.get_tensor_by_name('labels:0')

        # make sure the learning_phase flag is False (inference mode)
        learning_phase = compr_model_sess.graph.get_tensor_by_name('keras_learning_phase/input:0')
        self.assertFalse(compr_model_sess.run(learning_phase))

        compr_model_sess.run(accuracy_tensor, feed_dict={input_tensor: input_data, label_tensor: labels})

        # close original session
        sess.close()
        # close compressed model session
        compr_model_sess.close()

        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))


class ChannelPruningAcceptanceTests(unittest.TestCase):


    def test_channel_pruning_manual_vgg16_keras(self):
        """
        :return:
        """
        AimetLogger.set_level_for_all_areas(logging.INFO)
        tf.compat.v1.reset_default_graph()
        batch_size = 1
        input_data = np.random.rand(100, 224, 224, 3)
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.batch(batch_size=batch_size)

        graph = tf.Graph()

        with graph.as_default():

            _ = VGG16(weights=None, input_shape=(224, 224, 3))
            init = tf.compat.v1.global_variables_initializer()

        # Grow GPU memory as needed at the cost of fragmentation.
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member

        sess = tf.compat.v1.Session(graph=graph, config=config)

        with sess.graph.as_default():

            # predicted value of the model
            y_hat = sess.graph.get_tensor_by_name('predictions/Softmax:0')
            # place holder for the labels
            y = tf.compat.v1.placeholder(tf.int64, shape=[None, 1000], name='labels')
            # prediction Op
            correct_prediction = tf.equal(tf.argmax(y_hat, axis=1), tf.argmax(y, axis=1))
            # accuracy Op
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        sess.run(init)

        block1_conv1_op = sess.graph.get_operation_by_name('block1_conv1/Conv2D')
        block1_conv2_op = sess.graph.get_operation_by_name('block1_conv2/Conv2D')
        block2_conv1_op = sess.graph.get_operation_by_name('block2_conv1/Conv2D')
        block2_conv2_op = sess.graph.get_operation_by_name('block2_conv2/Conv2D')

        list_of_module_comp_ratio_pairs = [ModuleCompRatioPair(block1_conv2_op, 0.5),
                                           ModuleCompRatioPair(block2_conv2_op, 0.5)]

        # input op name in VGG-16 keras model is 'input_1'
        input_op_names = ['input_1']
        output_op_names = ['predictions/Softmax']

        manual_params = aimet_tensorflow.defs.ChannelPruningParameters.ManualModeParams(list_of_module_comp_ratio_pairs=
                                                                                       list_of_module_comp_ratio_pairs)

        params = aimet_tensorflow.defs.ChannelPruningParameters(input_op_names=input_op_names,
                                                                output_op_names=output_op_names, data_set=dataset,
                                                                batch_size=32, num_reconstruction_samples=50,
                                                                allow_custom_downsample_ops=True,
                                                                mode=aimet_tensorflow.defs.ChannelPruningParameters.Mode.
                                                                manual,
                                                                params=manual_params, multiplicity=8)
        # channels_last data format
        input_shape = (32, 224, 224, 3)

        # mocke eval function, baseline - 87, compressed - 64
        mocked_eval = unittest.mock.MagicMock()
        mocked_eval.side_effect = [87, 64]

        compr_model_sess, stats = ModelCompressor.compress_model(sess=sess,
                                                                 working_dir=None,
                                                                 eval_callback=mocked_eval,
                                                                 eval_iterations=1, input_shape=input_shape,
                                                                 compress_scheme=aimet_common.defs.CompressionScheme.
                                                                 channel_pruning,
                                                                 cost_metric=aimet_common.defs.CostMetric.mac,
                                                                 parameters=params)

        print(stats)

        red_block1_conv1_op = compr_model_sess.graph.get_operation_by_name('reduced_block1_conv1/Conv2D')
        red_block1_conv2_op = compr_model_sess.graph.get_operation_by_name('reduced_block1_conv2/Conv2D')
        red_block2_conv1_op = compr_model_sess.graph.get_operation_by_name('reduced_block2_conv1/Conv2D')
        red_block2_conv2_op = compr_model_sess.graph.get_operation_by_name('reduced_block2_conv2/Conv2D')

        # conv2_2 after and before compression weight shapes
        self.assertEqual(compr_model_sess.run(red_block2_conv2_op.inputs[1]).shape, (3, 3, 64, 128))
        self.assertEqual(sess.run(block2_conv2_op.inputs[1]).shape, (3, 3, 128, 128))

        # conv2_1 after and before compression weight shapes
        self.assertEqual(compr_model_sess.run(red_block2_conv1_op.inputs[1]).shape, (3, 3, 64, 64))
        self.assertEqual(sess.run(block2_conv1_op.inputs[1]).shape, (3, 3, 64, 128))

        # conv1_2 after and before compression weight shapes
        self.assertEqual(compr_model_sess.run(red_block1_conv2_op.inputs[1]).shape, (3, 3, 32, 64))
        self.assertEqual(sess.run(block1_conv2_op.inputs[1]).shape, (3, 3, 64, 64))

        # conv1_1 after and before compression weight shapes
        self.assertEqual(compr_model_sess.run(red_block1_conv1_op.inputs[1]).shape, (3, 3, 3, 32))
        self.assertEqual(sess.run(block1_conv1_op.inputs[1]).shape, (3, 3, 3, 64))

        # forward pass to the model with random data and labels
        input_data = np.random.rand(32, 224, 224, 3)
        labels = np.random.randint(low=2, size=(32, 1000))
        accuracy_tensor = compr_model_sess.graph.get_tensor_by_name('accuracy:0')
        input_tensor = compr_model_sess.graph.get_tensor_by_name('input_1:0')
        label_tensor = compr_model_sess.graph.get_tensor_by_name('labels:0')

        compr_model_sess.run(accuracy_tensor, feed_dict={input_tensor: input_data, label_tensor: labels})
        # close original session
        sess.close()
        # close compressed model session
        compr_model_sess.close()

        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))

    def test_channel_pruning_auto_mobilenetv1(self):
        """
        Auto mode test fot MobileNetv1 model
        :return:
        """
        AimetLogger.set_level_for_all_areas(logging.INFO)
        tf.compat.v1.reset_default_graph()
        # Grow GPU memory as needed at the cost of fragmentation.
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        model = MobileNet(weights=None, input_shape=(224, 224, 3))
        _ = update_keras_bn_ops_trainable_flag(model, False, "./t")
        sess = tf.compat.v1.keras.backend.get_session()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        # Grow GPU memory as needed at the cost of fragmentation.
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member

        with sess.graph.as_default():

            # predicted value of the model
            y_hat = sess.graph.get_tensor_by_name('reshape_2/Reshape:0')
            # place holder for the labels
            y = tf.compat.v1.placeholder(tf.int64, shape=[None, 1000], name='labels')
            # prediction Op
            correct_prediction = tf.equal(tf.argmax(y_hat, axis=1), tf.argmax(y, axis=1))
            # accuracy Op
            _ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        sess.run(init)

        modules_to_ignore = []
        all_ops = sess.graph.get_operations()
        for op in all_ops:
            if op.type == 'Conv2D':
                modules_to_ignore.append(op)

        del modules_to_ignore[2:4]

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.92),
                                                                    num_comp_ratio_candidates=4,
                                                                    use_monotonic_fit=True,
                                                                    saved_eval_scores_dict=None)

        auto_params = aimet_tensorflow.defs.ChannelPruningParameters.AutoModeParams(greedy_select_params=greedy_params,
                                                                                    modules_to_ignore=modules_to_ignore)

        input_op_names = ['input_1']
        output_op_names = ['reshape_2/Reshape']
        dataset = np.random.rand(1, 1, 224, 224, 3)
        dataset = tf.convert_to_tensor(dataset)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

        params = aimet_tensorflow.defs.ChannelPruningParameters(input_op_names=input_op_names,
                                                                output_op_names=output_op_names, data_set=dataset,
                                                                batch_size=32, num_reconstruction_samples=50,
                                                                allow_custom_downsample_ops=True,
                                                                mode=aimet_tensorflow.defs.ChannelPruningParameters.
                                                                Mode.auto,
                                                                params=auto_params, multiplicity=8)
        # channels_last data format
        input_shape = (32, 224, 224, 3)

        mocked_eval = unittest.mock.MagicMock()
        mocked_eval.side_effect = [70.123, 70.124, 70.125,
                                   70.123, 70.124, 70.125,
                                   70.123, 70.124, 70.125]

        compr_model_sess, stats = ModelCompressor.compress_model(sess=sess,
                                                                 working_dir=None,
                                                                 eval_callback=mocked_eval,
                                                                 eval_iterations=1, input_shape=input_shape,
                                                                 compress_scheme=aimet_common.defs.CompressionScheme.
                                                                 channel_pruning,
                                                                 cost_metric=aimet_common.defs.CostMetric.mac,
                                                                 parameters=params)

        print(stats)
        conv_pw_1_op = sess.graph.get_operation_by_name('conv_pw_1/Conv2D')
        conv_pw_2_op = sess.graph.get_operation_by_name('conv_pw_2/Conv2D')
        conv_pw_3_op = sess.graph.get_operation_by_name('conv_pw_3/Conv2D')
        reduced_conv_pw_1_op = compr_model_sess.graph.get_operation_by_name('reduced_conv_pw_1/Conv2D')
        reduced_conv_pw_2_op = compr_model_sess.graph.get_operation_by_name('reduced_reduced_conv_pw_2/Conv2D')
        reduced_conv_pw_3_op = compr_model_sess.graph.get_operation_by_name('reduced_conv_pw_3/Conv2D')

        # conv_pw_1 after and before compression weight shapes
        self.assertEqual(compr_model_sess.run(reduced_conv_pw_1_op.inputs[1]).shape, (1, 1, 32, 32))
        self.assertEqual(sess.run(conv_pw_1_op.inputs[1]).shape, (1, 1, 32, 64))

        # conv_pw_2 after and before compression weight shapes
        # TODO: uncomment below check after dangling winnowed nodes in ConnectedGraph are removed
        self.assertEqual(compr_model_sess.run(reduced_conv_pw_2_op.inputs[1]).shape, (1, 1, 32, 64))
        self.assertEqual(sess.run(conv_pw_2_op.inputs[1]).shape, (1, 1, 64, 128))

        # conv_pw_3 after and before compression weight shapes
        self.assertEqual(compr_model_sess.run(reduced_conv_pw_3_op.inputs[1]).shape, (1, 1, 64, 128))
        self.assertEqual(sess.run(conv_pw_3_op.inputs[1]).shape, (1, 1, 128, 128))

        # forward pass to the model with random data and labels
        input_data = np.random.rand(32, 224, 224, 3)
        labels = np.random.randint(low=2, size=(32, 1000))
        accuracy_tensor = compr_model_sess.graph.get_tensor_by_name('accuracy:0')
        input_tensor = compr_model_sess.graph.get_tensor_by_name('input_1:0')
        label_tensor = compr_model_sess.graph.get_tensor_by_name('labels:0')

        compr_model_sess.run(accuracy_tensor, feed_dict={input_tensor: input_data, label_tensor: labels})
        # close original session
        sess.close()
        # close compressed model session
        compr_model_sess.close()

        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))

    def test_channel_pruning_manual_resnet50_keras(self):
        """
        Manual mode test for Keras Resnet50
        """
        AimetLogger.set_level_for_all_areas(logging.INFO)
        tf.compat.v1.reset_default_graph()
        # Grow GPU memory as needed at the cost of fragmentation.
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member

        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        model = ResNet50(weights=None, input_shape=(224, 224, 3))
        _ = update_keras_bn_ops_trainable_flag(model, False, "./t")
        sess = tf.compat.v1.keras.backend.get_session()
        init = tf.compat.v1.global_variables_initializer()

        output_op_names = ['probs/Softmax']
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            output_op_names = ['predictions/Softmax']

        # predicted value of the model
        y_hat = sess.graph.get_tensor_by_name(output_op_names[0]+":0")

        with sess.graph.as_default():
            # place holder for the labels
            y = tf.compat.v1.placeholder(tf.int64, shape=[None, 1000], name='labels')
            # prediction Op
            correct_prediction = tf.equal(tf.argmax(y_hat, axis=1), tf.argmax(y, axis=1))
            # accuracy Op
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # initialize all the variables
        sess.run(init)

        batch_size = 32
        input_data = np.random.rand(100, 224, 224, 3)
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.batch(batch_size=batch_size)

        conv2_block1_1_conv = sess.graph.get_operation_by_name('conv2_block1_1_conv/Conv2D')
        conv2_block1_2_conv = sess.graph.get_operation_by_name('conv2_block1_2_conv/Conv2D')

        conv3_block1_1_conv = sess.graph.get_operation_by_name('conv3_block1_1_conv/Conv2D')
        conv3_block1_2_conv = sess.graph.get_operation_by_name('conv3_block1_2_conv/Conv2D')

        list_of_module_comp_ratio_pairs = [ModuleCompRatioPair(conv2_block1_2_conv, 0.5),
                                           ModuleCompRatioPair(conv3_block1_2_conv, 0.5)]

        input_op_names = ['input_1']

        manual_params = aimet_tensorflow.defs.ChannelPruningParameters.ManualModeParams(list_of_module_comp_ratio_pairs=
                                                                                        list_of_module_comp_ratio_pairs)

        params = aimet_tensorflow.defs.ChannelPruningParameters(input_op_names=input_op_names,
                                                                output_op_names=output_op_names,
                                                                data_set=dataset,
                                                                batch_size=32, num_reconstruction_samples=50,
                                                                allow_custom_downsample_ops=True,
                                                                mode=aimet_tensorflow.defs.ChannelPruningParameters.Mode.
                                                                manual,
                                                                params=manual_params, multiplicity=8)
        # channels_last data format
        input_shape = (32, 224, 224, 3)

        mocked_eval = unittest.mock.MagicMock()
        mocked_eval.side_effect = [70.123, 70.124]

        compr_model_sess, stats = ModelCompressor.compress_model(sess=sess,
                                                                 working_dir=None,
                                                                 eval_callback=mocked_eval,
                                                                 eval_iterations=1, input_shape=input_shape,
                                                                 compress_scheme=aimet_common.defs.CompressionScheme.
                                                                 channel_pruning,
                                                                 cost_metric=aimet_common.defs.CostMetric.mac,
                                                                 parameters=params)

        print(stats)
        all_ops = compr_model_sess.graph.get_operations()
        for op in all_ops:
            if op.type == 'Conv2D':
                print(op.name)

        # conv2_block1_2_conv
        reduced_conv2_block1_1_conv = compr_model_sess.graph.get_operation_by_name('reduced_conv2_block1_1_conv/Conv2D')
        reduced_conv2_block1_2_conv = compr_model_sess.graph.get_operation_by_name('reduced_conv2_block1_2_conv/Conv2D')

        # conv3_block1_2_conv
        reduced_conv3_block1_1_conv = compr_model_sess.graph.get_operation_by_name('reduced_conv3_block1_1_conv/Conv2D')
        reduced_conv3_block1_2_conv = compr_model_sess.graph.get_operation_by_name('reduced_conv3_block1_2_conv/Conv2D')

        # conv2_block1_1_conv after and before compression weight shapes
        self.assertEqual(compr_model_sess.run(reduced_conv2_block1_1_conv.inputs[1]).shape, (1, 1, 64, 32))
        self.assertEqual(sess.run(conv2_block1_1_conv.inputs[1]).shape, (1, 1, 64, 64))

        # conv2_block1_1_conv after and before compression weight shapes
        self.assertEqual(compr_model_sess.run(reduced_conv2_block1_2_conv.inputs[1]).shape, (3, 3, 32, 64))
        self.assertEqual(sess.run(conv2_block1_2_conv.inputs[1]).shape, (3, 3, 64, 64))

        # conv3_block1_1_conv after and before compression weight shapes
        self.assertEqual(compr_model_sess.run(reduced_conv3_block1_1_conv.inputs[1]).shape, (1, 1, 256, 64))
        self.assertEqual(sess.run(conv3_block1_1_conv.inputs[1]).shape, (1, 1, 256, 128))

        # conv3_block1_2_conv after and before compression weight shapes
        self.assertEqual(compr_model_sess.run(reduced_conv3_block1_2_conv.inputs[1]).shape, (3, 3, 64, 128))
        self.assertEqual(sess.run(conv3_block1_2_conv.inputs[1]).shape, (3, 3, 128, 128))

        # Test forward pass through the model with random data and labels
        new_input_data = np.random.rand(32, 224, 224, 3)
        labels = np.random.randint(low=2, size=(32, 1000))
        accuracy_tensor = compr_model_sess.graph.get_tensor_by_name('accuracy:0')
        input_tensor = compr_model_sess.graph.get_tensor_by_name('input_1:0')
        label_tensor = compr_model_sess.graph.get_tensor_by_name('labels:0')
        compr_model_sess.run(accuracy_tensor, feed_dict={input_tensor: new_input_data, label_tensor: labels})

        # close original session
        sess.close()

        # close compressed model session
        compr_model_sess.close()

        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))

    def test_channel_pruning_auto_resnet50(self):
        """
        Auto mode test for Keras ResNet-50.
        """
        AimetLogger.set_level_for_all_areas(logging.INFO)
        tf.compat.v1.reset_default_graph()
        # Grow GPU memory as needed at the cost of fragmentation.
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        model = ResNet50(weights=None, input_shape=(224, 224, 3))
        _ = update_keras_bn_ops_trainable_flag(model, False, "./t")
        sess = tf.compat.v1.keras.backend.get_session()

        init = tf.compat.v1.global_variables_initializer()

        output_op_names = ['probs/Softmax']
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            output_op_names = ['predictions/Softmax']

        # predicted value of the model
        y_hat = sess.graph.get_tensor_by_name(output_op_names[0]+":0")

        with sess.graph.as_default():
            # place holder for the labels
            y = tf.compat.v1.placeholder(tf.int64, shape=[None, 1000], name='labels')
            # prediction Op
            correct_prediction = tf.equal(tf.argmax(y_hat, axis=1), tf.argmax(y, axis=1))
            # accuracy Op
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # initialize all the variables
        sess.run(init)

        # create dataset using random numpy array
        batch_size = 32
        input_data = np.random.rand(100, 224, 224, 3)
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.batch(batch_size=batch_size)

        # compressing only two Conv2D layers
        modules_to_ignore = list()
        all_ops = sess.graph.get_operations()
        for op in all_ops:
            if op.type == 'Conv2D':
                modules_to_ignore.append(op)

        del modules_to_ignore[5:7]

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.9),
                                                                    num_comp_ratio_candidates=4,
                                                                    use_monotonic_fit=False)

        auto_params = aimet_tensorflow.defs.ChannelPruningParameters.AutoModeParams(greedy_select_params=greedy_params,
                                                                                    modules_to_ignore=modules_to_ignore)

        input_op_names = ['input_1']
        params = aimet_tensorflow.defs.ChannelPruningParameters(input_op_names=input_op_names,
                                                                output_op_names=output_op_names, data_set=dataset,
                                                                batch_size=32, num_reconstruction_samples=50,
                                                                allow_custom_downsample_ops=True,
                                                                mode=aimet_tensorflow.defs.ChannelPruningParameters.Mode.
                                                                auto,
                                                                params=auto_params, multiplicity=1)
        # channels_last data format
        input_shape = (32, 224, 224, 3)

        mocked_eval = unittest.mock.MagicMock()
        mocked_eval.side_effect = [70.123, 70.124, 70.125,
                                   87, 64, 99,
                                   22, 33]

        compr_model_sess, stats = ModelCompressor.compress_model(sess=sess,
                                                                 working_dir=None,
                                                                 eval_callback=mocked_eval,
                                                                 eval_iterations=1, input_shape=input_shape,
                                                                 compress_scheme=aimet_common.defs.CompressionScheme.
                                                                 channel_pruning,
                                                                 cost_metric=aimet_common.defs.CostMetric.mac,
                                                                 parameters=params)

        print(stats)

        # updated conv2_block2_1_conv
        updated_conv2_block2_1_conv = compr_model_sess.graph.get_operation_by_name('reduced_reduced_conv2_block2_1_conv/Conv2D')

        # updated conv2_block2_2_conv
        updated_conv2_block2_2_conv = compr_model_sess.graph.get_operation_by_name('reduced_conv2_block2_2_conv/Conv2D')

        # conv2_block2_1_conv after compression weight shapes
        self.assertEqual(compr_model_sess.run(updated_conv2_block2_1_conv.inputs[1]).shape, (1, 1, 64, 16))

        # conv2_block2_2_conv after compression weight shapes
        self.assertEqual(compr_model_sess.run(updated_conv2_block2_2_conv.inputs[1]).shape, (3, 3, 16, 64))

        # Test forward pass through the model with random data and labels
        new_input_data = np.random.rand(32, 224, 224, 3)
        labels = np.random.randint(low=2, size=(32, 1000))
        accuracy_tensor = compr_model_sess.graph.get_tensor_by_name('accuracy:0')
        input_tensor = compr_model_sess.graph.get_tensor_by_name('input_1:0')
        label_tensor = compr_model_sess.graph.get_tensor_by_name('labels:0')
        compr_model_sess.run(accuracy_tensor, feed_dict={input_tensor: new_input_data, label_tensor: labels})

        # close original session
        sess.close()

        # close compressed model session
        compr_model_sess.close()

        # delete temp directory
        shutil.rmtree(str('./temp_meta/'))


class SvdAndChannelPruningAcceptanceTests(unittest.TestCase):


    def test_svd_followed_by_channel_pruning(self):
        """ Test that a model can be run through spatial svd and then channel pruning """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            _ = model_with_three_convs()
            init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        conv2d_1 = sess.graph.get_operation_by_name('conv2d_1/Conv2D')

        list_of_module_comp_ratio_pairs = [ModuleCompRatioPair(conv2d_1, 0.5)]

        manual_params = aimet_tensorflow.defs.SpatialSvdParameters.ManualModeParams(list_of_module_comp_ratio_pairs=
                                                                                    list_of_module_comp_ratio_pairs)

        input_op_names = ['input_1']
        output_op_names = ['three_convs/Softmax']
        params = aimet_tensorflow.defs.SpatialSvdParameters(input_op_names=input_op_names,
                                                            output_op_names=output_op_names,
                                                            mode=aimet_tensorflow.defs.SpatialSvdParameters.Mode.manual,
                                                            params=manual_params, multiplicity=1)

        mocked_eval = unittest.mock.MagicMock()
        mocked_eval.side_effect = [87, 87]

        input_shape = (1, 8, 8, 3)
        sess, _ = ModelCompressor.compress_model(sess=sess, working_dir=None, eval_callback=mocked_eval,
                                                 eval_iterations=5, input_shape=input_shape,
                                                 compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
                                                 cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                                 trainer=None)

        # Check that svd added these ops
        _ = sess.graph.get_operation_by_name('conv2d_1_a/Conv2D')
        _ = sess.graph.get_operation_by_name('conv2d_1_b/Conv2D')

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.5),
                                                                    num_comp_ratio_candidates=4,
                                                                    use_monotonic_fit=True,
                                                                    saved_eval_scores_dict=None)

        conv_to_ignore = sess.graph.get_operation_by_name('conv2d/Conv2D')
        auto_params = aimet_tensorflow.defs.ChannelPruningParameters.AutoModeParams(greedy_select_params=greedy_params,
                                                                                    modules_to_ignore=[conv_to_ignore])

        dataset = np.random.rand(1, 1, 8, 8, 3)
        dataset = tf.convert_to_tensor(dataset)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

        params = aimet_tensorflow.defs.ChannelPruningParameters(input_op_names=input_op_names,
                                                                output_op_names=output_op_names, data_set=dataset,
                                                                batch_size=32, num_reconstruction_samples=50,
                                                                allow_custom_downsample_ops=False,
                                                                mode=aimet_tensorflow.defs.ChannelPruningParameters.
                                                                Mode.auto,
                                                                params=auto_params, multiplicity=1)

        mocked_eval = unittest.mock.MagicMock()
        mocked_eval.side_effect = [0, .5, 1,
                                   0, .5, 1,
                                   0, .5, 1,
                                   0, 1]

        sess, _ = ModelCompressor.compress_model(sess=sess,
                                                 working_dir=None,
                                                 eval_callback=mocked_eval,
                                                 eval_iterations=1, input_shape=input_shape,
                                                 compress_scheme=aimet_common.defs.CompressionScheme.
                                                 channel_pruning,
                                                 cost_metric=aimet_common.defs.CostMetric.mac,
                                                 parameters=params)

        # Check that these ops were added by cp
        _ = sess.graph.get_operation_by_name('reduced_reduced_conv2d_1_a/Conv2D')
        _ = sess.graph.get_operation_by_name('reduced_reduced_conv2d_1_b/Conv2D')
        _ = sess.graph.get_operation_by_name('reduced_conv2d_2/Conv2D')
        sess.close()

