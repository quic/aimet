# /usr/bin/env python3.5
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
""" Code examples to demonstrate Keras model with AIMET """

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from keras.applications.vgg16 import preprocess_input

import numpy as np

from aimet_common.defs import CompressionScheme, CostMetric
from aimet_tensorflow.defs import SpatialSvdParameters
from aimet_tensorflow.compress import ModelCompressor
from aimet_tensorflow.defs import ModuleCompRatioPair

from aimet_tensorflow.utils.convert_tf_sess_to_keras import save_tf_session_single_gpu, save_as_tf_module_multi_gpu, \
    load_tf_sess_variables_to_keras_single_gpu, load_keras_model_multi_gpu


def train(model):
    """
    Trains using fake dataset
    :param model: Keras model
    :return: trained model
    """
    # Create a fake dataset
    x_train = np.random.rand(32, 224, 224, 3)
    y_train = np.random.rand(32, )
    x_train = preprocess_input(x_train)
    y_train = tf.keras.utils.to_categorical(y_train, 1000)

    model.compile('rmsprop', 'mse')
    model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False)
    return model


def get_sess_from_keras_model():
    """
    Gets TF session from keras model
    :return: TF session
    """
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(1)
    _ = MobileNet(weights=None, input_shape=(224, 224, 3))
    sess = tf.compat.v1.keras.backend.get_session()
    return sess


def compress_session(sess, compressible_ops):
    """
    Compressed TF session
    :param sess: Tf session
    :param compressible_ops: layers to compress
    :return: compressed session
    """
    layer_a = sess.graph.get_operation_by_name(compressible_ops[0])
    list_of_module_comp_ratio_pairs = [ModuleCompRatioPair(layer_a, 0.5)]
    manual_params = SpatialSvdParameters.ManualModeParams(
        list_of_module_comp_ratio_pairs=list_of_module_comp_ratio_pairs)
    params = SpatialSvdParameters(input_op_names=['input_1'], output_op_names=['act_softmax/Softmax'],
                                  mode=SpatialSvdParameters.Mode.manual, params=manual_params)
    scheme = CompressionScheme.spatial_svd
    metric = CostMetric.mac

    # pylint: disable=unused-argument
    def evaluate(sess, iterations, use_cuda):
        return 1

    sess, _ = ModelCompressor.compress_model(sess=sess,
                                             working_dir="./",
                                             eval_callback=evaluate,
                                             eval_iterations=None,
                                             input_shape=(1, 3, 224, 224),
                                             compress_scheme=scheme,
                                             cost_metric=metric,
                                             parameters=params)
    return sess


def convert_tf_session_to_keras_model():
    """
    Convert an AIMET  spatial SVD compressed session to a Keras model and train the Keras model with MirroredStrategy
    """
    sess = get_sess_from_keras_model()

    # For instance, if the first conv layer in MobilNetV1 graph is compressed, then:
    compressed_ops = ['conv1/Conv2D']
    compressed_sess = compress_session(sess, compressed_ops)

    # Defining the input and output convs of the session for MobileNet model
    input_op_name, output_op_name = "input_1:0", "act_softmax/Softmax:0"

    # Step 1: Single Saving the compressed session
    path = './saved_model_single_gpu'
    save_tf_session_single_gpu(compressed_sess, path, input_op_name, output_op_name)
    tf.keras.backend.clear_session()

    # Step 2: Loading the correspnding Keras Model
    tf.keras.backend.set_learning_phase(1)
    model = load_tf_sess_variables_to_keras_single_gpu(path, compressed_ops)

    # Single GPU training of the loaded Keras Model
    train(model)

    # To be able to do multi-gpu training the next two steps needs to be followed:
    # Step 3: Re-Saving the Keras model to make it compatible with distribution strategy
    saving_path = './saved_model_multi_gpu'
    save_as_tf_module_multi_gpu(path, saving_path, compressed_ops, input_shape=(224, 224, 3))

    tf.keras.backend.clear_session()

    with tf.distribute.MirroredStrategy().scope():
        tf.keras.backend.set_learning_phase(1)
        # Step 4: Loading the keras model and  Multi gpu training the model on given dataset
        model = load_keras_model_multi_gpu(saving_path, input_shape=[224, 224, 3])
        # Train model on Multi-GPU
        train(model)
