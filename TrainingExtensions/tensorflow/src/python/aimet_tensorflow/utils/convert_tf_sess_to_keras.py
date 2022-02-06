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
""" Utilities to convert TF session to Keras """
# pylint: skip-file
import shutil
from typing import List, Tuple
import tensorflow as tf


def save_tf_session_single_gpu(sess: tf.compat.v1.Session(), path: 'str', input_tensor: 'str', output_tensor: 'str'):
    """
    Saves TF session, meta graph and variables in the provided path

    :param sess: Input: tf.compat.v1.Session
    :param path: Path to save the session
    :param input_tensor: Name of starting op to the given graph
    :param output_tensor: Name of output op of the graph
    :return: None

    """

    # Initilzing the given Tensorflow session
    with sess.graph.as_default():
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

    # Getting the input and output tensors of the graph using provided names
    inputs = sess.graph.get_tensor_by_name(input_tensor)
    train_out = sess.graph.get_tensor_by_name(output_tensor)

    # Saving the input session, meta graph and variables in the provided path
    with sess.graph.as_default():
        train_signature = tf.compat.v1.saved_model.predict_signature_def(inputs={'x': inputs}, outputs={'out': train_out})
        shutil.rmtree(path, ignore_errors=True)
        builder = tf.compat.v1.saved_model.Builder(path)
        builder.add_meta_graph_and_variables(sess, ['serve'], signature_def_map={'train': train_signature})
        builder.save()


def change_name_of_compressed_op(x: str):
    """
    Splits op name and adds kernel:0 to it
    :param x: Name of op
    :return:
    """
    return x.split('/')[0]+'/kernel'+':0'


def load_tf_sess_variables_to_keras_single_gpu(path: 'str', compressed_ops: List['str']) -> tf.compat.v1.keras.Model:
    """
    Creates a Keras model subclass and loads the saved session, meta graph and variables to Keras model

    :param path: Path to load the tf session saved using save_session_graph_and_variables
    :param compressed_ops: List of ops names skipped in Keras model creations. These are the the ops
                            that AIMET compressed and are isolated from rest of the graph.
    :return: Subclassed Keras Model

    """

    to_ignore = map(change_name_of_compressed_op, compressed_ops)

    class Model(tf.compat.v1.keras.Model):
        """ Keras Model subclassing and loading the saved variables"""
        def __init__(self):
            super(Model, self).__init__()
            self.imported = tf.compat.v1.saved_model.load_v2(path)
            self.variables_list = [v for v in self.imported.variables if v.name not in to_ignore]

        def call(self, inputs, training=None):
            """
            Creates a Keras model from the saved object in path
            :param inputs: Input to model
            :param training: If model is to be trained
            :return:
            """
            if training:
                return self.imported.signatures['train'](inputs)
            return self.imported.signatures['serving_default'](input)

    return Model()


def save_as_tf_module_multi_gpu(loading_path: 'str', saving_path: 'str', compressed_ops: List['str'], input_shape: Tuple):
    """
    Loads a Keras model and re-saves the loaded object in the form of tf.Module

    :param loading_path: Path to load the Keras Model
    :param saving_path: Path to save the object
    :param  compressed_ops: List of ops names for which we need to skip in Keras model creation. These are the the
                            ops that AIMET compressed and are isolated from rest of the graph.
    :param input_shape: shape of input to the model
    :return: None

    """

    def trace_model(inputs):
        tf.keras.backend.set_learning_phase(1)
        model = load_tf_sess_variables_to_keras_single_gpu(loading_path, compressed_ops)
        train_out = model(inputs, training=True)
        return train_out

    def export():
        tf.keras.backend.clear_session()
        with tf.compat.v1.keras.backend.get_session() as sess:

            fn = tf.wrap_function(trace_model, signature=[tf.TensorSpec((None, input_shape[0], input_shape[1],
                                                                         input_shape[2]), tf.float32)])
            train_fn = fn.prune(feeds=fn.inputs[0], fetches=fn.outputs[0])
            obj = tf.Module()
            obj.variables_list = list(fn.graph.variables)
            sess.run(tf.compat.v1.global_variables_initializer())
            tf.saved_model.save(obj, saving_path, {'train': train_fn, 'serving_default': train_fn})

    export()


def load_keras_model_multi_gpu(loading_path: 'str', input_shape: List):
    """
    This function loads the Keras model back, which can be used for funetuning within a strategy

    :param loading_path: Path to load the Keras Model
    :param  input_shape: the shape of  stating tensor in graph ; for instance (224,224,3) for ResNet50 and MoblinetV1
    :return: subclassed Keras model
    """

    class Model(tf.compat.v1.keras.Model):
        """ Keras Model subclassing and loading the saved variables """
        def __init__(self):
            super(Model, self).__init__()
            self.imported = tf.compat.v1.saved_model.load_v2(loading_path)
            self.variables_list = self.imported.variables_list

        def call(self, inputs, training=None):
            """
             Creates a Keras model from the saved object in path
            :param inputs: Input to model
            :param training: If training is True or False
            :return:
            """
            if training:
                return self.imported.signatures['train'](inputs)
            return self.imported.signatures['serving_default'](inputs)

    tf.keras.backend.set_learning_phase(1)

    x = tf.keras.Input(shape=tuple(input_shape))
    return tf.compat.v1.keras.Model(x, Model()(x, training=True))
