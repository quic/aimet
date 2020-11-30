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
""" utilities for Tf graph save and load"""

import datetime
import os
import shutil

import tensorflow as tf

from aimet_common.defs import EvalFunction


def save_model_to_meta(model: tf.compat.v1.Session, meta_path: str):
    """
    Utility function to save a graph
    :param model: tf.compat.v1.Session
    :param meta_path: path to meta file
    :return:
    """

    with model.graph.as_default():
        saver = tf.compat.v1.train.Saver()

    saver.save(sess=model, save_path=meta_path)


def load_model_from_meta(meta_path, checkpoint_path=None) -> tf.compat.v1.Session:
    """
    Utility function to load graph from meta file
    :param meta_path: path to meta file
    :return: tf.compat.v1.Session
    """

    if not checkpoint_path:
        checkpoint_path = meta_path.split(".meta")[0]

    # Grow GPU memory as needed at the cost of fragmentation.

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=no-member

    sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)

    with sess.graph.as_default():
        # open the graph and restore the parameters
        saver = tf.compat.v1.train.import_meta_graph(meta_path)

    saver.restore(sess, checkpoint_path)
    return sess


def save_and_load_graph(meta_path: str, sess: tf.compat.v1.Session) -> tf.compat.v1.Session:
    """
    saves and loads a graph and returns the new session obtained.
    :param meta_path: path to save the file
    :param sess: session to be saved and loaded back
    :return: new sess after load and save
    """

    unique_id = str(datetime.datetime.now()).replace(' ', '_')

    meta_path = meta_path + "_" + unique_id

    if not os.path.exists(meta_path):
        os.mkdir(meta_path)

    save_model_to_meta(model=sess, meta_path=str(meta_path + '/temp'))
    new_sess = load_model_from_meta(meta_path=str(meta_path + '/temp.meta'))

    # delete temp folder created
    shutil.rmtree(meta_path)

    # return new session after load and save
    return new_sess


def get_meta_and_checkpoint_path(working_dir: str) -> str:
    """
    Returns the path to store meta and checkpoint files
    If working_dir is None, creates ./temp_meta/ directory and stores all the
    intermediate meta and checkpoint files

    :param working_dir: working directory
    :return: path to meta file
    """

    if working_dir is not None:

        if not os.path.exists(working_dir):
            raise ValueError("working_dir : {} does not exist".format(working_dir))

        meta_path = working_dir

    # create temporary directory './temp_meta/'
    else:
        meta_path = './temp_meta/'
        if not os.path.exists(meta_path):
            os.mkdir(meta_path)

    return meta_path


def wrapper_func(eval_func: EvalFunction):
    """
    :param eval_func: eval function
    :return: wrapper function
    """
    # 'save_and_reload_tf_sess' is a Wrapper function in which the argument - 'eval_func' is called
    def save_and_reload_tf_sess(*args, **kwargs):

        # convert to list to update arg
        args = list(args)

        # if it's empty or first arg not of type tf.compat.v1.Session, raise error message
        if not args or not isinstance(args[0], tf.compat.v1.Session):
            raise ValueError('First argument to eval function should be Session!')

        # In TF after making changes to the graph you must save and reload, then evaluate
        updated_sess = save_and_load_graph('./saver', args[0])

        # update the argument with new session
        args[args.index(args[0])] = updated_sess

        # convert back to tuple
        args = tuple(args)

        # returning the actual function now inside the wrapper function.
        return eval_func(*args, **kwargs)

    return save_and_reload_tf_sess
