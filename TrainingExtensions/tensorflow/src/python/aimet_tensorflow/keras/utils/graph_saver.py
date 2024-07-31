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

""" utilities for Tf graph save and load"""

import datetime
import os
import shutil
import tensorflow as tf

from aimet_common.defs import EvalFunction


def keras_wrapper_func(eval_func: EvalFunction):
    """
    :param eval_func: eval function

    :return: wrapper function
    """
    # 'save_and_reload_keras_model' is a Wrapper function in which
    # the argument - 'eval_func' is called
    def save_and_reload_keras_model(*args, **kwargs):

        # convert to list to update arg
        args = list(args)

        # if it's empty or first arg not of type tf.keras.Model, raise error message
        if not args or not isinstance(args[0], tf.keras.Model):
            raise ValueError('First argument to eval function should be keras model')

        # In keras after making changes to the graph you must save and reload, then evaluate
        tmp_dir = './data/saved_model'
        model = keras_save_and_load_graph(tmp_dir, args[0])

        # update the argument with new session
        args[args.index(args[0])] = model

        # convert back to tuple
        args = tuple(args)

        # returning the actual function now inside the wrapper function.
        return eval_func(*args, **kwargs)

    return save_and_reload_keras_model


def keras_save_and_load_graph(directory_path: str, model: tf.keras.Model) -> tf.keras.Model:
    """
    saves and loads a keras model and returns the new model obtained.

    :param directory_path: path to save the file
    :param model: model to be saved and loaded back
    :return: new model after load and save
    """

    unique_id = str(datetime.datetime.now()).replace(' ', '_')

    directory_path = directory_path + "_" + unique_id
    os.makedirs(directory_path, exist_ok=True)

    tf.keras.models.save_model(model, directory_path)
    new_model = tf.keras.models.load_model(directory_path)

    # delete temp folder created
    shutil.rmtree(directory_path)

    # return new session after load and save
    return new_model


def keras_remove_hanging_nodes(model: tf.keras.Model) -> tf.keras.Model:
    """
    Removes all the hanging nodes in the keras model.
    This method assumes that all the hanging nodes are present towards the end
    which is the case if the model is loaded from the file.

    :param model: keras model from which hanging nodes needs to be removed.
    :return udpated keras model with hanging nodes removed
    """

    return tf.keras.Model(model.inputs, model.outputs)
