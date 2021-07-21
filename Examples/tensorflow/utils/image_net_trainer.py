# !/usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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


"""
Creates trainer for Image-Net dataset
"""
import os
import logging
from typing import List
import progressbar
import tensorflow as tf

from aimet_tensorflow.common import graph_eval

from Examples.tensorflow.utils.image_net_data_loader import ImageNetDataLoader
from Examples.tensorflow.utils.image_net_evaluator import ImageNetEvaluator
from Examples.common import image_net_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logger = logging.getLogger('Train')


class ImageNetTrainer:
    """
    For training the model using the ImageNet dataset.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=dangerous-default-value
    def __init__(self, tfrecord_dir: str, training_inputs: List[str], data_inputs: List[str],
                 validation_inputs: List[str], image_size: int = 224, batch_size: int = 128,
                 num_epochs: int = 1, format_bgr: bool = False, model_type: str = 'resnet'):
        """
        Constructor
        :param tfrecord_dir: The path to the TFRecords directory
        :param training_inputs: List of training ops names of the model
        :param data_inputs: List of input ops names of the model
        :param validation_inputs: List of validation ops names of the model
        :param image_size: Required size for images. Images will be resized to image_size x image_size
        :param batch_size: The batch size to use for validation
        :param num_epochs: Number of epochs to use in training
        :param format_bgr: Indicates to generate dateset images in BGR format
        :param model_type: Used to choose pre-processing function for one of
                           the 'resnet' or 'mobilenet' type model
        """

        if not data_inputs:
            raise ValueError("data_inputs list cannot be empty for imagenet")
        self._data_inputs = data_inputs

        if not validation_inputs:
            raise ValueError("validation_inputs list cannot be empty for imagenet")
        self._validation_inputs = validation_inputs

        if not training_inputs:
            raise ValueError("training_inputs list cannot be empty for imagenet")
        self._training_inputs = training_inputs

        self._train_data_loaders = ImageNetDataLoader(tfrecord_dir=tfrecord_dir, image_size=image_size,
                                                      batch_size=batch_size, num_epochs=num_epochs,
                                                      format_bgr=format_bgr, is_training=True,
                                                      model_type=model_type)
        self._tfrecord_dir = tfrecord_dir
        self._image_size = image_size
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._format_bgr = format_bgr
        self._model_type = model_type

    @staticmethod
    def remove_disconnected_ops(ops_list):
        """
        :param ops_list: List of ops to remove disconnected ops from
        """

        connected_ops = []
        for op in ops_list:
            # Do graph search on each ops_list. If 'input' is seen, it must
            # be connected to main graph, so add it to connected_ops
            stack = [op]
            seen = set(op.name)
            while stack:
                n_op = stack.pop()
                for z in n_op.outputs:
                    if z.op.name not in seen:
                        stack.append(z.op)
                        seen.add(z.op.name)
                for z in n_op.inputs:
                    if z.op.name not in seen:
                        stack.append(z.op)
                        seen.add(z.op.name)

            if 'input_1' in seen:
                connected_ops.append(op)

        return connected_ops

    def _evaluate_(self, session: tf.Session) -> float:
        """
        :param session: Tensorflow session to operate on
        """
        image_net_eval = ImageNetEvaluator(tfrecord_dir=self._tfrecord_dir, training_inputs=self._training_inputs,
                                           data_inputs=self._data_inputs, validation_inputs=self._validation_inputs,
                                           image_size=self._image_size, batch_size=self._batch_size,
                                           format_bgr=self._format_bgr, model_type=self._model_type)
        return image_net_eval.evaluate(session, iterations=None)

    def train(self, session: tf.Session, update_ops_name: List[str] = None, iterations: int = None,
              learning_rate: float = 0.001, decay_rate: float = 0.1, decay_steps: int = None,
              debug_steps: int = 1000):
        """
        :param session: Tensorflow session to operate on
        :param update_ops_name: list of name of update ops (mostly BatchNorms' moving averages).
                                tf.GraphKeys.UPDATE_OPS collections is always used
                                in addition to this list
        :param iterations: No of batches to use for training
        :param learning_rate: Learning rate
        :param decay_rate: Multiplicative factor of learning rate decay
        :param decay_steps: Adjust(decay) the learning rate by decay_rate after every decay_steps epochs
        :param debug_steps: number of training iterations to report accuracy/loss metrics
        """

        # pylint: disable-msg=too-many-locals
        if iterations is None:
            iterations = image_net_config.dataset['test_images_len'] // self._batch_size

        input_label_tensors = [session.graph.get_tensor_by_name(input_label)
                               for input_label in tuple(self._data_inputs)+tuple(self._validation_inputs)]
        train_tensors = [session.graph.get_tensor_by_name(training_input)
                         for training_input in self._training_inputs]

        train_tensors_dict = dict.fromkeys(train_tensors, False)

        if not update_ops_name:
            update_ops_name = []

        # Take only those ops from update_ops_name which are present in graph
        graph_all_ops_name = [op.name for op in session.graph.get_operations()]
        update_ops_name = set(update_ops_name).intersection(graph_all_ops_name)

        # Find the ops based on their name from update_ops_name
        update_ops = [session.graph.get_operation_by_name(op_name) for op_name in update_ops_name]

        with session.graph.as_default():
            loss_op = tf.get_collection(tf.GraphKeys.LOSSES)[0]

            global_step_op = tf.train.get_global_step()
            if global_step_op is None:
                global_step_op = tf.train.create_global_step()

            if decay_steps:
                learning_rate_op = tf.train.exponential_decay(learning_rate,
                                                              global_step=global_step_op,
                                                              decay_steps=decay_steps * iterations,
                                                              decay_rate=decay_rate,
                                                              staircase=True,
                                                              name='exponential_decay_learning_rate')
            else:
                learning_rate_op = learning_rate

            # Define an optimizer
            optimizer_op = tf.train.MomentumOptimizer(learning_rate=learning_rate_op, momentum=0.9)

            # Ensures that we execute the update_ops before performing the train_op
            update_ops = set(update_ops).union(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            connected_update_ops = self.remove_disconnected_ops(update_ops)
            with tf.control_dependencies(connected_update_ops):
                train_op = optimizer_op.minimize(loss_op, global_step=global_step_op)

            graph_eval.initialize_uninitialized_vars(session)

        logger.info('Training graph for %d iterations with batch_size %d for %d Epochs',
                    iterations, self._batch_size, self._num_epochs)

        for current_epoch in range(1, self._num_epochs + 1):
            avg_loss = 0.0
            curr_iter = 1
            with progressbar.ProgressBar(max_value=iterations) as progress_bar:
                for input_label in self._train_data_loaders:
                    input_label_tensors_dict = dict(zip(input_label_tensors, input_label))

                    feed_dict = {**input_label_tensors_dict, **train_tensors_dict}

                    with session.graph.as_default():
                        batch_loss_val, _ = session.run([loss_op, train_op], feed_dict=feed_dict)

                    avg_loss += batch_loss_val

                    progress_bar.update(curr_iter)

                    if curr_iter % debug_steps == 0:
                        eval_accuracy = self._evaluate_(session)
                        logger.info('Epoch #%d/%d: iteration #%d/%d: Global Avg Loss=%f, Eval Accuracy=%f',
                                    current_epoch, self._num_epochs, curr_iter, iterations,
                                    avg_loss / curr_iter, eval_accuracy)

                    curr_iter += 1
                    if curr_iter > iterations:
                        break

            eval_accuracy = self._evaluate_(session)
            logger.info('At the end of Epoch #%d/%d: Global Avg Loss=%f, Eval Accuracy=%f',
                        current_epoch, self._num_epochs, avg_loss / iterations, eval_accuracy)
