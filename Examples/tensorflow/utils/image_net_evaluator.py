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
Creates Evaluator for Image-Net dataset
"""
import logging
from typing import List
import progressbar
import tensorflow as tf

from Examples.tensorflow.utils.image_net_data_loader import ImageNetDataLoader
from Examples.common import image_net_config

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logger = logging.getLogger('Eval')


class ImageNetEvaluator:
    """
    For validation of a trained model using the ImageNet dataset.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=dangerous-default-value
    def __init__(self, tfrecord_dir: str, training_inputs: List[str], data_inputs: List[str],
                 validation_inputs: List[str], image_size: int = 224, batch_size: int = 128,
                 format_bgr: bool = False, model_type: str = 'resnet'):
        """
        Constructor
        :param tfrecord_dir: The path to the TFRecords directory
        :param training_inputs: List of training ops names of the model
        :param data_inputs: List of input ops names of the model
        :param validation_inputs: List of validation ops names of the model
        :param image_size: Required size for images. Images will be resized to image_size x image_size
        :param batch_size: The batch size to use for validation
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

        self._val_data_loaders = ImageNetDataLoader(tfrecord_dir=tfrecord_dir, image_size=image_size,
                                                    batch_size=batch_size, num_epochs=1,
                                                    format_bgr=format_bgr, is_training=False,
                                                    model_type=model_type)
        self._batch_size = batch_size

    def evaluate(self, session: tf.Session, iterations: int = None) -> float:
        """
        :param session: Tensorflow session to operate on
        :param iterations: No of batches to use. Default is complete dataset
        :return: Top-1 accuracy
        """

        # pylint: disable-msg=too-many-locals
        if iterations is None:
            iterations = image_net_config.dataset['val_images_len'] // self._batch_size

        input_label_tensors = [session.graph.get_tensor_by_name(input_label)
                               for input_label in tuple(self._data_inputs)+tuple(self._validation_inputs)]
        train_tensors = [session.graph.get_tensor_by_name(training_input)
                         for training_input in self._training_inputs]

        train_tensors_dict = dict.fromkeys(train_tensors, True)

        eval_names = ['top1-acc', 'top5-acc']
        eval_outputs = [session.graph.get_operation_by_name(name).outputs[0] for name in eval_names]

        # Run the graph and verify the data is being updated properly for each iteration
        avg_acc_top1 = 0
        avg_acc_top5 = 0

        logger.info("Evaluating graph for %d iterations with batch_size %d", iterations, self._batch_size)

        curr_iter = 1
        with progressbar.ProgressBar(max_value=iterations) as progress_bar:
            for input_label in self._val_data_loaders:
                input_label_tensors_dict = dict(zip(input_label_tensors, input_label))

                feed_dict = {**input_label_tensors_dict, **train_tensors_dict}

                with session.graph.as_default():
                    output_data = session.run(eval_outputs, feed_dict=feed_dict)

                avg_acc_top1 += output_data[0]
                avg_acc_top5 += output_data[1]

                progress_bar.update(curr_iter)

                curr_iter += 1
                if curr_iter > iterations:
                    break

        logger.info('Avg accuracy Top 1: %f Avg accuracy Top 5: %f on validation Dataset',
                    avg_acc_top1 / iterations, avg_acc_top5 / iterations)

        return avg_acc_top1 / iterations
