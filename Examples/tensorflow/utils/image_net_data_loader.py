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
Creates data-loader for Image-Net dataset
"""
import os
import multiprocessing
from typing import Tuple
import wget
import numpy as np
import tensorflow as tf

from Examples.common import image_net_config

# pylint: disable-msg=import-error
if not os.path.isfile("vgg_preprocessing.py"):
    wget.download(
        "https://raw.githubusercontent.com/tensorflow/models/r1.13.0/research/slim/preprocessing/vgg_preprocessing.py")

if not os.path.isfile("inception_preprocessing.py"):
    wget.download(
        "https://raw.githubusercontent.com/tensorflow/models/r1.13.0/research/slim/preprocessing/inception_preprocessing.py")

# pylint: disable-msg=wrong-import-position
from vgg_preprocessing import preprocess_image as vgg_resnet_preprocess_image
from inception_preprocessing import preprocess_image as inception_mobilenet_preprocess_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# The number of input Datasets to interleave from in parallel, used in tf.contrib.data.parallel_interleave
TRAIN_CYCLE_LEN = 10  # interleave from 10 different dataset to increase randomization
EVAL_CYCLE_LEN = 1  # No interleave/randomization for evaluation

# Represents the number of elements from the dataset from which the new dataset will sample.
SHUFFLE_BUFFER_SIZE = 8192  # Used in training for data randomization


class ImageNetDataLoader:
    """ Dataset generator for TfRecords of ImageNet """

    def __init__(self, tfrecord_dir: str, image_size: int = 224, batch_size: int = 128, num_epochs: int = 1,
                 format_bgr: bool = False, is_training: bool = False, model_type: str = 'resnet'):
        """
        :param tfrecord_dir: The path to the TFRecords directory
        :param image_size: Required size for images. Images will be resized to image_size x image_size
        :param batch_size: The batch size to use for training and validation
        :param num_epochs: How many times to repeat the dataset
        :param format_bgr: Indicates to generate dateset images in BGR format
        :param is_training: Indicates whether to load the training or validation data
        :param model_type: Used to choose pre-processing function for one of
                           the 'resnet' or 'mobilenet' type model
        :return: A new TfRecord generator used to generate data for model analysis
        """

        self._image_size = image_size
        self._batch_size = batch_size
        self._format_bgr = format_bgr
        self._is_training = is_training

        if model_type == 'mobilenet':
            self._preprocess_image = inception_mobilenet_preprocess_image
        else:
            self._preprocess_image = vgg_resnet_preprocess_image

        with tf.Graph().as_default():
            # Setup the Dataset reader

            if is_training:
                self._dataset = tf.data.Dataset.list_files(os.path.join(tfrecord_dir, 'train*'),
                                                           shuffle=True)
                cycle_length = TRAIN_CYCLE_LEN
            else:
                self._dataset = tf.data.Dataset.list_files(os.path.join(tfrecord_dir, 'validation*'),
                                                           shuffle=False)
                cycle_length = EVAL_CYCLE_LEN

            self._dataset = self._dataset.interleave(lambda tfrecord: (tf.data.TFRecordDataset(tfrecord)
                                                                       .map(self.parse,
                                                                            num_parallel_calls=multiprocessing.cpu_count())),
                                                     cycle_length=cycle_length)

            if is_training:
                self._dataset = self._dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

            self._dataset = self._dataset.repeat(num_epochs).batch(batch_size)

            # Creates a Dataset that prefetches elements from the dataset
            # This will buffer 1 dataset during prefetching that improves latency and throughput.
            self._dataset = self._dataset.prefetch(1)

            # Initialize the iterator. This must be allocated during init when the
            # generator is to be used manually. Otherwise the generator will generate a
            # new iterator each time it's used as an iterator
            self._iterator = self._dataset.make_one_shot_iterator()
            self._data_labels = self._iterator.get_next()
            self._sess = tf.Session()

    def __del__(self):
        """
        Closes tf session
        """
        self._sess.close()

    def __iter__(self):
        """
        Iter method for the generator
        :return:
        """
        return self

    def __next__(self) -> Tuple[np.ndarray]:
        """
        Return the next set of batched data

        **NOTE** This function will not return new batches until the previous batches have
        actually been used by a call to tensorflow. Eg used in a graph with a call to
        'run' etc. If it's unused the same tensors will be returned over and over again.

        :return: Tuple of multiple Input images followed by their corresponding labels
        """
        try:
            np_images_labels = self._sess.run(self._data_labels)
            return np_images_labels
        except tf.errors.OutOfRangeError:
            raise StopIteration

    def parse(self, serialized_example: tf.python.ops.Tensor) -> Tuple[tf.python.ops.Tensor]:
        """
        Parse one example
        :param serialized_example: single TFRecord file
        :return: Tuple of multiple Input Images tensors followed by their corresponding labels
        """
        features = tf.parse_single_example(serialized_example,
                                           features={'image/class/label': tf.FixedLenFeature([], tf.int64),
                                                     'image/encoded': tf.FixedLenFeature([], tf.string)})
        image_data = features['image/encoded']
        label = tf.cast(features['image/class/label'], tf.int32) - 1
        labels = tf.one_hot(indices=label, depth=image_net_config.dataset['images_classes'])

        # Decode the jpeg
        with tf.name_scope('prep_image', values=[image_data], default_name=None):
            # decode and reshape to default self._image_size x self._image_size
            # pylint: disable=no-member
            image = tf.image.decode_jpeg(image_data, channels=image_net_config.dataset['image_channels'])
            image = self._preprocess_image(image, self._image_size, self._image_size, is_training=self._is_training)
            if self._format_bgr:
                image = tf.reverse(image, axis=[-1])

        return (image,) + (labels,)

    @property
    def batch_size(self) -> int:
        """
        Returns the batch size
        :return: a value representing batch size
        """
        return self._batch_size

    @property
    def dataset(self) -> tf.data.Dataset:
        """
        Returns the dataset
        :return: a tf.data.Dataset representing dataset
        """
        return self._dataset
