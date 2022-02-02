# /usr/bin/env python2.7
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Data generator code for MNIST and ImageNet datasets. Works on TF Record format. """

import tensorflow as tf

class MnistParser:
    """ Parses MNIST dataset """

    def __init__(self, data_inputs=None, validation_inputs=None, batch_size=10):
        """
        Constructor
        :param data_inputs: List of input ops for the model
        :param validation_inputs: List of validation ops for the model
        :param batch_size: Batch size for the data
        """
        if not data_inputs:
            data_inputs = ['data']

        if len(data_inputs) > 1:
            raise ValueError("Only one data input supported for mnist")
        self._data_inputs = data_inputs

        if not validation_inputs:
            validation_inputs = ['labels']

        if len(validation_inputs) > 1:
            raise ValueError("Only one validation input supported for mnist")
        self._validation_inputs = validation_inputs

        self._batch_size = batch_size

    @staticmethod
    def parse(serialized_example):
        """
        Parse one example
        :param serialized_example:
        :return: Input image and labels
        """
        dim = 28
        features = tf.compat.v1.parse_single_example(serialized_example,
                                                     features={'label': tf.compat.v1.FixedLenFeature([], tf.int64),
                                                               'image_raw': tf.compat.v1.FixedLenFeature([], tf.string)})

        # Mnist examples are flattened. Since we aren't performing an augmentations
        # these can remain flattened.
        image = tf.compat.v1.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([dim*dim])

        # Convert from bytes to floats 0 -> 1.
        image = tf.cast(image, tf.float32) / 255
        label = tf.cast(features['label'], tf.int32)
        labels = tf.one_hot(indices=label, depth=10)

        return image, labels

    def get_batch(self, iterator):
        """
        Get the next batch of data
        :param iterator: Data iterator
        :return: Input images and labels in feed_dict form
        """

        data, labels = iterator.get_next()
        with tf.compat.v1.Session(graph=data.graph) as sess:
            np_images, np_labels = sess.run([data, labels])
        return {self._data_inputs[0]: np_images, self._validation_inputs[0]: np_labels}

    def get_batch_size(self):
        """
        Returns the batch size
        :return:
        """
        return self._batch_size

    def get_data_inputs(self):
        """
        Get a list of data input
        :return: List of data input ops
        """
        return self._data_inputs

    def get_validation_inputs(self):
        """
        Get a list of validation input
        :return: List of validation input ops
        """
        return self._validation_inputs


class ImagenetParser:
    """ Parses ImageNet dataset """

    def __init__(self, data_inputs=None, validation_inputs=None, batch_size=1):
        """
        Constructor
        :param data_inputs: List of input ops for the model
        :param validation_inputs: List of validation ops for the model
        :param batch_size: Batch size for the data
        """

        if not data_inputs:
            data_inputs = ['data']

        if len(data_inputs) > 1:
            raise ValueError("Only one data input supported for imagenet")
        self._data_inputs = data_inputs

        if not validation_inputs:
            validation_inputs = ['labels']

        if len(validation_inputs) > 1:
            raise ValueError("Only one validation input supported for imagenet")
        self._validation_inputs = validation_inputs
        self._batch_size = batch_size

    @staticmethod
    def parse(serialized_example):
        """
        Parse one example
        :param serialized_example:
        :return: Input image and labels
        """
        dim = 224

        features = tf.compat.v1.parse_single_example(serialized_example,
                                                     features={
                                                         'image/class/label': tf.FixedLenFeature([], tf.int64),
                                                         'image/encoded': tf.FixedLenFeature([], tf.string)})
        image_data = features['image/encoded']
        label = tf.cast(features['image/class/label'], tf.int32)
        labels = tf.one_hot(indices=label, depth=1000)

        # Decode the jpeg
        with tf.compat.v1.name_scope('prep_image', [image_data], None):
            # decode and reshape to default 224x224
            # pylint: disable=no-member
            image = tf.image.decode_jpeg(image_data, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [dim, dim])

        return image, labels

    def get_batch(self, iterator):
        """
        Get the next batch of data
        :param iterator: Data iterator
        :return: Input images and labels in feed_dict form
        """
        data, labels = iterator.get_next()
        with tf.compat.v1.Session(graph=data.graph) as sess:
            np_images, np_labels = sess.run([data, labels])
        return {self._data_inputs[0]: np_images, self._validation_inputs[0]: np_labels}

    def get_batch_size(self):
        """
        Returns the batch size
        :return:
        """
        return self._batch_size

    def get_data_inputs(self):
        """
        Get a list of data input
        :return: List of data input ops
        """
        return self._data_inputs

    def get_validation_inputs(self):
        """
        Get a list of validation input
        :return: List of validation input ops
        """
        return self._validation_inputs


class TfRecordGenerator:

    """ Dataset generator for TfRecords"""

    def __init__(self, tfrecords, parser=MnistParser(), num_gpus=1, num_epochs=None):
        """
        Constructor
        :param tfrecords: A list of TfRecord files
        :param parser: Defaults to use the mnist tfrecord parser, but any custom
                parser function can be passed to read a custom tfrecords format.
        :param num_gpus: The number of GPUs being used. Data batches must be generated for each GPU device
        :param num_epochs: How many times to repeat the dataset. Default is forever. Then the
                amount of data generated is determined by the number of iterations the model is run and the batch
                size. If set to a specific number the dataset will only provide the amount of the total dataset
                'num_epochs' times.
        :return: A new TfRecord generator used to generate data for model analysis
        """

        self._parser = parser
        self._num_gpus = num_gpus

        # Setup the Dataset reader
        self._dataset = tf.data.TFRecordDataset(tfrecords).repeat(num_epochs)
        batch_size = parser.get_batch_size()
        self._dataset = self._dataset.map(parser.parse, num_parallel_calls=batch_size)
        self._dataset = self._dataset.batch(batch_size)

        # Initialize the iterator. This must be allocated during init when the
        # generator is to be used manually. Otherwise the generator will generate a
        # new iterator each time it's used as an iterator
        self._iterator = tf.compat.v1.data.make_one_shot_iterator(self._dataset)

    def __iter__(self):
        """
        Iter method for the generator
        :return:
        """
        # creating one shot iterator ops in same graph as dataset ops
        # TODO: this will keep adding iterator ops in the same graph every time this iter method is being called, need
        #  better solution

        # pylint: disable=protected-access
        with self._dataset._graph.as_default():
            self._iterator = tf.compat.v1.data.make_one_shot_iterator(self._dataset)

        return self

    def __next__(self):
        """
        Return the next set of batched data

        **NOTE** This function will not return new batches until the previous batches have
        actually been used by a call to tensorflow. Eg used in a graph with a call to
        'run' etc. If it's unused the same tensors will be returned over and over again.

        :return:
        """
        return self._parser.get_batch(self._iterator)

    # Map next for python27 compatibility
    next = __next__

    def get_data_inputs(self):
        """
        Returns a list of data input ops
        :return:
        """
        return self._parser.get_data_inputs()

    def get_validation_inputs(self):
        """
        Returns a list of validation input ops
        :return:
        """
        return self._parser.get_validation_inputs()
