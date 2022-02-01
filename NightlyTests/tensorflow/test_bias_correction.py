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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pytest
import unittest
import tensorflow as tf

from aimet_tensorflow.bias_correction import BiasCorrection, QuantParams, BiasCorrectionParams
from aimet_tensorflow.utils.graph_saver import load_model_from_meta
tf.compat.v1.disable_eager_execution()

mnist_model_path = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'mnist/models/')
mnist_tfrecords_path = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'mnist/data/')


class TestBiasCorrection(unittest.TestCase):

    @pytest.mark.cuda
    def test_correct_bias_on_mnist(self):
        """  Test correct bias using mnist model and data """

        def modified_parse(serialized_example):
            """
            Modified mnist dataset parser to provide only image data, without labels
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

            return image

        tf.compat.v1.reset_default_graph()
        batch_size = 2
        num_samples = 10
        dataset = tf.data.TFRecordDataset([os.path.join(mnist_tfrecords_path, 'validation.tfrecords')]).repeat(1)
        dataset = dataset.map(modified_parse, num_parallel_calls=batch_size)
        dataset = dataset.batch(batch_size=batch_size)

        quant_params = QuantParams()
        bias_correction_params = BiasCorrectionParams(batch_size=batch_size,
                                                      num_quant_samples=num_samples,
                                                      num_bias_correct_samples=num_samples,
                                                      input_op_names=['reshape_input'],
                                                      output_op_names=['dense_1/BiasAdd'])

        meta_path = os.path.join(mnist_model_path, 'mnist_save.meta')
        checkpoint_path = os.path.join(mnist_model_path, 'mnist_save')
        sess = load_model_from_meta(meta_path=meta_path, checkpoint_path=checkpoint_path)
        BiasCorrection.correct_bias(sess, bias_correction_params, quant_params, dataset)
        self.assertTrue(1)      # Add some actual error check
        sess.close()

    @pytest.mark.cuda
    def test_correct_bias_on_mnist_with_analytical_bc(self):
        """  Test correct bias using mnist model and data (analytical bias correction)"""

        def modified_parse(serialized_example):
            """
            Modified mnist dataset parser to provide only image data, without labels
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

            return image

        tf.compat.v1.reset_default_graph()
        batch_size = 2
        num_samples = 10
        dataset = tf.data.TFRecordDataset([os.path.join(mnist_tfrecords_path, 'validation.tfrecords')]).repeat(1)
        dataset = dataset.map(modified_parse, num_parallel_calls=batch_size)
        dataset = dataset.batch(batch_size=batch_size)

        quant_params = QuantParams()
        bias_correction_params = BiasCorrectionParams(batch_size=batch_size,
                                                      num_quant_samples=num_samples,
                                                      num_bias_correct_samples=num_samples,
                                                      input_op_names=['reshape_input'],
                                                      output_op_names=['dense_1/BiasAdd'])

        meta_path = os.path.join(mnist_model_path, 'mnist_save.meta')
        checkpoint_path = os.path.join(mnist_model_path, 'mnist_save')
        sess = load_model_from_meta(meta_path=meta_path, checkpoint_path=checkpoint_path)
        BiasCorrection.correct_bias(sess, bias_correction_params, quant_params, dataset,
                                    perform_only_empirical_bias_corr=False)
        self.assertTrue(1)      # Add some actual error check
        sess.close()

    def test_dummy(self):
        # pytest has a 'feature' that returns an error code when all tests for a given suite are not selected
        # to be executed
        # So adding a dummy test to satisfy pytest
        pass
