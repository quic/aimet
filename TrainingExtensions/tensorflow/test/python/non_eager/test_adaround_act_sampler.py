# /usr/bin/env python3.6
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

""" Unit tests for Adaround Activation Sampler """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import unittest.mock
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2DTranspose

from aimet_common.utils import AimetLogger
from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.adaround.activation_sampler import ActivationSampler

tf.compat.v1.disable_eager_execution()


class TestAdaroundActivationSampler(unittest.TestCase):
    """
     AdaRound Activation Sampler Unit Test Cases
    """
    def _activation_sampler_conv(self, device):
        """ Test ActivationSampler for a Conv op """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.compat.v1.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(device):
                _ = keras_model()
                init = tf.compat.v1.global_variables_initializer()

        inp_op_names = ['conv2d_input']
        conv = graph.get_operation_by_name('conv2d/Conv2D')

        session = tf.compat.v1.Session(graph=graph)
        session.run(init)

        dataset_size = 16
        batch_size = 8
        possible_batches = dataset_size // batch_size
        input_data = np.random.rand(dataset_size, 16, 16, 3)

        graph = tf.Graph()
        with graph.as_default():
            dataset = tf.data.Dataset.from_tensor_slices(input_data)
            dataset = dataset.batch(batch_size=batch_size)

        act_sampler = ActivationSampler(dataset)
        inp_data, out_data = act_sampler.sample_activation(conv, conv, session, session, inp_op_names, possible_batches)

        self.assertEqual(list(inp_data.shape), [batch_size * possible_batches, 16, 16, 3])
        self.assertEqual(list(out_data.shape), [batch_size * possible_batches, 15, 15, 8])

        session.close()

    def test_activation_sampler_conv(self):
        """ Test ActivationSampler for a Conv op CPU """
        device = '/cpu:0'
        self._activation_sampler_conv(device)

    def test_get_output_tensor(self):
        """ Test get output tensor utility """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.compat.v1.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            _ = keras_model()

        op = graph.get_operation_by_name('conv2d/Conv2D')

        output_tensor = ActivationSampler._get_output_tensor(op)
        self.assertEqual(output_tensor.op.type, 'BiasAdd')
        self.assertNotEqual(output_tensor.op.type, 'Conv2D')

    def test_get_conv_transpose_input_tensor(self):
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.compat.v1.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            _ = Sequential([Conv2DTranspose(8, (2, 2), input_shape=(16, 16, 3,))])

        op = graph.get_operation_by_name('conv2d_transpose/conv2d_transpose')

        inp_tensor = ActivationSampler._get_input_tensor(op)
        assert inp_tensor == graph.get_tensor_by_name('conv2d_transpose_input:0')
