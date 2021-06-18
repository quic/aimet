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

""" AdaRound Weights Unit Test Cases """

import pytest
import logging
import os
import json
import numpy as np
import unittest.mock
import tensorflow as tf

from aimet_common.utils import AimetLogger
from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.adaround.adaround_weight import Adaround, AdaroundParameters

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestAdaroundWeight(unittest.TestCase):
    """
    AdaRound Weights Unit Test Cases
    """
    def test_get_ordered_list_of_ops(self):
        """ Test get ordered list of ops for Adaround supported ops """
        tf.compat.v1.reset_default_graph()
        _ = keras_model()

        ordered_ops = Adaround._get_ordered_list_of_ops(tf.compat.v1.get_default_graph(),
                                                        input_op_names=['conv2d_input'],
                                                        output_op_names=['keras_model/Softmax'])
        self.assertEqual(len(ordered_ops), 3)

    def test_get_act_func(self):
        """ Test get activation func """
        tf.compat.v1.reset_default_graph()
        _ = keras_model()

        conv = tf.compat.v1.get_default_graph().get_operation_by_name('conv2d/Conv2D')
        act_func = Adaround._get_act_func(conv)
        self.assertEqual(act_func, None)

    def _apply_adaround(self, device):
        """ Test apply adaround and export functionality """
        np.random.seed(1)
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.compat.v1.reset_default_graph()

        with tf.device(device):
            graph = tf.Graph()
            with graph.as_default():
                tf.compat.v1.set_random_seed(1)
                _ = keras_model()
                init = tf.compat.v1.global_variables_initializer()

        session = tf.compat.v1.Session(graph=graph)
        session.run(init)

        dataset_size = 32
        batch_size = 16
        possible_batches = dataset_size // batch_size
        input_data = np.random.rand(dataset_size, 16, 16, 3)
        input_data = input_data.astype(dtype=np.float64)

        graph = tf.Graph()
        with graph.as_default():
            dataset = tf.data.Dataset.from_tensor_slices(input_data)
            dataset = dataset.batch(batch_size=batch_size)

        params = AdaroundParameters(data_set=dataset, num_batches=possible_batches, default_num_iterations=10)
        starting_op_names = ['conv2d_input']
        output_op_names = ['keras_model/Softmax']

        with tf.device(device):
            _ = Adaround.apply_adaround(session, starting_op_names, output_op_names, params, path='./',
                                        filename_prefix='dummy')
        session.close()

        # Test export functionality
        with open('./dummy.encodings') as json_file:
            encoding_data = json.load(json_file)
            print(encoding_data)

        param_keys = list(encoding_data.keys())
        print(param_keys)

        self.assertTrue(param_keys[0] == "conv2d/Conv2D/ReadVariableOp:0")
        self.assertTrue(isinstance(encoding_data["conv2d/Conv2D/ReadVariableOp:0"], list))
        param_encoding_keys = encoding_data["conv2d/Conv2D/ReadVariableOp:0"][0].keys()
        self.assertTrue("offset" in param_encoding_keys)
        self.assertTrue("scale" in param_encoding_keys)

        # Delete encodings file
        if os.path.exists("./dummy.encodings"):
            os.remove("./dummy.encodings")

    @pytest.mark.cuda
    def test_apply_adaround_gpu(self):
        """ Test apply adaround and export functionality for GPU """
        device = '/gpu:0'
        self._apply_adaround(device)

    def test_apply_adaround(self):
        """ Test apply adaround and export functionality for CPU """
        device = '/cpu:0'
        self._apply_adaround(device)
