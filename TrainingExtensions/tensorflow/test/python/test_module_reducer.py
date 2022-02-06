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
""" This file contains unit tests for testing the tf module reducer. """

import unittest
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from aimet_common.connected_graph.connectedgraph import get_ordered_ops
from aimet_tensorflow.examples.test_models import single_residual
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.winnow.module_reducer import _insert_downsample_or_upsample_ops_if_needed

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()


class TestModuleReducer(unittest.TestCase):
    """ Test ModuleReducer module """

    def test_downsample(self):
        """ Test utility for inserting downsample op """

        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(2, 2, 3,))

        # Remove middle channel
        output = _insert_downsample_or_upsample_ops_if_needed(inputs, [1, 1, 1], [1, 0, 1])
        self.assertEqual([None, 2, 2, 2], output.shape.as_list())

        with tf.compat.v1.Session() as sess:
            random_input = tf.random.uniform(shape=(1, 2, 2, 3))
            inp = random_input.eval(session=sess)
            inputs_eval = sess.run(inputs, feed_dict={inputs: inp})
            output_eval = sess.run(output, feed_dict={inputs: inp})
        sess.close()

        # Check that inputs_eval and output_eval are the same when the middle channel is removed from inputs_eval
        self.assertTrue(np.array_equal(np.concatenate((inputs_eval[:, :, :, 0:1],
                                                       inputs_eval[:, :, :, 2:]),
                                                      axis=-1),
                                       output_eval))

    def test_upsample(self):
        """ Test utility for inserting upsample op """

        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(2, 2, 3,))

        # Insert channel in index position 1
        output = _insert_downsample_or_upsample_ops_if_needed(inputs, [1, 0, 1, 1], [1, 1, 1, 1])
        self.assertEqual([None, 2, 2, 4], output.shape.as_list())

        with tf.compat.v1.Session() as sess:
            random_input = tf.random.uniform(shape=(1, 2, 2, 3))
            inp = random_input.eval(session=sess)
            inputs_eval = sess.run(inputs, feed_dict={inputs: inp})
            output_eval = sess.run(output, feed_dict={inputs: inp})
        sess.close()

        # Check that inputs_eval and output_eval are equal once channel 1 is removed from output_eval
        self.assertTrue(np.array_equal(np.concatenate((output_eval[:, :, :, 0:1],
                                                       output_eval[:, :, :, 2:]),
                                                      axis=-1),
                                       inputs_eval))
        # Check that output_eval channel 1 is all zeros
        self.assertTrue(np.array_equal(np.zeros(output_eval[:, :, :, 1:2].shape), output_eval[:, :, :, 1:2]))

    def test_downsample_upsample_checks(self):
        """ Test various assert conditions in downsample/upsample utility """

        tf.compat.v1.reset_default_graph()
        inputs = tf.keras.Input(shape=(2, 2, 4,))

        # Assert if parent mask and child mask are not the same sizes
        with self.assertRaises(AssertionError):
            _ = _insert_downsample_or_upsample_ops_if_needed(inputs, [1, 1, 1, 1], [1, 1, 1, 1, 1])

        # Assert if child mask and parent mask have the same channel number but have different channels winnowed
        with self.assertRaises(AssertionError):
            _ = _insert_downsample_or_upsample_ops_if_needed(inputs, [1, 1, 0, 1, 1], [1, 1, 1, 0, 1])

        # Assert if child mask has fewer channels but has a one in a position that the parent mask does not
        with self.assertRaises(AssertionError):
            _ = _insert_downsample_or_upsample_ops_if_needed(inputs, [1, 1, 0, 1, 1], [1, 0, 1, 0, 1])

        # Assert if child mask has more channels but does not have a one in a position that the parent mask does
        with self.assertRaises(AssertionError):
            _ = _insert_downsample_or_upsample_ops_if_needed(inputs, [1, 0, 1, 0, 1, 1], [1, 1, 0, 1, 1, 1])

    def test_get_ordered_operations(self):
        """ Test the creation of the ordered operations list """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            _ = single_residual()
            conn_graph = ConnectedGraph(sess.graph, ["input_1"], ['Relu_2'])
            ordered_ops = get_ordered_ops(conn_graph.starting_ops)

        # check that there are the same number of modules in the ordered ops list as there are in the main ops dict
        self.assertEqual(len(ordered_ops), len(conn_graph.get_all_ops()))

        # check that for any module in the ordered ops list, all of its parent modules are earlier in the list
        seen_ops = set()
        for op in ordered_ops:
            input_products = op.get_input_products()
            for product in input_products:
                self.assertTrue(product.producer in seen_ops)
            seen_ops.add(op)
        sess.close()
