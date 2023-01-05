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

""" AdaRound Nightly Tests """

import pytest
import json
import unittest
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from packaging import version
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.adaround.adaround_weight import Adaround, AdaroundParameters

tf.compat.v1.disable_eager_execution()


class AdaroundAcceptanceTests(unittest.TestCase):
    """
    AdaRound test cases
    """
    @pytest.mark.cuda
    def test_adaround_mobilenet_only_weights(self):
        """ test end to end adaround with only weight quantized """

        def dummy_forward_pass(session: tf.compat.v1.Session):
            """ Dummy forward pass """
            input_data = np.random.rand(1, 224, 224, 3)
            input_tensor = session.graph.get_tensor_by_name('input_1:0')
            output_tensor = session.graph.get_tensor_by_name('conv_preds/BiasAdd:0')
            output = session.run(output_tensor, feed_dict={input_tensor: input_data})
            return output

        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.compat.v1.reset_default_graph()

        _ = MobileNet(weights=None, input_shape=(224, 224, 3))
        init = tf.compat.v1.global_variables_initializer()

        dataset_size = 128
        batch_size = 64
        possible_batches = dataset_size // batch_size
        input_data = np.random.rand(dataset_size, 224, 224, 3)
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.batch(batch_size=batch_size)

        session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        session.run(init)

        params = AdaroundParameters(data_set=dataset, num_batches=possible_batches, default_num_iterations=1,
                                    default_reg_param=0.01, default_beta_range=(20, 2), default_warm_start=0.2)

        starting_op_names = ['input_1']
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            output_op_names = ['predictions/Softmax']
        else:
            output_op_names = ['act_softmax/Softmax']

        adarounded_session = Adaround.apply_adaround(session, starting_op_names, output_op_names, params, path='./',
                                                     filename_prefix='mobilenet', default_param_bw=4,
                                                     default_quant_scheme=QuantScheme.post_training_tf_enhanced)

        orig_output = dummy_forward_pass(session)
        adarounded_output = dummy_forward_pass(adarounded_session)
        self.assertEqual(orig_output.shape, adarounded_output.shape)

        # Test exported encodings JSON file
        with open('./mobilenet.encodings') as json_file:
            encoding_data = json.load(json_file)
            print(encoding_data)

        self.assertTrue(isinstance(encoding_data["conv1/Conv2D/ReadVariableOp:0"], list))

        session.close()
        adarounded_session.close()

        # Delete encodings JSON file
        if os.path.exists("./mobilenet.encodings"):
            os.remove("./mobilenet.encodings")

    def test_adaround_resnet18_followed_by_quantsim(self):
        """ test end to end adaround with weight 4 bits and output activations 8 bits quantized """

        def dummy_forward_pass(session: tf.compat.v1.Session, _):
            """ Dummy forward pass """
            input_data = np.random.rand(32, 16, 16, 3)
            input_tensor = session.graph.get_tensor_by_name('conv2d_input:0')
            output_tensor = session.graph.get_tensor_by_name('keras_model/Softmax:0')
            output = session.run(output_tensor, feed_dict={input_tensor: input_data})
            return output

        np.random.seed(1)
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.compat.v1.reset_default_graph()

        _ = keras_model()
        init = tf.compat.v1.global_variables_initializer()
        dataset_size = 32
        batch_size = 16
        possible_batches = dataset_size // batch_size
        input_data = np.random.rand(dataset_size, 16, 16, 3)
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.batch(batch_size=batch_size)

        session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        session.run(init)

        params = AdaroundParameters(data_set=dataset, num_batches=possible_batches, default_num_iterations=10)
        starting_op_names = ['conv2d_input']
        output_op_names = ['keras_model/Softmax']

        # W4A8
        param_bw = 4
        output_bw = 8
        quant_scheme = QuantScheme.post_training_tf_enhanced

        adarounded_session = Adaround.apply_adaround(session, starting_op_names, output_op_names, params, path='./',
                                                     filename_prefix='dummy', default_param_bw=param_bw,
                                                     default_quant_scheme=quant_scheme, default_config_file=None)

        # Read exported param encodings JSON file
        with open('./dummy.encodings') as json_file:
            encoding_data = json.load(json_file)

        encoding = encoding_data["conv2d/Conv2D/ReadVariableOp:0"][0]
        before_min, before_max, before_delta, before_offset = encoding.get('min'), encoding.get('max'), \
                                                              encoding.get('scale'), encoding.get('offset')

        print(before_min, before_max, before_delta, before_offset)

        # Create QuantSim using adarounded_model, set and freeze parameter encodings and then invoke compute_encodings
        sim = QuantizationSimModel(adarounded_session, starting_op_names, output_op_names, quant_scheme,
                                   default_output_bw=output_bw, default_param_bw=param_bw, use_cuda=False)

        sim.set_and_freeze_param_encodings(encoding_path='./dummy.encodings')
        sim.compute_encodings(dummy_forward_pass, None)

        quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')
        encoding = quantizer.get_encoding()
        after_min, after_max, after_delta, after_offset = encoding.min, encoding.max, encoding.delta, encoding.offset

        print(after_min, after_max, after_delta, after_offset)

        self.assertEqual(before_min, after_min)
        self.assertEqual(before_max, after_max)
        self.assertAlmostEqual(before_delta, after_delta, places=4)
        self.assertEqual(before_offset, after_offset)

        session.close()
        adarounded_session.close()

        # Delete encodings file
        if os.path.exists("./dummy.encodings"):
            os.remove("./dummy.encodings")
