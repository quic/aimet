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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import numpy as np
import unittest.mock
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2DTranspose

from aimet_common.utils import AimetLogger
from aimet_common.quantsim_config.json_config_importer import JsonConfigImporter
from aimet_tensorflow.examples.test_models import keras_model, single_residual
from aimet_tensorflow.adaround.adaround_weight import Adaround, AdaroundParameters, tf_op_type_to_onnx_type_dict

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
tf.compat.v1.disable_eager_execution()


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

    def test_get_is_symmetric_flag_for_op_param(self):
        """ test get_is_symmetric_flag_for_op_param() """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            _ = single_residual()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
        conv_op = sess.graph.get_operation_by_name("conv2d/Conv2D")
        matmul_op = sess.graph.get_operation_by_name("single_residual/MatMul")

        # default case
        config = {
            "defaults": {
                "ops": {},
                "params": {}
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }
        with open('./config.json', 'w') as f:
            json.dump(config, f)

        try:
            configs = JsonConfigImporter.import_json_config_file(config_file='./config.json')
            assert not Adaround.get_is_symmetric_flag_for_op_param(configs, conv_op.type,
                                                                   param_name="weight",
                                                                   framework_to_onnx_type_dict=tf_op_type_to_onnx_type_dict)
        finally:
            if os.path.isfile('./config.json'):
                os.remove('./config.json')

        # All params having is_symmetric True.
        config = {
            "defaults": {
                "ops": {},
                "params": {
                    "is_symmetric": "True"
                }
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }
        with open('./config.json', 'w') as f:
            json.dump(config, f)

        try:
            configs = JsonConfigImporter.import_json_config_file(config_file='./config.json')
            assert Adaround.get_is_symmetric_flag_for_op_param(configs, conv_op.type,
                                                               param_name="weight",
                                                               framework_to_onnx_type_dict=tf_op_type_to_onnx_type_dict)
        finally:
            if os.path.isfile('./config.json'):
                os.remove('./config.json')

        # All params having is_symmetric False, but "weight" parameters have is_symmetric True.
        config = {
            "defaults": {
                "ops": {},
                "params": {
                    "is_symmetric": "False"
                }
            },
            "params": {
                "weight": {
                    "is_symmetric": "True"
                }
            },
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }
        with open('./config.json', 'w') as f:
            json.dump(config, f)

        try:
            configs = JsonConfigImporter.import_json_config_file(config_file='./config.json')
            assert Adaround.get_is_symmetric_flag_for_op_param(configs, conv_op.type,
                                                               param_name="weight",
                                                               framework_to_onnx_type_dict=tf_op_type_to_onnx_type_dict)
        finally:
            if os.path.isfile('./config.json'):
                os.remove('./config.json')

        # All params having is_symmetric False, but "weight" parameters of type Conv have is_symmetric True.
        config = {
            "defaults": {
                "ops": {},
                "params": {
                    "is_symmetric": "False"
                }
            },
            "params": {
                "weight": {
                    "is_symmetric": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "params": {
                        "weight": {
                            "is_symmetric": "True"
                        },
                        "bias": {
                            "is_symmetric": "False"
                        }
                    }
                }
            },
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }
        with open('./config.json', 'w') as f:
            json.dump(config, f)

        try:
            configs = JsonConfigImporter.import_json_config_file(config_file='./config.json')
            assert Adaround.get_is_symmetric_flag_for_op_param(configs, conv_op.type,
                                                               param_name="weight",
                                                               framework_to_onnx_type_dict=tf_op_type_to_onnx_type_dict)
            # For matmul op, is_symmetric should be False.
            assert not Adaround.get_is_symmetric_flag_for_op_param(configs, matmul_op.type,
                                                                   param_name="weight",
                                                                   framework_to_onnx_type_dict=tf_op_type_to_onnx_type_dict)
        finally:
            if os.path.isfile('./config.json'):
                os.remove('./config.json')

        sess.close()

    def test_apply_adaround_per_channel(self):
        """
        Test apply per channel adaround and export functionality for CPU
        """
        device = '/cpu:0'
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

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {
                    "is_symmetric": "True"
                },
                "per_channel_quantization": "True",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }

        with open('./config.json', 'w') as f:
            json.dump(quantsim_config, f)

        with tf.device(device):
            _ = Adaround.apply_adaround(session, starting_op_names, output_op_names, params, path='./',
                                        filename_prefix='dummy', default_config_file='./config.json')
        session.close()

        # Test export functionality

        with open('./dummy.encodings') as json_file:
            encoding_data = json.load(json_file)
        param_keys = list(encoding_data.keys())

        self.assertTrue(param_keys[0] == "conv2d/Conv2D/ReadVariableOp:0")
        self.assertTrue(isinstance(encoding_data["conv2d/Conv2D/ReadVariableOp:0"], list))
        self.assertTrue(len(encoding_data["conv2d/Conv2D/ReadVariableOp:0"]) == 8)

        self.assertTrue(param_keys[1] == "conv2d_1/Conv2D/ReadVariableOp:0")
        self.assertTrue(isinstance(encoding_data["conv2d_1/Conv2D/ReadVariableOp:0"], list))
        self.assertTrue(len(encoding_data["conv2d_1/Conv2D/ReadVariableOp:0"]) == 4)

        # Delete encodings file
        if os.path.exists("./dummy.encodings"):
            os.remove("./dummy.encodings")

    def test_apply_adaround_per_channel_conv2d_transpose(self):
        """
        Test apply per channel adaround and export functionality for conv transpose
        """
        device = '/cpu:0'
        np.random.seed(1)
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.compat.v1.reset_default_graph()

        with tf.device(device):
            graph = tf.Graph()
            with graph.as_default():
                tf.compat.v1.set_random_seed(1)
                _ = Sequential([Conv2DTranspose(8, (2, 2), input_shape=(16, 16, 3,))])
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
        starting_op_names = ['conv2d_transpose_input']
        output_op_names = ['conv2d_transpose/BiasAdd']

        quantsim_config = {
            "defaults": {
                "ops": {},
                "params": {
                    "is_symmetric": "True"
                },
                "per_channel_quantization": "True",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {
                "is_output_quantized": "True"
            }
        }

        with open('./config.json', 'w') as f:
            json.dump(quantsim_config, f)

        with tf.device(device):
            adarounded_session = Adaround.apply_adaround(session, starting_op_names, output_op_names, params, path='./',
                                                         filename_prefix='conv2d_transpose',
                                                         default_config_file='./config.json')
        session.close()

        # Test export functionality

        with open('./conv2d_transpose.encodings') as json_file:
            encoding_data = json.load(json_file)

        param_keys = list(encoding_data.keys())

        self.assertTrue(param_keys[0] == "conv2d_transpose/conv2d_transpose/ReadVariableOp:0")
        conv_transpose_encoding_data = encoding_data["conv2d_transpose/conv2d_transpose/ReadVariableOp:0"]
        self.assertTrue(isinstance(conv_transpose_encoding_data, list))

        self.assertTrue(isinstance(conv_transpose_encoding_data, list))
        self.assertTrue(len(conv_transpose_encoding_data) == 8)


        def dummy_forward_pass(session: tf.compat.v1.Session, _):
            """
            This is intended to be the user-defined model evaluation function.
            AIMET requires the above signature. So if the user's eval function does not
            match this signature, please create a simple wrapper.
            :param session: Session with model to be evaluated
            :param _: These argument(s) are passed to the forward_pass_callback as-is. Up to
                    the user to determine the type of this parameter. E.g. could be simply an integer representing the number
                    of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
                    If set to None, forward_pass_callback will be invoked with no parameters.
            :return: single float number (accuracy) representing model's performance
            """
            input_data = np.random.rand(1, 16, 16, 3)
            input_tensor = session.graph.get_tensor_by_name('conv2d_transpose_input:0')
            output_tensor = session.graph.get_tensor_by_name('conv2d_transpose/BiasAdd:0')
            output = session.run(output_tensor, feed_dict={input_tensor: input_data})
            return output

        from aimet_tensorflow.quantsim import QuantizationSimModel, QuantScheme
        qsim = QuantizationSimModel(adarounded_session, starting_op_names, output_op_names,
                                    quant_scheme=QuantScheme.post_training_tf, config_file='./config.json')
        # Set and freeze encodings to use same quantization grid and then invoke compute encodings
        qsim.set_and_freeze_param_encodings(encoding_path='./conv2d_transpose.encodings')
        qsim.compute_encodings(dummy_forward_pass, None)

        # Delete encodings file
        if os.path.exists("./conv2d_transpose.encodings"):
            os.remove("./conv2d_transpose.encodings")
