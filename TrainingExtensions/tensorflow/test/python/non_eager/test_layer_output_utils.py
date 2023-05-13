# /usr/bin/env python3.8
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
import shutil

import numpy as np
import tensorflow as tf

from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.layer_output_utils import LayerOutputUtil, LayerOutput
from aimet_tensorflow.utils.common import iterate_tf_dataset

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()


def cpu_session():
    tf.compat.v1.reset_default_graph()
    with tf.device('/cpu:0'):
        model = keras_model()
        init = tf.compat.v1.global_variables_initializer()
    session = tf.compat.v1.Session()
    session.run(init)
    return session


def quantsim_forward_pass_callback(session, dummy_input):
    model_input = session.graph.get_tensor_by_name('conv2d_input:0')
    model_output = session.graph.get_tensor_by_name('keras_model/Softmax_quantized:0')
    return session.run(model_output, feed_dict={model_input: dummy_input})


def original_model_forward_pass_callback(session, dummy_input):
    model_input = session.graph.get_tensor_by_name('conv2d_input:0')
    model_output = session.graph.get_tensor_by_name('keras_model/Softmax:0')
    return session.run(model_output, feed_dict={model_input: dummy_input})


def get_quantsim_artifacts():
    # Load sample model into cpu session
    session = cpu_session()
    dummy_input = np.random.randn(1, 16, 16, 3)

    # Obtain quantsim object
    quantsim = QuantizationSimModel(session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
    quantsim.compute_encodings(quantsim_forward_pass_callback, dummy_input)

    # Layer-output names by manually looking at the model
    layer_output_names = [
        'conv2d_input_0',
        'conv2d_BiasAdd_0',
        'batch_normalization_cond_Identity_0',
        'average_pooling2d_AvgPool_0',
        'max_pooling2d_MaxPool_0',
        'batch_normalization_1_cond_Identity_0',
        'conv2d_1_BiasAdd_0',
        'conv2d_1_Tanh_0',
        'keras_model_BiasAdd_0',
        'keras_model_Softmax_0',
    ]

    return quantsim, ['conv2d_input'], ['keras_model/Softmax_quantized'], layer_output_names, dummy_input


def get_original_model_artifacts():
    # Load sample model into cpu session
    session = cpu_session()
    dummy_input = np.random.randn(1, 16, 16, 3)

    # Layer-output names by manually looking at the model
    layer_output_names = [
        'conv2d_input_0',
        'conv2d_BiasAdd_0',
        'batch_normalization_cond_Identity_0',
        'average_pooling2d_AvgPool_0',
        'max_pooling2d_MaxPool_0',
        'batch_normalization_1_cond_Identity_0',
        'conv2d_1_BiasAdd_0',
        'conv2d_1_Tanh_0',
        'keras_model_BiasAdd_0',
        'keras_model_Softmax_0',
    ]

    return session, ['conv2d_input'], ['keras_model/Softmax'], layer_output_names, dummy_input


class TestLayerOutput:
    def test_get_quantsim_outputs(self):
        """ Test whether outputs are generated for all the layers of a quantsim model """

        # Get quantsim artifacts
        quantsim, starting_ops, output_ops, output_names, dummy_input = get_quantsim_artifacts()

        temp_dir_path = os.path.dirname(os.path.abspath(__file__))
        temp_dir_path = os.path.join(temp_dir_path, 'temp_dir')

        # Obtain layer-outputs of quantsim model
        layer_output_obj = LayerOutput(session=quantsim.session, starting_op_names=starting_ops, output_op_names=output_ops, dir_path=temp_dir_path)
        feed_dict = {quantsim.session.graph.get_tensor_by_name('conv2d_input:0'): dummy_input}
        layer_output_name_to_layer_output_dict = layer_output_obj.get_outputs(feed_dict)

        # Verify whether outputs are generated for all the layers
        for output_name in output_names:
            assert output_name in layer_output_name_to_layer_output_dict, "Output not generated for " + output_name

        # Verify whether outputs are quantized outputs. This can only be checked for final output of the model.
        model_output = quantsim_forward_pass_callback(quantsim.session, dummy_input)
        captured_model_output = layer_output_name_to_layer_output_dict['keras_model_Softmax_0']
        assert np.array_equal(model_output, captured_model_output), \
            "Output of last layer of quantsim model doesn't match with captured layer-output"

        # Delete temp_dir
        shutil.rmtree(temp_dir_path, ignore_errors=False, onerror=None)

        quantsim.session.close()
        del quantsim

    def test_get_original_model_outputs(self):
        """ Test whether outputs are generated for all the layers of an original model """

        # Get original model artifacts
        session, starting_ops, output_ops, output_names, dummy_input = get_original_model_artifacts()

        temp_dir_path = os.path.dirname(os.path.abspath(__file__))
        temp_dir_path = os.path.join(temp_dir_path, 'temp_dir')

        # Obtain layer-outputs of original model
        layer_output_obj = LayerOutput(session=session, starting_op_names=starting_ops, output_op_names=output_ops, dir_path=temp_dir_path)
        feed_dict = {session.graph.get_tensor_by_name('conv2d_input:0'): dummy_input}
        layer_output_name_to_layer_output_dict = layer_output_obj.get_outputs(feed_dict)

        # Verify whether outputs are generated for all the layers
        for output_name in output_names:
            assert output_name in layer_output_name_to_layer_output_dict, "Output not generated for " + output_name

        # Verify whether outputs are correct. This can only be checked for final output of the model.
        model_output = original_model_forward_pass_callback(session, dummy_input)
        captured_model_output = layer_output_name_to_layer_output_dict['keras_model_Softmax_0']
        assert np.array_equal(model_output, captured_model_output), \
            "Output of last layer of original model doesn't match with captured layer-output"

        # Delete temp_dir
        shutil.rmtree(temp_dir_path, ignore_errors=False, onerror=None)

        session.close()


def get_dataset_artifacts():
    dataset_size = 4
    batch_size = 2
    input_data = np.random.rand(dataset_size, 16, 16, 3)
    dataset = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size=batch_size)
    return dataset, dataset_size, input_data[0]


class TestLayerOutputUtil:
    def test_generate_layer_outputs(self):
        """ Test whether input files and corresponding layer-output files are generated """

        # Get quantsim artifacts
        quantsim, starting_ops, output_ops, output_names, dummy_input = get_quantsim_artifacts()

        # Get dataset artifacts
        dummy_dataset, data_count, first_input = get_dataset_artifacts()

        temp_dir_path = os.path.dirname(os.path.abspath(__file__))
        temp_dir_path = os.path.join(temp_dir_path, 'temp_dir')

        # Generate layer-outputs
        layer_output_util = LayerOutputUtil(session=quantsim.session, starting_op_names=starting_ops,
                                            output_op_names=output_ops, dir_path=temp_dir_path)
        iterator = iterate_tf_dataset(dummy_dataset)
        for input_batch in iterator:
            layer_output_util.generate_layer_outputs(input_batch)

        # Verify number of inputs
        assert data_count == len(os.listdir(os.path.join(temp_dir_path, 'inputs')))

        # Verify number of layer-output folders
        assert data_count == len(os.listdir(os.path.join(temp_dir_path, 'outputs')))

        # Verify number of layer-outputs
        saved_layer_outputs = os.listdir(os.path.join(temp_dir_path, 'outputs', 'layer_outputs_0'))
        saved_layer_outputs = [i[:-len('.raw')] for i in saved_layer_outputs]
        for name in output_names:
            assert name in saved_layer_outputs

        # Ensure generated layer-outputs can be correctly loaded for layer-output comparison
        saved_last_layer_output = np.fromfile(os.path.join(temp_dir_path, 'outputs', 'layer_outputs_0', 'keras_model_Softmax_0.raw'), dtype=np.float32).reshape((1, 2))
        last_layer_output = quantsim_forward_pass_callback(quantsim.session, np.expand_dims(first_input, axis=0))
        assert np.array_equal(saved_last_layer_output, last_layer_output)

        # Delete temp_dir
        shutil.rmtree(temp_dir_path, ignore_errors=False, onerror=None)

        quantsim.session.close()
        del quantsim
