# /usr/bin/env python2.7
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

import json
import pytest
import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
# Import the tensorflow quantizer
import aimet_common.libpymo as libpymo
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops
from aimet_tensorflow import quantsim
from aimet_tensorflow.common import tfrecord_generator as tf_gen
from aimet_tensorflow.common import graph_eval
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.utils import graph_saver
from aimet_tensorflow.examples.test_models import multiple_input_model, single_residual

tf.compat.v1.disable_eager_execution()

mnist_model_path = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'mnist/models/')
mnist_tfrecords_path = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'mnist/data/')


class Quantization(unittest.TestCase):

    @pytest.mark.cuda
    def test_gpu_quantize_mnist(self):
        """
        Running Quantize Test with GPU ops
        """
        tf.compat.v1.reset_default_graph()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Allocate the generator you wish to use to provide the network with data
        parser2 = tf_gen.MnistParser(batch_size=32, data_inputs=['reshape_input'])
        generator = tf_gen.TfRecordGenerator(tfrecords=[os.path.join(mnist_tfrecords_path, 'validation.tfrecords')],
                                             parser=parser2)

        meta_path = os.path.join(mnist_model_path, 'mnist_save.meta')
        checkpoint_path = os.path.join(mnist_model_path, 'mnist_save')
        sess = graph_saver.load_model_from_meta(meta_path, checkpoint_path)

        # Allocate the quantizer and quantize the network using the default 8 bit params/activations
        sim = quantsim.QuantizationSimModel(sess, ['reshape_input'], ['dense_1/BiasAdd'], quant_scheme='tf')

        def forward_callback(session, iterations):
            graph_eval.evaluate_graph(session, generator, ['accuracy'], graph_eval.default_eval_func, iterations)

        sim.compute_encodings(forward_callback, forward_pass_callback_args=1)

        # Try some fine-tuning
        g = sim.session.graph
        sess = sim.session
        with g.as_default():

            parser2 = tf_gen.MnistParser(batch_size=32, data_inputs=['reshape_input'])
            generator2 = tf_gen.TfRecordGenerator(tfrecords=[os.path.join(mnist_tfrecords_path, 'validation.tfrecords')],
                                                  parser=parser2)
            cross_entropy = g.get_operation_by_name('xent')
            train_step = g.get_operation_by_name("Adam")

            # do training: learn weights and architecture simultaneously
            x = sim.session.graph.get_tensor_by_name("reshape_input:0")
            y = g.get_tensor_by_name("labels:0")
            fc1_w = g.get_tensor_by_name("dense_1/MatMul/ReadVariableOp:0")

            perf = graph_eval.evaluate_graph(sess, generator2, ['accuracy'], graph_eval.default_eval_func, 1)
            print('Quantized performance: ' + str(perf * 100))

            ce = g.get_tensor_by_name("xent:0")
            ts = tf.compat.v1.train.AdamOptimizer(1e-3, name="TempAdam").minimize(ce)
            graph_eval.initialize_uninitialized_vars(sess)

            input_data = np.random.rand(32, 784)
            labels = np.random.randint(low=2, size=(32, 10))
            for i in range(20):
                _, fc1_w_value = sess.run([ts, fc1_w], feed_dict={x: input_data, y: labels})
                if i != 0:
                    assert not np.allclose(fc1_w_value, fc1_w_value_old), "Weights are not changing. Fine-tuning is not working"
                else:
                    fc1_w_value_old = fc1_w_value

                if i % 10 == 0:
                    perf = graph_eval.evaluate_graph(sess, generator2, ['accuracy'], graph_eval.default_eval_func, 1)
                    print('Quantized performance: ' + str(perf * 100))

        # close session
        sess.close()

    @pytest.mark.tf1
    @pytest.mark.cuda
    def test_gpu_quantize_resnet50(self):

        print('Running Quantize Test with GPU ops')

        tf.compat.v1.reset_default_graph()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        graph = tf.Graph()
        sess = tf.compat.v1.Session(graph=graph)
        with sess.as_default():
            with sess.graph.as_default():
                model = tf.keras.applications.resnet50.ResNet50(weights=None)
                init = tf.compat.v1.global_variables_initializer()
                sess.run(init)

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {},
            "supergroups": [
                {
                    "op_list": ["Conv", "BatchNormalization", "Relu"]
                },
                {
                    "op_list": ["Add", "Relu"]
                }
            ],
            "model_input": {},
            "model_output": {}
        }
        with open('./data/quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        conn_graph = ConnectedGraph(sess.graph, ['input_1'], ['probs/Softmax'])
        sim = quantsim.QuantizationSimModel(sess, ['input_1'], ['probs/Softmax'],
                                            config_file='./data/quantsim_config.json')

        ops_with_deactivated_output_quantizers = set()
        found_supergroup_1 = False
        found_supergroup_2 = False
        for op in conn_graph.get_all_ops().values():
            if op.type == 'Conv2D' and op.output.consumers[0].type == 'FusedBatchNormV3' and \
                    op.output.consumers[0].output.consumers[0].type == 'Relu':
                ops_with_deactivated_output_quantizers.add(op)
                ops_with_deactivated_output_quantizers.add(op.output.consumers[0])
                found_supergroup_1 = True
            elif op.type in ['Add', 'AddV2'] and op.output.consumers[0].type == 'Relu':
                ops_with_deactivated_output_quantizers.add(op)
                found_supergroup_2 = True
            elif op in get_all_input_ops(conn_graph):
                ops_with_deactivated_output_quantizers.add(op)
        ops_with_deactivated_output_quantizers_names = [op.name for op in ops_with_deactivated_output_quantizers]
        activation_quantizers = [op for op in sim.session.graph.get_operations() if op.type == 'QcQuantize' and
                                 'ReadVariableOp' not in op.name]
        param_quantizers = [op for op in sim.session.graph.get_operations() if op.type == 'QcQuantize' and
                            'ReadVariableOp' in op.name]
        for quantize_op in activation_quantizers:
            op_mode_tensor = sim.session.graph.get_tensor_by_name(quantize_op.name + '_op_mode:0')
            conn_graph_op = conn_graph.get_op_from_module_name(quantize_op.inputs[0].op.name)
            if conn_graph_op.name in ops_with_deactivated_output_quantizers_names:
                self.assertEqual(sim.session.run(op_mode_tensor), int(libpymo.TensorQuantizerOpMode.passThrough))
            else:
                self.assertEqual(sim.session.run(op_mode_tensor), int(libpymo.TensorQuantizerOpMode.updateStats))

        for quantize_op in param_quantizers:
            op_mode_tensor = sim.session.graph.get_tensor_by_name(quantize_op.name + '_op_mode:0')
            if 'BiasAdd' in quantize_op.name:
                self.assertEqual(sim.session.run(op_mode_tensor), int(libpymo.TensorQuantizerOpMode.passThrough))
            else:
                self.assertEqual(sim.session.run(op_mode_tensor),
                                 int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize))

        def dummy_forward_pass(session: tf.compat.v1.Session, args):
            out_tensor = session.graph.get_tensor_by_name(model.outputs[0].name)
            in_tensor = session.graph.get_tensor_by_name(model.inputs[0].name)
            dummy_input = np.random.rand(1, 224, 224, 3)
            session.run(out_tensor, feed_dict={in_tensor: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)

        all_quant_ops = [op for op in sim.session.graph.get_operations() if op.type == 'QcQuantize']
        self.assertEqual(286, len(all_quant_ops))
        self.assertTrue(found_supergroup_1 and found_supergroup_2)

        if os.path.exists('./data/quantsim_config.json'):
            os.remove('./data/quantsim_config.json')

    def test_cpu_quantize(self):
        """
        Running Quantize Test with CPU ops
        """

        print('Running Quantize Test with CPU ops')

        tf.compat.v1.reset_default_graph()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Allocate the generator you wish to use to provide the network with data
        parser2 = tf_gen.MnistParser(batch_size=32, data_inputs=['reshape_input'])
        generator = tf_gen.TfRecordGenerator(tfrecords=[os.path.join(mnist_tfrecords_path, 'validation.tfrecords')],
                                             parser=parser2)

        meta_path = os.path.join(mnist_model_path, 'mnist_save.meta')
        checkpoint_path = os.path.join(mnist_model_path, 'mnist_save')
        sess = graph_saver.load_model_from_meta(meta_path, checkpoint_path)

        # Allocate the quantizer and quantize the network using the default 8 bit params/activations
        sim = quantsim.QuantizationSimModel(sess, ['reshape_input'], ['dense_1/BiasAdd'], quant_scheme='tf',
                                            use_cuda=False)

        def forward_callback(session, iterations):
            graph_eval.evaluate_graph(session, generator, ['accuracy'], graph_eval.default_eval_func, iterations)

        sim.compute_encodings(forward_callback, forward_pass_callback_args=1)

        # Try some fine-tuning
        g = sim.session.graph
        sess = sim.session
        with g.as_default():

            parser2 = tf_gen.MnistParser(batch_size=32, data_inputs=['reshape_input'])
            generator2 = tf_gen.TfRecordGenerator(tfrecords=[os.path.join(mnist_tfrecords_path, 'validation.tfrecords')],
                                                  parser=parser2)
            cross_entropy = g.get_operation_by_name('xent')
            train_step = g.get_operation_by_name("Adam")

            # do training: learn weights and architecture simultaneously
            x = g.get_tensor_by_name("reshape_input:0")
            y = g.get_tensor_by_name("labels:0")

            perf = graph_eval.evaluate_graph(sess, generator2, ['accuracy'], graph_eval.default_eval_func, 1)
            print('Quantized performance: ' + str(perf * 100))

            input_data = np.random.rand(32, 784)
            labels = np.random.randint(low=2, size=(32, 10))
            for i in range(20):
                _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x: input_data, y: labels})
                if i % 10 == 0:
                    perf = graph_eval.evaluate_graph(sess, generator2, ['accuracy'], graph_eval.default_eval_func, 1)
                    print('Quantized performance: ' + str(perf * 100))

        # close session
        sess.close()

    def test_skip_bias_quantization(self):

        tf.compat.v1.reset_default_graph()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Allocate the generator you wish to use to provide the network with data
        parser2 = tf_gen.MnistParser(batch_size=32, data_inputs=['reshape_input'])
        generator = tf_gen.TfRecordGenerator(tfrecords=[os.path.join(mnist_tfrecords_path, 'validation.tfrecords')],
                                             parser=parser2)

        meta_path = os.path.join(mnist_model_path, 'mnist_save.meta')
        checkpoint_path = os.path.join(mnist_model_path, 'mnist_save')
        sess = graph_saver.load_model_from_meta(meta_path, checkpoint_path)

        # Allocate the quantizer and quantize the network using the default 8 bit params/activations
        sim = quantsim.QuantizationSimModel(sess, ['reshape_input'], ['dense_1/BiasAdd'], quant_scheme='tf',
                                            use_cuda=False)

        def forward_callback(session, iterations):
            graph_eval.evaluate_graph(session, generator, ['accuracy'], graph_eval.default_eval_func, iterations)

        sim.compute_encodings(forward_callback, forward_pass_callback_args=1)

        conv1_bias_add = sim.session.graph.get_operation_by_name('conv1/BiasAdd')
        self.assertEqual('QcQuantize', conv1_bias_add.inputs[1].op.type)
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.passThrough),
                         sim.session.run(conv1_bias_add.inputs[1].op.inputs[1]))

        conv2_bias_add = sim.session.graph.get_operation_by_name('conv2/BiasAdd')
        self.assertEqual('QcQuantize', conv2_bias_add.inputs[1].op.type)
        self.assertEqual(int(libpymo.TensorQuantizerOpMode.passThrough),
                         sim.session.run(conv2_bias_add.inputs[1].op.inputs[1]))

        # close session
        sess.close()

    def test_quantize_multi_input_mode(self):
        """ Test fill_op_product_graph() on a multiple input graph """

        tf.compat.v1.reset_default_graph()
        model_output = multiple_input_model()
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        graph_eval.initialize_uninitialized_vars(sess)

        sim = quantsim.QuantizationSimModel(sess, starting_op_names=['input1', 'input2'],
                                            output_op_names=['multiple_input_model/Softmax'],
                                            use_cuda=False)

        def forward_callback(session: tf.compat.v1.Session, iterations):
            input_tensor1 = np.random.rand(32, 10, 10, 3)
            input_tensor2 = np.random.rand(32, 12, 12, 3)

            input1 = session.graph.get_tensor_by_name('input1:0')
            input2 = session.graph.get_tensor_by_name('input2:0')
            output = session.graph.get_operation_by_name('multiple_input_model/Softmax').outputs[0]

            feed_dict = {input1: input_tensor1, input2: input_tensor2}
            session.run(output, feed_dict=feed_dict)

        sim.compute_encodings(forward_callback, forward_pass_callback_args=1)
        quant_graph = sim.session.graph
        # Checks to make sure that quantized operations exist for both input branches
        quant_graph.get_operation_by_name('conv1a/BiasAdd_quantized')
        quant_graph.get_operation_by_name('conv1b/BiasAdd_quantized')
        quant_graph.get_operation_by_name('conv2/BiasAdd_quantized')

        # close session
        sess.close()

    @pytest.mark.cuda
    def test_per_channel_quantization(self):
        print('Running Per channel Quantization Test')
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                },
                "per_channel_quantization": "True",
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with open('./quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with tf.device('/gpu:0'):
            _ = single_residual()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

        # Allocate the quantizer and quantize the network using the default 8 bit params/activations
        sim = quantsim.QuantizationSimModel(sess, ['input_1'], ['single_residual/Softmax'], quant_scheme='tf',
                                            use_cuda=True,
                                            config_file='./quantsim_config.json')

        param_quantizers = sim._param_quantizers
        for quant_op_name, quantizer_info in param_quantizers.items():

            assert len(quantizer_info.tensor_quantizer) > 1

        def forward_callback(sess, ite):
            model_output = sess.graph.get_tensor_by_name('single_residual/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(1, 16, 16, 3)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(forward_callback, forward_pass_callback_args=1)

        for quant_op_name, quantizer_info in param_quantizers.items():
            encoding = quantizer_info.get_encoding()
            assert isinstance(encoding, list)
        os.remove('./quantsim_config.json')
        # close session
        sess.close()

    @pytest.mark.cuda
    def test_per_channel_quantization_sequential_mnist_like_model(self):
        print('Running Per channel Quantization Test')

        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                    "is_symmetric": "False"
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "True"
                },
                "per_channel_quantization": "True",
            },
            "params": {
            },
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with open('./quantsim_config.json', 'w') as f:
            json.dump(quantsim_config, f)

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with tf.device('/gpu:0'):
            _ = sequential_model()
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

        # Allocate the quantizer and quantize the network using the default 8 bit params/activations
        sim = quantsim.QuantizationSimModel(sess, ['input_1'], ['dense_1/Relu'], quant_scheme='tf',
                                            use_cuda=True, config_file='./quantsim_config.json')

        param_quantizers = sim._param_quantizers
        for quant_op_name, quantizer_info in param_quantizers.items():

            assert len(quantizer_info.tensor_quantizer) > 1

        def forward_callback(sess, ite):
            model_output = sess.graph.get_tensor_by_name('dense_1/Relu:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(1, 28, 28, 1)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(forward_callback, forward_pass_callback_args=1)
        for quant_op_name, quantizer_info in param_quantizers.items():
            if quantizer_info.is_encoding_valid():
                encoding = quantizer_info.get_encoding()
                assert isinstance(encoding, list)
        os.remove('./quantsim_config.json')
        # close session
        sess.close()


def sequential_model():
    inputs = tf.keras.Input(shape=(28, 28, 1,))
    x = tf.keras.layers.Conv2D(32, (5, 5), activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    return x
