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

import pytest
import unittest
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import json
import time

from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils.graph_saver import load_model_from_meta
from aimet_tensorflow.quantsim import save_checkpoint, load_checkpoint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()



class TestQuantSimRnn(unittest.TestCase):

    @pytest.mark.tf1
    def test_insert_quant_op_recurrent(self):

        """ test insertion of quant ops to recurrent layer with conditional blocks """

        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            # Add an RNN layer
            x = tf.keras.layers.SimpleRNN(12)(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="simplernn_model")(x)

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        ops = sess.graph.get_operations()
        quant_op_inside_while_block_name = "simple_rnn/while/MatMul/ReadVariableOp_quantized"
        self.assertFalse(quant_op_inside_while_block_name in [op.name for op in ops])

        # construct a quantization sim model
        sim = QuantizationSimModel(sess, ['input_1'], ['simplernn_model/Softmax'], use_cuda=False)

        # get ops and make sure we have a quantized op added to the conditional block
        ops = sim.session.graph.get_operations()
        self.assertTrue(quant_op_inside_while_block_name in [op.name for op in ops])
        sim.session.close()
        del sim

    @pytest.mark.tf1
    def test_matmul_param_selection_lstm(self):
        """ Test apis to select input params to MatMuls within LSTM for quantization """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            x = tf.keras.layers.LSTM(12, name='lstm0')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="matmul0")(x)

            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            # _ = tf.compat.v1.summary.FileWriter('./lstm', sess.graph)

            matmul_with_split_inside_lstm = "lstm0/while/MatMul"
            tf_split_op_in = sess.graph.get_operation_by_name("lstm0/while/split")
            tf_matmul_with_split_inside_lstm = sess.graph.get_operation_by_name(matmul_with_split_inside_lstm)
            param_in_through_split = sess.graph.get_tensor_by_name("lstm0/while/split/ReadVariableOp:0")

            can_modify_op, param_in = QuantizationSimModel._get_op_to_modify_with_param_in(
                tf_matmul_with_split_inside_lstm, 1)

            self.assertEqual(can_modify_op, tf_split_op_in)
            self.assertEqual(param_in, param_in_through_split)

            matmul_with_slice_inside_lstm = "lstm0/while/MatMul_5"
            tf_strided_slice_op_in = sess.graph.get_operation_by_name("lstm0/while/strided_slice_1")
            tf_matmul_with_slice_inside_lstm = sess.graph.get_operation_by_name(matmul_with_slice_inside_lstm)
            param_in_through_strided_slice = sess.graph.get_tensor_by_name("lstm0/while/ReadVariableOp_1:0")

            can_modify_op, param_in = QuantizationSimModel._get_op_to_modify_with_param_in(
                tf_matmul_with_slice_inside_lstm, 1)

            self.assertEqual(can_modify_op, tf_strided_slice_op_in)
            self.assertEqual(param_in, param_in_through_strided_slice)

            sess.close()

    def _get_quant_ops_from_tf_graph(self, gr: tf.Graph):
        """
        utility to get quant op names in given graph
        :param graph: tf.Graph
        :return:
        """
        ops = gr.get_operations()
        quantized_graph_op_names = [op.name for op in ops if op.type in ["QcQuantize", "QcQuantizeRecurrentParam"]]

        return quantized_graph_op_names

    def validate_simple_rnn_auto_insertion_and_forward_pass(self, sess):
        """
        common api to validate auto quant node insertion and forward pass for simple rnn layer
        :param sess: TensorFlow session
        :return:
        """

        np.random.seed(0)
        tf.set_random_seed(0)

        ops = sess.graph.get_operations()
        matmul_param_quant_op_inside_while_block_name = "simple_rnn/while/MatMul/ReadVariableOp_quantized"
        self.assertFalse(matmul_param_quant_op_inside_while_block_name in [op.name for op in ops])
        # _ = tf.summary.FileWriter('./test_simple_rnn_keras', sess.graph)
        # construct a quantization sim model
        sim = QuantizationSimModel(sess, ['input_1'], ['simplernn_model/Softmax'], use_cuda=False)

        # params that must have quantizers
        matmul_2_param_quant_op_inside_while_block_name = "simple_rnn/while/MatMul_1/ReadVariableOp_quantized"
        # check biasadd param quantizers are disabled
        param_quantizers = sim._param_quantizers
        for p_quantizer in param_quantizers.keys():
            if 'BiasAdd' in p_quantizer:
                p_quant_config = sim.quantizer_config(p_quantizer)
                self.assertFalse(p_quant_config.enabled)

        # activations with quantizers
        activation_bias_add_op_inside_while_block_name = "simple_rnn/while/BiasAdd_quantized"
        add_op_inside_while_block_name = "simple_rnn/while/add_quantized"

        # these should not have activation quantizers
        activation_matmul_op_inside_while_block_name = "simple_rnn/while/MatMul_quantized"
        activation_matmul_2_op_inside_while_block_name = "simple_rnn/while/MatMul_1_quantized"

        # get ops and make sure we have a quantized op added to the conditional block
        quantized_graph_op_names = self._get_quant_ops_from_tf_graph(sim.session.graph)

        # while block ops
        # bias and kernel quantizers
        self.assertTrue(matmul_param_quant_op_inside_while_block_name in quantized_graph_op_names)
        self.assertTrue(matmul_2_param_quant_op_inside_while_block_name in quantized_graph_op_names)

        # output quantizers
        self.assertFalse(activation_bias_add_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(add_op_inside_while_block_name in quantized_graph_op_names)

        self.assertFalse(activation_matmul_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_2_op_inside_while_block_name in quantized_graph_op_names)

        # check for input quantizers
        input_matmul_op_inside_while_block_name = "simple_rnn/while/TensorArrayReadV3_quantized"
        input_matmul_2_op_inside_while_block_name = "simple_rnn/while/Identity_2_quantized"
        self.assertTrue(input_matmul_op_inside_while_block_name in quantized_graph_op_names)
        self.assertTrue(input_matmul_2_op_inside_while_block_name in quantized_graph_op_names)

        # validate encodings
        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('simplernn_model/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(16, 3, 100)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        def eval(sess, input_tensor):
            model_output = sess.graph.get_tensor_by_name('simplernn_model/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            out = sess.run(model_output, feed_dict={model_input: input_tensor})
            return out

        sim.compute_encodings(dummy_forward_pass, None)
        random_tensor = np.random.randn(16, 3, 100)
        orig_out = eval(sess, random_tensor)

        sim.compute_encodings(dummy_forward_pass, None)

        # check encoding min and max got updated
        with sim.session.graph.as_default():
            quantized_out = eval(sim.session, random_tensor)


        # check quantized output with orig output
        self.assertFalse(np.allclose(orig_out, quantized_out))

        # close tf sessions
        sess.close()
        sim.session.close()
        del sim

    @pytest.mark.tf1
    def test_insert_quant_op_forward_pass_simple_rnn(self):

        """ test insertion of quant ops to recurrent layer with conditional blocks """

        tf.reset_default_graph()
        np.random.seed(0)
        tf.set_random_seed(0)
        sess = tf.Session()

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer
            x = tf.keras.layers.SimpleRNN(12,
                                          kernel_initializer='glorot_uniform',
                                          recurrent_initializer='orthogonal')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="simplernn_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)

        self.validate_simple_rnn_auto_insertion_and_forward_pass(sess)
        sess.close()

    @pytest.mark.tf1
    def test_insert_quant_op_forward_pass_simple_rnn_with_relu(self):

        """ test insertion of quant ops to simple rnn with relu """

        tf.reset_default_graph()
        np.random.seed(0)
        tf.set_random_seed(0)
        sess = tf.Session()

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer
            x = tf.keras.layers.SimpleRNN(12, activation='relu',
                                          kernel_initializer='glorot_uniform',
                                          recurrent_initializer='orthogonal')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="simplernn_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)

        self.validate_simple_rnn_auto_insertion_and_forward_pass(sess)
        sess.close()

    @pytest.mark.tf1
    def test_insert_quant_op_forward_pass_simple_rnn_multiple_layers(self):

        """ test insertion of quant ops to simple rnn with multiple layes """

        tf.reset_default_graph()
        np.random.seed(0)
        tf.set_random_seed(0)
        sess = tf.Session()

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer
            x = tf.keras.layers.SimpleRNN(12, activation='tanh',
                                          kernel_initializer='glorot_uniform',
                                          recurrent_initializer='orthogonal',
                                          return_sequences=True)(inputs)
            x = tf.keras.layers.SimpleRNN(12, name='rnn1', activation='relu', return_sequences=True)(x)
            x = tf.keras.layers.SimpleRNN(12, name='rnn2', activation='tanh')(x)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="simplernn_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)

        self.validate_simple_rnn_auto_insertion_and_forward_pass(sess)
        # note - we will need to disable quantizers on identity nodes in this case

        sess.close()

    @pytest.mark.tf1
    def test_backward_pass_time_taken_simple_rnn(self, is_quantized=True, iterations=10, time_steps=1):
        """ perform backward pass with quantized simple RNN block"""

        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        batches = 16

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(1, 100))

            # Add an RNN layer with 12 internal units.
            x = tf.keras.layers.SimpleRNN(12, kernel_initializer='glorot_uniform',
                                          recurrent_initializer='orthogonal')(inputs)

            _ = tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                      name="simplernn_model")(x)

            init = tf.global_variables_initializer()
            sess.run(init)
        curr_sess = sess

        if is_quantized:
            sim = QuantizationSimModel(sess, ['input_1'], ['simplernn_model/Softmax'], use_cuda=False)

            def dummy_forward_pass(sess, args):
                model_output = sess.graph.get_tensor_by_name('simplernn_model/Softmax:0')
                model_input = sess.graph.get_tensor_by_name('input_1:0')
                dummy_input = np.random.randn(batches, 1, 100)
                sess.run(model_output, feed_dict={model_input: dummy_input})

            sim.compute_encodings(dummy_forward_pass, None)

            curr_sess = sim.session

        inp_tensor = curr_sess.graph.get_tensor_by_name('input_1:0')
        np.random.seed(0)
        w_shape = inp_tensor.shape

        inp_data = np.random.rand(batches, w_shape[1], w_shape[2])
        logits = curr_sess.graph.get_tensor_by_name('simplernn_model/MatMul:0')

        labels = np.random.randint(10, size=batches)
        one_hot_labels = np.eye(10)[labels]

        with curr_sess.graph.as_default():
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            labels_placeholder = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_placeholder, logits=logits)

            update_ops = []
            global_step = tf.train.create_global_step()
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
            gradients = optimizer.compute_gradients(loss, var_list)

            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            init = tf.group(init_global, init_local)
            curr_sess.run(init)

            grad_updates = optimizer.apply_gradients(gradients, global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)

            with tf.control_dependencies([update_op]):
                train_op = tf.identity(loss, name='train_op')

            # start training
            time_taken_by_default_grad = 0
            for i in range(iterations):
                start_time = time.perf_counter()
                _ = curr_sess.run(train_op, feed_dict={inp_tensor: inp_data, labels_placeholder: one_hot_labels})
                exec_time = time.perf_counter() - start_time
                time_taken_by_default_grad = time_taken_by_default_grad + exec_time

            default_grad_avg_time = time_taken_by_default_grad / iterations

        # close session
        sess.close()
        if is_quantized:
            sim.session.close()
            del sim

        return default_grad_avg_time

    # Keeping this disabled, this is for study purpose
    def _test_compare_simple_rnn_training_processing_time_increase(self):
        """
        Test to compare time taken by simple rnn node quantized versus no quantization
        There is no validation criterion for this test. It is only for study.
        :return:
        """

        # compare with and without quantize nodes
        itr = 1
        no_quant_simple_rnn_train_avg_time = self.test_backward_pass_time_taken_simple_rnn(is_quantized=False,
                                                                                           iterations=itr)
        quantized_simple_rnn_train_avg_time = self.test_backward_pass_time_taken_simple_rnn(is_quantized=True,
                                                                                            iterations=itr)

        print('\nquantized_simple_rnn_train_avg_time = ', quantized_simple_rnn_train_avg_time)
        print('\nno_quant_simple_rnn_train_avg_time = ', no_quant_simple_rnn_train_avg_time)
        print(' There is a ', ((quantized_simple_rnn_train_avg_time - no_quant_simple_rnn_train_avg_time)
                               / no_quant_simple_rnn_train_avg_time),
              'x increase in processing time with quant nodes in rnn block')

    def validate_internal_lstm_quantisim_nodes(self, quantized_graph_op_names, block_name='lstm',
                                               is_stacked=False, is_time_major=False):
        """
        Given a list of quantized_graph_op_names, this is utility function to validate
        the quantisim nodes are properly inserted
        :return:
        """
        # params that must have quantizers
        bias_param_quant_op_inside_while_block_name = block_name + "/while/split_1/ReadVariableOp_quantized"
        kernel_param_quant_op_inside_while_block_name = block_name + "/while/split/ReadVariableOp_quantized"
        recurrent_kenel_param_quant_op_inside_while_block_name_cp1 = block_name + "/while/ReadVariableOp_quantized"
        recurrent_kenel_param_quant_op_inside_while_block_name_cp2 = block_name + "/while/ReadVariableOp_1_quantized"
        recurrent_kenel_param_quant_op_inside_while_block_name_cp3 = block_name + "/while/ReadVariableOp_2_quantized"
        recurrent_kenel_param_quant_op_inside_while_block_name_cp4 = block_name + "/while/ReadVariableOp_3_quantized"

        # these should not have activation quantizers
        activation_matmul_op_inside_while_block_name = block_name + "/while/MatMul_quantized"
        activation_matmul_1_op_inside_while_block_name = block_name + "/while/MatMul_1_quantized"
        activation_matmul_2_op_inside_while_block_name = block_name + "/while/MatMul_2_quantized"
        activation_matmul_3_op_inside_while_block_name = block_name + "/while/MatMul_3_quantized"

        activation_matmul_4_op_inside_while_block_name = block_name + "/while/MatMul_4_quantized"
        activation_matmul_5_op_inside_while_block_name = block_name + "/while/MatMul_5_quantized"
        activation_matmul_6_op_inside_while_block_name = block_name + "/while/MatMul_6_quantized"
        activation_matmul_7_op_inside_while_block_name = block_name + "/while/MatMul_7_quantized"

        activation_bias_add_op_inside_while_block_name = block_name + "/while/BiasAdd_quantized"
        activation_bias_add_1_op_inside_while_block_name = block_name + "/while/BiasAdd_1_quantized"
        activation_bias_add_2_op_inside_while_block_name = block_name + "/while/BiasAdd_2_quantized"
        activation_bias_add_3_op_inside_while_block_name = block_name + "/while/BiasAdd_3_quantized"

        activation_add_op_inside_while_block_name = block_name + "/while/add_quantized"
        activation_add_2_op_inside_while_block_name = block_name + "/while/add_2_quantized"
        activation_add_4_op_inside_while_block_name = block_name + "/while/add_4_quantized"
        activation_add_6_op_inside_while_block_name = block_name + "/while/add_6_quantized"

        activation_mul_op_inside_while_block_name = block_name + "/while/Mul_quantized"
        activation_mul_1_op_inside_while_block_name = block_name + "/while/Mul_1_quantized"
        activation_mul_4_op_inside_while_block_name = block_name + "/while/Mul_4_quantized"

        activation_add_1_op_inside_while_block_name = block_name + "/while/Add_1_quantized"
        activation_add_3_op_inside_while_block_name = block_name + "/while/Add_3_quantized"
        activation_add_7_op_inside_while_block_name = block_name + "/while/Add_7_quantized"

        activation_mul_2_op_inside_while_block_name = block_name + "/while/mul_2_quantized"
        activation_mul_3_op_inside_while_block_name = block_name + "/while/mul_3_quantized"
        activation_mul_5_op_inside_while_block_name = block_name + "/while/mul_5_quantized"

        activation_add_5_op_inside_while_block_name = block_name + "/while/add_5_quantized"

        # while block ops
        # bias and kernel quantizers
        self.assertTrue(bias_param_quant_op_inside_while_block_name in quantized_graph_op_names)
        self.assertTrue(kernel_param_quant_op_inside_while_block_name in quantized_graph_op_names)
        self.assertTrue(recurrent_kenel_param_quant_op_inside_while_block_name_cp1 in quantized_graph_op_names)
        self.assertTrue(recurrent_kenel_param_quant_op_inside_while_block_name_cp2 in quantized_graph_op_names)
        self.assertTrue(recurrent_kenel_param_quant_op_inside_while_block_name_cp3 in quantized_graph_op_names)
        self.assertTrue(recurrent_kenel_param_quant_op_inside_while_block_name_cp4 in quantized_graph_op_names)

        # output quantizers: no activation quantizer is added for eAI
        # activations that are not quantized
        self.assertFalse(activation_matmul_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_1_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_2_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_3_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_4_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_5_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_6_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_matmul_7_op_inside_while_block_name in quantized_graph_op_names)

        self.assertFalse(activation_bias_add_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_bias_add_1_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_bias_add_2_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_bias_add_3_op_inside_while_block_name in quantized_graph_op_names)

        self.assertFalse(activation_add_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_1_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_2_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_3_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_4_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_5_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_6_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_add_7_op_inside_while_block_name in quantized_graph_op_names)

        self.assertFalse(activation_mul_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_mul_1_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_mul_2_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_mul_3_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_mul_4_op_inside_while_block_name in quantized_graph_op_names)
        self.assertFalse(activation_mul_5_op_inside_while_block_name in quantized_graph_op_names)

        # check for input quantizers
        input_x_op_inside_while_block_name = block_name + "/while/TensorArrayReadV3_quantized"
        input_h_op_inside_while_block_name = block_name + "/while/Identity_2_quantized"
        self.assertTrue(input_x_op_inside_while_block_name in quantized_graph_op_names)
        self.assertTrue(input_h_op_inside_while_block_name in quantized_graph_op_names)

        # check for input quantizer in stacked mode
        if is_stacked:
            if is_time_major:
                input_h_op_pass_to_last_lstm_name = block_name + "/TensorArrayStack/TensorArrayGatherV3_quantized"
            else:
                input_h_op_pass_to_last_lstm_name = block_name + "/transpose_1_quantized"
            self.assertTrue(input_h_op_pass_to_last_lstm_name in quantized_graph_op_names)

    def validate_general_lstm_forward_pass_and_encoding(self, sess, sim,
                                                        num_activation_quantizer=6, num_param_quantizer=8):
        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('lstm_model/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(16, 3, 100)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        def eval(sess, input_tensor):
            model_output = sess.graph.get_tensor_by_name('lstm_model/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            out = sess.run(model_output, feed_dict={model_input: input_tensor})
            return out

        sim.compute_encodings(dummy_forward_pass, None)
        random_tensor = np.random.randn(16, 3, 100)
        orig_out = eval(sess, random_tensor)

        activation_quantizers = sim._activation_quantizers
        param_quantizers = sim._param_quantizers

        # check the number of quantizers
        self.assertEqual(len(activation_quantizers), num_activation_quantizer)

        # kernel, recurrent kernelx4, bias
        # one bias and kernel of dense layer MatMul
        self.assertEqual(len(param_quantizers), num_param_quantizer)

        # Check if encodings have been calculated
        for name, quantizer in activation_quantizers.items():
            if quantizer.enabled:
                self.assertTrue(quantizer.tensor_quantizer.isEncodingValid,
                                "enabled quantizer: {} does not have a valid encoding set ".format(name))

        # check encoding min and max got updated
        with sim.session.graph.as_default():
            quantized_out = eval(sim.session, random_tensor)

        # quantized moddel output is different from orig model
        self.assertFalse(np.allclose(orig_out, quantized_out))

    @pytest.mark.tf1
    def test_quantize_lstm_default_quantsim_and_forward_pass(self):
        """ Test connected graph construction on a model with lstm op """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add a LSTM layer with 12 internal units.
            x = tf.keras.layers.LSTM(12)(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="lstm_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)
        # _ = tf.summary.FileWriter('./lstm', sess.graph)

        sim = QuantizationSimModel(sess, ['input_1'], ['lstm_model/Softmax'],
                                   use_cuda=False)

        # validate quantsim
        # get ops and make sure we have a quantized op added to the conditional block
        quantized_graph_op_names = self._get_quant_ops_from_tf_graph(sim.session.graph)

        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names)

        # validate forward pass
        self.validate_general_lstm_forward_pass_and_encoding(sess, sim)

        # close tf sessions
        sess.close()
        sim.session.close()
        del sim

    @pytest.mark.tf1
    def test_quantize_simple_rnn_export(self):
        """ Test model export for recurrent models """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            x = tf.keras.layers.SimpleRNN(10, name='rnn1', return_sequences=True)(inputs)
            x = tf.keras.layers.SimpleRNN(10, name='rnn2')(x)

            _ = tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                      name="fc")(x)

            init = tf.global_variables_initializer()
            sess.run(init)

        sim = QuantizationSimModel(sess, ['input_1'], ['fc/Softmax'],
                                   use_cuda=False)

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('fc/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(1, 3, 100)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        sim.compute_encodings(dummy_forward_pass, None)
        sim.export('./data', 'rnn_quantsim')

        new_sess = load_model_from_meta('./data/rnn_quantsim.meta')

        dummy_forward_pass(new_sess, None)

        all_op_types = [op.type for op in new_sess.graph.get_operations()]
        self.assertNotIn('QcQuantize', all_op_types)
        self.assertNotIn('QcQuantizeRecurrentParam', all_op_types)

        # Load the encodings file to check if the encodings were exported correctly
        with open("./data/rnn_quantsim.encodings", "r") as encodings_file:
            encodings = json.load(encodings_file)
            self.assertEqual(8, len(encodings['activation_encodings']))
            self.assertEqual(5, len(encodings['param_encodings']))

        # close tf sessions
        sess.close()
        sim.session.close()
        del sim

    @pytest.mark.tf1
    def test_quantize_simple_rnn_save_and_load_checkpoint(self):
        """ Test model export for recurrent models """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add an RNN layer with 12 internal units.
            x = tf.keras.layers.SimpleRNN(10, name='rnn1', return_sequences=True)(inputs)
            x = tf.keras.layers.SimpleRNN(10, name='rnn2')(x)

            _ = tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                      name="fc")(x)

            init = tf.global_variables_initializer()
            sess.run(init)

        sim = QuantizationSimModel(sess, ['input_1'], ['fc/Softmax'],
                                   use_cuda=False)

        def eval(sess, input_tensor):
            model_output = sess.graph.get_tensor_by_name('fc/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            out = sess.run(model_output, feed_dict={model_input: input_tensor})
            return out

        def dummy_forward_pass(sess, args):
            dummy_input = np.random.randn(1, 3, 100)
            eval(sess, dummy_input)

        sim.compute_encodings(dummy_forward_pass, None)
        random_tensor = np.random.randn(1, 3, 100)
        old_out = eval(sim.session, random_tensor)

        save_checkpoint(sim, './data/', 'simple_rnn_save')
        new_sim = load_checkpoint('./data', 'simple_rnn_save')

        # Check to make sure that inference through the new sim produces exactly the same output as the old sim
        # This checks that quantization parameters have been restored correctly
        # Also checks that we are able to invoke quantize-dequantize ops in the new session (so pymo objects were
        # restored correctly etc.)
        new_out = eval(new_sim.session, random_tensor)
        self.assertTrue(np.allclose(old_out, new_out))
        sim.session.close()
        del sim

    @pytest.mark.tf1
    def test_quantize_lstm_sigmoid_quantsim_and_forward_pass(self):
        """ Test connected graph construction on a model with lstm op """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add a LSTM layer with 12 internal units.
            x = tf.keras.layers.LSTM(12, recurrent_activation='sigmoid')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="lstm_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)
        # _ = tf.summary.FileWriter('./lstm', sess.graph)

        sim = QuantizationSimModel(sess, ['input_1'], ['lstm_model/Softmax'],
                                   use_cuda=False)

        # validate quantsim
        # get ops and make sure we have a quantized op added to the conditional block
        quantized_graph_op_names = self._get_quant_ops_from_tf_graph(sim.session.graph)

        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names)

        # validate forward pass
        self.validate_general_lstm_forward_pass_and_encoding(sess, sim)

        # close tf sessions
        sess.close()
        sim.session.close()
        del sim

    @pytest.mark.tf1
    def test_quantize_lstm_time_major_true_quantsim_and_forward_pass(self):
        """ Test connected graph construction on a model with lstm op """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add a LSTM layer with 12 internal units.
            x = tf.keras.layers.LSTM(12, time_major=True, name='lstm_tm')(inputs)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="lstm_model")(x)

        init = tf.global_variables_initializer()
        sess.run(init)
        # _ = tf.summary.FileWriter('./lstm', sess.graph)

        sim = QuantizationSimModel(sess, ['input_1'], ['lstm_model/Softmax'],
                                   use_cuda=False)

        # validate quantsim
        # get ops and make sure we have a quantized op added to the conditional blocks
        quantized_graph_op_names = self._get_quant_ops_from_tf_graph(sim.session.graph)

        batches = 32
        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('lstm_model/Softmax:0')
            model_input = sess.graph.get_tensor_by_name('input_1:0')
            dummy_input = np.random.randn(batches, 3, 100)
            sess.run(model_output, feed_dict={model_input: dummy_input})

        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names, 'lstm_tm')

        # validate forward pass
        self.validate_general_lstm_forward_pass_and_encoding(sess, sim)

        # close tf sessions
        sess.close()
        sim.session.close()
        del sim

    @pytest.mark.tf1
    def test_quantize_lstm_deepspeech_time_major_true_quantsim_and_forward_pass(self):
        """ Test connected graph construction on a model with lstm op """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add a LSTM layer with 12 internal units.
            x = tf.keras.layers.LSTM(12,
                                     unit_forget_bias=False,
                                     time_major=True,
                                     return_sequences=True,
                                     name='lstm_stacked')(inputs)
            x2 = tf.keras.layers.LSTM(12, name='last_lstm')(x)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="lstm_model")(x2)

        init = tf.global_variables_initializer()
        sess.run(init)
        # _ = tf.summary.FileWriter('./lstm', sess.graph)

        sim = QuantizationSimModel(sess, ['input_1'], ['lstm_model/Softmax'],
                                   use_cuda=False)

        # validate quantsim
        # get ops and make sure we have a quantized op added to the conditional block
        quantized_graph_op_names = self._get_quant_ops_from_tf_graph(sim.session.graph)

        # _ = tf.summary.FileWriter('./lstm_tm', sess.graph)
        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names, 'lstm_stacked', True, True)
        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names, 'last_lstm')

        # validate forward pass
        self.validate_general_lstm_forward_pass_and_encoding(sess, sim, 9, 14)
        self.validate_general_lstm_forward_pass_and_encoding(sess, sim, 9, 14)

        # close tf sessions
        sess.close()
        sim.session.close()
        del sim

    @pytest.mark.tf1
    def test_quantize_lstm_deepspeech_time_major_false_quantsim_and_forward_pass(self):
        """ Test connected graph construction on a model with lstm op """
        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)

        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(3, 100))

            # Add a LSTM layer with 12 internal units.
            x, state_h, state_c = tf.keras.layers.LSTM(12, return_state=True,
                                                       return_sequences=True,
                                                       name='lstm_stacked')(inputs)
            x2 = tf.keras.layers.LSTM(12, name='last_lstm')(x)
            _ = tf.keras.layers.Dense(12, activation=tf.nn.softmax,
                                      name="lstm_model")(x2)

        init = tf.global_variables_initializer()
        sess.run(init)

        sim = QuantizationSimModel(sess, ['input_1'], ['lstm_model/Softmax'],
                                   use_cuda=False)

        # validate quantsim
        # get ops and make sure we have a quantized op added to the conditional block
        quantized_graph_op_names = self._get_quant_ops_from_tf_graph(sim.session.graph)

        # _ = tf.summary.FileWriter('./lstm_tm', sess.graph)
        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names, 'lstm_stacked', True, False)
        self.validate_internal_lstm_quantisim_nodes(quantized_graph_op_names, 'last_lstm')

        # validate forward pass
        self.validate_general_lstm_forward_pass_and_encoding(sess, sim, 9, 14)
        self.validate_general_lstm_forward_pass_and_encoding(sess, sim, 9, 14)

        # close tf sessions
        sess.close()
        sim.session.close()
        del sim

    @pytest.mark.tf1
    def test_backward_pass_time_taken_lstm(self, is_quantized=True, iterations=1):
        """ perform backward pass with quantized lstm block"""

        tf.reset_default_graph()

        sess = tf.Session()
        np.random.seed(0)
        tf.set_random_seed(0)
        timesteps = 5
        with sess.graph.as_default():
            inputs = tf.keras.Input(shape=(timesteps, 100))

            # Add a lstm layer with 12 internal units.
            x = tf.keras.layers.LSTM(12)(inputs)

            _ = tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                                      name="lstm_model")(x)

            init = tf.global_variables_initializer()
            sess.run(init)
        curr_sess = sess
        if is_quantized:
            sim = QuantizationSimModel(sess, ['input_1'], ['lstm_model/Softmax'], use_cuda=False)

            def dummy_forward_pass(sess, args):
                model_output = sess.graph.get_tensor_by_name('lstm_model/Softmax:0')
                model_input = sess.graph.get_tensor_by_name('input_1:0')
                dummy_input = np.random.randn(32, 5, 100)  # time_steps = 5
                sess.run(model_output, feed_dict={model_input: dummy_input})

            sim.compute_encodings(dummy_forward_pass, None)

            curr_sess = sim.session

        inp_tensor = curr_sess.graph.get_tensor_by_name('input_1:0')
        np.random.seed(0)
        w_shape = inp_tensor.shape
        batches = 32
        inp_data = np.random.rand(batches, w_shape[1], w_shape[2])
        logits = curr_sess.graph.get_tensor_by_name('lstm_model/MatMul:0')

        labels = np.random.randint(10, size=batches)
        one_hot_labels = np.eye(10)[labels]

        with curr_sess.graph.as_default():
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            labels_placeholder = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_placeholder, logits=logits)

            update_ops = []
            global_step = tf.train.create_global_step()
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
            gradients = optimizer.compute_gradients(loss, var_list)

            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            init = tf.group(init_global, init_local)
            curr_sess.run(init)

            grad_updates = optimizer.apply_gradients(gradients, global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)

            with tf.control_dependencies([update_op]):
                train_op = tf.identity(loss, name='train_op')

            # start training
            time_taken_by_default_grad = 0
            for i in range(iterations):
                start_time = time.perf_counter()
                _ = curr_sess.run(train_op, feed_dict={inp_tensor: inp_data, labels_placeholder: one_hot_labels})
                exec_time = time.perf_counter() - start_time
                time_taken_by_default_grad = time_taken_by_default_grad + exec_time

            default_grad_avg_time = time_taken_by_default_grad / iterations

        # close session
        sess.close()
        if is_quantized:
            sim.session.close()
            del sim

        return default_grad_avg_time

    # Keeping this disabled, this is for study purpose
    def _test_compare_lstm_training_processing_time_increase(self):
        """
        Test to compare time taken by lstm node quantized versus no quantization
        There is no validation criterion for this test. It is only for study.
        :return:
        """

        # compare with and without quantize nodes
        itr = 10000
        no_quant_lstm_train_avg_time = self.test_backward_pass_time_taken_lstm(is_quantized=False,
                                                                               iterations=itr)
        quantized_lstm_train_avg_time = self.test_backward_pass_time_taken_lstm(is_quantized=True,
                                                                                iterations=itr)

        print('\nquantized_lstm_train_avg_time = ', quantized_lstm_train_avg_time)
        print('\nno_quant_lstm_train_avg_time = ', no_quant_lstm_train_avg_time)
        print(' There is a ', ((quantized_lstm_train_avg_time - no_quant_lstm_train_avg_time)
                               / no_quant_lstm_train_avg_time),
              'x increase in processing time with quant nodes in lstm block')

