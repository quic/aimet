# /usr/bin/env python2.7
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
# pylint: skip-file

# start of import statements
import os
import tensorflow as tf

# Import the tensorflow quantisim
from aimet_tensorflow import quantsim
from aimet_tensorflow.common import tfrecord_generator as tf_gen
from aimet_tensorflow.common import graph_eval
from aimet_tensorflow.utils import graph_saver
from aimet_common.defs import QuantScheme
from tensorflow.examples.tutorials.mnist import input_data

# End of import statements


def pass_calibration_data(session: tf.Session):
    """
    The User of the QuantizationSimModel API is expected to write this function based on their data set.
    This is not a working function and is provided only as a guideline.

    :param session: Model's session
    :return:
    """

    # User action required
    # The following line of code is an example of how to use the ImageNet data's validation data loader.
    # Replace the following line with your own dataset's validation data loader.
    data_loader = None  # Your Dataset's data loader

    # User action required
    # For computing the activation encodings, around 1000 unlabelled data samples are required.
    # Edit the following 2 lines based on your dataloader's batch size.
    # batch_size * max_batch_counter should be 1024
    batch_size = 64
    max_batch_counter = 16

    input_tensor = None  # input tensor in session
    train_tensor = None  # train tensor in session

    current_batch_counter = 0
    for input_data, _ in data_loader:
        feed_dict = {input_tensor: input_data,
                     train_tensor: False}

        session.run([], feed_dict=feed_dict)

        current_batch_counter += 1
        if current_batch_counter == max_batch_counter:
            break



def quantization_aware_training_range_learning():
    """
    Running Quantize Range Learning Test
    """
    tf.reset_default_graph()

    # Allocate the generator you wish to use to provide the network with data
    parser2 = tf_gen.MnistParser(batch_size=100, data_inputs=['reshape_input'])
    generator = tf_gen.TfRecordGenerator(tfrecords=[os.path.join('data', 'mnist', 'validation.tfrecords')],
                                         parser=parser2)

    sess = graph_saver.load_model_from_meta('models/mnist_save.meta', 'models/mnist_save')

    # Create quantsim model to quantize the network using the default 8 bit params/activations
    # quant scheme set to range learning
    sim = quantsim.QuantizationSimModel(sess, ['reshape_input'], ['dense_1/BiasAdd'],
                                        quant_scheme=QuantScheme.training_range_learning_with_tf_init)

    # Initialize the model with encodings
    sim.compute_encodings(pass_calibration_data, forward_pass_callback_args=None)

    # Train the model to fine-tune the encodings
    g = sim.session.graph
    sess = sim.session

    with g.as_default():

        parser2 = tf_gen.MnistParser(batch_size=100, data_inputs=['reshape_input'])
        generator2 = tf_gen.TfRecordGenerator(tfrecords=['data/mnist/validation.tfrecords'], parser=parser2)
        cross_entropy = g.get_operation_by_name('xent')
        train_step = g.get_operation_by_name("Adam")

        # do training: learn weights and architecture simultaneously
        x = sim.session.graph.get_tensor_by_name("reshape_input:0")
        y = g.get_tensor_by_name("labels:0")
        fc1_w = g.get_tensor_by_name("dense_1/MatMul/ReadVariableOp:0")

        perf = graph_eval.evaluate_graph(sess, generator2, ['accuracy'], graph_eval.default_eval_func, 1)
        print('Quantized performance: ' + str(perf * 100))

        ce = g.get_tensor_by_name("xent:0")
        train_step = tf.train.AdamOptimizer(1e-3, name="TempAdam").minimize(ce)
        graph_eval.initialize_uninitialized_vars(sess)
        mnist = input_data.read_data_sets('./data', one_hot=True)

        for i in range(100):
            batch = mnist.train.next_batch(50)
            sess.run([train_step, fc1_w], feed_dict={x: batch[0], y: batch[1]})
            if i % 10 == 0:
                perf = graph_eval.evaluate_graph(sess, generator2, ['accuracy'], graph_eval.default_eval_func, 1)
                print('Quantized performance: ' + str(perf * 100))

    # close session
    sess.close()
