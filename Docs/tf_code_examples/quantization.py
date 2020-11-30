# /usr/bin/env python2.7
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

import tensorflow as tf

# Import the tensorflow quantisim
from aimet_tensorflow import quantsim
from aimet_tensorflow.common import graph_eval
from aimet_tensorflow.utils import graph_saver
from aimet_common.defs import QuantScheme
from tensorflow.examples.tutorials.mnist import input_data


def quantize_model(generator):
    tf.compat.v1.reset_default_graph()

    # load graph
    sess = graph_saver.load_model_from_meta('models/mnist_save.meta', 'models/mnist_save')

    def forward_callback(session, iterations):
        graph_eval.evaluate_graph(session, generator, ['accuracy'], graph_eval.default_eval_func, iterations)

    # Create quantsim model to quantize the network using the default 8 bit params/activations
    sim = quantsim.QuantizationSimModel(sess, starting_op_names=['reshape_input'], output_op_names=['dense_1/BiasAdd'],
                                        quant_scheme=QuantScheme.post_training_tf_enhanced,
                                        config_file='../../../TrainingExtensions/common/src/python/aimet_common/'
                                                    'quantsim_config/default_config.json')

    # Compute encodings
    sim.compute_encodings(forward_callback, forward_pass_callback_args=1)

    # Do some fine-tuning
    training_helper(sim, generator)


def training_helper(sim, generator):
    """A Helper function to fine-tune MNIST model"""
    g = sim.session.graph
    sess = sim.session
    with g.as_default():
        x = sim.session.graph.get_tensor_by_name("reshape_input:0")
        y = g.get_tensor_by_name("labels:0")
        fc1_w = g.get_tensor_by_name("dense_1/MatMul/ReadVariableOp:0")

        ce = g.get_tensor_by_name("xent:0")
        # Using Adam optimizer
        train_step = tf.compat.v1.train.AdamOptimizer(1e-3, name="TempAdam").minimize(ce)
        graph_eval.initialize_uninitialized_vars(sess)
        # Input data for MNIST
        mnist = input_data.read_data_sets('./data', one_hot=True)

        # Using 100 iterations and batch of size 50
        for i in range(100):
            batch = mnist.train.next_batch(50)
            sess.run([train_step, fc1_w], feed_dict={x: batch[0], y: batch[1]})
            if i % 10 == 0:
                # Find accuracy of model every 10 iterations
                perf = graph_eval.evaluate_graph(sess, generator, ['accuracy'], graph_eval.default_eval_func, 1)
                print('Quantized performance: ' + str(perf * 100))

    # close session
    sess.close()

