#!/usr/bin/env python3.5
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

import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# Import the tensorflow quantizer
from aimet_tensorflow.common import tfrecord_generator as tf_gen

mnist_model_path = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'mnist/models/')
mnist_tfrecords_path = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'mnist/data/')


class Generator_Example(unittest.TestCase):

    def test_generator(self):
        # create tf.compat.v1.Session and initialize the weights
        tf.compat.v1.reset_default_graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        # Allocate the generator you wish to use to provide the network with data
        parser = tf_gen.MnistParser(batch_size=100, data_inputs=['reshape_input'])
        generator = tf_gen.TfRecordGenerator(tfrecords=[os.path.join(mnist_tfrecords_path, 'test.tfrecords')],
                                             parser=parser)

        saver = tf.compat.v1.train.import_meta_graph(os.path.join(mnist_model_path, 'mnist_save.meta'))
        saver.restore(sess, os.path.join(mnist_model_path, 'mnist_save'))
        graph = tf.compat.v1.get_default_graph()

        # Get the input nodes
        data_names = generator.get_data_inputs()
        data = graph.get_tensor_by_name(data_names[0]+':0')
        label_names = generator.get_validation_inputs()
        labels = graph.get_tensor_by_name(label_names[0]+':0')

        # Get output accuracy node
        accuracy_op = graph.get_operation_by_name('accuracy')
        accuracy_output = accuracy_op.outputs[0]

        # Run the graph and verify the data is being updated properly for each iteration
        for i in range(5):
            batch = generator.next()
            if len(batch) != 2:
                raise ValueError('Got batch with '+len(batch)+' but expected 2')
            if all(x not in batch for x in ['data', 'labels']):
                ValueError('Expected "data" and "labels" in batch')

            acc_val = sess.run(accuracy_output, feed_dict={data: batch['reshape_input'], labels: batch['labels']})
            print('Accuracy: '+str(acc_val))
        sess.close()
