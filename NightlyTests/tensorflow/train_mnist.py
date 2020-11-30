# =============================================================================
#
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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
#
# =============================================================================
# Train a MNIST classifier and store the graph and weights.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import subprocess
import os
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def train_mnist(output_dir, script_dir='/tensorflow/tensorflow/examples/how_tos/reading_data/convert_to_records.py'):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Check if we need to generate the data. If not, return
    output_files = [os.path.join(output_dir, 'models', 'mnist_save.meta'),
                    os.path.join(output_dir, 'data', 'mnist', 'test.tfrecords'),
                    os.path.join(output_dir, 'data', 'mnist', 'train.tfrecords'),
                    os.path.join(output_dir, 'data', 'mnist', 'validation.tfrecords')]

    # if all(os.path.isfile(x) for x in output_files):
    #     print('Mnist model generation not needed')
    #     return 0
    # else:
    #     print('Mnist model generation needed')

    try:
        shutil.rmtree(os.path.join(output_dir, 'models'))
        shutil.rmtree(os.path.join(output_dir, 'data'))
    except:
        print('No model present, training mnist and copying data to %s' % output_dir)

    # Read the input data and convert to tf records, then define the network
    mnist = input_data.read_data_sets(os.path.join(output_dir, 'data'), one_hot=True, seed=0)
    subprocess.call([sys.executable, script_dir, '--directory', os.path.join(output_dir, 'data/mnist/')])
    model = sequential_model()
    x = model.input
    y_hat = model.output
    tf.add_to_collection('data', x)
    y = tf.compat.v1.placeholder(tf.float32, [None, 10], 'labels')
    tf.add_to_collection('labels', y)

    # calculate loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat), name='xent')
    tf.add_to_collection('loss', cross_entropy)
    train_step = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.add_to_collection('accuracy', acc)

    # create tf.compat.v1.Session and initialize the weights
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # do training: learn weights and architecture simultaneously
    for i in range(1000):
        batch = mnist.train.next_batch(50, shuffle=False)
        _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y: batch[1]})
        if i % 100 == 0:
            acc_val = sess.run(acc, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print('step {0:4d}, test accuracy {1:0.2f}, loss {2:0.4f}'.format(i, acc_val, loss_val))

    saver = tf.compat.v1.train.Saver()
    saver.save(sess, os.path.join(output_dir, 'models', 'mnist_save'))
    writer = tf.compat.v1.summary.FileWriter(os.path.join(output_dir, "models"), sess.graph)


def conv(x, shape, scope):
    # initialize conv weights, if not done yet
    W = tf.compat.v1.get_variable(scope + '_w', initializer=tf.random.truncated_normal(shape, stddev=0.1, seed=0))
    b = tf.compat.v1.get_variable(scope + '_b', initializer=tf.constant(0.1, shape=shape[3:]))
    # do CONV forward path
    acts = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)
    return acts


def fc(x, shape, scope):
    # initialize fc weights, if not done yet
    W = tf.compat.v1.get_variable(scope + '_w', initializer=tf.random.truncated_normal(shape, stddev=0.1, seed=0))
    b = tf.compat.v1.get_variable(scope + '_b', initializer=tf.constant(0.1, shape=shape[1:2]))
    # do FC forward path
    y = tf.matmul(x, W) + b
    return y


def sequential_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(784,), name='reshape'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, name='conv1',
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(padding='same'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, name='conv2', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    return model


if len(sys.argv) == 1:
    train_mnist('./')
else:
    train_mnist('./', sys.argv[1])
