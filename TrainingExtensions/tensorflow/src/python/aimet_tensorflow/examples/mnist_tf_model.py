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
""" MNIST model using Keras API"""

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Reshape, MaxPool2D, Conv2D, Flatten, Dropout, Dense


#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
def create_model(data_format):

    """Model to recognize digits in the MNIST data set.
      Network structure is equivalent to:
      https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
      and
      https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py
      But uses the tf.keras API.
      Args:
        data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
          typically faster on GPUs while 'channels_last' is typically faster on
          CPUs. See
          https://www.tensorflow.org/performance/performance_guide#data_formats
      Returns:
        A tf.keras.Model. """

    # pylint: disable=no-member

    if data_format == 'channels_first':
        input_shape = [1, 28, 28]
    else:
        assert data_format == 'channels_last'
        input_shape = [28, 28, 1]

    return Sequential(
        [
            Reshape(target_shape=input_shape, input_shape=(28 * 28,)),
            Conv2D(32, 5, padding='same', data_format=data_format, activation=tf.nn.relu,
                   kernel_initializer='random_uniform'),
            MaxPool2D((2, 2), (2, 2), padding='same', data_format=data_format),
            Conv2D(64, 5, padding='same', data_format=data_format, activation=tf.nn.relu,
                   kernel_initializer='random_uniform'),
            MaxPool2D((2, 2), (2, 2), padding='same', data_format=data_format),
            Flatten(),
            Dense(1024, activation=tf.nn.relu, kernel_initializer='random_uniform'),
            Dropout(0.4),
            Dense(10, kernel_initializer='random_uniform')
        ])
