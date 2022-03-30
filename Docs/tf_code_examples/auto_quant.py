# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Code example for AutoQuant """

# Step 0. Import statements
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50

from aimet_tensorflow.utils.common import iterate_tf_dataset
from aimet_tensorflow.adaround.adaround_weight import AdaroundParameters
from aimet_tensorflow.auto_quant import AutoQuant
from aimet_tensorflow.utils.graph import update_keras_bn_ops_trainable_flag

tf.compat.v1.disable_eager_execution()
# End step 0

# Step 1. Define constants and helper functions
EVAL_DATASET_SIZE = 5000
CALIBRATION_DATASET_SIZE = 2000
BATCH_SIZE = 100

_sampled_datasets = {}

def _create_sampled_dataset(dataset, num_samples):
    if num_samples in _sampled_datasets:
        return _sampled_datasets[num_samples]

    with dataset._graph.as_default():
        SHUFFLE_BUFFER_SIZE = 300 # NOTE: Adjust the buffer size as necessary.
        SHUFFLE_SEED = 22222
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=SHUFFLE_SEED)\
                         .take(num_samples)\
                         .batch(BATCH_SIZE)
        _sampled_datasets[num_samples] = dataset
        return dataset
# End step 1

# Step 2. Prepare model and dataset
input_shape = (224, 224, 3)
num_classes = 1000

model = ResNet50(weights='imagenet', input_shape=input_shape)
model = update_keras_bn_ops_trainable_flag(model, False, load_save_path='./')

input_tensor_name = model.input.name
input_op_name, _ = input_tensor_name.split(":")
output_tensor_name = model.output.name
output_op_name, _ = output_tensor_name.split(":")

# NOTE: In the actual use cases, a real dataset should provide by the users.
images = np.random.rand(100, *input_shape)
labels = np.random.randint(num_classes, size=(100,))

image_dataset = tf.compat.v1.data.Dataset.from_tensor_slices(images)\
                                         .repeat()\
                                         .take(EVAL_DATASET_SIZE)
label_dataset = tf.compat.v1.data.Dataset.from_tensor_slices(labels)\
                                         .repeat()\
                                         .take(EVAL_DATASET_SIZE)
eval_dataset = tf.compat.v1.data.Dataset.zip((image_dataset, label_dataset))
# End step 2

# Step 3. Prepare unlabeled dataset
# NOTE: In the actual use cases, the users should implement this part to serve
#       their own goals if necessary.
unlabeled_dataset = image_dataset.batch(BATCH_SIZE)
                                 
# End step 3

# Step 4. Prepare eval callback
# NOTE: In the actual use cases, the users should implement this part to serve
#       their own goals if necessary.
def eval_callback(sess: tf.compat.v1.Session,
                  num_samples: Optional[int] = None) -> float:
    if num_samples is None:
        num_samples = EVAL_DATASET_SIZE

    sampled_dataset = _create_sampled_dataset(eval_dataset, num_samples)

    with sess.graph.as_default():
        sess.run(tf.compat.v1.global_variables_initializer())
        input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
        output_tensor = sess.graph.get_tensor_by_name(output_tensor_name)

        num_correct_predictions = 0
        for images, labels in iterate_tf_dataset(sampled_dataset):
            prob = sess.run(output_tensor, feed_dict={input_tensor: images})
            predictions = np.argmax(prob, axis=1)
            num_correct_predictions += np.sum(predictions == labels)

        return int(num_correct_predictions) / num_samples
# End step 4

# Step 5. Create AutoQuant object
auto_quant = AutoQuant(allowed_accuracy_drop=0.01,
                       unlabeled_dataset=unlabeled_dataset,
                       eval_callback=eval_callback)
# End step 5

# Step 6. (Optional) Set adaround params
ADAROUND_DATASET_SIZE = 2000
adaround_dataset = _create_sampled_dataset(image_dataset, ADAROUND_DATASET_SIZE)
adaround_params = AdaroundParameters(adaround_dataset,
                                     num_batches=ADAROUND_DATASET_SIZE // BATCH_SIZE)
auto_quant.set_adaround_params(adaround_params)
# End step 6

# Step 7. Run AutoQuant
sess, accuracy, encoding_path =\
    auto_quant.apply(tf.compat.v1.keras.backend.get_session(),
                     starting_op_names=[input_op_name],
                     output_op_names=[output_op_name])
# End step 7
