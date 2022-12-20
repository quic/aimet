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

import json
import unittest.mock

import tensorflow as tf
from packaging import version
import numpy as np

from aimet_tensorflow.keras.quantsim import QuantizationSimModel, QuantScheme

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


def save_config_file_bias_quantized_for_per_channel_quantization():
    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": "False"
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": "False"
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


class TestPerChannelQuantizationKeras(unittest.TestCase):
    def test_per_channel_range_learning(self):
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            tf.keras.backend.clear_session()

            inputs = tf.keras.layers.Input(shape=(32, 32, 4,))
            conv_op = tf.keras.layers.Conv2D(2, (3, 3),
                                             kernel_initializer=tf.random_uniform_initializer(-1, 2),
                                             bias_initializer='random_uniform',
                                             padding='SAME')(inputs)
            relu_op = tf.keras.layers.ReLU()(conv_op)
            reshape = tf.keras.layers.Flatten()(relu_op)
            dense = tf.keras.layers.Dense(10, bias_initializer='random_uniform')(reshape)
            model = tf.keras.Model(inputs=inputs, outputs=dense, name="conv_functional")

            save_config_file_bias_quantized_for_per_channel_quantization()

            qsim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                        default_param_bw=8, default_output_bw=8, config_file='./quantsim_config.json')

            for wrapper in qsim.quant_wrappers():
                wrapper.input_quantizers[0].disable()

            input_shape = inputs.shape.as_list()
            batches = 32

            input_data = np.random.rand(batches, input_shape[1], input_shape[2], input_shape[3])
            labels = np.random.randint(10, size=batches)
            one_hot_labels = np.eye(10)[labels]

            # model.predict(input_data)

            qsim.compute_encodings(lambda m, _: m.predict(input_data), None)
            qsim.compile(optimizer=tf.keras.optimizers.Adam(),
                         loss=tf.keras.losses.MeanSquaredError())

            _get_value = tf.keras.backend.get_value

            encoding_min_before_train = _get_value(qsim.model.layers[1].param_quantizers[0].encoding_min)
            encoding_max_before_train = _get_value(qsim.model.layers[1].param_quantizers[0].encoding_max)

            conv2d_output_encoding_min_before_train = _get_value(
                qsim.model.layers[1].output_quantizers[0]._encoding_min)
            conv2d_output_encoding_max_before_train = _get_value(
                qsim.model.layers[1].output_quantizers[0]._encoding_min)

            dense_bias_encoding_min_before_train = _get_value(qsim.model.layers[4].output_quantizers[0]._encoding_min)
            dense_bias_encoding_max_before_train = _get_value(qsim.model.layers[4].output_quantizers[0]._encoding_min)

            for _ in range(10):
                qsim.fit(input_data, one_hot_labels)

            encoding_min_after_train = _get_value(qsim.model.layers[1].param_quantizers[0].encoding_min)
            encoding_max_after_train = _get_value(qsim.model.layers[1].param_quantizers[0].encoding_max)

            conv2d_output_encoding_min_after_train = _get_value(qsim.model.layers[1].output_quantizers[0]._encoding_min)
            conv2d_output_encoding_max_after_train = _get_value(qsim.model.layers[1].output_quantizers[0]._encoding_min)

            dense_bias_encoding_min_after_train = _get_value(qsim.model.layers[4].output_quantizers[0]._encoding_min)
            dense_bias_encoding_max_after_train = _get_value(qsim.model.layers[4].output_quantizers[0]._encoding_min)

            assert not np.array_equal(encoding_min_before_train, encoding_min_after_train)
            assert not np.array_equal(encoding_max_before_train, encoding_max_after_train)
            assert not np.array_equal(conv2d_output_encoding_min_before_train, conv2d_output_encoding_min_after_train)
            assert not np.array_equal(conv2d_output_encoding_max_before_train, conv2d_output_encoding_max_after_train)
            assert not np.array_equal(dense_bias_encoding_min_before_train, dense_bias_encoding_min_after_train)
            assert not np.array_equal(dense_bias_encoding_max_before_train, dense_bias_encoding_max_after_train)
