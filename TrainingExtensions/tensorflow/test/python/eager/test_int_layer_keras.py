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
            "per_channel_quantization": "False",
        },
        "params": {},
        "op_type": {},
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }

    with open('./quantsim_config.json', 'w') as f:
        json.dump(quantsim_config, f)


class TestIntLayerKeras(unittest.TestCase):

    def test_int_layer_pass_through(self):
        """
        Ensure pass-through for layers with unsupported input/output types like int32 and int64
        """
        if version.parse(tf.version.VERSION) >= version.parse("2.00"):
            tf.keras.backend.clear_session()

            img = tf.keras.Input(shape=(128, 256, 3), name='img')
            x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d')(img)
            x = tf.keras.layers.MaxPooling2D((2, 2), name='maxpool')(x)
            x = tf.keras.layers.Reshape(target_shape=(64, 64, 64), name='reshape')(x)
            x = tf.math.argmax(x, axis=1)  # ArgMax output type is int64
            x = tf.cast(x, tf.int32)  # Input type is int64 and output type is int32
            x = tf.keras.layers.Flatten()(x)  # Both input and output type is int32
            output = tf.keras.layers.Dense(10)(x)  # Input type is int32 but output type is float

            model = tf.keras.Model(inputs=[img], outputs=[output])
            model.summary()

            save_config_file_bias_quantized_for_per_channel_quantization()

            qsim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                        default_param_bw=8, default_output_bw=8, config_file='./quantsim_config.json')

            for wrapper in qsim.quant_wrappers():
                wrapper.input_quantizers[0].disable()

            input_shape = img.shape.as_list()
            batches = 64

            input_data = np.random.rand(batches, input_shape[1], input_shape[2], input_shape[3])
            labels = np.random.randint(10, size=batches)
            one_hot_labels = np.eye(10)[labels]

            model.predict(input_data)

            qsim.compute_encodings(lambda m, _: m.predict(input_data), None)
            qsim.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                               loss=tf.keras.losses.MeanSquaredError())

            _get_value = tf.keras.backend.get_value

            argmax_output_encoding_max_before_training = _get_value(
                qsim.model.layers[4].output_quantizers[0]._encoding_max)
            cast_input_encoding_max_before_training = _get_value(qsim.model.layers[5].input_quantizers[0]._encoding_max)
            dense_output_encoding_max_before_train = _get_value(qsim.model.layers[7].output_quantizers[0]._encoding_max)

            for _ in range(10):
                _ = qsim.model.fit(input_data, one_hot_labels)

            argmax_output_encoding_max_after_training = _get_value(
                qsim.model.layers[4].output_quantizers[0]._encoding_max)
            cast_input_encoding_max_after_training = _get_value(qsim.model.layers[5].input_quantizers[0]._encoding_max)

            dense_output_encoding_max_after_train = _get_value(qsim.model.layers[7].output_quantizers[0]._encoding_max)

            assert argmax_output_encoding_max_before_training == argmax_output_encoding_max_after_training
            assert cast_input_encoding_max_before_training == cast_input_encoding_max_after_training
            assert not dense_output_encoding_max_before_train == dense_output_encoding_max_after_train
