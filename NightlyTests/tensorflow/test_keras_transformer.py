# /usr/bin/env python3.8
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
import pytest

from tensorflow import keras
import tensorflow as tf
import numpy as np

from aimet_tensorflow.keras.quantsim import QuantizationSimModel

@pytest.mark.skip("Disable tests that requires eager execution")
def test_quantizable_mha_export_backwards_pass():
    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = keras.layers.Input(shape=(maxlen,))
    # Embedding Layer
    positions = tf.range(start=0, limit=maxlen, delta=1)
    positions = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)(positions)
    x = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    x = x + positions

    # Transformer Block
    x = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = keras.layers.Dense(ff_dim, activation="relu")(x)
    x = keras.layers.Dense(embed_dim)(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Output layers
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(20, activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    functional_model = keras.Model(inputs=inputs, outputs=outputs)

    # STAGE 3 MODEL - model created using QuantSim
    quantized_model = QuantizationSimModel(functional_model)

    train_inputs = np.random.randint(1, 20000, (1024, 200))
    train_outputs = np.random.randint(0, 2, (1024,))

    val_inputs = np.random.randint(1, 20000, (256, 200))
    val_outputs = np.random.randint(0, 2, (256,))

    quantized_model.compute_encodings(lambda m, _: m(val_inputs), None)
    quantized_model.export('./data', 'pre_qat_mha')

    for wrapper in quantized_model.quant_wrappers():
        for quantizer in wrapper.input_quantizers:
            quantizer.enable()
        for quantizer in wrapper.output_quantizers:
            quantizer.enable()

    with open("./data/pre_qat_mha.encodings", "r") as encodings_file:
        pre_encodings = json.load(encodings_file)

    quantized_model.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    quantized_model.model.fit(
        train_inputs, train_outputs, batch_size=32, epochs=1, validation_data=(val_inputs, val_outputs)
    )

    quantized_model.compute_encodings(lambda m, _: m(val_inputs), None)
    quantized_model.export('./data', 'post_qat_mha')

    with open("./data/post_qat_mha.encodings", "r") as encodings_file:
        post_encodings = json.load(encodings_file)

    assert pre_encodings != post_encodings
