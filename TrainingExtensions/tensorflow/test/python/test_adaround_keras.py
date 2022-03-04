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

""" AdaRound Weights for Keras Unit Test Cases """
import pytest
pytestmark = pytest.mark.skip("Disable tests that requires eager execution")
import json
import os
import numpy as np
import tensorflow as tf

from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.keras.adaround_weight import Adaround, AdaroundParameters
from aimet_tensorflow.keras.adaround.activation_sampler import ActivationSampler

def test_apply_adaround():
    input_data = np.random.rand(32, 16, 16, 3)
    input_data = input_data.astype(dtype=np.float64)
    batch_size = 2
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)

    model = keras_model()
    params = AdaroundParameters(data_set=dataset, num_batches=2, default_num_iterations=10)

    _ = Adaround.apply_adaround(model, params, path='./data/', filename_prefix='dummy')

    # Test export functionality
    with open('./data/dummy.encodings') as json_file:
        encoding_data = json.load(json_file)

    param_keys = list(encoding_data.keys())

    assert param_keys[0] == "conv2d/kernel:0"
    assert isinstance(encoding_data["conv2d/kernel:0"], list)
    param_encoding_keys = encoding_data["conv2d/kernel:0"][0].keys()
    assert "offset" in param_encoding_keys
    assert "scale" in param_encoding_keys

    # Delete encodings file
    if os.path.exists("./data/dummy.encodings"):
        os.remove("./data/dummy.encodings")

def test_activation_sampler():
    input_data = np.random.rand(32, 16, 16, 3)
    batch_size = 2
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)

    model = keras_model()
    conv_op = model.layers[5]

    activation_sampler = ActivationSampler(dataset)
    inp_data, out_data = activation_sampler.sample_activation(conv_op, model, conv_op, model)
    assert inp_data.shape == (32, 3, 3, 8)
    assert out_data.shape == (32, 2, 2, 4)
