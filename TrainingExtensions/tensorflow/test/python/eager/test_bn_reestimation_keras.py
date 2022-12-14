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

""" bn_reestimation for Keras Unit Test Cases """

import tensorflow as tf
import numpy as np
from packaging import version
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.bn_reestimation import reestimate_bn_stats, _get_bn_submodules

def _reestimate_and_compare_results(model, dataset):
    it = iter(dataset)
    dummy_inputs = next(it)

    bn_layers = _get_bn_submodules(model)
    bn_mean_ori = {layer.name: layer.moving_mean.numpy() for layer in bn_layers}
    bn_var_ori = {layer.name: layer.moving_variance.numpy() for layer in bn_layers}
    bn_momentum_ori = {layer.name: layer.momentum for layer in bn_layers}
    output_ori = model(dummy_inputs, training=False)

    with reestimate_bn_stats(model, dataset, 1):
        # check re_estimation mean, var, momentum
        bn_mean_est = {layer.name: layer.moving_mean.numpy() for layer in bn_layers}
        bn_var_est = {layer.name: layer.moving_variance.numpy() for layer in bn_layers}
        bn_momentum_est = {layer.name: layer.momentum for layer in bn_layers}
        assert not all(np.allclose(bn_mean_ori[key], bn_mean_est[key]) for key in bn_mean_est)
        assert not all(np.allclose(bn_var_ori[key], bn_var_est[key]) for key in bn_var_est)
        assert not (bn_momentum_ori == bn_momentum_est)
        output_est = model(dummy_inputs, training=False)
        assert not np.allclose(output_est, output_ori)

    # check restored  mean, var, momentum
    bn_mean_restored = {layer.name: layer.moving_mean.numpy() for layer in bn_layers}
    bn_var_restored = {layer.name: layer.moving_variance.numpy() for layer in bn_layers}
    bn_momentum_restored = {layer.name: layer.momentum for layer in bn_layers}

    assert all(np.allclose(bn_mean_ori[key], bn_mean_restored[key]) for key in bn_mean_ori)
    assert all(np.allclose(bn_var_ori[key], bn_var_restored[key]) for key in bn_var_ori)
    assert (bn_momentum_ori == bn_momentum_restored)

    output_restored = model(dummy_inputs, training=False)
    assert np.allclose(output_restored, output_ori)


def test_bn_reestimation():
    """
    Test batchnorm reestimation
    """
    tf.keras.backend.clear_session()
    np.random.seed(0)
    input_data = np.random.randn(1024, 32,32,3).astype(np.float32)
    batch_size = 4
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)
    it = iter(dataset)
    dummy_inputs = next(it)

    inputs = tf.keras.Input(shape=(32, 32, 3,))
    bn = tf.keras.layers.BatchNormalization(fused=True, beta_initializer='random_uniform',
                                            gamma_initializer='random_uniform',
                                            moving_mean_initializer='random_uniform',
                                            moving_variance_initializer='ones') (inputs)
    model_fp32 = tf.keras.Model(inputs=inputs, outputs=bn)
    _reestimate_and_compare_results(model_fp32, dataset)

    qsim = QuantizationSimModel(model_fp32)

    qsim.compute_encodings(lambda m, _: m.predict(dummy_inputs+1), None)
    qsim.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError())
    _reestimate_and_compare_results(qsim.model, dataset)
