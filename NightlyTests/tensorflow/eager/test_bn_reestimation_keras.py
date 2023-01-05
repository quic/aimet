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
""" Keras bn_reestimation Nightly Tests """
import json
import tensorflow as tf
import numpy as np
from aimet_common.defs import QuantScheme
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.bn_reestimation import reestimate_bn_stats, _get_bn_submodules
from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms_to_scale


def _qsim_setup_for_fold_scale(model, dummy_inputs):
    default_config_per_channel = {
        "defaults":
            {
                "ops":
                    {
                        "is_output_quantized": "True"
                    },
                "params":
                    {
                        "is_quantized": "True",
                        "is_symmetric": "True"
                    },
                "strict_symmetric": "False",
                "unsigned_symmetric": "True",
                "per_channel_quantization": "True"
            },

        "params":
            {
                "bias":
                    {
                        "is_quantized": "False"
                    }
            },

        "op_type":
            {
                "Squeeze":
                    {
                        "is_output_quantized": "False"
                    },
                "Pad":
                    {
                        "is_output_quantized": "False"
                    },
                "Mean":
                    {
                        "is_output_quantized": "False"
                    }
            },

        "supergroups":
            [
                {
                    "op_list": ["Conv", "Relu"]
                },
                {
                    "op_list": ["Conv", "Clip"]
                },
                {
                    "op_list": ["Conv", "BatchNormalization", "Relu"]
                },
                {
                    "op_list": ["Add", "Relu"]
                },
                {
                    "op_list": ["Gemm", "Relu"]
                }
            ],

        "model_input":
            {
                "is_input_quantized": "True"
            },

        "model_output":
            {}
    }

    with open("/tmp/default_config_per_channel.json", "w") as f:
        json.dump(default_config_per_channel, f)

    qsim = QuantizationSimModel(model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                config_file="/tmp/default_config_per_channel.json")


    qsim.compute_encodings(lambda m, _: m.predict(dummy_inputs), None)

    return qsim


def _reestimate_and_compare_results(model, dataset):
    dummy_inputs = np.random.randn(1, 32, 32, 3).astype(np.float32)
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


def _fold_all_batch_norms_to_scale_and_compare_results(qsim, dummy_inputs, tolerance):
    output_before_batchnorm_folding = qsim.model(dummy_inputs)
    fold_all_batch_norms_to_scale(qsim)
    output_after_batchnorm_folding = qsim.model(dummy_inputs)
    assert np.allclose(output_before_batchnorm_folding, output_after_batchnorm_folding, atol = tolerance, equal_nan=True)


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
    dummy_inputs = np.random.randn(2, 32,32,3).astype(np.float32)
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=None, input_shape=(32, 32, 3))
    sub_model = tf.keras.Sequential()
    for layer in model.layers[0:12]:
        sub_model.add(layer)
    sub_model.build((32, 32, 3))
    qsim = _qsim_setup_for_fold_scale(sub_model, dummy_inputs)
    qsim.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError())
    _reestimate_and_compare_results(qsim.model, dataset)
    _fold_all_batch_norms_to_scale_and_compare_results(qsim, dummy_inputs, 5e-3)
