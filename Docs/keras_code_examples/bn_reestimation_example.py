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

import tensorflow as tf
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0




tf.keras.backend.clear_session()
inputs = tf.keras.Input(shape=(28, 28, 1,), name="inputs")
conv = tf.keras.layers.Conv2D(16, (3, 3), name ='conv1')(inputs)
bn = tf.keras.layers.BatchNormalization(fused=True)(conv)
relu = tf.keras.layers.ReLU()(bn)
pool = tf.keras.layers.MaxPooling2D()(relu)
conv2 = tf.keras.layers.Conv2D(8, (3, 3), name ='conv2')(pool)
flatten = tf.keras.layers.Flatten()(conv2)
dense  = tf.keras.layers.Dense(10)(flatten)
functional_model = tf.keras.Model(inputs=inputs, outputs=dense)

functional_model.summary()




loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

functional_model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

functional_model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test data using `evaluate`
print("Evaluate quantized model (post QAT) on test data")
results =  functional_model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)




import json
from aimet_common.defs import QuantScheme
from aimet_tensorflow.keras.quantsim import QuantizationSimModel

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



qsim = QuantizationSimModel(functional_model, quant_scheme=QuantScheme.training_range_learning_with_tf_init, config_file="/tmp/default_config_per_channel.json")


qsim.compute_encodings(lambda m, _: m(x_test[0:100]), None)

qsim.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError())
print("Evaluate quantized model on test data")
results = qsim.model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)


quantized_callback = tf.keras.callbacks.TensorBoard(log_dir="./log/quantized")
history = qsim.model.fit(
        x_train[0:1024], y_train[0:1024], batch_size=32, epochs=1, validation_data=(x_test, y_test),
        callbacks=[quantized_callback]
    )


print("Evaluate quantized model (post QAT) on test data")
results =  qsim.model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)


import numpy as np
from aimet_tensorflow.keras.bn_reestimation import reestimate_bn_stats, _get_bn_submodules

# start BatchNorm Re-estimation
batch_size = 4
dataset = tf.data.Dataset.from_tensor_slices(x_train[0:100])
dataset = dataset.batch(batch_size=batch_size)
it = iter(dataset)
dummy_inputs = next(it)

bn_layers = _get_bn_submodules(qsim.model)
bn_mean_ori = {layer.name: layer.moving_mean.numpy() for layer in bn_layers}
bn_var_ori = {layer.name: layer.moving_variance.numpy() for layer in bn_layers}
bn_momentum_ori = {layer.name: layer.momentum for layer in bn_layers}
output_ori = qsim.model(dummy_inputs, training=False)

with reestimate_bn_stats(qsim.model, dataset, 1):
    # check re_estimation mean, var, momentum
    bn_mean_est = {layer.name: layer.moving_mean.numpy() for layer in bn_layers}
    bn_var_est = {layer.name: layer.moving_variance.numpy() for layer in bn_layers}
    bn_momentum_est = {layer.name: layer.momentum for layer in bn_layers}
    assert not all(np.allclose(bn_mean_ori[key], bn_mean_est[key]) for key in bn_mean_est)
    assert not all(np.allclose(bn_var_ori[key], bn_var_est[key]) for key in bn_var_est)
    assert not (bn_momentum_ori == bn_momentum_est)
    output_est = qsim.model(dummy_inputs, training=False)
    assert not np.allclose(output_est, output_ori)
# end BatchNorm Re-estimation

# check restored  mean, var, momentum
bn_mean_restored = {layer.name: layer.moving_mean.numpy() for layer in bn_layers}
bn_var_restored = {layer.name: layer.moving_variance.numpy() for layer in bn_layers}
bn_momentum_restored = {layer.name: layer.momentum for layer in bn_layers}

assert all(np.allclose(bn_mean_ori[key], bn_mean_restored[key]) for key in bn_mean_ori)
assert all(np.allclose(bn_var_ori[key], bn_var_restored[key]) for key in bn_var_ori)
assert (bn_momentum_ori == bn_momentum_restored)


from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms_to_scale
fold_all_batch_norms_to_scale(qsim)

import os
os.makedirs('./output/', exist_ok=True)
qsim.export(path='./output/', filename_prefix='mnist_after_bn_re_estimation_qat_range_learning')