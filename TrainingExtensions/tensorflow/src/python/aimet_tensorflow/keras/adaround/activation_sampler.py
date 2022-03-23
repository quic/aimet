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
""" Sample input activation to quantized op and output activation from original op for Adaround feature """
import tensorflow as tf

class ActivationSampler:
    """
    Collect op's output activation data from unquantized model and input activation data from quantized model with
    all the preceding op's weights are quantized
    """
    def __init__(self, data_set: tf.data.Dataset, num_batches: int):
        """
        Activation sampler initializer.
        :param data_set: Data set
        :param num_batches: Number of batches of data to use during sample activation
        """
        # pylint: disable=protected-access
        self._data_set = data_set
        self._num_batches = num_batches

    def sample_activation(self, orig_module: tf.keras.layers.Layer, orig_model: tf.keras.Model,
                          quant_module: tf.keras.layers.Layer, quant_model: tf.keras.Model):
        """
        Using dataloader data, obtain inputs to orig_module and outputs of quant_module.
        :param orig_module: Module to obtain input data for
        :param orig_model: Model containing orig_module
        :param quant_module: Module to obtain output data for
        :param quant_model: Model containing quant_module
        :return: Tuple containing orig_module input data and quant_module output data
        """
        temp_orig_model = tf.keras.Model(inputs=orig_model.inputs, outputs=[orig_module.output])
        temp_quant_model = tf.keras.Model(inputs=quant_model.inputs, outputs=[quant_module.input])
        quant_input_data = temp_quant_model.predict(self._data_set, steps=self._num_batches)
        orig_output_data = temp_orig_model.predict(self._data_set, steps=self._num_batches)
        return quant_input_data, orig_output_data
