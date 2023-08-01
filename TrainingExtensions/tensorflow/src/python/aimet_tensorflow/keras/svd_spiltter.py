# /usr/bin/env python3.8
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

""" Implementation of layer splitting logic for spatial svd schemes """
import numpy as np
import tensorflow as tf

from aimet_tensorflow.keras.utils.op.conv import get_strides_for_split_conv_ops
from aimet_tensorflow.keras.layer_database import Layer
from aimet_tensorflow.keras.utils.common import replace_layer_in_functional_model
from aimet_tensorflow.keras.utils.weight_tensor_utils import WeightTensorUtils
from aimet_common.utils import AimetLogger
from aimet_common.svd_pruner import SpatialSvdPruner

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class SpatialSvdModuleSplitter:
    """ Spatial SVD module splitter"""

    @staticmethod
    def split_module(model: tf.keras.Model, layer: Layer, rank: int) -> (tf.keras.layers.Layer, tf.keras.layers.Layer):
        """
        :param  model: Keras Model whose layer we want to split
        :param layer: Module to be split
        :param rank: rank for splitting
        :return: Two split modules
        """

        h, v = SpatialSvdModuleSplitter.get_svd_matrices(layer, rank)

        conv_a_stride, conv_b_stride = get_strides_for_split_conv_ops(layer=layer.module)

        data_format_channels = layer.module.data_format
        padding = layer.module.padding

        conv_a = tf.keras.layers.Conv2D(filters=v.shape[3], kernel_size=(v.shape[0], v.shape[1]),
                                        strides=conv_a_stride, data_format=data_format_channels,
                                        activation=None, padding=padding,
                                        name=layer.module.name + '_a', use_bias=False)

        # get the succeeding bias tensor if present
        use_bias = False
        if len(layer.module.get_weights()) > 1:
            use_bias = True

        conv_b = tf.keras.layers.Conv2D(filters=h.shape[3], kernel_size=(h.shape[0], h.shape[1]),
                                        strides=conv_b_stride,
                                        name=layer.module.name + '_b',
                                        data_format=data_format_channels, padding=padding, use_bias=use_bias)

        # Replace the layer in the model
        replace_layer_in_functional_model(model, layer.module, [conv_a, conv_b])

        # Check if the weight shape are equal or not
        assert conv_a.get_weights()[0].shape == v.shape
        assert conv_b.get_weights()[0].shape == h.shape

        # Set the weights (kernel) for conv_a
        conv_a.set_weights([v])

        # Set the weights (kernel and bias )for conv_b
        conv_b_weight_tensor = [h]

        if use_bias:
            bias_tensor = layer.module.get_weights()[1]
            conv_b_weight_tensor.append(bias_tensor)

        conv_b.set_weights(conv_b_weight_tensor)

        return conv_a, conv_b

    @staticmethod
    def get_svd_matrices(layer: Layer, rank: int) -> (np.array, np.array):
        """
        :param layer: Module to be split
        :param rank: rank for splitting
        :return: v and h matrices after Single Value Decomposition
        """

        # get the weight parameters
        weight_tensor = layer.module.get_weights()[0]

        # Conv2d weight shape in TensorFlow  [kh, kw, Nic, Noc]
        # re order in the common shape  [Noc, Nic, kh, kw]
        weight_tensor = WeightTensorUtils.transpose_from_tf_to_libpymo_format(weight_tensor, layer.module)

        out_channels, in_channels, height, width = weight_tensor.shape

        h, v = SpatialSvdPruner.lingalg_spatial_svd(weight_tensor, rank, in_channels, out_channels, height, width)

        # h, v matrices are in the common shape [Noc, Nic, kh, kw]
        # re order in TensorFlow Conv2d shape [kh, kw, Nic, Noc]
        h = WeightTensorUtils.transpose_from_libpymo_to_tf_format(h, layer.module)
        v = WeightTensorUtils.transpose_from_libpymo_to_tf_format(v, layer.module)

        return h, v
