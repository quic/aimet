# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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

import aimet_tensorflow.utils.common
from aimet_tensorflow.utils.op.conv import get_strides_for_split_conv_ops, WeightTensorUtils, BiasUtils
from aimet_tensorflow.layer_database import Layer

from aimet_common.utils import AimetLogger
from aimet_common.svd_pruner import SpatialSvdPruner

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class SpatialSvdModuleSplitter:
    """ Spatial SVD module splitter"""

    @staticmethod
    def split_module(layer: Layer, rank: int) -> (tf.Operation, tf.Operation):
        """

        :param layer: Module to be split
        :param rank: rank for splitting
        :return: Two split modules
        """

        h, v = SpatialSvdModuleSplitter.get_svd_matrices(layer, rank)

        conv_a_stride, conv_b_stride = get_strides_for_split_conv_ops(op=layer.module)

        with layer.model.graph.as_default():
            # Use last_last_index to get name of conv layer prior to '/Conv2D' suffix
            last_slash_index = layer.module.name.rfind('/')
            data_format = layer.module.get_attr('data_format').decode('utf-8')
            if data_format == "NHWC":
                data_format_channels = "channels_last"
            else:
                data_format_channels = "channels_first"
            padding = layer.module.get_attr('padding').decode('utf-8')

            # Conv weight indices follow 'HWIO'
            conv_a_out = tf.keras.layers.Conv2D(filters=v.shape[3], kernel_size=(v.shape[0], v.shape[1]),
                                                strides=conv_a_stride, name=layer.module.name[:last_slash_index] + '_a',
                                                data_format=data_format_channels, padding=padding, use_bias=False)\
                (layer.module.inputs[0])

            # Update weights for conv_a
            WeightTensorUtils.update_tensor_for_op(layer.model, conv_a_out.op, v)

            # get the succeeding bias tensor
            bias_tensor = aimet_tensorflow.utils.common.get_succeeding_bias_tensor(layer.module)
            use_bias = bias_tensor is not None
            conv_b_out = tf.keras.layers.Conv2D(filters=h.shape[3], kernel_size=(h.shape[0], h.shape[1]),
                                                strides=conv_b_stride,
                                                name=layer.module.name[:last_slash_index] + '_b',
                                                data_format=data_format_channels, padding=padding, use_bias=use_bias)\
                (conv_a_out)
            if use_bias:
                # Find and write bias value into newly created bias variable
                bias_value = BiasUtils.get_bias_as_numpy_data(layer.model, layer.module)
                BiasUtils.update_bias_for_op(layer.model, conv_b_out.op.inputs[0].op, bias_value)
                conv_b_out = conv_b_out.op.inputs[0]        # set conv_b_out to be output tensor of Conv2D op

            # Update weights for conv_b
            WeightTensorUtils.update_tensor_for_op(layer.model, conv_b_out.op, h)
        return conv_a_out.op, conv_b_out.op

    @staticmethod
    def get_svd_matrices(layer: Layer, rank: int) -> (np.array, np.array):
        """
        :param layer: Module to be split
        :param rank: rank for splitting
        :return: v and h matrices after Single Value Decomposition
        """

        # get the weight parameters
        weight_tensor = layer.module.inputs[1].eval(session=layer.model)

        # Conv2d weight shape in TensorFlow  [kh, kw, Nic, Noc]
        # re order in the common shape  [Noc, Nic, kh, kw]
        weight_tensor = np.transpose(weight_tensor, (3, 2, 0, 1))

        out_channels, in_channels, height, width = weight_tensor.shape

        h, v = SpatialSvdPruner.lingalg_spatial_svd(weight_tensor, rank, in_channels, out_channels, height, width)

        # h, v matrices are in the common shape [Noc, Nic, kh, kw]
        # re order in TensorFlow Conv2d shape [kh, kw, Nic, Noc]
        h = np.transpose(h, (2, 3, 1, 0))
        v = np.transpose(v, (2, 3, 1, 0))

        return h, v
