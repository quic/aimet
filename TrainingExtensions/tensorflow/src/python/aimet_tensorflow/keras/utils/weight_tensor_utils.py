# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""Weight tensor utility"""
import typing
import numpy as np
import tensorflow as tf


class WeightTensorUtils:
    """
    Utility class to handle weight tensor
    """
    @staticmethod
    def transpose_from_libpymo_to_tf_format(tensor: np.ndarray,
                                            layer: tf.keras.layers.Layer) -> np.ndarray:
        """
        Transpose the weight tensor shape from libpymo format to TensorFlow format
        """
        if isinstance(layer, (tf.keras.layers.Conv2DTranspose, tf.keras.layers.DepthwiseConv2D)):
            # libpymo shape            [out_channels, in_channels, kernel_height, kernel_width] ->
            # TF Conv2DTranspose shape [kernel_height, kernel_width, out_channels, in_channels]
            transposed_tensor = tensor.transpose((2, 3, 0, 1))
        elif isinstance(layer, tf.keras.layers.Conv2D):
            # libpymo shape            [out_channels, in_channels, kernel_height, kernel_width] ->
            # TF Conv2D shape          [kernel_height, kernel_width, in_channels, out_channels]
            transposed_tensor = tensor.transpose((2, 3, 1, 0))
        else:
            raise ValueError("Only Conv2D or it's subclass is currently supported")

        return transposed_tensor

    @staticmethod
    def transpose_from_tf_to_libpymo_format(tensor: np.ndarray,
                                            layer: tf.keras.layers.Layer) -> np.ndarray:
        """
        Transpose the weight tensor shape from TensorFlow format to libpymo format
        """
        if isinstance(layer, (tf.keras.layers.Conv2DTranspose, tf.keras.layers.DepthwiseConv2D)):
            # TF Conv2DTranspose shape [kernel_height, kernel_width, out_channels, in_channels] ->
            # libpymo shape            [out_channels, in_channels, kernel_height, kernel_width]
            transposed_tensor = tensor.transpose((2, 3, 0, 1))
        elif isinstance(layer, tf.keras.layers.Conv2D):
            # TF Conv2D shape          [kernel_height, kernel_width, in_channels, out_channels] ->
            # libpymo shape            [out_channels, in_channels, kernel_height, kernel_width]
            transposed_tensor = tensor.transpose((3, 2, 0, 1))
        else:
            raise ValueError("Only Conv2D or it's subclass is currently supported")

        return transposed_tensor

    @staticmethod
    def get_max_abs_val_per_channel(
            layer: tf.keras.layers.Conv2D, axis: typing.Tuple
    ) -> np.ndarray:
        """
        Conv2D kernel tensor shape ->
          (kernel_height, kernel_width, in_channels, out_channels)
        Conv2DTranspose kernel tensor shape ->
          (kernel_height, kernel_width, out_channels, in_channels)

        e.g.,
        _get_max_val_per_channel(conv, axis=(2, 0, 1)) means
        max values of each output channels in Conv2D
        because axis are set as (in_channels, kernel_height, kernel_width)

        _get_max_val_per_channel(conv_transpose, axis=(2, 0, 1)) means
        max values of each input channels in Conv2DTranspose
        because axis are set as (out_channels, kernel_height, kernel_width)

        :param layer: Conv2D or layer inherited from Conv2D
        :param axis: Axis or axes along which to operate
        :return: Argmax absolute value per channel
        """
        param_tensors = layer.get_weights()
        weight_tensor = param_tensors[0]
        return np.amax(np.abs(weight_tensor), axis=axis)
