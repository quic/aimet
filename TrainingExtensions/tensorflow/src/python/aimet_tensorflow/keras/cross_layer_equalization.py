# /usr/bin/env python3.5
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
"""Cross Layer Equalization"""

import collections
import typing

import tensorflow as tf
from aimet_common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.CrosslayerEqualization)

ClsSet = typing.Union[typing.Tuple[tf.keras.layers.Conv2D,
                                   tf.keras.layers.Conv2D],
                      typing.Tuple[tf.keras.layers.Conv2D,
                                   tf.keras.layers.DepthwiseConv2D,
                                   tf.keras.layers.Conv2D]]


class GraphSearchUtils:
    """Implements graph search utils required by CLE feature"""

    def __init__(self):
        pass

    @staticmethod
    def convert_layer_group_to_cls_sets(layer_group: typing.List[tf.keras.layers.Conv2D]) \
            -> typing.List[ClsSet]:
        """
        Helper function to convert a layer group to a list of cls sets
        :param layer_group: Given layer group to convert
        :return: List of cls sets
        """
        cls_sets = []

        layer_group = collections.deque(layer_group)
        prev_layer_to_scale = layer_group.popleft()
        while layer_group:
            next_layer_to_scale = layer_group.popleft()

            if isinstance(next_layer_to_scale, tf.keras.layers.DepthwiseConv2D):
                next_non_depthwise_conv_layer = layer_group.popleft()
                # DepthwiseConv layer right after DepthwiseConv layer is not currently supported
                if isinstance(next_non_depthwise_conv_layer, tf.keras.layers.DepthwiseConv2D):
                    _logger.error("Consecutive DepthwiseConv layer not currently supported")
                    raise NotImplementedError

                cls_sets.append(
                    (prev_layer_to_scale, next_layer_to_scale, next_non_depthwise_conv_layer))
                prev_layer_to_scale = next_non_depthwise_conv_layer
            else:
                cls_sets.append((prev_layer_to_scale, next_layer_to_scale))
                prev_layer_to_scale = next_layer_to_scale

        return cls_sets
