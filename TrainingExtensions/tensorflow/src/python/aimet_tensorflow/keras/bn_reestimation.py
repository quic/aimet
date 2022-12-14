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

"""BatchNorm Reestimation"""
from typing import List, Dict
import numpy as np
import tensorflow as tf
from aimet_common.utils import Handle, AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

def _get_bn_submodules(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    bn_layers = []
    for layer in model.submodules:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            bn_layers.append(layer)
    return bn_layers


def _reset_bn_stats(bn_layers: List[tf.keras.layers.Layer], bn_mean_checkpoints: Dict, bn_var_checkpoints: Dict, bn_momentum_checkpoints: Dict) -> Handle:
    """
    reset bn stats
    :param bn_layers: keras bn_layers
    :param bn_mean_checkpoints: Dict for original bn mean
    :param bn_var_checkpoints: Dict for original bn var
    :param bn_momentum_checkpoints: Dict for original bn momentum
    :return:
    """

    def cleanup():
        """
        Restore Bn stats
        """
        for layer in bn_layers:
            move_mean = bn_mean_checkpoints[layer.name]
            move_var = bn_var_checkpoints[layer.name]
            gamma, beta, _, _ = layer.get_weights()
            layer.set_weights([gamma, beta, move_mean, move_var])
            layer.momentum = bn_momentum_checkpoints[layer.name]

    try:
        for layer in bn_layers:
            layer.momentum = 0.0
        return Handle(cleanup)
    except:
        cleanup()
        raise ValueError('exception for reset_bn_stats')

# pylint: disable=too-many-locals
def reestimate_bn_stats(model: tf.keras.Model, bn_re_estimation_dataset: tf.data.Dataset,
                        bn_num_batches: int = 100) -> Handle:
    """
    top level api for end user directly call
    :param model: tf.keras.Model
    :param bn_re_estimation_dataset: Training dataset
    :param bn_num_batches: The number of batches to be used for reestimation
    :returns: Handle that undos the effect of BN reestimation upon handle.remove()
    """

    bn_layers = _get_bn_submodules(model)

    # save checkpoints
    bn_mean_ori = {layer.name: layer.moving_mean.numpy() for layer in bn_layers}
    bn_var_ori = {layer.name: layer.moving_variance.numpy() for layer in bn_layers}
    bn_momentum_ori = {layer.name: layer.momentum for layer in bn_layers}
    # 1. switch to re-estimation mode and setup remove
    handle = _reset_bn_stats(bn_layers, bn_mean_ori, bn_var_ori, bn_momentum_ori)

    # 2. mean &var initialization
    mean_sum_dict = {layer.name: np.zeros(layer.moving_mean.shape, dtype=layer.moving_mean.dtype.as_numpy_dtype) for layer in bn_layers}
    var_sum_dict = {layer.name: np.zeros(layer.moving_variance.shape, dtype=layer.moving_variance.dtype.as_numpy_dtype) for  layer in bn_layers}

    # 3 per batch forward for BN re-estimation, accumulate into mean&var buffers
    bn_dataset_iterator = iter(bn_re_estimation_dataset)
    for batch_index in range(bn_num_batches):
        try:
            batch_data = next(bn_dataset_iterator)
            model(batch_data, training=True)
            for layer in bn_layers:
                mean_sum_dict[layer.name] += layer.moving_mean.numpy()
                var_sum_dict[layer.name] += layer.moving_variance.numpy()
            if batch_index == bn_num_batches - 1:
                break
        except tf.errors.OutOfRangeError:
            logger.info("tf.errors.OutOfRangeError:: End of dataset.")
            break

    # 4 average mean&var buffers, Override BN stats with the reestimated stats
    for layer in bn_layers:
        move_mean = mean_sum_dict[layer.name]/bn_num_batches
        move_var = var_sum_dict[layer.name]/bn_num_batches
        gamma, beta, _, _ = layer.get_weights()
        layer.set_weights([gamma, beta, move_mean, move_var])

    return handle
