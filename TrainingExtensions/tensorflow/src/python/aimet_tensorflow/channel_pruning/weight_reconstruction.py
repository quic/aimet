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

""" This module contains code to reconstruct weights post winnowing for the channel pruning feature """

from typing import Union, List
import numpy as np
from sklearn import linear_model

# Import aimet specific modules
from aimet_tensorflow.layer_database import Layer
import aimet_tensorflow.utils.common
from aimet_tensorflow.utils.op.conv import WeightTensorUtils, BiasUtils
from aimet_common.utils import AimetLogger
from aimet_common.winnow.winnow_utils import get_zero_positions_in_binary_mask

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ChannelPruning)


class WeightReconstructor:
    """
    Class enables weights to be reconstructed for a channel-pruned layer
    """

    @staticmethod
    def _linear_regression(input_data: np.ndarray, output_data: np.ndarray, bias: bool) \
            -> (np.ndarray, np.ndarray):

        """
        least square linear regression
        Given a matrix of input_data (X) and output_data (y), linear regression attempts to find solution vector (W)
        that will approximate y = W * X + b.

        :param input_data: input_data, in the shape of [n_samples, n_features]
        :param output_data: output_data, in the shape of [n_samples, n_targets]
        :param bias: whether to calculate the intercept for this model.

        :return: (new weight [n_samples, n_features], new bias [n_samples])
        """

        assert len(input_data.shape) == 2

        assert len(output_data.shape) == 2

        assert input_data.shape[0] == output_data.shape[0]

        # least squares Linear Regression
        # copy_X : If True, X (input_data) will be copied; else, it may be overwritten
        regressor = linear_model.LinearRegression(copy_X=True, fit_intercept=bias)

        regressor.fit(X=input_data, y=output_data)

        logger.info("finished linear regression fit ")

        # after regression, the coefficients attrib in the regressor form the new conv weights
        new_weight = regressor.coef_

        new_bias = regressor.intercept_ if bias is True else None

        return new_weight, new_bias

    @staticmethod
    def _update_layer_params(layer: Layer, new_weight: np.ndarray, new_bias: Union[np.ndarray, None]):
        """
        update parameters (weights and bias) for given layer

        :param layer: layer to be updated
        :param new_weight: newer weights
        :param new_bias: new bias
        :return:
        """
        assert layer.module.type == 'Conv2D'

        with layer.model.graph.as_default():

            # update existing weight tensor with new_weight in place
            WeightTensorUtils.update_tensor_for_op(layer.model, op=layer.module, tensor_as_numpy_array=new_weight)

            if new_bias is not None:

                # update existing bias tensor with new_bias in place
                BiasUtils.update_bias_for_op(layer.model, op=layer.module, bias_as_numpy_array=new_bias)

    @classmethod
    def reconstruct_params_for_conv2d(cls, layer: Layer, input_data: np.ndarray, output_data: np.ndarray,
                                      output_mask: List[int]):
        """
        Reconstruction of conv2d params (weights and biases) is performed using linear regression from sckit -learn.

        :param layer        : The layer to prune
        :param input_data   : input_data to the current layer, in the shape of (Ns * Nb, Nic, k_h, k_w)
        :param output_data  : output_data that match with the provided input_data should be of shape [Ns * Nb, Noc]
        :param output_mask  : output mask that specifies certain output channels to remove
        :return: new weight

        Ns = number of samples in image
        Nb = total number of images (batch size * number of batches)
        Nic, Noc = input and output channels of given layer
        k_h, k_w = kernel dimensions of given layer (height, width)
        """

        assert layer.module.type == 'Conv2D'
        assert len(input_data.shape) == 4
        assert len(output_data.shape) == 2

        assert input_data.shape[0] == output_data.shape[0]
        input_data = input_data.reshape(input_data.shape[0], -1)

        assert len(input_data.shape) == 2
        # Check that the output shape is same as number of ones in output mask
        assert layer.weight_shape[0] == sum(output_mask)

        calculate_bias = bool(aimet_tensorflow.utils.common.get_succeeding_bias_op(op=layer.module))

        # reconstruct newer weight and bias
        new_weight, new_bias = cls._linear_regression(input_data=input_data, output_data=output_data,
                                                      bias=calculate_bias)

        # reshape the new weights to common shape [Noc, Nic, kh, kw]
        _, n_ic, kh, kw = layer.weight_shape
        new_weight = new_weight.reshape(len(output_mask), n_ic, kh, kw)
        output_ch_indices_to_reduce = get_zero_positions_in_binary_mask(output_mask)
        new_weight = np.delete(new_weight, output_ch_indices_to_reduce, 0)
        if new_bias is not None:
            new_bias = np.delete(new_bias, output_ch_indices_to_reduce, 0)

        # re order in TensorFlow Conv2d shape [kh, kw, Nic, Noc]
        new_weight = np.transpose(new_weight, (2, 3, 1, 0))

        #TODO: PyLint crashes here with the error: "RecursionError: maximum recursion depth exceeded"
        cls._update_layer_params(layer=layer, new_weight=new_weight, new_bias=new_bias) # pylint: disable=all
