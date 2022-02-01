# /usr/bin/env python3.5
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
import tensorflow as tf

from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.keras.batch_norm_fold import fold_given_batch_norms
from aimet_tensorflow.keras.utils.model_transform_utils import replace_relu6_with_relu

def cross_layer_equalization_auto_stepwise():
    """
    Individual api calls to perform cross layer equalization one step at a time. Pairs to fold and
    scale are found automatically.
    1. Replace Relu6 with Relu
    2. Fold batch norms
    3. Perform cross layer scaling
    4. Perform high bias fold
    """

    # Note: Cross layer scaling and high bias fold auto stepwise functions still under development.

    # Load the model to equalize
    model = tf.keras.applications.resnet50.ResNet50(weights=None, classes=10)

    # 1. Replace Relu6 layer with Relu
    replace_relu6_with_relu(model)

    # 2. Fold all batch norms
    fold_all_batch_norms(model)


def cross_layer_equalization_manual():
    """
    Individual api calls to perform cross layer equalization one step at a time. Pairs to fold and
    scale are provided by the user.
    1. Replace Relu6 with Relu
    2. Fold batch norms
    3. Perform cross layer scaling
    4. Perform high bias fold
    """

    # Note: Cross layer scaling and high bias fold manual functions still under development.

    # Load the model to equalize
    model = tf.keras.applications.resnet50.ResNet50(weights=None, classes=10)

    # replace any ReLU6 layers with ReLU
    replace_relu6_with_relu(model)

    # pick potential pairs of conv and bn ops for fold
    layer_pairs = get_example_layer_pairs_resnet50_for_folding(model)

    # fold given layers
    fold_given_batch_norms(model, layer_pairs=layer_pairs)


def get_example_layer_pairs_resnet50_for_folding(model):
    """
    Function to pick example conv-batchnorm layer pairs for folding.
    :param model: Keras model containing conv batchnorm pairs to fold
    :return: pairs of conv and batchnorm layers for batch norm folding in Resnet50 model.
    """

    conv_op_1 = model.layers[2]
    bn_op_1 = model.layers[3]

    conv_op_2 = model.layers[7]
    bn_op_2 = model.layers[8]

    conv_op_3 = model.layers[10]
    bn_op_3 = model.layers[11]

    # make a layer pair list with potential the conv op and bn_op pair along with a flag
    # to indicate if given bn op can be folded upstream or downstream.
    # example of two pairs of conv and bn op  shown below
    layer_pairs = [(conv_op_1, bn_op_1, True),
                   (conv_op_2, bn_op_2, True),
                   (conv_op_3, bn_op_3, True)]

    return layer_pairs
