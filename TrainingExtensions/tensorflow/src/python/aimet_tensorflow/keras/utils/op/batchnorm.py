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

""" BN Utilities for tf 2.x """

import tensorflow as tf
import numpy as np


class BNUtils:
    """
    Batch Norm/ fused Batch Norm op related utils
    """
    @staticmethod
    def modify_bn_params_to_make_as_passthrough(bn: tf.keras.layers.BatchNormalization):
        """
        To change the batch normalization parameters to work as no-op operation

        :bn: Batch normalization layer that should be worked as passthrough op (no-op)
        """
        bn.trainable = False
        gamma = np.ones(shape=bn.gamma.shape, dtype=np.float32)
        beta = np.zeros(shape=bn.beta.shape, dtype=np.float32)
        move_mean = np.zeros(shape=bn.moving_mean.shape, dtype=np.float32)
        move_var = np.ones(shape=bn.moving_variance.shape, dtype=np.float32)

        # Note: The original gamma and beta is used for HighBiasFold, so save it separately
        original_gamma, original_beta, _, _ = bn.get_weights()
        setattr(bn, "original_gamma", original_gamma)
        setattr(bn, "original_beta", original_beta)

        bn.set_weights([gamma, beta, move_mean, move_var])
        bn.epsilon = 0
