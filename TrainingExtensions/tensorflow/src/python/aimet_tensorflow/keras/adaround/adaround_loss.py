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

""" Loss function for Keras Adaround """

from typing import Tuple
import numpy as np
import tensorflow.keras.backend as K


def compute_beta(max_iter: int, cur_iter: int, beta_range: Tuple, warm_start: float) -> float:
    """
    Compute beta parameter used in regularization function using cosine decay
    :param max_iter: total maximum number of iterations
    :param cur_iter: current iteration
    :param beta_range: range for beta decay (start_beta, end_beta)
    :param warm_start: warm up period, during which rounding loss has zero effect
    :return: parameter beta
    """
    #  Start and stop beta for annealing of rounding loss (start_beta, end_beta)
    start_beta, end_beta = beta_range

    # iteration at end of warm start period, which is 20% of max iterations
    warm_start_end_iter = warm_start * max_iter

    # compute relative iteration of current iteration
    rel_iter = (cur_iter - warm_start_end_iter) / (max_iter - warm_start_end_iter)
    beta = end_beta + 0.5 * (start_beta - end_beta) * (1 + K.cos(rel_iter * np.pi))

    return beta
