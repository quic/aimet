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

""" Channel Pruning functions that are common to both PyTorch and TensorFlow """

import numpy as np


def select_channels_to_prune(weight_data: np.array
                             , comp_ratio: float, num_in_channels: int) -> list:
    """
    Based on the weight date, compression ratio and the number of input channels, return the
    input channel indices to prune.

    :param weight_data: numpy array. weight data to use to select the channels to be pruned.
    :param comp_ratio: compression ratio to compress
    :param num_in_channels: the number of input channels for the module
    :return: list of input channel indices that must be pruned.
    """

    assert 0 < comp_ratio <= 1

    # Weight data is of shape [Noc, Nic, k_h, k_w]

    # Calculate squared magnitudes of weight data along all the axis except input channels (dim = 1)
    magnitudes = np.sum(np.sum(np.sum((weight_data ** 2), 3), 2), 0)

    all_indices = list(range(num_in_channels))

    # get number of input channels to keep
    keep_inp_channels = max(1, int(num_in_channels * comp_ratio))

    # get the top indices
    keep_indices = np.argpartition(magnitudes, -keep_inp_channels)[-keep_inp_channels:]

    # get input channel indices to prune
    prune_indices = list(set(all_indices) - set(keep_indices))
    return prune_indices
