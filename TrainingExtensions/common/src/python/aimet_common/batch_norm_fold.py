# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Batch normalization fold """

import math
from typing import List, Tuple, Union
import numpy as np


def expand_shape_to_4d(shape: Tuple) -> Union[List, np.ndarray]:
    """
    Expand the shape of the weight into 4d.

    :param shape:
    :return: 4d shape.
    """
    shape = list(shape)
    dims = len(shape)

    if dims > 5:
        raise RuntimeError

    if dims == 4:
        _4d_shape = shape

    else:
        if dims < 4:
            # If we have less dimensions, we add 1s to make 4 dimensions
            _4d_shape = np.append(shape, [1 for _ in range(4 - dims)]).astype(int)
        else:
            # If we have more dimensions, we concatenate all the dimensions beyond 3 into one dimension
            _4d_shape = np.array(shape[:3] + [math.prod(shape[3:])])
    return _4d_shape


def batch_norm_fold(weight: np.ndarray, bias: np.ndarray, gamma: np.ndarray, beta: np.ndarray, mu: np.ndarray,
                    sigma: np.ndarray, fold_backward: bool) -> [np.ndarray, np.ndarray]:
    """
    :param weight: conv/linear weight
    :param bias: conv/linear bias
    :param gamma: Batch Norm layer weight
    :param beta: Batch Norm layer bias
    :param mu: Batch Norm layer running mean
    :param sigma: Batch Norm layer running variance (calculated as square root of running variance)
    :param fold_backward: True if BatchNorm comes after Conv/Linear layer
    :return: Updated weight, bias
    """
    assert len(weight.shape) == 4

    assert not np.any(sigma == 0)
    scale = gamma / sigma

    if fold_backward:
        _weight = weight * scale[:, None, None, None]
        _bias = beta - (mu - bias) * scale
    else:
        _w_2d = weight.sum(3).sum(2)
        mu_hat = np.matmul(_w_2d, mu * scale)
        beta_hat = np.matmul(_w_2d, beta)
        _weight = weight * scale[None, :, None, None]
        _bias = beta_hat - mu_hat + bias
    return _weight, _bias
