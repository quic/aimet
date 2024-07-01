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

import numpy as np

def make_4d_tensor(tensor: np.ndarray) -> np.ndarray:
    """
    Return 4 dimensional tensor by adding a dimension if the tensor is not 4d.
    :param tensor: Input tensor.
    :return: Output tensor.
    """
    dims = len(tensor.shape)

    if dims > 4:
        raise RuntimeError

    if dims == 4:
        return tensor

    while len(tensor.shape) < 4:
        tensor = tensor[..., None]

    return tensor


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
