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
"""Utility methods for working with GPTVQ"""

import torch
from torch.linalg import LinAlgError

from aimet_torch.gptvq.defs import DAMPENING_PERCENTAGE


def generate_codebook(weight_block: torch.Tensor,
                      num_of_centroids: int):
    """
    Generate and optimize codebook using K-means and return it

    :param weight_block: Weight block
    :param num_of_centroids: Number of centroids
    :return: Optimized codebook
    """
    initial_codebook = hacky_mahalanobis_init(weight_block, num_of_centroids)

    # TODO: Add K-means optimization with Hessian tensor
    return initial_codebook


def hacky_mahalanobis_init(tensor: torch.Tensor, num_of_centroids: int) -> torch.Tensor:
    """
    Initialize centroids using hacky Mahalanobis

    :param tensor: num_blocks_per_column x N x vector_dim weight tensor
    :param num_of_centroids: Number of centroids
    :return: Initialized codebook
    """
    vector_dim = tensor.shape[-1]
    mu = tensor.mean(1).unsqueeze(1)
    x_centered = tensor - mu
    sigma = torch.bmm(x_centered.transpose(1, 2), x_centered)  # num_blocks_per_column x vector_dim x vector_dim

    diag = torch.arange(sigma.shape[-1], device=sigma.device)
    damp = DAMPENING_PERCENTAGE * torch.mean(sigma[:, diag, diag].abs(), dim=-1)
    sigma[:, diag, diag] += damp[..., None]

    try:
        lambda_ = torch.linalg.inv(sigma)
    except LinAlgError:
        lambda_ = torch.zeros_like(sigma)
        lambda_[:, diag, diag] = 1.0

    dists = (torch.bmm(x_centered, lambda_) * x_centered).sum(-1)  # num_blocks_per_column x N
    sorted_dists = torch.argsort(dists, dim=1)  # num_blocks_per_column x N
    idx = torch.round(torch.linspace(0, x_centered.shape[1] - 1, num_of_centroids)).long()  # num_of_centroids

    # num_blocks_per_column x num_of_centroids --> num_blocks_per_column x num_of_centroids x 1 --> num_blocks_per_column x num_of_centroids x vector_dim
    idx = (sorted_dists[:, idx].unsqueeze(-1).expand(-1, -1, vector_dim))
    return torch.gather(tensor, dim=1, index=idx)


def get_assignments(tensor: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Calculate nearest centroid index tensor

    :param tensor: num_blocks_per_column x N x vector_dim
    :param centroids: num_blocks_per_column x num_centroids x vector_dim
    :return: nearest centroid index tensor
    """
    centroids = centroids.unsqueeze(1)  # num_blocks_per_column x 1 x num_centroids x vector_dim
    tensor = tensor.unsqueeze(2)        # num_blocks_per_column x N x       1       x vector_dim
    distance = (tensor - centroids).pow(2).sum(-1)
    assignments = distance.argmin(-1)

    return assignments  # num_blocks_per_column x N
