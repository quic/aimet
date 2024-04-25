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
"""GPTVQ optimizer"""

import torch
from torch import nn


_DEFAULT_GROUP_SHAPE = (32, 256)


class GPTVQOptimizer:
    """
    Optimizes the weight rounding of quantized wrapper module
    """

    # pylint: disable=too-many-locals
    @classmethod
    def _weight_update(cls, module: nn.Module, block_size: int = 128):
        """
        Optimizes the weights

        :param module: nn.Module
        :param block_size: used to process columns to perform weight update in the optimization
        """
        original_weight = module.weight
        num_rows, num_cols = original_weight.shape

        columns_per_group = _DEFAULT_GROUP_SHAPE[1]
        num_groups_per_column = num_rows // _DEFAULT_GROUP_SHAPE[0]

        vector_dim = 2
        rounded_weight = torch.zeros_like(original_weight)
        dummy_codebook = None
        for i1 in range(0, num_cols, block_size):
            i2 = min(i1 + block_size, num_cols)
            count = i2 - i1

            weight_block = original_weight[:, i1:i2].clone()
            for i in range(count):
                if (i1 + i) % columns_per_group == 0:
                    dummy_codebook = _generate_dummy_codebook(
                        num_groups_per_column, vector_dim
                    )

                if i % vector_dim == 0:
                    weight_block[:, i:i + vector_dim] = cls._update_weight_block(
                        weight_block,
                        dummy_codebook,
                        start_index=i,
                        vector_dim=vector_dim,
                        num_groups_per_column=num_groups_per_column,
                    )

            rounded_weight[:, i1:i2] = weight_block

        module.weight.data = rounded_weight.reshape(original_weight.shape)

    @staticmethod
    def _update_weight_block(
            weight_block: torch.Tensor,
            codebook: torch.Tensor,
            start_index: int,
            vector_dim: int,
            num_groups_per_column: int,
    ) -> torch.Tensor:
        """
        Update weight block using codebook

        :param weight_block: Weight block to be updated
        :param codebook: Codebook containing centroids
        :param start_index: Starting index to be used weight block slicing
        :param vector_dim: Vector dimension
        :param num_groups_per_column: Number of groups per column
        :return: Updated weight block
        """
        sliced_weight = weight_block[:, start_index:start_index + vector_dim]
        sliced_weight_shape = sliced_weight.shape
        # Before: num_rows x vector_dim -> After: num_groups_per_column x N x vector_dim
        sliced_weight = sliced_weight.reshape(num_groups_per_column, -1, vector_dim)

        indices = get_assignments(sliced_weight, codebook)
        centroids = torch.gather(
            codebook,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, vector_dim),
        )
        # Before: num_groups_per_column x N x vector_dim -> After: num_rows x vector_dim
        centroids = centroids.view(sliced_weight_shape)
        return centroids


def _generate_dummy_codebook(num_groups_per_column: int, vector_dim: int) -> torch.Tensor:
    """
    Generate dummy codebook

    :param num_groups_per_column: number of groups per column
    :param vector_dim: dimension of vector
    :return: dummy codebook
    """
    bits_per_dim = 3
    num_centroids = 2 ** (vector_dim * bits_per_dim)
    return torch.randn(num_groups_per_column, num_centroids, vector_dim)


def get_assignments(tensor: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Calculate nearest centroid index tensor

    :param tensor: num_groups_per_column x N x vector_dim
    :param centroids: num_groups_per_column x num_centroids x vector_dim
    :return: nearest centroid index tensor
    """
    centroids = centroids.unsqueeze(1)  # num_groups_per_column x 1 x num_centroids x vector_dim
    tensor = tensor.unsqueeze(2)        # num_groups_per_column x N x       1       x vector_dim
    distance = (tensor - centroids).pow(2).sum(-1)
    assignments = distance.argmin(-1)

    return assignments  # num_groups_per_column x N
