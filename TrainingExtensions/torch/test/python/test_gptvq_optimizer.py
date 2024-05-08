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
"""Test GPTVQ optimizer"""

import pytest
import torch

from aimet_torch.gptvq.gptvq_optimizer import GPTVQOptimizer


class TestGPTVQOptimizer:
    @pytest.mark.parametrize("vector_dim", [1, 2, 4])
    @pytest.mark.parametrize("num_groups_per_column", [24, 12, 6])
    @pytest.mark.parametrize("min_value", [0, 16])
    def test_update_weight_block(self, vector_dim, num_groups_per_column, min_value):
        start_index = 0
        num_of_centroids = 64

        weight_block = torch.zeros(768, 128)
        original_sliced_weight = weight_block[:, start_index:start_index + vector_dim].clone()
        codebook = torch.arange(
            start=min_value,
            end=min_value + num_groups_per_column * num_of_centroids * vector_dim,
            dtype=torch.float32,
        )
        codebook = codebook.reshape(num_groups_per_column, num_of_centroids, vector_dim)

        updated_weight_block = GPTVQOptimizer._update_weight_block(
            weight_block,
            codebook,
            start_index=start_index,
            vector_dim=vector_dim,
            num_groups_per_column=num_groups_per_column,
        )

        assert updated_weight_block.shape == original_sliced_weight.shape
        assert not torch.allclose(updated_weight_block, original_sliced_weight)

        updated_weight_block = updated_weight_block.reshape(num_groups_per_column, -1, vector_dim)
        for group_index in range(num_groups_per_column):
            current_group_weight = updated_weight_block[group_index]
            corresponding_codebook = codebook[group_index]

            # Since weights is zero tensor, the rounded weight should be first vector in codebook, which is the nearest vector
            nearest_vector = corresponding_codebook[0]
            assert all([row.equal(nearest_vector) for row in current_group_weight])
