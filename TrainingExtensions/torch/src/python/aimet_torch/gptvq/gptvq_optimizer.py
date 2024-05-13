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

import aimet_torch.v2.quantization as Q
from aimet_torch.gptvq.defs import GPTVQParameters
from aimet_torch.gptvq.utils import get_assignments, generate_codebook


class GPTVQOptimizer:
    """
    Optimizes the weight rounding of quantized wrapper module
    """

    # pylint: disable=too-many-locals
    @classmethod
    def _weight_update(cls, module: nn.Module, gptvq_params: GPTVQParameters, block_stride: int = 128):
        """
        Optimizes the weights

        :param module: nn.Module
        :param block_stride: used to process columns to perform weight update in the optimization
        """
        original_weight = module.weight
        num_rows, num_cols = original_weight.shape

        assert num_rows % gptvq_params.rows_per_block == 0, f"The number of rows in weight (#: {num_rows}) should be divided by rows per block (#: {gptvq_params.rows_per_block})"
        columns_per_block = gptvq_params.cols_per_block
        num_blocks_per_column = num_rows // gptvq_params.rows_per_block

        vector_dim = gptvq_params.vector_dim
        num_of_centroids = 2 ** gptvq_params.index_bw
        rounded_weight = torch.zeros_like(original_weight)
        codebook = None
        for block_start_idx in range(0, num_cols, block_stride):
            block_end_idx = min(block_start_idx + block_stride, num_cols)
            count = block_end_idx - block_start_idx

            weight_block = original_weight[:, block_start_idx:block_end_idx].clone()
            for i in range(count):
                if (block_start_idx + i) % columns_per_block == 0:
                    weight_block_for_codebook = original_weight[:, (block_start_idx + i):(block_start_idx + i + columns_per_block)]
                    weight_block_for_codebook = weight_block_for_codebook.reshape(num_blocks_per_column, -1, vector_dim)
                    codebook = generate_codebook(weight_block_for_codebook, num_of_centroids)

                if i % vector_dim == 0:
                    updated_weight_block = cls._update_weight_block(
                        weight_block[:, i:i + vector_dim],
                        codebook,
                        vector_dim=vector_dim,
                        num_blocks_per_column=num_blocks_per_column,
                    )
                    qdq_weight_block = cls._quantize_dequantize_weight_block(
                        updated_weight_block,
                        quantizer=module.param_quantizers["weight"],
                        num_blocks_per_column=num_blocks_per_column,
                    )
                    # TODO: divide err by hessian
                    # pylint: disable=unused-variable
                    err = updated_weight_block - qdq_weight_block
                    weight_block[:, i:i + vector_dim] = updated_weight_block

            rounded_weight[:, block_start_idx:block_end_idx] = weight_block

        with torch.no_grad():
            module.weight.copy_(rounded_weight.reshape(original_weight.shape))

    @staticmethod
    def _update_weight_block(
            weight_block: torch.Tensor,
            codebook: torch.Tensor,
            vector_dim: int,
            num_blocks_per_column: int,
    ) -> torch.Tensor:
        """
        Update weight block using codebook

        :param weight_block: Weight block to be updated
        :param codebook: Codebook containing centroids
        :param vector_dim: Vector dimension
        :param num_blocks_per_column: Number of blocks per column
        :return: Updated weight block
        """
        weight_block_shape = weight_block.shape
        # Before: num_rows x vector_dim -> After: num_blocks_per_column x N x vector_dim
        sliced_weight = weight_block.reshape(num_blocks_per_column, -1, vector_dim)

        indices = get_assignments(sliced_weight, codebook)
        centroids = torch.gather(
            codebook,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, vector_dim),
        )
        # Before: num_blocks_per_column x N x vector_dim -> After: num_rows x vector_dim
        centroids = centroids.view(weight_block_shape)
        return centroids

    @staticmethod
    def _quantize_dequantize_weight_block(weight_block: torch.Tensor,
                                          quantizer: nn.Module,
                                          num_blocks_per_column: int) -> torch.Tensor:
        """
        Quantize-Dequantize rounded weight block

        :param weight_block: Rounded weight block
        :param quantizer: Quantizer
        :param num_blocks_per_column: Number of blocks per column
        :return: Quantize-Dequantized weight block
        """
        qdq_weight_block = Q.affine.quantize_dequantize(
            weight_block.reshape(num_blocks_per_column, -1),
            quantizer.get_scale(),
            quantizer.get_offset(),
            quantizer.bitwidth,
            quantizer.symmetric,
        )
        return qdq_weight_block.reshape(weight_block.shape)
