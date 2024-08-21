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
import dataclasses
from typing import Optional, Tuple

import torch

from aimet_common.utils import AimetLogger
from aimet_torch.gptvq import utils as gptvq_utils
from aimet_torch.gptvq.defs import GPTVQParameters, DAMPENING_PERCENTAGE, BLOCK_STRIDE
from aimet_torch.gptvq.utils import (
    get_assignments,
    generate_codebook,
    fine_tune_codebook,
    quantize_dequantize_codebook,
    get_2d_tensor_shape,
)
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.quantization.affine.encoding import AffineEncoding, VectorEncoding
from aimet_torch.v2.quantization.tensor import DequantizedTensor

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class GPTVQOptimizer:
    """
    Optimizes the weight rounding of quantized wrapper module
    """

    # pylint: disable=too-many-locals, too-many-statements
    @classmethod
    def weight_update(cls, module: BaseQuantizationMixin, gptvq_params: GPTVQParameters, hessian: torch.Tensor):
        """
        Update the weights of module via GPTVQ optimization

        :param module: Concrete class object of BaseQuantizationMixin
        :param gptvq_params: Data carrier including GPTVQ parameters
        :param hessian: Hessian tensor to be used for computing GPTVQ optimization
        """
        if gptvq_utils.DO_CODEBOOK_FINE_TUNING:
            original_weight = module.weight.clone()
            original_hessian = hessian.clone()
        else:
            original_weight = module.weight
            original_hessian = hessian

        num_rows, num_cols = get_2d_tensor_shape(module)
        assert num_rows % gptvq_params.rows_per_block == 0, f"The number of rows in weight (#: {num_rows}) should be divided by rows per block (#: {gptvq_params.rows_per_block})"
        columns_per_block = gptvq_params.cols_per_block
        num_blocks_per_column = num_rows // gptvq_params.rows_per_block

        # 1-1. if any diagonal elements are zero, mark them as 1, and corresponding weight as 0
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        if isinstance(module, torch.nn.Conv2d):
            cls._convert_weight_to_2d_tensor(
                module, num_blocks_per_column, gptvq_params.rows_per_block, num_cols
            )
        module.weight[:, dead] = 0

        # 1-2. After setting dead columns to module weight, compute and overwrite parameter encoding
        module.compute_param_encodings()

        # 2. apply per channel dampening to have stability in hessian inverse computation
        damp = DAMPENING_PERCENTAGE * torch.mean(torch.diag(hessian))
        diag = torch.arange(num_cols)
        hessian[diag, diag] += damp
        hessian_inv = cls.compute_inverse(hessian)

        vector_dim = gptvq_params.vector_dim
        num_of_centroids = 2 ** gptvq_params.index_bw
        rounded_weight = module.weight.float()
        codebook = None
        codebooks = []
        assignments = []
        for block_start_idx in range(0, num_cols, BLOCK_STRIDE):
            block_end_idx = min(block_start_idx + BLOCK_STRIDE, num_cols)
            count = block_end_idx - block_start_idx

            rounded_weight_block = rounded_weight[:, block_start_idx:block_end_idx]
            error_block = torch.zeros_like(rounded_weight_block)
            hessian_inverse_block = hessian_inv[block_start_idx:block_end_idx, block_start_idx:block_end_idx]

            for i in range(count):
                if (block_start_idx + i) % columns_per_block == 0:
                    weight_block_for_codebook = rounded_weight[:, (block_start_idx + i):(block_start_idx + i + columns_per_block)]
                    weight_block_for_codebook = weight_block_for_codebook.reshape(num_blocks_per_column, -1, vector_dim)

                    hessian_diagonal = torch.diag(hessian_inv)[(block_start_idx + i):(block_start_idx + i + columns_per_block)]
                    inverse_hessian_diagonal = cls._get_inverse_hessian_diagonal(hessian_diagonal, gptvq_params)
                    codebook = generate_codebook(weight_block_for_codebook, num_of_centroids,
                                                 inverse_hessian_diagonal=inverse_hessian_diagonal,
                                                 assignment_chunk_size=gptvq_params.assignment_chunk_size,
                                                 kmeans_iteration=gptvq_params.num_of_kmeans_iterations)

                    codebook = quantize_dequantize_codebook(
                        codebook,
                        quantizer=module.param_quantizers["weight"],
                        num_blocks_per_column=num_blocks_per_column,
                    )
                    codebooks.append(codebook)
                    assignments.append([])

                if i % vector_dim == 0:
                    original_weight_chunk = rounded_weight_block[:, i:i + vector_dim]
                    diagonal = torch.diag(hessian_inverse_block)[i:i+vector_dim].unsqueeze(0)

                    if gptvq_params.vector_dim > 1 and gptvq_utils.HESSIAN_WEIGHTED_LOOKUP:
                        inverse_hessian_diagonal = 1. / diagonal
                    else:
                        inverse_hessian_diagonal = None

                    updated_weight_chunk, indices = cls._update_weight_block(
                        original_weight_chunk,
                        codebook,
                        vector_dim=vector_dim,
                        num_blocks_per_column=num_blocks_per_column,
                        inverse_hessian_diagonal=inverse_hessian_diagonal,
                        assignment_chunk_size=gptvq_params.assignment_chunk_size,
                    )
                    assignments[-1].append(indices)

                    err = (original_weight_chunk - updated_weight_chunk) / diagonal
                    update = torch.bmm(
                        err.transpose(0, 1).unsqueeze(-1),
                        hessian_inverse_block[i:i + vector_dim, i + vector_dim:].unsqueeze(1)
                    ).sum(0)
                    rounded_weight_block[:, i:i + vector_dim] = updated_weight_chunk
                    rounded_weight_block[:, i + vector_dim:] -= update
                    error_block[:, i:i + vector_dim] = err

            rounded_weight[:, block_start_idx:block_end_idx] = rounded_weight_block
            rounded_weight[:, block_end_idx:] -= error_block.matmul(hessian_inv[block_start_idx:block_end_idx, block_end_idx:])

        if gptvq_utils.DO_CODEBOOK_FINE_TUNING:
            rounded_weight = fine_tune_codebook(
                original_weight,
                original_hessian,
                codebooks,
                assignments,
                gptvq_params,
                module.param_quantizers["weight"],
            )

        qtzr = module.param_quantizers["weight"]
        rounded_weight = qtzr(
            rounded_weight.reshape(module.weight.shape).to(module.weight.dtype)
        )
        # At this point, rounded_weight is a quantized tensor with affine encoding
        # since quantizer is an affine quantizer
        assert isinstance(rounded_weight, DequantizedTensor)
        assert isinstance(rounded_weight.encoding, AffineEncoding)
        e = rounded_weight.encoding
        # NOTE: Need to revert 2D tensor shape to original weight shape such as (O, I, H, W)
        rounded_weight = rounded_weight.reshape(original_weight.shape)
        # Convert affine encoding to vector encoding
        rounded_weight.encoding = VectorEncoding(
            e.scale,
            e.offset,
            e.bitwidth,
            e.signed,
            e.symmetry,
            block_size=e.block_size,
            **dataclasses.asdict(gptvq_params),
        )
        module.weight = torch.nn.Parameter(rounded_weight)
        # Remove associated quantizer since the weight is holding already-quantized values
        module.param_quantizers["weight"] = None

    @staticmethod
    def _convert_weight_to_2d_tensor(
        module: BaseQuantizationMixin,
        num_blocks_per_column: int,
        rows_per_block: int,
        num_cols: int,
    ):
        """
        Convert module weight and corresponding quantizer to 2D tensor compatible form

        :param module: Module to convert
        :param num_blocks_per_column: Num blocks per column
        :param rows_per_block: Rows per block
        :param num_cols: Num of columns in 2D tensor
        """
        if isinstance(module, torch.nn.Conv2d):
            module.weight = torch.nn.Parameter(module.weight.flatten(1))
            original_quantizer = module.param_quantizers["weight"]
            q = QuantizeDequantize(
                shape=(num_blocks_per_column, 1),
                bitwidth=original_quantizer.bitwidth,
                symmetric=original_quantizer.symmetric,
                block_size=(rows_per_block, num_cols),
            ).to(module.weight.device)
            module.param_quantizers["weight"] = q

    @staticmethod
    def _update_weight_block(
            weight_block: torch.Tensor,
            codebook: torch.Tensor,
            vector_dim: int,
            num_blocks_per_column: int,
            inverse_hessian_diagonal: Optional[torch.Tensor] = None,
            assignment_chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update weight block using codebook

        :param weight_block: Weight block to be updated
        :param codebook: Codebook containing centroids
        :param vector_dim: Vector dimension
        :param num_blocks_per_column: Number of blocks per column
        :return: Updated weight block and corresponding indices
        """
        weight_block_shape = weight_block.shape
        # Before: num_rows x vector_dim -> After: num_blocks_per_column x N x vector_dim
        sliced_weight = weight_block.reshape(num_blocks_per_column, -1, vector_dim)

        indices = get_assignments(sliced_weight, codebook, inverse_hessian_diagonal, assignment_chunk_size)

        centroids = torch.gather(
            codebook,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, vector_dim),
        )
        # Before: num_blocks_per_column x N x vector_dim -> After: num_rows x vector_dim
        centroids = centroids.view(weight_block_shape)
        return centroids, indices

    @classmethod
    def compute_inverse(cls, hessian: torch.tensor) -> torch.Tensor:
        """
        computes the inverse of the hessian matrix using the Cholesky decomposition

        :param hessian: hessian for the module used to do weight update
        :returns the inverse of the passed hessian matrix
        """
        hessian = torch.linalg.cholesky(hessian)
        hessian = torch.cholesky_inverse(hessian)
        hessian = torch.linalg.cholesky(hessian, upper=True)
        return hessian

    @staticmethod
    def _get_inverse_hessian_diagonal(hessian_diagonal: torch.Tensor, gptvq_params: GPTVQParameters) -> Optional[torch.Tensor]:
        """
        Get manipulated inverse of Hessian diagonal tensor for codebook generation

        :param hessian_diagonal: Hessian diagonal tensor
        :param gptvq_params: GPTVQ parameters
        :return: Manipulated inverse of Hessian diagonal or None if hessian_weighted_lookup option is disabled
        """
        if gptvq_params.vector_dim > 1 and gptvq_utils.HESSIAN_WEIGHTED_LOOKUP:
            inverse_hessian_diagonal = (
                (1.0 / hessian_diagonal)
                .tile(gptvq_params.rows_per_block)
                .reshape(1, -1, gptvq_params.vector_dim)
            )
            return inverse_hessian_diagonal

        return None
