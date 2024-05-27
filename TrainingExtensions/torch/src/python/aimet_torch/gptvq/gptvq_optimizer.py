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

import math
from functools import reduce
from typing import Optional, Callable, Any, Tuple

import psutil
import torch
from torch import nn
from torch.utils.data import Dataset

import aimet_torch.v2.quantization as Q
from aimet_common.utils import AimetLogger
from aimet_torch import utils
from aimet_torch.gptvq.activation_sampler import ActivationSampler
from aimet_torch.gptvq.defs import GPTVQParameters, DAMPENING_PERCENTAGE, BLOCK_STRIDE
from aimet_torch.gptvq.utils import get_assignments, generate_codebook
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantsim import QuantizationSimModel

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
EMPIRICAL_THRESHOLD = 3 / 4
DATA_SIZE_IN_BITS = 32
BATCH_SIZE = 1

class GPTVQOptimizer:
    """
    Optimizes the weight rounding of quantized wrapper module
    """

    # pylint: disable=too-many-locals
    @classmethod
    def gptvq_module(cls,
                     quant_module: BaseQuantizationMixin,
                     gptvq_params: GPTVQParameters,
                     sim: QuantizationSimModel,
                     forward_fn: Callable[[nn.Module, Any], Any],
                     cached_dataset: Dataset):
        """
        Run layer-wise GPTVQ optimization

        :param quant_module: Quantization module
        :param gptvq_params: Data holder containing GPTVQ parameters
        :param sim: QuantizationSimModel object
        :param forward_fn: Callable object to run forward pass
        :param cached_dataset: Cached dataset
        """
        assert isinstance(quant_module, BaseQuantizationMixin), "%s is not BaseQuantizationMixin" % quant_module
        assert quant_module.param_quantizers["weight"], "%s does not have weight quantizer" % quant_module

        original_weight = quant_module.weight
        _, num_cols = original_weight.shape
        device = original_weight.device
        hessian = torch.zeros((num_cols, num_cols)).to(device)

        # Check if we can cache intermediate activation data.
        model_inputs = cached_dataset[0]
        act_sampler = ActivationSampler(quant_module, sim.model, forward_fn)
        inp_data = act_sampler.sample_acts(model_inputs)

        use_cache_acts_data = cls._can_cache_acts_data(len(cached_dataset), inp_data.shape)
        if use_cache_acts_data:
            _logger.debug("Caching intermediate activations data for optimization.")
            all_inp_data = act_sampler.sample_and_place_all_acts_on_cpu(cached_dataset)
            # Try to put all cached activations data on GPU for faster optimization if possible.
            device = utils.get_device(quant_module)
            if 'cuda' in str(device):
                all_inp_data = cls._place_cached_acts_data(all_inp_data, device)

        # update the Hessian and the number of samples in place
        idx = 0
        n_samples = 0
        while idx < len(cached_dataset):
            end_idx = min(idx + BATCH_SIZE, len(cached_dataset))
            if use_cache_acts_data:
                inp_data = all_inp_data[idx:end_idx].to(device)
            else:
                model_inputs = cached_dataset[idx:end_idx]
                inp_data = act_sampler.sample_acts(model_inputs)
            idx = end_idx
            if len(inp_data.shape) == 1:
                inp_data = inp_data.unsqueeze(0)
            curr_batch_size = inp_data.shape[0]
            cls.update_hessian(inp_data, n_samples, curr_batch_size, hessian)
            n_samples += curr_batch_size

        with torch.no_grad():
            cls._weight_update(quant_module, gptvq_params, hessian)

    # pylint: disable=too-many-locals
    @classmethod
    def _weight_update(cls, module: BaseQuantizationMixin, gptvq_params: GPTVQParameters, hessian: torch.Tensor):
        """
        Update the weights of module via GPTVQ optimization

        :param module: Concrete class object of BaseQuantizationMixin
        :param gptvq_params: Data carrier including GPTVQ parameters
        :param hessian: Hessian tensor to be used for computing GPTVQ optimization
        """
        original_weight = module.weight.clone()
        original_weight = original_weight.float()
        num_rows, num_cols = original_weight.shape

        assert num_rows % gptvq_params.rows_per_block == 0, f"The number of rows in weight (#: {num_rows}) should be divided by rows per block (#: {gptvq_params.rows_per_block})"
        columns_per_block = gptvq_params.cols_per_block
        num_blocks_per_column = num_rows // gptvq_params.rows_per_block

        # 1) if any diagonal elements are zero, mark them as 1, and corresponding weight as 0
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        original_weight[:, dead] = 0

        # 2) apply per channel dampening to have stability in hessian inverse computation
        damp = DAMPENING_PERCENTAGE * torch.mean(torch.diag(hessian))
        diag = torch.arange(num_cols)
        hessian[diag, diag] += damp
        hessian_inv = cls.compute_inverse(hessian)

        vector_dim = gptvq_params.vector_dim
        num_of_centroids = 2 ** gptvq_params.index_bw
        rounded_weight = torch.zeros_like(original_weight)
        codebook = None
        for block_start_idx in range(0, num_cols, BLOCK_STRIDE):
            block_end_idx = min(block_start_idx + BLOCK_STRIDE, num_cols)
            count = block_end_idx - block_start_idx

            weight_block = original_weight[:, block_start_idx:block_end_idx].clone()
            rounded_weight_block = torch.zeros_like(weight_block)
            error_block = torch.zeros_like(weight_block)
            hessian_inverse_block = hessian_inv[block_start_idx:block_end_idx, block_start_idx:block_end_idx]

            for i in range(count):
                if (block_start_idx + i) % columns_per_block == 0:
                    weight_block_for_codebook = original_weight[:, (block_start_idx + i):(block_start_idx + i + columns_per_block)]
                    weight_block_for_codebook = weight_block_for_codebook.reshape(num_blocks_per_column, -1, vector_dim)
                    codebook = generate_codebook(weight_block_for_codebook, num_of_centroids,
                                                 inverse_hessian_diagonal=None,
                                                 assignment_chunk_size=gptvq_params.assignment_chunk_size,
                                                 kmeans_iteration=gptvq_params.num_of_kmeans_iterations)

                    codebook = cls._quantize_dequantize_codebook(
                        codebook,
                        quantizer=module.param_quantizers["weight"],
                        num_blocks_per_column=num_blocks_per_column,
                    )

                if i % vector_dim == 0:
                    original_weight_chunk = weight_block[:, i:i + vector_dim]
                    diagonal = torch.diag(hessian_inverse_block)[i:i+vector_dim].unsqueeze(0)

                    updated_weight_chunk = cls._update_weight_block(
                        original_weight_chunk,
                        codebook,
                        vector_dim=vector_dim,
                        num_blocks_per_column=num_blocks_per_column,
                        inverse_hessian_diagonal=None,
                        assignment_chunk_size=gptvq_params.assignment_chunk_size,
                    )

                    err = (original_weight_chunk - updated_weight_chunk) / diagonal
                    update = torch.bmm(
                        err.transpose(0, 1).unsqueeze(-1),
                        hessian_inverse_block[i:i + vector_dim, i + vector_dim:].unsqueeze(1)
                    ).sum(0)
                    rounded_weight_block[:, i:i + vector_dim] = updated_weight_chunk
                    weight_block[:, i + vector_dim:] -= update
                    error_block[:, i:i + vector_dim] = err

            rounded_weight[:, block_start_idx:block_end_idx] = rounded_weight_block
            original_weight[:, block_end_idx:] -= error_block.matmul(hessian_inv[block_start_idx:block_end_idx, block_end_idx:])

        with torch.no_grad():
            module.weight.copy_(rounded_weight.reshape(module.weight.shape).to(module.weight.dtype))

    @staticmethod
    def _update_weight_block(
            weight_block: torch.Tensor,
            codebook: torch.Tensor,
            vector_dim: int,
            num_blocks_per_column: int,
            inverse_hessian_diagonal: Optional[torch.Tensor] = None,
            assignment_chunk_size: Optional[int] = None,
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

        indices = get_assignments(sliced_weight, codebook, inverse_hessian_diagonal, assignment_chunk_size)

        centroids = torch.gather(
            codebook,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, vector_dim),
        )
        # Before: num_blocks_per_column x N x vector_dim -> After: num_rows x vector_dim
        centroids = centroids.view(weight_block_shape)
        return centroids

    @staticmethod
    def _quantize_dequantize_codebook(codebook: torch.Tensor,
                                      quantizer: nn.Module,
                                      num_blocks_per_column: int) -> torch.Tensor:
        """
        Quantize-Dequantize codebook

        :param codebook: Codebook
        :param quantizer: Quantizer
        :param num_blocks_per_column: Number of blocks per column
        :return: Quantize-Dequantized codebook
        """
        qdq_codebook = Q.affine.quantize_dequantize(
            codebook.reshape(num_blocks_per_column, -1),
            quantizer.get_scale(),
            quantizer.get_offset(),
            quantizer.bitwidth,
            quantizer.symmetric,
        )
        return qdq_codebook.reshape(codebook.shape)

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

    @classmethod
    def update_hessian(cls, inp: torch.tensor, n_samples: int, curr_batch_size: int, hessian: torch.tensor):
        """
        Updates the hessian matrix using the passed input data to the module and applies scaling

        :param inp: activation input passed to the given module
        :param hessian: hessian for the module used to do weight update
        :param n_samples: samples seen so far for hessian computation
        :param curr_batch_size: batch size of current input
        """
        #the hessian is of the shape [weight.shape[1], weight.shape[1]], i.e C*C columns
        # THIS SHOULD WORK FOR ALL THE DIMENSIONS, it makes the last dimension match the weight's column dimension, and first one as the reshaped/ adjusted sample size
        inp = inp.reshape((-1, inp.shape[-1]))

        ## we calculate the transpose of input to compute the Hessian in accordance with the weight shape
        inp = inp.T

        # scale the hessian matrix in place
        hessian *= n_samples / (n_samples + curr_batch_size)
        inp = math.sqrt(2 / (n_samples + curr_batch_size)) * inp.float()
        # update the hessian in place
        hessian += inp.matmul(inp.T)

    @staticmethod
    def _can_cache_acts_data(num_batches: int, input_shape: torch.Size) -> bool:
        """
        Function to check whether activations data can be cached and fit in CPU memory for given
        input and output shape in advance. The threshold CPU memory is determined by multiplying threshold and
        available CPU memory so that remaining CPU memory is available for other processes.

        NOTE: The threshold value is empirically chosen. Threshold ensures the safety from OOM for remaining run.

        :param num_batches: Number of batches.
        :param input_shape: Shape of input activations data.
        :return: True if we can cache, false otherwise.
        """
        can_cache_data = False

        # Available CPU memory in GB.
        threshold_mem = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        threshold_mem = threshold_mem * EMPIRICAL_THRESHOLD

        # required CPU memory in GB.
        req_mem = 0
        req_mem += reduce(lambda x, y: x * y, input_shape) * num_batches * DATA_SIZE_IN_BITS / (1024 * 1024 * 1024 * 8)

        if req_mem < threshold_mem:
            can_cache_data = True

        return can_cache_data

    @staticmethod
    def _place_cached_acts_data(inp_data: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function decides whether cached activation data can be placed on device or not. If yes, it puts
        cached activation data to given device. If there is not enough device memory, it keeps the
        cached activation data to CPU memory.

        NOTE: The threshold value is empirically chosen. Threshold ensures the safety from OOM for remaining run.

        :param inp_data: Input activations data.
        :param out_data: Output activations data.
        :param device: Device.
        :return: Input and output activations data.
        """
        torch.cuda.empty_cache()

        # Available GPU memory in GB
        threshold_mem = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
        threshold_mem = threshold_mem / (1024 * 1024 * 1024)
        threshold_mem = threshold_mem * EMPIRICAL_THRESHOLD

        # required GPU memory in GB
        req_mem = 0
        req_mem += reduce(lambda x, y: x * y, inp_data.size()) * DATA_SIZE_IN_BITS / (1024 * 1024 * 1024 * 8)

        if req_mem < threshold_mem:
            inp_data = inp_data.to(device)
            _logger.debug("Placing cached activations data on GPU.")

        return inp_data
