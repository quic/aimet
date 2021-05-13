# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Adaround optimizer """

from typing import Union, Tuple, List
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_torch.qc_quantize_op import QcPostTrainingWrapper
from aimet_torch.adaround.activation_sampler import ActivationSampler
from aimet_torch.adaround.adaround_loss import AdaroundLoss, AdaroundHyperParameters

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
BATCH_SIZE = 32


class AdaroundOptimizer:
    """
    Optimizes the weight rounding of quantized wrapper module
    """
    @classmethod
    def adaround_module(cls, module: torch.nn.Module, quant_module: QcPostTrainingWrapper,
                        orig_model: torch.nn.Module, quant_model: torch.nn.Module,
                        act_func: Union[torch.nn.Module, None], cached_dataset: Dataset,
                        opt_params: AdaroundHyperParameters):
        """
        Adaround module
        :param module: Original module
        :param quant_module: Quantized wrapper module
        :param orig_model: The original, un quantized, model
        :param quant_model: QuantSim model
        :param act_func: Activation function
        :param cached_dataset: Cached dataset
        :param opt_params: Optimization parameters
        """
        assert isinstance(quant_module, QcPostTrainingWrapper), '%s is not wrapper module.' % quant_module
        assert quant_module.param_quantizers['weight'], '%s does not have weight quantizer.' % quant_module

        # Get input and output data of batch size to compute reconstruction error of output activations
        # before and after optimization
        iterator = iter(cached_dataset)
        inp_data, out_data = ActivationSampler.sample_activation(module, quant_module, orig_model,
                                                                 quant_model, iterator, num_batches=1)

        recons_err_hard, recons_err_soft = cls._compute_recons_metrics(quant_module, act_func, inp_data, out_data)
        logger.debug("Before opt, Recons. error metrics using soft rounding=%f and hard rounding=%f", recons_err_soft,
                     recons_err_hard)

        # Optimize weight rounding
        cls._optimize_rounding(module, quant_module, orig_model, quant_model, act_func, cached_dataset, opt_params)

        recons_err_hard, recons_err_soft = cls._compute_recons_metrics(quant_module, act_func, inp_data, out_data)
        logger.debug("After opt, Recons. error metrics using soft rounding=%f and hard rounding=%f", recons_err_soft,
                     recons_err_hard)

        # After optimization, set the optimized layer's rounding mode to "Hard rounding"
        quant_module.param_quantizers['weight'].use_soft_rounding = False

    @classmethod
    def _optimize_rounding(cls, module: torch.nn.Module, quant_module: QcPostTrainingWrapper,
                           orig_model: torch.nn.Module, quant_model: torch.nn.Module,
                           act_func: Union[torch.nn.Module, None], cached_dataset: Dataset,
                           opt_params: AdaroundHyperParameters):
        """
        Optimizes the weight rounding of quantized wrapper module
        :param module: Original module
        :param quant_module: Quantized wrapper module
        :param orig_model: The original, un quantized, model
        :param quant_model: QuantSim model
        :param act_func: Activation function
        :param cached_dataset: Cached dataset
        :param opt_params: Optimization parameters
        """
        # pylint: disable=too-many-locals
        adaround_quantizer = quant_module.param_quantizers['weight']

        assert adaround_quantizer.use_soft_rounding, 'optimization should use soft rounding only.'
        assert adaround_quantizer.alpha is not None, 'alpha parameter should be initialized.'

        # Split total batches and iterations into chunks
        num_chunks = cls._compute_chunks_for_act_data(module, quant_module, orig_model, quant_model, cached_dataset)
        batches = cls._split_into_chunks(len(cached_dataset), num_chunks)
        iterations = cls._split_into_chunks(opt_params.num_iterations, num_chunks)
        logger.debug("Collecting activation data and optimizing layer using chunk(s)=%d", num_chunks)

        # Create and set up Adam optimizer with parameter 'alpha' to be optimized
        optimizer = torch.optim.Adam([adaround_quantizer.alpha])

        # Optimization using chunked input and output activation data
        cur_iteration = 0
        iterator = iter(cached_dataset)

        for chunk in range(num_chunks):

            # Collect input and output activations data in chunks
            all_inp_data, all_orig_out_data = ActivationSampler.sample_activation(module, quant_module, orig_model,
                                                                                  quant_model, iterator,
                                                                                  num_batches=batches[chunk])

            for _ in range(iterations[chunk]):

                # Get random indices of batch size and get original output and input activation data of batch size
                indices = torch.randperm(all_inp_data.size(0))[:BATCH_SIZE]
                inp_data = all_inp_data[indices]
                orig_out_data = all_orig_out_data[indices]

                # Clear alpha's gradients before optimization step
                optimizer.zero_grad()

                # Get the module's output activations using AdaRounded weights
                quant_out_data = cls._compute_output_with_adarounded_weights(quant_module, inp_data)

                # If followed by an activation function
                if act_func is not None:
                    orig_out_data = act_func(orig_out_data)
                    quant_out_data = act_func(quant_out_data)

                # Calculate total loss
                recon_loss = AdaroundLoss.compute_recon_loss(quant_out_data, orig_out_data)
                round_loss = AdaroundLoss.compute_round_loss(adaround_quantizer.alpha, opt_params, cur_iteration)
                total_loss = recon_loss + round_loss

                # Back propagate and Update the parameter 'alpha'
                total_loss.backward()
                optimizer.step()

                if cur_iteration == 0 or cur_iteration % 100 == 0:
                    logger.debug("After iterations=%d, Total loss=%5f, Recons. loss=%5f, Rounding loss=%5f",
                                 cur_iteration, float(total_loss), float(recon_loss), float(round_loss))

                cur_iteration += 1

            # Delete intermediate tensor references
            del all_inp_data
            del all_orig_out_data

    @classmethod
    def _compute_recons_metrics(cls, quant_module: QcPostTrainingWrapper, act_func, inp_data: torch.Tensor,
                                out_data: torch.Tensor) -> Tuple[float, float]:
        """
        Compute Mean square error of output activations using soft rounding which maps alpha parameter
        between zero and one and hard rounding which maps to exact zero and one
        :param quant_module: Quantized wrapper module
        :param act_func: Activation function
        :param inp_data: Input data to quantized wrapper module
        :param out_data: Output data from module
        :return: Reconstruction error using hard rounding and soft rounding
        """
        adaround_quantizer = quant_module.param_quantizers['weight']

        # Enable hard rounding and get quantized wrapper module's output
        adaround_quantizer.use_soft_rounding = False
        out_data_hard = cls._compute_output_with_adarounded_weights(quant_module, inp_data)

        # Enable soft rounding and get quantized wrapper module's output
        adaround_quantizer.use_soft_rounding = True
        out_data_soft = cls._compute_output_with_adarounded_weights(quant_module, inp_data)

        # If followed by an activation function
        if act_func is not None:
            out_data = act_func(out_data)
            out_data_soft = act_func(out_data_soft)
            out_data_hard = act_func(out_data_hard)

        recons_err_soft = functional.mse_loss(out_data_soft, out_data)
        recons_err_hard = functional.mse_loss(out_data_hard, out_data)

        return float(recons_err_hard), float(recons_err_soft)

    @staticmethod
    def _compute_output_with_adarounded_weights(quant_module: QcPostTrainingWrapper, inp_data: torch.Tensor):
        """
        Compute output of AdaroundSupportedModules with adarounded weights
        :param quant_module: Quantized wrapper module
        :param inp_data: The input data to be used for computing the output
        :return: output of the module computed with AdaRounded weights
        """
        # pylint: disable=protected-access
        module = quant_module._module_to_wrap
        adaround_quantizer = quant_module.param_quantizers['weight']

        # Compute adarounded weights
        adarounded_weights = adaround_quantizer.adaround_weights(module.weight)

        if isinstance(module, torch.nn.Conv2d):
            out_data = functional.conv2d(inp_data, adarounded_weights, bias=module.bias, stride=module.stride,
                                         dilation=module.dilation, padding=module.padding, groups=module.groups)
        elif isinstance(module, torch.nn.ConvTranspose2d):
            out_data = functional.conv_transpose2d(inp_data, adarounded_weights, bias=module.bias,
                                                   stride=module.stride, padding=module.padding,
                                                   output_padding=module.output_padding, groups=module.groups,
                                                   dilation=module.dilation)
        elif isinstance(module, torch.nn.Linear):
            out_data = functional.linear(inp_data, adarounded_weights, bias=module.bias)

        else:
            raise ValueError('AdaRound is not supported for the module: ', module)

        return out_data

    @staticmethod
    def _split_into_chunks(value: int, chunks: int) -> List:
        """
        Split a number into almost equal chunks
        :param value: Value to be split
        :param chunks: Chunks
        :return: List of splits
        """
        assert not value < chunks, 'Can not split {} into {} chunks'.format(value, chunks)

        if value % chunks == 0:
            splits = [value // chunks for _ in range(chunks)]

        else:
            splits = [value // chunks + 1 if c >= chunks - value % chunks else value // chunks for c in range(chunks)]

        return splits

    @staticmethod
    def _compute_chunks_for_act_data(module: torch.nn.Module, quant_module: QcPostTrainingWrapper,
                                     orig_model: torch.nn.Module, quant_model: torch.nn.Module,
                                     cached_dataset: Dataset) -> int:
        """
        Function computes number of possible chunks needed to split activation data that can be fit on
        device without running out of memory
        :param module: Original module
        :param quant_module: Quantized wrapper module
        :param orig_model: The original, un quantized, model
        :param quant_model: QuantSim model
        :param cached_dataset: Cached dataset
        :return: Number of chunks
        """
        num_chunks = 1

        while True:
            iterator = iter(cached_dataset)
            num_batches = int(len(cached_dataset) / num_chunks)
            try:
                ActivationSampler.sample_activation(module, quant_module, orig_model, quant_model, iterator,
                                                    num_batches=num_batches)
                break

            except RuntimeError:
                num_chunks += 1

        return num_chunks
