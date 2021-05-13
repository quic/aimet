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

""" Sample input to quantized wrapper module and output from original module for Adaround feature """

from collections.abc import Iterator
from typing import Tuple
import torch.nn

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_torch.utils import ModuleData, get_device
from aimet_torch.qc_quantize_op import QcPostTrainingWrapper

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class ActivationSampler:
    """
    For a module in the original model and the corresponding module in the weight quantized QuantSim model,
    collect the module's output and input activation data respectively
    """
    @staticmethod
    def sample_activation(orig_module: torch.nn.Module, quant_module: QcPostTrainingWrapper,
                          orig_model: torch.nn.Module, quant_model: torch.nn.Module,
                          iterator: Iterator, num_batches: int) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        From the original module, collect output activations and input activations to corresponding quantized module.
        :param orig_module: Single un quantized module from the original model
        :param quant_module: Corresponding quantized wrapper module from the QuantSim model
        :param orig_model: The original, un quantized, model
        :param quant_model: QuantSim model whose weights have been quantized using Nearest rounding
        :param iterator: Cached dataset iterator
        :param num_batches: Number of batches
        :return: Input data, output data
        """
        # Create ModuleData for original module
        orig_module_data = ModuleData(orig_model, orig_module)

        # Create ModuleData for quantized wrapper module
        quant_module_data = ModuleData(quant_model, quant_module)

        all_inp_data = []
        all_out_data = []

        for batch_index in range(num_batches):

            model_input = next(iterator)

            # Collect input activation data to quantized wrapper module (with all preceding weight modules quantized)
            inp_data, _ = quant_module_data.collect_inp_out_data(model_input, collect_input=True, collect_output=False)

            # Collect output activation data from original module
            _, out_data = orig_module_data.collect_inp_out_data(model_input, collect_input=False, collect_output=True)

            # Keep activation data on CPU memory
            all_inp_data.append(inp_data.cpu())
            all_out_data.append(out_data.cpu())

            if batch_index == num_batches - 1:
                break

        all_inp_data = torch.cat(all_inp_data, dim=0).to(get_device(quant_module))
        all_out_data = torch.cat(all_out_data, dim=0).to(get_device(quant_module))

        return all_inp_data, all_out_data
