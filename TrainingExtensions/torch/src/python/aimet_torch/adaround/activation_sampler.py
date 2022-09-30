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

from typing import Tuple, Union, List, Callable, Any
import torch
from torch.utils.data import Dataset

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_torch.utils import ModuleData
from aimet_torch.qc_quantize_op import QcQuantizeWrapper

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class ActivationSampler:
    """
    For a module in the original model and the corresponding module in the weight quantized QuantSim model,
    collect the module's output and input activation data respectively
    """
    def __init__(self, orig_module: torch.nn.Module, quant_module: QcQuantizeWrapper,
                 orig_model: torch.nn.Module, quant_model: torch.nn.Module,
                 forward_fn: Callable[[torch.nn.Module, Any], Any]):
        """
        :param orig_module: Module from original model.
        :param quant_module: Quant wrapper from sim model.
        :param orig_model: Original model.
        :param quant_model: Sim model.
        :param forward_fn: Adapter function that performs forward pass given a model and inputs
         yielded from the data loader.
        """
        self._orig_module = orig_module
        self._quant_module = quant_module
        self._orig_model = orig_model
        self._quant_model = quant_model
        self._orig_module_collector = ModuleData(orig_model, orig_module, forward_fn)
        self._quant_module_collector = ModuleData(quant_model, quant_module, forward_fn)

    def sample_and_place_all_acts_on_cpu(self, cached_dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        From the original module, collect output activations and input activations
        to corresponding quantized module.

        NOTE: Keeps collected activation data on CPU memory so this function should only be invoked
        if collected activation data can be fit entirely in CPU memory.

        :param cached_dataset: Cached dataset.
        :return: Input data, output data
        """
        all_inp_data = []
        all_out_data = []

        iterator = iter(cached_dataset)
        for batch_index in range(len(cached_dataset)):
            model_inputs = next(iterator)
            inp_data, out_data = self.sample_acts(model_inputs)

            # Keep activation data on CPU memory and then append.
            all_inp_data.append(inp_data.cpu())
            all_out_data.append(out_data.cpu())

            if batch_index == len(cached_dataset) - 1:
                break
        all_inp_data = torch.cat(all_inp_data, dim=0)
        all_out_data = torch.cat(all_out_data, dim=0)

        return all_inp_data, all_out_data

    def sample_acts(self, model_inputs: Union[torch.tensor, List, Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For given model_inputs, collect input activations data to quant module and
        output activations data from original module.

        :param model_inputs: Model inputs.
        :return: Input and output activations data.
        """
        # Collect input activation data to quantized wrapper module
        # (with all preceding weight modules quantized)
        inp_data, _ = self._quant_module_collector.collect_inp_out_data(model_inputs,
                                                                        collect_input=True,
                                                                        collect_output=False)
        # Collect output activation data from original module
        _, out_data = self._orig_module_collector.collect_inp_out_data(model_inputs,
                                                                       collect_input=False,
                                                                       collect_output=True)
        return inp_data, out_data
