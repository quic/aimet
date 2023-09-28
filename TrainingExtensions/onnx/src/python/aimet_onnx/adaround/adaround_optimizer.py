# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

from typing import Tuple, Dict
import torch
import torch.nn.functional as functional

from onnx import numpy_helper

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_onnx.adaround.utils import ModuleInfo, read_attributes_for_op
from aimet_torch.adaround.adaround_tensor_quantizer import AdaroundTensorQuantizer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
BATCH_SIZE = 32
EMPIRICAL_THRESHOLD = 3 / 4
DATA_SIZE_IN_BITS = 32


class AdaroundOptimizer:
    """
    Optimizes the weight rounding of quantized wrapper module
    """
    @classmethod
    def _compute_recons_metrics(cls, quant_module: ModuleInfo, act_func: torch.nn.Module, inp_data: torch.Tensor,
                                out_data: torch.Tensor, param_to_adaround_tensor_quantizer: Dict) -> Tuple[float, float]:
        """
        Compute Mean square error of output activations using soft rounding which maps alpha parameter
        between zero and one and hard rounding which maps to exact zero and one
        :param quant_module: Quantized wrapper module
        :param act_func: Activation function
        :param inp_data: Input data to quantized wrapper module
        :param out_data: Output data from module
        :param param_to_adaround_tensor_quantizer: Dict
        :return: Reconstruction error using hard rounding and soft rounding
        """
        adaround_quantizer = param_to_adaround_tensor_quantizer[quant_module.params['weight'].name]

        # Enable hard rounding and get quantized wrapper module's output
        adaround_quantizer.use_soft_rounding = False
        out_data_hard = cls._compute_output_with_adarounded_weights(quant_module, inp_data, adaround_quantizer)

        # Enable soft rounding and get quantized wrapper module's output
        adaround_quantizer.use_soft_rounding = True
        out_data_soft = cls._compute_output_with_adarounded_weights(quant_module, inp_data, adaround_quantizer)

        # If followed by an activation function
        if act_func is not None:
            out_data = act_func(out_data)
            out_data_soft = act_func(out_data_soft)
            out_data_hard = act_func(out_data_hard)

        recons_err_soft = functional.mse_loss(out_data_soft, out_data)
        recons_err_hard = functional.mse_loss(out_data_hard, out_data)

        return float(recons_err_hard), float(recons_err_soft)

    @staticmethod
    def _compute_output_with_adarounded_weights(quant_module, inp_data: torch.Tensor,
                                                adaround_quantizer: AdaroundTensorQuantizer):
        """
        Compute output of AdaroundSupportedModules with adarounded weights
        :param quant_module: Quantized wrapper module
        :param inp_data: The input data to be used for computing the output
        :param adaround_quantizer: Adaround tensor quantizer
        :return: output of the module computed with AdaRounded weights
        """
        # pylint: disable=protected-access
        # Compute adarounded weights
        weights = torch.from_numpy(numpy_helper.to_array(quant_module.params['weight'].tensor))
        adarounded_weights = adaround_quantizer.adaround_weights(weights)

        if quant_module.type == 'Conv':
            attributes = read_attributes_for_op(quant_module)
            bias = torch.from_numpy(numpy_helper.to_array(quant_module.params['bias'].tensor))
            out_data = functional.conv2d(inp_data, adarounded_weights, bias=bias, stride=attributes['strides'],
                                         dilation=attributes['dilations'], padding=attributes['pads'][0],
                                         groups=attributes['group'])
        elif quant_module.type == 'ConvTranspose':
            attributes = read_attributes_for_op(quant_module)
            bias = torch.from_numpy(numpy_helper.to_array(quant_module.params['bias'].tensor))
            out_data = functional.conv_transpose2d(inp_data, adarounded_weights, bias=bias, stride=attributes['strides'],
                                                   dilation=attributes['dilations'], padding=attributes['pads'][0],
                                                   groups=attributes['group'])
        elif quant_module.type in ['Gemm', 'MatMul']:
            bias = torch.from_numpy(numpy_helper.to_array(quant_module.params['bias'].tensor))
            out_data = functional.linear(inp_data, adarounded_weights, bias=bias)

        else:
            raise ValueError('AdaRound is not supported for the module type: ', quant_module.type)

        return out_data
