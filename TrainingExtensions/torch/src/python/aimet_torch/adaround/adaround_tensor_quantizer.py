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

""" Custom Tensor Quantizer for quantizing weights using Adaround """

import torch
import torch.nn

# Import AIMET specific modules
from aimet_common.defs import AdaroundConstants, QuantizationDataType, QuantScheme
from aimet_torch.tensor_quantizer import TensorQuantizer

class AdaroundTensorQuantizer(TensorQuantizer):
    """
    Simulates quantization for the given tensor post training using Adaround
    """
    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: QuantScheme, use_symmetric_encodings: bool,
                 enabled_by_default: bool, channel_axis: int):
        """
        Constructor
        :param bitwidth: Quantization bitwidth
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. Range Learning)
        :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
        :param channel_axis: Channel axis of parameter tensor. Only used during per channel Adaround.
        """
        #TODO Remove the hardcoding of data_type
        super(AdaroundTensorQuantizer, self).__init__(bitwidth, round_mode, quant_scheme, use_symmetric_encodings,
                                                      enabled_by_default, QuantizationDataType.int)
        self.encoding = None
        # V in System HLD
        self.alpha = None
        self.use_soft_rounding = True
        self._ch_axis = channel_axis

    def quantize_dequantize(self, tensor: torch.Tensor, _) -> torch.Tensor:
        """
        Quantize-dequantize the tensor, using the saved encoding for this tensor
        :param tensor: Tensor to quantize-dequantize
        :param _: Rounding mode parameter is not used
        :return: Resulting tensor
        """

        if self.enabled:
            quantized_tensor = self.adaround_weights(tensor)
        else:
            quantized_tensor = tensor

        return quantized_tensor

    def update_encoding_stats(self, tensor: torch.Tensor):
        """
        Update the stats for computing encoding
        :param tensor: Tensor to use for updating the encodings stats
        """
        # No action required for AdaroundTensorQuantizer.
        # Function is in place if QcQuantizeWrapper calls while in Training mode

    def compute_encoding(self):
        """
        Compute the quantization encoding for this tensor
        """
        # No action required for AdaroundTensorQuantizer.
        # Function is in place if QcQuantizeWrapper calls while in Training mode

    def reset_encoding_stats(self):
        """
        Resets the encodings stats
        """
        # No action required for AdaroundTensorQuantizer.
        # Function is in place if QcQuantizeWrapper calls while in Training mode

    def adaround_weights(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Adaround the weight tensor
        :param tensor: The weight tensor to be ada rounded.
        :return: AdaRounded weight tensor
        """
        assert self.encoding, 'Encoding needs to be set before Adaround the weight tensor.'

        def _broadcast_to_tensor(encoding, ch_axis):
            """
            This helper method takes n-dimension tensor and a 1-dimension encoding. And the encoding is broad-casted to
            match the n-dimensional tensor
            :param encoding: Encoding 1-dimensional tensor to broadcast
            :return: Broad-casted tensor
            """
            shape = list(tensor.shape)
            encoding = torch.Tensor(encoding).to(tensor.device)  # convert encoding to a tensor

            # Original tensor shape is OIHW/IOHW, we change the shape to IHWO. Encoding (which is of shape O) can naturally
            # broadcast to this shape
            # This will work if the original tensor shape was any dimensions as long as the first dimension matches the
            # encoding tensor shape
            num_channels = shape.pop(ch_axis)
            encoding = encoding * torch.ones(shape + [num_channels]).to(tensor.device)

            # we permute the resulting tensor back to OIHW/IOHW shape
            permute_dims = list(range(len(shape)))
            permute_dims.insert(ch_axis, len(shape))
            encoding = encoding.permute(permute_dims)

            return encoding

        if isinstance(self.encoding, list):
            delta = [enc.delta for enc in self.encoding]                # pylint: disable=not-an-iterable
            broadcasted_delta = _broadcast_to_tensor(delta, self._ch_axis)
            offset = [enc.offset for enc in self.encoding]              # pylint: disable=not-an-iterable
            broadcasted_offset = _broadcast_to_tensor(offset, self._ch_axis)
        else:
            broadcasted_delta = self.encoding.delta
            broadcasted_offset = self.encoding.offset

        # alpha is the "V" parameter in Equation 2 of the Systems HLD which is defined as a FP32 tensor of the
        # same shape as the weight tensor
        if self.alpha is None:
            self._initialize_alpha(tensor, broadcasted_delta)

        # Scale the tensor
        tensor = torch.floor(tensor / broadcasted_delta)

        # Soft rounding maps alpha parameter between zero and one using
        # rectified sigmoid function and hard rounding maps it to exactly zero or one
        if self.use_soft_rounding:
            h_alpha = torch.clamp(torch.sigmoid(self.alpha) * (AdaroundConstants.ZETA - AdaroundConstants.GAMMA) +
                                  AdaroundConstants.GAMMA, 0, 1)
        else:
            h_alpha = (self.alpha >= 0).float()


        # Adaround the tensor
        tensor = tensor + h_alpha

        # Quantize and de-quantize the tensor
        tensor_quant = torch.clamp(tensor - broadcasted_offset, 0, 2 ** self.bitwidth - 1)
        tensor_dequant = (tensor_quant + broadcasted_offset) * broadcasted_delta

        return tensor_dequant

    def _initialize_alpha(self, tensor: torch.Tensor, delta):
        """
        Initializes alpha parameter, same shape as the weight tensor
        :param tensor: The weight tensor to be ada rounded
        """
        tensor_floor = torch.floor(tensor / delta)

        tensor = (tensor / delta) - tensor_floor
        alpha = - torch.log((AdaroundConstants.ZETA - AdaroundConstants.GAMMA) / (tensor - AdaroundConstants.GAMMA) - 1)

        self.alpha = torch.nn.Parameter(alpha, requires_grad=True)
