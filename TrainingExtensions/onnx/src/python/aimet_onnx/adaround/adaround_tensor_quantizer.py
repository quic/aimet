# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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


class AdaroundTensorQuantizer: # pylint: disable=too-many-instance-attributes
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
        self._round_mode = round_mode
        self._quant_scheme = quant_scheme
        self.use_symmetric_encodings = use_symmetric_encodings
        self.use_strict_symmetric = False
        self.use_unsigned_symmetric = False
        self.bitwidth = bitwidth
        self._enabled = enabled_by_default
        self._data_type = QuantizationDataType.int
        self.encoding = None
        self.alpha = None
        self.use_soft_rounding = True
        self._ch_axis = channel_axis
        self.broadcasted_delta = None
        self.broadcasted_offset = None

    def adaround_weights(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Adaround the weight tensor
        :param tensor: The weight tensor to be ada rounded.
        :return: AdaRounded weight tensor
        """
        assert self.encoding, 'Encoding needs to be set before Adaround the weight tensor.'

        self.broadcast_offset_delta(tensor)

        # alpha is the "V" parameter in Equation 2 of the Systems HLD which is defined as a FP32 tensor of the
        # same shape as the weight tensor
        if self.alpha is None:
            self.initialize_alpha(tensor, self.broadcasted_delta)

        alpha = self.alpha.to(device=tensor.device, dtype=tensor.dtype)

        # Scale the tensor
        tensor = torch.floor(tensor / self.broadcasted_delta)

        # Soft rounding maps alpha parameter between zero and one using
        # rectified sigmoid function and hard rounding maps it to exactly zero or one

        if self.use_soft_rounding:
            h_alpha = torch.clamp(torch.sigmoid(alpha) * (AdaroundConstants.ZETA - AdaroundConstants.GAMMA) +
                                  AdaroundConstants.GAMMA, 0, 1)
        else:
            h_alpha = (alpha >= 0).to(tensor.dtype)

        # Adaround the tensor
        tensor = tensor + h_alpha

        # Quantize and de-quantize the tensor
        tensor_quant = torch.clamp(tensor - self.broadcasted_offset, 0, 2 ** self.bitwidth - 1)
        tensor_dequant = (tensor_quant + self.broadcasted_offset) * self.broadcasted_delta

        return tensor_dequant

    def broadcast_offset_delta(self, tensor: torch.Tensor):
        """
        Broadcast offset and delta

        :param tensor: The weight tensor to be adarounded
        """
        if self.broadcasted_delta is None or self.broadcasted_offset is None:
            if isinstance(self.encoding, list):
                # pylint: disable=not-an-iterable
                delta = torch.Tensor([enc.delta for enc in self.encoding]).to(device=tensor.device, dtype=tensor.dtype)
                offset = torch.Tensor([enc.offset for enc in self.encoding]).to(device=tensor.device, dtype=tensor.dtype)
            else:
                delta = self.encoding.delta
                offset = self.encoding.offset

            self.broadcasted_delta = self.broadcast_to_tensor(tensor, delta, self._ch_axis).to(tensor.dtype)
            self.broadcasted_offset = self.broadcast_to_tensor(tensor, offset, self._ch_axis).to(tensor.dtype)

    def initialize_alpha(self, tensor: torch.Tensor, delta):
        """
        Initializes alpha parameter, same shape as the weight tensor
        :param tensor: The weight tensor to be ada rounded
        """
        tensor_floor = torch.floor(tensor / delta)

        tensor = (tensor / delta) - tensor_floor
        alpha = - torch.log((AdaroundConstants.ZETA - AdaroundConstants.GAMMA) / (tensor - AdaroundConstants.GAMMA) - 1)

        # Even if the input is float16, alpha has to be kept in float32
        # in order to be updated by the optimizer
        self.alpha = torch.nn.Parameter(alpha.float(), requires_grad=True)

    @staticmethod
    def broadcast_to_tensor(tensor: torch.Tensor, encoding: torch.Tensor, ch_axis: int):
        """
        This helper method takes n-dimension tensor and a 1-dimension encoding. And the encoding is broad-casted to
        match the n-dimensional tensor

        :param tensor: Tensor to use as target for the broadcasting operation
        :param encoding: Encoding 1-dimensional tensor to broadcast
        :param ch_axis: Channel axis along which broadcasting happens
        :return: Broad-casted tensor
        """
        if not isinstance(encoding, torch.Tensor):
            encoding = torch.tensor(encoding).to(tensor.device)  # convert encoding to a tensor

        assert len(encoding.shape) <= 1  # Should be 1-dimensional tensor

        if encoding.numel() == 1:
            return encoding

        # Shape of encoding should match the channel dimension of the input
        assert encoding.numel() == tensor.shape[ch_axis]

        shape = tuple(dim if axis == ch_axis else 1
                      for axis, dim in enumerate(tensor.shape))
        return encoding.view(shape)
