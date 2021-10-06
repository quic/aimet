# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Custom Tensor Quantizers for PyTorch Op for quantizing weights and activations """

import io
from typing import Union, List

import torch

from aimet_common.defs import QuantScheme
from aimet_common.utils import AimetLogger
import aimet_torch.quantsim_straight_through_grad as ste
import libpymo
#TODO Pylint fails due an unknown import issue. We need to debug this later.
import AimetTensorQuantizer # pylint: disable=import-error

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class TensorQuantizer:
    """
    Base class for Simulation of quantization for a given tensor. This tensor can be a parameter in the model or an
    input to a layer or an output from a layer.
    """

    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: Union[QuantScheme, str],
                 use_symmetric_encodings: bool, enabled_by_default: bool):
        """
        Constructor
        :param bitwidth: Quantization bitwidth
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. Range Learning)
        :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
        """
        super(TensorQuantizer, self).__init__()
        self.round_mode = round_mode
        self.quant_scheme = quant_scheme
        self.use_symmetric_encodings = use_symmetric_encodings
        self.use_strict_symmetric = False
        self.use_unsigned_symmetric = True
        self.bitwidth = bitwidth
        self.enabled = enabled_by_default


class PickableState:
    """
    State variables in QcQuantizeBase that need to be saved separately when pickling
    """
    def __init__(self, builtin_dict, encoding, num_channels):
        self.dict = builtin_dict
        self.num_channels = num_channels

        if encoding:
            self.encodings = []
            for enc in encoding:
                self.encodings.append((enc.min, enc.max, enc.delta, enc.offset, enc.bw))


class StaticGridTensorQuantizer(TensorQuantizer):
    """
    Simulates quantization for the given tensor post training.
    """

    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: str, use_symmetric_encodings: bool,
                 enabled_by_default: bool):
        """
        Constructor
        :param bitwidth: Quantization bitwidth
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. tf, tf_enhanced)
        :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
        """
        super(StaticGridTensorQuantizer, self).__init__(bitwidth, round_mode, quant_scheme, use_symmetric_encodings,
                                                        enabled_by_default)
        self._cppOp = None
        self._encoding = None
        self._is_encoding_frozen = False

    def __str__(self):
        stream = io.StringIO(newline='\n')
        stream.write('Static Grid TensorQuantizer:\n')
        stream.write('  quant-scheme:{}, round_mode={}, bitwidth={}, enabled={}\n'.format(self.quant_scheme,
                                                                                          self.round_mode,
                                                                                          self.bitwidth,
                                                                                          self.enabled))
        if self._encoding:
            for enc in self._encoding:
                stream.write('  min:{}, max={}, delta={}, offset={}\n'.format(enc.min, enc.max,
                                                                              enc.delta, enc.offset))
        else:
            stream.write('  no encoding\n')

        return stream.getvalue()

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = PickableState(self.__dict__.copy(), self._encoding, len(self._cppOp))

        # Remove the unpicklable entries.
        del state.dict['_cppOp']
        del state.dict['_encoding']

        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state.dict)

        # Create the c++ op
        self._cppOp = []
        for _ in range(state.num_channels):
            self._cppOp.append(AimetTensorQuantizer.AimetTensorQuantizer(self.quant_scheme))

        # Create the encoding object
        if hasattr(state, 'encodings'):

            self._encoding = []
            for enc in state.encodings:

                enc_min, enc_max, delta, offset, bw = enc

                new_encoding = libpymo.TfEncoding()
                new_encoding.bw = bw
                new_encoding.max = enc_max
                new_encoding.min = enc_min
                new_encoding.delta = delta
                new_encoding.offset = offset

                self._encoding.append(new_encoding)
        else:
            self._encoding = None

    @property
    def encoding(self):
        """
        Property to get encoding
        :return: One (per-tensor) or list of many (per-channel) encodings
        """
        return self._encoding

    @encoding.setter
    def encoding(self, encoding):
        if not self._is_encoding_frozen:
            self._encoding = encoding

    def compute_encoding(self):
        """
        Compute the quantization encoding for this tensor
        """
        if self.enabled and not self._is_encoding_frozen:

            self._encoding = []
            for op in self._cppOp:
                encoding, is_encoding_valid = op.getEncoding(self.bitwidth, self.use_symmetric_encodings,
                                                             self.use_strict_symmetric,
                                                             self.use_unsigned_symmetric)

                if not is_encoding_valid:
                    self.enabled = False
                else:
                    self._encoding.append(encoding)

            # Check for the case when some cppOp encodings are not valid while others are.
            # In the case that a module is unused, all cppOp encodings will have is_encoding_valid False, and there
            # would be no entries in self._encoding. The only way for self._encoding to be non empty with self.enabled
            # False is if some encodings were valid while others were not.
            # TODO: add a test case for testing this assert
            if not self.enabled and self._encoding:
                _logger.error('At least one encoding for a multi-encoding quantizer is invalid.')
                assert not (not self.enabled and self._encoding)


    def quantize_dequantize(self, tensor, round_mode):
        """
        Quantize-dequantize the tensor, using the saved encoding for this tensor
        :param tensor: Tensor to quantize-dequantize
        :param round_mode: Rounding mode
        :return: Resulting tensor
        """
        output = QuantizeDequantize.apply(tensor, self, round_mode)
        return output

    def quantize(self, tensor, round_mode, use_int32=True):
        """
        Quantize the tensor, using the saved encoding for this tensor
        :param tensor: Tensor to quantize
        :param round_mode: Rounding mode
        :param use_int32: True if using int32 as output datatype, False for int64
        :return: Resulting tensor
        """
        output = Quantize.apply(tensor, self, round_mode, use_int32)
        return output

    def reset_encoding_stats(self):
        """
        Resets the encodings stats and set encoding to None
        """
        if not self._is_encoding_frozen:
            for op in self._cppOp:
                op.resetEncodingStats()
            self._encoding = None

    def freeze_encoding(self):
        """
        Freeze the encoding
        """
        self._is_encoding_frozen = True


class StaticGridPerTensorQuantizer(StaticGridTensorQuantizer):
    """
    Simulates quantization for the given tensor using a per-tensor scale/offset
    """

    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: str, use_symmetric_encodings: bool,
                 enabled_by_default: bool):
        """
        Constructor
        :param bitwidth: Quantization bitwidth
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. tf, tf_enhanced)
        :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
        """
        super(StaticGridPerTensorQuantizer, self).__init__(bitwidth, round_mode, quant_scheme, use_symmetric_encodings,
                                                           enabled_by_default)
        self._cppOp = [AimetTensorQuantizer.AimetTensorQuantizer(quant_scheme)]
        self._encoding = None
        self._is_encoding_frozen = False

    @property
    def encoding(self):
        if self._encoding:
            return self._encoding[0]

        return None

    @encoding.setter
    def encoding(self, encoding: libpymo.TfEncoding):
        if not self._is_encoding_frozen:
            self._encoding = [encoding]

    def update_encoding_stats(self, tensor):
        """
        Update the stats for computing encoding
        :param tensor: Tensor to use for updating the encodings stats
        """
        if self.enabled and not self._is_encoding_frozen:
            for op in self._cppOp:
                op.updateStats(tensor, tensor.is_cuda)


class StaticGridPerChannelQuantizer(StaticGridTensorQuantizer):
    """
    Simulates quantization for the given tensor using a per-channel scale/offset
    """

    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: str, use_symmetric_encodings: bool,
                 num_channels: int, enabled_by_default: bool):
        """
        Constructor
        :param bitwidth: Quantization bitwidth
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. tf, tf_enhanced)
        :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
        """
        super(StaticGridPerChannelQuantizer, self).__init__(bitwidth, round_mode, quant_scheme, use_symmetric_encodings,
                                                            enabled_by_default)
        self._cppOp = [AimetTensorQuantizer.AimetTensorQuantizer(quant_scheme) for _ in range(num_channels)]
        self._encoding = None
        self._is_encoding_frozen = False

    @property
    def encoding(self):
        return self._encoding

    @encoding.setter
    def encoding(self, encoding: List[libpymo.TfEncoding]):
        if not self._is_encoding_frozen:
            self._encoding = encoding

    def update_encoding_stats(self, tensor):
        """
        Update the stats for computing encoding
        :param tensor: Tensor to use for updating the encodings stats
        """
        if self.enabled and not self._is_encoding_frozen:
            for channel_idx, op in enumerate(self._cppOp):
                op.updateStats(tensor[channel_idx], tensor.is_cuda)


class QuantizeDequantize(torch.autograd.Function):
    """
    Custom gradient function for STE
    """

    @staticmethod
    def _per_tensor_quantize_dequantize(tensor, tensor_quantizer, round_mode):
        # pylint:disable = protected-access
        return tensor_quantizer._cppOp[0].quantizeDequantize(tensor, tensor_quantizer.encoding,
                                                             round_mode, tensor.is_cuda)

    @staticmethod
    def _per_channel_quantize_dequantize(tensor, tensor_quantizer, round_mode):
        quantized_tensors = []

        # pylint:disable = protected-access
        for index, op in enumerate(tensor_quantizer._cppOp):
            quantized_tensors.append(op.quantizeDequantize(tensor[index, :], tensor_quantizer._encoding[index],
                                                           round_mode, tensor.is_cuda))
        quantized_tensor = torch.stack(tuple(quantized_tensors))
        return quantized_tensor

    # pylint:disable = arguments-differ
    @staticmethod
    def forward(ctx, tensor, tensor_quantizer, round_mode):
        """
        Quantize-dequantize the tensor, using the saved encoding for this tensor
        :param tensor: Tensor to quantize-dequantize
        :param tensor_quantizer: Reference to the tensor quantizer
        :param round_mode: Rounding mode
        :return: Resulting tensor
        """
        if tensor_quantizer.enabled:

            # pylint: disable=protected-access
            if len(tensor_quantizer._cppOp) > 1:
                quantized_tensor = QuantizeDequantize._per_channel_quantize_dequantize(tensor, tensor_quantizer,
                                                                                       round_mode)
            else:
                quantized_tensor = QuantizeDequantize._per_tensor_quantize_dequantize(tensor, tensor_quantizer,
                                                                                      round_mode)

        else:
            quantized_tensor = tensor

        ctx.save_for_backward(quantized_tensor)

        ctx.tensor_quantizer = tensor_quantizer
        return quantized_tensor

    @staticmethod
    def backward(ctx, output_grad):
        tensor = ctx.saved_tensors
        tensor_quantizer = ctx.tensor_quantizer
        if tensor_quantizer.enabled:
            grad = ste.compute_dloss_by_dx(tensor[0], output_grad, tensor_quantizer.encoding.min,
                                           tensor_quantizer.encoding.max)
        else:
            grad = output_grad

        return grad, None, None


class Quantize(torch.autograd.Function):
    """
    Custom gradient function for STE
    """
    # pylint:disable = arguments-differ
    @staticmethod
    def forward(ctx, tensor, tensor_quantizer, round_mode, use_int32):
        """
        Quantize the tensor, using the saved encoding for this tensor
        :param tensor: Tensor to quantize
        :param tensor_quantizer: Reference to the tensor quantizer
        :param round_mode: Rounding mode
        :param use_int32: True if using int32 as output datatype, False for int64
        :return: Resulting tensor
        """
        if tensor_quantizer.enabled:
            # pylint:disable = protected-access
            shift_to_signed = False
            if tensor_quantizer.use_symmetric_encodings and tensor_quantizer.encoding.offset < 0:
                shift_to_signed = True
            quantized_tensor = tensor_quantizer._cppOp[0].quantize(tensor, tensor_quantizer.encoding, round_mode,
                                                                   tensor.is_cuda, shift_to_signed, use_int32)
        else:
            quantized_tensor = tensor

        ctx.save_for_backward(quantized_tensor)

        ctx.tensor_quantizer = tensor_quantizer
        return quantized_tensor

    @staticmethod
    def backward(ctx, output_grad):
        _logger.error('Backward pass for quantize only not implemented')
        raise AssertionError
