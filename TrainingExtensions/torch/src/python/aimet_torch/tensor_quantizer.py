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
from typing import List, Dict

import torch
from aimet_common.defs import QuantScheme, QuantizationDataType, MAP_QUANT_SCHEME_TO_PYMO
from aimet_common.utils import AimetLogger
import aimet_torch.quantsim_straight_through_grad as ste
import libpymo                  # pylint: disable=import-error

#TODO Pylint fails due an unknown import issue. We need to debug this later.
import AimetTensorQuantizer     # pylint: disable=import-error

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class TensorQuantizer:
    """
    Base class for Simulation of quantization for a given tensor. This tensor can be a parameter in the model or an
    input to a layer or an output from a layer.
    """

    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: QuantScheme,
                 use_symmetric_encodings: bool, enabled_by_default: bool,
                 data_type: QuantizationDataType = QuantizationDataType.int):
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
        self._quant_scheme = quant_scheme
        self.use_symmetric_encodings = use_symmetric_encodings
        self.use_strict_symmetric = False
        self.use_unsigned_symmetric = True
        self.bitwidth = bitwidth
        self.enabled = enabled_by_default
        self.data_type = data_type


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

    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: QuantScheme, use_symmetric_encodings: bool,
                 enabled_by_default: bool, data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Constructor
        :param bitwidth: Quantization bitwidth
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. tf, tf_enhanced)
        :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
        """
        super(StaticGridTensorQuantizer, self).__init__(bitwidth, round_mode, quant_scheme, use_symmetric_encodings,
                                                        enabled_by_default, data_type)
        self._cppOp = None
        self._encoding = None
        self._is_encoding_frozen = False

    def __str__(self):
        stream = io.StringIO(newline='\n')
        stream.write('StaticGrid TensorQuantizer:\n')
        stream.write('    quant-scheme:{}, round_mode={}, bitwidth={}, enabled={}\n'.format(self._quant_scheme,
                                                                                            self.round_mode,
                                                                                            self.bitwidth,
                                                                                            self.enabled))
        if self._encoding:
            for enc in self._encoding:
                stream.write('    min:{}, max={}, delta={}, offset={}\n'.format(enc.min, enc.max,
                                                                                enc.delta, enc.offset))
        else:
            stream.write('    no encoding\n')

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
        quant_scheme = MAP_QUANT_SCHEME_TO_PYMO[self._quant_scheme]
        for _ in range(state.num_channels):
            self._cppOp.append(AimetTensorQuantizer.AimetTensorQuantizer(quant_scheme))

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
    def quant_scheme(self) -> QuantScheme:
        """
        Property to get quant_scheme
        :return: QuantScheme
        """
        return self._quant_scheme

    @quant_scheme.setter
    def quant_scheme(self, quant_scheme: QuantScheme):
        """
        Property to set quant_scheme. When changing quantization schemes, it is necessary to re-instantiate the
        underlying C++ op (there is no way currently to set quant-scheme post instantiation for these objects). This
        will also automatically clear any accumulated statistics - which is a good side-effect.
        :param quant_scheme: Quantization scheme (see enum)
        """
        self._quant_scheme = quant_scheme
        quant_scheme = MAP_QUANT_SCHEME_TO_PYMO[quant_scheme]
        assert self._cppOp              # whether per-tensor or per-channel, there needs to be at least 1 op
        self._cppOp = [AimetTensorQuantizer.AimetTensorQuantizer(quant_scheme) for _ in self._cppOp]

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
            if self.data_type == QuantizationDataType.float:
                self._encoding = None
            else:
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
        output = output.clone()
        return output

    def quantize(self, tensor, round_mode):
        """
        Quantize the tensor, using the saved encoding for this tensor
        :param tensor: Tensor to quantize
        :param round_mode: Rounding mode
        :return: Resulting tensor
        """
        output = Quantize.apply(tensor, self, round_mode)
        output = output.clone()
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

    def get_stats_histogram(self) -> List[List]:
        """
        NOTE: Not to invoke when quantization scheme is not TF-Enhanced.

        Get histogram of statistics. Returns list of buckets where each bucket is
        tuple of two values - the float value representing the left edge of the
        bucket and a PDF of the values in this bucket relative to all the values
        seen across all buckets.

        :return: List of buckets where each bucket is (xLeft, PDF).
        """
        if self._quant_scheme != QuantScheme.post_training_tf_enhanced:
            raise RuntimeError("get_stats_histogram() can be invoked only when quantization scheme is TF-Enhanced.")

        if not self._encoding:
            raise RuntimeError("get_stats_histogram() can be invoked only when encoding is computed.")

        histogram = []
        for op in self._cppOp:
            histogram.append(op.getStatsHistogram())

        return histogram


class StaticGridPerTensorQuantizer(StaticGridTensorQuantizer):
    """
    Simulates quantization for the given tensor using a per-tensor scale/offset
    """

    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: QuantScheme, use_symmetric_encodings: bool,
                 enabled_by_default: bool, data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Constructor
        :param bitwidth: Quantization bitwidth
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. tf, tf_enhanced)
        :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
        """
        super(StaticGridPerTensorQuantizer, self).__init__(bitwidth, round_mode, quant_scheme, use_symmetric_encodings,
                                                           enabled_by_default, data_type)

        quant_scheme = MAP_QUANT_SCHEME_TO_PYMO[quant_scheme]
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

    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: QuantScheme, use_symmetric_encodings: bool,
                 num_channels: int, enabled_by_default: bool, ch_axis: int = 0):
        """
        Constructor
        :param bitwidth: Quantization bitwidth
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. tf, tf_enhanced)
        :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
        :param ch_axis: Channel Axis to use for per-channel quantization
        """
        super(StaticGridPerChannelQuantizer, self).__init__(bitwidth, round_mode, quant_scheme, use_symmetric_encodings,
                                                            enabled_by_default, data_type=QuantizationDataType.int)
        quant_scheme = MAP_QUANT_SCHEME_TO_PYMO[quant_scheme]
        self._cppOp = [AimetTensorQuantizer.AimetTensorQuantizer(quant_scheme) for _ in range(num_channels)]
        self._ch_axis = ch_axis
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
                tensor_slice = tensor.select(self._ch_axis, channel_idx).contiguous(memory_format=torch.contiguous_format)
                op.updateStats(tensor_slice, tensor.is_cuda)


class LearnedGridTensorQuantizer(TensorQuantizer):
    """
    Simulates quantization for a given tensor in the model, such that the scale/offset encodings are
    initialized and then "learnt" during training
    """

    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: QuantScheme, use_symmetric_encodings: bool,
                 enabled_by_default: bool, data_type: QuantizationDataType):
        """
        Constructor
        :param bitwidth: Quantization bitwidth
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. Range Learning)
        :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
        """

        if data_type != QuantizationDataType.int:
            raise ValueError('Only QuantizationDataType.int is supported for LearnedGridTensorQuantizer')

        super(LearnedGridTensorQuantizer, self).__init__(bitwidth, round_mode, quant_scheme, use_symmetric_encodings,
                                                         enabled_by_default, data_type)
        self.wrapper_ref = None
        self.name = None
        self.round_ste_func = ste.RoundStraightThrough.apply
        # p is the granularity/ steps (2^bw - 1)
        self.n, self.p = LearnedGridTensorQuantizer.get_n_and_p(self.bitwidth, self.use_symmetric_encodings)
        # Moving n and p device once so that we don't have to move it for every batch
        self.scaling, self.offset = None, None
        self.device = None
        # TODO pass it when instantiating the tensor quantizer
        self._ch_axis = 0

    @staticmethod
    def get_n_and_p(bitwidth, use_symmetric_encoding):
        """
        compute bounds n and p params given bitwidth and use_symmetric_encoding flag
        :param bitwidth: bitwidth configured
        :param use_symmetric_encoding: boolean flag indicates symmetric/ asymmetric encoding
        :return: n and p params computed as torch tensors
        """

        two_pow_bw = torch.pow(torch.Tensor([2]), bitwidth)
        two_pow_bw_minus_1 = torch.pow(torch.Tensor([2]), (bitwidth - 1))

        if use_symmetric_encoding:
            # symmetric case  : -2 ** (bw - 1) + 1  , p = 2 ** (bw - 1) - 1
            n = torch.add(-1.0 * two_pow_bw_minus_1, 1.0)
            p = torch.sub(two_pow_bw_minus_1, 1.0)
        else:
            # asymmetric case  : n = 0  , p = 2 ** (bw) - 1
            n = 0.0
            p = two_pow_bw - 1

        n = torch.Tensor([n])
        p = torch.Tensor([p])

        return n, p

    @property
    def encoding(self):
        """
        Gets the encodings from wrapper
        :return: encodings
        """
        # pylint:disable = protected-access
        if self.enabled:
            encoding_min = self.wrapper_ref._parameters[self.name + '_encoding_min']
            encoding_max = self.wrapper_ref._parameters[self.name + '_encoding_max']
            scaling, offsets = self.compute_scaling_offset(encoding_min, encoding_max)
            encodings = []
            for minimum, maximum, scale, offset in zip(encoding_min, encoding_max, scaling, offsets):
                tf_encoding = libpymo.TfEncoding()
                tf_encoding.min, tf_encoding.max, tf_encoding.offset, tf_encoding.delta, \
                tf_encoding.bw = minimum, maximum, offset, scale, self.bitwidth
                encodings.append(tf_encoding)
            # TODO: Remove when using only sequence of encodings (Done for backward compatibility)
            if len(encodings) == 1:
                encodings = encodings[0]
            return encodings
        return None

    @encoding.setter
    def encoding(self, encodings):
        """
        Sets encoding parameter using values obtained from encodings
        :param encodings: encodings value
        """
        # pylint:disable = protected-access
        if self.enabled:
            assert encodings is not None, "Encodings cannot be None if Quantizer is enabled"

            # pylint: disable = protected-access
            enc_min_param = self.name + '_encoding_min'
            enc_max_param = self.name + '_encoding_max'
            # TODO refactor to not call internal state of wrapper
            params = self.wrapper_ref._parameters

            # Todo: Remove this check when encodings is always a sequence
            if isinstance(encodings, List):
                assert isinstance(encodings[0], libpymo.TfEncoding), "Encodings should be a libpymo.TfEncoding() object"
                # Todo: Check for sequence
                encodings_min = [enc.min for enc in encodings]
                encodings_max = [enc.max for enc in encodings]
                self.bitwidth = encodings[0].bw
            else:
                assert isinstance(encodings, libpymo.TfEncoding), "Encodings should be a libpymo.TfEncoding() object"
                encodings_min = [encodings.min]
                encodings_max = [encodings.max]
                self.bitwidth = encodings.bw

            if params[enc_min_param] is None:
                params[enc_min_param] = torch.nn.Parameter(torch.FloatTensor(encodings_min).to(self.device),
                                                           requires_grad=True)

                params[enc_max_param] = torch.nn.Parameter(torch.FloatTensor(encodings_max).to(self.device),
                                                           requires_grad=True)

            else:
                params[enc_min_param].data = torch.FloatTensor(encodings_min)
                params[enc_max_param].data = torch.FloatTensor(encodings_max)

    def __str__(self):
        stream = io.StringIO(newline='\n')
        stream.write('LearnedGrid TensorQuantizer:\n')
        stream.write('    quant-scheme:{}, round_mode={}, bitwidth={}, enabled={}\n'.format(self._quant_scheme,
                                                                                            self.round_mode,
                                                                                            self.bitwidth,
                                                                                            self.enabled))
        if self.encoding:
            encoding = self.encoding
            # Todo: Remove this check when encodings is always a sequence
            if isinstance(encoding, libpymo.TfEncoding):
                encoding = [encoding]
            for tf_encoding in encoding:
                stream.write('    min:{}, max={}, delta={}, offset={}\n'.format(tf_encoding.min, tf_encoding.max,
                                                                                tf_encoding.delta, tf_encoding.offset))
        else:
            stream.write('    no encoding\n')

        return stream.getvalue()

    def compute_scaling_offset(self, encoding_min: float, encoding_max: float) -> [torch.Tensor, torch.Tensor]:
        """
        Computes scaling and offset for a given tensor using encoding min and max
        :param encoding_min: encoding min of a tensor
        :param encoding_max: encoding max of a tensor
        :return:
        """
        scaling = (encoding_max - encoding_min) / self.p.to(device=self.device)
        offset = self.round_ste_func(-encoding_min / scaling)
        return scaling, offset

    def quantize_dequantize(self, tensor: torch.Tensor, encoding_min: torch.nn.Parameter,
                            encoding_max: torch.nn.Parameter) -> torch.Tensor:
        """
        Quantize-dequantize the tensor, using the saved encoding for this tensor
        :param tensor: Tensor to quantize-dequantize
        :param encoding_min: minimum value of encoding for tensor
        :param encoding_max: maximum value of encoding for tensor
        :return: Quantized-dequantized tensor
        """
        self.p = self.p.to(tensor.device)
        self.n = self.n.to(tensor.device)

        if self.enabled:
            if encoding_max is None or encoding_min is None:
                raise RuntimeError("Forward pass used for compute_encodings differs from forward pass used during "
                                   "training")

            tensor = QuantizeDequantizeFunc.apply(tensor, encoding_min, encoding_max, self)
        return tensor


class QuantizeDequantizeFunc(torch.autograd.Function):
    """
    This functional is created explicitly for reducing the amount of intermediate tensors that would need to be
    maintained if we relied solely on PyTorch autograd for the forward function code.
    Instead, we specify a backward function that computes grads with respect to the tensor being quantized, and also
    the max and min learnable encodings.
    """

    # pylint:disable = arguments-differ
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, encoding_min: torch.nn.Parameter,
                encoding_max: torch.nn.Parameter, tensor_quantizer) -> torch.Tensor:
        # pylint: disable=protected-access
        ctx.save_for_backward(tensor, encoding_min, encoding_max, tensor_quantizer.n, tensor_quantizer.p,
                              torch.tensor(tensor_quantizer.bitwidth))

        delta = (encoding_max - encoding_min) / (2 ** tensor_quantizer.bitwidth - 1)
        offset = torch.round(-encoding_min / delta)
        if len(encoding_min) > 1:
            tensor = QuantizeDequantizeFunc._per_channel_quantize_dequantize(tensor, encoding_min, encoding_max, delta, offset, tensor_quantizer._ch_axis)
        else:
            tensor = QuantizeDequantizeFunc._per_tensor_quantize_dequantize(tensor, encoding_min, encoding_max, delta, offset)
        return tensor

    @staticmethod
    def _per_tensor_quantize_dequantize(tensor: torch.Tensor, encoding_min: torch.nn.Parameter, encoding_max: torch.nn.Parameter,
                                        delta: float, offset: float):
        """
        Quantize dequantize a tensor
        :param tensor: tensor which gets quantized and dequantized
        :param encoding_min: encoding's min
        :param encoding_max: encoding's max
        :param delta: encoding's delta
        :param offset: offset tensor
        :return: quantized dequantized tensor
        """
        tensor = torch.clamp(tensor, encoding_min.item(), encoding_max.item())
        tensor = torch.round(tensor / delta) + offset
        tensor = (tensor - offset) * delta
        return tensor

    @staticmethod
    def _per_channel_quantize_dequantize(tensor_to_quantize_dequantize: torch.Tensor, encoding_min: torch.nn.Parameter,
                                         encoding_max: torch.nn.Parameter, delta: torch.Tensor, offset: torch.Tensor, channel_axis: int):
        """
        Clamps tensor wrt min max per output channel
        :param tensor_to_quantize_dequantize: tensor which gets quantized and dequantized
        :param encoding_min: encoding's min
        :param encoding_max: encoding's max
        :param delta: encoding's delta
        :param offset: offset tensor
        :param channel_axis: Axis along which per channel quantize dequantize is performed
        :return: quantized dequantized tensor
        """
        quantized_tensors = []
        for i, minimum in enumerate(encoding_min):
            tensor_slice = tensor_to_quantize_dequantize.select(channel_axis, i).contiguous(memory_format=torch.contiguous_format)
            tensor = torch.clamp(tensor_slice, minimum.item(), encoding_max[i].item())
            tensor = torch.round(tensor / delta[i].item()) + offset[i].item()
            tensor = (tensor - offset[i].item()) * delta[i].item()
            quantized_tensors.append(tensor)
        quantized_tensor = torch.stack(tuple(quantized_tensors), dim=channel_axis)

        return quantized_tensor

    @staticmethod
    def backward(ctx, grad):
        # Retrieve saved tensors for gradient calculations
        tensor, encoding_min, encoding_max, n, p, bitwidth = ctx.saved_tensors

        scale = (encoding_max - encoding_min) / (2 ** bitwidth.data - 1)
        offset = torch.round(-encoding_min / scale)

        tensor_grad = ParameterQuantizer.compute_dloss_by_dx(tensor, grad, scale, offset, n, p)
        tensor_encoding_max_grad = ParameterQuantizer.compute_dloss_by_dmax(tensor, grad, scale, offset, n, p)
        tensor_encoding_min_grad = ParameterQuantizer.compute_dloss_by_dmin_using_dmax(tensor_encoding_max_grad)

        return tensor_grad, tensor_encoding_min_grad, tensor_encoding_max_grad, None


class ParameterQuantizer(torch.autograd.Function):
    """
    Helper class for simulating quantization for parameters for learned-grid quant wrappers
    """

    @staticmethod
    def compute_dloss_by_dmin_using_dmax(dloss_by_dmax):
        """
        compute derivative of loss w.r.t min, it is sign flipped version of derivative w.r.t max
        :param dloss_by_dmax derivative w.r.t max
        :return: derivative of loss w.r.t min
        """

        return -1 * dloss_by_dmax

    @staticmethod
    def compute_dloss_by_dmax(x, grad, scaling, offset, n, p):
        """
        helper function to compute derivative of loss w.r.t encoding max
        :param x: input
        :param grad: gradient
        :param scaling: scaling factor computed for given encoding min/max
        :param offset: offset computed
        :param n: lower bound
        :param p: upper bound
        :return: computed derivative of loss w.r.t encoding max
        """

        r_x_by_s_plus_round_o = torch.round(x / scaling) + torch.round(offset)
        # R(x/s)-(x/s)
        r_x_by_s_minus_x_by_s = torch.round(x / scaling) - (x / scaling)

        # compute dq_by_dmax
        # expr to be used if r_x_by_s_plus_round_o < n or > p
        false_expr = torch.clamp(r_x_by_s_plus_round_o.data, min=n.data.item(), max=p.data.item()) * (1 / p)
        inner_cond = torch.where(torch.le(r_x_by_s_plus_round_o.data, p), (r_x_by_s_minus_x_by_s * (1 / p)), false_expr)

        # we need a scalar value for dq_by_dmax, so reduce 4d value computed above
        # to single value before returning gradient
        # this uses chain rule, multiply by loss and sum it to get scalar.
        dq_by_dmax = torch.where(torch.le(n, r_x_by_s_plus_round_o.data), inner_cond, false_expr)
        dloss_by_dmax = torch.sum((dq_by_dmax * grad).flatten(), dim=0, keepdim=True)

        return dloss_by_dmax

    @staticmethod
    def compute_dloss_by_dx(x, grad, scaling, offset, n, p):
        """
        compute derivative w.r.t input
        :param grad: gradient
        :param scaling: scaling factor computed for given encoding min/max
        :param offset: offset computed
        :param n: lower bound
        :param p: upper bound
        :return: gradient w.r.t input
        """

        # R(x/s) + R(o)
        r_x_by_s_plus_round_o = torch.round(x / scaling) + offset

        # compute dloss_by_dx = dq_by_dx * grad
        inner_cond = torch.where(torch.le(r_x_by_s_plus_round_o.data, p.data),  # condition to check per value
                                 torch.ones_like(r_x_by_s_plus_round_o),  # execute if true
                                 torch.zeros_like(r_x_by_s_plus_round_o))  # execute if false

        dloss_by_dx = torch.where(torch.le(n.data, r_x_by_s_plus_round_o.data),  # condition to check per value
                                  inner_cond,  # execute if true
                                  torch.zeros_like(r_x_by_s_plus_round_o.data)) * grad

        return dloss_by_dx

    @staticmethod
    def compute_gradients(tensor: torch.Tensor, tensor_quantizer: LearnedGridTensorQuantizer, grad: torch.Tensor):
        """
        Computes gradients for tensor
        :param tensor: Given tensor
        :param tensor_quantizer: Trainable Tensor quantizer which holds encoding, n and p values
        :param grad: gradient using which other gradients will be calculated
        :return: grad with respect to tensor, grad of encoding min and max
        """
        device = tensor.device
        scaling, offset = tensor_quantizer.scaling, tensor_quantizer.offset
        tensor_quantizer.n, tensor_quantizer.p = tensor_quantizer.n.to(device), tensor_quantizer.p.to(device)
        tensor_grad = ParameterQuantizer.compute_dloss_by_dx(tensor, grad, scaling, offset, tensor_quantizer.n,
                                                             tensor_quantizer.p)
        tensor_encoding_max_grad = ParameterQuantizer.compute_dloss_by_dmax(tensor, grad, scaling, offset,
                                                                            tensor_quantizer.n, tensor_quantizer.p)

        tensor_encoding_min_grad = ParameterQuantizer.compute_dloss_by_dmin_using_dmax(tensor_encoding_max_grad)
        return tensor_grad, tensor_encoding_min_grad, tensor_encoding_max_grad

    @staticmethod
    def quantize_parameters(trainable_wrapper, encoding_params: List):
        """
        Quantizes layer parameters
        :param trainable_wrapper: LearnedGridQuantWrapper
        :param encoding_params: encoding min and max defined as torch parameters
        :return: original layer parameters values
        """
        # pylint:disable = protected-access
        for index, named_param in enumerate(trainable_wrapper._module_to_wrap.named_parameters()):
            name, param = named_param
            if trainable_wrapper.param_quantizers[name].enabled:
                encoding_min = encoding_params[index * 2]
                encoding_max = encoding_params[index * 2 + 1]
                param_quantizer = trainable_wrapper.param_quantizers[name]
                param_quantizer.scaling, param_quantizer.offset = \
                    param_quantizer.compute_scaling_offset(encoding_min.item(), encoding_max.item())
                param.data = param_quantizer.quantize_dequantize(param.data, encoding_min, encoding_max)

    @staticmethod
    def backward_pass_for_parameters(trainable_wrapper, shadow_params: Dict):
        """
        Calls custom gradient computation for each parameter
        :param trainable_wrapper: LearnedGridQuantWrapper
        :param shadow_params: Dictionary of original parameters of layer
        :return: Encoding min max params as list
        """
        param_encoding_grads = []
        # pylint:disable = protected-access
        for name, param in trainable_wrapper._module_to_wrap.named_parameters():
            param_quantizer = trainable_wrapper.param_quantizers[name]
            if param_quantizer.enabled:
                param.grad = ste.compute_dloss_by_dx(param, param.grad, param_quantizer.encoding.min,
                                                     param_quantizer.encoding.max)
                output_grad = param.grad

                _, param_encoding_min_grad, param_encoding_max_grad = ParameterQuantizer.compute_gradients(
                    shadow_params[name], param_quantizer, output_grad)
                param_encoding_grads.append(param_encoding_min_grad)
                param_encoding_grads.append(param_encoding_max_grad)
        return param_encoding_grads

    # pylint:disable = arguments-differ
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, trainable_wrapper, shadow_params: Dict,
                *encoding_params):
        """
        :param ctx: Context manager
        :param input_tensor: Input to the layer
        :param trainable_wrapper: LearnedGridQuantWrapper Wrapped around the layer
        :param shadow_params: Dict of original parameter values
        :param encoding_params: Unpacked List of Encoding min and max of parameters
        :return:
        """

        ParameterQuantizer.quantize_parameters(trainable_wrapper, encoding_params)
        ctx.trainable_module = trainable_wrapper
        ctx.shadow_params = shadow_params

        return input_tensor

    @staticmethod
    def backward(ctx, *output_grad):
        # Retrieve saved tensors for gradient calculations
        trainable_wrapper = ctx.trainable_module

        # Custom backward for parameters
        param_encoding_grads = ParameterQuantizer.backward_pass_for_parameters(trainable_wrapper,
                                                                               ctx.shadow_params)

        return (*output_grad, None, None, *param_encoding_grads)


class QuantizeDequantize(torch.autograd.Function):
    """
    Custom gradient function for STE
    """

    @staticmethod
    def _per_tensor_quantize_dequantize(tensor, tensor_quantizer, round_mode):
        """
        If the quantization data type is floating point, then call the pytorch functions to
        perform quantization followed by dequantization. Else call the custom function to
        get the new tensor
        """
        # pylint:disable = protected-access
        if tensor_quantizer.data_type == QuantizationDataType.float:
            if tensor_quantizer.bitwidth != 16:
                raise ValueError('float data_type only supports bitwidth=16')
            quantized_tensor = tensor.half()
            quantized_tensor = quantized_tensor.float()
        else:
            quantized_tensor = tensor_quantizer._cppOp[0].quantizeDequantize(tensor, tensor_quantizer.encoding,
                                                                             round_mode, tensor.is_cuda)
        return quantized_tensor

    @staticmethod
    def _per_channel_quantize_dequantize(tensor, tensor_quantizer, round_mode):
        quantized_tensors = []

        if tensor_quantizer.data_type == QuantizationDataType.float:
            raise ValueError('float data_type is not supported for per channel quantize-dequantize')

        # pylint:disable = protected-access
        for index, op in enumerate(tensor_quantizer._cppOp):
            tensor_slice = tensor.select(tensor_quantizer._ch_axis, index).contiguous(memory_format=torch.contiguous_format)
            # pylint:disable = protected-access
            computed_tensor = op.quantizeDequantize(tensor_slice, tensor_quantizer._encoding[index],
                                                    round_mode, tensor.is_cuda)
            quantized_tensors.append(computed_tensor)
        quantized_tensor = torch.stack(tuple(quantized_tensors), dim=tensor_quantizer._ch_axis)
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
        tensor_quantizer = ctx.tensor_quantizer
        if tensor_quantizer.enabled and tensor_quantizer.data_type == QuantizationDataType.int:
            tensor = ctx.saved_tensors
            # pylint: disable=protected-access
            if len(tensor_quantizer._cppOp) > 1:
                ch_axis = tensor_quantizer._ch_axis
            else:
                ch_axis = 0
            grad = ste.compute_dloss_by_dx(tensor[0], output_grad, tensor_quantizer.encoding.min,
                                           tensor_quantizer.encoding.max, ch_axis)
        else:
            grad = output_grad

        return grad, None, None


class Quantize(torch.autograd.Function):
    """
    Custom gradient function for STE
    """

    # pylint:disable = arguments-differ
    @staticmethod
    def forward(ctx, tensor, tensor_quantizer, round_mode):
        """
        Quantize the tensor, using the saved encoding for this tensor
        :param tensor: Tensor to quantize
        :param tensor_quantizer: Reference to the tensor quantizer
        :param round_mode: Rounding mode
        :return: Resulting tensor
        """
        if tensor_quantizer.enabled:
            # pylint:disable = protected-access
            shift_to_signed = False
            if tensor_quantizer.use_symmetric_encodings and tensor_quantizer.encoding.offset < 0:
                shift_to_signed = True
            quantized_tensor = tensor_quantizer._cppOp[0].quantize(tensor, tensor_quantizer.encoding, round_mode,
                                                                   tensor.is_cuda, shift_to_signed)
        else:
            quantized_tensor = tensor

        ctx.save_for_backward(quantized_tensor)

        ctx.tensor_quantizer = tensor_quantizer
        return quantized_tensor

    @staticmethod
    def backward(ctx, output_grad):
        _logger.error('Backward pass for quantize only not implemented')
        raise AssertionError
