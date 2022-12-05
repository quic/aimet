# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
import functools
import io
from typing import List, Union, Tuple

import torch

import aimet_common.AimetTensorQuantizer as AimetTensorQuantizer
import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme, QuantizationDataType, MAP_QUANT_SCHEME_TO_PYMO
from aimet_common.utils import AimetLogger
import aimet_torch.quantsim_straight_through_grad as grad_fn
from aimet_torch.quantsim_straight_through_grad import IntermediateResult
from aimet_torch.fp_quantization import fp8_quantizer, INIT_MAP


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
        :param data_type: Data type for quantization (e.g. int, float)
        """
        super(TensorQuantizer, self).__init__()
        self.round_mode = round_mode
        self._quant_scheme = quant_scheme
        self.use_symmetric_encodings = use_symmetric_encodings
        self.use_strict_symmetric = False
        self.use_unsigned_symmetric = False
        # NOTE: is_unsigned_symmetric flag is to check feasibility about unsigned symmetric quantization is possible
        #   The difference between use_unsigned_symmetric and is_unsigned_symmetric is
        #   is_unsigned_symmetric can be false if encoding_min < 0 < encoding_max range
        #   even if use_unsigned_symmetric is set to true.
        #
        #   In other words, is_unsigned_symmetric can be true
        #   when use_symmetric_encodings=True and use_unsigned_symmetric=True and (encoding_min >= 0 and encoding_max >= 0)
        self.is_unsigned_symmetric = False
        self.bitwidth = bitwidth
        self.enabled = enabled_by_default
        self.data_type = data_type
        self._is_encoding_frozen = False

    @property
    def quant_scheme(self):
        """Accessor to self._quant_scheme"""
        return self._quant_scheme

    @property
    def is_encoding_frozen(self):
        """Accessor to self._is_encoding_frozen"""
        return self._is_encoding_frozen

    @property
    def channel_axis(self):
        """ Returns channel axis, default None unless for per-channel quantizers """
        return None


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
        self.fp8_maxval = None

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
    def encoding(self) -> Union[None, libpymo.TfEncoding, List[libpymo.TfEncoding]]:
        """
        Property to get encoding.
        :return: One (per-tensor) or list of many (per-channel) encoding(s).
        """
        return self._encoding

    @encoding.setter
    def encoding(self, encoding) -> None:
        """
        Property to set encoding.
        :param encoding: One (per-tensor) or list of many (per-channel) encoding(s).
        """
        if self._is_encoding_frozen:
            raise RuntimeError("Encoding can be set only when it is not frozen.")

        self._encoding = encoding

    def compute_encoding(self):
        """
        Compute the quantization encoding for this tensor
        """
        if self.enabled and not self._is_encoding_frozen:
            self._encoding = []
            if self.data_type == QuantizationDataType.float:
                if self.bitwidth == 16:
                    self._encoding = None
                elif self.bitwidth == 8:
                    self._encoding = [libpymo.TfEncoding()]
                else:
                    raise ValueError("Only bitwidths [8, 16] allowed for float data type, not ", str(self.bitwidth))
            else:
                for op in self._cppOp:
                    encoding, is_encoding_valid = op.getEncoding(self.bitwidth, self.use_symmetric_encodings,
                                                                 self.use_strict_symmetric,
                                                                 self.use_unsigned_symmetric)

                    if not is_encoding_valid:
                        self.enabled = False
                    else:
                        self._encoding.append(encoding)

                # NOTE: Check feasibility about unsigned symmetric case
                #   whether encoding range is all positive
                is_in_positive_range = lambda enc_min, enc_max: enc_min >= 0 and enc_max >= 0
                self.is_unsigned_symmetric = self.use_symmetric_encodings and \
                                             self.use_unsigned_symmetric and \
                                             all([is_in_positive_range(enc.min, enc.max) for enc in self._encoding])

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

    def freeze_encoding(self) -> None:
        """
        Freeze the encoding.
        """
        if not self._encoding:
            raise RuntimeError("Encoding can be frozen only when it is not None.")

        self._is_encoding_frozen = True

    def set_percentile_value(self, percentile_value: float):
        """
        Set the percentile value to be used while computing encodings
        """
        for op in self._cppOp:
            op.setPercentileValue(percentile_value)

    def update_maxval(self, maxval):
        """
        Update the self.fp8_maxval member with a new value
        """
        if self.fp8_maxval is None:
            self.fp8_maxval = maxval
        else:
            self.fp8_maxval = 0.9 * self.fp8_maxval + 0.1 * maxval


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

    @property
    def encoding(self) -> Union[None, libpymo.TfEncoding]:
        """
        Property to get encoding.
        :return: Encoding.
        """
        if self._encoding:
            return self._encoding[0]

        return None

    @encoding.setter
    def encoding(self, encoding: Union[libpymo.TfEncoding, List[libpymo.TfEncoding]]) -> None:
        """
        Property to set encoding.
        :param encoding: Encoding.
        """
        if self._is_encoding_frozen:
            raise RuntimeError("Encoding can be set only when it is not frozen.")
        if isinstance(encoding, list) and len(encoding) == 1:
            self._encoding = encoding
        else:
            self._encoding = [encoding]

    def update_encoding_stats(self, tensor):
        """
        Update the stats for computing encoding
        :param tensor: Tensor to use for updating the encodings stats
        """
        if self.enabled and not self._is_encoding_frozen:
            if self.data_type == QuantizationDataType.float:
                if self.bitwidth == 8:
                    maxval = INIT_MAP[self.quant_scheme](tensor, self, False).to(tensor.device)
                    self.update_maxval(maxval)
                    ec = libpymo.TfEncoding()
                    ec.max = float(self.fp8_maxval)
                    ec.min = -ec.max
                    self.encoding = ec
            else:
                for op in self._cppOp:
                    op.updateStats(tensor, tensor.is_cuda)


class StaticGridPerChannelQuantizer(StaticGridTensorQuantizer):
    """
    Simulates quantization for the given tensor using a per-channel scale/offset
    """

    # pylint: disable=too-many-arguments
    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: QuantScheme, use_symmetric_encodings: bool,
                 num_channels: int, enabled_by_default: bool, ch_axis: int = 0,
                 data_type: QuantizationDataType = QuantizationDataType.int):
        """
        Constructor
        :param bitwidth: Quantization bitwidth
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. tf, tf_enhanced)
        :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
        :param ch_axis: Channel Axis to use for per-channel quantization
        :param data_type: data type of type QuantizationDataType to be used
        """
        super(StaticGridPerChannelQuantizer, self).__init__(bitwidth, round_mode, quant_scheme, use_symmetric_encodings,
                                                            enabled_by_default, data_type=data_type)
        quant_scheme = MAP_QUANT_SCHEME_TO_PYMO[quant_scheme]
        self._cppOp = [AimetTensorQuantizer.AimetTensorQuantizer(quant_scheme) for _ in range(num_channels)]
        self._ch_axis = ch_axis

    @property
    def encoding(self) -> Union[None, List[libpymo.TfEncoding]]:
        """
        Property to get encoding.
        :return: List of Encoding(s).
        """
        return self._encoding

    @encoding.setter
    def encoding(self, encoding: List[libpymo.TfEncoding]) -> None:
        """
        Property to set encoding.
        :param encoding: List of Encoding(s).
        """
        if self._is_encoding_frozen:
            raise RuntimeError("Encoding can be set only when it is not frozen.")

        self._encoding = encoding

    @property
    def channel_axis(self) -> int:
        """ Return private member _ch_axis """
        return self._ch_axis

    def update_encoding_stats(self, tensor):
        """
        Update the stats for computing encoding
        :param tensor: Tensor to use for updating the encodings stats
        """
        if self.enabled and not self._is_encoding_frozen:
            if self.data_type == QuantizationDataType.float:
                if self.bitwidth == 8:
                    maxval = INIT_MAP[self.quant_scheme](tensor, self, True).to(tensor.device)
                    self.update_maxval(maxval)
                    ecs = [libpymo.TfEncoding() for _ in range(self.fp8_maxval.shape[0])]
                    for idx, ec in enumerate(ecs):
                        ec.max = float(self.fp8_maxval[idx])
                        ec.min = -ec.max
                    self.encoding = ecs
            else:
                for channel_idx, op in enumerate(self._cppOp):
                    tensor_slice = tensor.select(self._ch_axis, channel_idx).contiguous(
                        memory_format=torch.contiguous_format)
                    op.updateStats(tensor_slice, tensor.is_cuda)


class LearnedGridTensorQuantizer(TensorQuantizer):
    """
    Simulates quantization for a given tensor in the model, such that the scale/offset encodings are
    initialized and then "learnt" during training
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: QuantScheme, use_symmetric_encodings: bool,
                 enabled_by_default: bool, data_type: QuantizationDataType):
        """
        Constructor
        :param bitwidth: Quantization bitwidth
        :param round_mode: Rounding mode (e.g. Nearest)
        :param quant_scheme: Quantization scheme (e.g. Range Learning)
        :param use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        :param enabled_by_default: True if quantization of tensor is enabled.  False otherwise.
        :param data_type: Type for quantization (e.g. int, float)
        """

        if data_type != QuantizationDataType.int:
            raise ValueError('Only QuantizationDataType.int is supported for LearnedGridTensorQuantizer')

        super(LearnedGridTensorQuantizer, self).__init__(bitwidth, round_mode, quant_scheme, use_symmetric_encodings,
                                                         enabled_by_default, data_type)
        self.wrapper_ref = None
        self.name = None
        self.round_ste_func = grad_fn.RoundStraightThrough.apply
        self.scaling, self.offset = None, None
        self.device = None
        self._ch_axis = 0

    @staticmethod
    @functools.lru_cache()
    def get_n_and_p(bitwidth: int,
                    use_symmetric_encoding: bool,
                    use_strict_symmetric: bool,
                    device: Union[torch.device, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute bounds n and p params given bitwidth and use_symmetric_encoding flag
        :param bitwidth: bitwidth configured
        :param use_symmetric_encoding: boolean flag indicates symmetric/asymmetric encoding
        :param use_strict_symmetric: boolean flag indicates strict or not when symmetric encoding
        :return: n and p params computed as torch tensors
        """
        if not use_symmetric_encoding and use_strict_symmetric:
            raise ValueError("Strict symmetric can be enabled only when using symmetric encoding")

        n = 0.0
        p = torch.pow(torch.tensor([2]), bitwidth) - 1

        if use_symmetric_encoding and use_strict_symmetric:
            p -= 1

        n = torch.tensor([n], device=device)
        p = torch.tensor([p], device=device)

        return n, p

    def n(self, device=None) -> torch.Tensor:
        """ Get n """
        n, _ = self.get_n_and_p(self.bitwidth, self.use_strict_symmetric, self.use_strict_symmetric, device or self.device)
        return n

    def p(self, device=None) -> torch.Tensor:
        """ Get p """
        _, p = self.get_n_and_p(self.bitwidth, self.use_strict_symmetric, self.use_strict_symmetric, device or self.device)
        return p

    @property
    def encoding(self) -> Union[None, libpymo.TfEncoding, List[libpymo.TfEncoding]]:
        """
        NOTE: encoding.getter first compute updated encoding and then return it.

        Property to get learned (up-to-date) encoding computed from encoding min and max parameters.
        :return: encoding(s).
        """
        # pylint:disable=protected-access
        if self.enabled:
            encoding = self._compute_updated_encoding()
            return encoding

        return None

    @encoding.setter
    def encoding(self, encoding: Union[libpymo.TfEncoding, List[libpymo.TfEncoding]]):
        """
        Property to set encoding.
        encoding.setter also sets encoding min and max parameters and recompute
        p and n tensors.
        :param encoding: encodings.
        """
        if self.enabled:
            if encoding is None:
                raise RuntimeError("Encodings cannot be None if Quantizer is enabled.")

            bitwidth = encoding[0].bw if isinstance(encoding, List) else encoding.bw

            if bitwidth != self.bitwidth:
                raise RuntimeError(
                    f"Bitwidth mismatched. The bitwidth for quantizer is {self.bitwidth}, but the bitwidth in encodings is {bitwidth}. "
                    f"If the intent is to change the bitwidth, please set quantizer bitwidth to {bitwidth} first."
                )

            if self._is_encoding_frozen:
                raise RuntimeError("Encoding can be set only when it is not frozen.")

            self._set_encoding_min_max_parameters(encoding)

    @property
    def channel_axis(self) -> int:
        """
        Accessor to self._ch_axis
        """
        return self._ch_axis

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

    def compute_scaling_offset(self, encoding_min: torch.Tensor, encoding_max: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes scaling and offset for a given tensor using encoding min and max
        :param encoding_min: encoding min of a tensor
        :param encoding_max: encoding max of a tensor
        :return:
        """
        scaling, offset, _ = grad_fn.get_computed_encodings(self.bitwidth, encoding_min, encoding_max, self.use_symmetric_encodings,
                                                            self.use_strict_symmetric, self.is_unsigned_symmetric)
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
        if self.enabled:
            if encoding_max is None or encoding_min is None:
                raise RuntimeError("Forward pass used for compute_encodings differs from forward pass used during "
                                   "training")
            tensor = QuantizeDequantizeFunc.apply(tensor, encoding_min, encoding_max, self)
        return tensor

    def _compute_updated_encoding(self) -> Union[libpymo.TfEncoding, List[libpymo.TfEncoding]]:
        """
        Computes updated encoding from encoding min and max parameters.
        :return: Up-to-date (learned) encoding(s).
        """
        # pylint:disable=protected-access

        encoding_min = getattr(self.wrapper_ref, self.name + '_encoding_min')
        encoding_max = getattr(self.wrapper_ref, self.name + '_encoding_max')

        encodings = []
        for minimum, maximum in zip(encoding_min, encoding_max):
            tf_encoding = libpymo.TfEncoding()
            scale, offset = self.compute_scaling_offset(minimum, maximum)
            tf_encoding.min, tf_encoding.max, tf_encoding.offset, tf_encoding.delta, \
            tf_encoding.bw = minimum, maximum, offset, scale, self.bitwidth
            encodings.append(tf_encoding)

        # TODO: Remove when using only sequence of encodings (Done for backward compatibility)
        if len(encodings) == 1:
            encodings = encodings[0]

        return encodings

    def _set_encoding_min_max_parameters(self, encodings: Union[libpymo.TfEncoding, List[libpymo.TfEncoding]]) -> None:
        """
        Set encoding min and max parameters.
        :param encodings: Encoding(s).
        """
        # pylint: disable=protected-access
        enc_min_param = self.name + '_encoding_min'
        enc_max_param = self.name + '_encoding_max'
        # TODO: refactor to not call internal state of wrapper
        params = self.wrapper_ref._parameters

        # TODO: Remove this check when encodings is always a sequence
        if isinstance(encodings, List):
            assert isinstance(encodings[0], libpymo.TfEncoding), "Encodings should be a libpymo.TfEncoding() object"
            # TODO: Check for sequence
            encodings_min = [enc.min for enc in encodings]
            encodings_max = [enc.max for enc in encodings]
        else:
            assert isinstance(encodings, libpymo.TfEncoding), "Encodings should be a libpymo.TfEncoding() object"
            encodings_min = [encodings.min]
            encodings_max = [encodings.max]

        params[enc_min_param] = torch.nn.Parameter(torch.FloatTensor(encodings_min).to(self.wrapper_ref.device),
                                                   requires_grad=True)
        params[enc_max_param] = torch.nn.Parameter(torch.FloatTensor(encodings_max).to(self.wrapper_ref.device),
                                                   requires_grad=True)

    def freeze_encoding(self):
        """
        Freeze encoding min and max parameters.
        """
        # pylint:disable=protected-access
        enc_min_param = self.name + '_encoding_min'
        enc_max_param = self.name + '_encoding_max'
        # TODO: refactor to not call internal state of wrapper.
        params = self.wrapper_ref._parameters

        if not params[enc_min_param] and not params[enc_max_param]:
            raise RuntimeError("Encoding can be frozen only when it is not None.")

        self._is_encoding_frozen = True
        params[enc_min_param].requires_grad = False
        params[enc_max_param].requires_grad = False



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
                encoding_max: torch.nn.Parameter, tensor_quantizer: LearnedGridTensorQuantizer) -> torch.Tensor:
        x_dequant, intermediate_result = grad_fn.calculate_forward_pass(tensor, tensor_quantizer,
                                                                        encoding_min, encoding_max)

        ctx.channel_axis = tensor_quantizer.channel_axis
        ctx.is_symmetric = intermediate_result.is_symmetric
        ctx.is_unsigned = intermediate_result.is_unsigned
        ctx.save_for_backward(tensor, intermediate_result.x_quant,
                              intermediate_result.delta, intermediate_result.offset,
                              intermediate_result.encoding_min, intermediate_result.encoding_max,
                              intermediate_result.mask_tensor, intermediate_result.num_steps)

        return x_dequant

    @staticmethod
    def backward(ctx, grad):
        # pylint: disable=too-many-locals
        # Retrieve saved tensors for gradient calculation
        tensor, x_quant, delta, offset, encoding_min, encoding_max, mask_tensor, num_steps = ctx.saved_tensors
        channel_axis = ctx.channel_axis
        is_symmetric = ctx.is_symmetric
        is_unsigned = ctx.is_unsigned

        intermediate_result = IntermediateResult(x_quant, encoding_min, encoding_max,
                                                 delta, offset, mask_tensor, num_steps,
                                                 is_symmetric, is_unsigned)

        tensor_grad, tensor_encoding_min_grad, tensor_encoding_max_grad = \
            grad_fn.calculate_gradients(tensor, grad, intermediate_result, channel_axis)

        return tensor_grad, tensor_encoding_min_grad, tensor_encoding_max_grad, None


class ParameterQuantizer(torch.autograd.Function):
    """
    Helper class for simulating quantization for parameters for learned-grid quant wrappers
    """
    @staticmethod
    def compute_gradients(tensor: torch.Tensor,
                          grad: torch.Tensor,
                          intermediate_result: IntermediateResult,
                          channel_axis: int):
        """
        Compute gradients of encoding min/max
        :param tensor: Given tensor
        :param grad: Gradient using which other gradients will be calculated
        :param intermediate_result: Intermediate result from forward pass
        :param channel_axis: Channel axis
        :return: grad with respect to tensor, grad of encoding min and max
        """
        tensor_grad, tensor_encoding_min_grad, tensor_encoding_max_grad = \
            grad_fn.calculate_gradients(tensor, grad, intermediate_result, channel_axis)

        tensor.grad = tensor_grad
        return tensor_encoding_min_grad, tensor_encoding_max_grad

    @staticmethod
    def quantize_parameters(trainable_wrapper, encoding_params: List):
        """
        Quantizes layer parameters
        :param trainable_wrapper: LearnedGridQuantWrapper
        :param encoding_params: encoding min and max defined as torch parameters
        :return: original layer parameters values
        """
        # pylint:disable = protected-access
        for index, named_param in enumerate(trainable_wrapper.get_named_parameters()):
            name, param = named_param
            if trainable_wrapper.param_quantizers[name].enabled:
                encoding_min = encoding_params[index * 2]
                encoding_max = encoding_params[index * 2 + 1]

                param_quantizer = trainable_wrapper.param_quantizers[name]
                param_quantizer.scaling, param_quantizer.offset = \
                    param_quantizer.compute_scaling_offset(encoding_min, encoding_max)

                if hasattr(trainable_wrapper, '_is_replica') and trainable_wrapper._is_replica:
                    param_tensor = param.data.clone()
                else:
                    param_tensor = param.data

                param.data = param_quantizer.quantize_dequantize(param_tensor, encoding_min, encoding_max)

    @staticmethod
    def backward_pass_for_parameters(trainable_wrapper):
        """
        Calls custom gradient computation for each parameter
        :param trainable_wrapper: LearnedGridQuantWrapper
        :return: Encoding min max params as list
        """
        param_encoding_grads = []
        for name, param in trainable_wrapper.get_named_parameters():
            param_quantizer = trainable_wrapper.param_quantizers[name]
            if param_quantizer.enabled and param.grad is not None:
                encoding_min = getattr(trainable_wrapper, f"{name}_encoding_min")
                encoding_max = getattr(trainable_wrapper, f"{name}_encoding_max")

                _, intermediate_result = grad_fn.calculate_forward_pass(param, param_quantizer, encoding_min, encoding_max)
                param_encoding_min_grad, param_encoding_max_grad = ParameterQuantizer.compute_gradients(param, param.grad,
                                                                                                        intermediate_result,
                                                                                                        param_quantizer.channel_axis)
                param_encoding_grads.append(param_encoding_min_grad)
                param_encoding_grads.append(param_encoding_max_grad)
            elif param_quantizer.enabled:
                param_encoding_grads.append(None)
                param_encoding_grads.append(None)
        return param_encoding_grads

    # pylint:disable = arguments-differ
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, trainable_wrapper, *encoding_params):
        """
        :param ctx: Context manager
        :param input_tensor: Input to the layer
        :param trainable_wrapper: LearnedGridQuantWrapper Wrapped around the layer
        :param encoding_params: Unpacked List of Encoding min and max of parameters
        :return:
        """

        ParameterQuantizer.quantize_parameters(trainable_wrapper, encoding_params)
        ctx.trainable_module = trainable_wrapper

        return input_tensor

    @staticmethod
    def backward(ctx, *output_grad):
        # Retrieve saved tensors for gradient calculations
        trainable_wrapper = ctx.trainable_module

        # Custom backward for parameters
        param_encoding_grads = ParameterQuantizer.backward_pass_for_parameters(trainable_wrapper)

        return (*output_grad, None, *param_encoding_grads)


class QuantizeDequantize(torch.autograd.Function):
    """
    Custom gradient function for STE
    """

    @staticmethod
    def _quantize_float(tensor, tensor_quantizer, per_channel):
        if tensor_quantizer.bitwidth == 16:
            quantized_tensor = tensor.half()
            quantized_tensor = quantized_tensor.float()
        elif tensor_quantizer.bitwidth == 8:
            quantized_tensor = fp8_quantizer(tensor, tensor_quantizer, per_channel)
        else:
            raise ValueError('float data_type only supports bitwidth in {16, 8}')

        return quantized_tensor

    @staticmethod
    def _per_tensor_quantize_dequantize(tensor, tensor_quantizer, round_mode):
        """
        If the quantization data type is floating point, then call the pytorch functions to
        perform quantization followed by dequantization. Else call the custom function to
        get the new tensor
        """
        # pylint:disable = protected-access
        if tensor_quantizer.data_type == QuantizationDataType.float:
            quantized_tensor = QuantizeDequantize._quantize_float(tensor, tensor_quantizer, False)
        else:
            quantized_tensor = tensor_quantizer._cppOp[0].quantizeDequantize(tensor, tensor_quantizer.encoding,
                                                                             round_mode, tensor.is_cuda)
        return quantized_tensor

    @staticmethod
    def _per_channel_quantize_dequantize(tensor, tensor_quantizer, round_mode):
        if tensor_quantizer.data_type == QuantizationDataType.float:
            quantized_tensor = QuantizeDequantize._quantize_float(tensor, tensor_quantizer, True)
        else:
            quantized_tensors = []
            # pylint: disable=protected-access
            for index, op in enumerate(tensor_quantizer._cppOp):
                # pylint: disable=protected-access
                tensor_slice = tensor.select(tensor_quantizer._ch_axis, index).contiguous(memory_format=torch.contiguous_format)
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
            if isinstance(tensor_quantizer, StaticGridPerChannelQuantizer):
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
            if isinstance(tensor_quantizer, StaticGridPerChannelQuantizer):
                ch_axis = tensor_quantizer._ch_axis
            else:
                ch_axis = 0
            grad = grad_fn.compute_dloss_by_dx(tensor[0], output_grad, tensor_quantizer.encoding.min,
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
        raise AssertionError('Backward pass for quantize only not implemented')
