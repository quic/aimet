# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=redefined-builtin
""" Affine quantizers """

import abc
import math
from typing import Optional, List, Dict
import contextlib
import functools

import torch
from torch import nn

from aimet_torch.v2.utils import patch_attr, _is_expandable, StatisticsNotFoundError
from aimet_torch.v2.quantization.encoding_analyzer import EncodingAnalyzer, MinMaxEncodingAnalyzer
from aimet_torch.v2.quantization.affine import AffineEncoding
from aimet_torch.v2.quantization.tensor import QuantizedTensor, DequantizedTensor
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine.backends import quantize, quantize_dequantize, torch_builtins
from aimet_torch.v2.utils import ste_round


__all__ = ['AffineQuantizerBase', 'MinMaxQuantizer', 'Quantize', 'QuantizeDequantize', 'Dequantize',
           'LpbqQuantizeDequantize']


class AffineQuantizerBase(QuantizerBase):
    """
    Base class for linear quantization modules.

    :param shape: Shape of the quantization parameters.
    :param bitwidth: Quantization bitwidth.
    :param symmetric: If True, performs symmetric quantization;
                      otherwise, performs asymmetric quantization.
    :param encoding_analyzer: Encoding analyzer for calibrating quantization encodings.
                              (default: absolute min-max encoding analyzer)
    """
    def __init__(self, shape, bitwidth: int, symmetric: bool, encoding_analyzer: EncodingAnalyzer = None,
                 block_size: Optional[list] = None):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.block_size = block_size
        self.bitwidth = bitwidth
        self._symmetric = symmetric
        # We support two quantization modes: (unsigned) asymmetric and signed-symmetric
        self._signed = symmetric

        self.encoding_analyzer = encoding_analyzer or \
                                 MinMaxEncodingAnalyzer(torch_builtins.get_encoding_shape_with_blocks(self.shape,
                                                                                                      self.block_size))

        if self.block_size is None and not _is_expandable(self.encoding_analyzer.observer.shape, self.shape):
            raise RuntimeError(f'Encoding analyzer of shape {self.encoding_analyzer.observer.shape} '
                               f'is incompatible with quantizer of shape {self.shape}.')

    @abc.abstractmethod
    def get_min(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization min to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        :param dtype: dtype of the computed min
        :return: Quantization min
        """

    @abc.abstractmethod
    def get_max(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization max to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        :param dtype: dtype of the computed max
        :return: Quantization max
        """

    @abc.abstractmethod
    def get_scale(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization scale to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        :param dtype: dtype of the computed scale
        :return: Quantization scale
        """

    @abc.abstractmethod
    def get_offset(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization offset to be used for forward pass.
        Return None f the quantizer is not initialized yet.

        :param dtype: dtype of the computed offset
        :return: Quantization offset
        """

    @abc.abstractmethod
    def set_range(self, min: torch.Tensor, max: torch.Tensor):
        """
        Set quantization parameters to the given min-max range
        """

    def get_encoding(self) -> Optional[AffineEncoding]:
        """
        Return the quantizer's encodings as an AffineEncoding object
        """
        if self.is_initialized():
            return AffineEncoding(self.get_scale(dtype=torch.float32),
                                  self.get_offset(dtype=torch.float32),
                                  self.bitwidth, self._signed, self._symmetric, self.block_size)
        return None

    @torch.no_grad()
    def get_legacy_encodings(self) -> Optional[List[Dict]]:
        """
        Returns a list of encodings, each represented as a List of Dicts
        """
        # pylint: disable=redefined-builtin

        if not self.is_initialized():
            return None

        min = self.get_min(dtype=torch.float32).flatten()
        max = self.get_max(dtype=torch.float32).flatten()
        scale = self.get_scale(dtype=torch.float32).flatten()
        offset = self.get_offset(dtype=torch.float32).flatten()
        if self._signed: # Legacy behavior is to use offset = 2 ** (bitwidth - 1) for signed symmetric
            offset -= 2 ** (self.bitwidth - 1)
        bitwidth = self.bitwidth
        dtype = "int"
        is_symmetric = self.symmetric

        return [
            {'min': float(min_), 'max': float(max_),
             'scale': float(scale_), 'offset': int(offset_),
             'bitwidth': bitwidth, 'dtype': dtype, 'is_symmetric': str(is_symmetric)}
            for min_, max_, scale_, offset_ in zip(min, max, scale, offset)
        ]

    @torch.no_grad()
    def set_legacy_encodings(self, encodings: List[Dict]):
        """
        Set encodings represented in the same format as the output of get_legacy_encodings as below:

        [
            {'min': float, 'max': float, 'scale': float, 'offset': float,
                     'bitwidth': int, 'dtype': str, 'is_symmetric': str},
            {'min': float, 'max': float, 'scale': float, 'offset': float,
                     'bitwidth': int, 'dtype': str, 'is_symmetric': str},
            ...
        ]
        """
        def str_to_bool(s: str):
            s = s.lower()
            if s == "false":
                return False
            if s == "true":
                return True
            raise ValueError

        self.bitwidth = encodings[0]['bitwidth']
        self.symmetric = str_to_bool(encodings[0]['is_symmetric'])
        min_ = torch.tensor([e['min'] for e in encodings]).view(self.shape)
        max_ = torch.tensor([e['max'] for e in encodings]).view(self.shape)
        self.set_range(min_, max_)

    def extra_repr(self) -> str:
        return f'shape={self.shape}, bitwidth={self.bitwidth}, symmetric={self.symmetric}'

    @property
    def symmetric(self) -> bool:
        """
        Indicates whether this quantizer uses symmetric quantization
        """
        return self._symmetric

    @symmetric.setter
    def symmetric(self, symmetric: bool):
        """
        Set the quantizer symmetry

        :param symmetric: If True, use symmetric encodings. Else, use asymmetric encodings
        """
        self._symmetric = symmetric

    @property
    def signed(self)-> bool:
        """
        Indicates whether this quantizer uses signed quantization
        """
        return self._signed

    @signed.setter
    def signed(self, signed: bool):
        """
        Set the quantizer to use signed or unsigned quantization

        :param signed: If True, use signed encodings, else use unsigned encodings
        """
        self._signed = signed


class MinMaxQuantizer(AffineQuantizerBase): # pylint: disable=abstract-method
    """
    Affine quantizer with min-max as trainable parameters
    """

    min: torch.nn.Parameter
    max: torch.nn.Parameter

    def __init__(self, shape, bitwidth: int, symmetric: bool, encoding_analyzer: EncodingAnalyzer = None,
                 block_size: Optional[List[int]] = None):
        super().__init__(shape, bitwidth, symmetric, encoding_analyzer, block_size)

        self.register_quantization_parameter('min', nn.Parameter(-torch.ones(self.shape)))
        self.register_quantization_parameter('max', nn.Parameter(torch.ones(self.shape)))

    @contextlib.contextmanager
    def compute_encodings(self):
        """
        Observe inputs and update quantization parameters based on the input statistics.
        During ``compute_encodings`` is enabled, the quantizer forward pass performs
        dynamic quantization using the batch statistics.
        """
        if not self.encoding_analyzer:
            yield
            return

        original_forward = self.forward

        @functools.wraps(original_forward)
        def forward_wrapper(input):
            expanded_input = torch_builtins.reshape_tensor_for_blocks(input, self.shape, self.block_size)
            batch_statistics = self.encoding_analyzer.update_stats(expanded_input)
            num_quant_bins = math.pow(2, self.bitwidth) - 1
            dynamic_min, dynamic_max =\
                    self.encoding_analyzer.compute_encodings_from_stats(batch_statistics,
                                                                        num_quant_bins,
                                                                        self.symmetric)
            if self.block_size is not None:
                dynamic_min = dynamic_min.view(self.min.shape)
                dynamic_max = dynamic_max.view(self.max.shape)
            dynamic_min = dynamic_min.to(dtype=self.min.dtype,
                                         device=self.min.device).expand_as(self.min)
            dynamic_max = dynamic_max.to(dtype=self.max.dtype,
                                         device=self.max.device).expand_as(self.max)

            with patch_attr(self, 'min', dynamic_min),\
                    patch_attr(self, 'max', dynamic_max):
                return original_forward(input)

        try:
            with patch_attr(self, 'forward', forward_wrapper):
                yield
        except: # pylint: disable=try-except-raise
            raise
        else:
            try:
                num_quant_bins = math.pow(2, self.bitwidth) - 1
                enc_min, enc_max = self.encoding_analyzer.compute_encodings(num_quant_bins, self.symmetric)
                if self.block_size is not None:
                    enc_min = enc_min.view(self.min.shape)
                    enc_max = enc_max.view(self.max.shape)

            except StatisticsNotFoundError:
                return

            if enc_min is None or enc_max is None:
                return

            self.set_range(enc_min, enc_max)

        finally:
            self.encoding_analyzer.reset_stats()

    def get_min(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization min to be used for forward pass.

        NOTE: self.min may not be equal to self.get_min().
              self.get_min() returns slightly recalibrated version of self.min.

        :param dtype: dtype of the computed min. Use of self.min.dtype by default.
        :return: Quantization min
        """
        if not self.is_initialized():
            return None
        num_negative_steps = 2 ** (self.bitwidth - 1) if self._signed else 0
        return self.get_scale(dtype) * (self.get_offset(dtype) - num_negative_steps)

    def get_max(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization max to be used for forward pass.

        NOTE: self.max may not be equal to self.get_max()
              self.get_max() returns slightly recalibrated version of self.max.

        :param dtype: dtype of the computed max. Use of self.min.dtype by default.
        :return: Quantization max
        """
        if not self.is_initialized():
            return None
        num_positive_steps = 2 ** (self.bitwidth - 1) - 1 if self._signed else 2 ** self.bitwidth - 1
        return self.get_scale(dtype) * (self.get_offset(dtype) + num_positive_steps)

    def get_scale(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization scale to be used for forward pass.

        :param dtype: dtype of the computed scale. Use of self.min.dtype by default.
        :return: Quantization scale
        """
        if not self.is_initialized():
            return None

        dtype = dtype or self.min.dtype
        num_bins = 2 ** self.bitwidth - 1

        scale = (self.max.to(dtype) - self.min.to(dtype)) / num_bins
        return scale.to(dtype)

    def get_offset(self, dtype=None) -> Optional[torch.Tensor]:
        """
        Compute quantization offset to be used for forward pass.

        :param dtype: dtype of the computed offset. Use of self.min.dtype by default.
        :return: Quantization offset
        """
        if not self.is_initialized():
            return None

        dtype = dtype or self.min.dtype

        if self.symmetric:
            offset = torch.zeros_like(self.min, requires_grad=False, dtype=dtype)
        else:
            offset = ste_round(self.min.to(dtype) / self.get_scale(dtype))

            if self._signed:
                offset += 2 ** (self.bitwidth - 1)

        return offset.to(dtype)

    def set_range(self, min: torch.Tensor, max: torch.Tensor):
        """
        Set quantization parameters to the given min-max range
        """
        with torch.no_grad():
            self.min.copy_(min)
            self.max.copy_(max)


class Quantize(MinMaxQuantizer):
    r"""
    Applies quantization to the input.

    Precisely,

    .. math::
        out = clamp\left(\left\lceil\frac{input}{scale}\right\rfloor - offset, qmin, qmax\right)

    where :math:`scale` and :math:`offset` are derived from learnable parameters
    :math:`\theta_{min}` and :math:`\theta_{max}`.

    :param shape: Shape of the quantization parameters
    :type shape: tuple
    :param bitwidth: Quantization bitwidth
    :type bitwidth: int
    :param symmetric: If True, performs symmetric quantization;
                      otherwise, performs asymmetric quantization
    :type symmetric: bool
    :param encoding_analyzer: Encoding analyzer for calibrating quantization encodings
                              (default: absolute min-max encoding analyzer)
    :type encoding_analyzer: EncodingAnalyzer, optional


    :ivar Tensor min: :math:`\theta_{min}` from which scale and offset will be derived.
    :ivar Tensor max: :math:`\theta_{max}` from which scale and offset will be derived.

    .. note::
        :class:`Quantize` cannot run :meth:`forward` until :attr:`min` and :attr:`max` are properly initialized,
        which can be done based on input statistics using :meth:`compute_encodings` or
        by manually assigning a new value to :attr:`min` and :attr:`max`.
        See the examples below.

    Examples:

        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.randn(5, 10)
        >>> q = Q.affine.Quantize(shape=(5, 1), bitwidth=8, symmetric=False)
        >>> q.is_initialized()
        False
        >>> with q.compute_encodings():
        ...     _ = q(input)
        ...
        >>> q.is_initialized()
        True
        >>> q(input)
        QuantizedTensor([[247.,   0.,  98.,  62.,  25.,  42., 209.,  71., 255., 129.],
                         [209., 152., 211., 163.,   0.,  90., 255., 221.,  87.,  67.],
                         [119., 245., 178., 255., 100., 182., 188., 150., 162.,   0.],
                         [204., 102.,   0., 255., 224., 249., 190., 176., 207., 137.],
                         [  0., 189.,  13., 255., 109.,  23.,  93.,  59.,  82., 195.]],
                        grad_fn=<AliasBackward0>)


        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.randn(5, 10)
        >>> q = Q.affine.Quantize(shape=(5, 1), bitwidth=8, symmetric=False)
        >>> q.is_initialized()
        False
        >>> q.min = torch.nn.Parameter(-torch.ones_like(q.min))
        >>> q.max = torch.nn.Parameter(torch.ones_like(q.max))
        >>> q.is_initialized()
        True
        >>> q(input)
        QuantizedTensor([[255.,   0.,  90.,  21.,   0.,   0., 255.,  38., 255., 148.],
                         [208., 108., 212., 127.,   0.,   0., 255., 230.,   0.,   0.],
                         [100., 255., 193., 255.,  70., 199., 208., 150., 168.,   0.],
                         [220.,  70.,   0., 255., 249., 255., 200., 180., 225., 121.],
                         [104., 248., 114., 255., 187., 121., 175., 149., 167., 253.]],
                        grad_fn=<AliasBackward0>)
    """
    def forward(self, input: torch.Tensor) -> QuantizedTensor:
        """
        :param input: Input to quantize
        :return: Quantized output and scale/offset associated with it
        """
        if not self.is_initialized():
            raise RuntimeError(
                'Failed to run Quantize since quantization parameters are not initialized.'
                ' Please initialize the quantization parameters using `compute_encodings()`.'
            )

        encoding = self.get_encoding()
        output = quantize(input,
                          encoding.scale.to(input.dtype),
                          encoding.offset.to(input.dtype),
                          encoding.bitwidth,
                          encoding.signed,
                          block_size=self.block_size)
        output = output.as_subclass(QuantizedTensor)
        output.encoding = encoding
        return output


class QuantizeDequantize(MinMaxQuantizer):
    r"""
    Applies fake-quantization by quantizing and dequantizing the input.

    Precisely,

    .. math::
        out = (x_{int} + offset) * scale

    where

    .. math::
        x_{int} = clamp\left(\left\lceil\frac{input}{scale}\right\rfloor - offset, qmin, qmax\right)

    and :math:`scale` and :math:`offset` are derived from learnable parameters
    :math:`\theta_{min}` and :math:`\theta_{max}`.

    :param shape: Shape of the quantization parameters
    :type shape: tuple
    :param bitwidth: Quantization bitwidth
    :type bitwidth: int
    :param symmetric: If True, performs symmetric quantization;
                      otherwise, performs asymmetric quantization
    :type symmetric: bool
    :param encoding_analyzer: Encoding analyzer for calibrating quantization encodings
                              (default: absolute min-max encoding analyzer)
    :type encoding_analyzer: EncodingAnalyzer, optional


    :ivar Tensor min: :math:`\theta_{min}` from which scale and offset will be derived.
    :ivar Tensor max: :math:`\theta_{max}` from which scale and offset will be derived.

    .. note::
        :class:`QuantizeDequantize` cannot run :meth:`forward` until :attr:`min` and :attr:`max` are properly initialized,
        which can be done based on input statistics using :meth:`compute_encodings` or
        by manually assigning a new value to :attr:`min` and :attr:`max`.
        See the examples below.

    Examples:

        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.randn(5, 10)
        >>> qdq = Q.affine.QuantizeDequantize(shape=(5, 1), bitwidth=8, symmetric=False)
        >>> qdq.is_initialized()
        False
        >>> with qdq.compute_encodings():
        ...     _ = qdq(input)
        ...
        >>> qdq.is_initialized()
        True
        >>> qdq(input)
        DequantizedTensor([[ 1.9185, -1.7549, -0.2974, -0.8328, -1.3831, -1.1303,
                             1.3534, -0.6990,  2.0375,  0.1636],
                           [ 0.6366, -0.1522,  0.6643,  0.0000, -2.2559, -1.0103,
                             1.2733,  0.8027, -1.0518, -1.3286],
                           [-0.2097,  1.3444,  0.5180,  1.4677, -0.4440,  0.5674,
                             0.6414,  0.1727,  0.3207, -1.6774],
                           [ 0.7324, -0.4534, -1.6393,  1.3254,  0.9650,  1.2556,
                             0.5697,  0.4069,  0.7673, -0.0465],
                           [-0.1790,  0.9488, -0.1014,  1.3427,  0.4714, -0.0418,
                             0.3759,  0.1731,  0.3103,  0.9846]],
                          grad_fn=<AliasBackward0>)


        >>> import aimet_torch.v2.quantization as Q
        >>> input = torch.randn(5, 10)
        >>> qdq = Q.affine.QuantizeDequantize(shape=(5, 1), bitwidth=8, symmetric=False)
        >>> qdq.is_initialized()
        False
        >>> qdq.min = torch.nn.Parameter(-torch.ones_like(qdq.min))
        >>> qdq.max = torch.nn.Parameter(torch.ones_like(qdq.max))
        >>> qdq.is_initialized()
        True
        >>> qdq(input)
        DequantizedTensor([[ 1.0039, -0.9961, -0.2902, -0.8314, -0.9961, -0.9961,
                             1.0039, -0.6980,  1.0039,  0.1647],
                           [ 0.6353, -0.1490,  0.6667,  0.0000, -0.9961, -0.9961,
                             1.0039,  0.8078, -0.9961, -0.9961],
                           [-0.2118,  1.0039,  0.5176,  1.0039, -0.4471,  0.5647,
                             0.6353,  0.1804,  0.3216, -0.9961],
                           [ 0.7294, -0.4471, -0.9961,  1.0039,  0.9569,  1.0039,
                             0.5725,  0.4157,  0.7686, -0.0471],
                           [-0.1804,  0.9490, -0.1020,  1.0039,  0.4706, -0.0471,
                             0.3765,  0.1725,  0.3137,  0.9882]],
                          grad_fn=<AliasBackward0>)
    """
    def forward(self, input: torch.Tensor) -> DequantizedTensor:
        """
        :param input: Input to quantize and dequantize
        :return: Quantize-dequantized output
        """
        if not self.is_initialized():
            raise RuntimeError(
                'Failed to run QuantizeDequantize since quantization parameters are not initialized.'
                ' Please initialize the quantization parameters using `compute_encodings()`.'
            )

        encoding = self.get_encoding()
        output = quantize_dequantize(input,
                                     encoding.scale.to(input.dtype),
                                     encoding.offset.to(input.dtype),
                                     encoding.bitwidth,
                                     encoding.signed,
                                     block_size=self.block_size)
        output = output.as_subclass(DequantizedTensor)
        output.encoding = encoding
        return output


class Dequantize(torch.nn.Module):
    """
    Applies dequantization to the input
    """
    def forward(self, input: QuantizedTensor) -> DequantizedTensor:
        # pylint: disable=no-self-use
        """
        :param input: Input to dequantize
        :return: Dequantized output
        """
        return input.dequantize()

class LpbqQuantizeDequantize(QuantizeDequantize):
    """ Class for performing Low-Power Blockwise Quantization (LPBQ) """
    def __init__(self, shape, bitwidth: int, symmetric: bool, decompressed_bw: int,
                 encoding_analyzer: EncodingAnalyzer = None, block_size: Optional[List[int]] = None,
                 block_grouping: Optional[List[int]] = None):
        """
        LPBQ Quantize Dequantize constructor.

        :param shape: Shape of the quantization parameters
        :type shape: tuple
        :param bitwidth: Quantization bitwidth
        :type bitwidth: int
        :param symmetric: If True, performs symmetric quantization;
                          otherwise, performs asymmetric quantization
        :type symmetric: bool
        :param decompressed_bw: Bitwidth used for decompression in LPBQ algorithm.
        :type decompressed_bw: int
        :param encoding_analyzer: Encoding analyzer for calibrating quantization encodings
                                  (default: absolute min-max encoding analyzer)
        :type encoding_analyzer: EncodingAnalyzer, optional
        :param block_size: Block size per dimension.
        :type block_size: List[int]
        :param block_grouping: Block grouping per dimension. If provided, every set of block_group scales will be
                               grouped together, and the maximum scale for all blocks in the group will be used to find
                               the scale in the decompressed_grid to be shared by all blocks in the group.
                               If no block_grouping is provided, default behavior uses a block group of 1 for all dims,
                               equivalent to Blockwise Quantization.
                               A value of -1 for a block group for a dimension is equivalent to grouping all blocks in
                               the dimension in one group. This is also equivalent to a block group value equal to the
                               number of blocks for that dimension.
        :type block_grouping: List[int]
        """
        super().__init__(shape, bitwidth, symmetric, encoding_analyzer, block_size)
        self.decompressed_bw = decompressed_bw
        self.block_grouping = block_grouping
        if self.block_grouping is None:
            # Default to BQ behavior with 1 for all block grouping dims if not provided
            self.block_grouping = [1] * len(self.shape)

        if block_grouping is not None:
            if len(block_grouping) != len(shape):
                raise RuntimeError(f'Length of block grouping {block_grouping} must equal length of shape {shape}.')
            for idx, block_group in enumerate(block_grouping):
                if block_group != -1 and shape[idx] % block_group != 0:
                    raise RuntimeError(f'Block size values must divide evenly with corresponding block grouping values '
                                       f' for block size {block_size} and block grouping {block_grouping}.')

        if self.decompressed_bw < self.bitwidth:
            raise RuntimeError(f'Decompressed bitwidth {decompressed_bw} cannot be smaller than self.bitwidth '
                               f'{bitwidth}')

        if not symmetric:
            raise RuntimeError(f'LPBQ only supports symmetric quantization.')

    def get_scale(self, dtype=None) -> torch.Tensor:
        """
        Compute quantization scale to be used for forward pass.
        Overrides QuantizeDequantize self.get_scale() to apply the LPBQ algorithm for calculating modified scales.

        :param dtype: dtype of the computed scale. Use of self.min.dtype by default.
        :return: LPBQ scale
        """
        orig_scale = super().get_scale(dtype)
        orig_scale_shape = orig_scale.shape
        reshaped_scale = orig_scale.view(self.get_expanded_scale_shape())
        max_scale = torch.amax(reshaped_scale, list(range(1, len(orig_scale_shape) * 2, 2)), keepdim=True)
        per_channel_scale = max_scale / 2 ** (self.decompressed_bw - self.bitwidth)
        updated_scale = torch.maximum(ste_round(reshaped_scale / per_channel_scale), torch.tensor(1.0))
        updated_scale = (updated_scale * per_channel_scale)
        return updated_scale.view(orig_scale_shape)

    def get_expanded_scale_shape(self) -> List[int]:
        """
        Get expanded scale shape which breaks each scale dimension into a pair of dimensions with sizes
        (original_shape / block_grouping, block_grouping).

        :return: Expanded scale shape
        """
        expanded_shape = []
        for idx, block_group in enumerate(self.block_grouping):
            # Block group of -1 is equivalent to grouping all blocks together
            if block_group == -1:
                expanded_shape.append(1)
                expanded_shape.append(self.shape[idx])
            else:
                expanded_shape.append(self.shape[idx] // block_group)
                expanded_shape.append(block_group)
        return expanded_shape
