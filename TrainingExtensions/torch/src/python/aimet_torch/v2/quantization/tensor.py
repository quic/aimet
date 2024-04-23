# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Quantized tensor class implementation """

import abc
import copy

import torch
from torch.utils._pytree import tree_map, tree_flatten

from aimet_torch.v2.quantization.base import EncodingBase


__all__ = ['QuantizedTensorBase', 'QuantizedTensor', 'DequantizedTensor', 'EncodingError']


class QuantizedTensorBase(torch.Tensor):
    """
    Abstract base class to define quantized tensor behavior.
    Represents a quantized or dequantized tensor as a subclass of :class:`torch.Tensor` which also holds the quantization encodings.
    This object can be safely quantized or dequantized through the :meth:`quantize` and :meth:`dequantize` methods without
    changing the represented data values.

    Example:

        >>> from aimet_torch.v2 import quantization as Q
        >>> quantizer = Q.affine.Quantize(shape=(2, 1), bitwidth=8, symmetric=True)
        >>> x = torch.tensor([[-1.20, 4.1, -0.21, 2.3],
        ...                   [0.2, 5.6, -1.0, -.1]])
        >>> with quantizer.compute_encodings():
        ...     x_q = quantizer(x)
        >>> torch.equal(x_q.encoding.scale, quantizer.get_scale())
        True
        >>> x_q
        QuantizedTensor([[-37., 127.,  -7.,  71.],
                         [  5., 127., -23.,  -2.]])
        >>> x_q.quantized_repr()
        tensor([[-37, 127,  -7,  71],
                [  5, 127, -23,  -2]], dtype=torch.int8)
        >>> x_q.dequantize()
        DequantizedTensor([[-1.1945,  4.1000, -0.2260,  2.2921],
                           [ 0.2205,  5.6000, -1.0142, -0.0882]])
    """

    encoding: EncodingBase

    _cast_ops = [
        torch.Tensor.half,
        torch.Tensor.float,
        torch.Tensor.double,
        torch.Tensor.char,
        torch.Tensor.short,
        torch.Tensor.int,
        torch.Tensor.long,
        torch.Tensor.cuda,
        torch.Tensor.cpu,
        torch.Tensor.to,
    ]

    # Operations that an encoding can always pass through
    _passthrough_ops = {
        torch.Tensor.clone,
        torch.Tensor.contiguous,
        torch.Tensor.detach,
        torch.detach,
        torch.detach_copy,
    }

    # Operations that a per-tensor encoding can pass through
    _pertensor_passthrough_ops = {
        torch.Tensor.broadcast_to,
        torch.Tensor.expand,
        torch.Tensor.expand_as,
        torch.Tensor.flatten,
        torch.Tensor.masked_select,
        torch.Tensor.permute,
        torch.Tensor.repeat,
        torch.Tensor.reshape,
        torch.Tensor.reshape_as,
        torch.Tensor.resize,
        torch.Tensor.resize_as,
        torch.Tensor.select,
        torch.Tensor.squeeze,
        torch.Tensor.swapaxes,
        torch.Tensor.swapdims,
        torch.Tensor.t,
        torch.Tensor.transpose,
        torch.Tensor.unflatten,
        torch.Tensor.unsqueeze,
        torch.Tensor.view,
        torch.Tensor.view_as,
        torch.as_strided,
        torch.as_strided_copy,
        torch.expand_copy,
        torch.flatten,
        torch.permute,
        torch.permute_copy,
        torch.reshape,
        torch.squeeze,
        torch.squeeze_copy,
        torch.swapdims,
        torch.t,
        torch.t_copy,
        torch.unflatten,
        torch.unsqueeze,
        torch.unsqueeze_copy,
        torch.view_copy
    }

    @abc.abstractmethod
    def quantize(self) -> "QuantizedTensor":
        """
        Quantizes ``self`` with the associated encoding

        .. note::
            This method must be an IDEMPOTENT function.
            The result of calling this method multiple times should be equal to calling it only once.
            In other words, calling this method multiple times should not result in duplicate quantization.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def dequantize(self) -> "DequantizedTensor":
        """
        Dequantizes ``self`` with the associated encoding

        .. note::
            This method must be an IDEMPOTENT function.
            The result of calling this method multiple times should be equal to calling it only once.
            In other words, calling this method multiple times should not result in duplicate dequantization.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def quantized_repr(self) -> torch.Tensor:
        """
        Return the quantized representation of ``self`` as a :class:`torch.Tensor` with data type :attr:`self.encoding.dtype`

        .. note::
            The result of this function may not be able to carry a gradient depending on the quantized data type.
            Thus, it may be necessary to call this only within an autograd function to allow for backpropagation.

        Example:

            >>> from aimet_torch.v2 import quantization as Q
            >>> quantizer = Q.affine.Quantize(shape=(2, 1), bitwidth=8, symmetric=True)
            >>> x = torch.randn((2, 4), requires_grad=True)
            >>> with quantizer.compute_encodings():
            ...     x_q = quantizer(x)
            >>> x_q
            QuantizedTensor([[  11.,  -57., -128.,   38.],
                             [  28.,   -0., -128.,  -40.]], grad_fn=<AliasBackward0>)
            >>> x_q.quantized_repr()
            tensor([[  11,  -57, -128,   38],
                    [  28,    0, -128,  -40]], dtype=torch.int8)
        """
        raise NotImplementedError

    @classmethod
    def __new__(cls, *args, **kwargs):
        encoding = kwargs.pop('encoding', None)
        ret = super().__new__(*args, **kwargs)
        if not ret.is_floating_point():
            raise RuntimeError(f"Non-floating point dtype `{ret.dtype}` is not allowed for quantized tensors.")
        ret.encoding = encoding
        return ret

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        ret = super().__torch_function__(func, types, args, kwargs)

        flattened_args, _ = tree_flatten((args, kwargs))
        if any(ret is arg for arg in flattened_args):
            # Return value is the same object as one of the arguments.
            # This implies that func is likely (but not necessarily) an in-place operator.
            return ret

        if func in cls._cast_ops:
            if not ret.dtype.is_floating_point:
                raise RuntimeError(
                    f"Type casting to non-floating point dtype `{ret.dtype}` is not allowed for quantized tensors. "
                    "To cast quantized tensors to integer, use `qtensor.quantzed_repr()`."
                )

            # Outputs of cast ops can inherit the same encoding as its parents
            self, *_ = args
            ret.encoding = copy.copy(self.encoding) # shallow copy

        if func in cls._passthrough_ops:
            self, *_ = args
            ret.encoding = copy.copy(self.encoding)

        if func in cls._pertensor_passthrough_ops:
            self, *_ = args
            if self.encoding.granularity == "pertensor":
                ret.encoding = copy.copy(self.encoding)

        def set_encoding(qtensor):
            if not hasattr(qtensor, 'encoding'):
                qtensor.encoding = None

            if qtensor.encoding is None:
                # If encoding does not exist, return a plain torch.Tensor
                return qtensor.as_subclass(torch.Tensor)

            # Change device of encoding
            # NOTE: We don't change the dtypes of encoding because scale/offset
            #       are sensitive to dtype
            qtensor.encoding = qtensor.encoding.to(device=qtensor.device)

            return qtensor

        return tree_map(lambda t: set_encoding(t) if isinstance(t, cls) else t, ret)


class QuantizedTensor(QuantizedTensorBase):
    """
    Represents a quantized tensor object. The object holds quantized values stored in a floating-point tensor along with
    an :class:`EncodingBase` object which holds the information necessary to map the quantized values back to the
    real/represented values.
    """

    def quantize(self) -> "QuantizedTensor":
        """
        Returns ``self``
        """
        if self.encoding is None:
            raise EncodingError("Encoding does not exist")
        return self

    def dequantize(self) -> "DequantizedTensor":
        """
        Dequantizes ``self`` using :attr:`self.encoding` to produce a :class:`DequantizedTensor` with the same encoding
        information.

        Example:

            >>> from aimet_torch.v2.quantization as Q
            >>> x = torch.tensor([[2.57, -2.312],
            ...                   [0.153, 0.205]])
            >>> quantizer = Q.affine.Quantize(shape=(1, ), bitwidth=8, symmetric=True)
            >>> quantizer.set_range(-128 * 0.1, 127 * 0.1)
            >>> x_q = quantizer(x)
            >>> x_q
            QuantizedTensor([[ 26., -23.],
                             [  2.,   2.]], grad_fn=<AliasBackward0>)
            >>> x_dq = x_q.dequantize()
            >>> x_dq
            DequantizedTensor([[ 2.6000, -2.3000],
                               [ 0.2000,  0.2000]], grad_fn=<AliasBackward0>)
            >>> torch.equal(x_dq.encoding.scale, x_q.encoding.scale)
            True
        """
        if self.encoding is None:
            raise EncodingError("Encoding does not exist")

        qtensor = self.encoding.dequantize(self.as_subclass(torch.Tensor))
        qtensor = qtensor.as_subclass(DequantizedTensor)
        qtensor.encoding = copy.copy(self.encoding)
        return qtensor

    def quantized_repr(self) -> torch.Tensor:
        # FIXME(kyunggeu): This only works for affine encodings.
        #                  Needs to be generalized for any kind of encodings
        return self.quantize().as_subclass(torch.Tensor).to(self.encoding.dtype)


class DequantizedTensor(QuantizedTensorBase):
    """
    Represents a tensor which has been quantized and subsequently dequantized. This object contains real floating point
    data as well as an :class:`EncodingBase` object which holds information about the quantization parameters with which
    the data was quantized. With this, a :class:`DequantizedTensor` can be converted back to its quantized representation
    without further loss in information.
    """

    def quantize(self) -> QuantizedTensor:
        """
        Quantizes ``self`` using :attr:`self.encoding` to produce a :class:`QuantizedTensor` with the same encoding
        information.

        Example:

            >>> import aimet_torch.v2.quantization as Q
            >>> x = torch.tensor([[0.39, 51.0], [3.521, 9.41]])
            >>> quant_dequant = Q.affine.QuantizeDequantize((1, ), 8, symmetric=False)
            >>> quant_dequant.set_range(-10, 41)
            >>> x_qdq = quant_dequant(x)
            >>> x_qdq
            DequantizedTensor([[ 0.4000, 41.0000],
                               [ 3.6000,  9.4000]], grad_fn=<AliasBackward0>)
            >>> x_qdq.quantize()
            QuantizedTensor([[ 52., 255.],
                             [ 68.,  97.]], grad_fn=<AliasBackward0>)
        """
        if self.encoding is None:
            raise EncodingError("Encoding does not exist")

        qtensor = self.encoding.quantize(self.as_subclass(torch.Tensor))
        qtensor = qtensor.as_subclass(QuantizedTensor)
        qtensor.encoding = copy.copy(self.encoding)
        return qtensor

    def dequantize(self) -> "DequantizedTensor":
        """
        Returns ``self``
        """
        if self.encoding is None:
            raise EncodingError("Encoding does not exist")
        return self

    def quantized_repr(self) -> torch.Tensor:
        """
        Return the quantized representation of ``self`` as a :class:`torch.Tensor` with data type :attr:`self.encoding.dtype`.

        .. note::
            The result of this function may not be able to carry a gradient depending on the quantized data type.
            Thus, it may be necessary to call this only within an autograd function to allow for backpropagation.

        Example:

            >>> import aimet_torch.v2.quantization as Q
            >>> x = torch.tensor([[0.39, 51.0], [3.521, 9.41]])
            >>> quant_dequant = Q.affine.QuantizeDequantize((1, ), 8, symmetric=False)
            >>> quant_dequant.set_range(-10, 41)
            >>> x_qdq = quant_dequant(x)
            >>> x_qdq
            DequantizedTensor([[ 0.4000, 41.0000],
                               [ 3.6000,  9.4000]], grad_fn=<AliasBackward0>)
            >>> x_qdq.quantized_repr()
            tensor([[ 52, 255],
                    [ 68,  97]], dtype=torch.uint8)
        """
        # FIXME(kyunggeu): This only works for affine encodings.
        #                  Needs to be generalized for any kind of encodings
        return self.quantize().as_subclass(torch.Tensor).to(self.encoding.dtype)


class EncodingError(RuntimeError):
    """Error that indicates an encoding is missing or invalid"""
