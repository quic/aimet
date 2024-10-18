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
# pylint: disable=too-many-lines, wrong-import-order, redefined-builtin
""" Quantized modules"""

from packaging import version
import contextlib
import itertools
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
from typing import Type, Any, Optional, Callable, Dict
from weakref import WeakKeyDictionary
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.overrides import BaseTorchFunctionMode, get_overridable_functions
from torch._VF import ( # pylint: disable=no-name-in-module
    gru as _gru,
    gru_cell as _gru_cell,
    lstm as _lstm,
    lstm_cell as _lstm_cell,
    rnn_relu as _rnn_relu,
    rnn_tanh as _rnn_tanh,
    rnn_relu_cell as _rnn_relu_cell,
    rnn_tanh_cell as _rnn_tanh_cell,
)

from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.tensor import QuantizedTensorBase
from aimet_torch.v2.utils import patch_attr, _ContextManager, allow_recompute
from .base import BaseQuantizationMixin # pylint: disable=import-error


def _quantize_if_applicable(data: Any, quantizer: Optional[QuantizerBase]):
    """
    Quantize data if it is a quantizable type and quantize is not None
    """
    if quantizer and isinstance(data, Tensor) and data.is_floating_point():
        if isinstance(data, QuantizedTensorBase):
            data = data.dequantize()
        return quantizer(data)

    if isinstance(data, QuantizedTensorBase):
        return data.quantize()

    return data


def _dequantize_if_applicable(data: torch.Tensor):
    return data.dequantize() if isinstance(data, QuantizedTensorBase) else data


def _quantize_dequantize_if_applicable(data, quantizer):
    if quantizer and isinstance(data, Tensor) and data.is_floating_point():
        if isinstance(data, QuantizedTensorBase):
            data = data.dequantize()
        data = quantizer(data)

    if isinstance(data, QuantizedTensorBase):
        return data.dequantize()

    return data


_QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS = WeakKeyDictionary()


def _is_computing_encodings(qmodule):
    return _QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS.get(qmodule, 0) > 0


def _enter_computing_encodings(qmodule):
    if qmodule not in _QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS:
        _QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS[qmodule] = 0
    _QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS[qmodule] += 1


def _exit_compute_encodings(qmodule):
    assert _QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS[qmodule] > 0
    _QUANTIZED_MODULES_UNDER_COMPUTE_ENCODINGS[qmodule] -= 1


class QuantizationMixinMeta(ABCMeta):
    """Sets :meth:`forward` to :meth:`quantized_forward` if only :meth:`quantized_forward` is defined
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        if "quantized_forward" in namespace and "forward" not in namespace:
            warnings.warn("Support for defining `quantized_forward` in place of `forward` method will be deprecated, "
                          "please use `forward` instead.",
                          DeprecationWarning, stacklevel=2)
            namespace["forward"] = namespace["quantized_forward"]
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class QuantizationMixin(BaseQuantizationMixin, metaclass=QuantizationMixinMeta): # pylint: disable=abstract-method
    """Mixin that adds quantization functionality on top of regular pytorch modules.

    Specifically, a quantized module will quantize input, output, and parameter tensors with
    its held :class:`QuantizerBase` objects during the :meth:`forward` method and use the inherited :class:`torch.nn.Module`
    forward method to compute the layer operation. If all input, output, and parameter quantizers are ``None``, a
    quantized module will behave exactly the same as its parent :class:`torch.nn.Module`.

    Attributes:
        input_quantizers: :class:`torch.nn.ModuleList` containing :class:`QuantizerBase` objects to be applied
            to the layer's input tensors
        output_quantizers: :class:`torch.nn.ModuleList` containing :class:`QuantizerBase` objects to be applied
            to the layer's output tensors
        param_quantizers: :class:`torch.nn.ModuleDict` mapping parameter names to associated :class:`QuantizerBase`
            objects

    Examples:

        >>> qlinear = QuantizedLinear(in_features=10, out_features=10)
        >>> print(qlinear)
        QuantizedLinear(
          in_features=10, out_features=10, bias=True
          (param_quantizers): ModuleDict(
            (weight): None
            (bias): None
          )
          (input_quantizers): ModuleList(
            (0): None
          )
          (output_quantizers): ModuleList(
            (0): None
          )
        )

    """

    cls_to_qcls = OrderedDict()  # quantized class -> original class
    qcls_to_cls = OrderedDict()  # original class -> quantized class

    _default_kernel: Optional[Callable] = None
    _kernels = WeakKeyDictionary()  # instance -> instance_kernel

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Computes a quantized version of the parent module's forward method.

        The :meth:`forward` method should perform the following logic in order:

            1) Apply existing input quantizers to input tensors
            2) Apply existing param quantizers to the layer's parameters
            3) Call the inherited :class:`torch.nn.Module` forward method with quantized inputs and parameters
            4) Apply existing output quantizers to the outputs of the forward method

        If all input, output, and parameter quantizers are ``None``, this method will behave exactly the same as
        its parent module's forward pass.
        """
        return super().forward(*args, **kwargs)

    @classmethod
    def set_default_kernel(cls, kernel: Callable):
        """Set default kernel for the class.

        In general, this signature will follow the signature of the equivalent :mod:`torch.nn.functional` function,
        but should return a :class:`QuantizedTensor` object and take in the additional keyword argument ``output_encodings``.

        Once set, all instances of cls will call into kernel in the forward pass unless:

            1) The instance is within the :meth:`compute_encodings` context, or
            2) The kernel has been overridden by a :meth:`set_kernel` call

        Args:
            kernel: Callable object to be used as the default kernel by all the instances of this class.


        Example:

            >>> from aimet_torch.v2 import quantization as Q
            >>> def int_multiply(a, b, output_encodings=None):
            ...     encodings = [a.encoding, b.encoding, output_encodings]
            ...     if not all(enc.mapping == "affine" for enc in encodings):
            ...             raise NotImplementedError
            ...     q_output = (a.quantized_repr() + a.encoding.offset) * (b.quantized_repr() + b.encoding.offset)
            ...     dq_output = q_output *  (a.encoding.scale * b.encoding.scale)
            ...     return Q.QuantizedTensor(output_encodings.quantize(dq_output), encoding=output_encodings)
            ...
            >>> QuantizedMultiply.set_default_kernel(int_multiply)
            >>> qmult = QuantizedMultiply()
            >>> qmult.get_kernel()
            <function int_multiply at ...>

        """
        cls._default_kernel = kernel

    @classmethod
    def get_default_kernel(cls) -> Optional[Callable]:
        """Return the default kernel of the class

        Returns:
            Default kernel of the class. None if the default kernel is not set.

        """
        return cls._default_kernel

    def set_kernel(self, kernel: Callable):
        """Set kernel for this instance of quantized module.

        In general, this signature will follow the signature of the equivalent :mod:`torch.nn.functional` function,
        but should return a :class:`QuantizedTensor` object and take in the additional keyword argument ``output_encodings``.

        Once set, the layer will call into ``kernel`` in the forward pass unless within the :meth:`compute_encodings`
        context.

        Args:
            kernel: Callable object to be used as the underlying kernel.

        Example:

            >>> from aimet_torch.v2 import quantization as Q
            >>> def int_multiply(a, b, output_encodings=None):
            ...     encodings = [a.encoding, b.encoding, output_encodings]
            ...     if not all(enc.mapping == "affine" for enc in encodings):
            ...             raise NotImplementedError
            ...     q_output = (a.quantized_repr() + a.encoding.offset) * (b.quantized_repr() + b.encoding.offset)
            ...     dq_output = q_output *  (a.encoding.scale * b.encoding.scale)
            ...     return Q.QuantizedTensor(output_encodings.quantize(dq_output), encoding=output_encodings)
            ...
            >>> qmult = QuantizedMultiply()
            >>> qmult.set_kernel(int_multiply)

        """
        QuantizationMixin._kernels[self] = kernel

    def get_kernel(self) -> Optional[Callable]:
        """Return the kernel to be used by this instance of quantized module.

        If the current instance does not have any kernel set, it will retrieve the default kernel of the class.

        Returns:
            The kernel to be used by this instance.

        """
        if self in QuantizationMixin._kernels:
            return QuantizationMixin._kernels[self]
        return self.get_default_kernel()

    @contextlib.contextmanager
    def compute_encodings(self):  # pylint: disable=missing-function-docstring
        ctx = _ContextManager(action=lambda: _enter_computing_encodings(self),
                              cleanup=lambda: _exit_compute_encodings(self))
        with super().compute_encodings(), ctx:
            yield

    def _patch_dequantized_parameters(self):
        stack = contextlib.ExitStack()
        for param_name, _ in self.param_quantizers.items():
            qparam = getattr(self, param_name)
            ctx = patch_attr(self, param_name, _dequantize_if_applicable(qparam))
            stack.enter_context(ctx)

        return stack

    @classmethod
    def wrap(cls, module_cls: Type[nn.Module]) -> Type[nn.Module]:
        """
        Wrap a regular module class into a quantized module class
        """
        if not issubclass(module_cls, nn.Module):
            raise ValueError("Expected module_cls to be a subclass of torch.nn.Module. "
                             f"Got {module_cls}.")
        if module_cls in cls.cls_to_qcls:
            return cls.cls_to_qcls[module_cls]

        quantized_cls_name = f"Quantized{module_cls.__name__}"
        base_classes = (cls, module_cls)
        quantized_cls = type(quantized_cls_name, base_classes, {'__module__': __name__})
        return cls.implements(module_cls)(quantized_cls)

    @classmethod
    def from_module(cls, module: nn.Module):
        r"""Create an instance of quantized module from a regular module instance.

        The resulting quantized module contains the same attributes and parameters as the original module, but may
        be assigned input, output and parameter quantizers.

        :param module: Floating point module to quantize
        :return: Quantized version of the original module

        Example:

            >>> linear = torch.nn.Linear(10, 10)
            >>> quantized_linear = QuantizationMixin.from_module(linear)
            >>> print(quantized_linear.param_quantizers)
            QuantizedLinear(
              in_features=10, out_features=10, bias=True
              (param_quantizers): ModuleDict(
                (weight): None
                (bias): None
              )
              (input_quantizers): ModuleList(
                (0): None
              )
              (output_quantizers): ModuleList(
                (0): None
              )
            )
            >>> print(quantized_linear.weight is linear.weight)
            True
        """
        return super().from_module(module)


# pylint: disable=too-many-ancestors


_dispatch_table: Dict[Callable, Optional[Callable]]
_dispatch_table = {
    torch_fn: None
    for torch_fn in itertools.chain(*get_overridable_functions().values())
}

# NOTE: ``torch.overrides.get_overridable_functions()`` doesn't include
#       F.hardswish, F.hardsigmoid, or Tensor.unflatten, even though
#       they are implemented in a perfectly dispatchable manner.
_dispatch_table[F.hardswish] = None
_dispatch_table[F.hardsigmoid] = None
_dispatch_table[Tensor.unflatten] = None


class _Dispatcher(BaseTorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        impl = _dispatch_table.get(func, None)

        if impl is None:
            impl = func

        return super().__torch_function__(impl, types, args, kwargs)


_dispatcher = _Dispatcher()
_stack_level = 0

@contextlib.contextmanager
def _dispatch(torch_func: Callable, custom_impl: Callable):
    # pylint: disable=global-statement
    global _stack_level
    orig_level = _stack_level

    try:
        orig = _dispatch_table[torch_func]
    except KeyError as e:
        raise RuntimeError(f"PyTorch doesn't support overriding {torch_func}") from e

    try:
        _dispatch_table[torch_func] = custom_impl

        if _stack_level == 0:
            _dispatcher.__enter__()
        _stack_level += 1

        yield
    finally:
        _dispatch_table[torch_func] = orig
        _stack_level = orig_level

        if _stack_level == 0:
            _dispatcher.__exit__(None, None, None)


class _DispatchMeta(QuantizationMixinMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        """
        Sanity check for class definitions of dispatch-based quantized modules
        """
        if '_builtin_torch_fn' in namespace:
            torch_fn = namespace['_builtin_torch_fn']
            if torch_fn and torch_fn not in _dispatch_table:
                raise RuntimeError(f"PyTorch doesn't support overriding {torch_fn}")
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class _DispatchMixin(metaclass=_DispatchMeta):
    _builtin_torch_fn: Optional[Callable] = None

    def _get_builtin_torch_fn(self):
        return type(self)._builtin_torch_fn

    def forward(self, *args, **kwargs):  # pylint: disable=missing-function-docstring
        kernel = self.get_kernel()
        builtin_torch_fn = self._get_builtin_torch_fn()

        if not kernel or _is_computing_encodings(self):
            kernel = self._builtin_torch_fn_helper(builtin_torch_fn)
        else:
            kernel = self._custom_kernel_helper(kernel)

        with self._patch_quantized_parameters():
            with _dispatch(builtin_torch_fn, kernel):
                output = super().forward(*args, **kwargs)

        return _dequantize_if_applicable(output)

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        def wrapper(*args, **kwargs):
            qtzd_args = (
                _quantize_dequantize_if_applicable(x, qtzr)
                for x, qtzr in zip(args, self.input_quantizers)
            )
            others = (
                _dequantize_if_applicable(x)
                for x in args[len(self.input_quantizers):]
            )
            kwargs = {
                key: _dequantize_if_applicable(value)
                for key, value in kwargs.items()
            }

            output = fn(*qtzd_args, *others, **kwargs)

            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return wrapper

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        def wrapper(*args, **kwargs):
            qtzd_args = (
                _quantize_if_applicable(x, qtzr)
                for x, qtzr in zip(args, self.input_quantizers)
            )
            others = args[len(self.input_quantizers):]

            output_encodings = self.output_quantizers[0].get_encoding() if self.output_quantizers[0] else None
            kwargs.update(output_encodings=output_encodings)
            return fn(*qtzd_args, *others, **kwargs)

        return wrapper


def __nullary__(self):
    super(type(self), self).__quant_init__()
    self.input_quantizers = nn.ModuleList([])


def __unary__(self):
    super(type(self), self).__quant_init__()


def __binary__(self):
    super(type(self), self).__quant_init__()
    self.input_quantizers = nn.ModuleList([None, None])


def __ternary__(self):
    super(type(self), self).__quant_init__()
    self.input_quantizers = nn.ModuleList([None, None, None])


@QuantizationMixin.implements(nn.AdaptiveAvgPool1d)
class QuantizedAdaptiveAvgPool1d(_DispatchMixin, QuantizationMixin, nn.AdaptiveAvgPool1d):
    """ Quantized AdaptiveAvgPool1d """
    _builtin_torch_fn = F.adaptive_avg_pool1d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.AdaptiveAvgPool2d)
class QuantizedAdaptiveAvgPool2d(_DispatchMixin, QuantizationMixin, nn.AdaptiveAvgPool2d):
    """ Quantized AdaptiveAvgPool2d """
    _builtin_torch_fn = F.adaptive_avg_pool2d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.AdaptiveAvgPool3d)
class QuantizedAdaptiveAvgPool3d(_DispatchMixin, QuantizationMixin, nn.AdaptiveAvgPool3d):
    """ Quantized AdaptiveAvgPool3d """
    _builtin_torch_fn = F.adaptive_avg_pool3d
    __quant_init__ = __unary__


# @QuantizationMixin.implements(nn.AdaptiveLogSoftmaxWithLoss)
# class QuantizedAdaptiveLogSoftmaxWithLoss(_DispatchMixin, QuantizationMixin, nn.AdaptiveLogSoftmaxWithLoss):
#     """ Quantized AdaptiveLogSoftmaxWithLoss """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.AdaptiveMaxPool1d)
class QuantizedAdaptiveMaxPool1d(_DispatchMixin, QuantizationMixin, nn.AdaptiveMaxPool1d):
    """ Quantized AdaptiveMaxPool1d """
    _builtin_torch_fn = F.adaptive_max_pool1d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.AdaptiveMaxPool2d)
class QuantizedAdaptiveMaxPool2d(_DispatchMixin, QuantizationMixin, nn.AdaptiveMaxPool2d):
    """ Quantized AdaptiveMaxPool2d """
    _builtin_torch_fn = F.adaptive_max_pool2d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.AdaptiveMaxPool3d)
class QuantizedAdaptiveMaxPool3d(_DispatchMixin, QuantizationMixin, nn.AdaptiveMaxPool3d):
    """ Quantized AdaptiveMaxPool3d """
    _builtin_torch_fn = F.adaptive_max_pool3d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.AlphaDropout)
class QuantizedAlphaDropout(_DispatchMixin, QuantizationMixin, nn.AlphaDropout):
    """ Quantized AlphaDropout """
    _builtin_torch_fn = F.alpha_dropout
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.AvgPool1d)
class QuantizedAvgPool1d(_DispatchMixin, QuantizationMixin, nn.AvgPool1d):
    """ Quantized AvgPool1d """
    _builtin_torch_fn = F.avg_pool1d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.AvgPool2d)
class QuantizedAvgPool2d(_DispatchMixin, QuantizationMixin, nn.AvgPool2d):
    """ Quantized AvgPool2d """
    _builtin_torch_fn = F.avg_pool2d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.AvgPool3d)
class QuantizedAvgPool3d(_DispatchMixin, QuantizationMixin, nn.AvgPool3d):
    """ Quantized AvgPool3d """
    _builtin_torch_fn = F.avg_pool3d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.BCELoss)
class QuantizedBCELoss(_DispatchMixin, QuantizationMixin, nn.BCELoss):
    """ Quantized BCELoss """
    _builtin_torch_fn = F.binary_cross_entropy
    __quant_init__ = __binary__


@QuantizationMixin.implements(nn.BCEWithLogitsLoss)
class QuantizedBCEWithLogitsLoss(_DispatchMixin, QuantizationMixin, nn.BCEWithLogitsLoss):
    """ Quantized BCEWithLogitsLoss """
    _builtin_torch_fn = F.binary_cross_entropy_with_logits
    __quant_init__ = __binary__


@QuantizationMixin.implements(nn.BatchNorm1d)
class QuantizedBatchNorm1d(_DispatchMixin, QuantizationMixin, nn.BatchNorm1d):
    """ Quantized BatchNorm1d """
    _builtin_torch_fn = F.batch_norm
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.BatchNorm2d)
class QuantizedBatchNorm2d(_DispatchMixin, QuantizationMixin, nn.BatchNorm2d):
    """ Quantized BatchNorm2d """
    _builtin_torch_fn = F.batch_norm
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.BatchNorm3d)
class QuantizedBatchNorm3d(_DispatchMixin, QuantizationMixin, nn.BatchNorm3d):
    """ Quantized BatchNorm3d """
    _builtin_torch_fn = F.batch_norm
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Bilinear)
class QuantizedBilinear(_DispatchMixin, QuantizationMixin, nn.Bilinear):
    """ Quantized Bilinear """
    _builtin_torch_fn = F.bilinear
    __quant_init__ = __binary__


@QuantizationMixin.implements(nn.CELU)
class QuantizedCELU(_DispatchMixin, QuantizationMixin, nn.CELU):
    """ Quantized CELU """
    _builtin_torch_fn = F.celu
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.CTCLoss)
class QuantizedCTCLoss(_DispatchMixin, QuantizationMixin, nn.CTCLoss):
    """ Quantized CTCLoss """
    _builtin_torch_fn = F.ctc_loss
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ChannelShuffle)
class QuantizedChannelShuffle(_DispatchMixin, QuantizationMixin, nn.ChannelShuffle):
    """ Quantized ChannelShuffle """
    _builtin_torch_fn = F.channel_shuffle
    __quant_init__ = __unary__


if version.parse(torch.__version__) >= version.parse("2.1.0"):
    @QuantizationMixin.implements(nn.CircularPad1d)
    class QuantizedCircularPad1d(_DispatchMixin, QuantizationMixin, nn.CircularPad1d):
        """ Quantized CircularPad1d """
        _builtin_torch_fn = F.pad
        __quant_init__ = __unary__


    @QuantizationMixin.implements(nn.CircularPad2d)
    class QuantizedCircularPad2d(_DispatchMixin, QuantizationMixin, nn.CircularPad2d):
        """ Quantized CircularPad2d """
        _builtin_torch_fn = F.pad
        __quant_init__ = __unary__


    @QuantizationMixin.implements(nn.CircularPad3d)
    class QuantizedCircularPad3d(_DispatchMixin, QuantizationMixin, nn.CircularPad3d):
        """ Quantized CircularPad3d """
        _builtin_torch_fn = F.pad
        __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ConstantPad1d)
class QuantizedConstantPad1d(_DispatchMixin, QuantizationMixin, nn.ConstantPad1d):
    """ Quantized ConstantPad2d """
    _builtin_torch_fn = F.pad
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ConstantPad2d)
class QuantizedConstantPad2d(_DispatchMixin, QuantizationMixin, nn.ConstantPad2d):
    """ Quantized ConstantPad2d """
    _builtin_torch_fn = F.pad
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ConstantPad3d)
class QuantizedConstantPad3d(_DispatchMixin, QuantizationMixin, nn.ConstantPad3d):
    """ Quantized ConstantPad3d """
    _builtin_torch_fn = F.pad
    __quant_init__ = __unary__


# @QuantizationMixin.implements(nn.Container)
# class QuantizedContainer(_DispatchMixin, QuantizationMixin, nn.Container):
#     """ Quantized Container """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.Conv1d)
class QuantizedConv1d(_DispatchMixin, QuantizationMixin, nn.Conv1d):  # pylint: disable=too-many-ancestors
    """ Quantized Conv1d """
    _builtin_torch_fn = F.conv1d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Conv2d)
class QuantizedConv2d(_DispatchMixin, QuantizationMixin, nn.Conv2d):  # pylint: disable=too-many-ancestors
    """ Quantized Conv2d """
    _builtin_torch_fn = F.conv2d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Conv3d)
class QuantizedConv3d(_DispatchMixin, QuantizationMixin, nn.Conv3d):  # pylint: disable=too-many-ancestors
    """ Quantized Conv3d """
    _builtin_torch_fn = F.conv3d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ConvTranspose1d)
class QuantizedConvTranspose1d(_DispatchMixin, QuantizationMixin, nn.ConvTranspose1d): # pylint: disable=too-many-ancestors
    """ Quantized ConvTranspose1d """
    _builtin_torch_fn = F.conv_transpose1d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ConvTranspose2d)
class QuantizedConvTranspose2d(_DispatchMixin, QuantizationMixin, nn.ConvTranspose2d): # pylint: disable=too-many-ancestors
    """ Quantized ConvTranspose2d """
    _builtin_torch_fn = F.conv_transpose2d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ConvTranspose3d)
class QuantizedConvTranspose3d(_DispatchMixin, QuantizationMixin, nn.ConvTranspose3d): # pylint: disable=too-many-ancestors
    """ Quantized ConvTranspose3d """
    _builtin_torch_fn = F.conv_transpose3d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.CosineEmbeddingLoss)
class QuantizedCosineEmbeddingLoss(_DispatchMixin, QuantizationMixin, nn.CosineEmbeddingLoss):
    """ Quantized CosineEmbeddingLoss """
    _builtin_torch_fn = F.cosine_embedding_loss
    __quant_init__ = __binary__


@QuantizationMixin.implements(nn.CosineSimilarity)
class QuantizedCosineSimilarity(_DispatchMixin, QuantizationMixin, nn.CosineSimilarity):
    """ Quantized CosineSimilarity """
    _builtin_torch_fn = F.cosine_similarity
    __quant_init__ = __binary__


@QuantizationMixin.implements(nn.CrossEntropyLoss)
class QuantizedCrossEntropyLoss(_DispatchMixin, QuantizationMixin, nn.CrossEntropyLoss):
    """ Quantized CrossEntropyLoss """
    _builtin_torch_fn = F.cross_entropy
    __quant_init__ = __binary__


# @QuantizationMixin.implements(nn.CrossMapLRN2d)
# class QuantizedCrossMapLRN2d(_DispatchMixin, QuantizationMixin, nn.CrossMapLRN2d):
#     """ Quantized CrossMapLRN2d """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.Dropout)
class QuantizedDropout(_DispatchMixin, QuantizationMixin, nn.Dropout):
    """ Quantized Dropout """
    _builtin_torch_fn = F.dropout
    __quant_init__ = __unary__


if version.parse(torch.__version__) >= version.parse("1.12.0"):
    @QuantizationMixin.implements(nn.Dropout1d)
    class QuantizedDropout1d(_DispatchMixin, QuantizationMixin, nn.Dropout1d):
        """ Quantized Dropout1d """
        _builtin_torch_fn = F.dropout1d
        __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Dropout2d)
class QuantizedDropout2d(_DispatchMixin, QuantizationMixin, nn.Dropout2d):
    """ Quantized Dropout2d """
    _builtin_torch_fn = F.dropout2d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Dropout3d)
class QuantizedDropout3d(_DispatchMixin, QuantizationMixin, nn.Dropout3d):
    """ Quantized Dropout3d """
    _builtin_torch_fn = F.dropout3d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ELU)
class QuantizedELU(_DispatchMixin, QuantizationMixin, nn.ELU):
    """ Quantized ELU """
    _builtin_torch_fn = F.elu
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Embedding)
class QuantizedEmbedding(_DispatchMixin, QuantizationMixin, nn.Embedding):
    """ Quantized Embedding """
    _builtin_torch_fn = F.embedding
    __quant_init__ = __nullary__


@QuantizationMixin.implements(nn.EmbeddingBag)
class QuantizedEmbeddingBag(_DispatchMixin, QuantizationMixin, nn.EmbeddingBag):
    """ Quantized EmbeddingBag """
    _builtin_torch_fn = F.embedding_bag

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        def embedding_bag(input: Tensor, # pylint: disable=redefined-builtin, too-many-arguments
                          weight: Tensor,
                          offsets: Optional[Tensor] = None,
                          max_norm: Optional[float] = None,
                          norm_type: float = 2,
                          scale_grad_by_freq: bool = False,
                          mode: str = "mean",
                          sparse: bool = False,
                          per_sample_weights: Optional[Tensor] = None,
                          include_last_offset: bool = False,
                          padding_idx: Optional[int] = None):

            if per_sample_weights is not None:
                qtzr = self.input_quantizers[0]
                per_sample_weights = _quantize_dequantize_if_applicable(per_sample_weights, qtzr)

            output = fn(input,
                        weight,
                        offsets=offsets,
                        max_norm=max_norm,
                        norm_type=norm_type,
                        scale_grad_by_freq=scale_grad_by_freq,
                        mode=mode,
                        sparse=sparse,
                        per_sample_weights=per_sample_weights,
                        include_last_offset=include_last_offset,
                        padding_idx=padding_idx)

            return _quantize_dequantize_if_applicable(output, self.output_quantizers[0])

        return embedding_bag

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        def embedding_bag(input: Tensor, # pylint: disable=redefined-builtin, too-many-arguments
                          weight: Tensor,
                          offsets: Optional[Tensor] = None,
                          max_norm: Optional[float] = None,
                          norm_type: float = 2,
                          scale_grad_by_freq: bool = False,
                          mode: str = "mean",
                          sparse: bool = False,
                          per_sample_weights: Optional[Tensor] = None,
                          include_last_offset: bool = False,
                          padding_idx: Optional[int] = None):

            if per_sample_weights is not None:
                qtzr = self.input_quantizers[0]
                per_sample_weights = _quantize_if_applicable(per_sample_weights, qtzr)

            output_encodings = self.output_quantizers[0].get_encoding() if self.output_quantizers[0] else None

            return fn(input,
                      weight,
                      offsets=offsets,
                      max_norm=max_norm,
                      norm_type=norm_type,
                      scale_grad_by_freq=scale_grad_by_freq,
                      mode=mode,
                      sparse=sparse,
                      per_sample_weights=per_sample_weights,
                      include_last_offset=include_last_offset,
                      padding_idx=padding_idx,
                      output_encodings=output_encodings)

        return embedding_bag


@QuantizationMixin.implements(nn.FeatureAlphaDropout)
class QuantizedFeatureAlphaDropout(_DispatchMixin, QuantizationMixin, nn.FeatureAlphaDropout):
    """ Quantized FeatureAlphaDropout """
    _builtin_torch_fn = F.feature_alpha_dropout
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Flatten)
class QuantizedFlatten(_DispatchMixin, QuantizationMixin, nn.Flatten):
    """ Quantized Flatten """
    def _get_builtin_torch_fn(self):
        return Tensor.flatten

    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Fold)
class QuantizedFold(_DispatchMixin, QuantizationMixin, nn.Fold):
    """ Quantized Fold """
    _builtin_torch_fn = F.fold
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.FractionalMaxPool2d)
class QuantizedFractionalMaxPool2d(_DispatchMixin, QuantizationMixin, nn.FractionalMaxPool2d):
    """ Quantized FractionalMaxPool2d """
    _builtin_torch_fn = F.fractional_max_pool2d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.FractionalMaxPool3d)
class QuantizedFractionalMaxPool3d(_DispatchMixin, QuantizationMixin, nn.FractionalMaxPool3d):
    """ Quantized FractionalMaxPool3d """
    _builtin_torch_fn = F.fractional_max_pool3d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.GELU)
class QuantizedGELU(_DispatchMixin, QuantizationMixin, nn.GELU):
    """ Quantized GELU """
    _builtin_torch_fn = F.gelu
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.GLU)
class QuantizedGLU(_DispatchMixin, QuantizationMixin, nn.GLU):
    """ Quantized GLU """
    _builtin_torch_fn = F.glu
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.GRU)
class QuantizedGRU(_DispatchMixin, QuantizationMixin, nn.GRU):
    """ Quantized GRU """
    _builtin_torch_fn = _gru

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None, None])

    def _quantize_inputs(self, args, apply):
        if args[1].is_floating_point():
            input, hx, *others = args
            batch_sizes = None
        else:
            input, batch_sizes, hx, *others = args

        input = apply(input, self.input_quantizers[0])
        hx = apply(hx, self.input_quantizers[1])

        if batch_sizes is None:
            return input, hx, *others
        return input, batch_sizes, hx, *others

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        assert fn == _gru
        apply = _quantize_dequantize_if_applicable

        def gru(*args):
            args = self._quantize_inputs(args, apply)
            output, h_n = fn(*args)
            return (
                apply(output, self.output_quantizers[0]),
                apply(h_n, self.output_quantizers[1]),
            )

        return gru

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        apply = _quantize_if_applicable

        def gru(*args):
            args = self._quantize_inputs(args, apply)
            output_encodings = tuple(qtzr and qtzr.get_encoding() for qtzr in self.output_quantizers)
            return fn(*args, output_encodings=output_encodings)

        return gru


@QuantizationMixin.implements(nn.GRUCell)
class QuantizedGRUCell(_DispatchMixin, QuantizationMixin, nn.GRUCell):
    """ Quantized GRUCell """
    _builtin_torch_fn = _gru_cell

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None])

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        assert fn == _gru_cell
        apply = _quantize_dequantize_if_applicable

        def gru_cell(input, hx, *args, **kwargs):
            input = apply(input, self.input_quantizers[0])
            hx = apply(hx, self.input_quantizers[1])
            output = fn(input, hx, *args, **kwargs)
            return apply(output, self.output_quantizers[0])

        return gru_cell

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        apply = _quantize_if_applicable

        def gru_cell(input, hx, *args, **kwargs):
            input = apply(input, self.input_quantizers[0])
            hx = apply(hx, self.input_quantizers[1])
            output_encodings = self.output_quantizers[0] and self.output_quantizers[0].get_encoding()
            return fn(input, hx, *args, **kwargs, output_encodings=output_encodings)

        return gru_cell


@QuantizationMixin.implements(nn.GaussianNLLLoss)
class QuantizedGaussianNLLLoss(_DispatchMixin, QuantizationMixin, nn.GaussianNLLLoss):
    """ Quantized GaussianNLLLoss """
    _builtin_torch_fn = F.gaussian_nll_loss
    __quant_init__ = __ternary__


@QuantizationMixin.implements(nn.GroupNorm)
class QuantizedGroupNorm(_DispatchMixin, QuantizationMixin, nn.GroupNorm):
    """ Quantized GroupNorm """
    _builtin_torch_fn = F.group_norm
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Hardshrink)
class QuantizedHardshrink(_DispatchMixin, QuantizationMixin, nn.Hardshrink):
    """ Quantized Hardshrink """
    _builtin_torch_fn = F.hardshrink
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Hardsigmoid)
class QuantizedHardsigmoid(_DispatchMixin, QuantizationMixin, nn.Hardsigmoid):
    """ Quantized Hardsigmoid """
    _builtin_torch_fn = F.hardsigmoid
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Hardswish)
class QuantizedHardswish(_DispatchMixin, QuantizationMixin, nn.Hardswish):
    """ Quantized Hardswish """
    _builtin_torch_fn = F.hardswish
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Hardtanh)
class QuantizedHardtanh(_DispatchMixin, QuantizationMixin, nn.Hardtanh):
    """ Quantized Hardtanh """
    _builtin_torch_fn = F.hardtanh
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.HingeEmbeddingLoss)
class QuantizedHingeEmbeddingLoss(_DispatchMixin, QuantizationMixin, nn.HingeEmbeddingLoss):
    """ Quantized HingeEmbeddingLoss """
    _builtin_torch_fn = F.hinge_embedding_loss
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.HuberLoss)
class QuantizedHuberLoss(_DispatchMixin, QuantizationMixin, nn.HuberLoss):
    """ Quantized HuberLoss """
    _builtin_torch_fn = F.huber_loss
    __quant_init__ = __binary__


# @QuantizationMixin.implements(nn.Identity)
# class QuantizedIdentity(_DispatchMixin, QuantizationMixin, nn.Identity):
#     """ Quantized Identity """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.InstanceNorm1d)
class QuantizedInstanceNorm1d(_DispatchMixin, QuantizationMixin, nn.InstanceNorm1d):
    """ Quantized InstanceNorm1d """
    _builtin_torch_fn = F.instance_norm
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.InstanceNorm2d)
class QuantizedInstanceNorm2d(_DispatchMixin, QuantizationMixin, nn.InstanceNorm2d):
    """ Quantized InstanceNorm2d """
    _builtin_torch_fn = F.instance_norm
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.InstanceNorm3d)
class QuantizedInstanceNorm3d(_DispatchMixin, QuantizationMixin, nn.InstanceNorm3d):
    """ Quantized InstanceNorm3d """
    _builtin_torch_fn = F.instance_norm
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.KLDivLoss)
class QuantizedKLDivLoss(_DispatchMixin, QuantizationMixin, nn.KLDivLoss):
    """ Quantized KLDivLoss """
    _builtin_torch_fn = F.kl_div
    __quant_init__ = __binary__


@QuantizationMixin.implements(nn.L1Loss)
class QuantizedL1Loss(_DispatchMixin, QuantizationMixin, nn.L1Loss):
    """ Quantized L1Loss """
    _builtin_torch_fn = F.l1_loss
    __quant_init__ = __binary__


@QuantizationMixin.implements(nn.LPPool1d)
class QuantizedLPPool1d(_DispatchMixin, QuantizationMixin, nn.LPPool1d):
    """ Quantized LPPool1d """
    _builtin_torch_fn = F.lp_pool1d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.LPPool2d)
class QuantizedLPPool2d(_DispatchMixin, QuantizationMixin, nn.LPPool2d):
    """ Quantized LPPool2d """
    _builtin_torch_fn = F.lp_pool2d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.LSTM)
class QuantizedLSTM(_DispatchMixin, QuantizationMixin, nn.LSTM):
    """ Quantized LSTM """
    _builtin_torch_fn = _lstm

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None])
        self.output_quantizers = nn.ModuleList([None, None, None])

    def _quantize_inputs(self, args, apply):
        if isinstance(args[1], Tensor):
            input, batch_sizes, hx, *others = args
        else:
            input, hx, *others = args
            batch_sizes = None

        input = apply(input, self.input_quantizers[0])
        h, c = hx
        h_qtzr, c_qtzr = self.input_quantizers[1:]
        hx = (apply(h, h_qtzr), apply(c, c_qtzr))

        if batch_sizes is None:
            return input, hx, *others
        return input, batch_sizes, hx, *others

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        assert fn == _lstm
        apply = _quantize_dequantize_if_applicable

        def lstm(*args):
            args = self._quantize_inputs(args, apply)
            output, h_n, c_n = fn(*args)
            return (
                apply(output, self.output_quantizers[0]),
                apply(h_n, self.output_quantizers[1]),
                apply(c_n, self.output_quantizers[2]),
            )

        return lstm

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        apply = _quantize_if_applicable

        def lstm(*args):
            args = self._quantize_inputs(args, apply)
            output_encodings = tuple(qtzr and qtzr.get_encoding() for qtzr in self.output_quantizers)
            return fn(*args, output_encodings=output_encodings)

        return lstm


@QuantizationMixin.implements(nn.LSTMCell)
class QuantizedLSTMCell(_DispatchMixin, QuantizationMixin, nn.LSTMCell):
    """ Quantized LSTMCell """
    _builtin_torch_fn = _lstm_cell

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None, None])
        self.output_quantizers = nn.ModuleList([None, None])

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        assert fn == _lstm_cell
        apply = _quantize_dequantize_if_applicable

        def lstm_cell(input, hx, *args, **kwargs):
            input = apply(input, self.input_quantizers[0])
            h, c = hx
            h_qtzr, c_qtzr = self.input_quantizers[1:]
            hx = (apply(h, h_qtzr), apply(c, c_qtzr))

            hx, cx = fn(input, hx, *args, **kwargs)
            return (
                apply(hx, self.output_quantizers[0]),
                apply(cx, self.output_quantizers[1]),
            )

        return lstm_cell

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        apply = _quantize_if_applicable

        def lstm_cell(input, hx, *args, **kwargs):
            input = apply(input, self.input_quantizers[0])
            h, c = hx
            h_qtzr, c_qtzr = self.input_quantizers[1:]
            hx = (apply(h, h_qtzr), apply(c, c_qtzr))

            output_encodings = tuple(qtzr and qtzr.get_encoding() for qtzr in self.output_quantizers)
            return fn(input, hx, *args, **kwargs, output_encodings=output_encodings)

        return lstm_cell


@QuantizationMixin.implements(nn.LayerNorm)
class QuantizedLayerNorm(_DispatchMixin, QuantizationMixin, nn.LayerNorm):
    """ Quantized LayerNorm """
    _builtin_torch_fn = F.layer_norm
    __quant_init__ = __unary__


# @QuantizationMixin.implements(nn.LazyBatchNorm1d)
# class QuantizedLazyBatchNorm1d(_DispatchMixin, QuantizationMixin, nn.LazyBatchNorm1d):
#     """ Quantized LazyBatchNorm1d """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyBatchNorm2d)
# class QuantizedLazyBatchNorm2d(_DispatchMixin, QuantizationMixin, nn.LazyBatchNorm2d):
#     """ Quantized LazyBatchNorm2d """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyBatchNorm3d)
# class QuantizedLazyBatchNorm3d(_DispatchMixin, QuantizationMixin, nn.LazyBatchNorm3d):
#     """ Quantized LazyBatchNorm3d """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyConv1d)
# class QuantizedLazyConv1d(_DispatchMixin, QuantizationMixin, nn.LazyConv1d):
#     """ Quantized LazyConv1d """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyConv2d)
# class QuantizedLazyConv2d(_DispatchMixin, QuantizationMixin, nn.LazyConv2d):
#     """ Quantized LazyConv2d """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyConv3d)
# class QuantizedLazyConv3d(_DispatchMixin, QuantizationMixin, nn.LazyConv3d):
#     """ Quantized LazyConv3d """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyConvTranspose1d)
# class QuantizedLazyConvTranspose1d(_DispatchMixin, QuantizationMixin, nn.LazyConvTranspose1d):
#     """ Quantized LazyConvTranspose1d """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyConvTranspose2d)
# class QuantizedLazyConvTranspose2d(_DispatchMixin, QuantizationMixin, nn.LazyConvTranspose2d):
#     """ Quantized LazyConvTranspose2d """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyConvTranspose3d)
# class QuantizedLazyConvTranspose3d(_DispatchMixin, QuantizationMixin, nn.LazyConvTranspose3d):
#     """ Quantized LazyConvTranspose3d """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyInstanceNorm1d)
# class QuantizedLazyInstanceNorm1d(_DispatchMixin, QuantizationMixin, nn.LazyInstanceNorm1d):
#     """ Quantized LazyInstanceNorm1d """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyInstanceNorm2d)
# class QuantizedLazyInstanceNorm2d(_DispatchMixin, QuantizationMixin, nn.LazyInstanceNorm2d):
#     """ Quantized LazyInstanceNorm2d """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyInstanceNorm3d)
# class QuantizedLazyInstanceNorm3d(_DispatchMixin, QuantizationMixin, nn.LazyInstanceNorm3d):
#     """ Quantized LazyInstanceNorm3d """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.LazyLinear)
# class QuantizedLazyLinear(_DispatchMixin, QuantizationMixin, nn.LazyLinear):
#     """ Quantized LazyLinear """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.LeakyReLU)
class QuantizedLeakyReLU(_DispatchMixin, QuantizationMixin, nn.LeakyReLU):
    """ Quantized LeakyReLU """
    _builtin_torch_fn = F.leaky_relu
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Linear)
class QuantizedLinear(_DispatchMixin, QuantizationMixin, nn.Linear):
    """ Quantized Linear """
    _builtin_torch_fn = F.linear
    __quant_init__ = __unary__

    # Only allow activation recompute (a.k.a activation checkpointing) for QuantizedLinear.
    # This is mainly to reduce memory footprint of QAT of large language models.
    @allow_recompute
    def forward(self, *args, **kwargs):
        # Workaround for deepspeed.
        # Deepspeed zero3 sometimes forcefully mokey-patches F.linear to torch.addmm,
        # which collides with the core assumption of our dispatch mechanism
        # that nn.Linear invokes F.linear.
        # To circumvent this issue, we temporarily restore the original F.linear
        # before running forward.
        with patch_attr(F, 'linear', type(self)._builtin_torch_fn):
            return super().forward(*args, **kwargs)


@QuantizationMixin.implements(nn.LocalResponseNorm)
class QuantizedLocalResponseNorm(_DispatchMixin, QuantizationMixin, nn.LocalResponseNorm):
    """ Quantized LocalResponseNorm """
    _builtin_torch_fn = F.local_response_norm
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.LogSigmoid)
class QuantizedLogSigmoid(_DispatchMixin, QuantizationMixin, nn.LogSigmoid):
    """ Quantized LogSigmoid """
    _builtin_torch_fn = F.logsigmoid
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.LogSoftmax)
class QuantizedLogSoftmax(_DispatchMixin, QuantizationMixin, nn.LogSoftmax):
    """ Quantized LogSoftmax """
    _builtin_torch_fn = F.log_softmax
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.MSELoss)
class QuantizedMSELoss(_DispatchMixin, QuantizationMixin, nn.MSELoss):
    """ Quantized MSELoss """
    _builtin_torch_fn = F.mse_loss
    __quant_init__ = __binary__


@QuantizationMixin.implements(nn.MarginRankingLoss)
class QuantizedMarginRankingLoss(_DispatchMixin, QuantizationMixin, nn.MarginRankingLoss):
    """ Quantized MarginRankingLoss """
    _builtin_torch_fn = F.margin_ranking_loss
    __quant_init__ = __binary__


@QuantizationMixin.implements(nn.MaxPool1d)
class QuantizedMaxPool1d(_DispatchMixin, QuantizationMixin, nn.MaxPool1d):
    """ Quantized MaxPool1d """
    _builtin_torch_fn = F.max_pool1d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.MaxPool2d)
class QuantizedMaxPool2d(_DispatchMixin, QuantizationMixin, nn.MaxPool2d):
    """ Quantized MaxPool2d """
    _builtin_torch_fn = F.max_pool2d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.MaxPool3d)
class QuantizedMaxPool3d(_DispatchMixin, QuantizationMixin, nn.MaxPool3d):
    """ Quantized MaxPool3d """
    _builtin_torch_fn = F.max_pool3d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.MaxUnpool1d)
class QuantizedMaxUnpool1d(_DispatchMixin, QuantizationMixin, nn.MaxUnpool1d):
    """ Quantized MaxUnpool1d """
    _builtin_torch_fn = F.max_unpool1d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.MaxUnpool2d)
class QuantizedMaxUnpool2d(_DispatchMixin, QuantizationMixin, nn.MaxUnpool2d):
    """ Quantized MaxUnpool2d """
    _builtin_torch_fn = F.max_unpool2d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.MaxUnpool3d)
class QuantizedMaxUnpool3d(_DispatchMixin, QuantizationMixin, nn.MaxUnpool3d):
    """ Quantized MaxUnpool3d """
    _builtin_torch_fn = F.max_unpool3d
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Mish)
class QuantizedMish(_DispatchMixin, QuantizationMixin, nn.Mish):
    """ Quantized Mish """
    _builtin_torch_fn = F.mish
    __quant_init__ = __unary__


# @QuantizationMixin.implements(nn.Module)
# class QuantizedModule(_DispatchMixin, QuantizationMixin, nn.Module):
#     """ Quantized Module """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.ModuleDict)
# class QuantizedModuleDict(_DispatchMixin, QuantizationMixin, nn.ModuleDict):
#     """ Quantized ModuleDict """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.ModuleList)
# class QuantizedModuleList(_DispatchMixin, QuantizationMixin, nn.ModuleList):
#     """ Quantized ModuleList """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.MultiLabelMarginLoss)
class QuantizedMultiLabelMarginLoss(_DispatchMixin, QuantizationMixin, nn.MultiLabelMarginLoss):
    """ Quantized MultiLabelMarginLoss """
    _builtin_torch_fn = F.multilabel_margin_loss
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.MultiLabelSoftMarginLoss)
class QuantizedMultiLabelSoftMarginLoss(_DispatchMixin, QuantizationMixin, nn.MultiLabelSoftMarginLoss):
    """ Quantized MultiLabelSoftMarginLoss """
    _builtin_torch_fn = F.multilabel_soft_margin_loss
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.MultiMarginLoss)
class QuantizedMultiMarginLoss(_DispatchMixin, QuantizationMixin, nn.MultiMarginLoss):
    """ Quantized MultiMarginLoss """
    _builtin_torch_fn = F.multi_margin_loss
    __quant_init__ = __unary__


# @QuantizationMixin.implements(nn.MultiheadAttention)
# class QuantizedMultiheadAttention(_DispatchMixin, QuantizationMixin, nn.MultiheadAttention):
#     """ Quantized MultiheadAttention """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.NLLLoss)
class QuantizedNLLLoss(_DispatchMixin, QuantizationMixin, nn.NLLLoss):
    """ Quantized NLLLoss """
    _builtin_torch_fn = F.nll_loss
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.NLLLoss2d)
class QuantizedNLLLoss2d(_DispatchMixin, QuantizationMixin, nn.NLLLoss2d):
    """ Quantized NLLLoss2d """
    _builtin_torch_fn = F.nll_loss
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.PReLU)
class QuantizedPReLU(_DispatchMixin, QuantizationMixin, nn.PReLU):
    """ Quantized PReLU """
    _builtin_torch_fn = F.prelu
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.PairwiseDistance)
class QuantizedPairwiseDistance(_DispatchMixin, QuantizationMixin, nn.PairwiseDistance):
    """ Quantized PairwiseDistance """
    _builtin_torch_fn = F.pairwise_distance
    __quant_init__ = __binary__


# @QuantizationMixin.implements(nn.ParameterDict)
# class QuantizedParameterDict(_DispatchMixin, QuantizationMixin, nn.ParameterDict):
#     """ Quantized ParameterDict """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.ParameterList)
# class QuantizedParameterList(_DispatchMixin, QuantizationMixin, nn.ParameterList):
#     """ Quantized ParameterList """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.PixelShuffle)
class QuantizedPixelShuffle(_DispatchMixin, QuantizationMixin, nn.PixelShuffle):
    """ Quantized PixelShuffle """
    _builtin_torch_fn = F.pixel_shuffle
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.PixelUnshuffle)
class QuantizedPixelUnshuffle(_DispatchMixin, QuantizationMixin, nn.PixelUnshuffle):
    """ Quantized PixelUnshuffle """
    _builtin_torch_fn = F.pixel_unshuffle
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.PoissonNLLLoss)
class QuantizedPoissonNLLLoss(_DispatchMixin, QuantizationMixin, nn.PoissonNLLLoss):
    """ Quantized PoissonNLLLoss """
    _builtin_torch_fn = F.poisson_nll_loss
    __quant_init__ = __binary__


@QuantizationMixin.implements(nn.RNN)
class QuantizedRNN(_DispatchMixin, QuantizationMixin, nn.RNN):
    """ Quantized RNN """
    def _get_builtin_torch_fn(self):
        assert self.mode in ('RNN_TANH', 'RNN_RELU')
        if self.mode == 'RNN_TANH':
            return _rnn_tanh
        return _rnn_relu

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None, None])

    def _quantize_inputs(self, args, apply):
        if args[1].is_floating_point():
            input, hx, *others = args
            batch_sizes = None
        else:
            input, batch_sizes, hx, *others = args

        input = apply(input, self.input_quantizers[0])
        hx = apply(hx, self.input_quantizers[1])

        if batch_sizes is None:
            return input, hx, *others
        return input, batch_sizes, hx, *others

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        assert fn in (_rnn_tanh, _rnn_relu)
        apply = _quantize_dequantize_if_applicable

        def rnn(*args):
            args = self._quantize_inputs(args, apply)
            output, h_n = fn(*args)
            return (
                apply(output, self.output_quantizers[0]),
                apply(h_n, self.output_quantizers[1]),
            )

        return rnn

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        apply = _quantize_if_applicable

        def rnn(*args):
            args = self._quantize_inputs(args, apply)
            output_encodings = tuple(qtzr and qtzr.get_encoding() for qtzr in self.output_quantizers)
            return fn(*args, output_encodings=output_encodings)

        return rnn


# @QuantizationMixin.implements(nn.RNNBase)
# class QuantizedRNNBase(_DispatchMixin, QuantizationMixin, nn.RNNBase):
#     """ Quantized RNNBase """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.RNNCell)
class QuantizedRNNCell(_DispatchMixin, QuantizationMixin, nn.RNNCell):
    """ Quantized RNNCell """
    def _get_builtin_torch_fn(self):
        assert self.nonlinearity in ("tanh", "relu")

        if self.nonlinearity == "tanh":
            return _rnn_tanh_cell
        return _rnn_relu_cell

    def __quant_init__(self):
        super().__quant_init__()
        # pylint: disable=attribute-defined-outside-init
        self.input_quantizers = nn.ModuleList([None, None])
        self.output_quantizers = nn.ModuleList([None])

    def _builtin_torch_fn_helper(self, fn: Callable[..., Tensor]):
        assert fn in (_rnn_tanh_cell, _rnn_relu_cell)
        apply = _quantize_dequantize_if_applicable

        def rnn_cell(input, hx, *args, **kwargs):
            input = apply(input, self.input_quantizers[0])
            hx = apply(hx, self.input_quantizers[1])
            output = fn(input, hx, *args, **kwargs)
            return apply(output, self.output_quantizers[0])

        return rnn_cell

    def _custom_kernel_helper(self, fn: Callable[..., QuantizedTensorBase]):
        apply = _quantize_if_applicable

        def rnn_cell(input, hx, *args, **kwargs):
            input = apply(input, self.input_quantizers[0])
            hx = apply(hx, self.input_quantizers[1])
            output_encodings = self.output_quantizers[0] and self.output_quantizers[0].get_encoding()
            return fn(input, hx, *args, **kwargs, output_encodings=output_encodings)

        return rnn_cell


# @QuantizationMixin.implements(nn.RNNCellBase)
# class QuantizedRNNCellBase(_DispatchMixin, QuantizationMixin, nn.RNNCellBase):
#     """ Quantized RNNCellBase """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.RReLU)
class QuantizedRReLU(_DispatchMixin, QuantizationMixin, nn.RReLU):
    """ Quantized RReLU """
    _builtin_torch_fn = F.rrelu
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ReLU)
class QuantizedReLU(_DispatchMixin, QuantizationMixin, nn.ReLU):
    """ Quantized ReLU """
    _builtin_torch_fn = F.relu
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ReLU6)
class QuantizedReLU6(_DispatchMixin, QuantizationMixin, nn.ReLU6):
    """ Quantized ReLU6 """
    _builtin_torch_fn = F.hardtanh
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ReflectionPad1d)
class QuantizedReflectionPad1d(_DispatchMixin, QuantizationMixin, nn.ReflectionPad1d):
    """ Quantized ReflectionPad1d """
    _builtin_torch_fn = F.pad
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ReflectionPad2d)
class QuantizedReflectionPad2d(_DispatchMixin, QuantizationMixin, nn.ReflectionPad2d):
    """ Quantized ReflectionPad2d """
    _builtin_torch_fn = F.pad
    __quant_init__ = __unary__


if version.parse(torch.__version__) >= version.parse("1.10.0"):
    @QuantizationMixin.implements(nn.ReflectionPad3d)
    class QuantizedReflectionPad3d(_DispatchMixin, QuantizationMixin, nn.ReflectionPad3d):
        """ Quantized ReflectionPad3d """
        _builtin_torch_fn = F.pad
        __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ReplicationPad1d)
class QuantizedReplicationPad1d(_DispatchMixin, QuantizationMixin, nn.ReplicationPad1d):
    """ Quantized ReplicationPad1d """
    _builtin_torch_fn = F.pad
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ReplicationPad2d)
class QuantizedReplicationPad2d(_DispatchMixin, QuantizationMixin, nn.ReplicationPad2d):
    """ Quantized ReplicationPad2d """
    _builtin_torch_fn = F.pad
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ReplicationPad3d)
class QuantizedReplicationPad3d(_DispatchMixin, QuantizationMixin, nn.ReplicationPad3d):
    """ Quantized ReplicationPad3d """
    _builtin_torch_fn = F.pad
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.SELU)
class QuantizedSELU(_DispatchMixin, QuantizationMixin, nn.SELU):
    """ Quantized SELU """
    _builtin_torch_fn = F.selu
    __quant_init__ = __unary__


# @QuantizationMixin.implements(nn.Sequential)
# class QuantizedSequential(_DispatchMixin, QuantizationMixin, nn.Sequential):
#     """ Quantized Sequential """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.SiLU)
class QuantizedSiLU(_DispatchMixin, QuantizationMixin, nn.SiLU):
    """ Quantized SiLU """
    _builtin_torch_fn = F.silu
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Sigmoid)
class QuantizedSigmoid(_DispatchMixin, QuantizationMixin, nn.Sigmoid):
    """ Quantized Sigmoid """
    _builtin_torch_fn = torch.sigmoid
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.SmoothL1Loss)
class QuantizedSmoothL1Loss(_DispatchMixin, QuantizationMixin, nn.SmoothL1Loss):
    """ Quantized SmoothL1Loss """
    _builtin_torch_fn = F.smooth_l1_loss
    __quant_init__ = __binary__


@QuantizationMixin.implements(nn.SoftMarginLoss)
class QuantizedSoftMarginLoss(_DispatchMixin, QuantizationMixin, nn.SoftMarginLoss):
    """ Quantized SoftMarginLoss """
    _builtin_torch_fn = F.soft_margin_loss
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Softmax)
class QuantizedSoftmax(_DispatchMixin, QuantizationMixin, nn.Softmax):
    """ Quantized Softmax """
    _builtin_torch_fn = F.softmax
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Softmax2d)
class QuantizedSoftmax2d(_DispatchMixin, QuantizationMixin, nn.Softmax2d):
    """ Quantized Softmax2d """
    _builtin_torch_fn = F.softmax
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Softmin)
class QuantizedSoftmin(_DispatchMixin, QuantizationMixin, nn.Softmin):
    """ Quantized Softmin """
    _builtin_torch_fn = F.softmin
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Softplus)
class QuantizedSoftplus(_DispatchMixin, QuantizationMixin, nn.Softplus):
    """ Quantized Softplus """
    _builtin_torch_fn = F.softplus
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Softshrink)
class QuantizedSoftshrink(_DispatchMixin, QuantizationMixin, nn.Softshrink):
    """ Quantized Softshrink """
    _builtin_torch_fn = F.softshrink
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Softsign)
class QuantizedSoftsign(_DispatchMixin, QuantizationMixin, nn.Softsign):
    """ Quantized Softsign """
    _builtin_torch_fn = F.softsign
    __quant_init__ = __unary__


# @QuantizationMixin.implements(nn.SyncBatchNorm)
# class QuantizedSyncBatchNorm(_DispatchMixin, QuantizationMixin, nn.SyncBatchNorm):
#     """ Quantized SyncBatchNorm """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.Tanh)
class QuantizedTanh(_DispatchMixin, QuantizationMixin, nn.Tanh):
    """ Quantized Tanh """
    _builtin_torch_fn = torch.tanh
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Tanhshrink)
class QuantizedTanhshrink(_DispatchMixin, QuantizationMixin, nn.Tanhshrink):
    """ Quantized Tanhshrink """
    _builtin_torch_fn = F.tanhshrink
    __quant_init__ = __unary__


# @QuantizationMixin.implements(nn.Threshold)
@QuantizationMixin.implements(nn.Threshold)
class QuantizedThreshold(_DispatchMixin, QuantizationMixin, nn.Threshold):
    """ Quantized Threshold """
    _builtin_torch_fn = F.threshold
    __quant_init__ = __unary__


# @QuantizationMixin.implements(nn.Transformer)
# class QuantizedTransformer(_DispatchMixin, QuantizationMixin, nn.Transformer):
#     """ Quantized Transformer """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.TransformerDecoder)
# class QuantizedTransformerDecoder(_DispatchMixin, QuantizationMixin, nn.TransformerDecoder):
#     """ Quantized TransformerDecoder """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.TransformerDecoderLayer)
# class QuantizedTransformerDecoderLayer(_DispatchMixin, QuantizationMixin, nn.TransformerDecoderLayer):
#     """ Quantized TransformerDecoderLayer """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.TransformerEncoder)
# class QuantizedTransformerEncoder(_DispatchMixin, QuantizationMixin, nn.TransformerEncoder):
#     """ Quantized TransformerEncoder """
#     _builtin_torch_fn = ...


# @QuantizationMixin.implements(nn.TransformerEncoderLayer)
# class QuantizedTransformerEncoderLayer(_DispatchMixin, QuantizationMixin, nn.TransformerEncoderLayer):
#     """ Quantized TransformerEncoderLayer """
#     _builtin_torch_fn = ...


@QuantizationMixin.implements(nn.TripletMarginLoss)
class QuantizedTripletMarginLoss(_DispatchMixin, QuantizationMixin, nn.TripletMarginLoss):
    """ Quantized TripletMarginLoss """
    _builtin_torch_fn = F.triplet_margin_loss
    __quant_init__ = __ternary__


@QuantizationMixin.implements(nn.TripletMarginWithDistanceLoss)
class QuantizedTripletMarginWithDistanceLoss(_DispatchMixin, QuantizationMixin, nn.TripletMarginWithDistanceLoss):
    """ Quantized TripletMarginWithDistanceLoss """
    _builtin_torch_fn = F.triplet_margin_with_distance_loss
    __quant_init__ = __ternary__


@QuantizationMixin.implements(nn.Unflatten)
class QuantizedUnflatten(_DispatchMixin, QuantizationMixin, nn.Unflatten):
    """ Quantized Unflatten """
    def _get_builtin_torch_fn(self):
        return Tensor.unflatten


@QuantizationMixin.implements(nn.Unfold)
class QuantizedUnfold(_DispatchMixin, QuantizationMixin, nn.Unfold):
    """ Quantized Unfold """
    _builtin_torch_fn = F.unfold
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.Upsample)
class QuantizedUpsample(_DispatchMixin, QuantizationMixin, nn.Upsample):
    """ Quantized Upsample """
    _builtin_torch_fn = F.interpolate
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.UpsamplingBilinear2d)
class QuantizedUpsamplingBilinear2d(_DispatchMixin, QuantizationMixin, nn.UpsamplingBilinear2d):
    """ Quantized UpsamplingBilinear2d """
    _builtin_torch_fn = F.interpolate
    __quant_init__ = __unary__


@QuantizationMixin.implements(nn.UpsamplingNearest2d)
class QuantizedUpsamplingNearest2d(_DispatchMixin, QuantizationMixin, nn.UpsamplingNearest2d):
    """ Quantized UpsamplingNearest2d """
    _builtin_torch_fn = F.interpolate
    __quant_init__ = __unary__


if version.parse(torch.__version__) >= version.parse("2.1.0"):
    @QuantizationMixin.implements(nn.ZeroPad1d)
    class QuantizedZeroPad1d(_DispatchMixin, QuantizationMixin, nn.ZeroPad1d):
        """ Quantized ZeroPad1d """
        _builtin_torch_fn = F.pad
        __quant_init__ = __unary__


@QuantizationMixin.implements(nn.ZeroPad2d)
class QuantizedZeroPad2d(_DispatchMixin, QuantizationMixin, nn.ZeroPad2d):
    """ Quantized ZeroPad2d """
    _builtin_torch_fn = F.pad
    __quant_init__ = __unary__


if version.parse(torch.__version__) >= version.parse("2.1.0"):
    @QuantizationMixin.implements(nn.ZeroPad3d)
    class QuantizedZeroPad3d(_DispatchMixin, QuantizationMixin, nn.ZeroPad3d):
        """ Quantized ZeroPad3d """
        _builtin_torch_fn = F.pad
        __quant_init__ = __unary__


del __nullary__
del __unary__
del __binary__
del __ternary__
