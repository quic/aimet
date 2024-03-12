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
""" Quantized modules"""

import contextlib
import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Type, Any, Tuple, Dict, Optional, overload

import torch
import torch.nn as nn
from torch import Tensor

from aimet_torch.experimental.v2.nn.quant_base import BaseQuantizationMixin
from aimet_torch.experimental.v2.nn.function_selector import FunctionSelector, _FunctionalLibrary
from aimet_torch.experimental.v2.quantization.quantizers.base import QuantizerBase
from aimet_torch.experimental.v2.quantization.quantized_tensor import QuantizedTensor
from aimet_torch.experimental.v2.utils import patch_attr, _ContextManager


_FUNCTION_SELECTOR = FunctionSelector([], strict=True)

def set_default_functional_library(library: _FunctionalLibrary, strict=True):
    """
    Set the default operator library(s) for quantized modules
    """
    global _FUNCTION_SELECTOR # pylint: disable=global-statement
    _FUNCTION_SELECTOR.set_libraries(library, strict)


def _set_default_function_selector(selector: FunctionSelector):
    global _FUNCTION_SELECTOR # pylint: disable=global-statement
    _FUNCTION_SELECTOR = selector

@contextlib.contextmanager
def _set_module_selector(modules, library, strict):
    modules = modules if isinstance(modules, list) else [modules]
    selector = FunctionSelector(library, strict)
    with contextlib.ExitStack() as stack:
        for module in modules:
            for m in module.modules():
                if isinstance(m, QuantizationMixin):
                    ctx = patch_attr(m, "_func_selector", lambda: selector)
                    stack.enter_context(ctx)

            if isinstance(module, QuantizationMixin):
                ctx = patch_attr(module, "_func_selector", lambda: selector)
                stack.enter_context(ctx)
        yield


def _set_global_function_selector(library, strict):
    selector = FunctionSelector(library, strict)
    old_selector = _FUNCTION_SELECTOR
    action = lambda: _set_default_function_selector(selector)
    cleanup = lambda: _set_default_function_selector(old_selector)
    return _ContextManager(action=action, cleanup=cleanup)


@overload
def set_functional_library(model: torch.nn.Module, library: _FunctionalLibrary, *, strict=True): # pylint:disable = unused-argument, function-redefined
    """
    Set the functional library for a given model

    :param model: torch.nn.Module(s) to set the library for
    :param library: operator library(s) to use
    :param strict: If True, raise an error when no valid kernel is found
    """

@overload
def set_functional_library(library: _FunctionalLibrary, *, strict=True): # pylint:disable = unused-argument, function-redefined
    """
    Set the functional library for a given model

    :param library: operator library(s) to use
    :param strict: If True, raise an error when no valid kernel is found
    """


def set_functional_library(*args, strict=True): # pylint:disable = function-redefined
    """
    Sets the functional library used by quantized layers
    """
    if len(args) == 1:
        library = args[0]
        return _set_global_function_selector(library, strict)
    if len(args) == 2:
        modules, library = args
        return _set_module_selector(modules, library, strict)
    raise RuntimeError("Invalid arguments, expected either (model, library, strict) or (library, strict)")


def _quantize_if_applicable(data: Any, quantizer: Optional[QuantizerBase]):
    """
    Quantize data if it is a quantizable type and quantize is not None
    """
    if quantizer and isinstance(data, Tensor) and data.is_floating_point():
        return quantizer(data)
    return data

def _dequantize_if_applicable(data: torch.Tensor):
    return data.dequantize() if isinstance(data, QuantizedTensor) else data


class QuantizationMixin(BaseQuantizationMixin, ABC): # pylint: disable=abstract-method
    """
    Mixin that allows dispatch to quantized operator libraries in place of native pytorch operations
    """

    cls_to_qcls = OrderedDict()  # quantized class -> original class
    qcls_to_cls = OrderedDict()  # original class -> quantized class
    op_key: str

    @contextlib.contextmanager
    def compute_encodings(self):
        def no_op(tensor_in: Tensor):
            return tensor_in

        with contextlib.ExitStack() as stack:
            for quantizer in itertools.chain(self.input_quantizers, self.output_quantizers):
                if quantizer is None:
                    continue
                # NOTE: This behavior is for backward-compatibility with V1 quantsim.
                stack.enter_context(patch_attr(quantizer, 'forward', no_op))
            dummy_funciton_selector = FunctionSelector([], strict=False)
            stack.enter_context(patch_attr(self, "_func_selector", lambda: dummy_funciton_selector))
            with super().compute_encodings():
                yield

    @contextlib.contextmanager
    def _patch_dequantized_parameters(self):
        with contextlib.ExitStack() as stack:
            for param_name, _ in self.param_quantizers.items():
                qparam = getattr(self, param_name)
                ctx = patch_attr(self, param_name, _dequantize_if_applicable(qparam))
                stack.enter_context(ctx)
            yield

    def _func_selector(self): # pylint:disable = no-self-use
        return _FUNCTION_SELECTOR

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
    def implements(cls, module_cls, op_key=None):
        """
        Decorator for registering quantized implementation of the given base class.
        """

        def wrapper(quantized_cls):
            quantized_cls.op_key = op_key or module_cls.__name__.lower()
            cls.cls_to_qcls[module_cls] = quantized_cls
            cls.qcls_to_cls[quantized_cls] = module_cls
            return quantized_cls

        return wrapper


# pylint: disable=arguments-differ, abstract-method

class _QuantizedUnaryOpMixin(QuantizationMixin, ABC):

    def quantized_forward(self, *args, **kwargs):
        x, *args = args
        x = _quantize_if_applicable(x, self.input_quantizers[0])

        with self._patch_quantized_parameters():
            kernel_args, kernel_kwargs = self.get_functional_args(x, *args, **kwargs)
            output_encodings = self.output_quantizers[0].get_encoding() if self.output_quantizers[0] else None
            kernel = self._func_selector().get_impl(self.op_key, *kernel_args, **kernel_kwargs, output_encodings=output_encodings)

            if kernel:
                output = kernel(*kernel_args, **kernel_kwargs, output_encodings=output_encodings)
            else:
                with self._patch_dequantized_parameters():
                    output = super().forward(_dequantize_if_applicable(x), *args, **kwargs)
                output = _quantize_if_applicable(output, self.output_quantizers[0])

        return output

    @abstractmethod
    def get_functional_args(self, x, *args, **kwargs) -> Tuple[Tuple, Dict]:
        """
        Return the args and keyword args to the layer's kernel call
        """


class _QuantizedBinaryOpMixin(QuantizationMixin, ABC):

    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = nn.ModuleList([None, None])

    def quantized_forward(self, *args, **kwargs):
        x, y, *args = args
        x = _quantize_if_applicable(x, self.input_quantizers[0])
        y = _quantize_if_applicable(y, self.input_quantizers[1])

        with self._patch_quantized_parameters():
            kernel_args, kernel_kwargs = self.get_functional_args(x, y, *args, **kwargs)
            output_encodings = self.output_quantizers[0].get_encoding if self.output_quantizers[0] else None
            kernel = self._func_selector().get_impl(self.op_key, *kernel_args, **kernel_kwargs,
                                                    output_encodings=output_encodings)

            if kernel:
                output = kernel(*args, **kwargs, output_encodings=output_encodings)
            else:
                with self._patch_dequantized_parameters():
                    output = super().forward(_dequantize_if_applicable(x), _dequantize_if_applicable(y), *args, **kwargs)
                output = _quantize_if_applicable(output, self.output_quantizers[0])

        return output

    @abstractmethod
    def get_functional_args(self, x, y, *args, **kwargs) -> Tuple[Tuple, Dict]:
        """
        Return the args and keyword args to the layer's kernel call
        """


@QuantizationMixin.implements(nn.Linear)
class QuantizedLinear(_QuantizedUnaryOpMixin, nn.Linear):
    """ Quantized Linear """
    def get_functional_args(self, x):
        return (x, self.weight), {"bias": self.bias}


@QuantizationMixin.implements(nn.GELU)
class QuantizedGELU(_QuantizedUnaryOpMixin, nn.GELU):
    """ Quantized GELU """

    def get_functional_args(self, x):
        return (x, ), {"approximate": self.approximate}


@QuantizationMixin.implements(nn.LayerNorm)
class QuantizedLayerNorm(_QuantizedUnaryOpMixin, nn.LayerNorm):
    """ Quantized LayerNorm """

    def get_functional_args(self, x):
        return (x, self.normalized_shape), {"weight": self.weight, "bias": self.bias, "eps": self.eps}

@QuantizationMixin.implements(nn.Softmax)
class QuantizedSoftmax(_QuantizedUnaryOpMixin, nn.Softmax):
    """ Quantized Softmax """

    def get_functional_args(self, x):
        return (x, self.dim), {}

@QuantizationMixin.implements(nn.Sigmoid)
class QuantizedSigmoid(_QuantizedUnaryOpMixin, nn.Sigmoid):
    """ Quantized Sigmoid """

    def get_functional_args(self, x):
        return (x, ), {}
