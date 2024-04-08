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
"""Base class of quantized modules"""

import abc
import contextlib
import functools
import itertools
from typing import Type, List, Dict, Union, Iterable

import torch.nn as nn

from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.utils import patch_attr, _ContextManager


def _flatten_nn_module_list(module):
    """
    Flatten nested list of nn.Modules into a flat list
    """
    def flat_iter(mod):
        if isinstance(mod, (list, tuple, nn.ModuleList)):
            for x in mod:
                yield from flat_iter(x)
        else:
            yield mod

    return list(flat_iter(module))


class BaseQuantizationMixin(abc.ABC):
    """
    Mixin that implements quantization on top of regular pytorch modules.
    """

    input_quantizers: nn.ModuleList
    output_quantizers: nn.ModuleList
    param_quantizers: nn.ModuleDict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__quant_init__()

    def __quant_init__(self):
        """
        Initializer for quantized module. This method will be invoked right after __init__.
        """
        self.param_quantizers = nn.ModuleDict({
            name: None for name, _ in self.named_parameters(recurse=False)
        })
        # Currently assume single input & output
        self.input_quantizers = nn.ModuleList([None])
        self.output_quantizers = nn.ModuleList([None])

    def __call__(self, *args, **kwargs):
        self._compute_param_encodings(overwrite=False)

        with patch_attr(self, 'forward', self.quantized_forward):
            return super().__call__(*args, **kwargs)

    @abc.abstractmethod
    def quantized_forward(self, *args, **kwargs):
        """
        Forward function for quantized module.
        This method will replace the original forward function.
        """

    @contextlib.contextmanager
    def _patch_quantized_parameters(self):
        with contextlib.ExitStack() as stack:
            for param_name, param_quantizer in self.param_quantizers.items():
                if param_quantizer:
                    orig_param = getattr(self, param_name)
                    quantized_param = param_quantizer(orig_param)
                    ctx = patch_attr(self, param_name, quantized_param)
                    stack.enter_context(ctx)
            yield

    def _compute_param_encodings(self, overwrite: bool):
        for param_name, param_quantizer in self.param_quantizers.items():
            if not param_quantizer:
                continue

            if not param_quantizer.is_initialized() or overwrite:
                param = getattr(self, param_name)
                if param is not None:
                    with param_quantizer.compute_encodings():
                        _ = param_quantizer(param)

    @contextlib.contextmanager
    def compute_encodings(self):
        """
        Observe inputs and update quantization parameters based on the input statistics.
        """
        self._compute_param_encodings(overwrite=True)

        with contextlib.ExitStack() as stack:
            input_quantizers = _flatten_nn_module_list(self.input_quantizers)
            output_quantizers = _flatten_nn_module_list(self.output_quantizers)

            for quantizer in itertools.chain(input_quantizers, output_quantizers):
                if not isinstance(quantizer, QuantizerBase):
                    continue
                ctx = quantizer.compute_encodings()
                stack.enter_context(ctx)
            yield

    @classmethod
    @abc.abstractmethod
    def wrap(cls, module_cls: Type[nn.Module]):
        """
        Wrap a regular module class into a quantized module class
        """

    @classmethod
    def from_module(cls, module: nn.Module):
        """
        Create an instance of quantized module from a regular moudle instance
        """
        # pylint: disable=protected-access
        module_cls = type(module)
        qtzn_module_cls = cls.cls_to_qcls.get(module_cls, None)

        if not qtzn_module_cls:
            raise RuntimeError(
                f'The quantized module definition of {module_cls} is not registered. '
                f'Please register the quantized module definition of {module_cls} '
                f'using `@{cls.__name__}.implements({module_cls.__name__})` decorator.'
            )

        qtzn_module = cls.__new__(qtzn_module_cls)

        qtzn_module.__dict__ = module.__dict__.copy()
        qtzn_module._modules = module._modules.copy()
        qtzn_module._parameters = module._parameters.copy()
        qtzn_module._buffers = module._buffers.copy()

        qtzn_module.__quant_init__()
        return qtzn_module

    def export_input_encodings(self) -> List[List[Dict]]:
        """
        Returns a list of input encodings, each represented as a List of Dicts
        """
        return [
            quantizer.get_legacy_encodings() if isinstance(quantizer, QuantizerBase) else None
            for quantizer in _flatten_nn_module_list(self.input_quantizers)
        ]

    def import_input_encodings(self, encodings: Dict[str, Dict], ignore_when_quantizer_disabled=False,
                               disable_quantizer_without_encoding=True, freeze: bool = False):
        """
        Import input encodings represented in below format:
        {
            '0': dict,
            '1': dict,
            ...
        }

        :param encodings: Dictionary mapping quantizer index (str) to encoding (dict)
        :param ignore_when_quantizer_disabled: If True, does not raise RuntimeError when a quantizer is disabled
        :param disable_quantizer_without_encoding: If True, disable any quantizer without an encoding in `encodings`
        :param freeze: If True, freezes the quantizer's encodings after loading
        """
        for i, quantizer in enumerate(list(self.input_quantizers)):
            encoding = encodings.get(str(i), None)
            if not encoding:
                if disable_quantizer_without_encoding:
                    self.input_quantizers[i] = None
                continue
            if quantizer is None and not ignore_when_quantizer_disabled:
                raise RuntimeError
            if isinstance(encoding, dict):
                encoding = [encoding]
            quantizer.set_legacy_encodings(encoding)
            if freeze:
                quantizer._freeze_encoding() # pylint:disable = protected-access

    def export_output_encodings(self) -> List[List[Dict]]:
        """
        Returns a list of output encodings, each represented as a List of Dicts
        """
        return [
            quantizer.get_legacy_encodings() if isinstance(quantizer, QuantizerBase) else None
            for quantizer in _flatten_nn_module_list(self.output_quantizers)
        ]

    def import_output_encodings(self, encodings: Dict[str, Dict], ignore_when_quantizer_disabled=False,
                                disable_quantizer_without_encoding=True, freeze: bool = False):
        """
        Import output encodings represented in below format:
        {
            '0': dict,
            '1': dict,
            ...
        }

        :param encodings: Dictionary mapping quantizer index (str) to encoding (dict)
        :param ignore_when_quantizer_disabled: If True, does not raise RuntimeError when a quantizer is disabled
        :param disable_quantizer_without_encoding: If True, disable any quantizer without an encoding in `encodings`
        :param freeze: If True, freezes the quantizer's encodings after loading
        """
        for i, quantizer in enumerate(list(self.output_quantizers)):
            encoding = encodings.get(str(i), None)
            if not encoding:
                if disable_quantizer_without_encoding:
                    self.output_quantizers[i] = None
                continue
            if quantizer is None and not ignore_when_quantizer_disabled:
                raise RuntimeError
            if isinstance(encoding, dict):
                encoding = [encoding]
            quantizer.set_legacy_encodings(encoding)
            if freeze:
                quantizer._freeze_encoding() # pylint:disable = protected-access

    def export_param_encodings(self) -> Dict[str, List[Dict]]:
        """
        Returns a dict of {param name: param encodings}, with each encoding represented as a List of Dicts
        """
        return {
            param_name: quantizer.get_legacy_encodings() if isinstance(quantizer, QuantizerBase) else None
            for param_name, quantizer in self.param_quantizers.items()
        }

    def import_param_encodings(self, encodings: Dict[str, List[Dict]], ignore_when_quantizer_disabled=False,
                               disable_quantizer_without_encoding=True, freeze: bool = False):
        """
        Import parameter encodings represented in below format:
        {
            'param_name_0': [dict, dict, ...],
            'param_name_1': [dict, dict, ...],
            ...
        }

        :param encodings: Dictionary mapping quantizer parameter name (str) to encodings (dict)
        :param ignore_when_quantizer_disabled: If True, does not raise RuntimeError when a quantizer is disabled
        :param disable_quantizer_without_encoding: If True, disable any quantizer without an encoding in `encodings`
        :param freeze: If True, freezes the quantizer's encodings after loading
        """
        for param_name, quantizer in dict(self.param_quantizers).items():
            encoding = encodings.get(param_name, None)
            if not encoding:
                if disable_quantizer_without_encoding:
                    self.param_quantizers[param_name] = None
                continue
            if quantizer is None and not ignore_when_quantizer_disabled:
                raise RuntimeError
            if isinstance(encoding, dict):
                encoding = [encoding]
            quantizer.set_legacy_encodings(encoding)
            if freeze:
                quantizer._freeze_encoding() # pylint:disable = protected-access

    def get_original_module(self) -> nn.Module:
        """
        Returns the floating point version of quantized module
        """
        # pylint: disable=protected-access

        qtzn_module_cls = type(self)
        orig_module_cls = self.qcls_to_cls.get(qtzn_module_cls)

        orig_module = self.__new__(orig_module_cls)
        orig_module.__dict__ = self.__dict__.copy()
        orig_module.__dict__.pop('forward', None)

        orig_module._parameters = self._parameters.copy()
        orig_module._buffers = self._buffers.copy()
        orig_module._modules = self._modules.copy()
        del orig_module._modules['input_quantizers']
        del orig_module._modules['output_quantizers']
        del orig_module._modules['param_quantizers']

        return orig_module

    @property
    def _super_forward(self):
        # This is a manually/explicitly rewritten version of super().forward.
        # NOTE: This is an ad-hoc solution and will be removed in the later versions
        is_staticmethod = False
        is_classmethod = False

        for cls in type(self).__mro__:
            if not 'forward' in cls.__dict__:
                continue
            super_forward = cls.__dict__['forward']

            if isinstance(super_forward, staticmethod):
                is_staticmethod = True
            if isinstance(super_forward, classmethod):
                is_classmethod = True
            break
        else:
            raise RuntimeError

        super_forward = self.qcls_to_cls[type(self)].forward

        if is_staticmethod or is_classmethod:
            return super_forward

        return functools.partial(super_forward, self)

    def _remove_input_quantizers(self, indices: Union[int, Iterable[int]] = None):
        """
        Remove input quantizers
        :param indices: Indices of input quantizers to remove.
                If None, all input quantizers will be removed.
        """
        if isinstance(indices, int):
            indices = [indices]
        elif indices is None:
            indices = list(range(len(self.input_quantizers)))
        return _remove_quantizers(self.input_quantizers, indices)

    def _remove_param_quantizers(self, keys: Union[str, Iterable[str]] = None):
        """
        Remove parameter quantizers
        :param indices: Indices of parameter quantizers to remove.
                If None, all input quantizers will be removed.
        """
        if isinstance(keys, str):
            keys = [keys]
        elif keys is None:
            keys = list(self.param_quantizers.keys())
        return _remove_quantizers(self.param_quantizers, keys)

    def _remove_output_quantizers(self, indices: Union[int, Iterable[int]] = None):
        """
        Remove output quantizers
        :param indices: Indices of input quantizers to remove.
                If None, all input quantizers will be removed.
        """
        if isinstance(indices, int):
            indices = [indices]
        elif indices is None:
            indices = list(range(len(self.output_quantizers)))
        return _remove_quantizers(self.output_quantizers, indices)


def _remove_quantizers(quantizers, keys):
    orig_quantizers = {key: quantizers[key] for key in keys}

    def restore_quantizers():
        for key, orig_qtzr in orig_quantizers.items():
            quantizers[key] = orig_qtzr

    ctx = _ContextManager(action=lambda: None,
                          cleanup=restore_quantizers)

    try:
        for key in keys:
            quantizers[key] = None
    except Exception:
        ctx._cleanup() # pylint: disable=protected-access
        raise
    else:
        return ctx
