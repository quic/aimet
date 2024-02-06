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
import itertools
from typing import Type

import torch.nn as nn

from aimet_torch.experimental.v2.utils import patch_attr


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
            for quantizer in itertools.chain(self.input_quantizers, self.output_quantizers):
                if not quantizer:
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
