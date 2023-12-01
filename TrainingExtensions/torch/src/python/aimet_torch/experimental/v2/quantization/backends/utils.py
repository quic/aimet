# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=all
from typing import Protocol

import torch

from aimet_torch.experimental.v2.utils import _ContextManager


class _QuantizationBackendProtocol(Protocol):
    def quantize(self, input: torch.Tensor) -> torch.Tensor:
        ...

    def dequantize(self,
                   input: torch.Tensor,
                   scale: torch.Tensor,
                   offset: torch.Tensor) -> torch.Tensor:
        ...

    def quantize_dequantize(self, input: torch.Tensor) -> torch.Tensor:
        ...


_CURRENT_BACKEND = 'default'

_SUPPORTED_BACKENDS = {
    'default': None,
}


def set_global_backend(name: str):
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = name


def set_backend(name: str) -> _ContextManager:
    if name not in _SUPPORTED_BACKENDS:
        supported_backend_names = ", ".join(_SUPPORTED_BACKENDS.keys())
        raise RuntimeError(f"Backend '{name}' is not supported. "
                           f"Please choose one of: {supported_backend_names}")

    old_backend = _CURRENT_BACKEND
    action = lambda: set_global_backend(name)
    cleanup = lambda: set_global_backend(old_backend)
    return _ContextManager(action=action, cleanup=cleanup)



def get_backend() -> _QuantizationBackendProtocol:
    if _SUPPORTED_BACKENDS[_CURRENT_BACKEND] is None:
        # Lazy import
        import importlib
        module_name = f'aimet_torch.experimental.v2.quantization.backends.{_CURRENT_BACKEND}'
        _SUPPORTED_BACKENDS[_CURRENT_BACKEND] = importlib.import_module(module_name)

    return _SUPPORTED_BACKENDS[_CURRENT_BACKEND]


__all__ = ['set_global_backend', 'set_backend', 'get_backend']
