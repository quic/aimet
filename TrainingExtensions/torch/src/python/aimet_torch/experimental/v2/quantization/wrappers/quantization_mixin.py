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
# pylint: skip-file
""" Placeholder for _QuantizationMixin definition, to be deleted/moved/updated """

from abc import ABC, abstractmethod
from typing import Union, Tuple, Iterable, Optional, Mapping
from dataclasses import dataclass
from torch import nn


@dataclass
class _TensorSpec:
    """ Spec class for quantizer initialization """
    shape: Tuple[int, ...]
    bitwidth: int
    symmetric: bool
    qscheme: None


@dataclass
class _ModuleSpec:
    """ Spec class for wrapper initialization """
    input_spec: Iterable[Optional[_TensorSpec]]
    param_spec: Mapping[str, Optional[_TensorSpec]]
    output_spec: Iterable[Optional[_TensorSpec]]


class _QuantizationMixin(ABC):
    """ Base class for quantized modules """

    @classmethod
    def from_module(cls, module: nn.Module, spec: _ModuleSpec) -> Union['_QuantizationMixin', nn.Module]:
        ...

    @abstractmethod
    def input_quantizers(self):
        ...

    @abstractmethod
    def output_quantizers(self):
        ...

    @abstractmethod
    def param_quantizers(self):
        ...