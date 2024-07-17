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
# pylint: disable=redefined-builtin

""" Utilities to use deepspeed """
import contextlib

import torch

from aimet_torch.v2.utils import (
    patch_attr,
    _ContextManager
)
from aimet_torch.v2.quantization.affine import MinMaxQuantizer

def gathered_parameter(param)-> _ContextManager:
    """
    A context that collects parameters that were partitioned via a
    :class:`deepspeed.zero.Init` context. The parameters are partitioned
    again upon exit.

    :param The parameter to be gathered and partitioned
    """

    action = lambda: param.all_gather() if hasattr(param, "ds_id") else None
    cleanup = lambda: param.partition() if hasattr(param, "ds_id") else None

    return _ContextManager(action, cleanup)

@contextlib.contextmanager
def transfer_quant_params(model: torch.nn.Module, requires_grad: bool = False):
    """
    Context manager to temporarily transfer the quantization parameters to buffer
    :param connected_graph: Connected graph associated with the model.
    :param model (torch.nn.Module): The model containing the modules to be modified.
    :param requires_grad (bool, optional): If the parameter requires gradient. See
        :ref:`locally-disable-grad-doc` for more details. Default: `False`.
    """
    with contextlib.ExitStack() as stack:
        model_parameters = []
        for _, module in model.named_modules():
            if not isinstance(module, MinMaxQuantizer):
                continue
            min_val = getattr(module, 'min').data.requires_grad_(requires_grad)
            max_val = getattr(module, 'max').data.requires_grad_(requires_grad)
            stack.enter_context(patch_attr(module, 'min', min_val))
            stack.enter_context(patch_attr(module, 'max', max_val))
            stack.enter_context(patch_attr(module, '_parameters', {}))
            model_parameters += [min_val, max_val]
        yield model_parameters

def shallow_copy(dict_like):
    """
    Create a shallow copy for dict-like objects with variables
    """
    copy = dict_like.__new__(type(dict_like))
    copy.update(dict_like.items())
    if hasattr(dict_like, '__dict__'):
        copy.__dict__.update(dict_like.__dict__)

    return copy
