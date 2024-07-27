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

try:
    import deepspeed as ds
    gathered_parameters = ds.runtime.zero.GatheredParameters
except ImportError:
    def gathered_parameters(*args, **kwargs): # pylint: disable=unused-argument
        """ Dummy placeholder in case deepspeed doesn't exist """
        return contextlib.nullcontext()


_ds_ctx = {}

def _all_gather(module, _):
    ctx = gathered_parameters(module.parameters(recurse=False))
    ctx.__enter__()
    _ds_ctx[module] = ctx

def _release(module, *_):
    ctx = _ds_ctx.pop(module, None)
    if ctx:
        ctx.__exit__(None, None, None)

@contextlib.contextmanager
def _register_zero3_forward_hooks(model: torch.nn.Module):
    handles = []

    try:
        for module in model.modules():
            handle = module.register_forward_pre_hook(_all_gather)
            handles.append(handle)
            handle = module.register_forward_hook(_release)
            handles.append(handle)
        yield
    finally:
        for handle in handles:
            handle.remove()


def _shallow_copy(dict_like):
    """
    Create a shallow copy for dict-like objects with variables
    """
    copy = dict_like.__new__(type(dict_like))
    copy.update(dict_like.items())
    if hasattr(dict_like, '__dict__'):
        copy.__dict__.update(dict_like.__dict__)

    return copy


def _get_shape(tensor: torch.Tensor):
    return getattr(tensor, 'ds_shape', tensor.shape)
