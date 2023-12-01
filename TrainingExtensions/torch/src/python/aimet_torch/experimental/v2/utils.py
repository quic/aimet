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
# pylint: disable=redefined-builtin
""" Common utility functions """
from typing import Callable, Tuple
import functools
import itertools

import torch


def _is_expandable(src_shape: Tuple[int, ...],
                   target_shape: Tuple[int, ...]) -> bool:
    """
    Returns true if source shape can be expanded as target shape
    """
    if len(src_shape) > len(target_shape):
        return False

    for src_dim, dst_dim in zip(src_shape[::-1], target_shape[::-1]):
        if src_dim not in (1, dst_dim):
            return False

    return True


def _is_reducible(src_shape: Tuple[int, ...],
                  target_shape: Tuple[int, ...]) -> bool:
    """
    Returns true if source shape can be reduced as target shape
    """
    return _is_expandable(target_shape, src_shape)


def reduce(input: torch.Tensor, shape: Tuple[int, ...], reduce_op: Callable):
    """
    Reduce input into given shape.

    :param input: Input to reduce
    :param shape: Shape of the reduced output
    :param reduce_op: Reduce operation
    """
    if not _is_reducible(input.shape, shape):
        raise RuntimeError(
            f"Input of shape {list(input.shape)} can't be reduced to shape {list(shape)}"
        )

    padded_shape = (
        *itertools.repeat(1, len(input.shape) - len(shape)),
        *shape
    )
    reduce_dims = tuple(axis for axis, dim in enumerate(padded_shape) if dim == 1)
    other_dims = tuple(axis for axis, dim in enumerate(padded_shape) if dim > 1)
    permute_dims = reduce_dims + other_dims

    return reduce_op(input.permute(permute_dims).reshape(-1, *shape), dim=0, keepdim=False)


class _ContextManager:
    def __init__(self, action: Callable[[], None], cleanup: Callable[[], None]):
        self._action = action
        self._cleanup = cleanup

    def __enter__(self):
        self._action()
        return self

    def __exit__(self, *_):
        self._cleanup()

    def __call__(self, fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapper


def patch_attr(obj, attr_name, new_attr)-> _ContextManager:
    """
    Temporarily overwrite object attribute
    """
    old_attr = getattr(obj, attr_name)
    action = lambda: setattr(obj, attr_name, new_attr)
    cleanup = lambda: setattr(obj, attr_name, old_attr)
    return _ContextManager(action, cleanup)


def patch_param(module: torch.nn.Module, param_name: str, new_param: torch.Tensor) -> _ContextManager:
    """
    Temporarily substitute the reference to the a parameter with the quantized parameter.
    Under the scope of this function, ``getattr(module, param_name)`` will return
    ``new_param`` instead of the original parameter.

    :param module: Module that owns the parameter
    :param param_name: Name of the parameter
    :param new_param: New parameter to replace the original parameter
    """
    original_param = getattr(module, param_name)
    if original_param is not None:
        assert original_param.shape == new_param.shape

    # Modify module.__dict__.
    # module.__dict__ is the primary lookup table which has higher priority than __getattr__ method.
    # Once we overwrite module.__dict__[param_name] with quantized_params,
    # getattr(module, param_name) will return module.__dict__[param_name] directly
    # without falling back to torch.nn.Module's __getattr__ method which returns
    # the original parameter stored in module._parameters.
    action = lambda: module.__dict__.update({param_name: new_param})

    if param_name in module.__dict__:
        # Some non-standard modules (e.g. replicas of torch.nn.DataParallel) store their parameters
        # directly to module.__dict__. In that case, the cleanup function should restore the dict
        # so that module.__dict__[param_name] points back to the original parameter again.
        assert module.__dict__[param_name] is original_param
        cleanup = lambda: module.__dict__.update({param_name: original_param})
    else:
        assert module._parameters[param_name] is original_param # pylint: disable=protected-access
        cleanup = lambda: module.__dict__.pop(param_name)


    return _ContextManager(action, cleanup)


class _StraightThroughEstimator(torch.autograd.Function): # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, op, *args, **kwargs): # pylint: disable=arguments-differ
        return op(*args, **kwargs)

    @staticmethod
    def backward(ctx, *grad):
        return (None, *grad)


def ste_round(*args, **kwargs):
    """
    Applies straight-through rounding
    """
    return _StraightThroughEstimator.apply(torch.round, *args, **kwargs)
