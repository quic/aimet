# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""BatchNorm Reestimation"""

import itertools
from typing import Iterable, List, Callable, Any

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.modules.batchnorm import _BatchNorm
from aimet_torch.utils import in_eval_mode, in_train_mode
from aimet_common.utils import Handle

def _get_active_bn_modules(model: torch.nn.Module) -> Iterable[_BatchNorm]:
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            bn = module
            if bn.running_mean is not None and bn.running_var is not None:
                yield bn


def _for_each_module(modules: Iterable[torch.nn.Module],
                     action: Callable[[torch.nn.Module], Handle]) -> Handle:
    """
    Apply an undoable action to each module.

    :param modules: Modules to apply the action.
    :param action: Action to be applied to the modules.
    :returns: Handle that undos the applied action.
    """

    handles: List[Handle] = []

    def cleanup():
        for handle in handles:
            handle.remove()

    try:
        for module in modules:
            handle = action(module)
            assert isinstance(handle, Handle)
            handles.append(handle)
        return Handle(cleanup)
    except:
        cleanup()
        raise


def _reset_bn_stats(module: _BatchNorm) -> Handle:
    """
    Reset BN statistics to the initial values.

    :param module: BatchNorm module.
    :returns: Handle that restores the original BN statistics upon handle.remove().
    """
    orig_running_mean = module.running_mean.clone()
    orig_running_var = module.running_var.clone()
    orig_num_batches_tracked = module.num_batches_tracked.clone()

    def cleanup():
        module.running_mean.copy_(orig_running_mean)
        module.running_var.copy_(orig_running_var)
        module.num_batches_tracked.copy_(orig_num_batches_tracked)

    try:
        module.reset_running_stats()
        return Handle(cleanup)
    except:
        cleanup()
        raise


def _reset_momentum(module: _BatchNorm) -> Handle:
    """
    Set BN momentum to 1.0.

    :param module: BatchNorm module.
    :returns: Handle that restores the original BN momentum upon handle.remove().
    """
    momentum = module.momentum

    def cleanup():
        module.momentum = momentum

    try:
        module.momentum = 1.0
        return Handle(cleanup)
    except:
        cleanup()
        raise


DEFAULT_NUM_BATCHES = 100


def reestimate_bn_stats(model: torch.nn.Module,
                        dataloader: DataLoader,
                        num_batches: int = DEFAULT_NUM_BATCHES,
                        forward_fn: Callable[[torch.nn.Module, Any], Any] = None) -> Handle:
    """
    Reestimate BatchNorm statistics (running mean and var).

    :param model: Model to reestimate the BN stats.
    :param dataloader: Training dataset.
    :param num_batches: The number of batches to be used for reestimation.
    :param forward_fn: Optional adapter function that performs forward pass
                       given a model and a input batch yielded from the data loader.
    :returns: Handle that undos the effect of BN reestimation upon handle.remove().
    """
    forward_fn = forward_fn or (lambda model, data: model(data))
    bn_modules = tuple(_get_active_bn_modules(model))

    # Set all the layers to eval mode except batchnorm layers
    with in_eval_mode(model), in_train_mode(bn_modules), torch.no_grad():
        with _for_each_module(bn_modules, action=_reset_momentum):
            handle = _for_each_module(bn_modules, action=_reset_bn_stats)

            try:
                # Batchnorm statistics accumulation buffer
                buffer = {
                    bn: {"sum_mean": torch.zeros_like(bn.running_mean),
                         "sum_var":  torch.zeros_like(bn.running_var)}
                    for bn in bn_modules
                }

                num_batches = min(len(dataloader), num_batches)
                dataloader_slice = itertools.islice(dataloader, num_batches)

                for data in tqdm(dataloader_slice,
                                 total=num_batches,
                                 desc="batchnorm reestimation"):
                    forward_fn(model, data)

                    for bn in bn_modules:
                        buffer[bn]["sum_mean"] += bn.running_mean
                        buffer[bn]["sum_var"] += bn.running_var

                for bn in bn_modules:
                    sum_mean = buffer[bn]["sum_mean"]
                    sum_var = buffer[bn]["sum_var"]

                    # Override BN stats with the reestimated stats.
                    bn.running_mean.copy_(sum_mean / min(len(dataloader), num_batches))
                    bn.running_var.copy_(sum_var / min(len(dataloader), num_batches))

                return handle
            except:
                handle.remove()
                raise
