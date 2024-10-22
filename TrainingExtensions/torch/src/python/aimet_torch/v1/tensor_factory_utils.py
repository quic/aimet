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
"""
Tensor factory utility method
"""
import functools
import torch


@functools.lru_cache
def constant_tensor_factory(data,
                            *,
                            dtype=None,
                            device=None,
                            pin_memory=False) -> torch.Tensor:
    """
    Factory function that returns a cached constant scalar tensor with given data.
    This function is intentionally aligned with ``torch.tensor`` API except that
    it doesn't take ``requires_grad`` parameter.

    NOTE: The returned tensor is strictly read-only. However, currently there is no way
          one can fundamentally prevent performing in-place oeprations to the returned
          tensors. It is up to the callers to make sure the returned tensors are not
          modified in-place in any cases.

    :param data: Data for the return tensor.
    :param dtype: Data type of the returned tensor.
    :param device: Device of the returned tensor.
    :param pin_memory: Whether to allocate the tensor to pinned memory. Valid only for CPU tensors.
    :return: Constant read-only tensor
    """
    if not isinstance(data, (float, int)):
        raise ValueError(f"Expected data to be an instance of float or int. Got {type(data)}).")

    return torch.tensor(data,
                        dtype=dtype,
                        device=device,
                        requires_grad=False,
                        pin_memory=pin_memory)


def constant_like(data, tensor):
    """
    Factory function that returns a cached constant scalar tensor with the same
    dtype and device as the given tensor. If the input tensor is allocated in
    the pinned memory, the returned tensor will be also allocated to the pinned memory.

    :param data: Data for the return tensor.
    :param tensor: Tensor which the returned tensors will copy properties from.
    :return: Constant read-only tensor
    """
    return constant_tensor_factory(data,
                                   dtype=tensor.dtype,
                                   device=tensor.device,
                                   pin_memory=tensor.is_pinned())
