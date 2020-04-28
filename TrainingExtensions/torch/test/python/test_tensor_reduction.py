# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

import unittest
import torch
from aimet_common.polyslice import PolySlice
from aimet_torch.winnow.winnow_utils import reduce_tensor


def tensor_contains(tensor, value):
    return (tensor == value).nonzero().numel() > 0


class TestTrainingExtensionsTensorReduction(unittest.TestCase):

    def test_tensor_reduction(self):
        shape = [3, 2, 4]
        tensor = torch.zeros(shape, dtype=torch.int8)
        view = tensor.reshape([-1])
        for i in range(tensor.numel()):
            view[i] = 101 + i

        reduct = PolySlice(dim=0, index=1)
        result = reduce_tensor(tensor, reduct)
        assert list(result.shape) == [2, 2, 4]

        assert tensor_contains(result, 101)
        assert tensor_contains(result, 108)
        assert not tensor_contains(result, 109)
        assert not tensor_contains(result, 116)
        assert tensor_contains(result, 117)
        assert tensor_contains(result, 124)

        reduct.set(dim=2, index=[0])
        reduct.add(dim=2, index=3)
        result = reduce_tensor(tensor, reduct)
        assert list(result.shape) == [2, 2, 2]

        assert not tensor_contains(result, 101)
        assert tensor_contains(result, 102)
        assert tensor_contains(result, 103)
        assert not tensor_contains(result, 104)

        assert not tensor_contains(result, 117)
        assert not tensor_contains(result, 121)
        assert tensor_contains(result, 122)
        assert tensor_contains(result, 123)
        assert not tensor_contains(result, 124)
