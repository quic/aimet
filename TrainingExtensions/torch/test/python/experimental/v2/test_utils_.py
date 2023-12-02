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
import pytest
import torch

from aimet_torch.experimental.v2.utils import reduce

@pytest.mark.parametrize('reduce_dim, target_shape', [
    # | reduce dim   | target shape |
    # | -------------|--------------|
    (   [0,1,2,3],     []          ),

    (   [0,1,2],       [6]         ),
    (   [0,1,2],       [1,6]       ),
    (   [0,1,2],       [1,1,6]     ),
    (   [0,1,2],       [1,1,1,6]   ),
    (   [0,1,3],       [5,1]       ),
    (   [0,1,3],       [1,5,1]     ),
    (   [0,1,3],       [1,1,5,1]   ),
    (   [0,2,3],       [4,1,1]     ),
    (   [0,2,3],       [1,4,1,1]   ),
    (   [1,2,3],       [3,1,1,1]   ),

    (   [0,1],         [5,6]       ),
    (   [0,1],         [1,5,6]     ),
    (   [0,1],         [1,1,5,6]   ),
    (   [0,2],         [4,1,6]     ),
    (   [0,2],         [1,4,1,6]   ),
    (   [1,2],         [3,1,1,6]   ),
    (   [0,3],         [4,5,1]     ),
    (   [0,3],         [1,4,5,1]   ),
    (   [1,3],         [3,1,5,1]   ),
    (   [2,3],         [3,4,1,1]   ),

    (   [0],           [4,5,6]     ),
    (   [0],           [1,4,5,6]   ),
    (   [1],           [3,1,5,6]   ),
    (   [2],           [3,4,1,6]   ),
    (   [3],           [3,4,5,1]   ),
])
def test_reduce(reduce_dim, target_shape):
    x = torch.arange(start=0, end=3*4*5*6).view(3,4,5,6)
    out = reduce(x, target_shape, torch.sum)
    expected = torch.sum(x, dim=reduce_dim, keepdim=True)
    assert list(out.shape) == list(target_shape)
    assert torch.allclose(out, expected)
