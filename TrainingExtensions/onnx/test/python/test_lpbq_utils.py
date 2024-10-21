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

import numpy as np

from aimet_onnx import lpbq_utils


class TestLPBQUtils:

    def test_get_per_group_scale_factor(self):
        bw = 4
        scale = np.asarray(
            [
                [100.2, 32.1, .001, .4],
                [23.1, 22.1, 10., 9.]
            ], np.float32
        )
        per_group_scale = lpbq_utils._get_per_group_scale_factor(scale, (1, 4), bw)
        expected_scale = np.asarray([100.2, 23.1]) / 2 ** bw
        assert np.allclose(per_group_scale, expected_scale.reshape(per_group_scale.shape))

        per_group_scale = lpbq_utils._get_per_group_scale_factor(scale, (2, 1), bw)
        expected_scale = np.asarray([100.2, 32.1, 10., 9.]) / 2 ** bw
        assert np.allclose(per_group_scale, expected_scale.reshape(per_group_scale.shape))

    def test_per_group_int_scales(self):
        scale = np.asarray([
            [16, 1.6, .16],
            [.12, 1.111, .033]
        ])
        grouping = (2, 1)
        expected_int_scale = [16, 16, 16, 1, 11, 3]
        int_scale, scale_factor = lpbq_utils.grouped_dynamic_quantize(scale, grouping, 4)
        assert scale_factor.flatten().tolist() == [1, .1, .01]
        assert int_scale.flatten().tolist() == expected_int_scale

    def test_compress_encoding_scales(self):
        scale_bw = 8
        scale = np.asarray(
            [
                [25.6, 11.111],
                [256., 25.555]
            ], np.float32
        )
        offset = np.asarray(
            [
                [-128, -128],
                [-128, -128]
            ]
        )

        encodings = lpbq_utils.scale_offset_arrays_to_encodings(scale, offset, 4)
        lpbq_encodings = lpbq_utils.compress_encoding_scales(encodings, scale.shape, (1, 2), scale_bw)
        lpbq_scale, lpbq_offset = lpbq_utils.encodings_to_scale_offset_arrays(lpbq_encodings, scale.shape)

        expected_lpbq_scale = np.asarray(
            [
                [25.6, 11.1],
                [256., 26.],
            ], np.float32
        )

        assert np.allclose(lpbq_scale, expected_lpbq_scale)
        assert np.allclose(lpbq_offset, offset)

