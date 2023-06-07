# /usr/bin/env python3.8
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession

from aimet_onnx.cross_layer_equalization import equalize_model
import test_models


class TestCLEAcceptance:
    """ Acceptance test for AIMET ONNX """
    def test_cle_mv2(self):
        """ Test for E2E quantization """
        np.random.seed(0)
        test_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        model = test_models.mobilenetv2()
        session = _build_session(model)
        output_before_cle = session.run(None, {'input': test_data})
        equalize_model(model)
        session = _build_session(model)
        output_after_cle = session.run(None, {'input': test_data})
        assert np.allclose(output_after_cle, output_before_cle, rtol=1e-2)


def _build_session(model):
    """
    Build and return onnxruntime inference session
    :param providers: providers to execute onnxruntime
    """
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    session = InferenceSession(
        path_or_bytes=model.model.SerializeToString(),
        sess_options=sess_options,
        providers=['CPUExecutionProvider'],
    )
    return session