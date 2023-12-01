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

""" Unit tests for Adaround Weights """
import json
from packaging import version
import numpy as np
import torch
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
import pytest

from aimet_onnx.adaround.adaround_weight import Adaround, AdaroundParameters
import models.models_for_tests as test_models

class TestAdaround:
    """
    AdaRound Weights Unit Test Cases
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="This unit-test is meant to be run on GPU")
    def test_apply_adaround(self):
        if version.parse(torch.__version__) >= version.parse("1.13"):
            np.random.seed(0)
            torch.manual_seed(0)
            model = test_models.single_residual_model()
            data_loader = dataloader()
            dummy_input = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
            sess = build_session(model)
            out_before_ada = sess.run(None, dummy_input)
            def callback(session, args):
                in_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
                session.run(None, in_tensor)

            params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=5, forward_fn=callback,
                                        forward_pass_callback_args=None)
            ada_rounded_model = Adaround.apply_adaround(model, params, './', 'dummy')
            sess = build_session(ada_rounded_model)
            out_after_ada = sess.run(None, dummy_input)
            assert not np.array_equal(out_before_ada[0], out_after_ada[0])

            with open('./dummy.encodings') as json_file:
                encoding_data = json.load(json_file)

            param_keys = list(encoding_data.keys())
            assert 'onnx::Conv_43' in param_keys


def dataloader():
    class DataLoader:
        """
        Example of a Dataloader which can be used for running AMPv2
        """
        def __init__(self, batch_size: int):
            """
            :param batch_size: batch size for data loader
            """
            self.batch_size = batch_size

        def __iter__(self):
            """Iterates over dataset"""
            dummy_input = np.random.rand(1, 3, 32, 32).astype(np.float32)
            yield dummy_input

        def __len__(self):
            return 4

    dummy_dataloader = DataLoader(batch_size=2)
    return dummy_dataloader

def build_session(model):
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