# /usr/bin/env python3.8
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
import os
import torch
import numpy as np
from torchvision import models
from onnx import load_model
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme

WORKING_DIR = '/tmp/quantsim'


class TestQuantizeAcceptance:
    """ Acceptance test for AIMET ONNX """
    def test_quantize_resnet18(self):
        """ Test for E2E quantization """
        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR)

        model = models.resnet18(pretrained=False)

        torch.onnx.export(model, torch.rand(1, 3, 224, 224), os.path.join(WORKING_DIR, 'resnet18.onnx'),
                          training=torch.onnx.TrainingMode.PRESERVE,
                          input_names=['input'], output_names=['output'])

        onnx_model = load_model(os.path.join(WORKING_DIR, 'resnet18.onnx'))
        sim = QuantizationSimModel(onnx_model, quant_scheme=QuantScheme.post_training_tf, default_param_bw=8,
                                   default_activation_bw=8)

        sim.compute_encodings(forward_pass_function, None)

        forward_pass_function(sim.session, None)

        for name, qc_op in sim.get_qc_quantize_op().items():
            assert qc_op.tensor_quantizer.isEncodingValid is True


def forward_pass_function(session, args=None):
    """
    Dummy forward pass function

    :param session: onnx runtime session
    :param args: arguments for forward pass function
    """
    session.run(None, {'input': np.random.rand(1, 3, 224, 224).astype(np.float32)})

