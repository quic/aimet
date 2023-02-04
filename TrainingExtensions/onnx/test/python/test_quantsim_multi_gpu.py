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
import json
import os
import torch
import numpy as np
from onnx import load_model
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.qc_quantize_op import OpMode
from aimet_torch.quantsim import QuantizationSimModel as PtQuantizationSimModel
from aimet_torch.examples.test_models import SingleResidual
from test_models import build_dummy_model


class DummyModel(SingleResidual):
    """
    Model
    """
    def __init__(self):
        super().__init__()
        # change padding size to 0, onnxruntime only support input size is the factor of output size for pooling
        self.conv4 = torch.nn.Conv2d(32, 8, kernel_size=2, stride=2, padding=0, bias=True)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        del self.bn1
        del self.bn2

    def forward(self, inputs):
        x = self.conv1(inputs)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Save the output of MaxPool as residual.
        residual = x

        x = self.conv2(x)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        # x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        residual = self.conv4(residual)
        residual = self.ada(residual)
        x += residual
        x = self.relu3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class TestQuantSim:
    """Tests for QuantizationSimModel"""


    def test_quantsim_device_1(self):
        """Test to compare encodings with PT"""
        if not os.path.exists('/tmp'):
            os.mkdir('/tmp')

        def pytorch_callback(model, inputs):
            model.eval()
            model(torch.as_tensor(inputs))

        def onnx_callback(session, inputs):
            in_tensor = {'input': inputs}
            session.run(None, in_tensor)

        inputs = np.random.rand(128, 3, 32, 32).astype(np.float32)
        inputs_torch = torch.as_tensor(inputs)
        model = DummyModel()
        model.eval()

        torch.onnx.export(model, torch.as_tensor(inputs), '/tmp/dummy_model.onnx', training=torch.onnx.TrainingMode.PRESERVE,
                          input_names=['input'], output_names=['output'])
        import time
        pt_start = time.time()


        onnx_model = load_model('/tmp/dummy_model.onnx')

        onnx_sim = QuantizationSimModel(onnx_model, use_cuda=True, device=1)
        for idx in range(1000):
            onnx_callback(onnx_sim.session, inputs)

        activation_encodings_map = {'12': '9', '15': '10', '21': '12', '24': '13', '27': '14', '30': '15',
                                    '34': '17', '38': '19', 't.1': 'input'}
        ort_start = time.time()
        for idx in range(100):
            onnx_sim.compute_encodings(onnx_callback, inputs)
        ort_time = time.time() - ort_start


        onnx_sim.export('/tmp', 'onnx_sim')

        with open('/tmp/pt_sim.encodings') as f:
            pt_encodings = json.load(f)
        with open('/tmp/onnx_sim.encodings') as f:
            onnx_encodings = json.load(f)

