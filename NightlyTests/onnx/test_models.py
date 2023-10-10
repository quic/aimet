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
import torch
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from onnx import load_model

from torchvision.models import MobileNetV2, mobilenet_v3_large

def mobilenetv2():
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model = MobileNetV2().eval()

    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_mobilenetv2.onnx",
                      training=torch.onnx.TrainingMode.PRESERVE,
                      export_params=True,
                      do_constant_folding=False,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={
                          'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'},
                      }
                      )
    model = ONNXModel(load_model('./model_mobilenetv2.onnx'))
    return model

def mobilenetv3_large_model():
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model = mobilenet_v3_large().eval()

    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "./model_mobilenetv3.onnx",
                      training=torch.onnx.TrainingMode.PRESERVE,
                      export_params=True,
                      do_constant_folding=False,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={
                          'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'},
                      }
                      )
    model = ONNXModel(load_model('./model_mobilenetv3.onnx'))
    return model