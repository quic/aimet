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

import numpy as np
import pytest
import torch
from onnx import load_model
from torchvision import models

from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_torch.quantsim import QuantizationSimModel as PtQuantizationSimModel
from torch_utils import _get_cifar10_data_loaders, train_cifar10

WORKING_DIR = '/tmp/quantsim'

image_size = 32
batch_size = 64
num_workers = 4


def model_eval_onnx(session, val_loader):
    """
    :param model: model to be evaluated
    :param early_stopping_iterations: if None, data loader will iterate over entire validation data
    :return: top_1_accuracy on validation data
    """

    corr = 0
    total = 0
    for (i, batch) in enumerate(val_loader):
        x, y = batch[0].numpy(), batch[1].numpy()
        in_tensor = {'input': x}
        out = session.run(None, in_tensor)[0]
        corr += np.sum(np.argmax(out, axis=1) == y)
        total += x.shape[0]
    print(f'Accuracy: {corr / total}')
    return corr / total


class TestQuantizeAcceptance:
    """ Acceptance test for AIMET ONNX """

    def test_quantize_resnet18(self):
        """ Test for E2E quantization """
        np.random.seed(0)
        torch.manual_seed(0)

        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR)

        inputs = np.random.rand(1, 3, 224, 224).astype(np.float32)

        model = models.resnet18(pretrained=False)

        # model = model.to(torch.device('cuda'))

        # layers_to_ignore = [model.conv1]
        sim_pt = PtQuantizationSimModel(model, quant_scheme=QuantScheme.post_training_tf, default_param_bw=8,
                                        default_output_bw=8, dummy_input=torch.as_tensor(inputs))

        def dummy_forward_pass_pt(model, _):
            model.eval()
            model(torch.as_tensor(inputs))

        # If 'iterations'set to None, will iterate over all the validation data
        sim_pt.compute_encodings(dummy_forward_pass_pt, forward_pass_callback_args=None)

        torch.onnx.export(model, torch.as_tensor(inputs), os.path.join(WORKING_DIR, 'resnet18.onnx'),
                          training=torch.onnx.TrainingMode.PRESERVE,
                          input_names=['input'], output_names=['output'])

        onnx_model = load_model(os.path.join(WORKING_DIR, 'resnet18.onnx'))
        sim = QuantizationSimModel(onnx_model, quant_scheme=QuantScheme.post_training_tf, default_param_bw=8,
                                   default_activation_bw=8)

        def dummy_forward_pass_onnx(session, _):
            in_tensor = {'input': inputs}
            session.run(None, in_tensor)

        sim.compute_encodings(dummy_forward_pass_onnx, None)

        pytorch_forward_pass_output = model(torch.as_tensor(inputs))
        onnx_forward_pass_output = sim.session.run(None, {'input': inputs})
        assert np.all(np.abs(
            np.asarray(pytorch_forward_pass_output.detach().numpy()) - np.asarray(onnx_forward_pass_output)) < 0.05)

    @pytest.mark.cuda
    def test_quantized_accuracy(self):
        np.random.seed(0)
        torch.manual_seed(0)
        model = models.resnet18(pretrained=False, num_classes=10)
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            model.to(device)

        train_cifar10(model, 2)
        train_loader, val_loader = _get_cifar10_data_loaders()

        torch.onnx.export(model, torch.rand(batch_size, 3, 32, 32).cuda(), os.path.join(WORKING_DIR, 'resnet18.onnx'),
                          training=torch.onnx.TrainingMode.PRESERVE,
                          input_names=['input'], output_names=['output'])

        onnx_model = load_model(os.path.join(WORKING_DIR, 'resnet18.onnx'))

        sim = QuantizationSimModel(onnx_model, quant_scheme=QuantScheme.post_training_tf, default_param_bw=8,
                                   default_activation_bw=8, use_cuda=True)

        def onnx_callback(session, iters):
            for i, batch in enumerate(train_loader):
                x = batch[0].detach().cpu().numpy()
                in_tensor = {'input': x}
                session.run(None, in_tensor)
                if i >= iters:
                    break

        sim.compute_encodings(onnx_callback, 10)

        onnx_qs_acc = model_eval_onnx(sim.session, val_loader)

        assert onnx_qs_acc > 0.5
