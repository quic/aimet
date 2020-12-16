# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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
import unittest
import copy
import time

import torch
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.utils import AimetLogger
import aimet_torch.examples.mnist_torch_model as mnist_model

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


def forward_pass(model, args):
    torch.manual_seed(1)
    device = next(model.parameters()).device

    rand_input = torch.randn((10, 1, 28, 28)).to(device)
    model(rand_input)


class QuantizerCpuGpu(unittest.TestCase):

    @pytest.mark.cuda
    def test_and_compare_quantizer_no_fine_tuning_CPU_and_GPU(self):

        torch.manual_seed(1)
        torch.backends.cudnn.deterministic = True

        start_time = time.time()

        # create model on CPU
        model_cpu = mnist_model.Net().to('cpu')
        model_gpu = copy.deepcopy(model_cpu).to('cuda')
        cpu_sim_model = QuantizationSimModel(model_cpu, quant_scheme='tf', in_place=True,
                                             dummy_input=torch.rand(1, 1, 28, 28))
        # Quantize
        cpu_sim_model.compute_encodings(forward_pass, None)

        print("Encodings for cpu model calculated")
        print("Took {} secs".format(time.time() - start_time))
        start_time = time.time()

        # create model on GPU
        gpu_sim_model = QuantizationSimModel(model_gpu, quant_scheme='tf', in_place=True,
                                             dummy_input=torch.rand(1, 1, 28, 28).cuda())
        # Quantize
        gpu_sim_model.compute_encodings(forward_pass, None)

        print("Encodings for gpu model calculated")
        print("Took {} secs".format(time.time() - start_time))

        # check the encodings only min and max
        # Test that first and second are approximately (or not approximately)
        # equal by computing the difference, rounding to the given number of
        # decimal places (default 7), and comparing to zero. Note that these
        # methods round the values to the given number of decimal places
        # (i.e. like the round() function) and not significant digits
        # excluding fc1 since it is part of Matmul->Relu supergroup
        # can't use assertEqual for FC2, so using assertAlmostEquals for FC2
        self.assertAlmostEqual(model_gpu.conv1.output_quantizers[0].encoding.min,
                               model_cpu.conv1.output_quantizers[0].encoding.min, delta=0.001)
        self.assertAlmostEqual(model_gpu.conv1.output_quantizers[0].encoding.max,
                               model_cpu.conv1.output_quantizers[0].encoding.max, delta=0.001)

        self.assertAlmostEqual(model_gpu.conv2.output_quantizers[0].encoding.min,
                               model_cpu.conv2.output_quantizers[0].encoding.min, delta=0.001)
        self.assertAlmostEqual(model_gpu.conv2.output_quantizers[0].encoding.max,
                               model_cpu.conv2.output_quantizers[0].encoding.max, delta=0.001)

        self.assertAlmostEqual(model_gpu.fc2.output_quantizers[0].encoding.min,
                               model_cpu.fc2.output_quantizers[0].encoding.min, delta=0.001)
        self.assertAlmostEqual(model_gpu.fc2.output_quantizers[0].encoding.max,
                               model_cpu.fc2.output_quantizers[0].encoding.max, delta=0.001)

        gpu_sim_model.export("./data/", "quantizer_no_fine_tuning__GPU", (1, 1, 28, 28))
        cpu_sim_model.export("./data/", "quantizer_no_fine_tuning__CPU", (1, 1, 28, 28))

        self.assertEqual(torch.device('cuda:0'), next(model_gpu.parameters()).device)
        self.assertEqual(torch.device('cpu'), next(model_cpu.parameters()).device)

