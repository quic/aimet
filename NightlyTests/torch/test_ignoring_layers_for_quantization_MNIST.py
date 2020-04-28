# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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
import torch.nn as nn

from aimet_torch import quantizer as q
from aimet_common.utils import AimetLogger
import aimet_torch.examples.mnist_torch_model as mnist_model

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class IgnoreLayers(unittest.TestCase):

    def test_quantizer_with_ignoring_layers(self):

        torch.cuda.empty_cache()

        net = mnist_model.Net()
        model = net.to(torch.device('cpu'))
        quantizer = q.Quantizer(model=model, use_cuda=False)

        layers_to_ignore = [net.conv1, net.fc2]
        quantizer.quantize_net(bw_params=8, bw_acts=8, run_model=mnist_model.evaluate, iterations=1,
                               layers_to_ignore=layers_to_ignore)
        self.assertTrue(isinstance(net.conv1, nn.Conv2d))
        self.assertFalse(isinstance(net.conv2, nn.Conv2d))
        self.assertTrue(isinstance(net.fc2, nn.Linear))

        print("Quantized Model", model)
