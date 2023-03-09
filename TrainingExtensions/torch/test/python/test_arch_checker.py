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

import torch
from aimet_torch.arch_checker.arch_checker import ArchChecker

class Model(torch.nn.Module):
    """
    Model that uses functional modules instead of nn.Modules.
    Expects input of shape (1, 3, 32, 32)
    """

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv3 = torch.nn.Conv2d(64, 48, kernel_size=2, stride=2, padding=2, bias=False)
        self.relu1 = torch.nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu1(x)
        return x

class TestArchChecker():
    """ Class for testing arch (architechture) checker. """
    model = Model()

    def test_check_arch(self):
        """ Test check_arch function with self defined model."""
        arch_checker_report = ArchChecker.check_arch(self.model, (1, 3, 224, 224))

        # Model.conv1 has input channel = 3, should fail _check_conv_channel_32_base and
        # _check_conv_channel_larger_than_32
        assert "_check_conv_channel_32_base" in arch_checker_report['Model.conv1']
        assert "_check_conv_channel_larger_than_32" in arch_checker_report['Model.conv1']

        # Model.conv2 should pass all the checks. No return.
        assert 'Model.conv2' not in arch_checker_report

        # Model.conv3 has output channel = 48. should fail _check_conv_channel_32_base.
        assert "_check_conv_channel_32_base" in arch_checker_report['Model.conv3']


    def test_add_check(self):
        """
        Test add_check function is arch_checker. Add a test that will always fail: pass if relu is
        conv2d. The added check will always fail to return a failure record.
        """
        def _temp_check_relu_is_conv2d(node)-> bool:
            """ Temp check pass if relu is conv2d. This should always fail. """
            if not isinstance(node, torch.nn.modules.conv.Conv2d):
                return False
            return True

        ArchChecker.add_check(torch.nn.ReLU, _temp_check_relu_is_conv2d)

        arch_checker_report = ArchChecker.check_arch(self.model, (1, 3, 224, 224))

        # 'relu1'node is ReLU not Conv2d, so failed the _relu_is_Conv2d test.
        assert _temp_check_relu_is_conv2d.__name__ in arch_checker_report['Model.relu1']
