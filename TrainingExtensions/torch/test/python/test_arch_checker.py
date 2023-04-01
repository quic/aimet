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
from aimet_torch import utils
from aimet_torch.arch_checker.arch_checker import ArchChecker

class Model(torch.nn.Module):
    """
    Model that uses functional modules instead of nn.Modules.
    Expects input of shape (1, 3, 32, 32)
    """
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.relu2 = torch.nn.ReLU()

        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 48, kernel_size=2, stride=2, padding=2, bias=False)

        self.conv4 = torch.nn.Conv2d(48, 20, 3)
        self.bn3 = torch.nn.BatchNorm2d(20)
        self.bn4 = torch.nn.BatchNorm2d(20)

        self.fc1 = torch.nn.Linear(1280, 10)
        self.prelu = torch.nn.PReLU()
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        # Regular case - conv followed by bn
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Non-linearity between conv and bn, not a candidate for fold
        x = self.conv2(x)
        x = self.relu2(x)

        # Case where BN can fold into an immediate downstream conv
        x = self.bn2(x)
        x = self.conv3(x)

        # No fold if there is a split between conv and BN
        x = self.conv4(x)
        bn1_out = self.bn3(x)
        bn2_out = self.bn4(x)

        x = bn1_out + bn2_out

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.prelu(x)
        x = self.silu(x)
        return x

class TestArchChecker():
    """ Class for testing arch (architechture) checker. """
    model = Model()
    dummy_input = utils.create_rand_tensors_given_shapes((2, 10, 64, 64), utils.get_device(model))

    def test_check_arch(self):
        """ Test check_arch function with self defined model."""
        arch_checker_report = ArchChecker.check_model_arch(self.model, self.dummy_input)
        # Node check unit test
        # Model.conv1 has input channel = 3, should fail _check_conv_channel_32_base and
        # _check_conv_channel_larger_than_32
        assert "_check_conv_channel_32_base" in arch_checker_report['Model.conv1'].failed_checks
        assert "_check_conv_channel_larger_than_32" in arch_checker_report['Model.conv1'].failed_checks

        # Model.conv2 should pass all the checks. No return
        assert 'Model.conv2' not in arch_checker_report

        # Model.conv3 has output channel = 48. should fail _check_conv_channel_32_base
        assert "_check_conv_channel_32_base" in arch_checker_report['Model.conv3'].failed_checks

        # prelu and silu should not pass not prelu check.
        assert "_activation_checks" in arch_checker_report['Model.prelu'].failed_checks
        assert "_activation_checks" in arch_checker_report['Model.silu'].failed_checks

        # relu should pass all checks
        assert "Model.relu1" not in arch_checker_report
        assert "Model.relu2" not in arch_checker_report

        # Pattern check unit test
        # bn1 can be folded into conv1
        assert "_check_batch_norm_fold" not in arch_checker_report

        # bn2 can be folded into conv3
        assert "_check_batch_norm_fold" not in arch_checker_report

        # bn3 and bn4 has a split between conv4, can not be folded
        assert "_check_batch_norm_fold" in arch_checker_report['Model.bn3'].failed_checks
        assert "_check_batch_norm_fold" in arch_checker_report['Model.bn4'].failed_checks

    def test_add_node_check(self):
        """
        Test add_check function is arch_checker. Add a test that will always fail: pass if relu is
        conv2d. The added check will always fail to return a failure record.
        """
        def _temp_check_relu_is_conv2d(node)-> bool:
            """ Temp check pass if relu is conv2d. This should always fail. """
            if not isinstance(node, torch.nn.modules.conv.Conv2d):
                return False
            return True

        ArchChecker.add_node_check(torch.nn.ReLU, _temp_check_relu_is_conv2d)

        arch_checker_report = ArchChecker.check_model_arch(self.model, self.dummy_input)

        # 'relu1'node is ReLU not Conv2d, so failed the _relu_is_Conv2d test.
        assert _temp_check_relu_is_conv2d.__name__ in arch_checker_report['Model.relu1'].failed_checks
        assert "_activation_checks" not in arch_checker_report['Model.relu1'].failed_checks

    def test_add_pattern_check(self):
        """
        Test add_check function is arch_checker. Add a test that will always fail: pass if relu is
        conv2d. The added check will always fail to return a failure record.
        """
        def _temp_check_get_all_bns(connected_graph):
            """ Temp check pass if relu is conv2d. This should always fail. """
            _bn_linear_optypes = ['BatchNormalization', 'BatchNorm3d']
            bn_ops = [op for op in connected_graph.get_all_ops().values() if op.type in _bn_linear_optypes]
            return bn_ops

        ArchChecker.add_pattern_check(_temp_check_get_all_bns)

        arch_checker_report = ArchChecker.check_model_arch(self.model, self.dummy_input)

        # all bns should be listed
        assert _temp_check_get_all_bns.__name__ in arch_checker_report['Model.bn1'].failed_checks
        assert _temp_check_get_all_bns.__name__ in arch_checker_report['Model.bn2'].failed_checks
        assert _temp_check_get_all_bns.__name__ in arch_checker_report['Model.bn3'].failed_checks
        assert _temp_check_get_all_bns.__name__ in arch_checker_report['Model.bn4'].failed_checks
