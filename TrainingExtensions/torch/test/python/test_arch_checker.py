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
from collections import OrderedDict
import torch
import torchvision
import os

from aimet_torch import utils
from aimet_torch.meta.operation import Op

from aimet_torch.arch_checker.arch_checker import ArchChecker
from aimet_torch.arch_checker.arch_checker_rules import TorchActivations
from aimet_torch.arch_checker.arch_checker_utils import _get_pd_dataframe
from aimet_torch.arch_checker.constants import ArchCheckerReportConstants as report_const

class Model_inter_pad_with_BN(torch.nn.Module):
    """
    Conv -> Activation -> BN -> Conv
    Model for testing intermediate padding check.
    Expects input of shape (batch_size, 3, 32, 32)
    4 cases are included:
    1) conv1(pad) conv2(pad) -> failed
    2) conv1      conv2(pad) -> pass
    3) conv1(pad) conv2      -> pass
    4) conv1      conv2      -> pass
    """
    def __init__(self):
        super(Model_inter_pad_with_BN, self).__init__()
        self.except_shape = (2, 3, 32, 32)

        # conv2 has intermediate paddings
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=2, bias=False)

        # conv4 has no intermediate paddings since conv3 has no paddings
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0, bias=False)
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=2, bias=False)

        # conv5 has intermediate paddings when consider (conv3, conv4, conv5) 
        # conv6 has no intermediate paddings
        self.conv5 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=2, bias=False)
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.conv6 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=0, bias=False)

        # conv7 has no intermediate paddings
        self.conv7 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=0, bias=False)
        self.relu4 = torch.nn.ReLU()
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.conv8 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)

        x = self.conv3(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.conv4(x)

        x = self.conv5(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.conv6(x)

        x = self.conv7(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.conv8(x)

        return x

class Model_inter_pad_without_BN(torch.nn.Module):
    """
    Conv -> Activation -> Conv
    Model for testing intermediate padding check.
    Expects input of shape (batch_size, 3, 32, 32)
    4 cases are included:
    1) conv1(pad) conv2(pad) -> failed
    2) conv1      conv2(pad) -> pass
    3) conv1(pad) conv2      -> pass
    4) conv1      conv2      -> pass
    """
    def __init__(self):
        super(Model_inter_pad_without_BN, self).__init__()
        self.except_shape = (2, 3, 32, 32)

        # conv2 has intermediate paddings
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=2, bias=False)

        # conv4 has intermediate paddings consider (conv1, conv2, conv3, conv4)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0, bias=False)
        self.relu2 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=2, bias=False)
        
        # PReLU is not a supported activation, stop examining the nodes after for (conv1, relu1, conv2) pattern.
        self.prelu = torch.nn.PReLU()
        # conv6 has no intermediate paddings
        self.conv5 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=2, bias=False)
        self.relu3 = torch.nn.ReLU()
        self.conv6 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=0, bias=False)

        # conv7, conv8 has no intermediate paddings
        self.conv7 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=0, bias=False)
        self.relu4 = torch.nn.ReLU()
        self.conv8 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x = self.conv3(x)
        x = self.relu2(x)
        x = self.conv4(x)

        x = self.prelu(x)
        x = self.conv5(x)
        x = self.relu3(x)
        x = self.conv6(x)

        x = self.conv7(x)
        x = self.relu4(x)
        x = self.conv8(x)

        return x

class Model_inter_pad_act_type(torch.nn.Module):
    """
    Conv -> Activation -> Conv
    Model for testing intermediate padding check.
    Expects input of shape (batch_size, 3, 32, 32)
    Support type: ("Relu", "Tanh", "HardSwish")
    """
    def __init__(self):
        super(Model_inter_pad_act_type, self).__init__()
        self.except_shape = (2, 3, 32, 32)

        # relu is supported type. conv2 has intermediate paddings
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=2, bias=False)

        # tanh is supported type. conv4 has intermediate paddings
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.tanh = torch.nn.Tanh()
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=2, bias=False)

        # hardswich is supported type. conv6 has intermediate paddings
        self.conv5 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=2, bias=False)
        self.hardswich = torch.nn.Hardswish()
        self.conv6 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=2, bias=False)

        # prelu is not supported type. conv8 has intermediate paddings
        self.conv7 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=2, bias=False)
        self.prelu = torch.nn.PReLU()
        self.conv8 = torch.nn.Conv2d(32, 32, kernel_size=2, padding=2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.prelu(x) # Break point for variable length

        x = self.conv3(x)
        x = self.tanh(x)
        x = self.conv4(x)
        x = self.prelu(x) # Break point for variable length

        x = self.conv5(x)
        x = self.hardswich(x)
        x = self.conv6(x)
        x = self.prelu(x) # Break point for variable length

        x = self.conv7(x)
        x = self.prelu(x)
        x = self.conv8(x)

        return x

class Model(torch.nn.Module):
    """
    Model for testing general arch_checker.
    Expects input of shape (batch_size, 3, 32, 32)
    """
    def __init__(self):
        super(Model, self).__init__()
        self.except_shape = (2, 3, 32, 32)

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.relu2 = torch.nn.ReLU()

        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 48, kernel_size=2, stride=2, padding=2, bias=False)

        self.conv4 = torch.nn.Conv2d(48, 20, 3)
        self.bn3 = torch.nn.BatchNorm2d(20)
        self.bn4 = torch.nn.BatchNorm2d(20)

        self.fc1 = torch.nn.Linear(320, 10)
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

class FPN_model(torch.nn.Module):
    def __init__(self) -> None:
        super(FPN_model, self).__init__()
        self.dummy_input = OrderedDict()
        self.dummy_input['feat0'] = torch.rand(4, 10, 64, 64)
        self.dummy_input['feat2'] = torch.rand(4, 20, 16, 16)
        self.dummy_input['feat3'] = torch.rand(4, 30, 8, 8)

        self.fpn = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        self.conv1 = torch.nn.Conv2d(5, 5, 3, stride = 8, padding = 1 )
        self.conv2 = torch.nn.Conv2d(5, 5, 3, stride = 2, padding = 1 )
        self.conv3 = torch.nn.Conv2d(5, 5, 1)
        self.conv4 = torch.nn.Conv2d(10, 5, 1)
        self.conv5 = torch.nn.Conv2d(10, 5, 1)

        self.bn1 = torch.nn.BatchNorm2d(10)
        self.bn2 = torch.nn.BatchNorm2d(25)

    def forward(self, x):
        fpn_output = self.fpn(x)
        out2 = self.conv2(fpn_output['feat2'])
        out3 = self.conv3(fpn_output['feat3'])
        concate = torch.cat((out2, out3), 1)
        x = self.bn1(concate)

        out0 = self.conv1(fpn_output['feat0'])
        out4 = self.conv4(x)
        out5 = self.conv5(x)
        concate = torch.cat((x, out0, out4, out5), 1)
        x = self.bn2(concate)

        return x

class TestArchChecker():
    """ Class for testing arch (architechture) checker. """
    model = Model()
    dummy_input = utils.create_rand_tensors_given_shapes(model.except_shape, utils.get_device(model))

    def test_check_conv_conv_cat_bn(self):
        model = FPN_model()
        dummy_input = model.dummy_input
        ArchChecker.check_model_arch(model, dummy_input)

        assert "_check_foldable_bn_with_split" in ArchChecker._arch_checker_report.raw_report["(FPN_model.conv2, FPN_model.conv3)-> Concat_44-> FPN_model.bn1"].failed_checks
        assert "_check_foldable_bn_with_split" in ArchChecker._arch_checker_report.raw_report["(FPN_model.bn1, FPN_model.conv1, FPN_model.conv4, FPN_model.conv5)-> Concat_51-> FPN_model.bn2"].failed_checks
        filepath = ArchChecker._arch_checker_report._get_write_path(".html")
        if os.path.exists(filepath):
            os.remove(filepath)

        ArchChecker._arch_checker_report.reset_raw_report()

    def test_intermediate_padding(self):
        # Test sequence: Conv -> Activation -> BN -> Conv
        model = Model_inter_pad_with_BN()
        ArchChecker.check_model_arch(model, self.dummy_input)
        arch_checker_report = ArchChecker._arch_checker_report
        assert "_check_intermediate_padding" in arch_checker_report.raw_report["Model_inter_pad_with_BN.conv2"].failed_checks
        assert "_check_intermediate_padding" in arch_checker_report.raw_report["Model_inter_pad_with_BN.conv5"].failed_checks
        assert "Model_inter_pad_with_BN.conv4" not in arch_checker_report.raw_report
        assert "Model_inter_pad_with_BN.conv6" not in arch_checker_report.raw_report
        assert "Model_inter_pad_with_BN.conv8" not in arch_checker_report.raw_report
        arch_checker_report.reset_raw_report()

        # Test sequence: Conv -> Activation -> Conv
        model = Model_inter_pad_without_BN()
        ArchChecker.check_model_arch(model, self.dummy_input)
        arch_checker_report = ArchChecker._arch_checker_report
        assert "_check_intermediate_padding" in arch_checker_report.raw_report["Model_inter_pad_without_BN.conv2"].failed_checks
        assert "_check_intermediate_padding" in arch_checker_report.raw_report["Model_inter_pad_without_BN.conv4"].failed_checks
        assert "Model_inter_pad_without_BN.conv6" not in arch_checker_report.raw_report
        assert "Model_inter_pad_without_BN.conv8" not in arch_checker_report.raw_report
        arch_checker_report.reset_raw_report()

        model = Model_inter_pad_act_type()
        ArchChecker.check_model_arch(model, self.dummy_input)
        arch_checker_report = ArchChecker._arch_checker_report

        assert "_check_intermediate_padding" not in arch_checker_report.raw_report["Model_inter_pad_act_type.conv1"].failed_checks
        assert "_check_intermediate_padding" in arch_checker_report.raw_report["Model_inter_pad_act_type.conv2"].failed_checks
        assert "Model_inter_pad_act_type.conv3" not in arch_checker_report.raw_report
        assert "_check_intermediate_padding" in arch_checker_report.raw_report["Model_inter_pad_act_type.conv4"].failed_checks
        assert "Model_inter_pad_act_type.conv5" not in arch_checker_report.raw_report
        assert "_check_intermediate_padding" in arch_checker_report.raw_report["Model_inter_pad_act_type.conv6"].failed_checks
        assert "Model_inter_pad_act_type.conv7" not in arch_checker_report.raw_report
        assert "Model_inter_pad_act_type.conv8" not in arch_checker_report.raw_report

        arch_checker_report.reset_raw_report()

        filepath = ArchChecker._arch_checker_report._get_write_path(".html")
        if os.path.exists(filepath):
            os.remove(filepath)

    def test_arch_checker_report(self):
        """ Test exported functions in ArchCheckerReport Class. """
        def get_export_dict_from_df(dataframe):
            # -1 to remove header column.
            column_length = len(dataframe[report_const.DF_GRAPH_NODENAME])

            export_dict = {}
            for idx in range(column_length):
                module_name, issue, recomm = dataframe.loc[idx]
                if module_name not in export_dict:
                    export_dict[module_name] = {report_const.DF_ISSUE: {issue},
                                                report_const.DF_RECOMM: {recomm}}
                else:
                    export_dict[module_name][report_const.DF_ISSUE].update({issue})
                    export_dict[module_name][report_const.DF_RECOMM].update({recomm})
            return export_dict

        ArchChecker.check_model_arch(self.model, self.dummy_input)
        arch_checker_report = ArchChecker._arch_checker_report
    
        # Add undefined check results to raw result.
        test_op = Op(name="test_op", dotted_name="test_dotted_name", output_shape =None, 
                 is_anonymous=False, op_type="test_type", residing_module=None)
        unknown_check_name = "unknown_check"

        arch_checker_report.raw_report["Model.conv1"].add_failed_checks({unknown_check_name})
        arch_checker_report.update_raw_report(test_op, {unknown_check_name} ) 

        # Read from dataframe file.
        dataframe = _get_pd_dataframe(arch_checker_report._raw_report)
        export_dict = get_export_dict_from_df(dataframe)

        # unknown_check_name raises undefined message.
        assert report_const.UNDEFINED_ISSUE.format(unknown_check_name) in export_dict[test_op.dotted_name][report_const.DF_ISSUE]
        assert report_const.UNDEFINED_RECOMM.format(unknown_check_name) in export_dict[test_op.dotted_name][report_const.DF_RECOMM]

        assert report_const.UNDEFINED_ISSUE.format(unknown_check_name) in export_dict["Model.conv1"][report_const.DF_ISSUE]
        assert report_const.UNDEFINED_RECOMM.format(unknown_check_name) in export_dict["Model.conv1"][report_const.DF_RECOMM]

        # Test .html is exported
        test_export_path = arch_checker_report._get_write_path(".html")
        arch_checker_report.export_to_html()
        assert os.path.exists(test_export_path)
        os.remove(test_export_path)
        assert not os.path.exists(test_export_path)

    def test_check_arch(self):
        """ Test check_arch function with self defined model."""
        ArchChecker.check_model_arch(self.model, self.dummy_input)
        arch_checker_report = ArchChecker._arch_checker_report
        # Node check unit test
        # Model.conv1 has input channel = 3, should fail _check_conv_channel_32_base and
        # _check_conv_channel_larger_than_32
        assert "_check_conv_channel_32_base" in arch_checker_report.raw_report['Model.conv1'].failed_checks
        assert "_check_conv_channel_larger_than_32" in arch_checker_report.raw_report['Model.conv1'].failed_checks

        # Model.conv2 should pass all the checks. No return.
        assert 'Model.conv2' not in arch_checker_report.raw_report

        # Model.conv3 has output channel = 48. should fail _check_conv_channel_32_base
        assert "_check_conv_channel_32_base" in arch_checker_report.raw_report['Model.conv3'].failed_checks

        # prelu and silu should not pass not prelu check.
        assert "_activation_checks" in arch_checker_report.raw_report['Model.prelu'].failed_checks
        assert "_activation_checks" in arch_checker_report.raw_report['Model.silu'].failed_checks

        # relu should pass all checks
        assert "Model.relu1" not in arch_checker_report.raw_report
        assert "Model.relu2" not in arch_checker_report.raw_report

        # Pattern check unit test
        # bn1 can be folded into conv1
        assert "_check_batch_norm_fold" not in arch_checker_report.raw_report

        # bn2 can be folded into conv3
        assert "_check_batch_norm_fold" not in arch_checker_report.raw_report

        # bn3 and bn4 has a split between conv4, can not be folded
        assert "_check_batch_norm_fold" in arch_checker_report.raw_report['Model.bn3'].failed_checks
        assert "_check_batch_norm_fold" in arch_checker_report.raw_report['Model.bn4'].failed_checks
        arch_checker_report.reset_raw_report()

        filepath = ArchChecker._arch_checker_report._get_write_path(".html")
        if os.path.exists(filepath):
            os.remove(filepath)

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

        ArchChecker.check_model_arch(self.model, self.dummy_input)
        arch_checker_report = ArchChecker._arch_checker_report

        # Relu is TorchActivations. Should under TorchActivations checks.
        assert torch.nn.ReLU not in ArchChecker._node_check_dict

        # _temp_check_relu_is_conv2d subject to Relu(TorchActivations) same func.__name__.
        assert _temp_check_relu_is_conv2d.__name__ in [_check.__name__ for _check in ArchChecker._node_check_dict[TorchActivations]]

        # _temp_check_relu_is_conv2d subject to Relu(TorchActivations). prelu and swish should node should return True without being checked.
        assert _temp_check_relu_is_conv2d.__name__ not in arch_checker_report.raw_report['Model.prelu'].failed_checks
        assert _temp_check_relu_is_conv2d.__name__ not in arch_checker_report.raw_report['Model.silu'].failed_checks

        # 'relu1'node is ReLU not Conv2d, so failed the _relu_is_Conv2d test.
        assert _temp_check_relu_is_conv2d.__name__ in arch_checker_report.raw_report['Model.relu1'].failed_checks
        assert "_activation_checks" not in arch_checker_report.raw_report['Model.relu1'].failed_checks
        arch_checker_report.reset_raw_report()

        filepath = ArchChecker._arch_checker_report._get_write_path(".html")
        if os.path.exists(filepath):
            os.remove(filepath)

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

        ArchChecker.check_model_arch(self.model, self.dummy_input)
        arch_checker_report = ArchChecker._arch_checker_report

        # all bns should be listed
        assert _temp_check_get_all_bns.__name__ in arch_checker_report.raw_report['Model.bn1'].failed_checks
        assert _temp_check_get_all_bns.__name__ in arch_checker_report.raw_report['Model.bn2'].failed_checks
        assert _temp_check_get_all_bns.__name__ in arch_checker_report.raw_report['Model.bn3'].failed_checks
        assert _temp_check_get_all_bns.__name__ in arch_checker_report.raw_report['Model.bn4'].failed_checks
        arch_checker_report.reset_raw_report()

        filepath = ArchChecker._arch_checker_report._get_write_path(".html")
        if os.path.exists(filepath):
            os.remove(filepath)
