# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from aimet_torch import QuantizationSimModel


class TestDisableSupergroups:
    """
    TODO: As of QAIRT 2.37, following supergroups are not supported by HTP:
    1. Conv3d / ConvTranspose3d -> ...
    2. Depthwise Conv -> ...

    Disabling pattern matching for above two convolution cases in AIMET for short-term
    Issue #5597: Remove this test case when respective support is added in HTP and remove work-around.
    """

    def test_disable_conv3d_supergroup(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv3d(3, 3, 1)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv3d(3, 3, 1)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                x1 = self.relu1(x1)
                x2 = self.relu2(x2)
                return x1 + x2

        model = Model()
        x = torch.randn((1, 3, 24, 24, 24))
        sim = QuantizationSimModel(model, x, config_file="htp_v81")

        assert sim.model.conv1.output_quantizers[0]
        assert sim.model.relu1.output_quantizers[0]
        assert sim.model.conv2.output_quantizers[0]
        assert sim.model.relu2.output_quantizers[0]

    def test_disable_dynamic_conv3d_supergroup(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv3d(3, 3, 1)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv3d(3, 3, 1)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                x1 = self.relu1(x1)
                x2 = self.relu2(x2)
                return x1 + x2

        model = Model()
        x = torch.randn((1, 3, 24, 24, 24))
        sim = QuantizationSimModel(model, x, config_file="htp_v81")

        assert sim.model.conv1.output_quantizers[0]
        assert sim.model.relu1.output_quantizers[0]
        assert sim.model.conv2.output_quantizers[0]
        assert sim.model.relu2.output_quantizers[0]

    def test_disable_conv_transpose3d_supergroup(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.ConvTranspose3d(3, 3, 1)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.ConvTranspose3d(3, 3, 1)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                x1 = self.relu1(x1)
                x2 = self.relu2(x2)
                return x1 + x2

        model = Model()
        x = torch.randn((1, 3, 24, 24, 24))
        sim = QuantizationSimModel(model, x, config_file="htp_v81")

        assert sim.model.conv1.output_quantizers[0]
        assert sim.model.relu1.output_quantizers[0]
        assert sim.model.conv2.output_quantizers[0]
        assert sim.model.relu2.output_quantizers[0]

    def test_disable_depthwise_conv_supergroup(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 1, groups=3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(3, 6, 1, groups=3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.conv2(x)
                x = self.relu2(x)
                return x

        model = Model()
        x = torch.randn((1, 3, 24, 24))
        sim = QuantizationSimModel(model, x, config_file="htp_v81")

        assert sim.model.conv1.output_quantizers[0]
        assert sim.model.relu1.output_quantizers[0]
        assert sim.model.conv2.output_quantizers[0]
        assert sim.model.relu2.output_quantizers[0]

    def test_disable_depthwise_conv_transpose_supergroup(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.ConvTranspose2d(3, 3, 1, groups=3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.ConvTranspose2d(3, 6, 1, groups=3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.conv2(x)
                x = self.relu2(x)
                return x

        model = Model()
        x = torch.randn((1, 3, 24, 24))
        sim = QuantizationSimModel(model, x, config_file="htp_v81")

        assert sim.model.conv1.output_quantizers[0]
        assert sim.model.relu1.output_quantizers[0]
        assert sim.model.conv2.output_quantizers[0]
        assert sim.model.relu2.output_quantizers[0]

    def test_disable_depthwise_conv_supergroup_trivial_case(self):
        """
        When: in_channels == num_groups == 1
        Then: Conv should be treated as regular Conv even though in_channels == num_groups.
              This is a trivial case which can be interpreted as both regular and depthwise conv
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1, groups=1)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.ConvTranspose2d(1, 1, 1, groups=1)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.conv2(x)
                x = self.relu2(x)
                return x

        model = Model()
        x = torch.randn((1, 1, 24, 24))
        sim = QuantizationSimModel(model, x, config_file="htp_v81")

        assert not sim.model.conv1.output_quantizers[0]
        assert sim.model.relu1.output_quantizers[0]
        assert not sim.model.conv2.output_quantizers[0]
        assert sim.model.relu2.output_quantizers[0]
