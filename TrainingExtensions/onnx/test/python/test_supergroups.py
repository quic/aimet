# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from aimet_onnx.quantsim import QuantizationSimModel
import onnx
from .utils import tmp_dir
from .models import models_for_tests


class TestDisableSupergroups:
    """
    TODO: As of QAIRT 2.37, following supergroups are not supported by HTP:
    1. Conv3d / ConvTranspose3d -> ...
    2. Depthwise Conv -> ...

    Disabling pattern matching for above two convolution cases in AIMET for short-term
    Issue #5597: Remove this test case when respective support is added in HTP and remove work-around.
    """

    def test_disable_conv3d_supergroup(self, tmp_dir):
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
        onnx_model_path = str(os.path.join(tmp_dir, "conv3d.onnx"))
        x = torch.randn((1, 3, 24, 24, 24))
        torch.onnx.export(
            model,
            x,
            onnx_model_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=16,
            dynamo=False,
        )
        onnx_model = onnx.load_model(onnx_model_path)

        sim = QuantizationSimModel(onnx_model)

        assert sim.qc_quantize_op_dict["/conv1/Conv_output_0"].enabled
        assert sim.qc_quantize_op_dict["/relu1/Relu_output_0"].enabled
        assert sim.qc_quantize_op_dict["/conv2/Conv_output_0"].enabled
        assert sim.qc_quantize_op_dict["/relu2/Relu_output_0"].enabled

    def test_disable_dynamic_conv3d_supergroup(self, tmp_dir):
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
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "output_height", 3: "output_width"},
        }
        x = torch.randn((1, 3, 24, 24, 24))

        onnx_model_path = str(os.path.join(tmp_dir, "dynamic_conv3d.onnx"))
        torch.onnx.export(
            model,
            x,
            onnx_model_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=16,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        onnx_model = onnx.load_model(onnx_model_path)
        sim = QuantizationSimModel(onnx_model)

        assert sim.qc_quantize_op_dict["/conv1/Conv_output_0"].enabled
        assert sim.qc_quantize_op_dict["/relu1/Relu_output_0"].enabled
        assert sim.qc_quantize_op_dict["/conv2/Conv_output_0"].enabled
        assert sim.qc_quantize_op_dict["/relu2/Relu_output_0"].enabled

    def test_disable_conv_transpose3d_supergroup(self, tmp_dir):
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
        onnx_model_path = str(os.path.join(tmp_dir, "convtranspose3d.onnx"))
        x = torch.randn((1, 3, 24, 24, 24))
        torch.onnx.export(
            model,
            x,
            onnx_model_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=16,
            dynamo=False,
        )
        onnx_model = onnx.load_model(onnx_model_path)

        sim = QuantizationSimModel(onnx_model)

        assert sim.qc_quantize_op_dict["/conv1/ConvTranspose_output_0"].enabled
        assert sim.qc_quantize_op_dict["/relu1/Relu_output_0"].enabled
        assert sim.qc_quantize_op_dict["/conv2/ConvTranspose_output_0"].enabled
        assert sim.qc_quantize_op_dict["/relu2/Relu_output_0"].enabled

    def test_disable_depthwise_conv_supergroup(self, tmp_dir):
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

        onnx_model_path = str(os.path.join(tmp_dir, "depthwise_conv.onnx"))
        x = torch.randn((1, 3, 24, 24))
        torch.onnx.export(
            model,
            x,
            onnx_model_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=16,
            dynamo=False,
        )
        onnx_model = onnx.load_model(onnx_model_path)
        sim = QuantizationSimModel(onnx_model)

        assert sim.qc_quantize_op_dict["/conv1/Conv_output_0"].enabled
        assert sim.qc_quantize_op_dict["/conv2/Conv_output_0"].enabled
        assert sim.qc_quantize_op_dict["output"].enabled

    def test_disable_depthwise_conv_transpose_supergroup(self, tmp_dir):
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

        onnx_model_path = str(os.path.join(tmp_dir, "depthwise_conv_transpose.onnx"))
        x = torch.randn((1, 3, 24, 24))
        torch.onnx.export(
            model,
            x,
            onnx_model_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=16,
            dynamo=False,
        )
        onnx_model = onnx.load_model(onnx_model_path)
        sim = QuantizationSimModel(onnx_model)

        assert sim.qc_quantize_op_dict["/conv1/ConvTranspose_output_0"].enabled
        assert sim.qc_quantize_op_dict["/conv2/ConvTranspose_output_0"].enabled
        assert sim.qc_quantize_op_dict["output"].enabled

    def test_disable_depthwise_conv_supergroup_trivial_case(self, tmp_dir):
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

        onnx_model_path = str(os.path.join(tmp_dir, "depthwise_conv.onnx"))
        x = torch.randn((1, 1, 24, 24))
        torch.onnx.export(
            model,
            x,
            onnx_model_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=16,
            dynamo=False,
        )
        onnx_model = onnx.load_model(onnx_model_path)
        sim = QuantizationSimModel(onnx_model)

        assert not sim.qc_quantize_op_dict["/conv1/Conv_output_0"].enabled
        assert not sim.qc_quantize_op_dict["/conv2/ConvTranspose_output_0"].enabled
        assert sim.qc_quantize_op_dict["output"].enabled

    def test_dynamic_matmul_add(self):
        """
        When: Model contains dynamic Matmul + Add pattern
        Then: Should not treat it as a fused matmul add
        """
        model = models_for_tests.matmul_add_with_transpose(dynamic_weight=True)
        sim = QuantizationSimModel(model, config_file="htp_v79")
        assert sim.qc_quantize_op_dict["matmul_out"].enabled
        assert sim.qc_quantize_op_dict["bias"].enabled
