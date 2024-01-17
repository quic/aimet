# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
import tempfile

import pytest
import torch.nn
import copy
import os
import json

from aimet_torch.experimental.v2.quantization.quantization_mixin import _QuantizationMixin
from aimet_torch.elementwise_ops import Add
from aimet_torch import onnx_utils
from aimet_torch.quantsim import QuantizationSimModel, OnnxExportApiArgs

from models_.models_to_test import SimpleConditional, ModelWithTwoInputs, ModelWith5Output, SoftMaxAvgPoolModel

# Key/values don't matter
dummy_encoding = {"min": 0,
                  "max": 2,
                  "scale": 2/255,
                  "offset": 0,
                  "bitwidth": 8,
                  "dtype": "int",
                  "is_symmetric": "False"}


class DummyMixin(_QuantizationMixin, torch.nn.Module):
    """ Dummy class for testing QuantSim export logic """

    def __init__(self, module, num_inputs, num_outputs, has_input_encodings, has_output_encodings):
        super(DummyMixin, self).__init__()
        # Assign a dummy output quantizer (since a real mixin will have child quantizers)
        self.output_quantizer = torch.nn.Identity()
        # Hide module inside list so it doesnt show up as a child (We will not actually have a wrapped module)
        self.module = [copy.deepcopy(module)]
        self._parameters = self.module[0]._parameters
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.has_input_encodings = has_input_encodings
        self.has_output_encodings = has_output_encodings
        self.dummy_encoding = copy.deepcopy(dummy_encoding)

    @classmethod
    def from_module(cls, module: torch.nn.Module, num_inputs=1, num_outputs=1, has_input_encodings=False, has_output_encodings=True):
        return cls(module, num_inputs, num_outputs, has_input_encodings, has_output_encodings)

    def forward(self, *inputs):
        return self.output_quantizer(self.module[0](*inputs))

    def export_input_encodings(self):
        enc = [self.dummy_encoding] if self.has_input_encodings else None
        return [enc] * self.num_inputs

    def export_output_encodings(self):
        enc = [self.dummy_encoding] if self.has_output_encodings else None
        return [enc] * self.num_outputs

    def export_param_encodings(self):
        enc_dict = {}
        for name, param in self.module[0].named_parameters():
            if name == "weight":
                enc_dict[name] = [self.dummy_encoding] * param.shape[0]
            else:
                enc_dict[name] = None
        return enc_dict

    def get_original_module(self):
        return copy.deepcopy(self.module[0])

class DummyModel(torch.nn.Module):

    def __init__(self, in_channels):
        super(DummyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(10, 10, 3, padding=1)
        self.add = Add()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x_resid = self.relu(x)
        x = self.conv2(x_resid)
        x = self.add(x, x_resid)
        return self.softmax(x)

export_args = {'opset_version': None, 'input_names': None, 'output_names': None}

class TestQuantsimOnnxExport:

    def test_onnx_export(self):
        export_args = OnnxExportApiArgs(opset_version=10, input_names=["input"], output_names=["output"])
        input_shape = (1, 10, 32, 32)
        fname = "test_model"
        dummy_input = torch.randn(input_shape)
        model = DummyModel(in_channels=input_shape[1])
        sim_model = copy.deepcopy(model)
        for name, module in sim_model.named_children():
            has_input_encodings = False if name != "conv1" else True
            has_output_encodings = True if name != "conv1" else False
            num_inputs = 2 if name == "add" else 1
            sim_model.__setattr__(name, DummyMixin.from_module(module, num_inputs, 1, has_input_encodings, has_output_encodings))


        with tempfile.TemporaryDirectory() as path:
            QuantizationSimModel.export_onnx_model_and_encodings(path, fname, model, sim_model, dummy_input=dummy_input,
                                                                 onnx_export_args=export_args,
                                                                 propagate_encodings=False, module_marker_map={},
                                                                 is_conditional=False, excluded_layer_names=None,
                                                                 quantizer_args=None)

            file_path = os.path.join(path, fname + '.encodings')

            assert os.path.exists(file_path)

            with open(file_path) as f:
                encoding_dict = json.load(f)

        # Format is "/layer_name/OnnxType_{output/input}_{idx}"
        expected_act_keys = {"input", "/relu/Relu_output_0", "/conv2/Conv_output_0", "/add/Add_output_0", "output"}
        expected_param_keys = {"conv1.weight", "conv2.weight"}

        assert set(encoding_dict["activation_encodings"].keys()) == expected_act_keys
        assert set(encoding_dict["param_encodings"].keys()) == expected_param_keys

        for encoding in encoding_dict["activation_encodings"].values():
            assert encoding[0] == dummy_encoding



    # From: https://github.com/quic/aimet/blob/ce3dafe75d81893cdb8b45ba8abf53a672c28187/TrainingExtensions/torch/test/python/test_quantizer.py#L2731
    def test_export_to_onnx_direct(self):
        model = ModelWithTwoInputs()
        sim_model = copy.deepcopy(model)
        dummy_input = (torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28))
        for name, layer in sim_model.named_children():
            has_input_encodings = name == "conv1_a"
            wrapped_layer = DummyMixin.from_module(layer, has_input_encodings=has_input_encodings)
            setattr(sim_model, name, wrapped_layer)

        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_utils.EXPORT_TO_ONNX_DIRECT = True
            QuantizationSimModel.export_onnx_model_and_encodings(temp_dir, "direct_onnx_export", model,
                                                                 sim_model, dummy_input, export_args,
                                                                 propagate_encodings=False)
            onnx_utils.EXPORT_TO_ONNX_DIRECT = False
            QuantizationSimModel.export_onnx_model_and_encodings(temp_dir, "onnxsaver_export", model,
                                                                 sim_model, dummy_input, export_args,
                                                                 propagate_encodings=False)

            with open(os.path.join(temp_dir, 'direct_onnx_export.encodings')) as direct_onnx_json:
                direct_onnx_encodings = json.load(direct_onnx_json)
            with open(os.path.join(temp_dir, 'onnxsaver_export.encodings')) as onnxsaver_json:
                onnxsaver_encodings = json.load(onnxsaver_json)

            assert len(direct_onnx_encodings['activation_encodings']) == \
                   len(onnxsaver_encodings['activation_encodings'])
            assert len(direct_onnx_encodings['param_encodings']) == len(onnxsaver_encodings['param_encodings'])
            direct_onnx_act_names = direct_onnx_encodings['activation_encodings'].keys()
            onnxsaver_act_names = onnxsaver_encodings['activation_encodings'].keys()
            assert direct_onnx_act_names != onnxsaver_act_names


    def test_encodings_propagation(self):
        """
        Test encodings are propagated correctly when more than
        one onnx node maps to the same torch module
        """
        export_args = OnnxExportApiArgs(opset_version=10, input_names=["input"], output_names=["output"])
        pixel_shuffel = torch.nn.PixelShuffle(2)
        model = torch.nn.Sequential(pixel_shuffel)
        sim_model = torch.nn.Sequential(DummyMixin.from_module(pixel_shuffel, num_inputs=1, has_input_encodings=True,
                                                               has_output_encodings=True))
        dummy_input = torch.randn(1, 4, 8, 8)

        # Save encodings
        with tempfile.TemporaryDirectory() as path:
            fname_no_prop = "encodings_propagation_false"
            fname_prop = "encodings_propagation_true"
            QuantizationSimModel.export_onnx_model_and_encodings(path, fname_no_prop, model, sim_model,
                                                                 dummy_input=dummy_input,
                                                                 onnx_export_args=export_args,
                                                                 propagate_encodings=False)
            QuantizationSimModel.export_onnx_model_and_encodings(path, fname_prop, model, sim_model,
                                                                 dummy_input=dummy_input,
                                                                 onnx_export_args=export_args,
                                                                 propagate_encodings=True)
            with open(os.path.join(path, fname_no_prop + ".encodings")) as f:
                encoding_dict_no_prop = json.load(f)["activation_encodings"]
            with open(os.path.join(path, fname_prop + ".encodings")) as f:
                encoding_dict_prop = json.load(f)["activation_encodings"]

        assert len(encoding_dict_no_prop) == 2
        assert len(encoding_dict_prop) == 4
        filtered_encoding_dict_prop = [{key: val} for key, val in encoding_dict_prop.items() if 'scale' in val[0]]
        assert len(filtered_encoding_dict_prop) == 2

    # From: https://github.com/quic/aimet/blob/ce3dafe75d81893cdb8b45ba8abf53a672c28187/TrainingExtensions/torch/test/python/test_quantizer.py#L3733
    def test_multi_output_onnx_op(self):
        """
        Test mapping and exporting of output encodings for multiple output onnx op.
        """
        model = ModelWith5Output()
        dummy_input = torch.randn(1, 3, 224, 224)
        sim_model = copy.deepcopy(model)
        class DummyMixinWithDisabledOutput(DummyMixin):
            def export_output_encodings(self):
                enc = [self.dummy_encoding]
                return [None] + ([enc] * (self.num_outputs - 1))

        sim_model.cust = DummyMixinWithDisabledOutput.from_module(sim_model.cust, num_inputs=1, num_outputs=5,
                                                                  has_input_encodings=True, has_output_encodings=True)

        with tempfile.TemporaryDirectory() as path:
            QuantizationSimModel.export_onnx_model_and_encodings(path, 'module_with_5_output', model, sim_model,
                                                                 dummy_input,
                                                                 onnx_export_args=(onnx_utils.OnnxExportApiArgs(opset_version=11)),
                                                                 propagate_encodings=False)
            with open(os.path.join(path, "module_with_5_output.encodings")) as json_file:
                activation_encodings = json.load(json_file)['activation_encodings']
                assert '7' not in activation_encodings
                assert set(['8', '9', '10', '11', 't.1']).issubset(activation_encodings.keys())
                for item in activation_encodings.values():
                    assert item[0] == sim_model.cust.dummy_encoding

    # From: https://github.com/quic/aimet/blob/ce3dafe75d81893cdb8b45ba8abf53a672c28187/TrainingExtensions/torch/test/python/test_quantizer.py#L1935
    def test_mapping_encoding_for_torch_module_with_multiple_onnx_ops(self):
        """
        Test the input and output encoding map to input/output at subgraph level when a torch module generates
        multiple onnx ops i.e. a sub-graph
        """
        dummy_input = torch.randn(1, 4, 256, 512)
        model = SoftMaxAvgPoolModel()

        sim_model  = copy.deepcopy(model)
        sim_model.sfmax = DummyMixin.from_module(sim_model.sfmax, 1, 1, True, True)
        sim_model.avgpool = DummyMixin.from_module(sim_model.avgpool, 1, 1, True, True)
        with tempfile.TemporaryDirectory() as path:
            QuantizationSimModel.export_onnx_model_and_encodings(path, "sfmaxavgpool_model", model, sim_model,
                                                                 dummy_input, export_args, propagate_encodings=False)
            with open(os.path.join(path, "sfmaxavgpool_model" + ".encodings")) as json_file:
                encoding_data = json.load(json_file)

        assert len(encoding_data["activation_encodings"]) == 3


    def test_conditional_export(self):
        """ Test exporting a model with conditional paths """
        model = SimpleConditional()
        model.eval()
        inp = torch.randn(1, 3)
        true_tensor = torch.tensor([1])
        false_tensor = torch.tensor([0])

        def forward_callback(model, _):
            model(inp, true_tensor)
            model(inp, false_tensor)

        sim_model = copy.deepcopy(model)

        qsim = QuantizationSimModel(model, dummy_input=(inp, true_tensor))
        qsim.compute_encodings(forward_callback, None)

        for name, module in sim_model.named_children():
            qsim_module = getattr(qsim.model, name)
            has_input_encodings = qsim_module.input_quantizers[0].enabled
            has_output_encodings = qsim_module.output_quantizers[0].enabled
            sim_model.__setattr__(name, DummyMixin.from_module(module, 1, 1, has_input_encodings, has_output_encodings))

        qsim.model = sim_model

        with tempfile.TemporaryDirectory() as path:
            qsim._export_conditional(path, 'simple_cond', dummy_input=(inp, false_tensor),
                                     forward_pass_callback=forward_callback, forward_pass_callback_args=None)

            with open(os.path.join(path, 'simple_cond.encodings')) as f:
                encodings = json.load(f)
                # verifying the encoding against default eAI HW cfg
                # activation encodings -- input, linear1 out, prelu1 out, linear2 out, prelu2 out, softmax out
                assert 6 == len(encodings['activation_encodings'])
                # param encoding -- linear 1 & 2 weight, prelu 1 & 2 weight
                assert 4 == len(encodings['param_encodings'])

        expected_encoding_keys = {"/linear1/Add_output_0",
                                  "/linear2/Add_output_0",
                                  "/prelu1/CustomMarker_1_output_0",
                                  "/prelu2/PRelu_output_0",
                                  "/softmax/CustomMarker_1_output_0",
                                  "_input.1",
                                  }
        assert encodings["activation_encodings"].keys() == expected_encoding_keys
