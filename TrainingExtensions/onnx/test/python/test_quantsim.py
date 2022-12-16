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
import json
import os
import torch
import numpy as np
from onnx import load_model
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.qc_quantize_op import OpMode, reset_qc_quantize_op_dict
from aimet_torch.quantsim import QuantizationSimModel as PtQuantizationSimModel
from aimet_torch.examples.test_models import SingleResidual
from test_models import build_dummy_model


class DummyModel(SingleResidual):
    """
    Model
    """
    def __init__(self):
        super().__init__()
        # change padding size to 0, onnxruntime only support input size is the factor of output size for pooling
        self.conv4 = torch.nn.Conv2d(32, 8, kernel_size=2, stride=2, padding=0, bias=True)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        del self.bn1
        del self.bn2

    def forward(self, inputs):
        x = self.conv1(inputs)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Save the output of MaxPool as residual.
        residual = x

        x = self.conv2(x)
        # TODO
        # remove bn layer for currently not supporting non-4 dim param tensors
        # x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        residual = self.conv4(residual)
        residual = self.ada(residual)
        x += residual
        x = self.relu3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class TestQuantSim:
    """Tests for QuantizationSimModel"""
    def test_insert_quantize_op_nodes(self):
        """ Test to insert qc quantize op to the graph"""
        model = build_dummy_model()
        sim = QuantizationSimModel(model)
        assert len(sim.model.nodes()) == 15

        node_ls = [node.op_type for node in sim.model.nodes()]
        assert node_ls == ['Conv', 'Relu', 'MaxPool', 'Flatten', 'Gemm'] + ['QcQuantizeOp'] * 10

        # Check if qc quantize op node is correctly connect to the corresponding onnx node
        assert sim.model.find_node_by_name('QcQuantizeOp_input', [], sim.model.graph()).output[0] == \
               sim.model.find_node_by_name('conv', [], sim.model.graph()).input[0]
        # Check if op_mode is set correctly for each qc quantize op node
        qc_quantize_op_dict = sim.get_qc_quantize_op()
        for name in sim.param_names:
            assert qc_quantize_op_dict[name].op_mode == OpMode.one_shot_quantize_dequantize
        for name in sim.activation_names:
            assert qc_quantize_op_dict[name].op_mode == OpMode.update_stats
        reset_qc_quantize_op_dict()

    def test_compute_encodings(self):
        """Test to perform compute encodings"""
        model = build_dummy_model()
        sim = QuantizationSimModel(model)

        for quantizer in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[quantizer].enabled = True

        for name, qc_op in sim.get_qc_quantize_op().items():
            assert qc_op.tensor_quantizer.isEncodingValid is False

        def callback(session, args):
            in_tensor = {'input': np.random.rand(1, 3, 32, 32).astype(np.float32)}
            session.run(None, in_tensor)

        sim.compute_encodings(callback, None)

        for name, qc_op in sim.get_qc_quantize_op().items():
            assert qc_op.encodings.bw == 8

        for name, qc_op in sim.get_qc_quantize_op().items():
            assert qc_op.tensor_quantizer.isEncodingValid is True
            assert qc_op.op_mode == OpMode.quantize_dequantize or OpMode.one_shot_quantize_dequantize
        reset_qc_quantize_op_dict()

    def test_export_model_with_quant_args(self):
        """Test to export encodings and model"""
        if not os.path.exists('./tmp'):
            os.mkdir('./tmp')
        model = build_dummy_model()
        sim = QuantizationSimModel(model, default_activation_bw=16, default_param_bw=16,
                                   quant_scheme=QuantScheme.post_training_tf)

        for quantizer in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[quantizer].enabled = True


        def dummy_callback(session, args):
            pass

        sim.compute_encodings(dummy_callback, None)

        sim.export('./tmp/', 'quant_sim_model_with_quant_args')
        with open('./tmp/quant_sim_model_with_quant_args.encodings') as json_file:
            encoding_data = json.load(json_file)

        assert "quantizer_args" in encoding_data
        quantizer_args = encoding_data["quantizer_args"]
        assert quantizer_args["activation_bitwidth"] == 16
        assert quantizer_args["param_bitwidth"] == 16
        assert not quantizer_args["per_channel_quantization"]
        assert quantizer_args["quant_scheme"] == QuantScheme.post_training_tf.name
        assert quantizer_args["dtype"] == "int"
        assert "is_symmetric" in quantizer_args

    def test_export_model(self):
        """Test to export encodings and model"""
        if not os.path.exists('/tmp'):
            os.mkdir('/tmp')
        model = build_dummy_model()
        sim = QuantizationSimModel(model)

        for quantizer in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[quantizer].enabled = True

        def dummy_callback(session, args):
            pass

        sim.compute_encodings(dummy_callback, None)
        sim.export('/tmp/', 'quant_sim_model')

        with open('/tmp/quant_sim_model.encodings', 'rb') as json_file:
            encoding_data = json.load(json_file)
        activation_keys = list(encoding_data["activation_encodings"].keys())
        assert activation_keys == ['3', '4', '5', '6', 'input', 'output']
        for act in activation_keys:
            act_encodings_keys = list(encoding_data["activation_encodings"][act].keys())
            assert act_encodings_keys == ['bitwidth', 'dtype', 'is_symmetric', 'max', 'min', 'offset', 'scale']

        param_keys = list(encoding_data['param_encodings'].keys())
        assert param_keys == ['conv_b', 'conv_w', 'fc_b', 'fc_w']
        for param in param_keys:
            param_encodings_keys = list(encoding_data["param_encodings"][param].keys())
            assert param_encodings_keys == ['bitwidth', 'dtype', 'is_symmetric', 'max', 'min', 'offset', 'scale']
        reset_qc_quantize_op_dict()

    def test_compare_encodings_with_PT(self):
        """Test to compare encodings with PT"""
        if not os.path.exists('/tmp'):
            os.mkdir('/tmp')

        def pytorch_callback(model, inputs):
            model.eval()
            model(torch.as_tensor(inputs))

        def onnx_callback(session, inputs):
            in_tensor = {'input': inputs}
            session.run(None, in_tensor)

        inputs = np.random.rand(1, 3, 32, 32).astype(np.float32)
        model = DummyModel()
        model.eval()

        torch.onnx.export(model, torch.as_tensor(inputs), '/tmp/dummy_model.onnx', training=torch.onnx.TrainingMode.PRESERVE,
                          input_names=['input'], output_names=['output'])

        pt_sim = PtQuantizationSimModel(model, dummy_input=torch.as_tensor(inputs))
        pt_sim.compute_encodings(pytorch_callback, inputs)
        pt_sim.export('/tmp', 'pt_sim', dummy_input=torch.as_tensor(inputs))

        onnx_model = load_model('/tmp/dummy_model.onnx')

        onnx_sim = QuantizationSimModel(onnx_model)

        activation_encodings_map = {'12': '9', '15': '10', '21': '12', '24': '13', '27': '14', '30': '15',
                                    '34': '17', '38': '19', 't.1': 'input'}
        onnx_sim.compute_encodings(onnx_callback, inputs)
        onnx_sim.export('/tmp', 'onnx_sim')

        with open('/tmp/pt_sim.encodings') as f:
            pt_encodings = json.load(f)
        with open('/tmp/onnx_sim.encodings') as f:
            onnx_encodings = json.load(f)

        for pt, onnx in activation_encodings_map.items():
            assert round(pt_encodings['activation_encodings'][pt][0]['max'], 4) == \
                   round(onnx_encodings['activation_encodings'][onnx]['max'], 4)
            assert round(pt_encodings['activation_encodings'][pt][0]['min'], 4) == \
                   round(onnx_encodings['activation_encodings'][onnx]['min'], 4)
            assert round(pt_encodings['activation_encodings'][pt][0]['scale'], 4) == \
                   round(onnx_encodings['activation_encodings'][onnx]['scale'], 4)
            assert pt_encodings['activation_encodings'][pt][0]['offset'] == \
                   onnx_encodings['activation_encodings'][onnx]['offset']

        for name in list(onnx_encodings['param_encodings'].keys()):
            assert round(pt_encodings['param_encodings'][name][0]['max'], 4) == \
                   round(onnx_encodings['param_encodings'][name]['max'], 4)
            assert round(pt_encodings['param_encodings'][name][0]['min'], 4) == \
                   round(onnx_encodings['param_encodings'][name]['min'], 4)
            assert round(pt_encodings['param_encodings'][name][0]['scale'], 4) == \
                   round(onnx_encodings['param_encodings'][name]['scale'], 4)
            assert pt_encodings['param_encodings'][name][0]['offset'] == \
                   onnx_encodings['param_encodings'][name]['offset']
