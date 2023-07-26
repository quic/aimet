# /usr/bin/env python3.8
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

import os
import shutil

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from aimet_onnx.utils import make_dummy_input
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.layer_output_utils import LayerOutput, LayerOutputUtil
from models.models_for_tests import build_dummy_model_with_dynamic_input


def get_original_model_artifacts():
    model = build_dummy_model_with_dynamic_input()
    output_names = [node.name for node in model.graph.input]
    for node in model.graph.node:
        output_names.extend(node.output)
    input_dict = make_dummy_input(model)
    return model, output_names, input_dict


def get_quantsim_artifacts():
    model, _, input_dict = get_original_model_artifacts()

    def callback(session, input_dict):
        session.run(None, input_dict)

    quantsim = QuantizationSimModel(model=model, dummy_input=input_dict, use_cuda=False)
    quantsim.compute_encodings(callback, input_dict)

    output_names = [node.name for node in quantsim.model.model.graph.input]
    for node in quantsim.model.model.graph.node:
        output_names.extend(node.output)
    output_names = [name[:-len('_updated')] for name in output_names if name.endswith('_updated')]

    return quantsim, output_names, input_dict


class TestLayerOutput:
    def test_get_original_model_outputs(self):
        """ Test whether outputs are generated for all the layers of an original model """

        # Get original model artifacts
        model, output_names, input_dict = get_original_model_artifacts()

        temp_dir_path = os.path.dirname(os.path.abspath(__file__))
        temp_dir_path = os.path.join(temp_dir_path, 'temp_dir')

        # Obtain layer-outputs of original model
        layer_output = LayerOutput(model, temp_dir_path)
        output_name_to_output_val_dict = layer_output.get_outputs(input_dict)

        # Verify whether outputs are generated for all the layers
        for name in output_names:
            assert name in output_name_to_output_val_dict, \
                "Output not generated for " + name

        # Verify whether captured outputs are correct. This can only be checked for final output of the model
        session = QuantizationSimModel.build_session(model, ['CPUExecutionProvider'])
        assert np.array_equal(session.run(None, input_dict)[0], output_name_to_output_val_dict['output'])

        # Delete temp_dir
        shutil.rmtree(temp_dir_path, ignore_errors=False, onerror=None)

    def test_get_quantsim_model_outputs(self):
        """ Test whether outputs are generated for all the layers of a quantsim model """

        # Get quantsim artifacts
        quantsim, output_names, input_dict = get_quantsim_artifacts()

        temp_dir_path = os.path.dirname(os.path.abspath(__file__))
        temp_dir_path = os.path.join(temp_dir_path, 'temp_dir')

        # Obtain layer-outputs of quantsim model
        layer_output = LayerOutput(quantsim.model.model, temp_dir_path)
        output_name_to_output_val_dict = layer_output.get_outputs(input_dict)

        # Verify whether outputs are generated for all the layers
        for name in output_names:
            assert name in output_name_to_output_val_dict, \
                "Output not generated for " + name

        # Verify whether captured outputs are correct. This can only be checked for final output of the model
        session = QuantizationSimModel.build_session(quantsim.model.model, ['CPUExecutionProvider'])
        assert np.array_equal(session.run(None, input_dict)[0], output_name_to_output_val_dict['output'])

        # Delete temp_dir
        shutil.rmtree(temp_dir_path, ignore_errors=False, onerror=None)
        pass


def get_dataset_artifacts():
    class DummyDataset(Dataset):
        def __init__(self, count):
            Dataset.__init__(self)
            self.random_img = []
            while count:
                self.random_img.append(torch.rand(3, 32, 32))
                count -= 1

        def __len__(self):
            return len(self.random_img)

        def __getitem__(self, idx):
            return self.random_img[idx]

    data_count = 4
    dummy_dataset = DummyDataset(data_count)
    dummy_dataloader = DataLoader(dataset=dummy_dataset, batch_size=2)

    return dummy_dataset, dummy_dataloader, data_count


class TestLayerOutputUtil:
    def test_generate_layer_outputs(self):
        """ Test whether input files and corresponding layer-output files are generated """

        # Get quantsim artifacts
        quantsim, output_names, input_dict = get_quantsim_artifacts()

        # Get dataset artifacts
        dummy_dataset, dummy_data_loader, data_count = get_dataset_artifacts()

        temp_dir_path = os.path.dirname(os.path.abspath(__file__))
        temp_dir_path = os.path.join(temp_dir_path, 'temp_dir')

        # Generate layer-outputs
        layer_output_util = LayerOutputUtil(model=quantsim.model.model, dir_path=temp_dir_path)
        for input_batch in dummy_data_loader:
            layer_output_util.generate_layer_outputs(input_batch.numpy())

        # Verify number of inputs
        assert data_count == len(os.listdir(os.path.join(temp_dir_path, 'inputs')))

        # Verify number of layer-output folders
        assert data_count == len(os.listdir(os.path.join(temp_dir_path, 'outputs')))

        # Verify number of layer-outputs
        saved_layer_outputs = os.listdir(os.path.join(temp_dir_path, 'outputs', 'layer_outputs_0'))
        saved_layer_outputs = [i[:-len('.raw')] for i in saved_layer_outputs]
        for name in output_names:
            assert name in saved_layer_outputs

        # Ensure generated layer-outputs can be correctly loaded for layer-output comparison
        saved_last_layer_output = np.fromfile(os.path.join(temp_dir_path, 'outputs', 'layer_outputs_0', 'output.raw'), dtype=np.float32).reshape((1, 10))
        session = QuantizationSimModel.build_session(quantsim.model.model, ['CPUExecutionProvider'])
        input_dict = {'input': np.expand_dims(dummy_dataset.__getitem__(0).numpy(), axis=0)}
        assert np.array_equal(session.run(None, input_dict)[0], saved_last_layer_output)

        # Delete temp_dir
        shutil.rmtree(temp_dir_path, ignore_errors=False, onerror=None)
