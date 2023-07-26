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

""" This module contains utilities to capture and save intermediate layer-outputs of a model """

import copy
from typing import List, Dict, Tuple, Union
import numpy as np
from onnx import onnx_pb

from aimet_common.utils import AimetLogger
from aimet_common.layer_output_utils import SaveInputOutput, save_layer_output_names

from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import create_input_dict, get_graph_intermediate_activations, add_hook_to_get_activation

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LayerOutputs)


class LayerOutputUtil:
    """ Implementation to capture and save outputs of intermediate layers of a model (fp32/quantsim) """

    def __init__(self, model: onnx_pb.ModelProto, dir_path: str):
        """
        Constructor - It initializes the utility classes that captures and saves layer-outputs

        :param model: ONNX model
        :param dir_path: Directory wherein layer-outputs will be saved
        """
        self.model = model

        # Utility to capture layer-outputs
        self.layer_output = LayerOutput(model=model, dir_path=dir_path)

        # Utility to save model inputs and their corresponding layer-outputs
        self.save_input_output = SaveInputOutput(dir_path, 'NCHW')

    def generate_layer_outputs(self, input_batch: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]]):
        """
        This method captures output of every layer of a model & saves the inputs and corresponding layer-outputs to disk.

        :param input_batch: Batch of inputs for which we want to obtain layer-outputs.
        :return:
        """
        logger.info("Generating layer-outputs for %d input instances", len(input_batch))

        input_dict = create_input_dict(self.model, input_batch)

        layer_output_dict = self.layer_output.get_outputs(input_dict)

        self.save_input_output.save(input_batch, layer_output_dict)

        logger.info('Layer-outputs generated for %d input instances', len(input_batch))


class LayerOutput:
    """
    This class creates a layer-output name to layer-output dictionary.
    """
    def __init__(self, model: onnx_pb.ModelProto, dir_path: str):
        """
        Constructor - It initializes few lists that are required for capturing and naming layer-outputs.

        :param model: ONNX model
        :param dir_path: directory to store topological order of layer-output names
        """
        self.model = copy.deepcopy(model)
        self.activation_names = LayerOutput.get_activation_names(self.model)

        quantized_activation_names = [name for name in self.activation_names if name.endswith('_updated')]
        if quantized_activation_names:
            self.activation_names = quantized_activation_names

        LayerOutput.register_activations(self.model, self.activation_names)

        self.session = QuantizationSimModel.build_session(self.model, ['CPUExecutionProvider'])
        self.sanitized_activation_names = [name[:-len('_updated')] if name.endswith('_updated') else name for name in self.activation_names]

        # Save activation names which are in topological order of model graph. This order can be used while comparing layer-outputs.
        save_layer_output_names(self.sanitized_activation_names, dir_path)

    def get_outputs(self, input_dict: Dict) -> Dict[str, np.ndarray]:
        """
        This function creates layer-output name to layer-output dictionary.

        :param input_dict: input name to input tensor map
        :return: layer-output name to layer-output dictionary
        """
        activation_values = self.session.run(self.activation_names, input_dict)
        return dict(zip(self.sanitized_activation_names, activation_values))

    @staticmethod
    def get_activation_names(model: onnx_pb.ModelProto) -> List[str]:
        """
        This function fetches the activation names (layer-output names) of the given onnx model.

        :param model: ONNX model
        :return: list of activation names
        """
        activation_names = get_graph_intermediate_activations(model.graph)
        activation_names.extend([node.name for node in model.graph.output])
        return activation_names

    @staticmethod
    def register_activations(model: onnx_pb.ModelProto, activation_names: List):
        """
        This function adds the intermediate activations into the model's ValueInfoProto so that they can be fetched via
        running the session.

        :param model: ONNX model
        :param activation_names: list of activation names to be registered
        :return:
        """
        for act_name in activation_names:
            _ = add_hook_to_get_activation(model, act_name)
