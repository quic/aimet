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

import re
from typing import List, Dict, Tuple, Union

import numpy as np
import tensorflow as tf

from aimet_common.utils import AimetLogger
from aimet_common.layer_output_utils import SaveInputOutput, save_layer_output_names

from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils.common import create_input_feed_dict

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LayerOutputs)


class LayerOutputUtil:
    """ Implementation to capture and save outputs of intermediate layers of a model (fp32/quantsim) """

    def __init__(self, session: tf.compat.v1.Session, starting_op_names: List[str], output_op_names: List[str],
                 dir_path: str):
        """
        Constructor for LayerOutputUtil.

        :param session: Session containing the model whose layer-outputs are needed.
        :param starting_op_names: List of starting op names of the model.
        :param output_op_names: List of output op names of the model.
        :param dir_path: Directory wherein layer-outputs will be saved.
        """
        self.session = session
        self.starting_op_names = starting_op_names

        # Utility to capture layer-outputs
        self.layer_output = LayerOutput(session=session, starting_op_names=starting_op_names,
                                        output_op_names=output_op_names, dir_path=dir_path)

        # Identify the axis-layout used for representing an image tensor
        axis_layout = 'NHWC' if tf.keras.backend.image_data_format() == 'channels_last' else 'NCHW'

        # Utility to save model inputs and their corresponding layer-outputs
        self.save_input_output = SaveInputOutput(dir_path, axis_layout)

    def generate_layer_outputs(self, input_batch: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]]):
        """
        This method captures output of every layer of a model & saves the inputs and corresponding layer-outputs to disk.

        :param input_batch: Batch of inputs for which we want to obtain layer-outputs.
        :return: None
        """
        logger.info("Generating layer-outputs for %d input instances", len(input_batch))

        feed_dict = create_input_feed_dict(self.session.graph, self.starting_op_names, input_batch)

        # Obtain layer-output name to output dictionary
        layer_output_batch_dict = self.layer_output.get_outputs(feed_dict)

        # Save inputs and layer-outputs
        self.save_input_output.save(input_batch, layer_output_batch_dict)

        logger.info('Layer-outputs generated for %d input instances', len(input_batch))


class LayerOutput:
    """
    This class creates a layer-output name to layer-output dictionary. The layer-output names are as per the AIMET exported
    tensorflow model.
    """
    def __init__(self, session: tf.compat.v1.Session, starting_op_names: List[str], output_op_names: List[str], dir_path: str):
        """
        Constructor - It initializes few lists that are required for capturing and naming layer-outputs.

        :param session: Session containing TF model.
        :param starting_op_names: List of starting op names of the model.
        :param output_op_names: List of output op names of the model.
        """
        self.session = session
        self.activation_tensor_names, self.activation_tensors = LayerOutput.get_activation_tensor_info(
            session, starting_op_names, output_op_names)

        # Save activation tensor names which are in topological order of model graph. This order can be used while comparing layer-outputs.
        save_layer_output_names(self.activation_tensor_names, dir_path)

    def get_outputs(self, feed_dict: Dict) -> Dict[str, np.ndarray]:
        """
        This function creates layer-output name to layer-output dictionary. The layer-output names are as per the AIMET
        exported TF model.

        :param feed_dict: input tensor to input batch map
        :return: layer-output name to layer-output dictionary
        """
        act_outputs = self.session.run(self.activation_tensors, feed_dict=feed_dict)
        return dict(zip(self.activation_tensor_names, act_outputs))

    @staticmethod
    def get_activation_tensor_info(session: tf.compat.v1.Session, starting_op_names: List[str], output_op_names: List[str]) -> Tuple[List, List]:
        """
        This function fetches the activation tensors and its names from the given TF model. These activation tensors contain
        the layer-outputs of the given TF model.

        :param session: Session containing TF model.
        :param starting_op_names: List of starting op names of the model.
        :param output_op_names: List of output op names of the model.
        :return: activation_tensor_names, activation_tensors
        """
        connected_graph = ConnectedGraph(session.graph, starting_op_names, output_op_names)
        # pylint: disable=protected-access
        activation_op_names = QuantizationSimModel._get_ops_to_quantize_activations_for(session.graph, connected_graph)

        # Get activation quantization ops
        activation_quant_op_names = [op_name for op_name in activation_op_names if op_name.endswith('_quantized')]

        # If activation quant ops are present then capture only their tensors
        if activation_quant_op_names:
            activation_op_names = activation_quant_op_names

        activation_tensor_names = []
        activation_tensors = []
        for activation_op_name in activation_op_names:
            activation_op = session.graph.get_operation_by_name(activation_op_name)
            for output in activation_op.outputs:
                activation_tensor_names.append(output.name)
                activation_tensors.append(output)

        # Update activation tensor names by removing 'quantized:' string and replacing '/' with '_'.
        activation_tensor_names = [re.sub(r'\W+', "_", name.replace('quantized:', '')) for name in activation_tensor_names]

        return activation_tensor_names, activation_tensors
