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

""" This module contains utilities to capture and save intermediate layer-outputs of a model. """

import os
import re
from collections import OrderedDict
import json
import tensorflow as tf
from keras.utils import tf_utils
from aimet_tensorflow.keras.quantsim import QcQuantizeWrapper
from aimet_common.layer_output_utils import SaveInputOutput
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LayerOutputs)

class LayerOutputUtil:
    """
    This class captures output of every layer of a keras (fp32/quantsim) model, creates a layer-output name to
    layer-output dictionary and saves the per layer outputs
    """
    def __init__(self, model: tf.keras.Model, save_dir: str = f"./KerasLayerOutput"):
        """
        Constructor - It initializes a few things that are required for capturing and naming layer-outputs.
        :param model: Keras (fp32/quantsim) model.
        :param save_dir: Directory to save the layer outputs.
        """
        self.model = model

        # Freeze the model, i.e, run in test mode
        self.model.trainable = False

        # Get Intermediate model for layer-outputs
        self.intermediate_model = self._get_intermediate_model(self.model)

        # Get Actual Layer output name to Modified Layer Output name dict
        self.layer_output_name_mapper = self._layer_output_name_mapper(self.model)

        # Saving the actual layer output name to modified layer output name (valid file name to save) in a json file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        json.dump(self.layer_output_name_mapper, open(os.path.join(save_dir, "LayerOutputNameMapper.json"), 'w'), indent=4)

        # Identify the axis-layout used for representing an image tensor
        axis_layout = 'NHWC' if tf.keras.backend.image_data_format() == 'channels_last' else 'NCHW'

        # Utility to save model inputs and their corresponding layer-outputs
        self.save_inp_out_obj = SaveInputOutput(save_dir, axis_layout=axis_layout)

        logger.info("Initialised LayerOutputUtil Class for Keras")

    @classmethod
    def _get_layer_name(cls, layer):
        if isinstance(layer, QcQuantizeWrapper):
            return layer.original_layer.output.name
        return layer.output.name

    @classmethod
    def _get_intermediate_model(cls, model):
        inputs = [model.layers[0].input]
        outputs = [layer.output for layer in model.layers]
        intermediate_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        intermediate_model.trainable = False
        return intermediate_model

    @classmethod
    def _layer_output_name_mapper(cls, model):
        layer_output_name_mapper = OrderedDict()
        for layer in model.layers:
            layer_output_name = cls._get_layer_name(layer)

            # Replace all Non-word characters with "_" to make it a valid file name for saving the results
            # For Eg.: "conv2d/BiasAdd:0" gets converted to "conv2d_BiasAdd_0"
            modified_layer_output_name = re.sub(r'\W+', "_", layer_output_name)

            layer_output_name_mapper[layer_output_name] = modified_layer_output_name

        return layer_output_name_mapper

    def get_outputs(self, input_batch: tf.Tensor):
        """
        This function captures layer-outputs and renames them as per the AIMET exported model.
        :param input_batch: Batch of inputs for which we want to obtain layer-outputs.
        :return: layer-output name to layer-output batch dict
        """
        outs = self.intermediate_model(input_batch)
        output_pred = tf_utils.sync_to_numpy_or_python_type(outs)

        return dict(zip(self.layer_output_name_mapper.values(), output_pred))

    def generate_layer_outputs(self, input_batch: tf.Tensor):
        """
        This method captures output of every layer of a keras model & saves the inputs and corresponding layer-outputs to disk.
        This allows layer-output comparison either between original fp32 model and quantization simulated model or quantization
        simulated model and actually quantized model on-target to debug accuracy miss-match issues.

        :param input_batch: Batch of Inputs for which layer output need to be generated
        :return: None
        """

        batch_layer_name_to_layer_output = self.get_outputs(input_batch)
        self.save_inp_out_obj.save(input_batch, batch_layer_name_to_layer_output)

        logger.info("Layer Outputs Saved")
