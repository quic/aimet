# /usr/bin/env python3.5
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

"""Stores and updates Layer Attributes"""
import copy
from typing import Tuple
import tensorflow as tf

from aimet_common.utils import AimetLogger
import aimet_common.layer_database

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class Layer(aimet_common.layer_database.Layer):
    """ Holds attributes for a given layer """

    def _set_type_specific_params(self, module: tf.keras.layers.Layer):

        if isinstance(module, tf.keras.layers.Conv2D):
            params = aimet_common.layer_database.Conv2dTypeSpecificParams(module.strides, module.padding, module.groups)
            self.type_specific_params = params

    def __init__(self, layer: tf.keras.layers.Layer, name: str, output_shape: Tuple):
        """
        Constructor
        :param layer: Reference to the layer
        :param name: Unique name of the layer
        :param output_shape: Shape of the output activations
        """
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            weight_shape = (layer.depth_multiplier, layer.input_shape[-1], layer.kernel_size[0], layer.kernel_size[1])
        elif isinstance(layer, tf.keras.layers.Conv2D):
            weight_shape = (layer.kernel.shape[3], layer.kernel.shape[2], layer.kernel.shape[0], layer.kernel.shape[1])
        elif isinstance(layer, tf.keras.layers.Dense):
            weight_shape = (layer.kernel.shape[1], layer.kernel.shape[0], 1, 1)
        else:
            raise AssertionError("Layer currently supports only Conv2D and Linear")

        aimet_common.layer_database.Layer.__init__(self, layer, name, weight_shape, output_shape)

        self.var_name_of_module_in_parent = None
        self.parent_module = None


class LayerDatabase(aimet_common.layer_database.LayerDatabase):
    """
    Stores, creates and updates the Layer database
    Also stores compressible layers to model optimization
    """

    def __init__(self, model: tf.keras.Model):
        """
        LayerDatabase constructor
        :param model: Model
        """
        aimet_common.layer_database.LayerDatabase.__init__(self, model)
        self._create_database(model)

    def __deepcopy__(self, memodict):

        # pylint: disable=protected-access

        # Allocate a new LayerDatabase
        layer_db = copy.copy(self)
        memodict[id(self)] = layer_db

        # Create a deep copy of the model
        layer_db._model = tf.keras.models.clone_model(self._model)
        layer_db._model.set_weights(self._model.get_weights())

        # Re-create the compressible layers dict
        layer_db._compressible_layers = {}

        modules_in_copy = list(layer_db._model.layers)

        # For all modules in the current model
        for index, module in enumerate(self._model.layers):

            # If this module is in the current layer database
            if id(module) in self._compressible_layers:
                existing_layer = self._compressible_layers[id(module)]
                new_layer = Layer(modules_in_copy[index], existing_layer.name,
                                  existing_layer.output_shape)
                new_layer.picked_for_compression = existing_layer.picked_for_compression
                layer_db._compressible_layers[id(modules_in_copy[index])] = new_layer

        return layer_db

    # pylint: disable=unused-argument
    def _collect_layer_attributes(self, layer: tf.keras.layers.Layer):
        """
        Custom hook function which will be applied to all the layers in the model and store following
        information :
        model name (which will be removed), model reference, input shape and output shape
        """
        output_activation_shape = list(layer.output_shape)
        if len(output_activation_shape) == 4:
            reorder = [0, 3, 1, 2]
            output_activation_shape = [output_activation_shape[idx] for idx in reorder]
        # activation dimension for FC layer is (1,1)
        if isinstance(layer, tf.keras.layers.Dense):
            output_activation_shape.extend([1, 1])

        layer_name = layer.name

        self._compressible_layers[id(layer)] = Layer(layer, layer_name, output_activation_shape)

    def _create_database(self, model: tf.keras.Model):
        """
        Register custom hook for the model with run_graph provided by user
        if the user wants to experiment with custom hook, we can support that option by
        exposing the hook parameter to compress_net method
        :param model: Model
        """
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                self._collect_layer_attributes(layer)

    def replace_layer_with_sequential_of_two_layers(self, layer_to_replace: Layer,
                                                    layer_a: Layer, layer_b: Layer):
        """
        Replaces original layer with two new layers in the graph.
        Adds two new layers in the database and remove the original layer from database.

        :param layer_to_replace: layer to replace
        :param layer_a: layer a
        :param layer_b: layer b
        """

        # Add the new layer to the database
        self._compressible_layers[id(layer_a.module)] = layer_a
        self._compressible_layers[id(layer_b.module)] = layer_b

        # Remove the layer being replaced from the database
        del self._compressible_layers[id(layer_to_replace.module)]

    def get_compressible_layers(self) -> Layer:
        """
        :return: Returns compressible layers
        """
        return self._compressible_layers

    def destroy(self):
        """
        Destroys the layer database
        """
        # clear the dictionary
        self._compressible_layers.clear()
        self._model = None
