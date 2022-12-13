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

""" Implementation to automatically prepare keras models for AIMET by converting them to a functional model """

from typing import List, Union
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional

from aimet_common.utils import AimetLogger
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ModelPreparer)


def get_original_models_weights_in_functional_model_order(original_model: tf.keras.Model,
                                                          functional_model: tf.keras.Model) -> np.ndarray:
    """
    Map the original model's weights to the functional model's weights
    :param original_model: The original model
    :param functional_model: The functional model
    :return: A list of the original model's weights in the order of the functional model's weights
    """

    # Make the original model's weights into a dictionary for quick lookup by name
    original_model_weights = {weight.name: weight for weight in original_model.weights}

    # Get the functional model's weights in order as a dictionary for quick lookup where the key is the weight name
    # and the position of the weight's order is the value
    functional_model_weight_order = {weight.name: position for position, weight in enumerate(functional_model.weights)}

    # Using the functional model's weights order, get the original model's weights in the same order. The lambda here
    # uses the weight's name to get position in the functional model's weights order and the sorts the original model's
    # weights by that position.
    weights_in_correct_order = [
        weight.numpy() for _, weight in
        sorted(original_model_weights.items(), key=lambda weight_info: functional_model_weight_order[weight_info[0]])
    ]

    return weights_in_correct_order


def _recursively_unwrap_subclass_layer(subclassed_layer: tf.keras.layers.Layer,
                                       prev_layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    """
    This function recursively unwraps the layers of a subclassed layer. While unwrapping, it also builds up the
    functional model by connecting the layers together.
    :return: The last layer in the model
    """

    # Base case: If the subclassed_layer is not subclassed, return the layer
    if not subclassed_layer.submodules:
        logger.debug("Base case: %s", subclassed_layer.name)
        return subclassed_layer(prev_layer)

    # First, get the input shape of the subclassed layer and create an input layer with that shape
    # This is used to great a model based on the subclassed layer
    input_shape = subclassed_layer.input.shape[1:]
    x = tf.keras.Input(shape=input_shape)

    # Create a model based on the subclassed layer.
    # This is done with the input layer created above being used for two reasons:
    # 1) The input layer is used to create the temporary functional model
    # 2) The input layer is used in the subclass layers call function as a symbolic tensor to get internal layers
    temp_model = tf.keras.Model(inputs=[x], outputs=subclassed_layer.call(x))

    # Recursively unwrap the layers of the temporary model. If the temporary model also has subclassed layers,
    # the recursive call will unwrap those layers as well.
    for layer in temp_model.layers[1:]:
        logger.debug("Unwrapping layer: %s", layer.name)
        prev_layer = _recursively_unwrap_subclass_layer(layer, prev_layer)

    return prev_layer


def _extract_functional_model(functional_model_found: Functional,
                              prev_layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    """
    This function extracts the layers from a Functional API model and returns the last layer in the model.
    :param functional_model_found: The Functional API model to be extracted
    :param prev_layer: The previous layer in the model
    :return: The last layer in the model
    """

    logger.debug("Functional model found. Extracting layers.")
    for func_layer in functional_model_found.layers[1:]:
        prev_layer = func_layer(prev_layer)
    logger.info("Functional model extracted.")

    return prev_layer


def _format_input_layer_and_get_layers_to_copy(original_model: tf.keras.Model,
                                               input_layer: Union[tf.keras.Input, List[tf.keras.Input]] = None) -> \
        Union[tf.keras.layers.Layer, tf.keras.layers.Layer, List[tf.keras.layers.Layer]]:

    try:
        input_layer = original_model.layers[0].input
        prev_layer = input_layer
        layers_to_copy = original_model.layers[1:]

    except AttributeError:
        logger.info("Input layer not found. Using input layer passed in.")
        if input_layer is None:
            raise ValueError("The top layer of this model is subclassed. Please provide an input layer via the "
                             "input_layer parameter.")
        prev_layer = input_layer
        layers_to_copy = original_model.layers

    return input_layer, prev_layer, layers_to_copy


def _set_functional_models_weights(original_model: tf.keras.Model, functional_model: tf.keras.Model) -> None:
    weights_in_correct_order = get_original_models_weights_in_functional_model_order(original_model, functional_model)
    try:
        functional_model.set_weights(weights_in_correct_order)
    except ValueError:
        logger.error(
            "Could not copy weights from original model to functional model. This can occur when "
            "custom sublayers are defined not in the same order as the sublayers call method. Please ensure that the "
            "sublayers internal layers are defined in the same order as the sublayers call method.")
        raise

    logger.debug("Functional model weights copied.")
    logger.info("Model prepared for AIMET in Functional API format.")


def prepare_model(original_model: tf.keras.Model, input_layer: Union[tf.keras.Input, List[tf.keras.Input]] = None):
    """
    This function prepares a Keras model before continuing on with AIMET. Specifically, it will convert the model into
    a purely Functional API model and copy over the original models weights.
    :param original_model: The original model to be prepared
    :param input_layer: The input layer to be used for the new model. By default, the input layer is set to None. If the
    beginning portion of the model is subclassed, then the input layer must be passed in.
    """

    input_layer, prev_layer, layers_to_copy = _format_input_layer_and_get_layers_to_copy(original_model, input_layer)

    for layer in layers_to_copy:
        if isinstance(layer, Functional):
            prev_layer = _extract_functional_model(layer, prev_layer)

        elif layer.submodules:
            logger.debug("Subclass layer found. Extracting layers.")
            prev_layer = _recursively_unwrap_subclass_layer(layer, prev_layer)

        else:
            prev_layer = layer(prev_layer)

    functional_model = tf.keras.Model(inputs=input_layer, outputs=prev_layer)
    logger.debug("Functional model architecture created.")

    _set_functional_models_weights(original_model, functional_model)

    return functional_model
