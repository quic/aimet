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
import re
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.layers.core import TFOpLambda
from aimet_common.utils import AimetLogger
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ModelPreparer)

regex_for_camel_case_to_snake_case = re.compile(r'(?<!^)(?=[A-Z])')

def _get_original_models_weights_in_functional_model_order(original_model: tf.keras.Model,
                                                           functional_model: tf.keras.Model,
                                                           class_names: List[str]) -> np.ndarray:
    """
    Map the original model's weights to the functional model's weights
    :param original_model: The original model
    :param functional_model: The functional model
    :param class_names: The names of the classes that the original model was subclassed from
    :return: A list of the original model's weights in the order of the functional model's weights
    """

    # Make the original model's weights into a dictionary for quick lookup by name
    # The original subclassed layers names are removed to match the new functional model's names
    original_model_weights = {}
    for weight in original_model.weights:
        # pop out class_names of weight name
        weight_name = weight.name
        for class_name in class_names:
            weight_name = weight_name.replace(class_name + '/', '')
        original_model_weights[weight_name] = weight.numpy()

    # Get the functional model's weights in order as a dictionary for quick lookup where the key is the weight name
    # and the position of the weight's order is the value
    functional_model_weight_order = {
        weight.name: position
        for position, weight in enumerate(functional_model.weights)
    }

    # Using the functional model's weights order, get the original model's weights in the same order. The lambda here
    # uses the weight's name to get position in the functional model's weights order and the sorts the original model's
    # weights by that position.
    weights_in_correct_order = [
        weight for _, weight in
        sorted(original_model_weights.items(), key=lambda weight_info: functional_model_weight_order[weight_info[0]])
    ]

    return weights_in_correct_order

def _set_functional_models_weights(original_model: tf.keras.Model, functional_model: tf.keras.Model, class_names) -> None:
    """
    Set the functional model's weights to the original model's weights in the correct order
    :param original_model: The original model
    :param functional_model: The functional model
    :param class_names: The names of the classes that the original model was subclassed from
    """

    weights_in_correct_order = \
        _get_original_models_weights_in_functional_model_order(original_model, functional_model, class_names)

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


def _format_input_layer(original_model: tf.keras.Model,
                        input_layer: Union[tf.keras.layers.InputLayer, List[tf.keras.layers.InputLayer]] = None) -> \
        Union[tf.keras.layers.Layer, tf.keras.layers.Layer, ]:
    """
    This function formats the input layer and gets the layers to be copied from the original model.
    :param original_model: The original model to be copied
    :param input_layer: The input layer to be used for the functional model
    :return: The input layer, the previous layer,
    """

    try:
        input_layer = original_model.layers[0].input

    except AttributeError:
        logger.info("Input layer not found. Using input layer passed in.")
        if input_layer is None:
            raise ValueError("The top layer of this model is subclassed. Please provide an input layer via the "
                             "\'input_layer\' parameter.")

    prev_layer = input_layer
    return input_layer, prev_layer


def _prepare_model_helper(model: tf.keras.Model, prev_layer: tf.keras.layers.Layer, class_names: List[str]) \
        -> tf.keras.layers.Layer:
    """
    Helper function to recursively prepare a model. This function will be recursively called if a subclassed layer is
    found. This function will extract the layers from the subclassed layer and add them to the functional model.
    Otherwise, it will add the layer to the functional model.
    :param model: The model to prepare
    :param prev_layer: The previous layer
    :param class_names: The names of the classes that the original model was subclassed from
    :return: The last layer of the model
    """

    for current_layer in model.layers:
        # Skip input layer
        if isinstance(current_layer, tf.keras.layers.InputLayer):
            continue

        # If the current layer is a subclassed layer, extract the layers from the subclassed layer
        if current_layer.submodules and "tensorflow.python.keras" not in current_layer.__module__:
            logger.debug("Subclass layer \'%s\' found. Extracting layers.", current_layer.name)
            # Converts CamelCase to snake_case of subclassed layers class name
            class_names.append(
                regex_for_camel_case_to_snake_case.sub("_", current_layer.__class__.__name__).lower())

            # First, get the input shape of the subclassed layer and create an input layer with that shape
            # This is used to great a model based on the subclassed layer
            temp_input_shape = prev_layer.shape[1:]
            temp_input = tf.keras.Input(shape=temp_input_shape)

            # Create a model based on the subclassed layer.
            # This is done with the input layer created above being used for two reasons:
            # 1) The input layer is used to create the temporary functional model
            # 2) The input layer is used in the subclass layers call function as a symbolic tensor to get internal layers
            temp_model = tf.keras.Model(inputs=[temp_input], outputs=current_layer.call(temp_input, training=False))
            logger.debug("Model created for layer %s", current_layer.name)
            temp_model.summary(print_fn=logger.debug)

            prev_layer = _prepare_model_helper(temp_model, prev_layer, class_names=class_names)

        # If the current layer is a functional layer, call the layer with the previous layer
        elif isinstance(current_layer, Functional):
            prev_layer = current_layer.call(prev_layer)

        # If a normal Keras defined layer
        elif not isinstance(current_layer, TFOpLambda) and not isinstance(current_layer, tf.keras.layers.MultiHeadAttention):
            prev_layer = current_layer(prev_layer)

        # Call entire model if the current model doesn't have any Functional or Subclassed layers
        else:
            return model.call(prev_layer)

    return prev_layer


def prepare_model(original_model: tf.keras.Model,
                  input_layer: Union[tf.keras.layers.InputLayer, List[tf.keras.layers.InputLayer]] = None):
    """
    This function prepares a Keras model before continuing on with AIMET. Specifically, it will convert the model into
    a purely Functional API model and copy over the original models weights.
    :param original_model: The original model to be prepared
    :param input_layer: The input layer to be used for the new model. By default, the input layer is set to None. If the
    beginning portion of the model is subclassed, then the input layer must be passed in.
    """
    logger.debug("Preparing model for AIMET. Original model architecture")
    original_model.summary(print_fn=logger.debug)
    functional_input_layer, prev_layer = _format_input_layer(original_model, input_layer)

    # Used to fix weight names at end of unwrapping
    # Originally set to the name of the original model's class in the case that their is an inherited model
    class_names = [regex_for_camel_case_to_snake_case.sub("_", original_model.__class__.__name__).lower()]
    prev_layer = _prepare_model_helper(original_model, prev_layer, class_names)

    functional_model = tf.keras.Model(inputs=functional_input_layer, outputs=prev_layer)

    # Cloning model to remove any references to the original model
    model_to_return = tf.keras.models.clone_model(functional_model)
    logger.debug("Functional model architecture created")
    model_to_return.summary(print_fn=logger.debug)

    # Copying over weights from original model to functional model
    _set_functional_models_weights(original_model, model_to_return, class_names)

    return model_to_return
