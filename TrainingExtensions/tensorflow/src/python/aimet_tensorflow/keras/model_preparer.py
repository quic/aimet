# /usr/bin/env python3.8
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

from typing import Dict, List, Union
import re
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.layers.core import TFOpLambda

from aimet_tensorflow.keras.utils.model_transform_utils import replace_separable_conv_with_depthwise_pointwise
from aimet_common.utils import AimetLogger
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ModelPreparer)

regex_for_camel_case_to_snake_case = re.compile(r'(?<!^)(?=[A-Z])')
_TEMP_MODEL_NAME = "temp_aimet_subclass_model"

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

    return input_layer

def _get_class_names_in_model(model: tf.keras.Model) -> List[str]:
    return [regex_for_camel_case_to_snake_case.sub("_", model.__class__.__name__).lower()]

def _is_subclass_layer(layer: tf.keras.layers.Layer) -> bool:
    """
    Checks if the given layer is a subclassed layer
    :param layer: The layer to check
    :return: True if the layer is a subclassed layer, False otherwise
    """
    return layer.submodules and "tensorflow.python.keras" not in layer.__module__

def _is_functional_model(layer: tf.keras.layers.Layer) -> bool:
    """
    Checks if the given layer is a functional layer
    :param layer: The layer to check
    :return: True if the layer is a functional layer, False otherwise
    """
    return isinstance(layer, Functional) and not isinstance(layer, tf.keras.Sequential) 

def _set_prepared_models_input_layer(model: tf.keras.Model, network_dict, input_layer: tf.keras.layers.InputLayer) -> None:
    """
    This function sets the input layer of the model to the input layer of the functional model
    :param model: The original model
    :param network_dict: The network dictionary
    """
    try:
        if isinstance(model.input, list):
            for inp in model.input:
                network_dict[NetworkDictProperities.NEW_OUTPUT_TENSOR_OF.value].update({inp.name: inp})
        else:
            network_dict[NetworkDictProperities.NEW_OUTPUT_TENSOR_OF.value].update({model.input.name: model.input})
    except AttributeError:
        # For models that are not connected
        logger.info("Model is not connected. Setting input layer to input layer passed in.")
        network_dict[NetworkDictProperities.INPUT_LAYERS_OF.value].update({model.layers[0].name: [input_layer.name]})
        network_dict[NetworkDictProperities.NEW_OUTPUT_TENSOR_OF.value].update({input_layer.name: input_layer})

# TODO: model_outputs type?? and network_dict
def _get_layer_input(layer: tf.keras.layers.Layer, network_dict: Dict) -> tf.keras.layers.Layer:
    """
    Helper function to get the input layer of a layer. This function will recursively call itself if the layer is a
    subclassed layer.
    :param layer: The layer to get the input layer of
    :return: The input layer of the layer
    """
    try:
        layer_input = [network_dict[NetworkDictProperities.NEW_OUTPUT_TENSOR_OF.value][layer_aux]
                    for layer_aux in network_dict[NetworkDictProperities.INPUT_LAYERS_OF.value][layer.name]]

        if len(layer_input) == 1:
            layer_input = layer_input[0]
    except KeyError:
        layer_input = _get_most_recently_added_output_tensor(network_dict)

    return layer_input


def _update_output_tensors_in_network_dict(
        layer: tf.keras.layers.Layer, new_output_tensor: KerasTensor, model: tf.keras.Model, network_dict: Dict,
        model_outputs: List[KerasTensor]) -> None:
    """
    Helper function to update the output tensors in the network dictionary
    :param layer: The layer to update the output tensors of
    :param network_dict: The network dictionary
    """
    # Set new output tensor (in this case, it will be the same as the original model)
    network_dict[NetworkDictProperities.NEW_OUTPUT_TENSOR_OF.value].update({layer.name: new_output_tensor})
    # Save tensor in output list if it is output in the initial model
    if model.output_names and layer.name in model.output_names and model.name is not _TEMP_MODEL_NAME:
        logger.debug("Layer '%s' added as output layer", layer.name)
        model_outputs.append(new_output_tensor)

def _get_most_recently_added_output_tensor(network_dict: Dict) -> KerasTensor:
    """
    Helper function to get the most recently added output tensor from the network dictionary
    :param network_dict: The network dictionary
    :return: The most recently added output tensor
    """
    return next(reversed(network_dict[NetworkDictProperities.NEW_OUTPUT_TENSOR_OF.value].items()))[-1]


def _handle_subclassed_layer(layer: tf.keras.layers.Layer, layer_input: KerasTensor, network_dict: Dict,
                             class_names: List[str], model_outputs: List[KerasTensor]) -> KerasTensor:
    """
    Helper function to handle subclassed layers
    :param layer: The layer to handle
    :param layer_input: The input layer of the layer
    :param network_dict: The network dictionary
    :param class_names: The list of class names
    :param model_outputs: The list of model outputs
    :return: The output tensor of the layer
    """
    logger.debug("Subclass layer \'%s\' found. Extracting layers.", layer.name)
    # Converts CamelCase to snake_case of subclassed layers class name
    class_names.append(_get_class_names_in_model(layer)[0])

    # Create a model based on the subclassed layer.
    # This is done with the layer input from the network dictionary.
    # 1) The input layer is used to create the temporary functional model
    # 2) The input layer is used in the subclass layers call function as a symbolic tensor to get internal layers
    temp_model = tf.keras.Model(inputs=[layer_input],
                                outputs=layer.call(layer_input, training=False),
                                name=_TEMP_MODEL_NAME)
    logger.debug("Model created for layer %s", layer.name)
    temp_model.summary(print_fn=logger.debug)

    # Get the network dictionary for the temporary model and merge it with the network dictionary for the
    # functional model. This is done so we can keep track of the sublayers and their inputs and outputs.
    temp_model_network_dict = WeightTensorUtils.get_weight_tensor_layer_mapping(temp_model)
    network_dict = WeightTensorUtils.merge_network_dicts(network_dict, temp_model_network_dict)

    _prepare_model_helper(temp_model, class_names=class_names,
                            network_dict=network_dict, model_outputs=model_outputs)

    # The output of the top level subclass layer will be the most recently added output tensor that was added
    # to the network dictionary in the recursive call above. This is used as it's output tensor also, to continue
    # building the functional model. For example, if we have a subclass layer with two convs, that last conv's
    # output tensor will be the output tensor of the subclass layer.
    # Original subclass model flow ------------------------->
    #   ______________        ______________
    #  |             |       |             |
    #  |    Input    |  ---> |  SubClass   |  --------------------------> (Subclass output tensor)
    #  |_____________|       |_____________|
    #
    # Functional model flow ------------------------------>               v Same output tensors ^
    #   ______________        ______________        ______________
    #  |             |       |             |       |             |
    #  |    Input    |  ---> |    Conv     |  ---> |  DepthWise  |  ---> (Subclass output tensor)
    #  |_____________|       |_____________|       |_____________|
    #

    return _get_most_recently_added_output_tensor(network_dict)

def _handle_functional_model(model: Functional, layer_input: KerasTensor, network_dict: Dict) -> \
        Union[KerasTensor, List[KerasTensor]]:
    """
    Helper function to handle functional models. This function will create a new output tensor for the Functional model 
    after it's unwrapped. Since this is just a Funcational model, there could be more custom layers in the model. Therefore,
    we need to recursively call the prepare model helper function to handle these layers.
    :param model: The functional model to handle
    :param layer_input: The input layer of the model
    :param network_dict: The network dictionary
    """
    logger.debug("Functional model \'%s\' found. Extracting layers.", model.name)

    temp_model_network_dict = WeightTensorUtils.get_weight_tensor_layer_mapping(model)
    _set_prepared_models_input_layer(model, temp_model_network_dict, layer_input)
    network_dict = WeightTensorUtils.merge_network_dicts(network_dict, temp_model_network_dict)

    class_names = _get_class_names_in_model(model)

    model_outputs = []
    _prepare_model_helper(model, class_names=class_names,
                          network_dict=network_dict, model_outputs=model_outputs)

    temp_model = tf.keras.Model(inputs=model.input, outputs=model_outputs)
    logger.debug("Functional model %s unwrapped", model.name)
    temp_model.summary(print_fn=logger.debug)

    return temp_model.call(layer_input)

def _handle_normal_keras_layer(layer: tf.keras.layers.Layer, layer_input: KerasTensor) -> KerasTensor:
    """
    Helper function to handle normal keras layers. This function will create a new output tensor for the layer
    and return it.
    :param layer: The layer to create the output tensor for
    :param layer_input: The input tensor to the layer
    :return: The output tensor of the layer
    """
    layer._inbound_nodes = []  # pylint: disable=protected-access
    # Special case for when there is a Lambda opertaion with multiple inputs. For example, x = y + z.
    if isinstance(layer, TFOpLambda) and isinstance(layer_input, List):
        new_output_tensor = layer(*layer_input)
    else:
        new_output_tensor = layer(layer_input)
    layer._outbound_nodes = [] # pylint: disable=protected-access
    return new_output_tensor

def _prepare_model_helper(
        model: tf.keras.Model, class_names: List[str],
        network_dict, model_outputs: List[KerasTensor]) -> tf.keras.layers.Layer:
    """
    Helper function to recursively prepare a model. This function will be recursively called if a subclassed layer is
    found. This function will extract the layers from the subclassed layer and add them to the functional model.
    Otherwise, it will add the layer to the functional model.
    :param model: The model to prepare
    :param class_names: The names of the classes that the original model was subclassed from
    :return: The last layer of the model
    """

    for current_layer in model.layers:
        # Skip input layer
        if isinstance(current_layer, tf.keras.layers.InputLayer):
            continue

        layer_input = _get_layer_input(current_layer, network_dict)

        # If the current layer is a subclassed layer, extract the layers from the subclassed layer
        if _is_subclass_layer(current_layer):
            new_output_tensor = _handle_subclassed_layer(current_layer, layer_input, network_dict, class_names, model_outputs)

        # If a functional model is found, then we unwrap the model and recursively call the prepare model helper
        elif _is_functional_model(current_layer):
            new_output_tensor = _handle_functional_model(current_layer, layer_input, network_dict)

        # If a normally defined layer is found, add it to the functional model
        else:
            new_output_tensor = _handle_normal_keras_layer(current_layer, layer_input)

        _update_output_tensors_in_network_dict(current_layer, new_output_tensor, model, network_dict, model_outputs)

    return

def _get_prepared_model(original_model: tf.keras.Model, input_layer: tf.keras.layers.Layer, class_names: List[str]) \
        -> tf.keras.Model:
    """
    Function to get the prepared model. This function sets up the input layer and calls the helper function to
    recursively prepare the model.
    :param original_model: The original model to prepare
    :param input_layer: The input layer to use for the new model
    :param class_names: The names of the classes that the original model was subclassed from
    :return: The prepared model
    """
    network_dict = WeightTensorUtils.get_weight_tensor_layer_mapping(original_model)
    _set_prepared_models_input_layer(original_model, network_dict, input_layer)

    model_outputs = []
    _prepare_model_helper(original_model, class_names, network_dict, model_outputs)

    # If the model outputs are empty, then we need to get the most recently added output tensor. This is the case
    # when a model might be sparse and not fully connected.
    model_outputs = _get_most_recently_added_output_tensor(network_dict) if not model_outputs else model_outputs
    return tf.keras.Model(inputs=input_layer, outputs=model_outputs)


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

    input_layer = _format_input_layer(original_model, input_layer)

    # Used to fix weight names at end of unwrapping
    # Originally set to the name of the original model's class in the case that their is an inherited model
    class_names = _get_class_names_in_model(original_model)

    prepared_model = _get_prepared_model(original_model, input_layer, class_names)

    # Cloning model to remove any references to the original model
    model_to_return = tf.keras.models.clone_model(functional_model)
    model_to_return, _ = replace_separable_conv_with_depthwise_pointwise(model_to_return)
    logger.debug("Functional model architecture created")
    model_to_return.summary(print_fn=logger.debug)

    # Copying over weights from original model to functional model
    _set_functional_models_weights(original_model, model_to_return, class_names)

    return model_to_return
