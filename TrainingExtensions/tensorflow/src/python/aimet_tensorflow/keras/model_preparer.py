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

from typing import Any, Dict, List, Set, Union
import re
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.engine.base_layer_utils import is_subclassed
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.layers.core import TFOpLambda
from tensorflow.python.keras.layers.merge import _Merge as MergeLayersParentClass
import tensorflow.keras.backend as K

from aimet_tensorflow.keras.utils.model_connection_utils import ModelLayerConnections, ModelLayerConnectionsProperties

from aimet_tensorflow.keras.utils.model_transform_utils import replace_separable_conv_with_depthwise_pointwise
from aimet_common.utils import AimetLogger
_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ModelPreparer)

regex_for_camel_case_to_snake_case = re.compile(r'(?<!^)(?=[A-Z])')
_TEMP_MODEL_NAME = "temp_aimet_intermediate_model"


def _get_original_models_weights_in_functional_model_order(original_model: tf.keras.Model,
                                                           functional_model: tf.keras.Model,
                                                           class_names: Set[str]) -> np.ndarray:
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


def _set_functional_models_weights(
        original_model: tf.keras.Model, functional_model: tf.keras.Model, class_names: Set[str]):
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
        _logger.error(
            "Could not copy weights from original model to functional model. This can occur when "
            "custom sublayers are defined not in the same order as the sublayers call method. Please ensure that the "
            "sublayers internal layers are defined in the same order as the sublayers call method.")
        raise

    _logger.debug("Functional model weights copied.")
    _logger.info("Model prepared for AIMET in Functional API format.")


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
        _logger.info("Input layer not found. Using input layer passed in.")
        if input_layer is None:
            raise ValueError("The top layer of this model is subclassed. Please provide an input layer via the "
                             "\'input_layer\' parameter.")

    return input_layer


def _get_class_names_in_model(model: tf.keras.Model) -> Set[str]:
    """
    Helper function to get the class name for a nested layer.
    :param model: the 'layer' or 'model' to get the class name
    :return: The class name
    """
    return {regex_for_camel_case_to_snake_case.sub("_", model.__class__.__name__).lower()}


def _is_nested_layer(layer: tf.keras.layers.Layer) -> bool:
    """
    Checks if the given layer is a nested layer
    :param layer: The layer to check
    :return: True if the layer is a nested layer, False otherwise
    """
    return is_subclassed(layer) or _is_functional_model(layer) or _is_sequential_model(layer)


def _is_functional_model(layer: tf.keras.layers.Layer) -> bool:
    """
    Checks if the given layer is a functional layer
    :param layer: The layer to check
    :return: True if the layer is a functional layer, False otherwise
    """
    return isinstance(layer, Functional) and not isinstance(layer, tf.keras.Sequential)


def _is_sequential_model(layer: tf.keras.layers.Layer) -> bool:
    """
    Checks if the given layer is a sequential layer
    :param layer: The layer to check
    :return: True if the layer is a sequential layer, False otherwise
    """
    return isinstance(layer, tf.keras.Sequential)


def _set_prepared_models_input_layer(
        model: tf.keras.Model, model_layers_connections, input_layer: tf.keras.layers.InputLayer):
    """
    This function sets the input layer of the model to the input layer of the functional model
    :param model: The original model
    :param model_layers_connections: The model layers connections dictionary
    """
    try:
        if isinstance(model.input, list):
            for inp in model.input:
                model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].update({inp.name: inp})
        else:
            model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].update(
                {model.input.name: model.input})
    except AttributeError:
        # For models that are not connected
        _logger.info("Model is not connected. Setting input layer to input layer passed in.")
        model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].update(
            {model.layers[0].name: [input_layer.name]})
        model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].update(
            {input_layer.name: input_layer})


def _get_layer_input(layer: tf.keras.layers.Layer, model_layers_connections: ModelLayerConnectionsProperties.TYPE) -> tf.keras.layers.Layer:
    """
    Helper function to get the input layer of a layer.
    :param layer: The layer to get the input layer of
    :param model_layers_connections: The model layers connections dict
    :return: The input layer of the layer
    """
    try:
        layer_input = [model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS][layer_aux]
                       for layer_aux in model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES][layer.name]]

        if len(layer_input) == 1:
            layer_input = layer_input[0]
    except KeyError:
        layer_input = _get_most_recently_added_output_tensor(model_layers_connections)
        model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].update({layer.name: [layer_input.name]})
        _logger.warning(
            "Could not find input tensor for layer: %s. Using %s as input, the most recent output tensor.", layer.name,
            layer_input.name)

    return layer_input


def _is_keras_tensor_input(arg: Any) -> bool:
    """
    Helper function to check if some argument is a valid Keras tensor
    :param arg: The argument in question
    :return: True if it is valid, False if not
    """
    if arg is not None:
        if isinstance(arg, List):
            return all(isinstance(x, KerasTensor) for x in arg) or all(isinstance(x, tf.Tensor) for x in arg)
        return isinstance(arg, (KerasTensor, tf.Tensor))
    return False


def _get_call_args(
        layer: tf.keras.layers.Layer,
        model_layers_connections: ModelLayerConnectionsProperties.TYPE) -> List[Union[KerasTensor, List[KerasTensor], Any]]:
    """
    Helper function to get the call arguments of a layer.
    :param layer: The layer to get the call arguments of
    :param model_layers_connections: The model layers connections dict
    :return: The call arguments of the layer
    """
    def _is_tf_tensor(arg: Any) -> bool:
        return isinstance(arg, tf.Tensor)

    try:
        original_call_args = model_layers_connections[ModelLayerConnectionsProperties.CALL_ARGS][layer.name]
    except KeyError:
        _logger.warning("Could not find call args for layer: '%s'. Using keras tensor only as input.", layer.name)
        return [_get_layer_input(layer, model_layers_connections)]

    call_args = []
    keras_tensor_found_flag = False
    for arg in original_call_args:
        if _is_keras_tensor_input(arg):
            if keras_tensor_found_flag:
                if _is_tf_tensor(arg):
                    call_args.append(arg)
                continue
            layer_input = _get_layer_input(layer, model_layers_connections)
            if isinstance(layer_input, list):
                call_args.extend(layer_input)
            else:
                call_args.append(layer_input)
            keras_tensor_found_flag = True
        else:
            call_args.append(arg)

    assert keras_tensor_found_flag, "No keras tensor found in call args of layer {}".format(layer.name)
    return call_args


def _get_call_kwargs(
        layer: tf.keras.layers.Layer,
        model_layers_connections: ModelLayerConnectionsProperties.TYPE) -> List[Union[KerasTensor, List[KerasTensor], Any]]:
    """
    Helper function to get call keyword arguments for a given layer.
    :param layer: The layer to get the call keyword arguments of
    :param model_layers_connections: The model layers connections dict
    :return: The call keyword arguments of the layer
    """
    if original_call_kwargs := model_layers_connections[ModelLayerConnectionsProperties.CALL_KWARGS][layer.name]:
        call_kwargs = {}
        for key, value in original_call_kwargs.items():
            # The Keras tensor is already in the call args, so we don't need to add it again. call_kwargs are for
            # keyword arguments that are not Keras tensors such as 'axis', 'training', etc.
            if _is_keras_tensor_input(value):
                continue
            else:
                call_kwargs[key] = value
    else:
        _logger.debug("No kwargs for layer: '%s'", layer.name)
        return {}
    return call_kwargs


def _update_output_tensors_in_model_layers_connections(
        layer: tf.keras.layers.Layer, new_output_tensor: KerasTensor, model: tf.keras.Model,
        model_layers_connections: ModelLayerConnectionsProperties.TYPE, model_outputs: List[KerasTensor]) -> None:
    """
    Helper function to update the output tensors in the model layers connections dictionary
    :param layer: The layer to update the output tensors of
    :param new_output_tensor: The new output tensor to update with
    :param model: The model currently being checked. Used to add model ouputs
    :param model_layers_connections: The model layers connections dictionary
    :param model_outputs: A list tracking the currently being checked models output tensors
    """
    if layer.name != new_output_tensor.name:
        new_name = new_output_tensor.name
        old_name_of_inputs = model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].pop(
            layer.name)
        model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].update(
            {new_name: old_name_of_inputs})

        # replace values in model_layers_connections[NetworkDictProperties.INBOUND_NODES] with new_name
        for value in model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES].values():
            if layer.name in value:
                idx = value.index(layer.name)
                value[idx] = new_name

        model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].update({new_name: new_output_tensor})
    else:
        # Set new output tensor (in this case, it will be the same as the original model)
        model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].update(
            {layer.name: new_output_tensor})
    # Save tensor in output list if it is output in the initial model
    if model.output_names and layer.name in model.output_names and model.name is not _TEMP_MODEL_NAME:
        _logger.debug("Layer '%s' added as output layer", layer.name)
        model_outputs.append(new_output_tensor)


def _get_most_recently_added_output_tensor(model_layers_connections: ModelLayerConnectionsProperties.TYPE) -> KerasTensor:
    """
    Helper function to get the most recently added output tensor from the model layers connections
    :param model_layers_connections: The model layers connections dictionary
    :return: The most recently added output tensor
    """
    return next(reversed(model_layers_connections[ModelLayerConnectionsProperties.OUTPUT_TENSORS].items()))[-1]


def _get_temporary_model(layer: tf.keras.layers.Layer, layer_input: tf.keras.layers.Layer) -> tf.keras.Model:
    """
    Helper function to create a temporary functional model from a layer
    :param layer: The layer to create the temporary model from
    :param layer_input: The input layer of the layer
    :return: The temporary model
    """
    temp_input = tf.keras.layers.Input(shape=layer_input.shape[1:], name=layer_input.name + "_temp_input")
    temp_model = tf.keras.Model(inputs=[temp_input],
                                outputs=layer.call(temp_input, training=False),
                                name=_TEMP_MODEL_NAME)
    _logger.debug("Model created for layer '%s'", layer.name)
    temp_model.summary(print_fn=_logger.debug)

    return temp_model


def _update_temporary_model_layers_connections_inbound_nodes(
        temp_model_model_layers_connections: ModelLayerConnectionsProperties.TYPE, temp_model: tf.keras.Model,
        layer_input: tf.keras.layers.Layer):
    """
    Helper function to update the input tensors of the temporary model layers connections dictionary
    :param temp_model_model_layer_connections: The temporary model layers connections dictionary
    :param temp_model: The temporary model
    :param layer_input: The input layer of the layer
    """
    for layers_name, input_tensor_name in temp_model_model_layers_connections[
            ModelLayerConnectionsProperties.INBOUND_NODES].items():
        for idx, current_input_name in enumerate(input_tensor_name):
            if current_input_name == temp_model.input.name:
                temp_model_model_layers_connections[ModelLayerConnectionsProperties.INBOUND_NODES][layers_name][
                    idx] = layer_input.name


def _handle_nested_layer(
        layer: tf.keras.layers.Layer, model_layers_connections: Dict, class_names: Set[str],
        model_outputs: List[KerasTensor]) -> KerasTensor:
    """
    Helper function to handle subclassed layers
    :param layer: The layer to handle
    :param model_layers_connections: The model layers connections dictionary
    :param class_names: The list of class names
    :param model_outputs: The list of model outputs
    :return: The output tensor of the layer
    """
    _logger.debug("Subclass layer '%s' found. Extracting layers.", layer.name)
    # Converts CamelCase to snake_case of subclassed layers class name
    class_names.update(_get_class_names_in_model(layer))

    # Create a model based on the subclassed layer.
    # This is done with the layer input from the model layers connections dictionary.
    # 1) The input layer is used to create the temporary functional model
    # 2) The input layer is used in the subclass layers call function as a symbolic tensor to get internal layers
    layer_input = _get_layer_input(layer, model_layers_connections)
    if _is_functional_model(layer) or _is_sequential_model(layer):
        # Unwrapping the functional or sequential model with an extra step of cloning the model to remove external nodes
        temp_model = tf.keras.models.clone_model(_get_temporary_model(layer, layer_input))
    else:
        temp_model = _get_temporary_model(layer, layer_input)

    # Get the model layers connections dictionary for the temporary model and merge it with the model layers connections dictionary for the
    # functional model. This is done so we can keep track of the sublayers and their inputs and outputs.
    temp_model_model_layers_connections = ModelLayerConnections.get_model_layers_connection_properties(temp_model)
    _update_temporary_model_layers_connections_inbound_nodes(
        temp_model_model_layers_connections, temp_model, layer_input)

    model_layers_connections = ModelLayerConnections.merge_model_layers_connections(
        model_layers_connections, temp_model_model_layers_connections)

    return _prepare_model_helper(temp_model, class_names=class_names,
                                 model_layers_connections=model_layers_connections, model_outputs=model_outputs)


def _handle_normal_keras_layer(layer: tf.keras.layers.Layer,
                               model_layers_connections: ModelLayerConnectionsProperties.TYPE) -> KerasTensor:
    """
    Helper function to handle normal keras layers. This function will create a new output tensor for the layer
    and return it.
    :param layer: The layer to create the output tensor for
    :param model_layers_connections: The models layer connections dictionary
    :return: The output tensor of the layer
    """
    call_args = _get_call_args(layer, model_layers_connections)

    if isinstance(layer, TFOpLambda):
        if call_kwargs := _get_call_kwargs(layer, model_layers_connections):
            # Special case for 'tf.concat' that takes a list of inputs with kwargs attached
            # may need to update in the future...
            if 'concat' in layer.name:
                new_output_tensor = layer.call([*call_args], **call_kwargs)
            else:
                new_output_tensor = layer.call(*call_args, **call_kwargs)
        else:
            new_output_tensor = layer.call(*call_args)
    # Special case for "Merge" layers that take a list of inputs such as "tf.keras.layers.Concatenate" and "tf.keras.layers.Add"
    elif isinstance(layer, MergeLayersParentClass):
        new_output_tensor = layer(call_args)
    else:
        new_output_tensor = layer(*call_args)

    return new_output_tensor


def _prepare_model_helper(
        model: tf.keras.Model, class_names: Set[str],
        model_layers_connections, model_outputs: List[KerasTensor]) -> tf.keras.layers.Layer:
    """
    Helper function to recursively prepare a model. This function will be recursively called if a subclassed layer is
    found. This function will extract the layers from the subclassed layer and add them to the functional model.
    Otherwise, it will add the layer to the functional model.
    :param model: The model to prepare
    :param class_names: The names of the classes that the original model was subclassed from
    :param model_layers_connections: The model layers connections dict
    :param model_outputs: The list tracking the models output tensors
    :return: The last layer of the model
    """
    for current_layer in model.layers:
        _logger.debug("Processing layer: '%s'", current_layer.name)
        # Skip input layers
        if isinstance(current_layer, tf.keras.layers.InputLayer):
            continue

        # If the current layer either a subclassed layer, sequential model or functional model, we need to extract the
        # layers from the subclassed layer and add them to the functional model.
        if _is_nested_layer(current_layer):
            new_output_tensor = _handle_nested_layer(
                current_layer, model_layers_connections, class_names, model_outputs)
        else:
            new_output_tensor = _handle_normal_keras_layer(current_layer, model_layers_connections)

        _update_output_tensors_in_model_layers_connections(
            current_layer, new_output_tensor, model, model_layers_connections, model_outputs)

    return new_output_tensor


def _get_prepared_model(original_model: tf.keras.Model, input_layer: tf.keras.layers.Layer, class_names: Set[str]) \
        -> tf.keras.Model:
    """
    Function to get the prepared model. This function sets up the input layer and calls the helper function to
    recursively prepare the model.
    :param original_model: The original model to prepare
    :param input_layer: The input layer to use for the new model
    :param class_names: The names of the classes that the original model was subclassed from
    :return: The prepared model
    """
    model_layers_connections = ModelLayerConnections.get_model_layers_connection_properties(original_model)
    _set_prepared_models_input_layer(original_model, model_layers_connections, input_layer)

    model_outputs = []
    _prepare_model_helper(original_model, class_names, model_layers_connections, model_outputs)

    # If the model outputs are empty, then we need to get the most recently added output tensor. This is the case
    # when a model might be sparse and not fully connected or when a Functional model is inside an inherited model.
    if not model_outputs:
        _logger.warning("No model outputs found. This usually occurs when a models is made by inheritting from "
                        "'tf.keras.Model' and placing a Functional model inside. "
                        "Using most recently added output tensor as prepared models output.")
        model_outputs = _get_most_recently_added_output_tensor(model_layers_connections)

    return tf.keras.Model(inputs=input_layer, outputs=model_outputs)


def _model_has_nested_layers(model: tf.keras.Model) -> bool:
    """
    Helper function to check if a model is needed to be prepared or not based on if the model has nested layers such as
    subclass, functional, or sequential.
    :param model: The model to check
    :return: If the model needs to be prepared or not
    """
    for layer in model.layers:
        if _is_nested_layer(layer):
            return True
    return False


def prepare_model(original_model: tf.keras.Model,
                  input_layer: Union[tf.keras.layers.InputLayer, List[tf.keras.layers.InputLayer]] = None) -> tf.keras.Model:
    """
    This function prepares a Keras model before continuing on with AIMET. Specifically, it will convert the model into
    a purely Functional API model and copy over the original models weights.
    :param original_model: The original model to be prepared
    :param input_layer: The input layer to be used for the new model. By default, the input layer is set to None. If the
    beginning portion of the model is subclassed, then the input layer must be passed in.
    :return: The prepared model if needed, or the original model
    """

    # Initial check to see if preparing model is necessary
    if not _model_has_nested_layers(original_model):
        _logger.debug("Model does not contain any nested layers. Original model being returned.")
        return original_model

    _logger.debug("Preparing model for AIMET. Original model architecture")
    original_model.summary(print_fn=_logger.debug)
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
    _logger.debug("Final class_names: %s", class_names)
    _set_functional_models_weights(original_model, model_to_return, class_names)

    return model_to_return
