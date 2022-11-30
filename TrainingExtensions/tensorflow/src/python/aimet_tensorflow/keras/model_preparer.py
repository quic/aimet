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

import inspect
from typing import List, Set, Union
import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional

from aimet_common.utils import AimetLogger
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ModelPreparer)


def _get_result_from_conditional(line: str, subclass_layer: tf.keras.layers.Layer):
    """
    Get the result of a conditional statement in a subclassed layer
    :param line: The line of code to evaluate
    :param subclass_layer: The subclassed layer
    :return: The result of the conditional statement
    """

    if 'else' in line:
        return True

    bool_to_eval = line.split('if')[1].split(':')[0]
    for attr in dir(subclass_layer):
        if not attr.startswith('__') and attr in bool_to_eval:
            bool_to_eval = bool_to_eval.replace(f'self.{attr}', f'{getattr(subclass_layer, attr)}')

    return eval(bool_to_eval)  # pylint: disable=eval-used


def _refractor_call_code(call_code: List[str], subclass_layer: tf.keras.layers.Layer) -> List[str]:
    """
    Refractor the sublayer's call code to be more readable
    :param call_code: The code to be cleaned up
    :return: The cleaned up code
    """
    # Remove docstring
    call_code = call_code[1:]
    # Remove any comments
    call_code = [line.split('#')[0] for line in call_code]

    find_conditionals = [(line_number, _get_result_from_conditional(line, subclass_layer))
                         for line_number, line in enumerate(call_code)
                         if any(cond in line for cond in ('if', 'elif', 'else'))]

    # remove index if the conditional is false up until next conditional
    for line_number, result in find_conditionals:
        if not result:
            call_code.pop(line_number)
            for line in call_code[line_number:]:
                call_code.pop(line_number)
                if any(cond in line for cond in ('if', 'elif', 'else', 'return')):
                    break

    return call_code


def _get_layer_call_order_and_validate(subclass_layer: tf.keras.layers.Layer, found_internal_layers: List[str]):
    """
    This function returns the call order of a layer. This is used to determine the order of the layers in the
    Functional API model.
    :param subclass_layer: The layer to get the call order of
    :param found_internal_layers: The list of layers that have been found
    :return: The call order of the layer
    """
    # TODO: Need to check for if conditions. Potentially autograd call will handle for us.
    # TODO: Using layers in computations. i.e. self.layer(x) * 2
    # Using tf.autograph to get the code used when the layer is called
    code_by_line = _refractor_call_code(inspect.getsource(subclass_layer.call).splitlines(), subclass_layer)

    call_order = []
    attr_layer_pattern = "self."
    for line in code_by_line:
        wrapped_layer_index = line.find(attr_layer_pattern) + len(attr_layer_pattern)
        if attr_layer_pattern in line:
            if (num_wrapped_layers_on_line := line.count(attr_layer_pattern)) > 1:
                nested_call_order = []
                for _ in range(num_wrapped_layers_on_line):
                    nested_call_order.append(line[wrapped_layer_index:line.find(",", wrapped_layer_index)])
                    wrapped_layer_index = line.find(attr_layer_pattern, wrapped_layer_index) + len(attr_layer_pattern)
                call_order.extend(nested_call_order[::-1])
            else:
                call_order.append(line[wrapped_layer_index:line.find("(", wrapped_layer_index)])

    assert all(layer in call_order for layer in found_internal_layers), \
        f"""
        Could not parse call order correctly for subclassed layer: \"{subclass_layer.name}\".
        Missing internal layers: {set(call_order) - set(found_internal_layers)}
        For easier parsing, consider updating the call method of subclassed layers to be functional.
        For example:

        def call(self, inputs):
            x = self.layer1(inputs)
            x = self.layer2(x)
            return x
        """
    return call_order


def _handle_sub_layers(sub_layer: tf.keras.layers.Layer, found_sub_layers: Set[tf.keras.layers.Layer],
                       prev_layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    """
    Go through each "Layers" properities, if the layer is subclassed, then we will have the wrapped
    layers as properties that can be extracted over and used to create a functioncal model.
    :param sub_layer: The layer to be handled
    :param found_sub_layers: The set of layers that have been found
    :param prev_layer: The previous layer in the model
    :return: The new layer to be used in the model
    """
    call_order = _get_layer_call_order_and_validate(sub_layer, found_sub_layers.keys())
    for sub_layer_name in call_order:
        prev_layer = found_sub_layers[sub_layer_name](prev_layer)

    return prev_layer


def prepare_model(original_model: tf.keras.Model, input_layer: Union[tf.keras.Input, List[tf.keras.Input]] = None):
    """
    This function prepares a Keras model before continuing on with AIMET. Specifically, it will convert the model into
    a purely Functional API model and copy over the original models weights.
    :param original_model: The original model to be prepared
    :param input_layer: The input layer to be used for the new model. By default, the input layer is set to None. If the
    beginning portion of the model is subclassed, then the input layer must be passed in.
    """

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

    for layer in layers_to_copy:
        # If the layer is another Functional model, we need to take out it's layer minus the input layer
        if isinstance(layer, Functional):
            logger.debug("Functional model found. Extracting layers.")
            for func_layer in layer.layers[1:]:
                prev_layer = func_layer(prev_layer)
            logger.info("Functional model extracted.")

        elif found_sub_layers := {object_name: sub_layer
                                  for object_name, sub_layer in layer.__dict__.items()
                                  if isinstance(sub_layer, tf.keras.layers.Layer)}:
            logger.debug("Subclassed layer %s found. Attempting to convert to Functional API.", layer.name)
            prev_layer = _handle_sub_layers(layer, found_sub_layers, prev_layer)
            logger.info("Subclassed layer %s converted to Functional API.", layer.name)
        else:
            prev_layer = layer(prev_layer)

    functional_model = tf.keras.Model(inputs=input_layer, outputs=prev_layer)
    logger.debug("Functional model architecture created.")

    try:
        functional_model.set_weights(original_model.get_weights())
    except ValueError:
        logger.error(
            "Could not copy weights from original model to functional model. This can occur when "
            "custom sublayers are defined not in the same order as the sublayers call method. Please ensure that the "
            "sublayers internal layers are defined in the same order as the sublayers call method.")
        raise
    logger.debug("Functional model weights copied.")
    logger.info("Model prepared for AIMET in Functional API format.")

    return functional_model
