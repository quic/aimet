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

""" Common Utilities for tf 2 keras """
import errno
import os
from typing import Union, List, Dict, Tuple, AnyStr
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_from_session_graph
from tensorflow.python.framework.graph_util_impl import remove_training_nodes


from aimet_common.utils import AimetLogger
from aimet_tensorflow.defs import AxisHandling

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

lambda_operators = ['__operators__.add', 'math.multiply', 'math.truediv', 'math.subtract']
per_channel_quantizeable_layers = (tf.keras.layers.Conv2D, tf.keras.layers.Conv2DTranspose,
                                   tf.keras.layers.DepthwiseConv2D, tf.keras.layers.SeparableConv2D,
                                   tf.keras.layers.Dense)

SUBMODULES_TO_SKIP = (tf.keras.layers.MultiHeadAttention,)


def is_lambda_operator(layer: tf.keras.layers.Layer) -> bool:
    """
    Check if layer is a known lambda operator (ex. when '+', '-', '*', '/' are used in the model definition)
    :param layer: Layer to check
    :return True if layer is a known lambda operator, False otherwise
    """
    config = layer.get_config()
    if 'function' in config:
        return config['function'] in lambda_operators
    return False


def module_to_name_map(cur_layer: (tf.keras.Model, tf.keras.layers.Layer)) \
        -> Dict[tf.keras.layers.Layer, Tuple[tf.keras.Model, str]]:
    """
    To find a variable name and parent reference of one module.

    :param cur_layer: model to obtain module_to_name_map
    :return: dictionary includes module_ref as a key, parent_ref and module_name as value
    """

    ref_name = {}
    # pylint: disable=protected-access
    for inner_layer in cur_layer._layers:
        if not isinstance(inner_layer, tf.keras.layers.Layer):
            continue
        if inner_layer.submodules:
            ref_name.update(module_to_name_map(inner_layer))
        else:
            for key, element in vars(cur_layer).items():
                if isinstance(element, tf.keras.layers.Layer) and element == inner_layer:
                    ref_name[element] = (cur_layer, key)

    return ref_name


def find_input_layers(node_layer_map: Dict) -> List[tf.keras.layers.InputLayer]:
    """
    helper to find the input layers of the model.

    :param node_layer_map: dictionary includes node_ref as a key, in_layers and out_layer as value
    :return: return list of input layers
    """

    input_layers = []

    for value in node_layer_map.values():
        if value[0] is None:
            input_layers.append(value[1])

    return input_layers


def _find_last_layers(cur_layer: (tf.keras.Model, tf.keras.layers.Layer)) \
        -> List[tf.keras.layers.Layer]:
    """
    helper to find the last layers of the model.

    :param cur_layer: model to obtain last layers
    :return: return list of last layers in the model
    """

    last_layers = []
    # pylint: disable=protected-access
    for inner_layer in cur_layer._layers:
        if not isinstance(inner_layer, tf.keras.layers.Layer):
            continue
        if inner_layer.outbound_nodes == []:
            if inner_layer.submodules:
                last_layers.extend(_find_last_layers(inner_layer))
            else:
                last_layers.append(inner_layer)

    return last_layers


def create_layer_to_out_node_map(cur_layer: (tf.keras.Model, tf.keras.layers.Layer)) -> Dict:
    """
    To find the outbound nodes of one layer.

    :param cur_layer: model to obtain layer_to_out_node_map
    :return: return dictionary includes layer_ref as a key, outbound nodes as value
    """

    layer_node_map = {}
    node_layer_map = create_node_to_layer_map(cur_layer)

    for node, (in_layers, _) in node_layer_map.items():
        if in_layers:
            for in_layer in in_layers:
                if in_layer not in layer_node_map:
                    layer_node_map[in_layer] = [node]
                else:
                    layer_node_map[in_layer].append(node)

    return layer_node_map


def _submodule_handler_node_to_layer_map(
        cur_layer: (tf.keras.Model, tf.keras.layers.Layer),
        node_layer_map: Dict) -> (Dict, str, List):
    """
    The utility to extract node_layer_map for the cur_layer submodule and
    provide the connectivity with the outer model.

    :param cur_layer: model to obtain node_layer_ref for
    :param node_layer_map: dictionary of node_layer_ref includes an item with submodule ref as
             first value index or first and second value index
    :return: return dictionary includes node_ref as a key,
        in_layers and out_layer as value for cur_layer,
        inbound node of the input layer and list of outbound nodes after input layer
    """

    # pylint: disable=too-many-locals

    # im stands for inner model
    im_node_layer_map = create_node_to_layer_map(cur_layer)

    im_input_layer = None
    im_node_input = None
    im_nodes_after_input_layer = []
    im_input_layer_succeeding_layers = []

    for node, (in_layers, out_layer) in node_layer_map.items():
        if out_layer == cur_layer:
            # iterating through inner model node_layer_map dict to find input_layer
            # and its inbound_node
            for im_node, (im_in_layers, im_out_layer) in im_node_layer_map.items():
                if im_in_layers is None:
                    im_input_layer = im_out_layer
                    im_node_input = im_node
            # iterating through inner model node_layer_map dict to find input layer
            # outbound nodes and its succeeding layers
            for im_node, (im_in_layers, im_out_layer) in im_node_layer_map.items():
                if im_in_layers == [im_input_layer]:
                    im_input_layer_succeeding_layers.append(im_out_layer)
                    im_nodes_after_input_layer.append(im_node)
            # If there are more than one layer which input layer goes to in inner model,
            # we need to build a connection between incoming layer of cur layer
            # and inner model input layer (branch will not be considered in pattern)
            if len(im_input_layer_succeeding_layers) > 1:
                node_layer_map.update({node: [in_layers, im_input_layer]})
            # otherwise we need to build a connection between incoming layer of cur layer
            # and succeeding layers to inner model input layer
            elif len(im_input_layer_succeeding_layers) == 1:
                node_layer_map.update({node: [in_layers, im_input_layer_succeeding_layers[0]]})
        elif in_layers and cur_layer in in_layers:
            im_last_layers = _find_last_layers(cur_layer)
            node_layer_map.update({node: [im_last_layers, out_layer]})

    return im_node_layer_map, im_node_input, im_nodes_after_input_layer


def create_node_to_layer_map(cur_layer: (tf.keras.Model, tf.keras.layers.Layer)) -> Dict:
    """
    To find the input layers and output layer of one node.

    :param cur_layer: model to obtain node_to_layer_map
    :return: return dictionary includes node_ref as a key, in_layers and out_layer as value
    """

    node_layer_map = {}
    # pylint: disable=protected-access
    for inner_layer in cur_layer._layers:
        if not isinstance(inner_layer, tf.keras.layers.Layer):
            continue
        for out_node in inner_layer.outbound_nodes:
            if out_node in node_layer_map:
                node_layer_map[out_node][0].append(inner_layer)
            else:
                node_layer_map[out_node] = [[inner_layer], None]
        for in_node in inner_layer.inbound_nodes:
            if in_node in node_layer_map:
                node_layer_map[in_node][1] = inner_layer
            else:
                node_layer_map[in_node] = [None, inner_layer]

        # If the layer has submodules we try to find out the input/output map for those layers as well.
        # But in case of few layers having submodules, internal connection for them is not present. It means
        # inbound and outbound nodes for the submodules are empty so, we treat them as a single layer only.
        if inner_layer.submodules and not isinstance(inner_layer, SUBMODULES_TO_SKIP):
            im_node_layer_map, im_node_input, im_nodes_after_input_layer = \
                _submodule_handler_node_to_layer_map(inner_layer, node_layer_map)
            if len(im_nodes_after_input_layer) == 1:
                del im_node_layer_map[im_nodes_after_input_layer[0]]
            del im_node_layer_map[im_node_input]

            node_layer_map.update(im_node_layer_map)

    return node_layer_map


def replace_layer_in_functional_model(model: tf.keras.Model, old_layer: tf.keras.layers.Layer,
                                      new_layers: Union[List, tf.keras.layers.Layer]):
    """
    Replace a layer in a model with a list of new layers to be called in sequential order.
    :param model: Model containing layer to replace
    :param old_layer: Layer to replace
    :param new_layers: Layer or list of new layers to insert into the model
    """
    # pylint: disable=protected-access
    if old_layer in model._input_layers:
        _logger.error('Replacement for input layers not currently supported')
        raise NotImplementedError('Replacement for input layers not currently supported')

    if len(old_layer.inbound_nodes) > 1:
        _logger.error('Replacement for layers used multiple times not currently supported')
        raise NotImplementedError('Replacement for layers used multiple times not currently supported')

    if not isinstance(new_layers, list):
        new_layers = [new_layers]

    residing_model = _get_residing_model_of_layer(model, old_layer)

    # Find layers before and after the old layer to replace
    # Need to use keras_inputs instead of input_layers due to bug where only one input layer will be shown
    parent_layers = [keras_input._keras_history.layer for keras_input in old_layer.inbound_nodes[0].keras_inputs]
    following_layers_and_inputs_dict = _get_following_layers_and_inputs(residing_model, old_layer)

    # Clear out certain outbound nodes of parent layers, inbound node of old layer, outbound nodes of old layer, and
    # certain inbound nodes of following layers.
    _clear_inbound_and_outbound_nodes(old_layer, parent_layers, following_layers_and_inputs_dict)

    # Remove network nodes from the residing model for old_layer and following layers
    _remove_network_nodes(residing_model, [old_layer] + list(following_layers_and_inputs_dict.keys()))

    # Call new layers in sequence to create new nodes
    out_tensor = _call_new_layers_sequentially(parent_layers, new_layers)

    # Connect the last new layer back to old child layer, creating a new node in the process
    if following_layers_and_inputs_dict:
        _link_following_layers_to_new_layer_output(out_tensor, following_layers_and_inputs_dict, old_layer)
    else:
        _update_model_output_info(residing_model, old_layer, out_tensor)

    # Update model's layers and network nodes
    residing_model._insert_layers(new_layers + list(following_layers_and_inputs_dict.keys()))


def _get_residing_model_of_layer(model: tf.keras.Model, layer: tf.keras.layers.Layer) -> Union[tf.keras.Model,
                                                                                               None]:
    """
    Get the model in which the layer resides, given a top level model. The residing model could be the topmost level
    model itself, or a submodel inside the topmost model.
    :param model: Top level model
    :param layer: Layer to get residing model of
    return: Residing model of layer
    """
    if layer in model.layers:
        return model

    # Iterate through layers of current model level and recursively call into any layer that is a model
    for inner_layer in model.layers:
        if isinstance(inner_layer, tf.keras.models.Model):
            residing_model = _get_residing_model_of_layer(inner_layer, layer)
            if residing_model is not None:
                return residing_model
    return None


def _get_following_layers_and_inputs(model: tf.keras.Model, layer: tf.keras.layers.Layer) -> \
        Dict[tf.keras.layers.Layer, List]:
    """
    Get a dictionary mapping following layers of the given layer to the keras inputs for the following layer.
    :param model: Model containing layer
    :param layer: Layer to get following layers for
    :return: Dictionary mapping following layers of the given layer to the keras inputs for the following layer
    """
    # pylint: disable=protected-access
    following_layers_and_inputs = {}
    if layer.outbound_nodes:
        for outbound_node in layer.outbound_nodes:
            following_layer = outbound_node.outbound_layer
            following_layers_and_inputs[following_layer] = []
            # Find all inputs of following_layer, of which the old layer will be one
            for keras_input in outbound_node.keras_inputs:
                following_layers_and_inputs[following_layer].append(keras_input)
    else:
        assert layer in model._output_layers
    return following_layers_and_inputs


def _remove_network_nodes(model: tf.keras.Model, layers: List[tf.keras.layers.Layer]):
    """
    Remove network nodes in the model corresponding to the given layers
    :param model: Model to remove network nodes from
    :param layers: List of layers corresponding to network nodes to remove
    """
    # pylint: disable=protected-access
    for layer in layers:
        node_key = layer.name + '_ib-0'
        model._network_nodes.remove(node_key)


def _call_new_layers_sequentially(parent_layers: List[tf.keras.layers.Layer],
                                  new_layers: List[tf.keras.layers.Layer]) -> tf.Tensor:
    """
    Call new layers sequentially to create nodes, starting from parent layers.
    :param parent_layers: Parent layers to start building new nodes from
    :param new_layers: New layers to be called sequentially
    :return: Output tensor from final new layer
    """
    # pylint: disable=protected-access
    # Use output tensors from parent layers to create new nodes
    # Assumption is that the order in which keras inputs showed up is the same order to call into new layers.
    curr_tensor = []
    for parent_layer in parent_layers:
        curr_tensor.append(parent_layer.output)

    # Flatten inputs if there is only one
    if len(curr_tensor) == 1:
        curr_tensor = curr_tensor[0]

    for layer in new_layers:
        curr_tensor = layer(curr_tensor)
    return curr_tensor


def _clear_inbound_and_outbound_nodes(layer_of_interest: tf.keras.layers.Layer,
                                      parent_layers: List[tf.keras.layers.Layer],
                                      following_layers_and_inputs_dict: Dict[tf.keras.layers.Layer,
                                                                             List[tf.Tensor]]):
    """
    Clear certain outbound nodes of parent layers, inbound node of layer_of_interest, outbound nodes of
    layer_of_interest, and inbound nodes of following layers.
    :param layer_of_interest: Layer to remove inbound and outbound nodes of
    :param parent_layers: Parent layers to remove outbound nodes of
    :param following_layers_and_inputs_dict: Dictionary containing following layers to remove inbound nodes of
    """
    # pylint: disable=protected-access
    # Clear out inbound node of layer_of_interest
    old_inbound_node = layer_of_interest.inbound_nodes[0]
    layer_of_interest._inbound_nodes.remove(old_inbound_node)

    # Clear out the inbound node from the outbound nodes lists of parent layers
    for parent_layer in parent_layers:
        parent_layer._outbound_nodes.remove(old_inbound_node)

    # Clear out old outbound node of old layer and corresponding inbound node of following layers.
    # Following layers may have other inputs as well; for those inputs, the correct outbound node must be removed too.
    # pylint: disable=too-many-nested-blocks
    if layer_of_interest.outbound_nodes:
        for outbound_node in layer_of_interest.outbound_nodes:
            # For all following layers of old layer, clear out inbound nodes corresponding to the old layer
            for following_layer in following_layers_and_inputs_dict.keys():
                if outbound_node in following_layer._inbound_nodes:
                    following_layer._inbound_nodes.remove(outbound_node)

                    # The following layer may have multiple inputs. In this case, since we are recreating the inbound
                    # node using all inputs, we need to clear out the corresponding outbound_node from the other input
                    # layers' outbound_nodes as well.
                    for keras_input in following_layers_and_inputs_dict[following_layer]:
                        # Don't modify outbound node of old_layer yet
                        if keras_input._keras_history.layer != layer_of_interest:
                            keras_input._keras_history.layer._outbound_nodes.remove(outbound_node)

        layer_of_interest._outbound_nodes = []


def _link_following_layers_to_new_layer_output(new_tensor_output: tf.Tensor,
                                               following_layers_and_inputs_dict: Dict[tf.keras.layers.Layer,
                                                                                      List[tf.Tensor]],
                                               replaced_layer: tf.keras.layers.Layer):
    """
    Link following layers to the given tensor.
    :param new_tensor_output: Tensor to link to following layers
    :param following_layers_and_inputs_dict: Dictionary containing following layers to link new tensor to
    :param replaced_layer: Layer that was replaced
    """
    # pylint: disable=protected-access
    for following_layer, keras_inputs in following_layers_and_inputs_dict.items():
        for idx, keras_input in enumerate(keras_inputs):
            if keras_input._keras_history.layer == replaced_layer:
                keras_inputs[idx] = new_tensor_output
        # Flatten list if only one input
        if isinstance(keras_inputs, list):
            keras_inputs = keras_inputs[0]
        _ = following_layer(keras_inputs)


def _update_model_output_info(residing_model: tf.keras.Model, replaced_layer: tf.keras.layers.Layer,
                              new_tensor_output: tf.Tensor):
    """
    Update model output layers, output coordinates, and nested outputs with the new output tensor.
    :param residing_model: Model to update output info for
    :param replaced_layer: Layer that was replaced
    :param new_tensor_output: New output tensor for the model
    """
    # pylint: disable=protected-access
    # Last layer in new_layers will be a new output op
    old_layer_index = residing_model._output_layers.index(replaced_layer)
    del residing_model._output_layers[old_layer_index]
    del residing_model._output_coordinates[old_layer_index]
    if isinstance(residing_model.output, list):
        del residing_model._nested_outputs[old_layer_index]
    else:
        assert old_layer_index == 0

    layer, node_index, tensor_index = new_tensor_output._keras_history  # pylint: disable=protected-access
    residing_model._output_layers.append(layer)
    residing_model._output_coordinates.append((layer, node_index, tensor_index))
    if isinstance(residing_model._nested_outputs, list):
        residing_model._nested_outputs.append(new_tensor_output)
    else:
        residing_model._nested_outputs = new_tensor_output


def parse_activation_layer(
        activation_layer: tf.keras.layers.Activation,
) -> List[str]:
    """
    Parse generic tf.keras.layers.Activation and convert it to corresponding onnx type

    :param activation_layer: tf.keras.layers.Activation
    :return: list of converted onnx type str
    """
    activation_name = tf.keras.activations.serialize(activation_layer.activation)

    keras_to_onnx_dict = {
        "softmax": "Softmax",
        "relu": "Relu",
        "selu": "Selu",
        "tanh": "Tanh",
        "sigmoid": "Sigmoid",
        "leaky_relu": "LeakyRelu",
        "softplus": "Softplus",
        "softsign": "Softsign",
        "elu": "Elu",
    }
    onnx_activation = keras_to_onnx_dict.get(activation_name)
    if onnx_activation is None:
        return ["Unknown"]

    return [onnx_activation]


def get_number_of_outputs_and_axis_handling(layer, weight_shape, param_type) -> Tuple[int, AxisHandling]:
    """
    Get number of output of channels and handling axis for a specific layers
    :param layer: tf.keras.layers.Layer
    :param weight_shape:
    :param param_type: str
    :return: Tuple[int, int]
    """
    axis_handling = AxisHandling.LAST_AXIS
    num_output_channels = weight_shape[-1]

    if isinstance(layer, (tf.keras.layers.Conv1DTranspose,
                          tf.keras.layers.Conv2DTranspose,
                          tf.keras.layers.Conv3DTranspose)) and param_type != 'bias':
        num_output_channels = weight_shape[-2]

    elif isinstance(layer, (tf.keras.layers.DepthwiseConv2D,
                            tf.keras.layers.SeparableConv2D)) and param_type != 'bias':
        num_output_channels *= weight_shape[-2]
        axis_handling = AxisHandling.LAST_TWO_AXES

    return num_output_channels, axis_handling


def log_param_quantizer_wrapper_details(layer, axis_handling=None, num_output_channels=None):
    """ Logging statements to which Keras Layers are wrapped in Param Quantizers """
    if axis_handling is None and num_output_channels is None:
        _logger.debug("%s to be wrapped in ParamPerTENSORQuantizer\n", layer)
    else:
        _logger.debug("%s to be wrapped in ParamPerCHANNELQuantizer\n"
                      "Axis: %d\n"
                      "Number of Output Channels: %d", layer,
                      axis_handling, num_output_channels)


def convert_h5_model_to_pb_model(h5_model_path: AnyStr, custom_objects: Dict = None):
    """
    This utility function converts a h5_model from Keras into a frozen pb for consumption by SNPE/QNN
    :param h5_model_path: Path to the saved h5 Keras Model
    :param custom_objects: If there are custom objects to load, Keras needs a dict of them to map them
    """

    # Function for validating if the file exist and is a h5
    def validate_model_path() -> Tuple[str, str]:
        if not os.path.exists(h5_model_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), h5_model_path)

        model_name_split = os.path.basename(h5_model_path).split('.')
        if model_name_split[1] != 'h5':
            raise ValueError("File must be a h5 model.")

        model_name = model_name_split[0] + '_converted.pb'
        save_path = os.path.dirname(h5_model_path)

        return model_name, save_path if save_path else os.getcwd()

    def freeze_session(session, output_names):
        graph = session.graph
        with graph.as_default():
            output_names += [v.op.name for v in tf.compat.v1.global_variables()]
            input_graph_def = graph.as_graph_def()

            # Unset all nodes device
            for node in input_graph_def.node:
                node.device = ""

            # Take session and output names to a frozen graph. Also converting training specific ops
            # to testing ops i.e. Identities
            frozen_graph = convert_variables_to_constants_from_session_graph(
                session, input_graph_def, output_names)
            frozen_graph = remove_training_nodes(frozen_graph)
        return frozen_graph

    model_name, save_path = validate_model_path()
    with tf.compat.v1.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            # Grab the session and set the learning phase to test to remove training nodes
            tf.compat.v1.keras.backend.get_session(sess)
            tf.compat.v1.keras.backend.set_learning_phase(0)

            # Try and load model. If there are custom objects, then user is logged how to pass custom objects and
            # raises again with the stacktrace.
            try:
                model = tf.keras.models.load_model(h5_model_path,
                                                   custom_objects=custom_objects,
                                                   compile=False)
            except ValueError:
                _logger.error("If using custom layers, pass a dict mapping them. "
                              "For example, {'CustomLayer': CustomLayer}")
                raise

            frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(),
                                          [out.op.name for out in model.outputs])
            tf.io.write_graph(frozen_graph, save_path, model_name, as_text=False)

            _logger.info("Success. The converted model is located at %s saved as %s", save_path, model_name)
