# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

import typing

import tensorflow as tf
from aimet_common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


def module_to_name_map(cur_layer: (tf.keras.Model, tf.keras.layers.Layer)) \
        -> typing.Dict[tf.keras.layers.Layer, typing.Tuple[tf.keras.Model, str]]:
    """
    To find a variable name and parent reference of one module.

    :param cur_layer: model to obtain module_to_name_map
    :return: dictionary includes module_ref as a key, parent_ref and module_name as value
    """

    ref_name = {}
    # pylint: disable=protected-access
    for inner_layer in cur_layer._layers:
        if inner_layer.submodules:
            ref_name.update(module_to_name_map(inner_layer))
        else:
            for key, element in vars(cur_layer).items():
                if isinstance(element, tf.keras.layers.Layer) and element == inner_layer:
                    ref_name[element] = (cur_layer, key)

    return ref_name


def find_input_layers(node_layer_map: typing.Dict) -> typing.List[tf.keras.layers.InputLayer]:
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
        -> typing.List[tf.keras.layers.Layer]:
    """
    helper to find the last layers of the model.

    :param cur_layer: model to obtain last layers
    :return: return list of last layers in the model
    """

    last_layers = []
    # pylint: disable=protected-access
    for inner_layer in cur_layer._layers:
        if inner_layer.outbound_nodes == []:
            if inner_layer.submodules:
                last_layers.extend(_find_last_layers(inner_layer))
            else:
                last_layers.append(inner_layer)

    return last_layers


def create_layer_to_out_node_map(cur_layer: (tf.keras.Model, tf.keras.layers.Layer)) -> typing.Dict:
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
        node_layer_map: typing.Dict) -> (typing.Dict, str, typing.List):
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


def create_node_to_layer_map(cur_layer: (tf.keras.Model, tf.keras.layers.Layer)) -> typing.Dict:
    """
    To find the input layers and output layer of one node.

    :param cur_layer: model to obtain node_to_layer_map
    :return: return dictionary includes node_ref as a key, in_layers and out_layer as value
    """

    node_layer_map = {}
    # pylint: disable=protected-access
    for inner_layer in cur_layer._layers:
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
        if inner_layer.submodules:
            im_node_layer_map, im_node_input, im_nodes_after_input_layer = \
                _submodule_handler_node_to_layer_map(inner_layer, node_layer_map)
            if len(im_nodes_after_input_layer) == 1:
                del im_node_layer_map[im_nodes_after_input_layer[0]]
            del im_node_layer_map[im_node_input]

            node_layer_map.update(im_node_layer_map)

    return node_layer_map


def replace_layer_for_non_subclassed_model(model: tf.keras.Model, old_layer: tf.keras.layers.Layer,
                                           new_layers: typing.Union[typing.List,
                                                                    tf.keras.layers.Layer]):
    """
    Replace a layer in a model with a list of new layers to be called in sequential order.
    :param model: Model containing layer to replace
    :param old_layer: Layer to replace
    :param new_layers: Layer or list of new layers to insert into the model
    """
    if not isinstance(new_layers, list):
        new_layers = [new_layers]

    if len(old_layer.inbound_nodes) > 1:
        _logger.error('Replacement for layer with multiple inputs not currently supported')
        raise NotImplementedError

    if len(old_layer.outbound_nodes) > 1:
        _logger.error('Replacement for layer with multiple outputs not currently supported')
        raise NotImplementedError

    # pylint: disable=protected-access
    if old_layer in model._input_layers:
        _logger.error('Replacement for input layer not currently supported')
        raise NotImplementedError

    if old_layer in model._output_layers:
        _logger.error('Replacement for output layer not currently supported')
        raise NotImplementedError

    # Find layers before and after the old layer to replace
    parent_layer = old_layer.inbound_nodes[0].inbound_layers
    following_layer = old_layer.outbound_nodes[0].outbound_layer

    # Remove any network nodes to do with any of the old and new layers to replace
    for layer in [old_layer] + new_layers:
        for idx, _ in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(idx)
            if node_key in model._network_nodes:
                model._network_nodes.remove(node_key)

    # Clear out inbound and outbound nodes before and after the old layer to replace
    old_layer._inbound_nodes = []
    old_layer._outbound_nodes = []
    parent_layer._outbound_nodes = []
    following_layer._inbound_nodes = []

    # Use output tensor from parent layer to create new nodes
    curr_tensor = parent_layer.output
    for layer in new_layers:
        curr_tensor = layer(curr_tensor)

    # Connect the last new layer back to old child layer, creating a new node in the process
    _ = following_layer(curr_tensor)

    # Update model's layers and network nodes
    model._insert_layers(new_layers + [following_layer])
