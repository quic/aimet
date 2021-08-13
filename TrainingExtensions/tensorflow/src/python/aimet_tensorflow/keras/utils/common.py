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

from typing import Dict, List

import tensorflow as tf


def module_to_name_map(cur_layer: tf.keras.Model) -> Dict[tf.keras.layers.Layer, List[str]]:
    """
    To find a variable name and parent reference of one module
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
                    ref_name[element] = [cur_layer, key]

    return ref_name

def find_input_layers(node_layer_ref: Dict[str, List[str]]) -> List[str]:
    """
    helper to find the input layers of the model
    :param node_layer_ref: dictionary includes node_ref as a key, in_layers and out_layer as value
    :return: return list of input layers
    """

    input_layers = []

    for value in node_layer_ref.values():
        if value[0] is None:
            input_layers.append(value[1])

    return input_layers

def find_last_layers(cur_layer: tf.keras.Model):
    """
    helper to find the last layers of the model
    :param cur_layer: model to obtain last layers
    :return: return list of last layers
    """
    last_layers = []
    # pylint: disable=protected-access
    for inner_layer in cur_layer._layers:
        if inner_layer.outbound_nodes == []:
            if inner_layer.submodules:
                last_layers += find_last_layers(inner_layer)
            else:
                last_layers.append(inner_layer)

    return  last_layers

def create_layer_to_out_node_map(cur_layer: tf.keras.Model) -> Dict[tf.keras.layers.Layer, str]:
    """
    To find the outbound nodes of one layer
    :param cur_layer: model to obtain layer_to_out_node_map
    :return: dictionary includes layer_ref as a key, outbound nodes as value
    """

    layer_node_ref = {}
    node_layer_ref = create_node_to_layer_map(cur_layer)

    for node, layers in node_layer_ref.items():
        if layers[0]:
            for in_layer in layers[0]:
                if in_layer not in layer_node_ref:
                    layer_node_ref[in_layer] = [node]
                else:
                    layer_node_ref[in_layer].append(node)

    return layer_node_ref

def _submodule_handler_node_to_layer_map(cur_layer: tf.keras.Model, node_layer_ref: Dict[str, List[str]]):
    """
    The utility to extract node_layer_map for the cur_layer submodule and provide the connectivity with the outer model
    :param cur_layer: model to obtain node_layer_ref for
    :param node_layer_ref: dictionary of node_layer_ref includes an item with submodule ref as
             first value index or first and second value index
    :return: dictionary includes node_layer_for for this submodules and nodes corresponding
             to input layer and the input layer outbounding nodes
    """

    node_layer_map2 = create_node_to_layer_map(cur_layer)
    input_layer = None
    node_input_layer = None
    node_after_input_layer = None
    for key, value in node_layer_ref.items():
        if value[1] == cur_layer:
            for key1, value1 in node_layer_map2.items():
                if value1[0] is None:
                    input_layer = value1[1]
                    node_input_layer = key1
            for key2, value2 in node_layer_map2.items():
                if value2[0] == [input_layer]:
                    next_layer = value2[1]
                    node_after_input_layer = key2
            node_layer_ref.update({key: [value[0], next_layer]})
        elif value[0] and cur_layer in value[0]:
            last_layers = find_last_layers(cur_layer)
            node_layer_ref.update({key: [last_layers, value[1]]})

    return node_layer_map2, node_input_layer, node_after_input_layer

def create_node_to_layer_map(cur_layer: tf.keras.Model) -> Dict[str, List[str]]:
    """
    To find the input layers and output layer of one node
    :param cur_layer: model to obtain node_to_layer_map
    :return: dictionary includes node_ref as a key, in_layers and out_layer as value
    """

    node_layer_ref = {}
    # pylint: disable=protected-access
    for inner_layer in cur_layer._layers:
        for out_node in inner_layer.outbound_nodes:
            if out_node in node_layer_ref:
                node_layer_ref[out_node][0].append(inner_layer)
            else:
                node_layer_ref[out_node] = [[inner_layer], None]
        for in_node in inner_layer.inbound_nodes:
            if in_node in node_layer_ref:
                node_layer_ref[in_node][1] = inner_layer
            else:
                node_layer_ref[in_node] = [None, inner_layer]
        if inner_layer.submodules:
            node_layer_map2, node_input_layer, node_after_input_layer = _submodule_handler_node_to_layer_map(inner_layer, node_layer_ref)
            del node_layer_map2[node_input_layer]
            del node_layer_map2[node_after_input_layer]
            node_layer_ref.update(node_layer_map2)

    return node_layer_ref
