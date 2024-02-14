# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""
Common base-class for Layer and LayerDatabase. Intent is to create training-framework agnostic abstractions so
the majority of the aimet model compression and quantization code can use this and not be dependent on a given
training framework.
"""

from typing import Tuple
from collections import OrderedDict
import abc


class Conv2dTypeSpecificParams:
    """
    Holds layer parameters specific to Conv2D layers
    """

    def __init__(self, stride: Tuple[int, int], padding: Tuple[int, int], groups: int):
        """
        :param stride: Stride
        :param padding: Padding
        :param groups: Groups
        """
        self.stride = stride
        self.padding = padding
        self.groups = groups


class Layer:
    """
    Holds attributes for a given layer. This is a training-framework-agnostic abstraction for a layer
    """

    # TODO Remove the default None value for input_shape and retrieve this info from the caller API.
    def __init__(self, module, name, weight_shape, output_shape, input_shape=None):
        """
        Constructor
        :param module: Reference to the layer
        :param name: Unique name of the layer
        :param weight_shape: Shape of the weight tensor
        :param output_shape: Shape of the output activations
        :param input_shape:  Shape of the inputs
        """
        self.module = module
        self.name = str(name)
        self.weight_shape = weight_shape
        self.output_shape = output_shape
        self.input_shape = input_shape

        self.picked_for_compression = False
        self.type_specific_params = None

        self._set_type_specific_params(module)

    @abc.abstractmethod
    def _set_type_specific_params(self, module):
        """
        Using the provided module set type-specific-params
        :param module: Training-extension specific module
        :return:
        """


class LayerDatabase:
    """
    Stores, creates and updates the Layer database
    Also stores compressible layers to model optimization
    """
    def __init__(self, model):
        self._model = model
        self._compressible_layers = OrderedDict()

    @property
    def model(self):
        """ Property to expose the underlying model """
        return self._model

    def __iter__(self):
        """
        Expose the underlying compressible_layers dictionary as an iterable
        :return:
        """
        return iter(self._compressible_layers.values())

    def find_layer_by_name(self, layer_name: str) -> Layer:
        """
        Find a layer in the database given the name of the layer
        :param layer_name: Name of the layer
        :return: Layer reference
        :raises KeyError if layer_name does not correspond to any layer in the database
        """

        for layer in self._compressible_layers.values():
            if layer.name == layer_name:
                return layer

        raise KeyError("Layer name %s does not exist in layer database" % layer_name)

    def find_layer_by_module(self, module) -> Layer:
        """
        Find a layer in the database given the name of the layer
        :param module: Module to find
        :return: Layer reference
        :raises KeyError if layer_name does not correspond to any layer in the database
        """
        return self._compressible_layers[id(module)]

    def mark_picked_layers(self, selected_layers):
        """
        Marks layers which are selected in the database
        :param selected_layers: layers which are selected for compression
        """
        for layer in self._compressible_layers.values():
            if layer in selected_layers:
                layer.picked_for_compression = True

    def get_selected_layers(self):
        """
        :return: Returns selected layers
        """
        selected_layers = [layer for layer in self._compressible_layers.values()
                           if layer.picked_for_compression is True]
        return selected_layers

    @abc.abstractmethod
    def destroy(self):
        """
        Destroys the layer database
        """
