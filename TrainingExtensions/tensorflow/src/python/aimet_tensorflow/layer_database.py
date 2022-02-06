# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
from collections import OrderedDict
from typing import Tuple, Set, Union, List

import tensorflow as tf

# Import aimet specific modules
from aimet_tensorflow import graph_editor
from aimet_tensorflow.utils.common import is_op_compressible, get_valid_ops
from aimet_tensorflow.utils import graph_saver
from aimet_tensorflow.utils.op.conv import get_output_activation_shape
import aimet_tensorflow.utils.op.conv
import aimet_common.layer_database
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)

OP_WEIGHT_INDICES = {'Conv2D': 1,
                     'MatMul': 1,
                     'DepthwiseConv2dNative': 1,
                     'Conv2DTranspose': 1,
                     'BiasAdd': 1,
                     'Add': 1}


class Layer(aimet_common.layer_database.Layer):
    """ Holds attributes for given Op """

    def _set_type_specific_params(self, module: tf.Operation):
        """
        Using the provided module set type-specific-params.

        :param module: Training-extension specific module
        """

        if module.type == 'Conv2D':

            strides, padding, groups = aimet_tensorflow.utils.op.conv.get_conv2d_op_params(module)

            params = aimet_common.layer_database.Conv2dTypeSpecificParams(strides, padding, groups)
            self.type_specific_params = params

    def __init__(self, model: tf.compat.v1.Session, op: tf.Operation, output_shape: List):
        """
        :param model: TensorFlow Session
        :param op: TensorFlow Op
        :param output_shape: output activation shape in Common format channels_first format (NCHW)
        """

        self.model = model

        # get the weight shape
        weight_shape = aimet_tensorflow.utils.op.conv.get_weight_shape(op)

        aimet_common.layer_database.Layer.__init__(self, module=op, name=op.name,
                                                   weight_shape=weight_shape, output_shape=output_shape)


class LayerDatabase(aimet_common.layer_database.LayerDatabase):
    """
    Stores, creates and updates the Layer database.
    Also stores compressible layers to model optimization.
    """

    def __init__(self, model: tf.compat.v1.Session, input_shape: Union[Tuple, List[Tuple]], working_dir: str,
                 starting_ops: List[str] = None, ending_ops: List[str] = None):
        """
        :param model: TensorFlow Session
        :param input_shape: tuple or list of tuples of input shapes to the model
        :param working_dir: path to working directory to store intermediate graphs
        :param starting_ops: Starting ops of the graph, used in a top down DFS search
        :param ending_ops: Ending ops of the graph, used in a bottom up DFS search
        """
        self.meta_path = graph_saver.get_meta_and_checkpoint_path(working_dir)
        if starting_ops:
            self.starting_ops = starting_ops
        else:
            self.starting_ops = []
        if ending_ops:
            self.ending_ops = ending_ops
        else:
            self.ending_ops = []

        self.input_shape = input_shape

        # Save the original model graph to meta path
        graph_saver.save_model_to_meta(model=model, meta_path=self.meta_path + 'original_model')

        aimet_common.layer_database.LayerDatabase.__init__(self, model)

        self._create_database()

    def __deepcopy__(self, memodict: dict):
        """
        Create deepcopy of Layer Database.

        :param memodict: id to object dictionary
        """

        # pylint: disable=protected-access

        # Allocate a new Layer Database
        layer_db = copy.copy(self)
        memodict[id(self)] = layer_db

        # Load the original model graph so we are operating on a fresh copy of the original model graph
        layer_db._model = graph_saver.load_model_from_meta(meta_path=self.meta_path + 'original_model.meta')

        layer_db._compressible_layers = OrderedDict()

        # all the ops in the existing graph
        for op in self._model.graph.get_operations():

            # If this module is in the current layer database
            if id(op) in self._compressible_layers:

                existing_layer = self._compressible_layers[id(op)]

                # get the corresponding op in new graph
                new_op = layer_db._model.graph.get_operation_by_name(existing_layer.name)

                # get the output activation shape
                output_shape = get_output_activation_shape(sess=layer_db._model, op=new_op,
                                                           input_op_names=self.starting_ops,
                                                           input_shape=self.input_shape)

                # create new layer
                new_layer = Layer(model=layer_db._model, op=new_op, output_shape=output_shape)

                new_layer.picked_for_compression = existing_layer.picked_for_compression

                layer_db._compressible_layers[id(new_op)] = new_layer

        return layer_db

    def _create_database(self):
        """
        Create Layer Database by populating with Conv2D and MatMul layers.
        """
        # get all the operations associated with graph in current session
        all_ops = self.model.graph.get_operations()
        # If starting ops is provided, use it to get a set of valid ops in the graph
        if self.starting_ops:
            valid_ops = get_valid_ops(self.model.graph, self.starting_ops, self.ending_ops)
            # Only keep ops in all_ops if it is a valid op
            all_ops = [op for op in all_ops if op in valid_ops]

        for op in all_ops:
            if is_op_compressible(op):

                output_shape = get_output_activation_shape(sess=self.model, op=op, input_op_names=self.starting_ops,
                                                           input_shape=self.input_shape)

                self._compressible_layers[id(op)] = Layer(model=self.model, op=op, output_shape=output_shape)

    def replace_layer_with_sequential_of_two_layers(self, layer_to_replace: Layer,
                                                    layer_a: Layer, layer_b: Layer):
        """
        Replaces original layer with two new layers in the graph.
        Adds two new layers in the database and remove the original layer from database.

        :param layer_to_replace: layer to replace
        :param layer_a: layer a
        :param layer_b: layer b
        """

        old_bias_op = aimet_tensorflow.utils.common.get_succeeding_bias_op(layer_to_replace.module)
        old_outputs = [old_bias_op.outputs[0]] if old_bias_op is not None else [layer_to_replace.module.outputs[0]]

        new_bias_op = aimet_tensorflow.utils.common.get_succeeding_bias_op(layer_b.module)
        new_outputs = [new_bias_op.outputs[0]] if new_bias_op is not None else [layer_b.module.outputs[0]]

        consumers = []

        for output in old_outputs:

            for consumer in output.consumers():
                consumers.append(consumer)

        # For each tensor's pair, replaces the end of [t1 = old_outputs] by the end of [t0 = new_outputs]
        # The end of the tensors in [ts1 = old_outputs] are left dangling
        _ = graph_editor.reroute_ts(ts0=new_outputs, ts1=old_outputs, can_modify=consumers)

        # Add the new layer to the database
        self._compressible_layers[id(layer_a.module)] = layer_a
        self._compressible_layers[id(layer_b.module)] = layer_b

        # Remove the the layer being replaced from the database
        del self._compressible_layers[id(layer_to_replace.module)]

    def update_database(self, model: tf.compat.v1.Session, detached_op_names: Set, update_model=False):
        """
        Update layer database with new provided session and exclude detached ops.

        :param model: TensorFlow Session
        :param detached_op_names: list of detached op names
        :param update_model: update model (session) with provided session if True
        """
        if update_model:
            # close the existing session
            self._model.close()
            # update the model (session) with new provided tf.compat.v1.Session
            self._model = model

        # clear the dictionary
        self._compressible_layers.clear()

        # get all the operations associated with graph in the provided session
        all_ops = self._model.graph.get_operations()
        # If starting ops is provided, use it to get a set of valid ops in the graph
        if self.starting_ops:
            valid_ops = get_valid_ops(self._model.graph, self.starting_ops, self.ending_ops)
            # Only keep ops in all_ops if it is a valid op
            all_ops = [op for op in all_ops if op in valid_ops]

        for op in all_ops:

            if is_op_compressible(op) and op.name not in detached_op_names:

                # get the output activation shape
                output_shape = get_output_activation_shape(sess=self._model, op=op, input_op_names=self.starting_ops,
                                                           input_shape=self.input_shape)

                self._compressible_layers[id(op)] = Layer(model=self._model, op=op, output_shape=output_shape)

    def destroy(self):
        """
        Close the session and reset the default graph if in case any graph is created without 'with' keyword.
        """
        # clear the dictionary
        self._compressible_layers.clear()
        # close the session
        self._model.close()
        tf.compat.v1.reset_default_graph()
