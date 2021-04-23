# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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


""" Utilities to load and save onnx models """

from typing import Union, List, Tuple, Dict
import os
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.onnx.symbolic_caffe2

import onnx

from aimet_common.utils import AimetLogger
import aimet_torch.utils
import aimet_torch.elementwise_ops as elementwise_ops
from aimet_torch.defs import OpToIOTensors

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


recurrent_onnx_optypes = ['LSTM', 'GRU', 'RNN']

# This is a dict that maps a PyTorch module type to the corresponding ONNX op type (as a string)
map_torch_types_to_onnx = {
    nn.Conv2d: ['Conv'],
    nn.Dropout: ['Dropout'],
    nn.Dropout2d: ['Dropout'],
    nn.BatchNorm1d: ['BatchNormalization'],
    nn.BatchNorm2d: ['BatchNormalization'],
    nn.ReLU: ['Relu'],
    nn.ReLU6: ['Clip'],
    nn.MaxPool2d: ['MaxPool'],
    nn.Linear: ['Gemm', 'MatMul'],
    nn.AdaptiveAvgPool2d: ['GlobalAveragePool', 'AveragePool'],
    nn.AvgPool2d: ['AveragePool'],
    nn.LogSoftmax: ['LogSoftmax'],
    nn.RNN:  ['RNN'],
    nn.LSTM: ['LSTM'],
    nn.GRU: ['GRU'],
    nn.ConvTranspose2d: ['ConvTranspose'],
    nn.Sigmoid: ['Sigmoid'],
    nn.Upsample: ['Upsample'],
    nn.PReLU: ['PRelu'],
    nn.LeakyReLU: ['LeakyRelu'],
    nn.Flatten: ['Flatten'],
    elementwise_ops.Add: ['Add'],
    elementwise_ops.Subtract: ['Sub'],
    elementwise_ops.Multiply: ['Mul'],
    elementwise_ops.Divide: ['Div'],
    elementwise_ops.Concat: ['Concat']

}

# Maps pytorch functional op string names to corresponding onnx types.
pytorch_functional_name_to_onnx_dict = {
    'add': 'Add',
    'cat': 'Concat',
    'mul': 'Mul',
    'div': 'Div'
}


class OnnxExportApiArgs:
    """
    configuration for torch onnx export api invocation
    """

    def __init__(self, opset_version: int = None, input_names: List[str] = None, output_names: List[str] = None):
        """
        Refer torch documentation https://pytorch.org/docs/1.7.1/onnx.html?highlight=onnx%20export#torch.onnx.export
        :param opset_version: onnx opset version to use to export the model
        :param input_names:  names to assign to the input nodes of the onnx graph, in order
        :param output_names: names to assign to the output nodes of the graph, in order
        """
        self.opset_version = opset_version
        self.input_names = input_names
        self.output_names = output_names

    @property
    def kwargs(self):
        """
        formats all override options into kwarg format to appended to onnx export call
        """
        return {'opset_version': self.opset_version,
                'input_names': self.input_names,
                'output_names': self.output_names}

class OnnxSaver:
    """
    Utilities to save/load onnx models
    """

    @classmethod
    def set_node_names(cls, onnx_model_path: str, pytorch_model: torch.nn.Module,
                       dummy_input: Union[torch.Tensor, Tuple],
                       onnx_export_args: OnnxExportApiArgs = OnnxExportApiArgs()):
        """
        This utility loads a given onnx model file and set the names of all the nodes (ops) to equivalent
        pytorch module names given the corresponding pytorch model.
        :param onnx_model_path: Path to the ONNX model file
        :param pytorch_model: Equivalent PyTorch model instance
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :param onnx_export_args:  override options for torch.onnx.export call
        :return:
        """

        cls._map_onnx_nodes_to_pytorch_modules(pytorch_model, dummy_input, onnx_model_path, onnx_export_args)

    @staticmethod
    def _create_map_of_tensor_to_node(onnx_model: onnx.ModelProto) -> Tuple[Dict[str, List[onnx.NodeProto]],
                                                                            Dict[str, onnx.NodeProto]]:
        """
        Create and return two dicts
            1. Tensor -> list of nodes that consume this tensor
            2. Tensor -> node that produces this tensor
        :param onnx_model: ONNX model object
        :return: The two dicts described above

        Note: The list in #1 is ordered exactly in the order that pytorch trace reaches these nodes. This is important
        because later on we will use pytorch layer hooks to match these nodes with the equivalent PyTorch modules.
        The expectation is that PyTorch trace and PyTorch hooks follow the same execution sequence
        """
        map_input_tensor_to_node = {}
        map_output_tensor_to_node = {}
        for node in onnx_model.graph.node:
            for in_tensor in node.input:
                if in_tensor in map_input_tensor_to_node:
                    map_input_tensor_to_node[in_tensor].append(node)
                else:
                    map_input_tensor_to_node[in_tensor] = [node]

            for output in node.output:
                assert output not in map_output_tensor_to_node, 'More than one node produces the same tensor'
                map_output_tensor_to_node[output] = node

        return map_output_tensor_to_node, map_input_tensor_to_node

    @classmethod
    def _add_markers(cls, starting_module, module_name_map):
        """Recursively add marker layers
        """

        class CustomMarkerFunc(torch.autograd.Function):
            """
            This function helps add a custom layer when exporting to ONNX
            Note the input tensor has a trivial operation performed on it (clamp). This is needed to force
            pytorch trace to not ignore the function.
            """

            @staticmethod
            def symbolic(g, inp, identifier, start):
                """
                Magic method that helps with exporting a custom ONNX node
                """
                return g.op('CustomMarker', inp, id_s=identifier, start_s=start)

            @staticmethod
            def forward(ctx, inp, _identifier, _start):     # pylint: disable=arguments-differ
                return inp.clamp(0)

            @staticmethod
            def backward(ctx, _grad):                       # pylint: disable=arguments-differ
                raise NotImplementedError()

        class CustomMarker(torch.nn.Module):
            """
            This is a temporary layer that in inserted next to a real layer to distinguish the real layer in the
            exported ONNX format
            """

            def __init__(self, module, identifier):
                super(CustomMarker, self).__init__()
                self.marked_module = module
                self.identifier = identifier

            def forward(self, *inputs):
                """
                Forward method for this CustomMarker layer
                """
                output = []
                for t in inputs:
                    if isinstance(t, torch.Tensor):
                        t = CustomMarkerFunc.apply(t, self.identifier, 'True')
                    output.append(t)

                x = self.marked_module(*output)
                if isinstance(x, torch.Tensor):
                    x = [x]

                output = []
                for t in x:
                    if isinstance(t, torch.Tensor):
                        t = CustomMarkerFunc.apply(t, self.identifier, 'False')
                    output.append(t)

                if len(output) == 1:
                    output = output[0]
                else:
                    output = tuple(output)

                return output

        for module_name, module_ref in starting_module.named_children():

            if aimet_torch.utils.is_leaf_module(module_ref):
                marker_layer = CustomMarker(module_ref, module_name_map[module_ref])
                setattr(starting_module, module_name, marker_layer)

            # recursively call children modules
            else:
                cls._add_markers(module_ref, module_name_map)

    @classmethod
    def _map_onnx_nodes_to_pytorch_modules(cls, pt_model, dummy_input, onnx_model_path, onnx_export_args):
        """
        Exports an onnx model, maps the nodes in the onnx model to corresponding pytorch modules and names
        them accordingly
        :param pt_model: PyTorch model
        :param dummy_input: Dummy input to run a fwd pass on @pt_model
        :param onnx_model_path: Path to the saved ONNX model
        :param onnx_export_args:  override options for torch.onnx.export call
        """
        working_dir = os.path.dirname(onnx_model_path)

        onnx_model = cls._create_onnx_model_with_markers(dummy_input, pt_model, working_dir, onnx_export_args)

        # Parse the ONNX model and create mapping from input and output tensors to corresponding nodes
        map_output_tensor_to_node, map_input_tensor_to_node = cls._create_map_of_tensor_to_node(onnx_model)

        # Find all marker nodes
        end_marker_map, start_marker_map = cls._create_map_of_marker_nodes(onnx_model)

        # Set names
        cls._set_onnx_node_names(map_input_tensor_to_node, start_marker_map)

        # Remove markers
        for markers in start_marker_map.values():
            for marker in markers:
                cls._detach_onnx_node(map_input_tensor_to_node, map_output_tensor_to_node, marker)

        for markers in end_marker_map.values():
            for marker in markers:
                cls._detach_onnx_node(map_input_tensor_to_node, map_output_tensor_to_node, marker)

        onnx_model = cls._remove_detached_nodes_from_onnx_graph(onnx_model)

        cls._fix_param_names(onnx_model)

        onnx.save(onnx_model, onnx_model_path)

    @classmethod
    def _fix_param_names(cls, onnx_model):
        """
        Parameter names have an additional level due to the name of the Marker module itself. This method removes that.
        :param onnx_model: Onnx Model
        """
        # Rename initializers
        for ini in onnx_model.graph.initializer:
            if 'marked_module' in ini.name:
                name = ini.name
                name = name.replace('marked_module.', '')
                ini.name = name

        # Change the references to initializers in each node
        for node in onnx_model.graph.node:
            indices_to_replace = []
            for index, inp_tensor in enumerate(node.input):
                if 'marked_module' in inp_tensor:
                    indices_to_replace.append(index)

            for index in indices_to_replace:
                param_name = node.input[index]
                node.input.remove(param_name)
                node.input.insert(index, param_name.replace('marked_module.', ''))


    @classmethod
    def _remove_detached_nodes_from_onnx_graph(cls, onnx_model):
        """
        Given a ONNX model removes any detached nodes from the graph
        :return: Updated onnx model
        """
        e = onnx.utils.Extractor(onnx_model)
        model_inputs = [inp.name for inp in onnx_model.graph.input]
        model_outputs = [output.name for output in onnx_model.graph.output]
        onnx_model = e.extract_model(model_inputs, model_outputs)
        return onnx_model

    @classmethod
    def _set_onnx_node_names(cls, map_input_tensor_to_node, start_marker_map):
        """
        Set names of the ONNX nodes using the identifier fields in the marker layers
        :param map_input_tensor_to_node: Map of tensor to node consuming that tensor
        :param start_marker_map: Map of start marker nodes in the ONNX graph
        :return:
        """
        def set_name_for_downstream_nodes(starting_nodes, name, depth):
            for node in starting_nodes:

                if node.op_type == 'CustomMarker':      # Recursion end condition
                    return

                if depth == 0:
                    node.name = name
                else:
                    node.name = name + "#" + str(depth)

                for tensor in node.output:
                    downstream_nodes = map_input_tensor_to_node.get(tensor, [])
                    set_name_for_downstream_nodes(downstream_nodes, name, depth + 1)

        for node_name, markers in start_marker_map.items():
            for marker in markers:
                out_tensor = marker.output[0]
                downstream_nodes = map_input_tensor_to_node.get(out_tensor, [])
                set_name_for_downstream_nodes(downstream_nodes, node_name, 0)

    @classmethod
    def _create_map_of_marker_nodes(cls, onnx_model):
        """
        Creates and returns maps of start and end marker nodes
        :param onnx_model: Onnx model
        :return: Map of end marker node, Map of start marker nodes
        """
        start_marker_map = defaultdict(list)
        end_marker_map = defaultdict(list)
        for node in onnx_model.graph.node:
            if node.op_type == 'CustomMarker':
                identifier = node.attribute[0].s.decode()
                is_start_marker = node.attribute[1].s.decode()

                if is_start_marker == 'True':
                    start_marker_map[identifier].append(node)
                else:
                    end_marker_map[identifier].append(node)
        print(start_marker_map.keys())
        print(end_marker_map.keys())
        return end_marker_map, start_marker_map

    @classmethod
    def _create_onnx_model_with_markers(cls, dummy_input, pt_model, working_dir, onnx_export_args):
        """
        Exports an onnx model with marker nodes inserted
        :param dummy_input: Dummy input
        :param pt_model: PyTorch model
        :param working_dir: Working directory for storing the exported onnx model
        :param onnx_export_args:  override options for torch.onnx.export call
        :return: Onnx model with marker layers
        """
        model = copy.deepcopy(pt_model).cpu()
        module_name_map = {}
        for module_name, module_ref in model.named_modules():
            if aimet_torch.utils.is_leaf_module(module_ref):
                module_name_map[module_ref] = module_name
        cls._add_markers(model, module_name_map)
        temp_file = os.path.join(working_dir, 'temp_onnx_model_with_markers.onnx')
        torch.onnx.export(model, dummy_input, temp_file, enable_onnx_checker=False, **onnx_export_args.kwargs)
        onnx_model = onnx.load(temp_file)
        return onnx_model

    @classmethod
    def _detach_onnx_node(cls, map_input_tensor_to_node, map_output_tensor_to_node, node_to_detach):
        """
        Given a ONNX node, detach it from the graph
        :param map_input_tensor_to_node: Map of tensor to node consuming the tensor
        :param map_output_tensor_to_node: Map of tensor to node producing the tensor
        :param node_to_detach: Reference to the ONNX node to detach
        """
        assert len(node_to_detach.input) == 1
        assert len(node_to_detach.output) == 1

        input_tensor = node_to_detach.input[0]
        output_tensor = node_to_detach.output[0]

        if input_tensor in map_output_tensor_to_node:
            prev_node = map_output_tensor_to_node[input_tensor]
            index = list(prev_node.output).index(input_tensor)
            prev_node.output.remove(input_tensor)
            prev_node.output.insert(index, output_tensor)

            map_output_tensor_to_node[output_tensor] = prev_node
            map_output_tensor_to_node.pop(input_tensor)

            for consumer in map_input_tensor_to_node[input_tensor]:
                if consumer is node_to_detach:
                    continue
                index = list(consumer.input).index(input_tensor)
                consumer.input.remove(input_tensor)
                consumer.input.insert(index, output_tensor)

            map_input_tensor_to_node.pop(input_tensor)

        else:
            for consumer in map_input_tensor_to_node[output_tensor]:
                index = list(consumer.input).index(output_tensor)
                consumer.input.remove(output_tensor)
                consumer.input.insert(index, input_tensor)

        node_to_detach.input.pop()
        node_to_detach.output.pop()

    @staticmethod
    def _collate_io_tensors_for_multi_layer_recurrent_nodes(onnx_model: onnx.NodeProto,
                                                            node_to_io_tensor_name_map: Dict):
        """
        Given an ONNX model and corresponding node-tensor map, consolidate multi-layer recurrent nodes
        into single map entries
        """
        recurrent_nodes = []
        for node in onnx_model.graph.node:
            if node.op_type in recurrent_onnx_optypes:
                recurrent_nodes.append(node.name)

        # Collection of recurrent nodes that includes only the first layer nodes
        recurrent_root_nodes = [node for node in recurrent_nodes if '#' not in node]

        for root_node in recurrent_root_nodes:
            # Find nodes corresponding to all other layers of the recurrent node
            other_layers = [node for node in recurrent_nodes if node.startswith(root_node + '#')]

            # sort the other layers using the depth value following the '#'
            other_layers = sorted(other_layers, key=lambda layer: int(layer.split('#')[1]))

            # Append the io_tensors for all layers for the current root recurrent node, in order
            io_tensor_list = [node_to_io_tensor_name_map[root_node]]
            for layer in other_layers:
                io_tensor_list.append(node_to_io_tensor_name_map[layer])
                del node_to_io_tensor_name_map[layer]
            node_to_io_tensor_name_map[root_node] = io_tensor_list

    @classmethod
    def get_onnx_node_to_io_tensor_names_map(cls, onnx_model: onnx.NodeProto) -> \
            (Dict[str, Union[OpToIOTensors, List[OpToIOTensors]]], set):
        """
        Given an ONNX model, gets the inputs and output tensor names for each node in the model.
        if multiple onnx nodes have the same name then the nodes are provided as a list of inputs and output tensor
         names, one for each onnx node.
        :param onnx_model: The ONNX model instance
        :return: Dictionary of ONNX node name and corresponding input and output tensor names and a set with all valid
        param names in model
        """

        node_to_io_tensor_name_map = {}
        valid_param_set = set()
        initializer_names = {initializer.name for initializer in onnx_model.graph.initializer}

        for node in onnx_model.graph.node:
            if node.name:
                onnx_node_io_tensors = OpToIOTensors(list(node.input), list(node.output))
                if (node.name not in node_to_io_tensor_name_map) or node.op_type in recurrent_onnx_optypes:
                    node_to_io_tensor_name_map[node.name] = onnx_node_io_tensors

            # update valid params list
            for input_tensor in list(node.input):
                if input_tensor in initializer_names:
                    valid_param_set.add(input_tensor)

        cls._collate_io_tensors_for_multi_layer_recurrent_nodes(onnx_model, node_to_io_tensor_name_map)

        return node_to_io_tensor_name_map, valid_param_set
