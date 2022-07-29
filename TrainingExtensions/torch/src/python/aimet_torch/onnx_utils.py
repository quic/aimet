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

from typing import Union, List, Tuple, Dict, Set
import os
import copy
from collections import defaultdict
import torch
import torch.nn as nn
import torch.onnx.symbolic_caffe2
import onnx
from packaging import version

from aimet_common.utils import AimetLogger
import aimet_torch.utils
import aimet_torch.elementwise_ops as elementwise_ops
from aimet_torch.defs import OpToIOTensors

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


recurrent_onnx_optypes = ['LSTM', 'GRU', 'RNN']

# This is a dict that maps a PyTorch module type to the corresponding ONNX op type (as a string)
map_torch_types_to_onnx = {
    nn.Conv1d: ['Conv'],
    nn.Conv2d: ['Conv'],
    nn.Dropout: ['Dropout'],
    nn.Dropout2d: ['Dropout'],
    nn.BatchNorm1d: ['BatchNormalization'],
    nn.BatchNorm2d: ['BatchNormalization'],
    # Note - Currently, both LayerNorm and GELU are not in the supported ops list in ONNX
    # Adding this entry here for usage by Connected graph
    nn.LayerNorm: ['LayerNorm'],
    nn.GELU: ['GELU'],
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
    nn.Softmax: ['Softmax'],
    nn.Tanh: ['Tanh'],
    nn.Softplus: ['Softplus'],
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


if version.parse(torch.__version__) >= version.parse("1.9"):
    onnx_subgraph_op_to_pytorch_module_param_name = {
        torch.nn.GroupNorm:
            {
                # '#depth', 'op_type': {input_index: torch module parameter name}
                ('#3', 'Mul'): {1: 'weight'},
                ('#4.end', 'Add'): {1: 'bias'}
            },
        torch.nn.Linear:
            {
                ('', 'MatMul'): {1: 'weight'},
                ('#1.end', 'Add'): {0: 'bias'}
            },
        torch.nn.PReLU:
            {
                ('', 'PRelu'): {1: 'weight'}
            }
    }
else:
    onnx_subgraph_op_to_pytorch_module_param_name = {
        torch.nn.GroupNorm:
            {
                # '#depth', 'op_type': {input_index: torch module parameter name}
                ('#3', 'Mul'): {1: 'weight'},
                ('#4.end', 'Add'): {1: 'bias'}
            },
        torch.nn.Linear:
            {
                ('', 'MatMul'): {1: 'weight'},
                ('#1.end', 'Add'): {1: 'bias'}
            },
        torch.nn.PReLU:
            {
                ('', 'PRelu'): {1: 'weight'}
            }
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
        if inp.dtype == torch.bool:
            return inp
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


class OnnxSaver:
    """
    Utilities to save/load onnx models
    """

    @classmethod
    def set_node_names(cls, onnx_model_path: str, pytorch_model: torch.nn.Module,
                       dummy_input: Union[torch.Tensor, Tuple], is_conditional=False, module_marker_map=None,
                       onnx_export_args: OnnxExportApiArgs = None):
        """
        This utility loads a given onnx model file and set the names of all the nodes (ops) to equivalent
        pytorch module names given the corresponding pytorch model.
        :param onnx_model_path: Path to the ONNX model file
        :param pytorch_model: Equivalent PyTorch model instance
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :param is_conditional: True if model is a conditional model, False otherwise
        :param module_marker_map: Maps module names to traced custom markers (only used for conditional models)
        :param onnx_export_args:  override options for torch.onnx.export call
        :return:
        """
        if module_marker_map is None:
            module_marker_map = {}
        if onnx_export_args is None:
            onnx_export_args = OnnxExportApiArgs()

        onnx_model = cls._map_onnx_nodes_to_pytorch_modules(pytorch_model, dummy_input,
                                                            onnx_model_path, onnx_export_args, is_conditional,
                                                            module_marker_map)

        onnx.save(onnx_model, onnx_model_path)

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
            OnnxSaver._populate_input_output_tensor_maps(map_input_tensor_to_node, map_output_tensor_to_node, node)
        return map_output_tensor_to_node, map_input_tensor_to_node

    @staticmethod
    def _populate_input_output_tensor_maps(map_input_tensor_to_node: Dict[str, List[onnx.NodeProto]],
                                           map_output_tensor_to_node: Dict[str, onnx.NodeProto], node: onnx.NodeProto):
        """
        Populate input tensor to node and output tensor to node maps given a node. If the node has a graph with nodes,
        recursively populate the maps for subgraph nodes.
        :param map_input_tensor_to_node: Dictionary mapping onnx tensor to nodes that consume it
        :param map_output_tensor_to_node: Dictionary mapping onnx tensor to node that generated it
        :param node: Node to extract input and output tensors for to populate maps
        """
        for in_tensor in node.input:
            if in_tensor in map_input_tensor_to_node:
                map_input_tensor_to_node[in_tensor].append(node)
            else:
                map_input_tensor_to_node[in_tensor] = [node]

        for output in node.output:
            assert output not in map_output_tensor_to_node, 'More than one node produces the same tensor'
            map_output_tensor_to_node[output] = node

        for attribute in node.attribute:
            if getattr(attribute, 'g', None) is not None:
                for subnode in getattr(attribute, 'g').node:
                    OnnxSaver._populate_input_output_tensor_maps(map_input_tensor_to_node, map_output_tensor_to_node,
                                                                 subnode)

    @classmethod
    def _add_markers(cls, starting_module, module_name_map, module_marker_map, use_trace):
        """Recursively add marker layers
        """
        for local_module_name, module_ref in starting_module.named_children():
            if aimet_torch.utils.is_leaf_module(module_ref):
                # local module name only contains the module's attribute name directly under the parent module
                # need to pick up the full name starting with top level module name from module_inputs_map
                full_module_name = module_name_map[module_ref]
                if use_trace:
                    assert full_module_name in module_marker_map
                    marker_layer = module_marker_map[full_module_name]
                else:
                    marker_layer = CustomMarker(module_ref, full_module_name)
                setattr(starting_module, local_module_name, marker_layer)

            # recursively call children modules
            else:
                cls._add_markers(module_ref, module_name_map, module_marker_map, use_trace)

    @classmethod
    def _map_onnx_nodes_to_pytorch_modules(cls, pt_model, dummy_input, onnx_model_path, onnx_export_args,
                                           is_conditional, module_marker_map):
        """
        Exports an onnx model, maps the nodes in the onnx model to corresponding pytorch modules and names
        them accordingly
        :param pt_model: PyTorch model
        :param dummy_input: Dummy input to run a fwd pass on @pt_model
        :param onnx_model_path: Path to the saved ONNX model
        :param onnx_export_args:  override options for torch.onnx.export call
        :param is_conditional: True if model is a conditional model, False otherwise
        :param module_marker_map: Maps module names to traced custom markers (only used for conditional models)
        """
        working_dir = os.path.dirname(onnx_model_path)

        onnx_model = cls._create_onnx_model_with_markers(dummy_input, pt_model, working_dir, onnx_export_args,
                                                         is_conditional, module_marker_map)
        graphs_list, output_names_list = OnnxSaver._get_graph_and_output_names_lists(onnx_model)

        # Parse the ONNX model and create mapping from input and output tensors to corresponding nodes
        map_output_tensor_to_node, map_input_tensor_to_node = cls._create_map_of_tensor_to_node(onnx_model)

        # Find all marker nodes
        end_marker_map, start_marker_map = cls._create_map_of_marker_nodes(onnx_model)

        # Set names
        cls._set_onnx_node_names(map_input_tensor_to_node, start_marker_map)

        # Remove markers
        cls._detach_start_and_end_markers(map_input_tensor_to_node, map_output_tensor_to_node, start_marker_map,
                                          end_marker_map, graphs_list, output_names_list)

        # Make sure we rename the model outputs to original names
        cls._set_output_names(onnx_model, graphs_list, output_names_list, map_output_tensor_to_node,
                              map_input_tensor_to_node)

        # Clean up the detached nodes
        # pylint: disable=no-member
        cls._remove_detached_nodes_from_onnx_graph(onnx_model.graph)

        cls._fix_param_names(onnx_model)
        cls._fix_initializer_names(onnx_model, pt_model)

        return onnx_model

    @staticmethod
    def _get_graph_and_output_names_lists(onnx_model: onnx.ModelProto) -> Tuple[List[onnx.GraphProto], List[List[str]]]:
        """
        Get a list of graphs in the model with outputs, and a list of lists of output names for the corresponding
        graphs.
        Length of the two lists should be identical. A dictionary would be the preferred structure, however
        onnx.GraphProto objects are unhashable.
        :param onnx_model: Onnx model to get list of graphs and output names for
        :return: A list of graphs in the model with outputs, and a list of lists of output names for the corresponding
        graphs.
        """
        graphs_list = []
        output_names_list = []
        OnnxSaver._populate_graph_and_output_names_lists(onnx_model.graph, graphs_list, output_names_list)
        return graphs_list, output_names_list

    @staticmethod
    def _populate_graph_and_output_names_lists(onnx_graph: onnx.GraphProto, graphs_list: List[onnx.GraphProto],
                                               output_names_list: List[List[str]]):
        """
        Helper function to populate lists of graph and output names for a particular onnx graph. Also recursively
        populates the lists for subgraphs of the given graph.
        :param onnx_graph: Onnx graph to populate graphs list and output names list with
        :param graphs_list: List of graphs with outputs
        :param output_names_list: List of list of output names for corresponding graphs
        """
        output_names_list_for_graph = []
        for output in onnx_graph.output:
            output_names_list_for_graph.append(output.name)
        if output_names_list_for_graph:
            graphs_list.append(onnx_graph)
            output_names_list.append(output_names_list_for_graph)

        for node in onnx_graph.node:
            for attribute in node.attribute:
                if getattr(attribute, 'g', None) is not None:
                    OnnxSaver._populate_graph_and_output_names_lists(attribute.g, graphs_list, output_names_list)

    @staticmethod
    def _get_onnx_node_map(onnx_graph: onnx.GraphProto, onnx_node_map: Dict[Tuple[str, str], onnx.NodeProto] = None) \
            -> Dict[Tuple[str, str], onnx.NodeProto]:
        """
        Get a mapping between tuples of onnx node names and op types, and the node itself.
        :param onnx_graph: Onnx graph used to populate map
        :param onnx_node_map: Map between tuples of onnx node names and op types, and the node itself
        """
        if onnx_node_map is None:
            onnx_node_map = {}

        for node in onnx_graph.node:
            onnx_node_map[(node.name, node.op_type)] = node

        for node in onnx_graph.node:
            for attribute in node.attribute:
                if getattr(attribute, 'g', None) is not None:
                    OnnxSaver._get_onnx_node_map(attribute.g, onnx_node_map)
        return onnx_node_map

    @classmethod
    def _fix_initializer_names(cls, onnx_model: onnx.NodeProto, pt_model: torch.nn.Module):
        """
        Parameter names in some case do not have reflect the torch param names. This method updates the onnx model
        with param names using a custom mapping.
        When exporting a scripted model, all modules with params will need to update their initializer names.
        :param onnx_model: Onnx Model
        :param pt_model: PyTorch Model
        """

        initializers = OnnxSaver._get_all_initializers(onnx_model.graph)
        initializer_names = [ini.name for ini in initializers]
        onnx_node_map = OnnxSaver._get_onnx_node_map(onnx_model.graph)

        for module_name, module_ref in pt_model.named_modules():

            # Not using isinstance since we want to check for the exact type and not any subclasses
            # pylint: disable=unidiomatic-typecheck
            if type(module_ref) in onnx_subgraph_op_to_pytorch_module_param_name:
                for (node_suffix, op_type), replace_pairs in \
                        onnx_subgraph_op_to_pytorch_module_param_name[type(module_ref)].items():
                    # Some modules like linear can take on various forms (e.g. Gemm versus MatMul and Add)
                    if (module_name + node_suffix, op_type) in onnx_node_map:
                        node = onnx_node_map[module_name + node_suffix, op_type]

                        cls._replace_param_name(initializers, initializer_names, module_name, node, replace_pairs)

    @classmethod
    def _replace_param_name(cls, initializers: List[onnx.TensorProto], initializer_names: List[str], module_name: str,
                            node: onnx.NodeProto, replace_pairs: Dict[int, str]):
        """
        helper method to replace parameter names at the corresponding input tensor index
        :param initializer_names: List of model initializer names
        :param module_name: PyTorch module name
        :param node: Onnx node part of sub-graph that maps to the torch module
        :param replace_pairs: dictionary of input tensor indices and param names
        """
        for input_index, param_name in replace_pairs.items():
            # If bias is not present for example, skip processing for the index
            if len(node.input) > input_index:
                new_param_name = module_name + '.' + param_name
                inp_tensor = node.input[input_index]
                # Check if inp_tensor name is already named as a parameter (name.param_name, instead of a number). If so,
                # sanity check that the name we want to replace it with is the same as the existing name, then continue.
                if '.' in inp_tensor:
                    assert inp_tensor == new_param_name
                    continue
                node.input.remove(inp_tensor)
                node.input.insert(input_index, new_param_name)

                # Find the index of the old initializer name and use it to update the corresponding initializer's name
                # in the actual initializers array
                initializer_index = initializer_names.index(inp_tensor)
                initializers[initializer_index].name = new_param_name

    @classmethod
    def _remove_marked_module_string_from_initializer_names(cls, onnx_model):
        """
        Remove 'marked_module' from all initializer names.
        :param onnx_model: Onnx model containing initializers to update
        """
        for ini in OnnxSaver._get_all_initializers(onnx_model.graph):
            if 'marked_module' in ini.name:
                name = ini.name
                name = name.replace('marked_module.', '')
                ini.name = name

    @classmethod
    def _remove_marked_module_string_from_node_input_names(cls, onnx_graph):
        """
        Remove 'marked_module' from all node input names. Also recursively updates subgraph node input names.
        :param onnx_graph: Onnx graph containing nodes and subgraphs to modify
        """
        for node in onnx_graph.node:
            indices_to_replace = []
            for index, inp_tensor in enumerate(node.input):
                if 'marked_module' in inp_tensor:
                    indices_to_replace.append(index)

            for index in indices_to_replace:
                param_name = node.input[index]
                node.input.remove(param_name)
                node.input.insert(index, param_name.replace('marked_module.', ''))

        for node in onnx_graph.node:
            for attribute in node.attribute:
                if getattr(attribute, 'g', None) is not None:
                    cls._remove_marked_module_string_from_node_input_names(attribute.g)

    @classmethod
    def _fix_param_names(cls, onnx_model):
        """
        Parameter names have an additional level due to the name of the Marker module itself. This method removes that.
        :param onnx_model: Onnx Model
        """
        # Rename initializers
        cls._remove_marked_module_string_from_initializer_names(onnx_model)

        # Change the references to initializers in each node
        cls._remove_marked_module_string_from_node_input_names(onnx_model.graph)

    @classmethod
    def _remove_detached_nodes_from_onnx_graph(cls, onnx_graph: onnx.GraphProto):
        """
        Given a ONNX model, remove any detached nodes from the graph. Recursively removes detached nodes from subgraphs.
        """
        marker_nodes = [node for node in onnx_graph.node if node.op_type == 'CustomMarker']

        for node in marker_nodes:
            onnx_graph.node.remove(node)

        for node in onnx_graph.node:
            for attribute in node.attribute:
                if getattr(attribute, 'g', None) is not None:
                    OnnxSaver._remove_detached_nodes_from_onnx_graph(attribute.g)

    @classmethod
    def _set_onnx_node_names(cls, map_input_tensor_to_node, start_marker_map):
        """
        Set names of the ONNX nodes using the identifier fields in the marker layers
        :param map_input_tensor_to_node: Map of tensor to node consuming that tensor
        :param start_marker_map: Map of start marker nodes in the ONNX graph
        :return:
        """
        node_name_count_map = dict()
        visited = set()

        def set_name_for_downstream_nodes(starting_nodes, name, depth):
            for node in starting_nodes:

                if id(node) in visited:
                    continue

                if node.op_type == 'CustomMarker':      # Recursion end condition
                    return

                if depth == 0:
                    node.name = name
                else:
                    node.name = name + "#" + str(depth)

                if node.name in node_name_count_map:
                    reuse_count = node_name_count_map[node.name]
                    node_name_count_map[node.name] = reuse_count + 1
                    node.name = f'{node.name}-{reuse_count}' if '#' in node.name else f'{node.name}#0-{reuse_count}'
                else:
                    node_name_count_map[node.name] = 1

                for tensor in node.output:
                    downstream_nodes = map_input_tensor_to_node.get(tensor, [])
                    set_name_for_downstream_nodes(downstream_nodes, name, depth + 1)

                    if depth != 0:
                        for dnode in downstream_nodes:
                            if dnode.op_type == 'CustomMarker':  #end marker
                                node.name += '.end'
                                break
                visited.add(id(node))

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
            OnnxSaver._populate_start_and_end_marker_maps(start_marker_map, end_marker_map, node)
        return end_marker_map, start_marker_map

    @staticmethod
    def _populate_start_and_end_marker_maps(start_marker_map: Dict[str, onnx.NodeProto],
                                            end_marker_map: Dict[str, onnx.NodeProto], node: onnx.NodeProto):
        """
        Populate dictionaries mapping identifier names to start and end markers. Recursively populates dictionaries for
        subgraphs of the given node.
        :param start_marker_map: Dictionary mapping identifier names to start markers
        :param end_marker_map: Dictionary mapping identifier names to end markers
        :param node: Onnx node to potentially include in a map
        """
        if node.op_type == 'CustomMarker':
            identifier = node.attribute[0].s.decode()
            is_start_marker = node.attribute[1].s.decode()

            if is_start_marker == 'True':
                start_marker_map[identifier].append(node)
            else:
                end_marker_map[identifier].append(node)

        for attribute in node.attribute:
            if getattr(attribute, 'g', None) is not None:
                for subnode in getattr(attribute, 'g').node:
                    OnnxSaver._populate_start_and_end_marker_maps(start_marker_map, end_marker_map, subnode)

    @classmethod
    def _create_onnx_model_with_markers(cls, dummy_input, pt_model, working_dir, onnx_export_args, is_conditional,
                                        module_marker_map) -> \
            onnx.ModelProto:
        """
        Exports an onnx model with marker nodes inserted
        :param dummy_input: Dummy input
        :param pt_model: PyTorch model
        :param working_dir: Working directory for storing the exported onnx model
        :param onnx_export_args:  override options for torch.onnx.export call
        :param is_conditional: True if model is a conditional model, False otherwise
        :param module_marker_map: Maps module names to traced custom markers (only used for conditional models)
        :return: Onnx model with marker layers
        """
        model = copy.deepcopy(pt_model).cpu()
        module_name_map = {}
        for module_name, module_ref in model.named_modules():
            if aimet_torch.utils.is_leaf_module(module_ref):
                module_name_map[module_ref] = module_name
        cls._add_markers(model, module_name_map, module_marker_map, is_conditional)
        temp_file = os.path.join(working_dir, 'temp_onnx_model_with_markers.onnx')
        if is_conditional:
            with aimet_torch.utils.in_eval_mode(model), torch.no_grad():
                dummy_output = model(*dummy_input)
            scripted_model = torch.jit.script(model)
            torch.onnx.export(scripted_model, dummy_input, temp_file, example_outputs=dummy_output,
                              enable_onnx_checker=False, **onnx_export_args.kwargs)
        else:
            torch.onnx.export(model, dummy_input, temp_file, enable_onnx_checker=False, **onnx_export_args.kwargs)
        onnx_model = onnx.load(temp_file)
        return onnx_model

    @classmethod
    def _detach_start_and_end_markers(cls, map_input_tensor_to_node: Dict[str, onnx.NodeProto],
                                      map_output_tensor_to_node: Dict[str, List[onnx.NodeProto]],
                                      start_marker_map: Dict[str, onnx.NodeProto],
                                      end_marker_map: Dict[str, onnx.NodeProto], graphs_list: List[onnx.GraphProto],
                                      output_names_list: List[List[str]]):
        """
        Detach start and end marker nodes from the onnx graph
        :param map_input_tensor_to_node: Map of tensor to node consuming the tensor
        :param map_output_tensor_to_node: Map of tensor to node producing the tensor
        :param start_marker_map: Map of start marker identifiers to start markers
        :param end_marker_map: Map of end marker identifiers to end markers
        :param graphs_list: List of graphs with outputs
        :param output_names_list: List of list of output names for corresponding graphs
        """
        for markers in start_marker_map.values():
            for marker in markers:
                cls._detach_start_marker_node(map_input_tensor_to_node, map_output_tensor_to_node, marker)

        for markers in end_marker_map.values():
            for marker in markers:
                cls._detach_end_marker_node(map_input_tensor_to_node, map_output_tensor_to_node,
                                            graphs_list, output_names_list, marker)

    @classmethod
    def _detach_start_marker_node(cls, map_input_tensor_to_node: Dict[str, onnx.NodeProto],
                                  map_output_tensor_to_node: Dict[str, List[onnx.NodeProto]],
                                  start_marker: onnx.NodeProto):
        """
        Given a ONNX start_marker node, detach it from the graph
        :param map_input_tensor_to_node: Map of tensor to node consuming the tensor
        :param map_output_tensor_to_node: Map of tensor to node producing the tensor
        :param start_marker: Reference to the ONNX node to detach
        """
        assert len(start_marker.input) == 1
        assert len(start_marker.output) == 1

        input_tensor = start_marker.input[0]
        output_tensor = start_marker.output[0]

        for next_node in map_input_tensor_to_node[output_tensor]:

            index = list(next_node.input).index(output_tensor)
            next_node.input.remove(output_tensor)
            next_node.input.insert(index, input_tensor)
            map_input_tensor_to_node[input_tensor].append(next_node)

        map_input_tensor_to_node[input_tensor].remove(start_marker)
        del map_output_tensor_to_node[output_tensor]        # No node should produce output tensor anymore
        del map_input_tensor_to_node[output_tensor]         # No node should consume output tensor anymore

        start_marker.input.pop()
        start_marker.output.pop()

    @classmethod
    def _detach_end_marker_node(cls, map_input_tensor_to_node: Dict[str, onnx.NodeProto],
                                map_output_tensor_to_node: Dict[str, List[onnx.NodeProto]],
                                graphs_list: List[onnx.GraphProto], output_names_for_all_graphs: List[List[str]],
                                end_marker: onnx.NodeProto):
        """
        Given a ONNX end_marker node, detach it from the graph
        :param onnx_model: ONNX model instance
        :param map_input_tensor_to_node: Map of tensor to node consuming the tensor
        :param map_output_tensor_to_node: Map of tensor to node producing the tensor
        :param end_marker: Reference to the ONNX node to detach
        """
        assert len(end_marker.input) == 1
        assert len(end_marker.output) == 1

        input_tensor = end_marker.input[0]
        output_tensor = end_marker.output[0]

        found_tensor = False
        for idx, output_names_list in enumerate(output_names_for_all_graphs):
            if output_tensor in output_names_list:
                graph = graphs_list[idx]
                # Degenerate case: somebody did a "return y, y" at the end of the model or something similar
                for index, model_output in enumerate(output_names_list):
                    if model_output == output_tensor:
                        graph.output[index].name = input_tensor
                found_tensor = True
                break

        if not found_tensor:
            for next_node in map_input_tensor_to_node[output_tensor]:
                index = list(next_node.input).index(output_tensor)
                next_node.input.remove(output_tensor)
                next_node.input.insert(index, input_tensor)
                map_input_tensor_to_node[input_tensor].append(next_node)

        map_input_tensor_to_node[input_tensor].remove(end_marker)
        if not map_input_tensor_to_node[input_tensor]:
            del map_input_tensor_to_node[input_tensor]

        del map_output_tensor_to_node[output_tensor]        # No node should produce output tensor anymore
        if output_tensor in map_input_tensor_to_node:
            del map_input_tensor_to_node[output_tensor]     # No node should consume output tensor anymore

        end_marker.input.pop()
        end_marker.output.pop()

    @staticmethod
    def _set_output_names(onnx_model: onnx.ModelProto, graphs_list: List[onnx.GraphProto],
                          output_names_for_all_graphs: List[List[str]],
                          map_output_tensor_to_node: Dict[str, List[onnx.NodeProto]],
                          map_input_tensor_to_node: Dict[str, onnx.NodeProto]):
        """
        Set output names for outputs of the onnx model. Also recursively sets output names for subgraphs in the model.
        :param onnx_model: Model to set output names for
        :param graphs_list: List of graphs with outputs
        :param output_names_for_all_graphs: List of list of output names for corresponding graphs
        :param map_output_tensor_to_node: Dictionary mapping onnx tensor to node that generated it
        :param map_input_tensor_to_node: Dictionary mapping onnx tensor to nodes that consume it
        """

        OnnxSaver._set_output_names_for_graph(onnx_model.graph, graphs_list, output_names_for_all_graphs,
                                              map_output_tensor_to_node, map_input_tensor_to_node)

        for node in onnx_model.graph.node:
            for attribute in node.attribute:
                if getattr(attribute, 'g', None) is not None:
                    OnnxSaver._set_output_names_for_graph(attribute.g, graphs_list, output_names_for_all_graphs,
                                                          map_output_tensor_to_node, map_input_tensor_to_node)

    @staticmethod
    def _set_output_names_for_graph(onnx_graph: onnx.ModelProto, graphs_list, output_names_for_all_graphs,
                                    map_output_tensor_to_node, map_input_tensor_to_node):
        """
        Set output names for the given onnx graph
        :param onnx_graph: Graph to set output names for
        :param graphs_list: List of graphs with outputs
        :param output_names_for_all_graphs: List of list of output names for corresponding graphs
        :param map_output_tensor_to_node: Dictionary mapping onnx tensor to node that generated it
        :param map_input_tensor_to_node: Dictionary mapping onnx tensor to nodes that consume it
        """
        try:
            graph_index = graphs_list.index(onnx_graph)
        except ValueError:
            return

        desired_model_output_names = output_names_for_all_graphs[graph_index]
        # Iterate over the model outputs
        for index, output in enumerate(onnx_graph.output):
            new_tensor = desired_model_output_names[index]
            old_tensor = output.name

            if old_tensor == new_tensor:  # Nothing to do
                continue

            if old_tensor in map_input_tensor_to_node:
                # Degenerate case: model output tensor also is an intermediate tensor that inputs into other nodes
                for consumer in map_input_tensor_to_node[old_tensor]:
                    index = list(consumer.input).index(old_tensor)
                    consumer.input.remove(old_tensor)
                    consumer.input.insert(index, new_tensor)
                    if new_tensor not in map_input_tensor_to_node:
                        map_input_tensor_to_node[new_tensor] = []
                    map_input_tensor_to_node[new_tensor].append(consumer)

                del map_input_tensor_to_node[old_tensor]  # No node should consume old tensor anymore

            producer = map_output_tensor_to_node[old_tensor]

            output.name = new_tensor
            index = list(producer.output).index(old_tensor)
            producer.output.remove(old_tensor)
            producer.output.insert(index, new_tensor)

            del map_output_tensor_to_node[old_tensor]
            map_output_tensor_to_node[new_tensor] = producer

            # If there were duplicate outputs with the same name, they need to be updated
            for output_node in onnx_graph.output:
                # Ugly double loop - cannot avoid
                if output_node.name == old_tensor:
                    output_node.name = new_tensor

    @staticmethod
    def _get_all_nodes(onnx_graph: onnx.GraphProto, all_nodes: Union[List[onnx.NodeProto], None] = None) -> \
            List[onnx.NodeProto]:
        """
        Get all nodes in an onnx graph, including nodes in subgraphs.
        :param onnx_graph: Onnx graph to get nodes for
        :param all_nodes: List of nodes in the graph
        :return: List of nodes in the graph
        """
        if all_nodes is None:
            all_nodes = []
        for node in onnx_graph.node:
            all_nodes.append(node)
            for attribute in node.attribute:
                if getattr(attribute, 'g', None) is not None:
                    OnnxSaver._get_all_nodes(attribute.g, all_nodes)
        return all_nodes

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
        root_nodes = dict()
        # onnx graph is maintained in topological order, the first occurrence of the onnx node with the module name will
        # be the root node of the recurrent module
        for node_name in recurrent_nodes:
            root_name = node_name.split('#')[0]
            if root_name not in root_nodes:
                root_nodes[root_name] = node_name

        for root_node in root_nodes.values():
            # Find nodes corresponding to all other layers of the recurrent node
            other_layers = [node for node in recurrent_nodes if node.startswith(root_node.split('#')[0])]

            # Remove the root node from other layers
            other_layers.remove(root_node)

            # sort the other layers using the depth value following the '#'
            def get_depth(layer_name):
                right_of_pound = layer_name.split('#')[1]
                but_ignore_dash = right_of_pound.split('-')[0]
                return int(but_ignore_dash)

            other_layers = sorted(other_layers, key=get_depth)

            # Append the io_tensors for all layers for the current root recurrent node, in order
            io_tensor_list = [node_to_io_tensor_name_map[root_node]]
            for layer in other_layers:
                io_tensor_list.append(node_to_io_tensor_name_map[layer])
                del node_to_io_tensor_name_map[layer]

            # Let's rename the root node, so we can identify it later when building encoding file e.g.
            del node_to_io_tensor_name_map[root_node]
            root_node = root_node.split('#')[0] + '#root_node'

            node_to_io_tensor_name_map[root_node] = io_tensor_list

    @staticmethod
    def _get_all_initializers(onnx_graph: onnx.GraphProto, initializers: Union[List[onnx.TensorProto], None] = None) \
            -> List[onnx.TensorProto]:
        """
        Get all initializer names in the onnx graph. Also recursively gets initializer names for subgraphs of the graph.
        :param onnx_graph: Onnx graph to get initializer names for
        :param initializer_names: List of initializer names in the graph
        :return List of initializer names in the graph
        """
        if initializers is None:
            initializers = []
        for initializer in onnx_graph.initializer:
            initializers.append(initializer)
        for node in onnx_graph.node:
            for attribute in node.attribute:
                if getattr(attribute, 'g', None) is not None:
                    OnnxSaver._get_all_initializers(attribute.g, initializers)
        return initializers

    @staticmethod
    def _populate_node_to_io_tensor_map_and_valid_param_set(
            onnx_graph: onnx.GraphProto, initializer_names: List[str],
            node_to_io_tensor_name_map: Dict[str, OpToIOTensors],
            valid_param_set: Set[str]):
        """
        Populate the node name to io tensor map and the valid param set.
        :param onnx_graph: Onnx graph used to populate maps
        :param initializer_names: List of initializer names in the graph
        :param node_to_io_tensor_name_map: Map of onnx node names to input and output tensors
        :param valid_param_set: Set containing valid parameter names in the graph
        """
        for node in onnx_graph.node:
            if node.name:
                onnx_node_io_tensors = OpToIOTensors(list(node.input), list(node.output))
                if (node.name not in node_to_io_tensor_name_map) or node.op_type in recurrent_onnx_optypes:
                    node_to_io_tensor_name_map[node.name] = onnx_node_io_tensors

            # update valid params list
            for input_tensor in list(node.input):
                if input_tensor in initializer_names:
                    valid_param_set.add(input_tensor)

            for attribute in node.attribute:
                if getattr(attribute, 'g', None) is not None:
                    OnnxSaver._populate_node_to_io_tensor_map_and_valid_param_set(attribute.g, initializer_names,
                                                                                  node_to_io_tensor_name_map,
                                                                                  valid_param_set)

    @classmethod
    def get_onnx_node_to_io_tensor_names_map(cls, onnx_model: onnx.NodeProto) -> \
            (Dict[str, Union[OpToIOTensors, List[OpToIOTensors]]], Set[str]):
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
        initializer_names = [ini.name for ini in OnnxSaver._get_all_initializers(onnx_model.graph)]
        OnnxSaver._populate_node_to_io_tensor_map_and_valid_param_set(onnx_model.graph, initializer_names,
                                                                      node_to_io_tensor_name_map, valid_param_set)
        cls._collate_io_tensors_for_multi_layer_recurrent_nodes(onnx_model, node_to_io_tensor_name_map)

        return node_to_io_tensor_name_map, valid_param_set
