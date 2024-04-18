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

""" Implementation for simulating models running on Quantized hardware """
from typing import List, Union, Dict, Tuple
import torch

from aimet_common.utils import AimetLogger
from aimet_torch.defs import OpToIOTensors
from aimet_torch.utils import is_leaf_module, run_hook_for_layers_with_given_input
from aimet_torch.meta.connectedgraph import ConnectedGraph

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# Map torch module types to normalized names to provide backward compatibility to
# trace code based construction
op_type_map = {
    torch.nn.Conv2d: 'convolution',
    torch.nn.ConvTranspose2d: 'convolution',
    torch.nn.BatchNorm1d: 'batch_norm',
    torch.nn.BatchNorm2d: 'batch_norm',
    torch.nn.ReLU: 'relu',
    torch.nn.ReLU6: 'hardtanh',
    torch.nn.MaxPool2d: 'max_pool2d',
    torch.nn.AdaptiveAvgPool2d: 'adaptive_avg_pool2d',
    torch.nn.AvgPool2d: 'avg_pool2d',
    torch.nn.Linear: 'addmm',
    torch.nn.Dropout: 'dropout',
    torch.nn.Dropout2d: 'feature_dropout',
    torch.nn.LogSoftmax: 'log_softmax',
    torch.nn.Sigmoid: 'sigmoid'
}


# pylint: disable=protected-access
class IrNode:
    """
    Representation for a module in torch graph.
    """
    def __init__(self, node_type: str, inputs: List[Union[List, torch._C.TensorType]],
                 outputs: List[Union[List, torch._C.TensorType]], module: torch.nn.Module):
        self.node_type = node_type
        self.inputs = inputs
        self.outputs = outputs
        self.module = module

    def __str__(self):
        return self.node_type


def _get_attribute_name(node: torch._C.Node) -> Dict[str, str]:
    """
    Retrieve the attributes associated with the graph node
    :param node: trace graph node
    :return: a dictionary of attributes associated with the node
    """
    attributes = {}
    # node description has pseudo-code of the form  '... torch_mangle_2.Module = prim::GetAttr[name="fc"](%self.1)'
    # for the above example attributeNames() iterator should return a string 'name'
    node_desc = str(node)
    for attribute_name in node.attributeNames():
        pattern = attribute_name + '="'
        if pattern in node_desc:
            attributes[attribute_name] = node_desc.split(pattern)[1].split('"')[0]
    return attributes


def _get_module_instance(node: torch._C.Node,
                         node_name_to_module: Dict[str, torch.nn.Module]) -> torch.nn.Module:
    """
    Get the torch.nn.Module referenced by the node.
    :param node: trace graph node
    :param node_name_to_module: dictionary of module index by output_name referenced in the sub-graph
    :return: torch module corresponding to the node
    """
    input_name: str = node.input().debugName()
    attributes = _get_attribute_name(node)
    model = node_name_to_module[input_name]
    sub_model = getattr(model, attributes['name'])
    return sub_model


def _parse_graph(graph: torch._C.Graph, model: torch.nn.Module) -> List[IrNode]:
    """
    Implements a depth-first graph extraction to obtain connectivity information in the form of an IrNodes list.
    Depth-first extraction is realized using recursion.

    :param trace: Pytorch JIT trace for model or a submodule
    :param model: Pytorch model to create connected graph from
    :return List of IrNodes created from traversing the trace graph
    """
    ir_nodes_list = []
    # pylint: disable=unnecessary-comprehension
    curr_inputs = [inp for inp in graph.inputs()]

    # A map of sub-graph models and node name that requires recursive parsing
    # modules that are being referenced within the sub-graph
    node_name_to_module = {curr_inputs[0].debugName(): model}
    for node in graph.nodes():
        # pylint: disable=unnecessary-comprehension
        outputs = [output for output in node.outputs()]

        # retrieving a module reference
        if 'GetAttr' in node.kind():
            # For GetAttr lines, the output name will be referring to the module, and not the module's output(s)
            assert len(outputs) == 1
            node_name = outputs[0].debugName()
            assert node_name not in node_name_to_module
            module = _get_module_instance(node, node_name_to_module)
            node_name_to_module[node_name] = module
        else:
            op_type: str = ConnectedGraph._parse_op_type(node)
            if "Constant" not in op_type:
                # pylint: disable=unnecessary-comprehension
                outputs = [output for output in node.outputs()]
                ir_node = IrNode(node_type=op_type,
                                 inputs=[inp for inp in node.inputs() if
                                         "Constant" not in ConnectedGraph._parse_op_type(inp.node())],
                                 outputs=outputs,
                                 module=None)
                ir_nodes_list.append(ir_node)

    for ir_node in ir_nodes_list:
        inputs = []
        for inp in ir_node.inputs:
            if "GetAttr" in inp.node().kind():
                if ir_node.node_type in op_type_map.values():
                    module = node_name_to_module[inp.node().input().debugName()]
                    assert is_leaf_module(module)
                    if ir_node.module is None:
                        ir_node.module = module
                    else:
                        assert ir_node.module == module
            else:
                inputs.append(inp)
        ir_node.inputs = inputs

    return ir_nodes_list


def _coalesce_add_and_mm_nodes(ir_nodes_list: List[IrNode]):
    """
    helper method to combine add and mm operation into addmm to map back to fc layer
    :param ir_nodes_list: List of ir_nodes to update connections for
    """

    del_node_indices = []
    for i, ir_node in enumerate(ir_nodes_list):
        if ir_node.node_type == 'add' and len(ir_node.inputs) == 1:
            producer_ir_node = ir_nodes_list[i-1]
            if producer_ir_node.node_type == 'mm' and len(producer_ir_node.outputs) == 1 and \
                    producer_ir_node.outputs[0] == ir_node.inputs[0]:
                producer_ir_node.outputs = ir_node.outputs
                producer_ir_node.node_type = 'addmm'
                del_node_indices.insert(0, i)

    for index in del_node_indices:
        del ir_nodes_list[index]


def get_node_to_io_tensor_names_map(model: torch.nn.Module,
                                    trace: torch.jit.TopLevelTracedModule,
                                    inputs: List[torch.Tensor]) -> \
        (Dict[str, Union[OpToIOTensors, List[OpToIOTensors]]], set):
    """
    Given an Torch model, gets the inputs and output tensor names for each node in the model.
    :param model: The user provided model instance
    :param trace: the mode in torch script format
    :param inputs: sample tensor inputs
    :return: Dictionary of torch script node name and corresponding input and output tensor names and
        a set with all valid param names in model
    """
    # pylint: disable=too-many-locals

    # Generates a look up dictionary for getting modules from their names.
    model_name = type(model).__name__
    module_to_name = {}
    for name, module in model.named_modules(prefix=model_name):
        module_to_name[module] = name
    if isinstance(inputs, torch.Tensor):
        graph = trace.graph_for(inputs)
    else:
        graph = trace.graph_for(*inputs)
    ir_nodes_list = _parse_graph(graph, model)
    _coalesce_add_and_mm_nodes(ir_nodes_list)

    node_to_io_tensor_name_map = {}
    valid_param_set = set()
    prefix_len = len(model_name) + 1

    modules = []

    def forward_hook(curr_module: torch.nn.Module, *_):
        """
        Custom forward hook function to add every module to module list.
        :param curr_module: Current module being traversed during forward pass.
        """
        if isinstance(curr_module, (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU)):
            raise NotImplementedError('exporting encoding for RNN module via torchscript not supported')

        if not isinstance(curr_module, torch.nn.Identity):
            modules.append(curr_module)

    run_hook_for_layers_with_given_input(model, inputs, forward_hook)
    index = 0
    # pylint: disable=unnecessary-comprehension
    module_types = [types for types in op_type_map.values()]
    for node in ir_nodes_list:
        if node.module is None:
            if node.node_type in module_types:
                node.module = modules[index]
                assert op_type_map[type(node.module)] == node.node_type
            else:
                continue
        module_name = module_to_name[node.module][prefix_len:]
        index = index + 1

        node_to_io_tensor_name_map[module_name] = \
            OpToIOTensors(
                [inp.debugName() for inp in node.inputs],
                [output.debugName() for output in node.outputs])

        for param_name, _ in node.module.named_parameters():
            valid_param_set.add(module_name + '.' + param_name)

    #assert index == len(modules)

    return node_to_io_tensor_name_map, valid_param_set


def create_torch_script_model(ts_path: str, original_model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple]):
    """
    This utility obtains an equivalent torchscript model for the given pytorch model. Whatever pre-processing/post-processing
    steps to be done on the resultant torchscript model must be done here.

    :param ts_path: Path to the torchscript model file
    :param original_model: Equivalent PyTorch model instance
    :param dummy_input: Dummy input to the model. Used to parse model graph
    :return:
    """
    trace = torch.jit.trace(original_model, dummy_input)
    trace.save(ts_path)
