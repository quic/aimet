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

""" Implementation to automatically prepare pytorch models for AIMET features """

import copy
from re import search
from typing import Any, Optional, Dict, Union
import torch
import torch.fx
from aimet_common.utils import AimetLogger
from aimet_torch.utils import get_device
import aimet_torch.elementwise_ops as elementwise_ops

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

functional_to_module_map = {

    # Non Linear activation functions
    'relu'          : torch.nn.ReLU,
    'relu6'         : torch.nn.ReLU6,
    'hardtanh'      : torch.nn.Hardtanh,
    'hardwish'      : torch.nn.Hardswish,
    'elu'           : torch.nn.ELU,
    'selu'          : torch.nn.SELU,
    'celu'          : torch.nn.CELU,
    'leaky_relu'    : torch.nn.LeakyReLU,
    'prelu'         : torch.nn.PReLU,
    'rrelu'         : torch.nn.RReLU,
    'glu'           : torch.nn.GLU,
    'gelu'          : torch.nn.GELU,
    'logsigmoid'    : torch.nn.LogSigmoid,
    'hardshrink'    : torch.nn.Hardshrink,
    'tanhshrink'    : torch.nn.Tanhshrink,
    'softsign'      : torch.nn.Softsign,
    'softplus'      : torch.nn.Softplus,
    'softmin'       : torch.nn.Softmin,
    'softmax'       : torch.nn.Softmax,
    'softshrink'    : torch.nn.Softshrink,
    'log_softmax'   : torch.nn.LogSoftmax,
    'tanh'          : torch.nn.Tanh,
    'sigmoid'       : torch.nn.Sigmoid,
    'hardsigmoid'   : torch.nn.Hardsigmoid,
    'silu'          : torch.nn.SiLU,

    # Elementwise operations
    'add'           : elementwise_ops.Add,
    'subtract'      : elementwise_ops.Subtract,
    'mul'           : elementwise_ops.Multiply,
    'div'           : elementwise_ops.Divide,
    'cat'           : elementwise_ops.Concat,
    'matmul'        : elementwise_ops.MatMul
}


def prepare_model(model: torch.nn.Module, concrete_args: Optional[Dict[str, Any]] = None) -> torch.fx.GraphModule:
    """
    Prepare and modify the pytorch model for AIMET features using torch.FX symbolic tracing API.

    #1 Replace torch.nn.functional by torch.nn.Module.
    #2 Create new independent torch.nn.Module instances for reused/duplicate module.

    Example #1 Replace torch.nn.functional by torch.nn.module::

        class ModelWithFunctionalReLU(torch.nn.Module):

            def __init__(self):
                super(ModelWithFunctionalReLU, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3)

            def forward(self, *inputs):
                x = self.conv1(inputs[0])
                x = torch.nn.functional.relu(x, inplace=True)
                return x

        model = ModelWithFunctionalReLU().eval()
        model_transformed = prepare_model(model)

    This function can replace the ReLU of type torch.nn.functional by type torch.nn.Module and make sure
    both the modified and original model are functionally same.

    Example #2 Create new module for reused/duplicate module::

        class ModelWithDuplicateReLU(torch.nn.Module):

            def __init__(self):
                super(ModelWithDuplicateReLU, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3)
                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, *inputs):
                x = self.relu(inputs[0])
                x = self.conv1(x)
                x = self.relu(x)
                return x

        model = ModelWithDuplicateReLU().eval()
        model_transformed = prepare_model(model)

    This function can create new independent torch.nn.ReLU type module for reused module and make sure
    both the modified and original model are functionally same.

    Limitations of torch.fx symbolic trace API:

    #1 Dynamic control flow where conditions depend on some of the input values. This limitation can be overcome by
    binding concrete values to arguments during symbolic tracing::

        def f(x, flag):
            if flag: return x
            else: return x*2

        torch.fx.symbolic_trace(f) # Fails!
        torch.fx.symbolic_trace(f, concrete_args={'flag': True}) # Passes!

    #2 Non-torch functions which does not use __torch_function__ mechanism is not supported by default in symbolic
    tracing. If we do not want to capture them in symbolic tracing then use torch.fx.wrap() API at module-scope level::

        import torch
        import torch.fx
        torch.fx.wrap('len')  # call the API at module-level scope.
        torch.fx.wrap('sqrt') # call the API at module-level scope.

        class ModelWithNonTorchFunction(torch.nn.Module):
            def __init__(self):
                super(ModelWithNonTorchFunction, self).__init__()
                self.conv = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2, bias=False)

            def forward(self, *inputs):
                x = self.conv(inputs[0])
                return x / sqrt(len(x))

        model = ModelWithNonTorchFunction().eval()
        model_transformed = prepare_model(model)

    :param model: pytorch Model to be modified
    :param concrete_args: Allows you to partially specialize your function, whether it's to remove control flow or
     data structures. If the model has control flow, torch.fx won't be able to trace the model. Check
     torch.fx.symbolic_trace API in detail.
    :return: Modified pytorch Model
    """
    model.eval()
    device = get_device(model)
    # Create a copy of model and keep it on cpu
    model_copy = copy.deepcopy(model).cpu()

    unique_nodes = set()
    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_traced_model = torch.fx.symbolic_trace(model_copy, concrete_args)

    # Modify the symbolically traced model by iterating over all the nodes
    for node in symbolic_traced_model.graph.nodes:

        # Create new module for functional nodes
        if node.op in ['call_function', 'call_method']:
            functional_name = _find_functional_name_for_node(node)
            if functional_name:
                # Instantiate new module for functional node
                new_module = _create_module_for_functional_node(node, functional_name)
                new_nodule_name = 'module_' + node.name
                setattr(symbolic_traced_model, new_nodule_name, new_module)
                # Create the node for new module in the graph
                _create_node_for_new_module(symbolic_traced_model, node, new_nodule_name)
                logger.info("Functional         : Adding new module for node: {%s} ", node.name)

        # Create new module for reused/duplicate nodes
        elif node.target in unique_nodes:
            if node.op == 'call_module':
                # Instantiate new module for reused node
                new_module = _create_module_for_reused_node(node, symbolic_traced_model)
                new_nodule_name = 'module_' + node.name
                setattr(symbolic_traced_model, new_nodule_name, new_module)
                # Create the node for new module in the graph
                _create_node_for_new_module(symbolic_traced_model, node, new_nodule_name)
                logger.info("Reused/Duplicate   : Adding new module for node: {%s} ", node.name)
        else:
            unique_nodes.add(node.target)

    # Perform some checks to make sure the graph is well formed
    _verify_symbolic_traced_model(symbolic_traced_model)

    symbolic_traced_model.eval()
    symbolic_traced_model.to(device)
    return symbolic_traced_model


def _verify_symbolic_traced_model(symbolic_traced_model: torch.fx.GraphModule):
    """
    Does some checks to make sure the graph is well formed and recompile the forward() method of symbolic_traced
    model from its graph
    :param symbolic_traced_model: Symbolically traced model
    :return: None
    """
    symbolic_traced_model.graph.lint()
    symbolic_traced_model.recompile()


def _create_node_for_new_module(symbolic_traced_model: torch.fx.GraphModule, node: torch.fx.node,
                                module_name: str):
    """
    Insert 'call module' node into graph and replace all the uses of 'node' with newly added node and erase the
    the old node from graph.
    :param symbolic_traced_model: Symbolically traced model
    :param node: Current node in the graph after which new node will be inserted
    :param module_name: Qualified module name in symbolic_traced_model hierarchy corresponding to new node
    :return: None
    """
    with symbolic_traced_model.graph.inserting_after(node):
        new_node = symbolic_traced_model.graph.call_module(module_name, args=node.args)
        node.replace_all_uses_with(new_node)
    symbolic_traced_model.graph.erase_node(node)


def _find_functional_name_for_node(node: torch.fx.node) -> Union[str, None]:
    """
    For given node, find corresponding functional name from functional_to_module lookup
    :param node: torch.fx Node
    :return: corresponding functional name if found, else None
    """
    for functional_name in functional_to_module_map:

        # \b boundary character to find the exact match from the functional_to_module lookup
        pattern = r"\b" + functional_name + r"\b"
        if search(pattern, str(node.target)):
            return functional_name

    return None


def _create_module_for_functional_node(node: torch.fx.node, functional_name: str) -> torch.nn.Module:
    """
    For given node and functional name, create torch.nn.Module with same parameters as functional node parameters
    :param node: torch.fx Node
    :param functional_name: Functional name for given node
    :return: New module
    """
    kwargs = node.kwargs

    # Instantiate new module from lookup
    module = functional_to_module_map[functional_name]()

    # Set the parameters for module from node.kwargs
    for key, value in kwargs.items():
        setattr(module, key, value)

    return module


def _create_module_for_reused_node(node: torch.fx.node, symbolic_traced_model: torch.fx.GraphModule) ->\
        torch.nn.Module:
    """
    For given reused/Duplicate node in symbolically traced model, create new module with same parameters as
    original module
    :param node: Reused/Duplicate torch.fx Node
    :param symbolic_traced_model: Symbolically traced model
    :return: New module
    """
    # Get the original module and return newly deep copied module
    module = _get_module_for_dotted_name(symbolic_traced_model, node.target)
    new_module = copy.deepcopy(module)

    return new_module


def _get_module_for_dotted_name(module: torch.fx.GraphModule, dotted_name: str) -> torch.nn.Module:
    """
    For given dotted name, find the module
    :param module: module to be found
    :param dotted_name: dotted name of module
    :return: module
    """
    if '.' in dotted_name:
        module_name, _, remainder = dotted_name.partition('.')
        return _get_module_for_dotted_name(module._modules[module_name], remainder) # pylint: disable=protected-access

    return getattr(module, dotted_name)
