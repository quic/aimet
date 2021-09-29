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

""" Implementation to automatically replace functional by module """

from re import search
from typing import Any, Optional, Dict
import torch
import torch.fx
import aimet_torch.elementwise_ops as elementwise_ops


functional_to_module_map = {
    'relu': torch.nn.ReLU,
    'add': elementwise_ops.Add
}


def replace_functional_by_module(model: torch.nn.Module, concrete_args: Optional[Dict[str, Any]] = None) ->\
        torch.fx.GraphModule:
    """
    This function replaces functional by nn.Modules in given model and returns GraphModule
    :param model: torch Model to be modified
    :param concrete_args: Inputs to be partially specialized
    :return: Modified Model
    """
    unique_nodes = set()

    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_trace = torch.fx.symbolic_trace(model, concrete_args)

    # Modify the symbolic_trace by iterating over all the nodes
    for node in symbolic_trace.graph.nodes:
        if node.op in ['call_function', 'call_method']:
            for functional_name in functional_to_module_map:
                if search(functional_name, str(node.target)):
                    with symbolic_trace.graph.inserting_after(node):
                        new_node_name = 'module_' + node.name
                        setattr(symbolic_trace, new_node_name, functional_to_module_map[functional_name]())
                        new_node = symbolic_trace.graph.call_module(new_node_name, args=node.args)
                        node.replace_all_uses_with(new_node)
                    symbolic_trace.graph.erase_node(node)

        elif node.target in unique_nodes:
            if node.op == 'call_module':
                for functional_name in functional_to_module_map:
                    if search(functional_name, str(node.target)):
                        with symbolic_trace.graph.inserting_after(node):
                            new_node_name = 'module_' + node.name
                            setattr(symbolic_trace, new_node_name, functional_to_module_map[functional_name]())
                            new_node = symbolic_trace.graph.call_module(new_node_name, args=node.args)
                            node.replace_all_uses_with(new_node)
                        symbolic_trace.graph.erase_node(node)
        else:
            unique_nodes.add(node.target)

    # Does some checks to make sure the graph is well formed
    symbolic_trace.graph.lint()

    # Recompile the forward() method of symbolic_trace from its Graph
    symbolic_trace.recompile()

    return symbolic_trace
