# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Utilities for ConnectedGraph """

from typing import Tuple, Union, List, Dict
import torch

# Import AIMET specific modules
from aimet_torch.meta.connectedgraph import ConnectedGraph

ActivationFunctionMap = {
        'relu': torch.nn.ReLU(),
        'relu6': torch.nn.ReLU6(),
        'sigmoid': torch.nn.Sigmoid(),
        'tanh': torch.nn.Tanh(),
        'hardtanh': torch.nn.Hardtanh()
}

def get_module_act_func_pair(model: torch.nn.Module, inp_tensor: Union[Tuple[torch.Tensor], List[torch.Tensor]]) -> \
        Dict[torch.nn.Module, Union[torch.nn.Module, None]]:
    """
    For given model, returns dictionary of module to immediate following activation function else mops
    module to None

    :param model: model
    :param inp_tensor:  list/tuple of input tensor to model
    :return: dictionary of module to activation function
    """

    def get_act_func_from_name(op_name: str) -> Union[torch.nn.Module, None]:
        """
        For given op name from connected graph return following activation function else return None
        :param op_name: op name
        :return: activation function or None
        """
        for act_func in ActivationFunctionMap:

            if act_func in op_name:
                return ActivationFunctionMap[act_func]

        return None

    model.eval()

    # Create the ConnectedGraph
    graph = ConnectedGraph(model, inp_tensor)

    # Maps module to next following activation function else None
    module_act_func_pair = {}

    # get all the ops
    all_ops = graph.get_all_ops()

    for _, op in all_ops.items():
        for _, module in model.named_children():

            if module == op.get_module():
                module_act_func_pair[module] = None

                if op.output:
                    assert len(op.output.consumers) == 1, 'op output should have at least one consumer.'
                    # get the next op
                    next_op = op.output.consumers[0]
                    # get the activation function
                    activation_function = get_act_func_from_name(next_op.name_op)

                    module_act_func_pair[module] = activation_function

    return module_act_func_pair
