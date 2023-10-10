# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Utility to classify MultiHeadAttention (MHA) module(s) """

from collections import deque
from dataclasses import dataclass
from typing import List, Union, Tuple, Optional
import torch

from aimet_torch.meta.operation import Op
from aimet_torch.meta.connectedgraph import ConnectedGraph


@dataclass
class MhaInfo:
    """
    MHA info contains MHA module type and qualified name.
    """
    type: str
    module_qualified_name: str


def find_mha_variant(model: torch.nn.Module,
                     dummy_input: Union[torch.Tensor, Tuple],
                     pattern: List[str]) -> Optional[List[MhaInfo]]:
    """
    For given model and MHA variant pattern, classify and locate the MHA variants in the model if it exists.

    NOTE: both model and MHA variant should be torch.jit.traceable.

    :param model: torch model.
    :param dummy_input: Dummy input to the model.
    :param pattern: A pattern is list of ordered connected graph op types for MHA variant to be found.
    :return: List of MHAInfo which consists of MHA variant's type and qualified name.
    """
    conn_graph = ConnectedGraph(model, dummy_input)
    mha_modules_info = pattern_exists(conn_graph.ordered_ops, pattern)
    return mha_modules_info


def pattern_exists(ordered_ops: List[Op],
                   pattern: List[str]) -> Optional[List[MhaInfo]]:
    """
    Determine if the ordered ops contain the given pattern or not using sliding window approach.

    :param ordered_ops: Orderered connected graph ops.
    :param pattern: A pattern is list of connected graph op types in order of occurence.
    :return: List of MHAInfo which consists of mha's type and qualified name.
    """
    mha_modules_info = []
    sliding_window = deque(maxlen=len(pattern))
    for index, op in enumerate(ordered_ops):
        sliding_window.append(op)
        sliced_pattern = [op.type for op in sliding_window]
        if sliced_pattern == pattern:
            _, parent_name = ordered_ops[index].dotted_name.split(".", 1)
            module_qualified_name, _ = parent_name.rsplit(".", 1)
            mha_info = MhaInfo(type(ordered_ops[index].residing_module),
                               module_qualified_name)
            mha_modules_info.append(mha_info)

    return mha_modules_info
