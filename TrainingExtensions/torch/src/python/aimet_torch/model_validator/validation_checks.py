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

""" Functions for validating pytorch models prior to using AIMET features """

from typing import Tuple, Union
import torch

from aimet_common.utils import AimetLogger
from aimet_torch import utils
from aimet_torch.meta import connectedgraph_utils


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def validate_for_reused_modules(model: torch.nn.Module, model_input: Union[torch.Tensor, Tuple]) -> bool:
    """
    Check if the model has any reused modules. Returns True if there are none, False otherwise.
    :param model: Pytorch model to create connected graph from
    :param model_input: Example input to model.  Can be a single tensor or a list/tuple of input tensors
    :return: True if model has no reused modules, False otherwise
    """
    reused_modules = utils.get_reused_modules(model, model_input)
    if reused_modules:
        logger.warning('The following modules are used more than once in the model: %s\n'
                       'AIMET features are not designed to work with reused modules. Please redefine your model '
                       'to use distinct modules for each instance.', [name for (name, _) in reused_modules])
    return not reused_modules


def validate_for_missing_modules(model: torch.nn.Module, model_input: Union[torch.Tensor, Tuple]) -> bool:
    """
    Check if the model has any ops with missing modules (excluding a set of known ops which can be functionals).
    Returns True if there are no ops with missing modules, False otherwise.
    :param model: Pytorch model to create connected graph from
    :param model_input: Example input to model.  Can be a single tensor or a list/tuple of input tensors
    :return: True if model has no ops with missing modules, False otherwise.
    """
    ops_with_missing_modules = connectedgraph_utils.get_ops_with_missing_modules(model, model_input)
    if ops_with_missing_modules:
        # TODO: replace with logger.error and assertion after rewriting unit tests to avoid using built in vgg,
        #  resnet, and inception models (since they use functionals in their models)
        logger.warning('Ops with missing modules: %s\n'
                       'This can be due to several reasons:\n'
                       '1. There is no mapping for the op in ConnectedGraph.op_type_map. Add a mapping for '
                       'ConnectedGraph to recognize and be able to map the op.\n'
                       '2. The op is defined as a functional in the forward function, instead of as a class '
                       'module. Redefine the op as a class module if possible. Else, check 3.\n'
                       '3. This op is one that cannot be defined as a class module, but has not been added to '
                       'ConnectedGraph.functional_ops. Add to continue.'
                       , ops_with_missing_modules)
    return not ops_with_missing_modules
