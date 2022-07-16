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
        warning_message = ('Functional ops were found in the model. Different AIMET features will expect ops of '
                           'certain types to be defined as torch.nn modules.\n'
                           'AIMET features which operate on the op may not work as intended. As an example, quantsim '
                           'will not be able to wrap functional ops and simulate quantization noise for them.\n'
                           'Consider the following choices: \n'
                           '1. The op can be redefined as a torch.nn.Module in the class definition.\n'
                           '2. The op can remain as a functional op due to not being an op type of interest, but the '
                           'op type has not been added to ConnectedGraph.functional_ops. \n'
                           'Add an entry to ignore the op.\n')
        warning_message += f'The following functional ops were found. The parent module is named for ease of ' \
                           f'locating the ops within the model definition.\n'
        max_name_len = 0
        for op in ops_with_missing_modules:
            if len(op.name) > max_name_len:
                max_name_len = len(op.name)
        for op in ops_with_missing_modules:
            warning_message += f'{op.name}{" " * (max_name_len + 10 - len(op.name))}parent module: ' \
                                      f'{op.residing_module}\n'
        logger.warning(warning_message)
    return not ops_with_missing_modules
