# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Manual mixed precision configurator """
from typing import overload, Union, Literal, TypeAlias, List, Tuple, Dict, get_args, Type
from dataclasses import dataclass, field
import torch

from aimet_common.defs import QuantizationDataType
from aimet_torch.v2.quantsim import QuantizationSimModel

SupportedDType: TypeAlias = Literal['Int16', 'Int8', 'Int4', 'Fp16']


@dataclass
class MpRequest:
    """ Internal data structure to save the request to act upon"""
    id: int = -1  # original request ID
    input_candidates: List[Tuple[QuantizationDataType, int]] = field(default_factory=list)
    output_candidate: Tuple[QuantizationDataType, int] = field(default_factory=tuple)
    param_candidate: Dict[str, Tuple[QuantizationDataType, int]] = field(default_factory=dict)


class MixedPrecisionConfigurator:
    """
    Mixed Precision Configurator helps set up a mixed precision profile in the QuantSim object. The user is expected to
    follow the below steps to set the sim in Mixed Precision.

    1. Create QuantSim object
    2. Create the MixedPrecisionConfigurator object by passing in the QuantSim object
    3. Make a series of set_precision/set_model_input_precision/set_model_output_precision calls
    4. Call apply() method by passing in the config file and strict flag
    5. Run compute_encodings on the above QuantSim object
    6. Export the encodings/onnx artifacts
    """

    def __init__(self, sim: QuantizationSimModel):
        """
        :param sim: QuantSim object
        """
        self._sim = sim

    @overload
    def set_precision(self, module: torch.nn.Module, act_candidate: SupportedDType = None,
                      param_candidate: SupportedDType = None): ...

    @overload
    def set_precision(self, module: Type[torch.nn.Module], act_candidate: SupportedDType = None,
                      param_candidate: SupportedDType = None): ...

    def set_precision(self, module: Union[torch.nn.Module, Type[torch.nn.Module]], act_candidate: SupportedDType = None,
                      param_candidate: SupportedDType = None):
        """
        :param module: Module can be of type torch.nn.Module or the type of the module.
        :param act_candidate: A string representing the activation dtype of the module(s)
        :param param_candidate: A string representing the param dtype of the module(s)

        - If the 'module' is a leaf-module(the module doesnt compose of other torch.nn.module), the specified settings
        would be applied to the module.
        - If the 'module' is a non-leaf-module (module is composed of other torch.nn.module), the specified settings
        would be applied to all the leaf modules in 'module'.
        - If the 'module' is Type of module, all the modules in the model which satisfy the specified module type would
        be set to the specified activation and param settings
        - If the same 'module' is specified through multiple set_precision(...) calls, the latest one will be applied.

        Examples: TODO

        """

        if act_candidate and act_candidate not in get_args(SupportedDType):
            raise ValueError("Supported inputs for act_candidate are ", get_args(SupportedDType))
        if param_candidate and param_candidate not in get_args(SupportedDType):
            raise ValueError("Supported inputs for param_candidate are ", get_args(SupportedDType))

        assert module in self._sim.model.modules()


    def set_model_input_precision(self, act_candidate):
        """
        Activation precision which needs to be set to the model inputs
        :param act_candidate: Activation dtypes for all the inputs of the model
        """

    def set_model_output_precision(self, act_candidate):
        """
        Activation precision which needs to be set to the model outputs
        :param act_candidate: Activation dtypes for all the outputs of the model
        """

    def apply(self, config: str = "", strict: bool = True, log_file: str = './mmp_log.txt'):
        """
        Apply the mp settings specified through the set_precision/set_model_input_precision/set_model_output_precision
        calls to the QuantSim object
        :param config: Config file to be used for backend awareness. If empty no backend awareness would be checked
        :param strict: Boolean flag to indicate whether to fail (strict=True) on incorrect/conflicting inputs made by
        the user or (strict=False) take a best-effort approach to realize the MP settings
        :param log_file: Log file to store the logs
        """
