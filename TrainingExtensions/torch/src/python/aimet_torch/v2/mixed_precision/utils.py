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
"""Utilities to achieve mixed precision"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Type, List, TypeAlias, Literal, Tuple, Optional, Union

import torch

from aimet_common.defs import QuantizationDataType
from aimet_common.utils import AimetLogger
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import BaseQuantizationMixin

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

SupportedDType: TypeAlias = Literal['Int16', 'Int8', 'Int4', 'Fp16']

TranslateUserDtypes = {
    'Int16': (QuantizationDataType.int, 16),
    'Int8': (QuantizationDataType.int, 8),
    'Int4': (QuantizationDataType.int, 4),
    'Fp16': (QuantizationDataType.float, 16),
}


@dataclass
class MpRequest:
    """ Internal data structure to save the request to act upon"""
    id: int = None  # original request ID
    input_candidates: List[Tuple[QuantizationDataType, int]] = field(default_factory=list)
    output_candidates: List[Tuple[QuantizationDataType, int]] = field(default_factory=list)
    param_candidate: Dict[str, Tuple[QuantizationDataType, int]] = field(default_factory=dict)


class RequestType(Enum):
    """Enum to represent the type of request made by the user"""
    set_precision_by_module = 1
    set_precision_by_module_type = 2
    set_model_input_precision = 3
    set_model_output_precision = 4


@dataclass
class UserRequest:
    """ Data structure to store user requests"""
    request_type: RequestType
    module: Union[torch.nn.Module, Type, None] = None
    activation: Union[List[SupportedDType], SupportedDType, None] = None
    param: Optional[Dict[str, SupportedDType]] = None


class MpHandler:
    """
    Mixed Precision handler provides the functionalities to generate the Mixed Precision profile from the user provided
    requests and apply to the sim

    """
    def __init__(self, sim: QuantizationSimModel):
        self._sim = sim
        self.mp_requests = {}

    @staticmethod
    def _get_candidate_from_user_dtype(user_dtype: Union[List[SupportedDType], SupportedDType, None] = None):
        """
        Converts user dtype to internal representation in AIMET (QuantizationDataType, Int)

        :param user_dtype: user input for an activation/param
        """
        candidate = None
        if user_dtype:
            if isinstance(user_dtype, List):
                candidate = []
                for dtype in user_dtype:
                    candidate.append(TranslateUserDtypes.get(dtype))
            else:
                candidate = TranslateUserDtypes.get(user_dtype)
        return candidate

    def _get_leaf_modules(self, torch_module: torch.nn.Module) -> List:
        """ Get all the leaf modules in the given module """
        for name, module in torch_module.named_modules():
            if module not in self._sim.model.modules():
                raise ValueError(f"Specified module {module} is not part of the sim object")
            if isinstance(module, BaseQuantizationMixin):
                yield name, module

    def _get_modules_of_type(self, module_type):
        """ Get all the modules of given type"""
        for name, module in self._sim.model.named_modules():
            if isinstance(module, BaseQuantizationMixin) and isinstance(module.get_original_module(), module_type):
                yield name, module

    def _process_user_requests(self, user_requests: Dict[int, UserRequest]):

        def create_mp_request(torch_module: BaseQuantizationMixin, module_name: str, idx: int,
                              activation: Union[List[SupportedDType], SupportedDType, None] = None,
                              param: Optional[Dict[str, SupportedDType]] = None):
            """ For a given leaf module, and the specified activation and param candidates, convert to MpRequest"""
            # TODO fill missing inputs
            if torch_module in mp_requests:
                prev_request = mp_requests[torch_module]
                logger.info(f"{module_name} was already encountered with request_id {prev_request.id} and request "
                            f"{user_requests[prev_request.id]}. This would be replaced with the new request "
                            f"{user_requests[idx]}")

            # multi-inputs would be wrong here
            input_candidates = self._get_candidate_from_user_dtype(activation)
            output_candidates = self._get_candidate_from_user_dtype(activation[0]) \
                if isinstance(activation, List) else self._get_candidate_from_user_dtype(activation)
            param_candidate = {}
            for param_name, dtype in param.items():
                param_candidate = {param_name: self._get_candidate_from_user_dtype(dtype)}
            mp_requests[torch_module] = MpRequest(id=idx, input_candidates=input_candidates,
                                                  output_candidates=output_candidates,
                                                  param_candidate=param_candidate)

        mp_requests = {}
        for request_id, user_request in user_requests.items():
            match user_request.request_type:
                case RequestType.set_precision_by_module_type:
                    for name, module in self._get_modules_of_type(user_request.module):
                        create_mp_request(module, name, request_id, user_request.activation,
                                          user_request.param)

                case RequestType.set_precision_by_module:
                    for name, module in self._get_leaf_modules(user_request.module):
                        create_mp_request(module, name, request_id, user_request.activation,
                                          user_request.param)

                case RequestType.set_model_input_precision:
                    ...

                case RequestType.set_model_output_precision:
                    ...

                case _:
                    raise RuntimeError(f"Unsupported request type {user_request.request_type} encountered")
        return mp_requests

    def _apply_backend_awareness(self, mp_requests: Dict, config: str = "", strict: bool = True) -> Dict:
        """
        Apply backend awareness to the requests from the user

        :param mp_requests: MP requests generated after processing user requests
        :param config: Config file to be used for backend awareness. If empty no backend awareness would be checked
        :param strict: Boolean flag to indicate whether to fail (strict=True) on incorrect/conflicting inputs made by
        the user or (strict=False) take a best-effort approach to realize the MP settings
        """
        return mp_requests

    def _propagate_requests(self, mp_requests: Dict, strict: bool = True) -> Dict:
        """
        Propagate requests to parent modules to achieve precision at given module

        :param mp_requests: MP requests generated after processing user requests
        :param strict: Boolean flag to indicate whether to fail (strict=True) on incorrect/conflicting inputs made by
        the user or (strict=False) take a best-effort approach to realize the MP settings
        """
        return mp_requests

    def _apply_to_sim(self, mp_requests: Dict):
        """
        Apply MP configuration to the sim object

        :param mp_requests: MP requests after preprocessing, applying backend awareness(if present), propagating to
        parent modules
        """

    def apply(self, user_requests: Dict[int, UserRequest], config: str = "", strict: bool = True,
              log_file: str = './mmp_log.txt'):
        """
        Apply the mp settings specified through the set_precision/set_model_input_precision/set_model_output_precision
        calls to the QuantSim object

        :param user_requests: Dict of request id and user request to apply to sim
        :param config: Config file to be used for backend awareness. If empty no backend awareness would be checked
        :param strict: Boolean flag to indicate whether to fail (strict=True) on incorrect/conflicting inputs made by
        the user or (strict=False) take a best-effort approach to realize the MP settings
        :param log_file: Log file to store the logs
        """
        mp_requests = self._process_user_requests(user_requests)
        mp_requests = self._apply_backend_awareness(mp_requests, config, strict)
        mp_requests = self._propagate_requests(mp_requests, strict)
        self._apply_to_sim(mp_requests)
