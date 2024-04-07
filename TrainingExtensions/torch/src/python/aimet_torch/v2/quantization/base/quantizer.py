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
""" Quantizer base class """

import abc
import copy
from collections import OrderedDict
import contextlib
import weakref
from typing import Optional, List, Dict

import torch
from torch import nn

from packaging import version
from aimet_torch.v2.quantization.base import EncodingBase
from aimet_torch.v2.quantization.encoding_analyzer import EncodingAnalyzer


__all__ = ['QuantizerBase']


class QuantizerBase(abc.ABC, torch.nn.Module):
    """
    Quantizer base class
    """
    encoding_analyzer: EncodingAnalyzer

    def __init__(self):
        super().__init__()

        # param_name -> (weakref of initial parameter, version info of the initial parameter)
        # This info will be used for judging whether the current parameter has ever been
        # initialized after it was instantiated.
        self._initial_parameters = OrderedDict()

    @abc.abstractmethod
    @contextlib.contextmanager
    def compute_encodings(self):
        """
        Observe inputs and update quantization parameters based on the input statistics.
        """

    @abc.abstractmethod
    def get_legacy_encodings(self) -> Optional[List[Dict]]:
        """
        Returns a list of encodings, each represented as a List of Dicts
        """

    @abc.abstractmethod
    def set_legacy_encodings(self, encodings: List[Dict]):
        """
        Set encodings represented in the same format as the output of get_legacy_encodings.
        """

    @abc.abstractmethod
    def get_encoding(self) -> Optional[EncodingBase]:
        """
        Return the quantizer's encodings as an EncodingBase object
        """

    def register_quantization_parameter(self, name: str, param: nn.Parameter):
        """
        Register quantization parameter.
        """
        # pylint: disable=protected-access

        self.register_parameter(name, param)
        param = getattr(self, name)
        self._initial_parameters[name] = (weakref.ref(param), param._version)

    def is_initialized(self) -> bool:
        """
        Returns true if the quantization parameters are initialized.
        """
        for param_name, _ in self.named_parameters():
            if not self._is_initialized(param_name):
                return False
        return True

    def _is_initialized(self, param_name) -> bool:
        # pylint: disable=protected-access

        initial_param_weakref, initial_param_version = self._initial_parameters[param_name]
        initial_param = initial_param_weakref()

        if initial_param is None:
            # The initial parameter object doesn't exist in memory space anymore.
            return True

        current_param = getattr(self, param_name)

        if current_param is initial_param and current_param._version == initial_param_version:
            # 1. Current parameter is the identical object as the initial parameter
            # 2. The version nubmer of the current parameter never changed
            return False

        return True

    def state_dict(self, *args, **kwargs): # pylint: disable=arguments-differ
        state_dict = super().state_dict(*args, **kwargs) # pylint: disable=missing-kwoa

        if version.parse(torch.__version__) < version.parse("1.10"):
            # This is for backward compatibility with torch < 1.10
            # which doesn't support get/set_extra_state() hooks
            prefix = kwargs['prefix']
            state_dict[f'{prefix}extra_state'] = self.get_extra_state()

        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True): # pylint:disable=arguments-differ
        if '_extra_state' not in state_dict:
            is_initialized = {
                param_name: True for param_name in state_dict
                if param_name in self._parameters
            }
            state_dict['_extra_state'] = is_initialized

        ret = super().load_state_dict(state_dict, strict)

        if version.parse(torch.__version__) < version.parse("1.10"):
            # This is for backward compatibility with torch < 1.10
            # which doesn't support get/set_extra_state() hooks
            self.set_extra_state(state_dict['_extra_state'])

        return ret

    def get_extra_state(self):
        """
        Get extra state that describes which parameters are initialized.
        """
        return {
            param_name: self._is_initialized(param_name)
            for param_name, _ in self.named_parameters()
        }

    @torch.no_grad()
    def set_extra_state(self, state):
        """
        Set extra state that describes which parameters are initialized.
        """
        is_initialized = state
        for param_name, param in self._parameters.items():
            if param_name in is_initialized:
                self.register_quantization_parameter(param_name, param)

                if is_initialized[param_name]:
                    # If the parameter has been already initialized,
                    # artificially increment the parameter version to mark as initialized
                    param.mul_(1.)

    @torch.no_grad()
    def __deepcopy__(self, memo):
        self_copy = self.__new__(type(self))
        self_copy.__dict__ = copy.deepcopy(self.__dict__, memo)
        self_copy.set_extra_state(self.get_extra_state())
        return self_copy

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_initial_parameters')
        state['is_initialized'] = self.get_extra_state()
        return state

    @torch.no_grad()
    def __setstate__(self, state):
        self._initial_parameters = OrderedDict()
        is_initialized = state.pop('is_initialized')
        self.__dict__.update(state)
        self.set_extra_state(is_initialized)

    def _freeze_encoding(self):
        """
        Freeze the encoding params so they won't be updated during training or compute_encodings
        """
        for param in self.parameters():
            param.requires_grad = False
        self.encoding_analyzer = None

    def _is_encoding_frozen(self):
        """
        Returns true if the encodings are frozen
        """
        for param in self.parameters():
            if param.requires_grad:
                return False
        return self.encoding_analyzer is None
