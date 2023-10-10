# -*- mode: python -*-
#  =============================================================================
#
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
#
#  =============================================================================

""" Module reducer abstract class. """

import abc
from typing import Dict, List

from aimet_common.connected_graph.operation import Op
from aimet_common.winnow.mask import Mask


class ModuleReducer(abc.ABC):
    """ The ModuleReducer class contains functionality to reduce a module's weight parameter and adjust the module's
    number of input and output channels.
    """

    def __init__(self, using_cuda: bool, reshape: bool, op_to_mask_dict: Dict[Op, Mask]):
        """
        ModuleReducer initialization.

        :param using_cuda: Indicates if a module is on GPU.
        :param reshape: If True, ModuleReducer will add DownsampleLayer and UpsampleLayer as needed.
                        If False, ModuleReducer will not add DownsampleLayer and UpsampleLayer.
        :param op_to_mask_dict: Dictionary mapping Op to mask
        """

        self._using_cuda = using_cuda
        self._reshape = reshape
        self._op_to_mask_dict = op_to_mask_dict

    @abc.abstractmethod
    def reduce_modules(self, list_of_ops_to_reduce: List):
        """
        For the Ops in the list, reduce the corresponding module.

        :param list_of_ops_to_reduce: list of Ops whose associated modules need to be reduced.
        """
