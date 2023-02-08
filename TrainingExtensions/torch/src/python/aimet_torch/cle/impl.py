# /usr/bin/env python3.8
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

""" Cross Layer Equalization base implementation """

import abc
from typing import Union, Tuple, Dict
import torch
import numpy as np

from aimet_common.cross_layer_equalization import ClsSetInfo


ClsSet = Union[Tuple[torch.nn.Conv2d, torch.nn.Conv2d],
               Tuple[torch.nn.Conv2d, torch.nn.Conv2d, torch.nn.Conv2d]]


class CLSImpl(abc.ABC):
    """
    The Implementation interface declares methods common to both MO (c++) and python versions of CLS algorithm.
    """
    @abc.abstractmethod
    def scale_cls_set_with_depthwise_layers(self, cls_set: ClsSet) -> [np.ndarray, np.ndarray]:
        """
        API to invoke equalize layer params for depth wise separable layers(update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers whose weights and biases need to be equalized.
                        Second Conv layer is a depth-wise conv and third conv layer is point-wise conv
        :return: Scaling factors S_12 and S_23 : numpy arrays
        """

    @abc.abstractmethod
    def scale_cls_set_with_conv_layers(self, cls_set: ClsSet) -> np.ndarray:
        """
        API to invoke equalize layer params for regular conv layers (update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized
        :return: Scaling factor S_12 for each conv layer pair: numpy array
        """

    @staticmethod
    def _transpose_tensor_in_common_format(module: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        """
        Transpose tensor in the common format [Noc, Nin, Kh, Kw].
        For transposed conv, weight are stored in [Nin, Noc, Kh, Kw] format.

        :param module: Module.
        :param tensor: Input tensor.
        :return: Output tensor.
        """
        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        if isinstance(module, torch.nn.ConvTranspose2d):
            tensor = tensor.permute(1, 0, 2, 3)
        if isinstance(module, torch.nn.ConvTranspose1d):
            tensor = tensor.permute(1, 0, 2)
        return tensor

    @staticmethod
    def _make_4d_tensor(module: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        """
        Return 4 dimensional tensor by adding a dimension on the end if the tensor is not 4d.

        :param module: Module.
        :param tensor: Input tensor.
        :return: Output tensor.
        """
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            assert len(tensor.shape) == 3, "Module should have 3d weight tensor."
            tensor = torch.unsqueeze(tensor, dim=-1)
        return tensor

    @staticmethod
    def _get_tensor_max(tensor: torch.Tensor, dim: Union[int, Tuple]) -> torch.Tensor:
        """
        Return the max value of the tensor in given dimension.

        :param tensor: Input Tensor.
        :param dim: dimension(s)
        :return: Output tensor.
        """
        return torch.amax(torch.abs(tensor), dim=dim)


class HBFImpl(abc.ABC):
    """
    The Implementation interface declares methods common to both MO (c++) and python versions of HBF algorithm.
    """
    @abc.abstractmethod
    def bias_fold(self, cls_pair_info: ClsSetInfo.ClsSetLayerPairInfo,
                  bn_layers: Dict[Union[torch.nn.Module], torch.nn.BatchNorm2d]):
        """
        Bias fold implementation.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param bn_layers: Dictionary with Key being Conv/Linear layer and value being corresponding folded BN layer.
        """
