# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Common type definitions that are used across AIMET """

from enum import Enum
from typing import List, Optional, Union
import warnings

import torch.utils.data

from aimet_common.defs import GreedySelectionParameters, TarRankSelectionParameters, RankSelectScheme


class ModuleCompRatioPair:
    """
    Pair of torch.nn.module and a compression-ratio

    :ivar module: Module of type torch.nn.module
    :ivar comp_ratio: Compression ratio. Compression ratio is the ratio of cost of compressed model
            to cost of the original model.
    """

    def __init__(self, module: torch.nn.Module, comp_ratio: float):
        self.module = module
        self.comp_ratio = comp_ratio


class OpToIOTensors:
    """
    Data class to store the input and output tensor names of an operation as a lists.
    """
    def __init__(self, node_inputs: List[str], node_outputs: List[str]):
        """
        :param node_inputs: name of inputs to the node
        :param node_outputs: name of output from the node
        """

        self.inputs = node_inputs
        self.outputs = node_outputs


class SpatialSvdParameters:
    """ Configuration parameters for spatial svd compression """

    class ManualModeParams:
        """
        Configuration parameters for manual-mode spatial svd compression
        """

        def __init__(self, list_of_module_comp_ratio_pairs: List[ModuleCompRatioPair]):
            """
            :param list_of_module_comp_ratio_pairs: List of (module, comp-ratio) pairs
            """
            self.list_of_module_comp_ratio_pairs = list_of_module_comp_ratio_pairs

    class AutoModeParams:
        """
        Configuration parameters for auto-mode compression
        """

        def __init__(self, greedy_select_params: GreedySelectionParameters,
                     modules_to_ignore: Optional[List[torch.nn.Module]] = None):
            """
            :param greedy_select_params: Params for greedy comp-ratio selection algorithm
            :param modules_to_ignore: List of modules to ignore (None indicates nothing to ignore)
            """
            self.greedy_params = greedy_select_params
            self.modules_to_ignore = [] if modules_to_ignore is None else modules_to_ignore

    class Mode(Enum):
        """ Mode enumeration """

        manual = 1
        """ Manual mode """

        auto = 2
        """ Auto mode """

    def __init__(self, mode: Mode, params: Union[ManualModeParams, AutoModeParams], multiplicity=1):
        """
        :param mode: Either auto mode or manual mode
        :param params: Parameters for the mode selected
        :param multiplicity: The multiplicity to which ranks/input channels will get rounded. Default: 1
        """
        self.mode = mode
        self.mode_params = params
        self.multiplicity = multiplicity


class ChannelPruningParameters:
    """ Configuration parameters for channel pruning compression """

    class ManualModeParams:
        """
        Configuration parameters for manual-mode channel pruning compression
        """

        def __init__(self, list_of_module_comp_ratio_pairs: List[ModuleCompRatioPair]):
            """
            :param list_of_module_comp_ratio_pairs: List of (module, comp-ratio) pairs
            """
            self.list_of_module_comp_ratio_pairs = list_of_module_comp_ratio_pairs

    class AutoModeParams:
        """
        Configuration parameters for auto-mode compression
        """

        def __init__(self, greedy_select_params: GreedySelectionParameters,
                     modules_to_ignore: Optional[List[torch.nn.Module]] = None):
            """
            :param greedy_select_params: Params for greedy comp-ratio selection algorithm
            :param modules_to_ignore: List of modules to ignore (None indicates nothing to ignore)
            """
            self.greedy_params = greedy_select_params
            self.modules_to_ignore = [] if modules_to_ignore is None else modules_to_ignore

    class Mode(Enum):
        """ Mode enumeration """

        manual = 1
        """ Manual mode: User specifies comp-ratio per layer """

        auto = 2
        """ Auto mode: AIMET computes optimal comp-ratio per layer """

    def __init__(self, data_loader: torch.utils.data.DataLoader, num_reconstruction_samples: int,
                 allow_custom_downsample_ops: bool,
                 mode: Mode, params: Union[ManualModeParams, AutoModeParams], multiplicity=1):
        self.data_loader = data_loader
        self.num_reconstruction_samples = num_reconstruction_samples
        self.allow_custom_downsample_ops = allow_custom_downsample_ops
        self.mode = mode
        self.mode_params = params
        self.multiplicity = multiplicity


class WeightSvdParameters:
    """ Configuration parameters for weight svd compression """

    class ManualModeParams:
        """
        Configuration parameters for manual-mode weight svd compression
        """

        def __init__(self, list_of_module_comp_ratio_pairs: List[ModuleCompRatioPair]):
            """
            :param list_of_module_comp_ratio_pairs: List of (module, comp-ratio) pairs
            """
            self.list_of_module_comp_ratio_pairs = list_of_module_comp_ratio_pairs

    class AutoModeParams:
        """
        Configuration parameters for auto-mode compression
        """

        def __init__(self,
                     rank_select_scheme: RankSelectScheme,
                     select_params: Union[GreedySelectionParameters,
                                          TarRankSelectionParameters],
                     modules_to_ignore: Optional[List[torch.nn.Module]] = None):
            """
            :param rank_select_scheme: supports two options greedy and tar
            :param select_params: Params for greedy/TAR comp-ratio selection algorithm
            :param modules_to_ignore: List of modules to ignore (None indicates nothing to ignore)
            """
            self.rank_select_scheme = rank_select_scheme
            self.select_params = select_params
            self.modules_to_ignore = [] if modules_to_ignore is None else modules_to_ignore

    class Mode(Enum):
        """ Mode enumeration """

        manual = 1
        """ Manual mode """

        auto = 2
        """ Auto mode """

    def __init__(self, mode: Mode, params: Union[ManualModeParams, AutoModeParams], multiplicity=1):
        """
        :param mode: Either auto mode or manual mode
        :param params: Parameters for the mode selected
        :param multiplicity: The multiplicity to which ranks/input channels will get rounded. Default: 1
        """
        self.mode = mode
        self.mode_params = params
        self.multiplicity = multiplicity


class PassThroughOp(torch.nn.Module):
    """
    This is a pass-through op, used for purpose of making an op a no-op
    """
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(inputx):
        """
        Forward pass for passthrough op
        """
        return inputx


class ConvInplaceLinear(torch.nn.Module):
    """ Convolution module that replaces a Linear layer inplace"""
    def __init__(self, linear):
        super(ConvInplaceLinear, self).__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.conv2d = torch.nn.Conv2d(linear.in_features, linear.out_features, 1, bias=True if linear.bias is not None else False)
        self.conv2d.weight.data.copy_(linear.weight.data[:, :, None, None])
        if linear.bias is not None:
            self.conv2d.bias.data.copy_(linear.bias.data)
        self.conv2d.to(linear.weight.data.device)

    def __getattr__(self, attr):
        conv2d = self._modules['conv2d']
        if attr == 'conv2d':
            return conv2d
        return getattr(conv2d, attr)

    def forward(self, x: torch.Tensor, scale: float = 1.0):
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0).unsqueeze(-1).permute(0, 2, 3, 1) # (emb_dim, C) -> (1, C, 1, emb_dim)
        elif ndim == 3:
            x = x.unsqueeze(-1).permute(0, 2, 3, 1) # (B, emb_dim, C) -> (B, C, 1, emb_dim)
        elif ndim == 4:
            x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
            warnings.warn("ConvInplaceLinear received an unexpected 4d input, assuming channels-last and proceeding.")
        else:
            raise NotImplementedError(f"ConvInplaceLinear could not handle input with shape {x.shape}")

        x = self.conv2d(x)

        if ndim == 2:
            return x.permute(0, 3, 1, 2).squeeze(-1).squeeze(0) # (1, C, 1, emb_dim) -> # (emb_dim, C)
        elif ndim == 3:
             return x.permute(0, 3, 1, 2).squeeze(-1) # (1, C, 1, emb_dim) -> # (B, emb_dim, C)
        elif ndim == 4:
            x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        return x
