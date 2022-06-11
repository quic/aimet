# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Common type definitions that are used across aimet """

from enum import Enum
from typing import List, Optional, Union

import tensorflow as tf

from aimet_common.defs import GreedySelectionParameters


class ModuleCompRatioPair:
    """
    Pair of tf.Operation and a compression-ratio

    :ivar module: Module of type tf.Operation
    :ivar comp_ratio: Compression ratio. Compression ratio is the ratio of cost of compressed model
            to cost of the original model.
    """

    def __init__(self, module: tf.Operation, comp_ratio: float):
        self.module = module
        self.comp_ratio = comp_ratio


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
                     modules_to_ignore: Optional[List[tf.Operation]] = None):
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

    def __init__(self, input_op_names: List[str], output_op_names: List[str], mode: Mode,
                 params: Union[ManualModeParams, AutoModeParams], multiplicity=1):
        """
        :param input_op_names: list of input op names to the model
        :param output_op_names: List of output op names of the model
        :param mode: Either auto mode or manual mode
        :param params: Parameters for the mode selected
        :param multiplicity: The multiplicity to which ranks/input channels will get rounded. Default: 1
        """
        self.input_op_names = input_op_names
        self.output_op_names = output_op_names
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
                     modules_to_ignore: Optional[List[tf.Operation]] = None):
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
        """ Auto mode: aimet computes optimal comp-ratio per layer """

    def __init__(self, input_op_names: List[str], output_op_names: List[str], data_set: tf.data.Dataset,
                 batch_size: int, num_reconstruction_samples: int, allow_custom_downsample_ops: bool, mode: Mode,
                 params: Union[ManualModeParams, AutoModeParams], multiplicity=1):
        """

        :param input_op_names: list of input op names to the model
        :param output_op_names: List of output op names of the model
        :param data_set: data set
        :param batch_size: batch size
        :param num_reconstruction_samples: number of samples to be used for reconstruction
        :param allow_custom_downsample_ops: If set to True, DownSampleLayer and UpSampleLayer will be added as required
        :param mode: indicates whether the mode is manual or auto
        :param params: ManualModeParams or AutoModeParams, depending on teh value of mode
        :param multiplicity: The multiplicity to which ranks/input channels will get rounded. Default: 1
        """

        # pylint: disable=too-many-arguments
        self.input_op_names = input_op_names
        self.output_op_names = output_op_names
        self.data_set = data_set
        self.batch_size = batch_size
        self.num_reconstruction_samples = num_reconstruction_samples
        self.allow_custom_downsample_ops = allow_custom_downsample_ops
        self.mode = mode
        self.mode_params = params
        self.multiplicity = multiplicity


class ParameterInfo:
    """ Store information required for parameter quantization """
    def __init__(self, param_type: str, op_with_param_name: List):
        self.param_type = param_type
        self.op_with_param_name = op_with_param_name


# Ways to handle getting number of channels from axes. Default is to get it from the last dimension. For depthwise
# conv2d, it will be obtained from the last two dimensions.
class AxisHandling(Enum):
    """
    Enum for axis handling used as input variable to QcQuantizePerChannelOp. This defines how to interpret the
    number of output channels from the weight dimensions.
    """
    LAST_AXIS = 0
    LAST_TWO_AXES = 1
