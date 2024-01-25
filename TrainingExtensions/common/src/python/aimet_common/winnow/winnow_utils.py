# -*- mode: python -*-
#  =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Contains Winnowing related utility functions are used by both PyTorch and TensorFlow Winnower. """

from typing import List, Set, Union
from enum import Enum

from aimet_common.connected_graph.connectedgraph_utils import CG_SPLIT
from aimet_common.utils import ModelApi


def get_one_positions_in_binary_mask(mask: List[int]) -> List[int]:
    """
    Return the indices of one positions in a binary mask.

    :param mask: a mask that contains either 0s or 1s
    :return: List of indices of positions of ones in a binary mask.
    """

    mask_one_positions = [idx for (idx, channel) in enumerate(mask) if channel]
    return mask_one_positions


def get_zero_positions_in_binary_mask(mask: List[int]) -> List[int]:
    """
    Return the indices of zero positions in a binary mask.

    :param mask: a mask that contains either 0s or 1s
    :return: List of indices of positions of zeros in a binary mask.
    """

    mask_zero_positions = [idx for (idx, channel) in enumerate(mask) if not channel]
    return mask_zero_positions


class ConnectivityType(Enum):
    """ Defines the channel types"""

    null = 1
    direct = 2
    split = 3
    add = 4
    concat = 5
    skip = 6
    stop = 7


class OpConnectivity:
    """
    Class containing mappings between modules and module connectivity, for both pytorch and tensorflow.
    Class is meant to be used statically, and not instantiated.
    """

    # TODO: remove all types used in old connected graph when it is completely removed.
    pytorch_dict = {'Conv': ConnectivityType.null,
                    'ConvTranspose': ConnectivityType.null,
                    'Linear': ConnectivityType.null,
                    'Gemm': ConnectivityType.null,
                    'DownsampleLayer': ConnectivityType.stop,
                    'Dropout': ConnectivityType.direct,
                    'Dropout2d': ConnectivityType.direct,
                    'Relu': ConnectivityType.direct,
                    'MaxPool': ConnectivityType.direct,
                    'MaxPool2d': ConnectivityType.direct,
                    'AveragePool': ConnectivityType.direct,
                    'AvgPool2d': ConnectivityType.direct,
                    'Neg': ConnectivityType.direct,
                    'BatchNorm2d': ConnectivityType.direct,
                    'Flatten': ConnectivityType.direct,
                    'flatten': ConnectivityType.direct,
                    'ReduceMean': ConnectivityType.direct,
                    'GlobalAveragePool': ConnectivityType.direct,
                    'AdaptiveAvgPool2d': ConnectivityType.direct,
                    'BatchNormalization': ConnectivityType.direct,
                    'BatchNorm1d': ConnectivityType.direct,
                    'Add': ConnectivityType.add,
                    'Concat': ConnectivityType.concat,
                    'Split': ConnectivityType.split,
                    CG_SPLIT: ConnectivityType.split,
                    'LogSoftmax': ConnectivityType.direct,
                    'Gather': ConnectivityType.skip,
                    'Reshape': ConnectivityType.skip,
                    'ListConstruct': ConnectivityType.skip,
                    'Pad': ConnectivityType.skip,
                    'Mul': ConnectivityType.skip,
                    'Clip': ConnectivityType.direct,
                    'Upsample': ConnectivityType.skip,
                    'index_select': ConnectivityType.null,
                    'mean': ConnectivityType.direct,
                    'floor': ConnectivityType.direct,
                    'cat': ConnectivityType.concat,
                    'add': ConnectivityType.add,
                    'size': ConnectivityType.skip,
                    'NumToTensor': ConnectivityType.skip,
                    'mul': ConnectivityType.skip,
                    'view': ConnectivityType.skip,
                    'reshape': ConnectivityType.skip,
                    'slice': ConnectivityType.skip,
                    'unsqueeze': ConnectivityType.skip,
                    'select': ConnectivityType.skip}

    # Including Reshape under null for tensorflow so that input to the layer below does not get propagated
    # to the output of the layer above.
    # Putting Placeholder under null so an output mask is generated for it, even though it will never be changed.
    # Output mask needed in a check in module_reducer
    tensorflow_dict = {'Conv2D': ConnectivityType.null,
                       'DepthwiseConv2dNative': ConnectivityType.null,
                       'Dense': ConnectivityType.null,
                       'Placeholder': ConnectivityType.null,
                       'PlaceholderWithDefault': ConnectivityType.null,
                       'Downsample': ConnectivityType.null,
                       'BatchNorm': ConnectivityType.direct,
                       'AvgPool': ConnectivityType.direct,
                       'FusedBatchNormV3': ConnectivityType.direct,
                       'Relu': ConnectivityType.direct,
                       'Relu6': ConnectivityType.direct,
                       'MaxPool': ConnectivityType.direct,
                       'Tanh': ConnectivityType.direct,
                       'Identity': ConnectivityType.direct,
                       'Dropout': ConnectivityType.direct,
                       'Pad': ConnectivityType.direct,
                       'PadV2': ConnectivityType.direct,
                       'MirrorPad': ConnectivityType.direct,
                       'Minimum': ConnectivityType.direct,
                       'Maximum': ConnectivityType.direct,
                       'Upsample2D': ConnectivityType.direct,
                       'LeakyRelu': ConnectivityType.direct,
                       'Add': ConnectivityType.add,
                       'AddN': ConnectivityType.add,
                       'AddV2': ConnectivityType.add,
                       'ConcatV2': ConnectivityType.concat,
                       'branch': ConnectivityType.split,
                       'Softmax': ConnectivityType.stop,
                       'Squeeze': ConnectivityType.stop,
                       'ArgMax': ConnectivityType.stop,
                       'Equal': ConnectivityType.stop,
                       'Cast': ConnectivityType.stop,
                       'Mean': ConnectivityType.stop,
                       'Reshape': ConnectivityType.stop,
                       'Shape': ConnectivityType.stop,
                       'Upsample': ConnectivityType.stop,
                       'Flatten': ConnectivityType.stop,
                       'GlobalMaxpool2D': ConnectivityType.stop}

    @classmethod
    def get_op_connectivity(cls, model_api: ModelApi, op_type: str) -> Union[ConnectivityType, None]:
        """
        Get op connectivity for a module, and return None if the module is not recognized.
        :param model_api: Represents either pytorch or tensorflow
        :param op_type: Type of the op, which is used to map to its connectivity
        :return: Op connectivity, or None if module is not recognized.
        """
        if model_api == ModelApi.pytorch:
            return cls.pytorch_dict.get(op_type, None)
        return cls.tensorflow_dict.get(op_type, ConnectivityType.stop)


def get_conv_ops_for_api(model_api: ModelApi) -> Set:
    """
    Return a set of op types that represent conv ops, based on the model api

    :param model_api: Enum for whether the api is pytorch or tensorflow
    """
    if model_api == ModelApi.pytorch:
        return {'Conv', 'ConvTranspose'}
    return {'Conv2D', 'DepthwiseConv2dNative'}


def get_linear_ops_for_api(model_api: ModelApi) -> Set:
    """
    Return a set of op types that represent linear ops, based on the model api

    :param model_api: Enum for whether the api is pytorch or tensorflow
    """
    if model_api == ModelApi.pytorch:
        return {'Gemm'}
    return {'Dense'}


def get_indices_among_ones_of_overlapping_ones(more_ones_mask: List[int], less_ones_mask: List[int]) -> List[int]:
    """
    :param more_ones_mask: Mask that has more ones
    :param less_ones_mask: Mask that has less ones
    :return: A list of indices of where the overlapping ones occur between the two masks, where the indices are counted
    looking only at the ones in more_ones_mask.
    It is assumed that wherever less_ones_mask has a 1, more_ones_mask also has a 1 in that position
    Example:
    more_ones_mask: 1, 0, 0, 1, 1, 0, 1, 0, 1, 1
    less_ones_mask: 1, 0, 0, 0, 1, 0, 0, 0, 1, 0
                    *           *           *
    Overlapping ones are represented by the stars above.  If we are indexing using the full list, we would say that
    the overlapping ones are indexes 0, 4, and 8.  However, we only consider the indexes by looking at the positions
    in more_ones_mask that have ones.  So we can see that it is the 0th, 2nd, and 4th ones in more_ones_mask that have
    overlapping ones with less_ones_mask.  Thus the index list that is returned will be [0, 2, 4].
    """

    indices = []
    more_ones_mask_ones_index = 0
    for index, mask_item in enumerate(more_ones_mask):
        if mask_item & less_ones_mask[index]:
            indices.append(more_ones_mask_ones_index)
        if mask_item:
            more_ones_mask_ones_index += 1

    return indices


def update_winnowed_channels(original_mask: List[int], new_mask: List[int]):
    """
    Update original mask with newly winnowed channels in new_mask
    Mask lengths can be different, but the length of new mask should be equal to the number of ones in original mask.
    Ones in the original mask will be set to zeros for each zero in new_mask.
    For determining which index positions to set to zero in original mask, zeros that were already present in original
    mask are not considered.
    Example:
    Original mask: 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1
    New mask:      1, 1, 0, 0, 1, 0, 1
    New mask has 7 entries, same as number of ones in original mask.  Each channel in new mask corresponds to a one
    in the original mask.
    In new mask, 2nd, 3rd, and 5th channels are winnowed.  Thus, we set the 2nd, 3rd, and 5th ones in original mask
    to zeros.  This results in an original mask of 1, 1, 0, 0*, 0, 0, 0*, 1, 0*, 0, 0, 1 (* represents ones set to zero)
    :param original_mask: Original mask representing a running tally of masks winnowed since the very beginning
    :param new_mask: Most recent mask after the most recent round of winnowing
    :return: original masks with updated channels winnowed according to new mask
    """
    assert len(new_mask) == sum(original_mask)
    original_mask_ones_indices = get_one_positions_in_binary_mask(original_mask)
    new_mask_zero_indices = get_zero_positions_in_binary_mask(new_mask)
    for idx in new_mask_zero_indices:
        original_mask[original_mask_ones_indices[idx]] = 0
