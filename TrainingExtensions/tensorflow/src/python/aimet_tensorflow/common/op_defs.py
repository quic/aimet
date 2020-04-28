# /usr/bin/env python2.7
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""
Map the tensorflow ops to known SNPE layer types. These determine where quantization ops
should be inserted. Add new defintions as needed to support adding them in new
locations

The format follows the patern: layer_name:[list of op types]
Note the layer name is only a rough name it doesn't need to match any SNPE layer name
exactly.
"""

from aimet_common.utils import AimetLogger

_log = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

default_op_map = {
    'Add': {'ScaledBatchNorm': ['Add', 'Rsqrt', 'Mul', 'Mul', 'Mul', 'Sub', 'Add'],
            'UnScaledBatchNorm': ['Add', 'Rsqrt', 'Mul', 'Mul', 'Sub', 'Add']},
    'Concat': {'Concat': ['Concat']},
    'Conv2D': {'Conv w/bias1': ['Conv2D', 'BiasAdd'],
               'Conv w/bias2': ['Conv2D', 'Add'],
               'Conv w/o bias': ['Conv2D']},
    'Conv2DTranspose': {'Deconv w/bias1': ['Conv2DTranspose', 'BiasAdd'],
                        'Deconv w/bias2': ['Conv2DTranspose', 'Add'],
                        'Deconv w/o bias': ['Conv2DTranspose']},
    'DepthwiseConv2dNative': {'Depthwise Convolution w/bias1': ['DepthwiseConv2dNative', 'BiasAdd'],
                              'Depthwise Convolution w/bias2': ['DepthwiseConv2dNative', 'Add'],
                              'Depthwise Convolution w/o bias': ['DepthwiseConv2dNative']},
    'ElementWiseSum': {'ElementWiseSum': ['ElementWiseSum']},
    'ElementWiseMul': {'ElementWiseMul': ['ElementWiseMul']},
    'ElementWiseMax': {'ElementWiseMax': ['ElementWiseMax']},
    'Maximum': {'Maximum': ['Maximum']},
    'MatMul': {'FC w/bias1': ['MatMul', 'BiasAdd'],
               'FC w/bias2': ['MatMul', 'Add'],
               'FC w/o bias': ['MatMul']},
    'LRN': {'LRN': ['LRN']},
    'AvgPool': {'Average Pooling': ['AvgPool']},
    'MaxPool': {'MaxPool': ['MaxPool']},
    'Placeholder': {'Data input w/reshape': ['Placeholder', 'Reshape'],
                    'Data input no reshape': ['Placeholder']},
    'Relu': {'Prelu': ['Relu', 'Abs', 'Sub', 'Mul', 'Mul', 'Add'],
             'Relu': ['Relu']},
    'Relu6': {'Relu6': ['Relu6']},
    'Sigmoid': {'Sigmoid': ['Sigmoid']},
    'Split': {'Split': ['Split']},
    'SplitV': {'SplitV': ['SplitV']},
    'SoftMax': {'Softmax': ['Softmax']},
    'Squeeze': {'Squeeze': ['Squeeze']},
    'Tanh': {'Tanh': ['Tanh']},
}


def _check_sequence(graph_seq, known_seq):
    """
    Check that the sequence matches up to a certain point for the known sequence. If it does
    return the index to where it matches. If it doesn't return 0.
    :param graph_seq: Actual sequence in the graph
    :param known_seq: One of the known sequences
    :return: Index up to which a match is found
    """
    if len(known_seq) < len(graph_seq):
        return 0

    if known_seq[0:len(graph_seq)] == graph_seq:
        match_len = len(graph_seq)
    else:
        match_len = 0

    return match_len


def check_match(op_list, op_map=None):
    """
    Check if the op_list matches, or partially matches, any supported op sequences.
    :param op_list: Current list of ops to find a match for
    :param op_map: Map of known sequences
    :return: Current match length, max match length possible for this sequence
    """
    if not op_list:
        raise ValueError('Empty op_list passed to check_match')

    if not op_map:
        op_map = default_op_map

    op_type_list = [op.type for op in op_list]
    _log.debug('Checking matches for op_type_list: %s', op_type_list)

    op_index = op_type_list[0]
    if op_index in op_map:
        ret_max_len = 0
        ret_match_len = 0
        op_map_entry = op_map[op_index]
        for _, sequence in op_map_entry.items():
            match_len = _check_sequence(op_type_list, sequence)
            max_len = len(sequence)
            if match_len > ret_match_len:
                ret_match_len = match_len
                ret_max_len = max_len
            elif match_len == ret_match_len:
                ret_max_len = max(max_len, ret_max_len)
        return ret_match_len, ret_max_len
    return 0, 0


def does_sequence_match(op_list, op_map=None):
    """
    Check if the op_list fully matches any supported op sequences
    :param op_list: List of ops to match
    :param op_map: Map of known sequences
    :return: True if full match found, False otherwise
    """
    if not op_list:
        raise ValueError('Empty op_list passed to does_sequence_match')

    if not op_map:
        op_map = default_op_map

    op_type_list = [op.type for op in op_list]
    _log.debug('Checking matches for op_type_list: %s', op_type_list)

    op_index = op_type_list[0]
    if op_index in op_map:
        op_map_entry = op_map[op_index]
        for _, sequence in op_map_entry.items():
            if sequence == op_type_list:
                return True
    return False
