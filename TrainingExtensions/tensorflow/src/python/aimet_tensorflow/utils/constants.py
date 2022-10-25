# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""  Module for holding all the constants """

import tensorflow as tf

# Op types with weights
OP_WEIGHT_TYPES = ['Conv2D', 'MatMul', 'DepthwiseConv2dNative', 'Conv2DTranspose']
OP_WEIGHT_INDICES = {'Conv2D': 1,
                     'MatMul': 1,
                     'DepthwiseConv2dNative': 1,
                     'Conv2DTranspose': 1,
                     'BiasAdd': 1,
                     'Add': 1,
                     'Conv2DBackpropInput': 1
                     }
OP_VAR_WEIGHT_INDEX = 0
OP_BIAS_TYPES = ['Conv2D', 'VariableV2']
OP_BIAS_INDICES = {'Conv2D': 0,
                   'VariableV2': 0,
                   'DepthwiseConv2dNative' : 0,
                   'MatMul': 0,
                   'Conv2DBackpropInput': 0
                   }
READ_VAR_OP_BIAS_INDICES = {'Conv2D': 1,
                            'DepthwiseConv2dNative' : 1,
                            'MatMul': 1
                           }

OP_WEIGHT_SHAPE_INDEX_FOR_BIAS = {'MatMul': 1,
                                  'DepthwiseConv2dNative': 2,
                                  'Conv2D': 3
                                 }

BN_OP_PARAM_INDICES = {'gamma': 1,
                       'beta': 2,
                       'movingmean': 3,
                       'movingvariance': 4
                      }

TF_VAR_WT_CONSUMERS_READ_OP_INDEX = 1
TF_VAR_BIAS_CONSUMERS_READ_OP_INDEX = 1
BIAS_ADD_CONSUMERS_INPUT_BIAS_READ_INDEX = 1
BIAS_ADD_READ_VAR_OP_BIAS_TENSOR_INDEX = 1

QUANT_ALLOWED_DTYPES = [tf.float32, tf.float64]

# add quantize op input indices with corresponding param name


class QuantizeOpIndices():
    """ QuantizeOp input indices enumeration """
    op_mode = 1
    tensor_quant_ref = 2
    encoding_min = 3
    encoding_max = 4
    bit_width = 5
    use_symmetric_encoding = 6
    time_steps = 7
    is_int_data_type = 7
    axis_handling = 8

class BNOpParamType():
    """ BN OP Param types """
    gamma = 1
    beta = 2
    moving_mean = 3
    moving_variance = 4


BN_OP_PARAM_NAME_SUFFIX = {BNOpParamType.gamma: '/gamma/Read/ReadVariableOp:0',
                           BNOpParamType.beta: '/beta/Read/ReadVariableOp:0',
                           BNOpParamType.moving_mean: '/moving_mean/Read/ReadVariableOp:0',
                           BNOpParamType.moving_variance: '/moving_variance/Read/ReadVariableOp:0'
                           }
