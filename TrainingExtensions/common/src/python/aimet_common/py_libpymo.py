# /usr/bin/env python3
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

""" Creating an alias for function/classes/methods to use AIMET without MO library  """

import typing

libpymo_attributes = [
    'ActivationType',
    'BNParams',
    'BNParamsHighBiasFold',
    'BatchNormFold',
    'BiasCorrection',
    'BnBasedBiasCorrection',
    'BnParamsBiasCorr',
    'COMPRESS_LAYER_TYPE',
    'COMP_MODE_CPU',
    'COMP_MODE_GPU',
    'COST_TYPE_MAC',
    'COST_TYPE_MEMORY',
    'ComputationMode',
    'CrossLayerScaling',
    'EncodingAnalyzerForPython',
    'EqualizationParams',
    'GetQuantizationEncodingAnalyzerInstance',
    'GetQuantizationInstance',
    'GetSVDInstance',
    'HighBiasFold',
    'LAYER_INPUT',
    'LAYER_OUTPUT',
    'LAYER_TYPE_CONV',
    'LAYER_TYPE_FC',
    'LAYER_TYPE_OTHER',
    'LayerAttributes',
    'LayerInOut',
    'LayerParams',
    'ModelOpDefParser',
    'NETWORK_COST_METRIC',
    'PtrToInt64',
    'QUANTIZATION_ENTROPY',
    'QUANTIZATION_MSE',
    'QUANTIZATION_PERCENTILE',
    'QUANTIZATION_RANGE_LEARNING',
    'QUANTIZATION_TF',
    'QUANTIZATION_TF_ENHANCED',
    'QnnDatatype',
    'QnnRank',
    'QuantizationEncodingAnalyzer',
    'QuantizationMode',
    'Quantizer',
    'ROUND_NEAREST',
    'ROUND_STOCHASTIC',
    'RescalingParamsVectors',
    'RoundingMode',
    'SVD_COMPRESS_TYPE',
    'Svd',
    'TYPE_NONE',
    'TYPE_SINGLE',
    'TYPE_SUCCESSIVE',
    'TensorParamBiasCorrection',
    'TensorParams',
    'TensorQuantizationSimForPython',
    'TensorQuantizer',
    'TensorQuantizerOpMode',
    'TfEncoding',
    'fold',
    'scaleDepthWiseSeparableLayer',
    'scaleLayerParams',
    'str_to_dtype',
    'str_to_rank',
    'updateBias'
]

for libpymo_attribute in libpymo_attributes:
    globals()[libpymo_attribute] = typing.Any


class COMPRESS_LAYER_TYPE:
    """
    COMPRESS_LAYER_TYPE
    """
    LAYER_TYPE_CONV = typing.Any
    LAYER_TYPE_FC = typing.Any
    LAYER_TYPE_OTHER = typing.Any


class ComputationMode:
    """
    ComputationMode
    """
    COMP_MODE_CPU = typing.Any
    COMP_MODE_GPU = typing.Any


class LayerInOut:
    """
    LayerInOut
    """
    LAYER_INPUT = typing.Any
    LAYER_OUTPUT = typing.Any


class NETWORK_COST_METRIC:
    """
    NETWORK_COST_METRIC
    """
    COST_TYPE_MAC = typing.Any
    COST_TYPE_MEMORY = typing.Any


class QuantizationMode:
    """
    QuantizationMode
    """
    QUANTIZATION_ENTROPY = typing.Any
    QUANTIZATION_MSE = typing.Any
    QUANTIZATION_PERCENTILE = typing.Any
    QUANTIZATION_RANGE_LEARNING = typing.Any
    QUANTIZATION_TF = typing.Any
    QUANTIZATION_TF_ENHANCED = typing.Any


class RoundingMode:
    """
    RoundingMode
    """
    ROUND_NEAREST = typing.Any
    ROUND_STOCHASTIC = typing.Any


class SVD_COMPRESS_TYPE:
    """
    SVD_COMPRESS_TYPE
    """
    TYPE_NONE = typing.Any
    TYPE_SINGLE = typing.Any
    TYPE_SUCCESSIVE = typing.Any


class QnnDatatype:
    """
    QnnDatatype
    """
    QNN_DATATYPE_BACKEND_SPECIFIC = typing.Any
    QNN_DATATYPE_BOOL_8 = typing.Any
    QNN_DATATYPE_FLOAT_16 = typing.Any
    QNN_DATATYPE_FLOAT_32 = typing.Any
    QNN_DATATYPE_INT_16 = typing.Any
    QNN_DATATYPE_INT_32 = typing.Any
    QNN_DATATYPE_INT_64 = typing.Any
    QNN_DATATYPE_INT_8 = typing.Any
    QNN_DATATYPE_SFIXED_POINT_16 = typing.Any
    QNN_DATATYPE_SFIXED_POINT_32 = typing.Any
    QNN_DATATYPE_SFIXED_POINT_8 = typing.Any
    QNN_DATATYPE_UFIXED_POINT_16 = typing.Any
    QNN_DATATYPE_UFIXED_POINT_32 = typing.Any
    QNN_DATATYPE_UFIXED_POINT_8 = typing.Any
    QNN_DATATYPE_UINT_16 = typing.Any
    QNN_DATATYPE_UINT_32 = typing.Any
    QNN_DATATYPE_UINT_64 = typing.Any
    QNN_DATATYPE_UINT_8 = typing.Any
    QNN_DATATYPE_UNDEFINED = typing.Any
