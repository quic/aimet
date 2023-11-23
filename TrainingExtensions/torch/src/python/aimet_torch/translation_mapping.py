#@@-COPYRIGHT-START-@@
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

""" Mapping information for AIMET and backend"""
# pylint: disable=import-error, no-name-in-module
from aimet_common.libpymo import QnnDatatype
from aimet_common.defs import QuantizationDataType

backend_datatype_to_aimet_map = {QnnDatatype.QNN_DATATYPE_INT_8: {'bitwidth': 8,
                                                                  'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_INT_16: {'bitwidth': 16,
                                                                   'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_INT_32: {'bitwidth': 32,
                                                                   'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_INT_64: {'bitwidth': 64,
                                                                   'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UINT_8: {'bitwidth': 8,
                                                                   'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UINT_16: {'bitwidth': 16,
                                                                    'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UINT_32: {'bitwidth': 32,
                                                                    'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UINT_64: {'bitwidth': 64,
                                                                    'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_FLOAT_16: {'bitwidth': 16,
                                                                     'dtype': QuantizationDataType.float},
                                 QnnDatatype.QNN_DATATYPE_FLOAT_32: {'bitwidth': 32,
                                                                     'dtype': QuantizationDataType.float},
                                 QnnDatatype.QNN_DATATYPE_SFIXED_POINT_8: {'bitwidth': 8,
                                                                           'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_SFIXED_POINT_16: {'bitwidth': 16,
                                                                            'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_SFIXED_POINT_32: {'bitwidth': 32,
                                                                            'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UFIXED_POINT_8: {'bitwidth': 8,
                                                                           'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UFIXED_POINT_16: {'bitwidth': 16,
                                                                            'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_UFIXED_POINT_32: {'bitwidth': 32,
                                                                            'dtype': QuantizationDataType.int},
                                 QnnDatatype.QNN_DATATYPE_BOOL_8: {'bitwidth': 8,
                                                                   'dtype': QuantizationDataType.int}
                                 }

aimet_op_to_backend_op_name_map = {"Conv1d":"Conv1d",
                                   "Conv2d":"Conv2d",
                                   "Conv3d":"Conv3d",
                                   "ConvTranspose1d":"TransposeConv1d",
                                   "ConvTranspose2d":"TransposeConv2d",
                                   "ConvTranspose3d":"TransposeConv3d",
                                   "ReLU":"Relu",
                                   "Tanh":"Tanh",
                                   "Sigmoid":"Sigmoid",
                                   "ELU":"Elu",
                                   "ReLU6":"Relu6",
                                   "Hardtanh":"ReluMinMax",
                                   "Hardswish":"HardSwish",
                                   "Add":"ElementWiseAdd",
                                   "Subtract":"ElementWiseSubtract",
                                   "Multiply":"ElementWiseMultiply",
                                   "Divide":"ElementWiseDivide",
                                   "Mul":"ElementWiseMultiply",
                                   "Div":"ElementWiseDivide",
                                   "Minimum":"ElementWiseMinimum",
                                   "Maximum":"ElementWiseMaximum",
                                   "Pow":"ElementWisePower",
                                   "Remainder":"ElementWiseMod",
                                   "Fmod":"ElementWiseFmod",
                                   "Exponential":"ElementWiseExp",
                                   "Log":"ElementWiseLog",
                                   "Sqrt":"ElementWiseRsqrt",
                                   "Abs":"ElementWiseAbs",
                                   "Neg":"ElementWiseNeg",
                                   "Erf":"Gelu",
                                   "Round":"ElementWiseRound",
                                   "Where":"ElementWiseSelect",
                                   "Equal":"ElementWiseEqual",
                                   "Greater":"ElementWiseGreater",
                                   "Less":"ElementWiseLess",
                                   "GreaterEqual":"ElementWiseGreaterEqual",
                                   "LessEqual":"ElementWiseLessEqual",
                                   "LogicalOr":"ElementWiseOr",
                                   "LogicalAnd":"ElementWiseAnd",
                                   "LogicalNot":"ElementWiseNot",
                                   "Mean":"ReduceMean",
                                   "Sum":"ReduceSum",
                                   "Prod":"ReduceProd",
                                   "ElementwiseCeil":"ElementWiseCeil",
                                   "ElementwiseFloor":"ElementWiseFloor",
                                   "Split":"Split",
                                   "Concat":"Concat",
                                   "MaxPool2d":"PoolMax2d",
                                   "MaxPool3d":"PoolMax3d",
                                   "AvgPool2d":"PoolAvg2d",
                                   "AvgPool3d":"PoolAvg3d",
                                   "LPPool2d":"L2Pool2d",
                                   "Reshape":"Reshape",
                                   "Permute":"Transpose",
                                   "Upsample":"Resize",
                                   "Linear":"FullyConnected",
                                   "Softmax":"Softmax",
                                   "LogSoftmax":"LogSoftmax",
                                   "LayerNorm":"LayerNorm",
                                   "Softplus":"ElementWiseSoftplus",
                                   "PReLU":"Prelu",
                                   "CustomGather":"Gather",
                                   "InstanceNorm1d":"InstanceNorm",
                                   "InstanceNorm2d":"InstanceNorm",
                                   "InstanceNorm3d":"InstanceNorm",
                                   "MatMul":"MatMul",
                                   "CumSum":"CumulativeSum",
                                   "Argmin":"Argmin",
                                   "Argmax":"Argmax",
                                   "Sin":"ElementWiseSin",
                                   "Cos":"ElementWiseCos",
                                   "Asin":"ElementWiseAsin",
                                   "Atan":"ElementWiseAtan",
                                   "Normalize":"L2Norm",
                                   "Gather":"Gather",
                                   "ChannelShuffle":"ChannelShuffle",
                                   "Pad":"Pad",
                                   "ElementwiseUnarySign":"ElementWiseUnary",
                                   "RoIPool":"RoiPooling",
                                   "PixelShuffle":"DepthToSpace",
                                   "DepthToSpaceDCRMode":"DepthToSpace",
                                   "PixelUnshuffle":"SpaceToDepth",
                                   "Min":"ReduceMin",
                                   "Max":"ReduceMax",
                                   "NonZero":"NonZero",
                                   "TopK":"TopK",
                                   "Shape":"Shape",
                                   "Tile":"Tile",
                                   "LocalResponseNorm":"Lrn",
                                   "LSTM":"Lstm",
                                   "ScatterND":"ScatterNd",
                                   "RoiAlign":"RoiAlign",
                                   "NonMaxSuppression":"NonMaxSuppression",
                                   "GatherNd":"GatherNd",
                                   "BatchNorm1d":"Batchnorm",
                                   "BatchNorm2d":"Batchnorm",
                                   "BatchNorm3d":"Batchnorm",
                                   "OneHot":"OneHot",
                                   "ScatterElements":"ScatterElements",
                                   "LeakyReLU":"Prelu",
                                   "GRU":"Gru",
                                   "CustomLayerNorm":"LayerNorm",
                                   "IndexSelect":"Gather",
                                   "Embedding":"Gather",
                                   "Expand":"ElementWiseMultiply"}


op_to_weight_index_map = {'Conv1d' : 1,
                          'Conv2d': 1,
                          'Conv3d': 1,
                          'DepthWiseConv1d' : 1,
                          'DepthWiseConv2d': 1,
                          'TransposeConv1d' : 1,
                          'TransposeConv2d': 1,
                          'TransposeConv3d': 1,
                          'Batchnorm': 1,
                          'FullyConnected': 1,
                          'LayerNorm': 1,
                          'InstanceNorm': 1,
                          'GroupNorm': 1,
                          'MatMul': 1
                          }
