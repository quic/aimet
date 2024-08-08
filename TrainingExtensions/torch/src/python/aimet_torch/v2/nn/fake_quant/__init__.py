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
"""
Placeholder of the deprecated aimet_torch/v2/nn/fake_quant.py for backward compatibility.

FakeQuantized- modules are now completely superseded by Quantized- modules,
and any legacy user code that tries to import FakeQuantized- modules will import Quantized- modules instead.
This package serves as a namespace that maps the legacy FakeQuantized- modules to the Quantized- equivalents
for backward compatibility.
"""

import torch
from packaging import version
from .. import true_quant as _nn

FakeQuantizationMixin = _nn.QuantizationMixin
FakeQuantizedAdaptiveMaxPool1d = _nn.QuantizedAdaptiveMaxPool1d
FakeQuantizedAdaptiveMaxPool2d = _nn.QuantizedAdaptiveMaxPool2d
FakeQuantizedAdaptiveMaxPool3d = _nn.QuantizedAdaptiveMaxPool3d
FakeQuantizedAlphaDropout = _nn.QuantizedAlphaDropout
FakeQuantizedAvgPool1d = _nn.QuantizedAvgPool1d
FakeQuantizedAvgPool2d = _nn.QuantizedAvgPool2d
FakeQuantizedAvgPool3d = _nn.QuantizedAvgPool3d
FakeQuantizedBCELoss = _nn.QuantizedBCELoss
FakeQuantizedBCEWithLogitsLoss = _nn.QuantizedBCEWithLogitsLoss
FakeQuantizedBatchNorm1d = _nn.QuantizedBatchNorm1d
FakeQuantizedBatchNorm2d = _nn.QuantizedBatchNorm2d
FakeQuantizedBatchNorm3d = _nn.QuantizedBatchNorm3d
FakeQuantizedBilinear = _nn.QuantizedBilinear
FakeQuantizedCELU = _nn.QuantizedCELU
FakeQuantizedCTCLoss = _nn.QuantizedCTCLoss
FakeQuantizedChannelShuffle = _nn.QuantizedChannelShuffle

if version.parse(torch.__version__) >= version.parse("2.1.0"):
    FakeQuantizedCircularPad1d = _nn.QuantizedCircularPad1d
    FakeQuantizedCircularPad2d = _nn.QuantizedCircularPad2d
    FakeQuantizedCircularPad3d = _nn.QuantizedCircularPad3d

FakeQuantizedConstantPad1d = _nn.QuantizedConstantPad1d
FakeQuantizedConstantPad2d = _nn.QuantizedConstantPad2d
FakeQuantizedConstantPad3d = _nn.QuantizedConstantPad3d
FakeQuantizedConv1d = _nn.QuantizedConv1d
FakeQuantizedConv2d = _nn.QuantizedConv2d
FakeQuantizedConv3d = _nn.QuantizedConv3d
FakeQuantizedConvTranspose1d = _nn.QuantizedConvTranspose1d
FakeQuantizedConvTranspose2d = _nn.QuantizedConvTranspose2d
FakeQuantizedConvTranspose3d = _nn.QuantizedConvTranspose3d
FakeQuantizedCosineEmbeddingLoss = _nn.QuantizedCosineEmbeddingLoss
FakeQuantizedCosineSimilarity = _nn.QuantizedCosineSimilarity
FakeQuantizedCrossEntropyLoss = _nn.QuantizedCrossEntropyLoss
FakeQuantizedDropout = _nn.QuantizedDropout

if version.parse(torch.__version__) >= version.parse("1.12.0"):
    FakeQuantizedDropout1d = _nn.QuantizedDropout1d

FakeQuantizedDropout2d = _nn.QuantizedDropout2d
FakeQuantizedDropout3d = _nn.QuantizedDropout3d
FakeQuantizedELU = _nn.QuantizedELU
FakeQuantizedEmbedding = _nn.QuantizedEmbedding
FakeQuantizedEmbeddingBag = _nn.QuantizedEmbeddingBag
FakeQuantizedFeatureAlphaDropout = _nn.QuantizedFeatureAlphaDropout
FakeQuantizedFlatten = _nn.QuantizedFlatten
FakeQuantizedFold = _nn.QuantizedFold
FakeQuantizedFractionalMaxPool2d = _nn.QuantizedFractionalMaxPool2d
FakeQuantizedFractionalMaxPool3d = _nn.QuantizedFractionalMaxPool3d
FakeQuantizedGELU = _nn.QuantizedGELU
FakeQuantizedGLU = _nn.QuantizedGLU
FakeQuantizedGRU = _nn.QuantizedGRU
FakeQuantizedGRUCell = _nn.QuantizedGRUCell
FakeQuantizedGaussianNLLLoss = _nn.QuantizedGaussianNLLLoss
FakeQuantizedGroupNorm = _nn.QuantizedGroupNorm
FakeQuantizedHardshrink = _nn.QuantizedHardshrink
FakeQuantizedHardsigmoid = _nn.QuantizedHardsigmoid
FakeQuantizedHardswish = _nn.QuantizedHardswish
FakeQuantizedHardtanh = _nn.QuantizedHardtanh
FakeQuantizedHingeEmbeddingLoss = _nn.QuantizedHingeEmbeddingLoss
FakeQuantizedHuberLoss = _nn.QuantizedHuberLoss
FakeQuantizedInstanceNorm1d = _nn.QuantizedInstanceNorm1d
FakeQuantizedInstanceNorm2d = _nn.QuantizedInstanceNorm2d
FakeQuantizedInstanceNorm3d = _nn.QuantizedInstanceNorm3d
FakeQuantizedKLDivLoss = _nn.QuantizedKLDivLoss
FakeQuantizedL1Loss = _nn.QuantizedL1Loss
FakeQuantizedLPPool1d = _nn.QuantizedLPPool1d
FakeQuantizedLPPool2d = _nn.QuantizedLPPool2d
FakeQuantizedLSTM = _nn.QuantizedLSTM
FakeQuantizedLSTMCell = _nn.QuantizedLSTMCell
FakeQuantizedLayerNorm = _nn.QuantizedLayerNorm
FakeQuantizedLeakyReLU = _nn.QuantizedLeakyReLU
FakeQuantizedLinear = _nn.QuantizedLinear
FakeQuantizedLocalResponseNorm = _nn.QuantizedLocalResponseNorm
FakeQuantizedLogSigmoid = _nn.QuantizedLogSigmoid
FakeQuantizedLogSoftmax = _nn.QuantizedLogSoftmax
FakeQuantizedMSELoss = _nn.QuantizedMSELoss
FakeQuantizedMarginRankingLoss = _nn.QuantizedMarginRankingLoss
FakeQuantizedMaxPool1d = _nn.QuantizedMaxPool1d
FakeQuantizedMaxPool2d = _nn.QuantizedMaxPool2d
FakeQuantizedMaxPool3d = _nn.QuantizedMaxPool3d
FakeQuantizedMaxUnpool1d = _nn.QuantizedMaxUnpool1d
FakeQuantizedMaxUnpool2d = _nn.QuantizedMaxUnpool2d
FakeQuantizedMaxUnpool3d = _nn.QuantizedMaxUnpool3d
FakeQuantizedMish = _nn.QuantizedMish
FakeQuantizedMultiLabelMarginLoss = _nn.QuantizedMultiLabelMarginLoss
FakeQuantizedMultiLabelSoftMarginLoss = _nn.QuantizedMultiLabelSoftMarginLoss
FakeQuantizedMultiMarginLoss = _nn.QuantizedMultiMarginLoss
FakeQuantizedNLLLoss = _nn.QuantizedNLLLoss
FakeQuantizedNLLLoss2d = _nn.QuantizedNLLLoss2d
FakeQuantizedPReLU = _nn.QuantizedPReLU
FakeQuantizedPairwiseDistance = _nn.QuantizedPairwiseDistance
FakeQuantizedPixelShuffle = _nn.QuantizedPixelShuffle
FakeQuantizedPixelUnshuffle = _nn.QuantizedPixelUnshuffle
FakeQuantizedPoissonNLLLoss = _nn.QuantizedPoissonNLLLoss
FakeQuantizedRNN = _nn.QuantizedRNN
FakeQuantizedRNNCell = _nn.QuantizedRNNCell
FakeQuantizedRReLU = _nn.QuantizedRReLU
FakeQuantizedReLU = _nn.QuantizedReLU
FakeQuantizedReLU6 = _nn.QuantizedReLU6
FakeQuantizedReflectionPad1d = _nn.QuantizedReflectionPad1d
FakeQuantizedReflectionPad2d = _nn.QuantizedReflectionPad2d

if version.parse(torch.__version__) >= version.parse("1.10.0"):
    FakeQuantizedReflectionPad3d = _nn.QuantizedReflectionPad3d

FakeQuantizedReplicationPad1d = _nn.QuantizedReplicationPad1d
FakeQuantizedReplicationPad2d = _nn.QuantizedReplicationPad2d
FakeQuantizedReplicationPad3d = _nn.QuantizedReplicationPad3d
FakeQuantizedSELU = _nn.QuantizedSELU
FakeQuantizedSiLU = _nn.QuantizedSiLU
FakeQuantizedSigmoid = _nn.QuantizedSigmoid
FakeQuantizedSmoothL1Loss = _nn.QuantizedSmoothL1Loss
FakeQuantizedSoftMarginLoss = _nn.QuantizedSoftMarginLoss
FakeQuantizedSoftmax = _nn.QuantizedSoftmax
FakeQuantizedSoftmax2d = _nn.QuantizedSoftmax2d
FakeQuantizedSoftmin = _nn.QuantizedSoftmin
FakeQuantizedSoftplus = _nn.QuantizedSoftplus
FakeQuantizedSoftshrink = _nn.QuantizedSoftshrink
FakeQuantizedSoftsign = _nn.QuantizedSoftsign
FakeQuantizedTanh = _nn.QuantizedTanh
FakeQuantizedTanhshrink = _nn.QuantizedTanhshrink
FakeQuantizedThreshold = _nn.QuantizedThreshold
FakeQuantizedTripletMarginLoss = _nn.QuantizedTripletMarginLoss
FakeQuantizedTripletMarginWithDistanceLoss = _nn.QuantizedTripletMarginWithDistanceLoss
FakeQuantizedUnflatten = _nn.QuantizedUnflatten
FakeQuantizedUnfold = _nn.QuantizedUnfold
FakeQuantizedUpsample = _nn.QuantizedUpsample
FakeQuantizedUpsamplingBilinear2d = _nn.QuantizedUpsamplingBilinear2d
FakeQuantizedUpsamplingNearest2d = _nn.QuantizedUpsamplingNearest2d

if version.parse(torch.__version__) >= version.parse("2.1.0"):
    FakeQuantizedZeroPad1d = _nn.QuantizedZeroPad1d

FakeQuantizedZeroPad2d = _nn.QuantizedZeroPad2d

if version.parse(torch.__version__) >= version.parse("2.1.0"):
    FakeQuantizedZeroPad3d = _nn.QuantizedZeroPad3d
