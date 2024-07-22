.. _api-torch-quantized-modules:

.. currentmodule:: aimet_torch.v2.nn

.. warning::
    This feature is under heavy development and API changes may occur without notice in future versions.

=================
Quantized Modules
=================

To simulate the effects of running networks at a reduced bitwidth, AIMET provides quantized versions of
standard torch.nn.Modules. These quantized modules serve as drop-in replacements for their PyTorch counterparts, but can
hold input, output, and parameter :ref:`quantizers<api-torch-quantizers>` to perform quantization operations during the
module's forward pass and compute quantization encodings.

A quantized module inherits both from an AIMET-defined quantization mixin type as well as a native pytorch `nn.Module` type. The
exact behavior and capabilities of the quantized module are determined by which type of quantization mixin it inherits from.

AIMET defines two types of quantization mixin:

    - :ref:`FakeQuantizationMixin<api-torch-fake-quantization-mixin>`: Simulates quantization by performing quantize-dequantize
      operations on tensors and calling into native pytorch floating-point operations

    - :ref:`QuantizationMixin<api-torch-quantization-mixin>`: Allows the user to register a custom kernel to perform
      a quantized forward pass and dequantizes the output. If no kernel is registered, the module will perform fake-quantization.

The functionality and state of a :ref:`QuantizationMixin<api-torch-quantization-mixin>` is a superset of that of a :ref:`FakeQuantizationMixin<api-torch-fake-quantization-mixin>`, meaning that
if one does not register a custom kernel, a :ref:`QuantizationMixin<api-torch-quantization-mixin>`-derived module behaves
exactly the same as a :ref:`FakeQuantizationMixin<api-torch-fake-quantization-mixin>`-derived module. AIMET provides
extensive coverage of :ref:`FakeQuantizationMixin<api-torch-fake-quantization-mixin>` for ``torch.nn.Module`` layer types, and more limited coverage for
:ref:`QuantizationMixin<api-torch-quantization-mixin>` layers. See the :ref:`table below<api-quantized-module-class-table>` for a full list of module coverage.

Top-level API
=============

.. autoclass:: aimet_torch.v2.nn.base.BaseQuantizationMixin
   :members: __quant_init__, forward, compute_encodings

Configuration
=============

The quantization behavior of a quantized module is controlled by the :ref:`quantizers<api-torch-quantizers>` contained within the input, output,
and parameter quantizer attributes listed below.

================= ====================== ========================================
Attribute         Type                   Description
================= ====================== ========================================
input_quantizers  torch.nn.ModuleList    List of quantizers for input tensors
param_quantizers  torch.nn.ModuleDict    Dict mapping parameter names to quantizers
output_quantizers torch.nn.ModuleList    List of quantizers for output tensors
================= ====================== ========================================

By assigning and configuring :ref:`quantizers<api-torch-quantizers>` to these structures, we define the type of quantization applied to the corresponding
input index, output index, or parameter name. By default, all the quantizers are set to `None`, meaning that no quantization
will be applied to the respective tensor.

Example: Create a linear layer which performs only per-channel weight quantization
    >>> import aimet_torch.v2 as aimet
    >>> import aimet_torch.quantization as Q
    >>> qlinear = aimet.nn.QuantizedLinear(out_features=10, in_features=5)
    >>> # Per-channel weight quantization is performed over the `out_features` dimension, so encodings are shape (10, 1)
    >>> per_channel_quantizer = Q.affine.QuantizeDequantize(shape=(10, 1), bitwidth=8, symmetric=True)
    >>> qlinear.param_quantizers["weight"] = per_channel_quantizer

Example: Create an elementwise multiply layer which quantizes only the output and the second input
    >>> qmul = aimet.nn.QuantizedMultiply()
    >>> qmul.output_quantizers[0] = Q.affine.QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
    >>> qmul.input_quantizers[1] = Q.affine.QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)

In some cases, it may make sense for multiple tensors to share the same quantizer. In this case, we can assign the same
quantizer to multiple indices.

Example: Create an elementwise add layer which shares the same quantizer between its inputs
    >>> qadd = aimet.nn.QuantizedAdd()
    >>> quantizer = Q.affine.QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
    >>> qadd.input_quantizers[0] = quantizer
    >>> qadd.input_quantizers[1] = quantizer

Computing Encodings
===================

Before a module can compute a quantized forward pass, all quantizers must first be calibrated inside a `compute_encodings`
context. When a quantized module enters the `compute_encodings` context, it first disables all input and output quantization
while the quantizers observe the statistics of the activation tensors passing through them. Upon exiting the context,
the quantizers calculate appropriate quantization encodings based on these statistics (exactly *how* the encodings are
computed is determined by each quantizer's :ref:`encoding analyzer<api-torch-encoding-analyzer>`).

Example:
    >>> qlinear = aimet.nn.QuantizedLinear(out_features=10, in_features=5)
    >>> qlinear.output_quantizers[0] = Q.affine.QuantizeDequantize((1, ), bitwidth=8, symmetric=False)
    >>> qlinear.param_quantizers[0] = Q.affine.QuantizeDequantize((10, 1), bitwidth=8, symmetric=True)
    >>> with qlinear.compute_encodings():
    ...     # Pass several samples through the layer to ensure representative statistics
    ...     for x, _ in calibration_data_loader:
    ...         qlinear(x)
    >>> print(qlinear.output_quantizers[0].is_initialized())
    True
    >>> print(qlinear.param_quantizers["weight"].is_initialized())
    True


Quantized Module Classes
========================
.. _api-quantized-module-class-table:

============================================= ============================================= ============================
nn.Module                                     FakeQuantizationMixin                         QuantizationMixin
============================================= ============================================= ============================
torch.nn.AdaptiveAvgPool1d                    FakeQuantizedAdaptiveAvgPool1d
torch.nn.AdaptiveAvgPool2d                    FakeQuantizedAdaptiveAvgPool2d
torch.nn.AdaptiveAvgPool3d                    FakeQuantizedAdaptiveAvgPool3d
torch.nn.AdaptiveMaxPool1d                    FakeQuantizedAdaptiveMaxPool1d
torch.nn.AdaptiveMaxPool2d                    FakeQuantizedAdaptiveMaxPool2d
torch.nn.AdaptiveMaxPool3d                    FakeQuantizedAdaptiveMaxPool3d
torch.nn.AlphaDropout                         FakeQuantizedAlphaDropout
torch.nn.AvgPool1d                            FakeQuantizedAvgPool1d
torch.nn.AvgPool2d                            FakeQuantizedAvgPool2d
torch.nn.AvgPool3d                            FakeQuantizedAvgPool3d
torch.nn.BatchNorm1d                          FakeQuantizedBatchNorm1d
torch.nn.BatchNorm2d                          FakeQuantizedBatchNorm2d
torch.nn.BatchNorm3d                          FakeQuantizedBatchNorm3d
torch.nn.CELU                                 FakeQuantizedCELU
torch.nn.ChannelShuffle                       FakeQuantizedChannelShuffle
torch.nn.ConstantPad1d                        FakeQuantizedConstantPad1d
torch.nn.ConstantPad2d                        FakeQuantizedConstantPad2d
torch.nn.ConstantPad3d                        FakeQuantizedConstantPad3d
torch.nn.Conv1d                               FakeQuantizedConv1d                           QuantizedConv1d
torch.nn.Conv2d                               FakeQuantizedConv2d                           QuantizedConv2d
torch.nn.Conv3d                               FakeQuantizedConv3d                           QuantizedConv3d
torch.nn.ConvTranspose1d                      FakeQuantizedConvTranspose1d
torch.nn.ConvTranspose2d                      FakeQuantizedConvTranspose2d
torch.nn.ConvTranspose3d                      FakeQuantizedConvTranspose3d
torch.nn.CrossMapLRN2d                        FakeQuantizedCrossMapLRN2d
torch.nn.Dropout                              FakeQuantizedDropout
torch.nn.Dropout2d                            FakeQuantizedDropout2d
torch.nn.Dropout3d                            FakeQuantizedDropout3d
torch.nn.ELU                                  FakeQuantizedELU
torch.nn.FeatureAlphaDropout                  FakeQuantizedFeatureAlphaDropout
torch.nn.Flatten                              FakeQuantizedFlatten
torch.nn.Fold                                 FakeQuantizedFold
torch.nn.FractionalMaxPool2d                  FakeQuantizedFractionalMaxPool2d
torch.nn.FractionalMaxPool3d                  FakeQuantizedFractionalMaxPool3d
torch.nn.GELU                                 FakeQuantizedGELU                             QuantizedGELU
torch.nn.GLU                                  FakeQuantizedGLU
torch.nn.GroupNorm                            FakeQuantizedGroupNorm
torch.nn.Hardshrink                           FakeQuantizedHardshrink
torch.nn.Hardsigmoid                          FakeQuantizedHardsigmoid
torch.nn.Hardswish                            FakeQuantizedHardswish
torch.nn.Hardtanh                             FakeQuantizedHardtanh
torch.nn.Identity                             FakeQuantizedIdentity
torch.nn.InstanceNorm1d                       FakeQuantizedInstanceNorm1d
torch.nn.InstanceNorm2d                       FakeQuantizedInstanceNorm2d
torch.nn.InstanceNorm3d                       FakeQuantizedInstanceNorm3d
torch.nn.LPPool1d                             FakeQuantizedLPPool1d
torch.nn.LPPool2d                             FakeQuantizedLPPool2d
torch.nn.LayerNorm                            FakeQuantizedLayerNorm                        QuantizedLayerNorm
torch.nn.LeakyReLU                            FakeQuantizedLeakyReLU
torch.nn.Linear                               FakeQuantizedLinear                           QuantizedLinear
torch.nn.LocalResponseNorm                    FakeQuantizedLocalResponseNorm
torch.nn.LogSigmoid                           FakeQuantizedLogSigmoid
torch.nn.LogSoftmax                           FakeQuantizedLogSoftmax
torch.nn.MaxPool1d                            FakeQuantizedMaxPool1d
torch.nn.MaxPool2d                            FakeQuantizedMaxPool2d
torch.nn.MaxPool3d                            FakeQuantizedMaxPool3d
torch.nn.MaxUnpool1d                          FakeQuantizedMaxUnpool1d
torch.nn.MaxUnpool2d                          FakeQuantizedMaxUnpool2d
torch.nn.MaxUnpool3d                          FakeQuantizedMaxUnpool3d
torch.nn.Mish                                 FakeQuantizedMish
torch.nn.PReLU                                FakeQuantizedPReLU
torch.nn.PixelShuffle                         FakeQuantizedPixelShuffle
torch.nn.PixelUnshuffle                       FakeQuantizedPixelUnshuffle
torch.nn.RReLU                                FakeQuantizedRReLU
torch.nn.ReLU                                 FakeQuantizedReLU
torch.nn.ReLU6                                FakeQuantizedReLU6
torch.nn.ReflectionPad1d                      FakeQuantizedReflectionPad1d
torch.nn.ReflectionPad2d                      FakeQuantizedReflectionPad2d
torch.nn.ReplicationPad1d                     FakeQuantizedReplicationPad1d
torch.nn.ReplicationPad2d                     FakeQuantizedReplicationPad2d
torch.nn.ReplicationPad3d                     FakeQuantizedReplicationPad3d
torch.nn.SELU                                 FakeQuantizedSELU
torch.nn.SiLU                                 FakeQuantizedSiLU
torch.nn.Sigmoid                              FakeQuantizedSigmoid                          QuantizedSigmoid
torch.nn.Softmax                              FakeQuantizedSoftmax                          QuantizedSoftmax
torch.nn.Softmax2d                            FakeQuantizedSoftmax2d
torch.nn.Softmin                              FakeQuantizedSoftmin
torch.nn.Softplus                             FakeQuantizedSoftplus
torch.nn.Softshrink                           FakeQuantizedSoftshrink
torch.nn.Softsign                             FakeQuantizedSoftsign
torch.nn.SyncBatchNorm                        FakeQuantizedSyncBatchNorm
torch.nn.Tanh                                 FakeQuantizedTanh
torch.nn.Tanhshrink                           FakeQuantizedTanhshrink
torch.nn.Threshold                            FakeQuantizedThreshold
torch.nn.Unflatten                            FakeQuantizedUnflatten
torch.nn.Unfold                               FakeQuantizedUnfold
torch.nn.Upsample                             FakeQuantizedUpsample
torch.nn.UpsamplingBilinear2d                 FakeQuantizedUpsamplingBilinear2d
torch.nn.UpsamplingNearest2d                  FakeQuantizedUpsamplingNearest2d
torch.nn.ZeroPad2d                            FakeQuantizedZeroPad2d
torch.nn.BCELoss                              FakeQuantizedBCELoss
torch.nn.BCEWithLogitsLoss                    FakeQuantizedBCEWithLogitsLoss
torch.nn.Bilinear                             FakeQuantizedBilinear
torch.nn.CTCLoss                              FakeQuantizedCTCLoss
torch.nn.CosineSimilarity                     FakeQuantizedCosineSimilarity
torch.nn.CrossEntropyLoss                     FakeQuantizedCrossEntropyLoss
torch.nn.HingeEmbeddingLoss                   FakeQuantizedHingeEmbeddingLoss
torch.nn.HuberLoss                            FakeQuantizedHuberLoss
torch.nn.KLDivLoss                            FakeQuantizedKLDivLoss
torch.nn.L1Loss                               FakeQuantizedL1Loss
torch.nn.MSELoss                              FakeQuantizedMSELoss
torch.nn.MultiLabelMarginLoss                 FakeQuantizedMultiLabelMarginLoss
torch.nn.MultiLabelSoftMarginLoss             FakeQuantizedMultiLabelSoftMarginLoss
torch.nn.MultiMarginLoss                      FakeQuantizedMultiMarginLoss
torch.nn.NLLLoss                              FakeQuantizedNLLLoss
torch.nn.NLLLoss2d                            FakeQuantizedNLLLoss2d
torch.nn.PairwiseDistance                     FakeQuantizedPairwiseDistance
torch.nn.PoissonNLLLoss                       FakeQuantizedPoissonNLLLoss
torch.nn.SmoothL1Loss                         FakeQuantizedSmoothL1Loss
torch.nn.SoftMarginLoss                       FakeQuantizedSoftMarginLoss
torch.nn.CosineEmbeddingLoss                  FakeQuantizedCosineEmbeddingLoss
torch.nn.GaussianNLLLoss                      FakeQuantizedGaussianNLLLoss
torch.nn.MarginRankingLoss                    FakeQuantizedMarginRankingLoss
torch.nn.TripletMarginLoss                    FakeQuantizedTripletMarginLoss
torch.nn.TripletMarginWithDistanceLoss        FakeQuantizedTripletMarginWithDistanceLoss
torch.nn.Embedding                            FakeQuantizedEmbedding
torch.nn.EmbeddingBag                         FakeQuantizedEmbeddingBag
torch.nn.GRU                                  FakeQuantizedGRU
torch.nn.RNN                                  FakeQuantizedRNN
torch.nn.GRUCell                              FakeQuantizedGRUCell
torch.nn.RNNCell                              FakeQuantizedRNNCell
torch.nn.LSTM                                 FakeQuantizedLSTM
torch.nn.LSTMCell                             FakeQuantizedLSTMCell
torch.nn.AdaptiveLogSoftmaxWithLoss           FakeQuantizedAdaptiveLogSoftmaxWithLoss
aimet_ops.ChannelShuffle                      FakeQuantizedChannelShuffle
aimet_ops.MaxPool2d                           FakeQuantizedMaxPool2d
aimet_ops.AdaptiveAvgPool2d                   FakeQuantizedAdaptiveAvgPool2d
aimet_ops.AvgPool2d                           FakeQuantizedAvgPool2d
aimet_ops.Cast                                FakeQuantizedCast
aimet_ops.DepthToSpaceDCRMode                 FakeQuantizedDepthToSpaceDCRMode
aimet_ops.OneHot                              FakeQuantizedOneHot
aimet_ops.Exponential                         FakeQuantizedExponential
aimet_ops.Erf                                 FakeQuantizedErf
aimet_ops.Sqrt                                FakeQuantizedSqrt
aimet_ops.Log                                 FakeQuantizedLog
aimet_ops.Abs                                 FakeQuantizedAbs
aimet_ops.Neg                                 FakeQuantizedNeg
aimet_ops.ElementwiseCeil                     FakeQuantizedElementwiseCeil
aimet_ops.ElementwiseFloor                    FakeQuantizedElementwiseFloor
aimet_ops.Sin                                 FakeQuantizedSin
aimet_ops.Cos                                 FakeQuantizedCos
aimet_ops.Asin                                FakeQuantizedAsin
aimet_ops.Atan                                FakeQuantizedAtan
aimet_ops.Round                               FakeQuantizedRound
aimet_ops.LogicalNot                          FakeQuantizedLogicalNot
aimet_ops.NonZero                             FakeQuantizedNonZero
aimet_ops.ElementwiseUnarySign                FakeQuantizedElementwiseUnarySign
aimet_ops.RSqrt                               FakeQuantizedRSqrt
aimet_ops.Square                              FakeQuantizedSquare
aimet_ops.Mean                                FakeQuantizedMean
aimet_ops.Sum                                 FakeQuantizedSum
aimet_ops.Prod                                FakeQuantizedProd
aimet_ops.Argmin                              FakeQuantizedArgmin
aimet_ops.Argmax                              FakeQuantizedArgmax
aimet_ops.Gather                              FakeQuantizedGather
aimet_ops.Reshape                             FakeQuantizedReshape
aimet_ops.RoiAlign                            FakeQuantizedRoiAlign
aimet_ops.Permute                             FakeQuantizedPermute
aimet_ops.IndexSelect                         FakeQuantizedIndexSelect
aimet_ops.TopK                                FakeQuantizedTopK
aimet_ops.Tile                                FakeQuantizedTile
aimet_ops.Norm                                FakeQuantizedNorm
aimet_ops.CumSum                              FakeQuantizedCumSum
aimet_ops.Interpolate                         FakeQuantizedInterpolate
aimet_ops.Normalize                           FakeQuantizedNormalize
aimet_ops.Pad                                 FakeQuantizedPad
aimet_ops.Shape                               FakeQuantizedShape
aimet_ops.Expand                              FakeQuantizedExpand
aimet_ops.StridedSlice                        FakeQuantizedStridedSlice
aimet_ops.MatMul                              FakeQuantizedMatMul
aimet_ops.Add                                 FakeQuantizedAdd                              QuantizedAdd
aimet_ops.Multiply                            FakeQuantizedMultiply                         QuantizedMultiply
aimet_ops.Subtract                            FakeQuantizedSubtract                         QuantizedSubtract
aimet_ops.Divide                              FakeQuantizedDivide
aimet_ops.FloorDivide                         FakeQuantizedFloorDivide
aimet_ops.Greater                             FakeQuantizedGreater
aimet_ops.Less                                FakeQuantizedLess
aimet_ops.GreaterEqual                        FakeQuantizedGreaterEqual
aimet_ops.LessEqual                           FakeQuantizedLessEqual
aimet_ops.NotEqual                            FakeQuantizedNotEqual
aimet_ops.Equal                               FakeQuantizedEqual
aimet_ops.Remainder                           FakeQuantizedRemainder
aimet_ops.Fmod                                FakeQuantizedFmod
aimet_ops.Pow                                 FakeQuantizedPow
aimet_ops.CustomSiLU                          FakeQuantizedCustomSiLU
aimet_ops.Maximum                             FakeQuantizedMaximum
aimet_ops.Max                                 FakeQuantizedMax
aimet_ops.Minimum                             FakeQuantizedMinimum
aimet_ops.Min                                 FakeQuantizedMin
aimet_ops.Bmm                                 FakeQuantizedBmm
aimet_ops.LogicalOr                           FakeQuantizedLogicalOr
aimet_ops.LogicalAnd                          FakeQuantizedLogicalAnd
aimet_ops.CustomGather                        FakeQuantizedCustomGather
aimet_ops.GatherNd                            FakeQuantizedGatherNd
aimet_ops.Baddbmm                             FakeQuantizedBaddbmm
aimet_ops.Addmm                               FakeQuantizedAddmm
aimet_ops.ScatterND                           FakeQuantizedScatterND
aimet_ops.DynamicConv2d                       FakeQuantizedDynamicConv2d
aimet_ops.ScatterElements                     FakeQuantizedScatterElements
aimet_ops.BatchNorm                           FakeQuantizedBatchNorm
aimet_ops.GroupNorm                           FakeQuantizedAimetGroupNorm
aimet_ops.NonMaxSuppression                   FakeQuantizedNonMaxSuppression
aimet_ops.Split                               FakeQuantizedSplit
aimet_ops.Concat                              FakeQuantizedConcat
aimet_ops.Where                               FakeQuantizedWhere
aimet_ops.MaskedFill                          FakeQuantizedMaskedFill
============================================= ============================================= ============================
