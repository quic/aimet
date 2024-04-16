.. _api-torch-quantized-modules:

.. currentmodule:: aimet_torch.v2.nn

.. warning::
    This feature is under heavy development and API changes may occur without notice in future verions.

=================
Quantized Modules
=================

To simulate the effects of running networks at an reduced (or integer) bitwidth, AIMET provides quantized versions of
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
if one does not intend to register a custom kernel, it does not matter which mixin a module is inherited from. AIMET provides
extensive coverage of :ref:`FakeQuantizationMixin<api-torch-fake-quantization-mixin>` for `torch.nn.Module` layer types, and more limited coverage for
:ref:`QuantizationMixin<api-torch-quantization-mixin>` layers.

Top-level API
=============

.. autoclass:: aimet_torch.v2.nn.base.BaseQuantizationMixin
   :exclude-members: wrap, export_input_encodings, export_output_encodings, export_param_encodings, import_input_encodings, import_output_encodings, import_param_encodings
   :members:
   :special-members: __quant_init__

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
    >>> from aimet_torch.v2.quantization import affine
    >>> qlinear = aimet_torch.v2.nn.QuantizedLinear(out_features=10, in_features=5)
    >>> # Per-channel weight quantization is performed over the `out_features` dimension, so encodings are shape (10, 1)
    >>> per_channel_quantizer = affine.QuantizeDequantize(shape=(10, 1), bitwidth=8, symmetric=True)
    >>> qlinear.param_quantizers["weight"] = per_channel_quantizer

Example: Create an elementwise multiply layer which quantizes only the output and the second input
    >>> qmul = aimet_torch.v2.nn.QuantizedMultiply()
    >>> qmul.output_quantizers[0] = affine.QuantizeDequantize(shape=(1, ), bitwidth=8, symmetric=False)
    >>> qmul.input_quantizers[1] = affine.QuantizeDequantize(shape=(1, ), bitwidth=8, symmetric=False)

In some cases, it may make sense for multiple tensors to share the same quantizer. In this case, we can assign the same
quantizer to multiple indices.

Example: Create an elementwise add layer which shares the same quantizer between its inputs
    >>> qadd = aimet_torch.v2.nn.QuantizedAdd()
    >>> quantizer = affine.QuantizeDequantize(shape=(1, ), bitwidth=8, symmetric=False)
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
    >>> qlinear = aimet_torch.v2.nn.QuantizedLinear(out_features=10, in_features=5)
    >>> qlinear.output_quantizers[0] = affine.QuantizeDequantize((1, ), bitwidth=8, symmetric=False)
    >>> qlinear.param_quantizers[0] = affine.QuantizeDequantize((10, 1), bitwidth=8, symmetric=True)
    >>> with qlinear.compute_encodings():
    ...     # Pass several samples through the layer to ensure representative statistics
    ...     for x, _ in calibration_data_loader:
    ...         qlinear(x)
    >>> print(qlinear.output_quantizers[0].is_initialized())
    True
    >>> print(qlinear.param_quantizers["weight"].is_initialized())
    True

