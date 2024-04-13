.. _api-torch-quantized-modules:

.. currentmodule:: aimet_torch.v2.nn

.. warning::
    This feature is under heavy development and API changes may occur without notice in future verions.

=================
Quantized Modules
=================

AIMET defines quantized versions of standard torch.nn.Modules. These behave as drop-in replacements for their PyTorch
counterparts but can hold input, output, and parameter :ref:`quantizers<api-torch-quantizers>` to perform quantization
operations during the module's forward pass.


Overview
========


AIMET defines quantized modules via two distinct types of mixins:

    - :ref:`QuantizationMixin<api-torch-quantization-mixin>`: performs "true" quantization by calling into custom
      quantized libraries after quantizing input and parameter tensors.

    - :ref:`FakeQuantizationMixin<api-torch-fake-quantization-mixin>`: performs fake quantization by performing quantize-dequantize
      operations on tensors and calling into native pytorch floating-point operations

While the forward-pass behavior of a quantized module is dependent on which mixin it was created with,
all quantized modules share the same basic structure for configuring and exporting layer quantizers.

Top-level API
=============

.. autoclass:: aimet_torch.v2.nn.base.BaseQuantizationMixin
   :exclude-members: wrap
   :members:
   :special-members: __quant_init__

Class attributes:
-----------------

================= ====================== ========================================
Attribute         Type                   Description
================= ====================== ========================================
input_quantizers  torch.nn.ModuleList    List of quantizers for input tensors
param_quantizers  torch.nn.ModuleDict    Dict mapping parameter names to quantizers
output_quantizers torch.nn.ModuleList    List of quantizers for output tensors
================= ====================== ========================================

