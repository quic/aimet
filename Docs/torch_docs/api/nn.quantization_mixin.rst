
.. _api-torch-quantization-mixin:

.. warning::
    This feature is under heavy development and API changes may occur without notice in future verions.


.. currentmodule:: aimet_torch.v2.nn

====================
nn.QuantizationMixin
====================

Mixin for adding full quantization functionality to `nn.Module` subclasses. This functionality includes both the ability
to set input, output, and parameter quantizers as well as the ability to register a quantized version of the layer's
forward operation.

Top-level API
=============
.. autoclass:: QuantizationMixin
   :members:


