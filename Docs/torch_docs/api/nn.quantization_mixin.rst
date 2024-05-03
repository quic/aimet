
.. _api-torch-quantization-mixin:

.. warning::
    This feature is under heavy development and API changes may occur without notice in future verions.


.. currentmodule:: aimet_torch.v2.nn

=================
QuantizationMixin
=================

.. autoclass:: QuantizationMixin
   :members: compute_encodings, from_module, implements, get_original_module, get_kernel, get_default_kernel

   .. automethod:: quantized_forward
   .. automethod:: __quant_init__
   .. automethod:: set_kernel
   .. automethod:: set_default_kernel


