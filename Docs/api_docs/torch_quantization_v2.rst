.. _api-torch-v2-quantization:

==============
aimet_torch.v2
==============

Introducing aimet_torch v2, a future version of aimet_torch with more powerful quantization features and PyTorch-friendly user interface!

What's New
==========
These are some of the powerful new features and interfaces supported in aimet_torch.v2

* :ref:`Blockwise Quantization<api-torch-blockwise-quantization>`
* :ref:`Low Power Blockwise Quantization (LPBQ)<api-torch-blockwise-quantization>`
* Dispatching Custom Quantized Kernels

Backwards Compatibility
========================
Good news! aimet_torch.v2 is carefully designed to be fully backwards-compatibile  with all previous public APIs of aimet_torch.
All you need is drop-in replacement of import statements from mod:`aimet_torch` to :mod:`aimet_torch.v2` as below!

.. code-block:: diff

   -from aimet_torch.quantsim import QuantizationSimModel
   +from aimet_torch.v2.quantsim import QuantizationSimModel

   -from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
   +from aimet_torch.v2.adaround import Adaround, AdaroundParameters

   -from aimet_torch.seq_mse import apply_seq_mse
   +from aimet_torch.v2.seq_mse import apply_seq_mse

   -from aimet_torch.quant_analyzer import QuantAnalyzer
   +from aimet_torch.v2.quant_analyzer import QuantAnalyzer


All the other APIs that didn't changed in or are orthogonal with aimet_torch.v2 will be still accessible via :mod:`aimet_torch` namespace as before.


For more detailed information about how to migrate to aimet_torch.v2, see :ref:`aimet_torch.v2 migration guide<tutorials-migration-guide>`


API Reference
=============
.. toctree::
   :titlesonly:
   :maxdepth: 1

   ../torch_docs/quantized_modules
   ../torch_docs/quantizer
   ../torch_docs/encoding_analyzer
   ../torch_docs/api/nn.fake_quantization_mixin
   ../torch_docs/api/nn.quantization_mixin
   ../torch_docs/api/quantization/affine/index
   ../torch_docs/api/quantization/float/index
