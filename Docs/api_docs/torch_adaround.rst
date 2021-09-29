:orphan:

.. _api-torch-adaround:

==================================
AIMET PyTorch AdaRound API
==================================

Top-level API
=============
.. autofunction:: aimet_torch.adaround.adaround_weight.Adaround.apply_adaround


Adaround Parameters
===================
.. autoclass:: aimet_torch.adaround.adaround_weight.AdaroundParameters
    :members:


Enum Definition
===============
**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:

|

Code Examples
=============

**Required imports**

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :start-after: # AdaRound imports
   :end-before: # End of import statements

**Evaluation function**

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :pyobject: dummy_forward_pass

**After applying AdaRound to ResNet18, the AdaRounded model and associated encodings are returned**

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :pyobject: apply_adaround_example
