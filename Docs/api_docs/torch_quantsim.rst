:orphan:

.. _api-torch-quantsim:

==================================
AIMET PyTorch Quantization SIM API
==================================

Top-level API
=============

.. autoclass:: aimet_torch.quantsim.QuantizationSimModel

|

**The following API can be used to Compute Encodings for Model**

.. automethod:: aimet_torch.quantsim.QuantizationSimModel.compute_encodings

|

**The following APIs can be used to save and restore the quantized model**

.. automethod:: aimet_torch.quantsim.save_checkpoint

|

.. automethod:: aimet_torch.quantsim.load_checkpoint

|

**The following API can be used to Export the Model to target**

.. automethod:: aimet_torch.quantsim.QuantizationSimModel.export

|


Enum Definition
===============
**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:

|

Code Examples
=============

**Required imports**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :lines: 43, 56, 52-53

**Evaluation function**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :pyobject: evaluate_model

**Quantize and fine-tune a trained model**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :pyobject: quantize_model
