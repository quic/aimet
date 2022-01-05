:orphan:

.. _api-torch-quantsim:

==================================
AIMET PyTorch Quantization SIM API
==================================

AIMET Quantization Sim requires PyTorch model definition to follow certain guidelines. These guidelines are described
in detail here. :ref:`Model Guidelines<api-torch-model-guidelines>`

AIMET provides Model Preparer API to allow user to prepare PyTorch model for AIMET Quantization features. The API and
usage examples are described in detail here. :ref:`Model Preparer API<api-torch-model-preparer>`

AIMET also includes a Model Validator utility to allow user to check their model definition. Please see the API and
usage examples for this utility here. :ref:`Model Validator API<api-torch-model-validator>`

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

Encoding format is described in the :ref:`Quantization Encoding Specification<api-quantization-encoding-spec>`

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

.. literalinclude:: ../torch_code_examples/quantsim_code_example.py
   :language: python
   :start-after: # Quantsim imports
   :end-before: # End of import statements

**Train function**

.. literalinclude:: ../torch_code_examples/quantsim_code_example.py
   :language: python
   :pyobject: train

**Evaluation function**

.. literalinclude:: ../torch_code_examples/quantsim_code_example.py
   :language: python
   :pyobject: evaluate

**Quantize and fine-tune a quantized model (QAT)**

.. literalinclude:: ../torch_code_examples/quantsim_code_example.py
   :language: python
   :pyobject: quantsim_example
