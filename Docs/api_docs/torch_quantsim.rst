:orphan:

.. _api-torch-quantsim:

==================================
AIMET PyTorch Quantization SIM API
==================================

AIMET Quantization Sim requires the model definitions to use certain constructs and avoid others. These constraints are
described in detail :ref:`here<api-torch-model-validator>`.

AIMET also includes a Model Validator tool to allow the users to check their model definition and find constructs that
might need to be replaced. Please see the API and usage examples for this tool also on the same page.

Top-level API
=============

.. autoclass:: aimet_torch.quantsim.QuantizationSimModel

|

**Note about Quantization Schemes** : AIMET offers multiple Quantization Schemes-
    1. Post Training Quantization- The encodings of the model are computed using TF or TF-Enhanced scheme
    2. Trainable Quantization- The min max of encodings are learnt during training
        * Range Learning with TF initialization - Uses TF scheme to initialize the encodings and then during training these encodings are fine-tuned to improve accuracy of the model
        * Range Learning with TF-Enhanced initialization - Uses TF-Enhanced scheme to initialize the encodings and then during training these encodings are fine-tuned to improve accuracy of the model

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


Code Example #1 - Post Training Quantization
============================================

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

|

Code Example #2 - Trainable Quantization
========================================

**Required imports**

.. literalinclude:: ../torch_code_examples/range_learning.py
   :language: python
   :start-after: # Eval function related import
   :end-before: # End of import statements

**Evaluation function to be used for computing initial encodings**

.. literalinclude:: ../torch_code_examples/code_examples.py
   :language: python
   :pyobject: evaluate_model

**Quantize and fine-tune a trained model to learn min max ranges**

.. literalinclude:: ../torch_code_examples/range_learning.py
   :language: python
   :pyobject: quantization_aware_training_range_learning
