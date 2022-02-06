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

**Note about Quantization Schemes** : AIMET offers multiple Quantization Schemes-
    1. Post Training Quantization- The encodings of the model are computed using TF or TF-Enhanced scheme
    2. Trainable Quantization- The min max of encodings are learnt during training.
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

Code Examples
=============

**Required imports**

.. literalinclude:: ../torch_code_examples/quantsim_code_example.py
   :language: python
   :start-after: # Quantsim imports
   :end-before: # End of import statements


**User should write this function to pass calibration data**

.. literalinclude:: ../torch_code_examples/quantsim_code_example.py
   :language: python
   :pyobject: pass_calibration_data

**Quantize the model and finetune (QAT)**

.. literalinclude:: ../torch_code_examples/quantsim_code_example.py
   :language: python
   :pyobject: quantize_and_finetune_example
