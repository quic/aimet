:orphan:

.. _api-tf-quantsim:

=====================================
AIMET TensorFlow Quantization SIM API
=====================================

User Guide Link
===============
To learn more about Quantization Simulation, please see :ref:`Quantization Sim<ug-quantsim>`

Examples Notebook Link
======================
For an end-to-end notebook showing how to use TensorFlow Quantization-Aware Training, please see :doc:`here<../Examples/tensorflow/quantization/qat>`.

Top-level API
=============

.. autoclass:: aimet_tensorflow.quantsim.QuantizationSimModel

|

**Note about Quantization Schemes** : AIMET offers multiple Quantization Schemes-
    1. Post Training Quantization- The encodings of the model are computed using TF or TF-Enhanced scheme
    2. Trainable Quantization- The min max of encodings are learnt during training.
        * Range Learning with TF initialization - Uses TF scheme to initialize the encodings and then during training these encodings are fine-tuned to improve accuracy of the model
        * Range Learning with TF-Enhanced initialization - Uses TF-Enhanced scheme to initialize the encodings and then during training these encodings are fine-tuned to improve accuracy of the model

**The following API can be used to Compute Encodings for Model**

.. automethod:: aimet_tensorflow.quantsim.QuantizationSimModel.compute_encodings


|

**The following API can be used to Export the Model to target**

.. automethod:: aimet_tensorflow.quantsim.QuantizationSimModel.export
   
|

Encoding format is described in the :ref:`Quantization Encoding Specification<api-quantization-encoding-spec>`

|


Code Examples
=============

**Required imports**

.. literalinclude:: ../tf_code_examples/quantization.py
    :language: python
    :lines: 39-46

**User should write this function to pass calibration data**


.. literalinclude:: ../tf_code_examples/quantization.py
   :language: python
   :pyobject: pass_calibration_data


**Quantize the model and finetune (QAT)**

.. literalinclude:: ../tf_code_examples/quantization.py
    :language: python
    :pyobject: quantize_model


**Quantize and finetune a trained model learn the encodings (Range Learning)**

.. literalinclude:: ../tf_code_examples/range_learning.py
   :language: python
   :pyobject: quantization_aware_training_range_learning