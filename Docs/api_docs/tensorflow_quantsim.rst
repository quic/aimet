:orphan:

.. _tf-quantsim:

===========================
QuantSim API for TensorFlow
===========================

Top-level API
=============

.. autoclass:: aimet_tensorflow.quantsim.QuantizationSimModel

|

**The following API can be used to Compute Encodings for Model**

.. automethod:: aimet_tensorflow.quantsim.QuantizationSimModel.compute_encodings


|

**The following API can be used to Export the Model to target**

.. automethod:: aimet_tensorflow.quantsim.QuantizationSimModel.export

|


Code Examples
=============

**Required imports**

.. literalinclude:: ../../NightlyTests/tensorflow/code_examples/quantization.py
    :language: python
    :lines: 39-46

**Quantize with Range Learning**

.. literalinclude:: ../../NightlyTests/tensorflow/code_examples/quantization.py
    :language: python
    :pyobject: quantize_model

**Example Fine-tuning step**

.. literalinclude:: ../../NightlyTests/tensorflow/code_examples/quantization.py
    :language: python
    :pyobject: training_helper

