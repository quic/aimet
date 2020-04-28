:orphan:

.. _tf-bias-correction:

====================================
AIMET TensorFlow Bias Correction API
====================================

Bias Correction API
====================

Bias correction is performed after Cross layer equalization on models.
Main api to perform bias correction on entire model is listed below.

.. autofunction:: aimet_tensorflow.bias_correction.BiasCorrection.correct_bias

Input Parameter Types
=====================

**Quantization Params**

.. autofunction:: aimet_tensorflow.bias_correction.QuantParams

**Bias Correction Params**

.. autofunction:: aimet_tensorflow.bias_correction.BiasCorrectionParams


Data Input Type
===============

**Format expected is tf.Data.DataSet type**

Dataset represents the input data as a tensor or nested structures of
tensors, one per input to model along with an iterator that operates on them.


Code Examples for Bias Correction
=================================

**Required imports**

.. literalinclude:: ../../NightlyTests/tensorflow/code_examples/post_training_techniques_examples.py
    :language: python
    :lines: 41-55

**Only Empirical Bias correction on a given model**

.. literalinclude:: ../../NightlyTests/tensorflow/code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: bias_correction_empirical

**Empirical and Analytical Bias correction on a given model**

.. literalinclude:: ../../NightlyTests/tensorflow/code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: bias_correction_empirical_analytical

**Empirical and Analytical Bias correction on a given model after performing CLE**

.. literalinclude:: ../../NightlyTests/tensorflow/code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: bias_correction_after_cle

Bias Correction Per Layer API
==============================

Empirical/ analytical Bias correction can also be performed on a
subset of selected layers in a given model using the api listed below.

.. autofunction:: aimet_tensorflow.bias_correction.BiasCorrection.bias_correction_per_layer
.. autofunction:: aimet_tensorflow.bias_correction.BiasCorrection.analytical_bias_correction_per_layer

Code Example for Per-Layer Bias Correction
===========================================

**Empirical Bias correction on one layer**

.. literalinclude:: ../../NightlyTests/tensorflow/code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: bias_correction_single_layer_empirical

**Analytical Bias correction on one layer**

.. literalinclude:: ../../NightlyTests/tensorflow/code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: bias_correction_single_layer_analytical