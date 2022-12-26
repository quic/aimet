:orphan:

.. _api-keras-bn-reestimation:

===============================================
AIMET Keras BatchNorm Re-estimation APIs
===============================================

Examples Notebook Link
======================
For an end-to-end notebook showing how to use Keras Quantization-Aware Training with BatchNorm Re-estimation, please see :doc:`here<../Examples/tensorflow/quantization/keras/bn_reestimation>`.

Introduction
============
AIMET functionality for Keras BatchNorm Re-estimation recalculates the batchnorm statistics based on the model after QAT. By doing so, we aim to make our model learn batchnorm statistics from from stable outputs after QAT, rather than from likely noisy outputs during QAT:


Top-level APIs
============================

**API(s) for  BatchNorm Re-estimation**

.. autofunction:: aimet_tensorflow.keras.bn_reestimation.reestimate_bn_stats


Code Example
============

**Required imports**

.. literalinclude:: ../keras_code_examples/bn_reestimation_example.py
    :language: python
    :lines: 44

**Perform BatchNorm Re-estimation**

.. literalinclude:: ../keras_code_examples/bn_reestimation_example.py
    :language: python
    :start-after: # start BatchNorm Re-estimation
    :end-before: # end BatchNorm Re-estimation


Limitations
===========
Please see The AIMET Keras ModelPreparer API limitations:
