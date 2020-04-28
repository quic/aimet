:orphan:

.. _tf-cle:

===============================================
AIMET TensorFlow Cross Layer Equalization APIs
===============================================
Introduction
============
AIMET functionality for TensorFlow Cross Layer Equalization supports three techniques-
   - BatchNorm Folding
   - Cross Layer Scaling
   - High Bias Fold


Cross Layer Equalization API
============================
Listed below is a comprehensive API to apply all available techniques under cross layer equalization.
It performs 'auto' detection of candidate layers and applies the techniques.
If there are no BatchNorm layers in a given model, BatchNorm fold and high bias fold shall be skipped.

**API(s) for Cross Layer Equalization**

.. autofunction:: aimet_tensorflow.cross_layer_equalization.equalize_model


Code Example
============

**Required imports**

.. literalinclude:: ../../NightlyTests/tensorflow/code_examples/post_training_techniques_examples.py
    :language: python
    :lines: 41-47

**Cross Layer Equalization in auto mode comprehensive**

.. literalinclude:: ../../NightlyTests/tensorflow/code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: cross_layer_equalization_auto


Primitive APIs
==============
If the user would like to call the APIs individually, then the following APIs can be used-

.. toctree::
    :titlesonly:
    :maxdepth: 1

    Primitive APIs for Cross Layer Equalization<tensorflow_primitive_apis_cle>
