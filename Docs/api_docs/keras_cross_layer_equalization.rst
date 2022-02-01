:orphan:

.. _api-keras-cle:

===============================================
AIMET Keras Cross Layer Equalization APIs
===============================================
Introduction
============
AIMET functionality for Keras Cross Layer Equalization supports three techniques:
   - BatchNorm Folding
   - Cross Layer Scaling
   - High Bias Fold


Cross Layer Equalization API
============================
The top level Cross Layer Equalization for Keras API is still under development. For the time being, the Primitive APIs
can be used to perform each step of Cross Layer Equalization separately.

Primitive APIs
==============
If the user would like to call the APIs individually, then the following APIs can be used:

.. toctree::
    :titlesonly:
    :maxdepth: 1

    Primitive APIs for Cross Layer Equalization<keras_primitive_apis_cle>
