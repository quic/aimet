.. _ug-apidocs:

================================================
Welcome to AI Model Efficiency Toolkit API Docs!
================================================

AI Model Efficiency Toolkit (AIMET) is a software toolkit that enables users to compress
and quantize ML models. The resulting models returned by AIMET can be further trained (or fine-tuned)
to dramatically improve accuracy lost due to quantization and compression.

AIMET is designed to work generically on any user-provided model. At present, AIMET supports
TensorFlow, Keras, and PyTorch training frameworks.

Please follow the links below to see AIMET APIs for either PyTorch, TensorFlow, or Keras.

.. note:: AIMET Keras API requires Tensorflow 2.4 or later.

.. toctree::
   :titlesonly:
   :maxdepth: 2

   AIMET APIs for PyTorch<torch>

.. toctree::
   :titlesonly:
   :maxdepth: 2

   AIMET APIs for TensorFlow<tensorflow>

.. toctree::
   :titlesonly:
   :maxdepth: 2

   AIMET APIs for Keras<keras>

|
|

.. note:: This documentation is auto-generated from the AIMET codebase using Sphinx

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
