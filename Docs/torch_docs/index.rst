
.. _ug-torch-apidocs:

========================================================
Welcome to AI Model Efficiency Toolkit PyTorch API Docs!
========================================================

AI Model Efficiency Toolkit (AIMET) is a software toolkit that enables users to quantize and compress models.
Quantization is a must for efficient edge inference using fixed-point AI accelerators.

AIMET optimizes pre-trained models (e.g., FP32 trained models) using post-training and fine-tuning techniques that
minimize accuracy loss incurred during quantization or compression.

AIMET PyTorch is designed to work on any user-provided PyTorch model. Please follow the links on this page to see
AIMET PyTorch tutorials, examples, features, and APIs. To view AIMET's Tensorflow, ONNX, Keras documentation,
please visit our `AIMET v1 Documentation <https://quic.github.io/aimet-pages/releases/latest/user_guide/index.html>`_ page.

.. image:: ../images/AIMET_index_no_fine_tune.png

The above picture shows a high-level view of the workflow when using AIMET. The user will start with a trained
floating-point model. This trained model is passed to AIMET using APIs
for compression and quantization. AIMET returns a compressed/quantized version of the model
that the users can fine-tune (or train further for a small number of epochs) to recover lost accuracy. Users can then
export via ONNX/torchscript to an on-target runtime like Qualcomm\ |reg| Neural Processing SDK.

Please visit :ref:`AIMET Installation <ug-installation>` for installation instructions.


.. toctree::
   :caption: Getting Started
   :titlesonly:
   :includehidden:
   :maxdepth: 1

   Installation <../install/index>
   tutorials/quickstart


.. toctree::
   :caption: Examples
   :glob:
   :titlesonly:

   examples/*




.. toctree::
   :caption: Feature Descriptions
   :titlesonly:
   :maxdepth: 1

    Adaptive Rounding (AdaRound) <../user_guide/adaround>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :glob:
   :caption: AIMET PyTorch API

   quantized_modules
   quantizer
   encoding_analyzer
   api/*




| |project| is a product of |author|
| Qualcomm\ |reg| Neural Processing SDK is a product of Qualcomm Technologies, Inc. and/or its subsidiaries.

.. |reg|    unicode:: U+000AE .. REGISTERED SIGN
