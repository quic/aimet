
.. _ug-torch-apidocs:

================================================
AIMET: AI Model Efficiency Toolkit Documentation
================================================

AI Model Efficiency Toolkit (AIMET) provides tools enabling users to quantize and compress PyTorch models. Quantization
is an essential step when deploying models to edge devices with fixed-point AI accelerators.

AIMET provides both post-training and fine-tuning techniques to minimize accuracy loss incurred when quantizing
floating-point models.

.. image:: ../images/AIMET_index_no_fine_tune.png
   :width: 800px

The above picture shows a high-level view of the workflow when using AIMET. The user passes a trained floating-point
model to AIMET's APIs for quantization. AIMET returns a new PyTorch model simulating low-precision inference, which users
can fine-tune to recover lost accuracy. Users can then export the quantized model via ONNX/torchscript to an on-target
runtime like Qualcomm\ |reg| Neural Processing SDK.


Getting Started
===============


.. toctree::
   :caption: Getting Started
   :hidden:
   :titlesonly:
   :includehidden:
   :maxdepth: 1

   Installation <../install/index>
   Quickstart Guide <tutorials/quickstart_guide>


**Pip Installation:**

.. code-block:: console

    apt-get install liblapacke
    python3 -m pip install aimet-torch

For more installation options, please visit the :ref:`AIMET installation instructions<ug-installation>`.


**Basic Usage:**

.. code-block:: Python

    import aimet_torch.v2 as aimet

    # Create quantization simulation model for your model
    sim = aimet.quantsim.QuantizationSimModel(model, sample_input)

    # Calibrate quantization encodings on sample data
    with aimet.nn.compute_encodings(sim.model):
        for data, _ in data_loader:
            sim.model(data)

    # Simulate quantized inference
    sample_output = sim.model(sample_input)

    # Export model and quantization encodings
    sim.export("./out_dir", "quantized_model", sample_input)

Please view the :ref:`Quickstart Guide<ug-torch-quickstart>` for a more in-depth guide to using AIMET quantsim.

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
   :caption: AIMET PyTorch API

   quantized_modules
   quantizer
   encoding_analyzer
   api/nn.fake_quantization_mixin
   api/nn.quantization_mixin
   api/quantization/affine/index
   api/quantization/float/index




| |project| is a product of |author|
| Qualcomm\ |reg| Neural Processing SDK is a product of Qualcomm Technologies, Inc. and/or its subsidiaries.

.. |reg|    unicode:: U+000AE .. REGISTERED SIGN
