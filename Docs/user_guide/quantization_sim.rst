:orphan:

.. _ug-quantsim:

=============================
AIMET Quantization Simulation
=============================
Overview
========
When ML models are run on quantized hardware, the runtime tools (like Qualcomm Neural Processing SDK) will convert the floating-point parameters of the model into fixed-point parameters. This conversion generally leads to a loss in accuracy. AIMET model quantization feature helps alleviate this problem. AIMET provides functionality to change the model to simulate the effects of quantized hardware. This allows the user to then re-train the model further (called fine-tuning) to recover the loss in accuracy. As a final step, AIMET provides functionality to export the model such that it can then be run on target via a runtime.

User Flow
=========

.. image:: ../images/quant_1.png

The above explains a typical work flow a AIMET user can follow to make use of the quantization support. The steps are as follows

#. The AIMET user will create their model in one of the supported training frameworks (PyTorch or TensorFlow)
#. User trains their model
#. After the user has a working and trained model, she/he can invoke the AIMET quantization APIs to created a quantized version of the model. During this step, AIMET uses a dataloader passed in by the user to analyze the model and determine the best quantization encodings on a per-layer basis.
#. User will further train the quantized version of the model. The user can re-train the model just like in Step 2 on smaller training dataset. This step is the key step where the benefit of AIMET quantization comes into effect. The model will learn from the effects of quantization simulation.
#. User uses AIMET to save the model and the per-layer quantization encodings
#. These can be fed to a runtime like Qualcomm Neural Processing SDK to run the model on target (AIMET Importing encodings into quantized runtimes)

Quantization Noise
==================
The diagram below explains how quantization noise is introduced to a model when its input, output or parameters are quantized and dequantized.

    .. image:: ../images/quant_3.png

Since dequantizated value may not be exactly the same as quantized value, the difference between the two values is the quantization noise.

What happens under the hood
===========================
As explained above, in Step 3, AIMET analyzes the model and determines the optimal quantization encodings per-layer.

.. image:: ../images/quant_2.png

To analyze, AIMET passes some training samples through the model and using hooks, captures the tensors as they are outputted from each layer. A histogram is created to model the distribution of the floating point numbers in the output tensor for each layer.

Using the distribution of the floating point numbers in the output tensor for each layer, AIMET will use a scheme called "Enhanced TensorFlow" to determine the best encodings to convert the floating point numbers to fixed point. An encoding for a layer consists of four numbers

- Min:     Numbers below these are clamped
- Max:    Numbers above these are clamped
- Delta:   Granularity of the fixed point numbers (is a function of the bit-width selected)
- Offset:  Offset from zero

The delta and offset can be calculated using min and max and vice versa using the equations-
    :math:`delta = \frac{min - max}{{2}^{bitwidth} - 1}` and :math:`offset = \frac{-min}{delta}`

During the fine-tuning phase in Step 4, the following happens in the forward pass

.. image:: ../images/quant_4.png

Weights from a given layer are first quantized to fixed point and then de-quantized back to floating point. And the same is done with the output tensor from the layer itself. AIMET achieves this by wrapping existing layers with a custom layer that add this functionality

.. image:: ../images/quant_5.png


In the backward pass, AIMET will backprop normally. This is achieved by keeping the full-resolution floating point weights as shadow weights to be used during backprop.
