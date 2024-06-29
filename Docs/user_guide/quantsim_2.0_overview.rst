.. _ug-aimet-torch-v2-overview:

QuantSim v2 
===============================
Welcome to QuantSim v2! 

When designing QuantSim v2, we were motivated to provide users with a clean design from the ground-up while maintaining familiar APIs. With newly designed building blocks, users have more flexibility to customize and extend the components. Developers can experience more control and transparency throughout the quantization process.

In a future release, :mod:`aimet_torch` Quantization Simulation will go through a major redesign of the basic building blocks that make up a simulated quantization model, referred to as QuantSim v2. While these changes have not yet been mainlined into :mod:`aimet_torch`, they have been made optionally available in the experimental :mod:`aimet_torch.v2` namespace. 

.. note::
  
  Please be advised that QuantSim v2 is an experimental feature whose APIs and behaviors are subject to change. 

Overview
--------------------
At a high level, QuantSim v2:

* Comprises of a different set of building blocks, quantized nn.Modules and Quantizers
* Enables dispatching to custom quantized kernels
* Allows components to be extended easily to support advanced quantization techniques 
* Moves all implementation code to Python for easier debugging and portability

Like QuantSim v1, QuantSim v2 upholds the same high level API such as ``compute_encodings()`` and ``export()``. Both QuantSim versions can perform fake quantization (quantization on floating point kernels) and support the same AIMET features like AdaRound, Sequential MSE, and QuantAnalyzer.

To learn more about the differences between QuantSim v1 and QuantSim v2 and how to migrate your code, please refer to this :ref:`guide<tutorials-migration-guide>`.


Using QuantSim v2
--------------------

All code that involves QuantSim v2 can be found in the :mod:`aimet_torch.v2` namespace. Please refer to following to navigate the namespace:

.. toctree::
   :maxdepth: 1
   :titlesonly:
   
   ../torch_docs/quantized_modules
   ../torch_docs/quantizer
   ../torch_docs/encoding_analyzer
   ../torch_docs/api/quantization/affine/index
   ../torch_docs/api/quantization/float/index

New Features
~~~~~~~~~~~~~~~~
We have now enabled blockwise quantization and low power blockwise quantization for QuantSim v2 users. When applied, these features obtain encoding parameters with a finer granularity, which produces a more optimized quantization grid. 

To learn more, please refer to the following documentation:

* :ref:`Blockwise Quantization<api-torch-blockwise-quantization>`
* :ref:`Low Power Blockwise Quantization<api-torch-blockwise-quantization>`
