.. _ug-aimet-torch-v2-overview:

QuantSim 2.0 
===============================
Welcome to QuantSim 2.0! 

When designing QuantSim 2.0, we were motivated to provide users with a clean design from the ground-up while maintaining familiar APIs. With newly designed building blocks, users have more flexibility to customize and extend the components. Developers can experience more control and transparency throughout the quantization process.

.. note::
    
    Please be advised that QuantSim 2.0 is an experimental feature whose APIs and behaviors are subject to change. 

Overview
--------------------
At a high level, QuantSim 2.0:

* Comprises of a different set of building blocks, quantized nn.Modules and Quantizers
* Enables true quantization (quantization on integer kernels)
* Allows components to be extended easily to support advanced quantization techniques 
* Moves all implementation code to Python for easier debugging and portability

Like QuantSim 1.0, QuantSim 2.0 upholds the same high level API such as ``compute_encodings()`` and ``export()``. Both QuantSim versions can perform fake quantization (quantization on floating point kernels) and support the same AIMET features like AutoQuant, AdaRound, and Sequential MSE. 

To learn more about the differences between QuantSim 1.0 and QuantSim 2.0 and how to migrate your code, please refer to this guide. 


Using QuantSim 2.0
--------------------

All code that involves QuantSim 2.0 can be found in the :mod:`aimet_torch.v2` namespace. Please refer to following to navigate the namespace:

.. list-table:: 
   :widths: 25 25
   :header-rows: 1

   * - Feature
     - Location
   * - QuantizationSimModel
     - aimet_torch.v2.quantsim.quantsim
   * - Quantized nn.Modules
     - aimet_torch.v2.nn  
   * - Quantizer
     - aimet_torch.v2.quantization
   * - Encoding Analyzer
     - aimet_torch.v2.quantization
   * - AdaRound
     - aimet_torch.v2.adaround.AdaRound
   * - AutoQuant
     - aimet_torch.v2.auto_quant.AutoQuant
   * - Sequential MSE 
     - aimet_torch.v2.seq_mse.apply_seq_mse
   * - QuantAnalyzer
     - aimet_torch.v2.quant_analyzer.QuantAnalyzer


To learn more about the new design and APIs, please refer to the following documentation:

* :ref:`Quantized nn.Modules<api-torch-quantized-modules>`
* :ref:`Quantizer<api-torch-quantizers>`
* :ref:`Encoding Analyzer<api-torch-encoding-analyzer>`


New Features
~~~~~~~~~~~~~~~~
We have now enabled blockwise quantization and low power blockwise quantization for QuantSim 2.0 users. When applied, these features obtain encoding parameters with a finer granularity, which produces a more optimized quantization grid. 

To learn more, please refer to the following documentation:

* Blockwise Quantization 
* Low Power Blockwise Quantization
