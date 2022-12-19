===============================
AIMET PyTorch Quantization APIs
===============================

In order to make full use of AIMET Quantization features, there are several guidelines users are encouraged to follow
when defining PyTorch models. AIMET provides APIs which can automate some of the model definition changes and checks
whether AIMET Quantization features can be applied on PyTorch model.

Users should first invoke Model Preparer API before using any of the AIMET Quantization features.
   - :ref:`Model Guidelines<api-torch-model-guidelines>`: Guidelines for defining PyTorch models
   - :ref:`Model Preparer API<api-torch-model-preparer>`: Allows user to automate model definition changes
   - :ref:`Model Validator API<api-torch-model-validator>`: Allows user to check whether AIMET Quantization feature can be applied on a PyTorch model

AIMET Quantization for PyTorch Models provides the following functionality.
   - :ref:`Quant Analyzer API<api-torch-quant-analyzer>`: Analyzes the model and points out sensitive layers to quantization
   - :ref:`Quantization Simulation API<api-torch-quantsim>`: Allows ability to simulate inference and training on quantized hardware
   - :ref:`Adaptive Rounding API<api-torch-adaround>`: Post-training quantization technique to optimize rounding of weight tensors
   - :ref:`Cross-Layer Equalization API<api-torch-cle>`: Post-training quantization technique to equalize layer parameters
   - :ref:`Bias Correction API<api-torch-bias-correction>`: Post-training quantization technique to correct shift in layer outputs due to quantization noise
   - :ref:`AutoQuant API<api-torch-auto-quant>`: Unified API that integrates the post-training quantization techniques provided by AIMET

If a user wants to use Multi-GPU with CLE or QAT, they can refer to:
    - :ref:`Multi-GPU guidelines<api-torch-multi-gpu>`: Guidelines to use PyTorch DataParallel API with AIMET features
