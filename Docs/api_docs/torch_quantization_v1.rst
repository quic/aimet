===========
aimet_torch
===========

In order to make full use of AIMET Quantization features, there are several guidelines users are encouraged to follow
when defining PyTorch models. AIMET provides APIs which can automate some of the model definition changes and checks
whether AIMET Quantization features can be applied on PyTorch model.

.. toctree::
   :titlesonly:
   :hidden:

    Model Guidelines <torch_model_guidelines>
    Architecture Checker API<torch_architecture_checker>
    Model Preparer API<torch_model_preparer>
    Model Validator API<torch_model_validator>
    Quant Analyzer API<torch_quant_analyzer>
    Quantization Simulation API<torch_quantsim>
    Adaptive Rounding API<torch_adaround>
    Cross-Layer Equalization API<torch_cross_layer_equalization>
    Bias Correction API<torch_bias_correction>
    AutoQuant API<torch_auto_quant>
    BN Re-estimation APIs<torch_batchnorm_re_estimation>
    Multi-GPU guidelines<torch_multi_gpu>
    PEFT LoRA APIs<torch_peft_lora>

Users should first invoke Model Preparer API before using any of the AIMET Quantization features.
   - :ref:`Model Guidelines<api-torch-model-guidelines>`: Guidelines for defining PyTorch models
   - :ref:`Architecture Checker API<api-torch-architecture_checker>`: Allows user to check for performance concern with the model.
   - :ref:`Model Preparer API<api-torch-model-preparer>`: Allows user to automate model definition changes
   - :ref:`Model Validator API<api-torch-model-validator>`: Allows user to check whether AIMET Quantization feature can be applied on a PyTorch model

AIMET Quantization for PyTorch Models provides the following functionality.
   - :ref:`Quant Analyzer API<api-torch-quant-analyzer>`: Analyzes the model and points out sensitive layers to quantization
   - :ref:`Quantization Simulation API<api-torch-quantsim>`: Allows ability to simulate inference and training on quantized hardware
   - :ref:`Adaptive Rounding API<api-torch-adaround>`: Post-training quantization technique to optimize rounding of weight tensors
   - :ref:`Cross-Layer Equalization API<api-torch-cle>`: Post-training quantization technique to equalize layer parameters
   - :ref:`Bias Correction API<api-torch-bias-correction>`: Post-training quantization technique to correct shift in layer outputs due to quantization noise
   - :ref:`AutoQuant API<api-torch-auto-quant>`: Unified API that integrates the post-training quantization techniques provided by AIMET
   - :ref:`BN Re-estimation APIs<api-torch-bn-reestimation>`: APIs that Re-estimate BN layers' statistics and fold the BN layers
   - :ref:`PEFT LoRA APIs<api-torch-peft-lora>`: APIs to integrate PEFT LoRA with AIMET Quantization flow

If a user wants to use Multi-GPU with CLE or QAT, they can refer to:
    - :ref:`Multi-GPU guidelines<api-torch-multi-gpu>`: Guidelines to use PyTorch DataParallel API with AIMET features
