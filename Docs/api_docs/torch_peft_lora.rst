.. _api-torch-peft-lora:

Top-level API
=============
.. autoclass:: aimet_torch.peft.AdapterMetaData

|

**The following API can be used to replace PEFT lora layers definition with AIMET lora layers definition**

.. automethod:: aimet_torch.peft.replace_lora_layers_with_quantizable_layers

|

**The following API can be used to save adapter weights if model adaptations were performed which change
the names/type of modules**

.. automethod:: aimet_torch.peft.save_lora_weights_after_adaptation

|

**The following API can be used to track lora meta data. To be passed to peft utilities**

.. automethod:: aimet_torch.peft.track_lora_meta_data

|

.. autoclass:: aimet_torch.peft.PeftQuantUtils
    :members:
|


User flow
===========

Example:

1) Create a PEFT model with one adapter

    >>> from peft import LoraConfig, get_peft_model
    >>> lora_config = LoraConfig(
    >>>    lora_alpha=16,
    >>>    lora_dropout=0.1,
    >>>    r=4,
    >>>    bias="none",
    >>>    target_modules=["linear"])
    >>> model = get_peft_model(model, lora_config)

2) Replace lora layer with AIMET lora layer

    >>> from aimet_torch.peft import replace_lora_layers_with_quantizable_layers
    >>> replace_lora_layers_with_quantizable_layers(model)

3) Save lora weights for adapter model

    >>> from aimet_torch.peft import save_lora_weights_after_adaptation
    >>> save_lora_weights_after_adaptation(model, tmp_dir, 'lora_weights_after_adaptation_for_adapter1')

4) Track meta data for lora layers

    >>> from aimet_torch.peft import track_lora_meta_data
    >>> meta_data = track_lora_meta_data(model, tmp_dir, 'meta_data')
    >>> ## If linear lora layers were replaced with ConvInplaceLinear then
    >>> meta_data = track_lora_meta_data(model, tmp_dir, 'meta_data', ConvInplaceLinear)

5) Create Quant utilities

    >>> from aimet_torch.peft import PeftQuantUtils
    >>> peft_utils = PeftQuantUtils(meta_data)
    >>> ## If we are using a prepared model, then load name to module dict that gets saved as a json file
    >>> peft_utils = PeftQuantUtils(meta_data, name_to_module_dict)

**Next step will be to create a QuantSim object (steps are not shown below, please refer to quantsim docs for reference)
Once Sim is created, we can use peft_utils to modify quantization attributes for lora layers in sim**

6) Disable lora adapters. To compute base model encodings without the effect of adapters

    >>> peft_utils.disable_lora_adapters(sim)

7) Compute Encodings for sim (Not shown below, refer to quantsim docs) & freeze base model encodings for params

    >>> peft_utils.freeze_base_model_param_quantizers(sim)

8) Export base model and encodings

    >>> sim.export(tmpdir, 'model', dummy_input=dummy_inputs, export_model=True, filename_prefix_encodings='base_encodings')

9) Load adapter weights

    >>> peft_utils.enable_adapter_and_load_weights(sim, 'tmpdir/lora_weights_after_adaptation_for_adapter1.safetensor', use_safetensor=True)

10) Configure lora adapter quantizers

    >>> for name, lora_module in peft_utils.get_quantized_lora_layer(sim):
    >>>     ### Change bitwidth
    >>>     lora_module.param_quantizers['weight'].bitwidth = 16
    >>>     ### Change per tensor to per channel
    >>>     lora_module.param_quantizers['weight'] = aimet.quantization.affine.QuantizeDequantize(shape=(1, 1, 1, 1), bitwidth=16, symmetric=True).to(module.weight.device)

11) Compute encodings for model & Export
    Note: while exporting the model directory should be the same for base_model export and consecutive exports

    >>> sim.export(tmpdir, 'model', dummy_input=dummy_inputs, export_model=False, filename_prefix_encodings='adapter1')
    >>> peft_utils.export_adapter_weights(sim, tmpdir, 'adapter1_weights', 'tmpdir/model.onnx')



