.. _ug-quant-analyzer:


===================
AIMET QuantAnalyzer
===================

Overview
========

The QuantAnalyzer feature, analyzes the model for quantization and points out sensitive parts/hotspots in the model.
It performs following:
1. Sensitivity analysis to weight and/or activation quantization:
    - which helps user to determine which AIMET quantization technique(s) will be more beneficial for model.
      For example, if model is more sensitive to activation quantization, then in such scenario, PTQ techniques
      like Adaptive Rounding or Cross Layer Equalization might not be very helpful.
2. Per layer analysis:
    - Per layer analysis by enabling and disabling quant wrappers (layer's weights and activations) and export
      the visualization in .html plot with model performance (y-axis) vs layers (x-axis).
      This helps user to further pinpoint culprit layer(s)/hotspots in the model.
3. Per layer encodings min-max ranges:
    - Exports encodings min and max ranges for all layers' weights and activations. If Per Channel Quantization (PCQ)
      is enabled, then it will also exports encodings min and max ranges for all the channels of layer's weight.
4. Per layer statistics histogram:
    - Exports histogram that represents a PDF of collected statistics by a quantizer for every quant wrappers.
5. Per layer MSE loss:
    - Computes MSE loss between FP32 model and QuantizationSimModel for every layers and exports plot with
      MSE loss (y-axis) and layers (x-axis). This step requires to feed some data through both the models.
      Approximately 256 samples/images are sufficient.
