.. _ug-quantization-aware-training:

=================================
AIMET Quantization Aware Training
=================================

Overview
========
In cases where PTQ techniques are not sufficient for mitigating quantization error, users can use quantization-aware
training (QAT). QAT models the quantization noise during training and allows the model to find better solutions
than post-training quantization. However, the higher accuracy comes with the usual costs of neural
network training, i.e. longer training times, need for labeled data and hyperparameter search.

QAT workflow
============
The QAT workflow is largely similar to the flow for using Quantization Simulation for inference. The only difference is
that a user can take the sim.model and use it in their training pipeline in order to fine-tune model parameters while
taking quantization noise into account. The user's training pipeline will not need to change in order to train the
sim.model compared to training the original model.

A typical pipeline is as follows:

1. Create a QuantSim sim object from a pretrained model.
2. Calibrate the sim using representative data samples to come up with initial encoding values for each quantizer node.
3. Pass the sim.model into a training pipeline to fine-tune the model parameters.
4. Evaluate the sim.model using an evaluation pipeline to check whether model accuracy has improved.
5. Export the sim to generate a model with updated weights and no quantization nodes, along with the accompanying
   encodings file containing quantization scale/offset parameters for each quantization node.

Observe that as compared to QuantSim inference, step 3 is the only addition when performing QAT.

QAT modes
=========
There are two variants of QAT, referred to as QAT without Range Learning and QAT with Range Learning.

In QAT without Range Learning, encoding values for activation quantizers are found once in the beginning during the
calibration step after QuantSim has been instantiated, and are not updated again subsequently throughout training.

In QAT with Range Learning, encoding values for activation quantizers are initially set during the calibration step, but
are free to update during training, allowing a more optimal set of scale/offset quantization parameters to be found
as training takes place.

In both variants, parameter quantizer encoding values will continue to update in accordance with the parameters
themselves updating during training.

Recommendations for Quantization-Aware Training
===============================================
Here are some general guidelines that can aid in improving performance or faster convergence with Quantization-aware Training (QAT):

* Initialization:
    - Often it can be beneficial to first apply post training quantization techniques like :ref:`AutoQuant<ug-auto-quant>` before applying QAT.
      This is especially beneficial if there is large drop in INT8 performance compared to the FP32 baseline.
* Hyper-parameters:
    - Number of epochs: 15-20 epochs are generally sufficient for convergence
    - Learning rate: Comparable (or one order higher) to FP32 model's final learning rate at convergence.
      Results in AIMET are with learning of the order 1e-6.
    - Learning rate schedule: Divide learning rate by 10 every 5-10 epochs
