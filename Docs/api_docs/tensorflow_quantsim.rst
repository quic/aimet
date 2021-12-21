:orphan:

.. _api-tf-quantsim:

=====================================
AIMET TensorFlow Quantization SIM API
=====================================

Top-level API
=============

.. autoclass:: aimet_tensorflow.quantsim.QuantizationSimModel

        QuantSim simulates the behavior of a Quantized model on Hardware. supports configurations of the scheme, bitwidth for quantization, configuration of hardware, rounding mode to achieve different configurations for simulation.
        Constructor

        :Parameters:    * **model** - Model to add simulation ops to
                        * **input_shapes** - List of input shapes to the model
                        * **quant_scheme** - Quantization scheme. Supported options for Post Training Quantization are 'tf_enhanced' or 'tf' or using Quant Scheme Enum QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced. Supported options for Range Learning are QuantScheme.training_range_learning_with_tf_init or QuantScheme.training_range_learning_with_tf_enhanced_init
                        * **rounding_mode** - Rounding mode. Supported options are 'nearest' or 'stochastic'
                        * **default_output_bw** - Default bitwidth (4-31) to use for quantizing layer inputs and outputs
                        * **default_param_bw** - Default bitwidth (4-31) to use for quantizing layer parameters
                        * **in_place** - If True, then the given 'model' is modified in-place to add quant-sim nodes. Only suggested use of this option is when the user wants to avoid creating a copy of the model
                        * **config_file** - Configuration file for model quantizers

|

**Note about Quantization Schemes** : AIMET offers multiple Quantization Schemes-
    1. Post Training Quantization- The encodings of the model are computed using TF or TF-Enhanced scheme
    2. Trainable Quantization- The min max of encodings are learnt during training
        * Range Learning with TF initialization - Uses TF scheme to initialize the encodings and then during training these encodings are fine-tuned to improve accuracy of the model
        * Range Learning with TF-Enhanced initialization - Uses TF-Enhanced scheme to initialize the encodings and then during training these encodings are fine-tuned to improve accuracy of the model

**The following API can be used to Compute Encodings for Model**

.. automethod:: aimet_tensorflow.quantsim.QuantizationSimModel.compute_encodings


|

**The following API can be used to Export the Model to target**

.. automethod:: aimet_tensorflow.quantsim.QuantizationSimModel.export
   
|

Encoding format is described in the :ref:`Quantization Encoding Specification<api-quantization-encoding-spec>`

|


Code Example #1 Post Training Quantization
===========================================

**Required imports**

.. literalinclude:: ../tf_code_examples/quantization.py
    :language: python
    :lines: 39-46

**Quantize and fine-tune a trained model**

.. literalinclude:: ../tf_code_examples/quantization.py
    :language: python
    :pyobject: quantize_model

**Example Fine-tuning step**

.. literalinclude:: ../tf_code_examples/quantization.py
    :language: python
    :pyobject: training_helper

|

Code Example #2 Trainable Quantization
======================================

**Required imports**

.. literalinclude:: ../tf_code_examples/range_learning.py
   :language: python
   :start-after: # start of import statements
   :end-before: # End of import statements

**Evaluation function to be used for computing initial encodings**

.. literalinclude:: ../../../aimet/Docs/tf_code_examples/code_examples.py
   :language: python
   :pyobject: evaluate_model

**Quantize and fine-tune a trained model to learn min max ranges**

.. literalinclude:: ../tf_code_examples/range_learning.py
   :language: python
   :pyobject: quantization_aware_training_range_learning
