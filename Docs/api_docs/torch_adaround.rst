:orphan:

.. _api-torch-adaround:

==================================
AIMET PyTorch AdaRound API
==================================

Top-level API
=============
.. autofunction:: aimet_torch.adaround.adaround_weight.Adaround.apply_adaround


        Multiple parameters can be specified by the users of the API.



        :Parameters:    * **model** - Model to apply AdaRound to
                        * **params** - AdaroundParameters, explained below the API parameters description
                        * **default_param_bw** - Default bitwidth (4-31) to use for initializing the encodings used for adaptive rounding. Default: 4
                        * **default_quant_scheme** - Default Quantization scheme used for initializing encodings used for adaptive rounding. Supported options are using Quant Enum QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced. Default: QuantScheme.post_training_tf_enhanced
                        * **config_file** - Configuration file for model quantizers

|

        :AdaroundParameters:    * **data_loader** - The Data Loader containing training data
                                * **num_batches** - The number of batches to use for adarounding
                                * **default_num_iterations** - Number of iterations to adaround each layer. Default: 10000
                                * **default_reg_param** - Regularization parameter, trading off between rounding loss vs reconstruction loss. Default: 0.01
                                * **default_beta_range** - Start and stop beta parameter for annealing of rounding loss (start_beta, end_beta). Default: (20, 2)
                                * **default_warm_start** - warm up period, during which rounding loss has zero effect. Default: 20% (0.2)

|

Enum Definition
===============
**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:

|

Code Examples
=============

**Required imports**

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :start-after: # AdaRound imports
   :end-before: # End of import statements

**Evaluation function**

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :pyobject: dummy_forward_pass

**After applying AdaRound to ResNet18, the AdaRounded model and associated encodings are returned**

.. literalinclude:: ../torch_code_examples/adaround.py
   :language: python
   :pyobject: apply_adaround_example
