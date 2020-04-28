:orphan:

.. _api-torch-bias-correction:

=================================
AIMET PyTorch Bias Correction API
=================================

Bias Correction API
===================

.. autofunction:: aimet_torch.bias_correction.correct_bias

|

ConvBnInfoType
==============
.. autoclass:: aimet_common.bias_correction.ConvBnInfoType

|

ActivationType
==============
.. autoclass:: aimet_common.defs.ActivationType
   :members:

Quantization Params
===================

.. autofunction:: aimet_torch.quantsim.QuantParams

|

Code Examples
=============

**Required imports**

.. literalinclude:: ../../NightlyTests/torch/code_examples/post_training_techniques_examples.py
    :language: python
    :lines: 51-55

**Empirical Bias correction**

.. literalinclude:: ../../NightlyTests/torch/code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: bias_correction_empirical

**Analytical + Empirical Bias correction**

.. literalinclude:: ../../NightlyTests/torch/code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: bias_correction_analytical_and_empirical

**Bias correction Data loader format example**

.. literalinclude:: ../../NightlyTests/torch/code_examples/post_training_techniques_examples.py
    :language: python
    :pyobject: BatchIterator
