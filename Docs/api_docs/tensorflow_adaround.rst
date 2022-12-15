:orphan:

.. _api-tf-adaround:

==================================
AIMET TensorFlow AdaRound API
==================================

User Guide Link
===============
To learn more about this technique, please see :ref:`AdaRound<ug-adaround>`

Examples Notebook Link
======================
For an end-to-end notebook showing how to use TensorFlow AdaRound, please see :doc:`here<../Examples/tensorflow/quantization/adaround>`.

Top-level API
=============
.. autofunction:: aimet_tensorflow.adaround.adaround_weight.Adaround.apply_adaround

Adaround Parameters
===================
.. autoclass:: aimet_tensorflow.adaround.adaround_weight.AdaroundParameters
    :members:

Enum Definition
===============
**Quant Scheme Enum**

.. autoclass:: aimet_common.defs.QuantScheme
    :members:

Code Examples
=============

**Required imports**

.. literalinclude:: ../tf_code_examples/adaround.py
   :language: python
   :start-after: # AdaRound imports
   :end-before: # End of import statements

**Evaluation function**

.. literalinclude:: ../tf_code_examples/adaround.py
   :language: python
   :pyobject: dummy_forward_pass

**After applying AdaRound to the model, the AdaRounded session and associated encodings are returned**

.. literalinclude:: ../tf_code_examples/adaround.py
   :language: python
   :pyobject: apply_adaround_example
