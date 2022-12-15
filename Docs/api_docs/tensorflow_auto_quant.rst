:orphan:

.. _api-tf-auto-quant:

==============================
AIMET TensorFlow AutoQuant API
==============================

User Guide Link
===============
To learn more about this technique, please see :ref:`AutoQuant<ug-auto-quant>`

Examples Notebook Link
======================
For an end-to-end notebook showing how to use TensorFlow AutoQuant, please see :doc:`here<../Examples/tensorflow/quantization/autoquant>`.

Top-level API
=============
.. autoclass:: aimet_tensorflow.auto_quant.AutoQuant
    :members: apply, set_adaround_params

Code Examples
=============

**Required imports**

.. literalinclude:: ../tf_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 0. Import statements
    :end-before: # End step 0

**Define constants and helper functions**

.. literalinclude:: ../tf_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 1. Define constants and helper functions
    :end-before: # End step 1

**Prepare model and dataset**

.. literalinclude:: ../tf_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 2. Prepare model and dataset
    :end-before: # End step 2

**Prepare unlabeled dataset**

.. literalinclude:: ../tf_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 3. Prepare unlabeled dataset
    :end-before: # End step 3

**Prepare eval callback**

.. literalinclude:: ../tf_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 4. Prepare eval callback
    :end-before: # End step 4

**Create AutoQuant object**

.. literalinclude:: ../tf_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 5. Create AutoQuant object
    :end-before: # End step 5

**(Optional) Set Adaround parameters**

For setting the num_batches parameter, use the following guideline.
The number of batches is used to evaluate the model while calculating the quantization encodings.
Typically we want AdaRound to use around 2000 samples.
For example, if the batch size is 32, num_batches is 64.
If the batch size you are using is different, adjust the num_batches accordingly.

.. literalinclude:: ../tf_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 6. (Optional) Set adaround params
    :end-before: # End step 6

**Run AutoQuant**

.. literalinclude:: ../tf_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 7. Run AutoQuant
    :end-before: # End step 7
