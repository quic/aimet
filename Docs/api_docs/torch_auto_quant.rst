:orphan:

.. _api-torch-auto-quant:

===========================
AIMET PyTorch AutoQuant API
===========================

User Guide Link
===============
To learn more about this technique, please see :ref:`AutoQuant<ug-auto-quant>`

Examples Notebook Link
======================
For an end-to-end notebook showing how to use PyTorch AutoQuant, please see :doc:`here<../Examples/torch/quantization/autoquant>`.

Top-level API
=============
.. autoclass:: aimet_torch.auto_quant.AutoQuant
    :members:

Code Examples
=============

**Required imports**

.. literalinclude:: ../torch_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 0. Import statements
    :end-before: # End step 0

**Define constants and helper functions**

.. literalinclude:: ../torch_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 1. Define constants and helper functions
    :end-before: # End step 1

**Prepare model and dataset**

.. literalinclude:: ../torch_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 2. Prepare model and dataset
    :end-before: # End step 2

**Prepare unlabeled dataset**

.. literalinclude:: ../torch_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 3. Prepare unlabeled dataset
    :end-before: # End step 3

**Prepare eval callback**

.. literalinclude:: ../torch_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 4. Prepare eval callback
    :end-before: # End step 4

**Create AutoQuant object**

.. literalinclude:: ../torch_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 5. Create AutoQuant object
    :end-before: # End step 5

**(Optional) Set Adaround parameters**

For setting the num_batches parameter, use the following guideline.
The number of batches is used to evaluate the model while calculating the quantization encodings.
Typically we want AdaRound to use around 2000 samples.
For example, if the batch size is 32, num_batches is 64.
If the batch size you are using is different, adjust the num_batches accordingly.

.. literalinclude:: ../torch_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 6. (Optional) Set adaround params
    :end-before: # End step 6

**Run AutoQuant**

.. literalinclude:: ../torch_code_examples/auto_quant.py
    :language: python
    :start-after: # Step 7. Run AutoQuant
    :end-before: # End step 7
