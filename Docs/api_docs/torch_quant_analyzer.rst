:orphan:

.. _api-torch-quant-analyzer:

================================
AIMET PyTorch Quant Analyzer API
================================

AIMET PyTorch Quant Analyzer analyzes the PyTorch model and points out sensitive layers to quantization in the model.
It checks model sensitivity to weight and activation quantization, performs per layer sensitivity and MSE analysis.
It also exports per layer encodings min and max ranges and statistics histogram for every layer.

Top-level API
=============
.. autoclass:: aimet_torch.quant_analyzer.QuantAnalyzer
    :members:

Code Examples
=============

**Required imports**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 0. Import statements
    :end-before: # End step 0

**Prepare eval callback**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 1. Prepare eval callback
    :end-before: # End step 1

**Prepare model**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 2. Prepare model
    :end-before: # End step 2

**Prepare calibration dataloader**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 3. Prepare calibration dataloader
    :end-before: # End step 3

**Wrap eval callback function**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 4. Wrap eval callback function
    :end-before: # End step 4

**Create QuantAnalyzer object**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 5. Create QuantAnalyzer object
    :end-before: # End step 5

**Run QuantAnalyzer**

.. literalinclude:: ../torch_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 6. Run QuantAnalyzer
    :end-before: # End step 6
