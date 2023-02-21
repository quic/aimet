:orphan:

.. _api-tensorflow-quant-analyzer:

================================
AIMET Tensorflow Quant Analyzer API
================================

AIMET Tensorflow Quant Analyzer analyzes the Tensorflow model and points out sensitive ops to quantization in the model.
It checks model sensitivity to weight and activation quantization, performs per op sensitivity and MSE analysis.
It also exports per op encodings min and max ranges and statistics histogram for every op.

Top-level API
=============
.. autoclass:: aimet_tensorflow.quant_analyzer.QuantAnalyzer
    :members:

Code Example
=============

**Required imports**

.. literalinclude:: ../tf_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 0. Import statements
    :end-before:  # End step 0

**Prepare forward pass callback**

.. literalinclude:: ../tf_code_examples/quant_analyzer_code_example.py
    :language: python
    :pyobject: forward_pass_callback

**Prepare eval callback**

.. literalinclude:: ../tf_code_examples/quant_analyzer_code_example.py
    :language: python
    :pyobject: eval_callback

**Create session**

.. literalinclude:: ../tf_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 3. Create TF session
    :end-before:  # End step 3

**Create QuantAnalyzer object**

.. literalinclude:: ../tf_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 4. Create QuantAnalyzer object
    :end-before: # End step 4

**Create unlabeled dataset and define num_batches**

.. literalinclude:: ../tf_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 5 (optional step)
    :end-before: # End step 5

**Run QuantAnalyzer**

.. literalinclude:: ../tf_code_examples/quant_analyzer_code_example.py
    :language: python
    :start-after: # Step 6. Run QuantAnalyzer
    :end-before: # End step 6
