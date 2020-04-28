
========================================
AIMET Visualization for Quantization API
========================================

Top-level API Quantization
==========================

.. autofunction:: aimet_torch.visualize_model.visualize_relative_weight_ranges_to_identify_problematic_layers

|

.. autofunction:: aimet_torch.visualize_model.visualize_weight_ranges


|

.. autofunction:: aimet_torch.visualize_model.visualize_changes_after_optimization

|

Code Examples
=============

**Required imports**

.. literalinclude:: ../../NightlyTests/torch/code_examples/visualization.py
    :language: python
    :lines: 40, 42-43, 47, 52-58

**Comparing Model After Optimization**

.. literalinclude:: ../../NightlyTests/torch/code_examples/visualization.py
    :language: python
    :pyobject: visualize_changes_in_model_after_and_before_cle

**Visualizing weight ranges in Model**

.. literalinclude:: ../../NightlyTests/torch/code_examples/visualization.py
    :language: python
    :pyobject: visualize_weight_ranges_model

**Visualizing Relative weight ranges in Model**

.. literalinclude:: ../../NightlyTests/torch/code_examples/visualization.py
    :language: python
    :pyobject: visualize_relative_weight_ranges_model



