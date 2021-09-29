=======================================================
AIMET Visualization for Quantization for TensorFlow API
=======================================================

Top-level API for Visualization of Weight tensors
=================================================

.. autofunction:: aimet_tensorflow.plotting_utils.visualize_weight_ranges_single_layer

|

.. autofunction:: aimet_tensorflow.plotting_utils.visualize_relative_weight_ranges_single_layer


|

Code Examples for Visualization of Weight tensors
=================================================

**Required imports**

.. literalinclude:: ../tf_code_examples/visualization.py
    :language: python
    :start-after: # Quantization visualization imports
    :end-before: # End of import statements

**Visualizing weight ranges for layer**

.. literalinclude:: ../tf_code_examples/visualization.py
    :language: python
    :pyobject: visualizing_weight_ranges_for_single_layer

**Visualizing Relative weight ranges for layer**

.. literalinclude:: ../tf_code_examples/visualization.py
    :language: python
    :pyobject: visualizing_relative_weight_ranges_for_single_layer