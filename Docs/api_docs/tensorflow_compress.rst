
================================
AIMET TensorFlow Compression API
================================

Introduction
============
AIMET supports the following model compression techniques for tensorflow models
   - Spatial SVD
   - Channel Pruning
   - Weight SVD

To learn more about these model compression techniques, please see :ref:`Model Compression User Guide<ug-model-compression>`

For the Spatial SVD and Channel Pruning compression techniques, there are two modes in which you can invoke the AIMET API
   - Auto Mode: In **Auto** mode, AIMET will determine the optimal way to compress each layer of
                the model given an overall target compression ratio. Greedy Compression Ratio Selection Algorithm is used to pick appropriate compression ratios for each layer.

   - Manual Mode: In **Manual** mode, the user can pass in the desired compression-ratio per layer
                  to AIMET. AIMET will apply the specified compression technique for each of the
                  layers to achieve the desired compression-ratio per layer. It is recommended that
                  the user start with Auto mode, and then tweak per-layer compression-ratios using
                  Manual mode if desired.

For Weight SVD, we use Tar-Based Rank selection. Auto and Manual modes are supported for Weight SVD as well.

|

Top-level API for Compression
=============================

.. autoclass:: aimet_tensorflow.compress.ModelCompressor

|

.. automethod:: aimet_tensorflow.compress.ModelCompressor.compress_model

|

Greedy Selection Parameters
===========================

.. autoclass:: aimet_common.defs.GreedySelectionParameters
   :members:
   :noindex:

|


Spatial SVD Configuration
=========================

.. autoclass:: aimet_tensorflow.defs.SpatialSvdParameters
   :members:

|

Channel Pruning Configuration
=============================

.. autoclass:: aimet_tensorflow.defs.ChannelPruningParameters
   :members:

|

Configuration Definitions
=========================

.. autoclass:: aimet_common.defs.CostMetric
   :members:

|

.. autoclass:: aimet_common.defs.CompressionScheme
   :members:

|

.. autoclass:: aimet_tensorflow.defs.ModuleCompRatioPair
   :members:

|


Code Examples
=============

**Required imports**

.. literalinclude:: ../tf_code_examples/code_examples.py
    :language: python
    :lines: 41-51

**Evaluation function**

.. literalinclude:: ../tf_code_examples/code_examples.py
    :language: python
    :pyobject: evaluate_model

**Compressing using Spatial SVD in auto mode with multiplicity = 8 for rank rounding**

.. literalinclude:: ../tf_code_examples/code_examples.py
    :language: python
    :pyobject: spatial_svd_auto_mode

**Compressing using Spatial SVD in manual mode**

.. literalinclude:: ../tf_code_examples/code_examples.py
    :language: python
    :pyobject: spatial_svd_manual_mode

**Compressing using Channel Pruning in auto mode**

.. literalinclude:: ../tf_code_examples/code_examples.py
   :language: python
   :pyobject: channel_pruning_auto_mode

**Compressing using Channel Pruning in manual mode**

.. literalinclude:: ../tf_code_examples/code_examples.py
   :language: python
   :pyobject: channel_pruning_manual_mode




Weight SVD Top-level API
========================

.. autoclass:: aimet_tensorflow.svd.Svd

|

.. automethod:: aimet_tensorflow.svd.Svd.compress_net

|

Code Examples for Weight SVD
============================

**Required imports**

.. literalinclude:: ../tf_code_examples/weight_svd.py
   :language: python
   :lines: 38-41

**Compressing using Weight SVD in auto mode**

.. literalinclude:: ../tf_code_examples/weight_svd.py
   :language: python
   :pyobject: weight_svd_auto_mode


**Compressing using Weight SVD in manual mode**

.. literalinclude:: ../tf_code_examples/weight_svd.py
   :language: python
   :pyobject: weight_svd_manual_mode
