.. _ug-examples:

.. image:: ../images/logo-quic-on@h68.png

##############
AIMET examples
##############

AIMET examples are *Jupyter Notebooks* that are intended to:

- Familiarize you with the AIMET APIs
- Demonstrate basic usage: how to apply AIMET to a model
- Teach you how to use AIMET quantization and compression

For more details on each of the features and APIs see :ref:`Links to User Guide and API Documentation<ug-index>`.

Browse the notebooks
====================

The following table provides links to browsable versions of the notebooks for several different AIMET features.

**Model Quantization Examples**

.. list-table::
   :widths: 32 12 12 12
   :header-rows: 1

   * - Feature
     - PyTorch
     - TensorFlow
     - ONNX
   * - QuantSim / Quantization-Aware Training (QAT)
     - `Link <../Examples/torch/quantization/qat.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/keras/qat.ipynb>`_
     - `Link <../Examples/onnx/quantization/quantsim.ipynb>`_  (no training)
   * - QAT with Range Learning
     - `Link <../Examples/torch/quantization/qat_range_learning.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/keras/qat_range_learning.ipynb>`_
     -
   * - Cross-Layer Equalization (CLE)
     - `Link <../Examples/torch/quantization/cle_bc.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/keras/quantsim_cle.ipynb>`_
     - `Link <../Examples/onnx/quantization/cle.ipynb>`_
   * - Adaptive Rounding (AdaRound)
     - `Link <../Examples/torch/quantization/adaround.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/keras/adaround.ipynb>`_
     - `Link <../Examples/onnx/quantization/adaround.ipynb>`_
   * - AutoQuant
     - `Link <../Examples/torch/quantization/autoquant_v2.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/keras/autoquant.ipynb>`_
     -

|
**Model Compression Examples**

.. list-table::
   :widths: 40 12
   :header-rows: 1

   * - Feature
     - PyTorch
   * - Channel Pruning
     - `Link <../Examples/torch/compression/channel_pruning.ipynb>`_
   * - Spatial SVD
     - `Link <../Examples/torch/compression/spatial_svd.ipynb>`_
   * - Spatial SVD + Channel Pruning
     - `Link <../Examples/torch/compression/spatial_svd_channel_pruning.ipynb>`_


|
Running the notebooks
=====================

To run the notebooks, follow the instructions below.

Prerequisites
-------------

#. Install the Jupyter metapackage using the following command. (If necessary, pre-pend the command with "sudo -H".)
  .. code-block:: shell

    python3 -m pip install jupyter
#. Start the notebook server as follows:
.. code-block:: shell
    
    jupyter notebook --ip=* --no-browser &
#. The command generates and displays a URL in the terminal. Copy and paste the URL into your browser.
#. Install AIMET and its dependencies using the instructions in :doc:`install`.


1. Download the example notebooks and related code
--------------------------------------------------

#. Clone the AIMET repo by running the following commands:
.. code-block:: shell
   WORKSPACE="<absolute_path_to_workspace>"
   mkdir $WORKSPACE && cd $WORKSPACE
   # Identify the release tag (<release_tag>) of the AIMET package that you're working with at: https://github.com/quic/aimet/releases.
   git clone https://github.com/quic/aimet.git --branch <release_tag>
   # Update the path environment variable:
   export PYTHONPATH=$PYTHONPATH:${WORKSPACE}/aimet
#. The dataloader, evaluator, and trainer used in the examples is for the ImageNet dataset.
  Download the ImageNet dataset from here: https://www.image-net.org/download.php

2. Run the notebooks
--------------------

#. Navigate to one of the following paths under the Examples directory and launch your chosen Jupyter Notebook (`.ipynb` extension):
  - `Examples/torch/quantization/`
  - `Examples/torch/compression/`
  - `Examples/tensorflow/quantization/keras/`
#. Follow the instructions in the notebook to execute the code.
