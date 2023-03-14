.. _ug-examples:

.. image:: ../images/logo-quic-on@h68.png

==============
AIMET Examples
==============

AIMET Examples provide reference code (in the form of *Jupyter Notebooks*) to learn how to
apply AIMET quantization and compression features. It is also a quick way to become
familiar with AIMET usage and APIs.

For more details on each of the features and APIs please refer:
:ref:`Links to User Guide and API Documentation<ug-index>`

Browse the notebooks
====================

The following table has links to browsable versions of the notebooks for different features.

**Model Quantization Examples**

.. list-table::
   :widths: 40 12 12 12 12
   :header-rows: 1

   * - Features
     - PyTorch
     - TensorFlow
     - Keras
     - ONNX
   * - Quantsim / Quantization-Aware Training (QAT)
     - `Link <../Examples/torch/quantization/qat.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/qat.ipynb>`_
     -
     - `Link <../Examples/onnx/quantization/quantsim.ipynb>`_  (no training)
   * - QAT with Range Learning
     - `Link <../Examples/torch/quantization/qat_range_learning.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/qat_range_learning.ipynb>`_
     -
     -
   * - Cross-Layer Equalization (CLE)
     - `Link <../Examples/torch/quantization/cle_bc.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/cle_bc.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/keras/quantsim_cle.ipynb>`_
     -
   * - Adaptive Rounding (AdaRound)
     - `Link <../Examples/torch/quantization/adaround.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/adaround.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/keras/adaround.ipynb>`_
     -
   * - AutoQuant
     - `Link <../Examples/torch/quantization/autoquant.ipynb>`_
     - `Link <../Examples/tensorflow/quantization/autoquant.ipynb>`_
     -
     -

|
**Model Compression Examples**

.. list-table::
   :widths: 40 12 12
   :header-rows: 1

   * - Features
     - PyTorch
     - TensorFlow
   * - Channel Pruning
     - `Link <../Examples/torch/compression/channel_pruning.ipynb>`_
     - `Link <../Examples/tensorflow/compression/channel_pruning.ipynb>`_
   * - Spatial SVD
     - `Link <../Examples/torch/compression/spatial_svd.ipynb>`_
     - `Link <../Examples/tensorflow/compression/spatial_svd.ipynb>`_
   * - Spatial SVD + Channel Pruning
     - `Link <../Examples/torch/compression/spatial_svd_channel_pruning.ipynb>`_
     - `Link <../Examples/tensorflow/compression/spatial_svd_channel_pruning.ipynb>`_


|
Running the notebooks
=====================

Install Jupyter
---------------
- Install the Jupyter metapackage as follows (pre-pend with "sudo -H" if appropriate):
`python3 -m pip install jupyter`

- Start the notebook server as follows (please customize the command line options if appropriate):
`jupyter notebook --ip=* --no-browser &`

- The above command will generate and display a URL in the terminal. Copy and paste it into your browser.


Download the Example notebooks and related code
------------------------------------------------
- Clone the AIMET repo as follows to any location:
  * WORKSPACE="<absolute_path_to_workspace>"
  * mkdir $WORKSPACE && cd $WORKSPACE
  *  Go to https://github.com/quic/aimet/releases and identify the release tag (`<release_tag>`) of the AIMET package that you're working with.
  * git clone https://github.com/quic/aimet.git --branch <release_tag>
  * Update the environment variable as follows: `export PYTHONPATH=$PYTHONPATH:${WORKSPACE}/aimet`

- The dataloader, evaluator, and trainer utilized in the examples is for the ImageNet dataset.
  To run the example, please download the ImageNet dataset from here: https://www.image-net.org/download.php

- Install AIMET and its dependencies using the instructions in the Installation section'

Run the notebooks
-----------------

- Navigate to one of the following paths under the Examples directory and launch your chosen Jupyter Notebook (`.ipynb` extension):
  - `Examples/torch/quantization/`
  - `Examples/torch/compression/`
  - `Examples/tensorflow/quantization/`
  - `Examples/tensorflow/compression/`
- Follow the instructions therein to execute the code.

