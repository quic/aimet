.. _ug-examples:

.. image:: ../images/logo-quic-on@h68.png

==============
AIMET Examples
==============

AIMET Examples provide reference code (in the form of *scripts* and *Jupyter Notebooks*) to learn how to load models,
apply AIMET quantization and compression features, fine tune and save your models. It is also a quick way to become
familiar with AIMET usage and APIs.

For more details on each of the features and APIs please refer:
:ref:`Links to User Guide and API Documentation<ug-index>`

Installation Instructions
=========================

- The dataloader, evaluator, and trainer utilized in the examples is for the ImageNet dataset.
  To run the example, please download it from here: https://www.image-net.org/download.php
- Install AIMET and its dependencies using the instructions in the Installation section'
- Clone the AIMET repo as follows to any location:

  * WORKSPACE="<absolute_path_to_workspace>"

  * mkdir $WORKSPACE && cd $WORKSPACE

  *  Go to https://github.com/quic/aimet/releases and identify the release tag (`<release_tag>`) of the AIMET package that you're working with.

  * git clone https://github.com/quic/aimet.git --branch <release_tag>

  * Update the environment variable as follows: `export PYTHONPATH=$PYTHONPATH:${WORKSPACE}/aimet`

Code Layout
===========

The code for AIMET Examples shares a common structure:

Examples/

  common/

  torch/

    utils/

    quantization/

    compression/

  tensorflow/

    utils/

    quantization/

    compression/

|
Running Examples via Jupyter Notebook
=====================================

- Install the Jupyter metapackage as follows (pre-pend with "sudo -H" if appropriate):
`python3 -m pip install jupyter`

- Start the notebook server as follows (please customize the command line options if appropriate):
`jupyter notebook --ip=* --no-browser &`

- The above command will generate and display a URL in the terminal. Copy and paste it into your browser.

- Navigate to one of the following paths under the Examples directory and launch your chosen Jupyter Notebook (`.ipynb` extension):
  - `Examples/torch/quantization/`
  - `Examples/torch/compression/`
  - `Examples/tensorflow/quantization/`
  - `Examples/tensorflow/compression/`
- Follow the instructions therein to execute the code.

Running Examples via Command Line
=================================

To run an AIMET example:

python example_name.py --dataset_dir path/to/dataset/ --use_cuda

For example, to run the channel pruning example run the following:

python channel_pruning.py --dataset_dir path/to/dataset/ --use_cuda --epochs 15 --learning_rate 1e-2 --learning_rate_schedule [5, 10]

Setting the hyperparameters epochs, learning rate, and learning rate scheduler is optional. If the values are not given, the default values will be used.