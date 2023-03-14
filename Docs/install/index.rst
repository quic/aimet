.. #==============================================================================
   #  @@-COPYRIGHT-START-@@
   #
   #  Copyright 2022 Qualcomm Technologies, Inc. All rights reserved.
   #  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
   #
   #  The party receiving this software directly from QTI (the "Recipient")
   #  may use this software as reasonably necessary solely for the purposes
   #  set forth in the agreement between the Recipient and QTI (the
   #  "Agreement"). The software may be used in source code form solely by
   #  the Recipient's employees (if any) authorized by the Agreement. Unless
   #  expressly authorized in the Agreement, the Recipient may not sublicense,
   #  assign, transfer or otherwise provide the source code to any third
   #  party. Qualcomm Technologies, Inc. retains all ownership rights in and
   #  to the software
   #
   #  This notice supersedes any other QTI notices contained within the software
   #  except copyright notices indicating different years of publication for
   #  different portions of the software. This notice does not supersede the
   #  application of any third party copyright notice to that third party's
   #  code.
   #
   #  @@-COPYRIGHT-END-@@
   #==============================================================================

.. _ug-installation:

##############################
AIMET Installation and Setup
##############################

This page provides instructions to install AIMET package on Ubuntu 18.04 LTS with Nvidia GPU. Please follow the instructions in the order provided, unless specified otherwise.

=============
Installation
=============

**NOTE:**
    #. Please pre-pend the "apt-get install" and "pip3 install" commands with "sudo -H" as appropriate.
    #. These instructions assume that pip packages will be installed in the path: /usr/local/lib/python3.8/dist-packages. If that is not the case, please modify it accordingly.


Install prerequisite packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the basic pre-requisite packages as follows:

.. code-block::

    apt-get update
    apt-get install python3.8 python3.8-dev python3-pip
    python3 -m pip install --upgrade pip
    apt-get install --assume-yes wget gnupg2

Install GPU packages
~~~~~~~~~~~~~~~~~~~~

**NOTE:**

#. Do this section ONLY for the PyTorch or Tensorflow GPU packages.
#. Visit this page https://developer.nvidia.com/cuda-11.1.1-download-archive to obtain the exact and up-to-date installation instructions for your environment.

.. code-block::

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
    dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
    apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
    apt-get update
    apt-get -y install cuda

    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    apt-get update


Install AIMET packages
~~~~~~~~~~~~~~~~~~~~~~~

Go to https://github.com/quic/aimet/releases and identify the release tag of the package you want to install.

Set the <variant_string> to ONE of the following depending on your desired variant

#. For the PyTorch GPU variant, use "torch_gpu"
#. For the PyTorch CPU variant, use "torch_cpu"
#. For the TensorFlow GPU variant, use "tf_gpu"
#. For the TensorFlow CPU variant, use "tf_cpu"

.. code-block::

    export AIMET_VARIANT=<variant_string>

Replace <variant_string> in the steps below with the appropriate tag:

.. code-block::

    export AIMET_VARIANT=<variant_string>

Set the package download URL as follows:

.. code-block::

    export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"

Set the common suffix for the package files as follows:

**NOTE:** Set wheel_file_suffix to cp38-cp38-linux_x86_64.whl OR cp36-cp36m-linux_x86_64 OR cp37-cp37m-linux_x86_64 OR py3-none-any as appropriate depending on the actual wheel filename(s) on the https://github.com/quic/aimet/releases.

.. code-block::

    export wheel_file_suffix="cp38-cp38-linux_x86_64.whl"

Install the AIMET packages in the order specified below:

**NOTE:** Python dependencies will automatically get installed.

.. code-block::

    python3 -m pip install ${download_url}/AimetCommon-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

    # Install ONE of the following depending on the variant
    python3 -m pip install ${download_url}/AimetTorch-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix} -f https://download.pytorch.org/whl/torch_stable.html
    # OR
    python3 -m pip install ${download_url}/AimetTensorflow-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

    python3 -m pip install ${download_url}/Aimet-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}


Install common debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the common debian packages as follows:

.. code-block::

    cat /usr/local/lib/python3.8/dist-packages/aimet_common/bin/reqs_deb_common.txt | xargs apt-get --assume-yes install

Install tensorflow GPU debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the TensorFlow GPU package.

.. code-block::

    cat /usr/local/lib/python3.8/dist-packages/aimet_tensorflow/bin/reqs_deb_tf_gpu.txt | xargs apt-get --assume-yes install

Install torch GPU debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the PyTorch GPU package.

.. code-block::

    cat /usr/local/lib/python3.8/dist-packages/aimet_torch/bin/reqs_deb_torch_gpu.txt | xargs apt-get --assume-yes install

Replace Pillow with Pillow-SIMD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Optional:** Replace the Pillow package with Pillow-SIMD as follows:

.. code-block::

    python3 -m pip uninstall -y pillow
    python3 -m pip install --no-cache-dir Pillow-SIMD==7.0.0.post3

Replace onnxruntime with onnxruntime-gpu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the PyTorch GPU package.

.. code-block::

    python3 -m pip uninstall -y onnxruntime
    python3 -m pip install --no-cache-dir onnxruntime-gpu==1.10.0

Post installation steps
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

    ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib

**NOTE:** Do the following step ONLY for the PyTorch or Tensorflow GPU packages.

.. code-block::

    # If you installed the CUDA 11.x drivers
    ln -s /usr/local/cuda-11.0 /usr/local/cuda
    # OR if you installed the CUDA 10.x drivers
    ln -s /usr/local/cuda-10.0 /usr/local/cuda

=================
Environment setup
=================

Set the common environment variables as follows:

.. code-block::

    source /usr/local/lib/python3.8/dist-packages/aimet_common/bin/envsetup.sh

