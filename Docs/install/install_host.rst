.. # =============================================================================
   #  @@-COPYRIGHT-START-@@
   #
   #  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
   #
   #  Redistribution and use in source and binary forms, with or without
   #  modification, are permitted provided that the following conditions are met:
   #
   #  1. Redistributions of source code must retain the above copyright notice,
   #     this list of conditions and the following disclaimer.
   #
   #  2. Redistributions in binary form must reproduce the above copyright notice,
   #     this list of conditions and the following disclaimer in the documentation
   #     and/or other materials provided with the distribution.
   #
   #  3. Neither the name of the copyright holder nor the names of its contributors
   #     may be used to endorse or promote products derived from this software
   #     without specific prior written permission.
   #
   #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   #  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   #  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   #  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   #  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   #  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   #  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   #  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   #  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   #  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   #  POSSIBILITY OF SUCH DAMAGE.
   #
   #  SPDX-License-Identifier: BSD-3-Clause
   #
   #  @@-COPYRIGHT-END-@@
   # =============================================================================

.. _installation-host:

##############################
AIMET Installation and Setup
##############################

This page provides instructions to install AIMET package on Ubuntu 22.04 LTS with Nvidia GPU. Please follow the instructions in the order provided, unless specified otherwise.

**NOTE:**
    #. Please pre-pend the "apt-get install" and "pip3 install" commands with "sudo -H" as appropriate.
    #. These instructions assume that pip packages will be installed in the path: /usr/local/lib/python3.10/dist-packages. If that is not the case, please modify it accordingly.


Install prerequisite packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the basic pre-requisite packages as follows:

.. code-block:: bash

    apt-get update
    apt-get install python3.10 python3.10-dev python3-pip
    python3 -m pip install --upgrade pip
    apt-get install --assume-yes wget gnupg2

If you have multiple python versions installed, set the default python version as follows:

.. code-block:: bash

    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
    update-alternatives --set python3 /usr/bin/python3.10

Install GPU packages
~~~~~~~~~~~~~~~~~~~~~

**NOTE:**

#. Do this section ONLY for the GPU variants.
#. The released AIMET GPU packages *were tested* with the following CUDA toolkit versions:
    #. PyTorch 2.1 GPU variant: `CUDA Toolkit 11.8.0 <https://developer.nvidia.com/cuda-11-8-0-download-archive>`_
    #. PyTorch 1.13 GPU variant: `CUDA Toolkit 11.7.1 <https://developer.nvidia.com/cuda-11-7-1-download-archive>`_
    #. TensorFlow GPU variant: `CUDA Toolkit 11.8.0 <https://developer.nvidia.com/cuda-11-8-0-download-archive>`_
    #. ONNX GPU variant: `CUDA Toolkit 11.7.1 <https://developer.nvidia.com/cuda-11-7-1-download-archive>`_
#. The instructions in the sub-sections below correspond to our tested versions above. Visit this page https://developer.nvidia.com/cuda-toolkit-archive to obtain the correct version of the CUDA toolkit for your environment.

Install GPU packages for PyTorch 2.1 or TensorFlow
===================================================

**NOTE:**

#. Do this section ONLY for the PyTorch 2.1 or TensorFlow GPU variant.
#. Visit this page https://developer.nvidia.com/cuda-11-8-0-download-archive to obtain the exact and up-to-date installation instructions for your environment.

.. code-block:: bash

    apt-get update && apt-get install -y gnupg2
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
    dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
    cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list
    apt-get update

Install GPU packages for PyTorch 1.13 or ONNX
=============================================

**NOTE:**

#. Do this section ONLY for the PyTorch 1.13 or ONNX GPU variants.
#. Visit this page https://developer.nvidia.com/cuda-11-7-1-download-archive to obtain the exact and up-to-date installation instructions for your environment.

.. code-block:: bash

    apt-get update && apt-get install -y gnupg2
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-ubuntu2204-11-7-local_11.7.1-515.65.01-1_amd64.deb
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
    dpkg -i cuda-repo-ubuntu2204-11-7-local_11.7.1-515.65.01-1_amd64.deb
    cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list  
    apt-get update

Install AIMET packages
~~~~~~~~~~~~~~~~~~~~~~~

From PyPI
=========

Aimet Torch GPU can install from pypi through the following method:

Go to https://pypi.org/project/aimet-torch to identify a version you wish to install

    - For PyTorch 1.13 GPU you should use aimet-torch==1.31.1
    - For Pytorch 2.1.2 GPU you should use aimet-torch >= 1.32.0

.. code-block:: bash

    sudo apt-get install liblapacke -y
    pip install aimet-torch


From Release Package
====================

Alternatively, we host .whl packages for each release at https://github.com/quic/aimet/releases. Identify the release tag
of the package you wish to install, then follow the instructions below to install AIMET from the .whl file.

Set the <variant_string> to ONE of the following depending on your desired variant

#. For the PyTorch 2.1 GPU variant, use "torch_gpu"
#. For the PyTorch 2.1 CPU variant, use "torch_cpu"
#. For the PyTorch 1.13 GPU variant, use "torch_gpu_pt113"
#. For the PyTorch 1.13 CPU variant, use "torch_cpu_pt113"
#. For the TensorFlow GPU variant, use "tf_gpu"
#. For the TensorFlow CPU variant, use "tf_cpu"
#. For the ONNX GPU variant, use "onnx_gpu"
#. For the ONNX CPU variant, use "onnx_cpu"

.. code-block:: bash

    export AIMET_VARIANT=<variant_string>

Replace <release_tag> in the steps below with the appropriate tag:

.. code-block:: bash

    export release_tag=<release_tag>

Set the package download URL as follows:

.. code-block:: bash

    export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"

Set the common suffix for the package files as follows:

**NOTE:** Set wheel_file_suffix to cp310-cp310-linux_x86_64.whl OR cp38-cp38-linux_x86_64.whl OR cp36-cp36m-linux_x86_64 OR cp37-cp37m-linux_x86_64 OR py3-none-any as appropriate depending on the actual wheel filename(s) on the https://github.com/quic/aimet/releases.

.. code-block:: bash

    export wheel_file_suffix="cp310-cp310-linux_x86_64.whl"

Install the AIMET packages in the order specified below:

**NOTE:** Python dependencies will automatically get installed.

.. code-block:: bash

    # Install ONE of the following depending on the variant
    python3 -m pip install ${download_url}/aimet_torch-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix} -f https://download.pytorch.org/whl/torch_stable.html
    # OR
    python3 -m pip install ${download_url}/aimet_tensorflow-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}
    # OR
    python3 -m pip install ${download_url}/aimet_onnx-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}


Install common debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the common debian packages as follows:

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_common/bin/reqs_deb_common.txt | xargs apt-get --assume-yes install

**NOTE:** Do the following ONLY for the PyTorch variant packages.

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_onnx/bin/reqs_deb_torch_common.txt | xargs apt-get --assume-yes install

**NOTE:** Do the following ONLY for the ONNX variant packages.

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_onnx/bin/reqs_deb_onnx_common.txt | xargs apt-get --assume-yes install

Install tensorflow GPU debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the TensorFlow GPU package.

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_tensorflow/bin/reqs_deb_tf_gpu.txt | xargs apt-get --assume-yes install

Install torch GPU debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the PyTorch GPU package.

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_torch/bin/reqs_deb_torch_gpu.txt | xargs apt-get --assume-yes install

Install ONNX GPU debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the ONNX GPU package.

.. code-block:: bash

    cat /usr/local/lib/python3.10/dist-packages/aimet_onnx/bin/reqs_deb_onnx_gpu.txt | xargs apt-get --assume-yes install

Replace Pillow with Pillow-SIMD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Optional:** Replace the Pillow package with Pillow-SIMD as follows:

.. code-block:: bash

    python3 -m pip uninstall -y pillow
    python3 -m pip install --no-cache-dir Pillow-SIMD==9.0.0.post1

Replace onnxruntime with onnxruntime-gpu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the PyTorch GPU package.

.. code-block:: bash

    export ONNXRUNTIME_VER=$(python3 -c 'import onnxruntime; print(onnxruntime.__version__)')
    python3 -m pip uninstall -y onnxruntime
    python3 -m pip install --no-cache-dir onnxruntime-gpu==$ONNXRUNTIME_VER

Post installation steps
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib

**NOTE:** Do the following step ONLY for the PyTorch or Tensorflow GPU packages.

.. code-block:: bash

    # NOTE: Please chose between the below command depending on the version of your CUDA driver toolkit
    ln -s /usr/local/cuda-11.7 /usr/local/cuda
    ln -s /usr/local/cuda-11.8 /usr/local/cuda

Environment setup
~~~~~~~~~~~~~~~~~

Set the common environment variables as follows:

.. code-block:: bash

    source /usr/local/lib/python3.10/dist-packages/aimet_common/bin/envsetup.sh

