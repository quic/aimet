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

This page provides instructions to install AIMET package on Ubuntu 20.04 LTS with Nvidia GPU. Please follow the instructions in the order provided, unless specified otherwise.

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

If you have multiple python versions installed, set the default python version as follows:

.. code-block::

    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
    update-alternatives --set python3 /usr/bin/python3.8

Install GPU packages for PyTorch or ONNX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:**

#. Do this section ONLY for the PyTorch or ONNX GPU variants.
#. Visit this page https://developer.nvidia.com/cuda-11.1.1-download-archive to obtain the exact and up-to-date installation instructions for your environment.

.. code-block::

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
    dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
    apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
    apt-get update
    apt-get -y install cuda

    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
    dpkg -i nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
    apt-get update

Install GPU packages for TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:**

#. Do this section ONLY for the TensorFlow GPU variant.
#. Visit this page https://developer.nvidia.com/cuda-11.2.2-download-archive to obtain the exact and up-to-date installation instructions for your environment.

.. code-block::

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb
    dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb
    apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
    apt-get update
    apt-get -y install cuda

    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
    dpkg -i nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
    apt-get update

Install AIMET packages
~~~~~~~~~~~~~~~~~~~~~~~

Go to https://github.com/quic/aimet/releases and identify the release tag of the package you want to install.

Set the <variant_string> to ONE of the following depending on your desired variant

#. For the PyTorch 1.9 GPU variant, use "torch_gpu"
#. For the PyTorch 1.9 CPU variant, use "torch_cpu"
#. For the PyTorch 1.13 GPU variant, use "torch_gpu_pt113"
#. For the PyTorch 1.13 CPU variant, use "torch_cpu_pt113"
#. For the TensorFlow GPU variant, use "tf_gpu"
#. For the TensorFlow CPU variant, use "tf_cpu"
#. For the ONNX GPU variant, use "onnx_gpu"
#. For the ONNX CPU variant, use "onnx_cpu"

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
    # OR
    python3 -m pip install ${download_url}/AimetOnnx-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

    python3 -m pip install ${download_url}/Aimet-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}


Install common debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the common debian packages as follows:

.. code-block::

    cat /usr/local/lib/python3.8/dist-packages/aimet_common/bin/reqs_deb_common.txt | xargs apt-get --assume-yes install

**NOTE:** Do the following ONLY for the PyTorch variant packages.

.. code-block::

    cat /usr/local/lib/python3.8/dist-packages/aimet_onnx/bin/reqs_deb_torch_common.txt | xargs apt-get --assume-yes install

**NOTE:** Do the following ONLY for the ONNX variant packages.

.. code-block::

    cat /usr/local/lib/python3.8/dist-packages/aimet_onnx/bin/reqs_deb_onnx_common.txt | xargs apt-get --assume-yes install

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

Install ONNX GPU debian packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the ONNX GPU package.

.. code-block::

    cat /usr/local/lib/python3.8/dist-packages/aimet_onnx/bin/reqs_deb_onnx_gpu.txt | xargs apt-get --assume-yes install

Replace Pillow with Pillow-SIMD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Optional:** Replace the Pillow package with Pillow-SIMD as follows:

.. code-block::

    python3 -m pip uninstall -y pillow
    python3 -m pip install --no-cache-dir Pillow-SIMD==9.0.0.post1

Replace onnxruntime with onnxruntime-gpu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NOTE:** Do this ONLY for the PyTorch GPU package.

.. code-block::

    export ONNXRUNTIME_VER=$(python3 -c 'import onnxruntime; print(onnxruntime.__version__)')
    python3 -m pip uninstall -y onnxruntime
    python3 -m pip install --no-cache-dir onnxruntime-gpu==$ONNXRUNTIME_VER

Post installation steps
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

    ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib

**NOTE:** Do the following step ONLY for the PyTorch or Tensorflow GPU packages.

.. code-block::

    # If you installed a CUDA driver other than 11.1, please modify the command accordingly
    ln -s /usr/local/cuda-11.1 /usr/local/cuda

Environment setup
~~~~~~~~~~~~~~~~~

Set the common environment variables as follows:

.. code-block::

    source /usr/local/lib/python3.8/dist-packages/aimet_common/bin/envsetup.sh

