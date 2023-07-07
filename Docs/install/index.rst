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

.. _ug-installation:

###################
AIMET Installation
###################

Release packages
~~~~~~~~~~~~~~~~

AIMET release packages are hosted at https://github.com/quic/aimet/releases. Each release includes multiple python packages of the following format:

.. code-block::

    <PACKAGE_PREFIX>-<VARIANT>_<VERSION>-cp38-cp38-linux_x86_64.whl

Please find more information below about each *VARIANT*.

PyTorch

#. **torch-gpu** for PyTorch 1.9 GPU package with Python 3.8 and CUDA 11.x - *Recommended for use* with PyTorch models
#. **torch-cpu** for PyTorch 1.9 CPU package with Python 3.8 - If installing on a machine without CUDA
#. **torch-gpu-pt113** for PyTorch 1.13 GPU package with Python 3.8 and CUDA 11.x
#. **torch-cpu-pt113** for PyTorch 1.13 CPU package with Python 3.8 - If installing on a machine without CUDA

TensorFlow

#. **tf-gpu** for TensorFlow 2.10 GPU package with Python 3.8 - *Recommended for use* with TensorFlow models
#. **tf-cpu** for TensorFlow 2.10 CPU package with Python 3.8 - If installing on a machine without CUDA

ONNX

#. **onnx-gpu** for ONNX 1.10.0 GPU package with Python 3.8 - *Recommended for use* with ONNX models
#. **onnx-cpu** for ONNX 1.10.0 CPU package with Python 3.8 - If installing on a machine without CUDA

System Requirements
~~~~~~~~~~~~~~~~~~~

The AIMET package requires the following host platform setup:

* 64-bit Intel x86-compatible processor
* Linux Ubuntu: 20.04 LTS
* bash command shell
* For GPU variants:
    * Nvidia GPU card (Compute capability 5.2 or later)
    * nvidia-docker - Installation instructions: https://github.com/NVIDIA/nvidia-docker

To use the GPU accelerated training modules an Nvidia CUDA enabled GPU with a minimum Nvidia driver version of 455+ is required. Using the latest driver is always recommended, especially if using a newer GPU. Both CUDA and cuDNN (the more advanced CUDA interface) enabled GPUs are supported.

Recommended host system hardware requirements:

* Intel i7 multicore CPU w/hyperthreading
* 16+ GB RAM
* 500GB+ SSD hard drive
* For GPU variants:
    * GPU: Nvidia GeForce GTX 1080 or Tesla V100

While these are not minimum requirements, they are recommended for good performance when training large networks.

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to setup and install AIMET:
    * On your host machine
    * Using our pre-built development `Docker images <https://artifacts.codelinaro.org/ui/native/codelinaro-aimet/aimet-dev>`_

Please click on the appropriate link for installation instructions:

.. toctree::
   :titlesonly:
   :maxdepth: 2

   Install in Host Machine <install_host>
   Install in Docker Container <install_docker>
