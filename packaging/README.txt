#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
#==============================================================================

========
Overview
========
AI Model Efficiency Toolkit (AIMET) is a library that provides advanced model
quantization and model compression techniques for trained neural network
models. It provides features that have been proven to improve run-time
performance of deep learning neural network models with lower compute and
memory requirements and minimal impact to task accuracy.

Features
========
AIMET supports the following features

- Model Quantization
  - Quantization simulation: Simulates on-target quantized inference.
    Specifically simulates Qualcomm SnapDragon DSP accelerators.
  - Quantization-aware training: Fine-tune models to improve on-target
    quantized accuracy
  - Data Free quantization: Post-training technique to improve quantized
    accuracy by equalizing model weights (Cross-Layer Equalization) and
    correcting shifts in layer outputs due to quantization (Bias Correction)

- Model Compression
  - Spatial SVD: Tensor decomposition technique to split a large layer
    into two smaller ones
  - Channel Pruning: Removes redundant input channels of convolutional
    layers and modifies the model graph accordingly
  - Compression-ratio Selection: Automatically selects per-layer compression
    ratios

Please see the user guide (...\Docs\user_guide\index.html) for more details.

====================
Package Requirements
====================
This section describes the AIMET package requirements.

Development host
================
The AIMET package has been developed and tested on the following host platform:
- 64-bit Intel x86-compatible processor
- Nvidia GPU card
- Linux Ubuntu 18.04 LTS
- nvidia-docker (optional) - Installation instructions: https://github.com/NVIDIA/nvidia-docker
- bash command shell

------------
Dependencies
------------
- See the INSTALL.txt file for details.

Run-time host
=============
The AIMET package supports the following run-time hosts:
- 64-bit x86 Linux (Ubuntu 18.04 LTS)
It may work on other hosts, but has not been tested to work with those.

Package tree
============
This is the top level directory hierarchy of the package:
├── Docs --> User guide and API documentation
└── pip --> AIMET pip packages

=================
Documentation
=================
Please refer to the user-guide at ...\Docs\user_guide\index.html. The user
guide also has links to the API documentation that show API details and quick
examples on how to invoke them.

=================
Using the Package
=================
Before it can be used, the package environment must be setup.

Environment setup
=================
To setup your environment and being using AIMET, complete the installation steps
from INSTALL.txt and follow configuration steps in envsetup.sh file


