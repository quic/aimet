AIMET-1.7.0

Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

SPDX-License-Identifier: BSD-3-Clause

========
Overview
========

The AIMET package (machine learning training tools) adds additional features and functionality to the
Tensorflow training framework. Features like network quantization and compression
enable neural networks to run in mobile environments and on new hardware (DSPs, NPUs, etc) that might not
otherwise be possible. Because this functionality is incorporated into the training framework the
networks can also be fine tuned enabling better performance while also benefiting from additional
compression.

Features
========
The AIMET release provides two main features:

-----------------------
Python modules
-----------------------
A set of python modules to enable network optimizaations like quantization and model compression.
These modules can be used from within any python script or can be run as standalone tools from the
command line.

----------------------
Training framework Ops
----------------------
The ops are written specifically for each training framework enabling features like quantization to directly
quantize data inline during network inference and training.

====================
Package Requirements
====================
This section describes the AIMET package requirements.

Development host
================
The AIMET package has been developed and tested on the following host platform:
- 64-bit Intel x86-compatible processor
- Linux Ubuntu 16.04 LTS
- bash command shell

------------
Dependencies
------------
- Python (>= 3.5) is required to use the python modules
<TODO: Add dependencies>

Run-time host
=============
The AIMET package supports the following run-time hosts:
- 64-bit x86 Linux (Ubuntu 16.04 LTS)
It may work on other hosts, but has not been tested to work with those.

Package tree
============
This is the top level directory hierarchy of the package:

<TODO: Add package hierarchy>

---------
bin
---------
- envsetup.sh:
  This script must be sourced in each shell when using AIMET features.
  See 'Environment setup' section.

Here is a brief description of the subdirectories and their content.

---
lib
---
This directory contains various AIMET libraries and python modules.

- python:
  This directory contains python modules used for optimizing neural networks.

- x86_64-linux-gnu:
  This directory contains the op libraries to be loaded py the python modules for
  a specific training framework

--------
examples
--------
This directory contains examples of different model architectures and training
paradigms. Each has instructions on training that model.

=================
Documentation
=================
Please refer to the following confluence page for more information on new
features and their usage.

<TODO: Add link to documentation>
=================
Using the Package
=================
Before it can be used, the package environment must be setup.

Environment setup
=================
- 1 -
To setup your environment to use the package, source the envsetup.sh script in
the 'bin' directory. This will setup the necessary paths to allow the use of the
code within the package.

> source <package-path>/bin/envsetup.sh

This will update: PATH, LD_LIBRARY_PATH, PYTHONPATH to include the relevant
paths from the package.
It also sets AIMET_ROOT to the <package-path> directory.

Note: this step must be done in each command shell when using the AIMET package.

Example code
============
Example code is located in <package-path>/examples/tensorflow/.
