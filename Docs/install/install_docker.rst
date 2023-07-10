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

.. _installation-docker:

##############################
AIMET Installation in Docker
##############################

This page provides instructions to install AIMET package inside a development docker container.

Set variant
~~~~~~~~~~~
Set the `<variant_string>` to ONE of the following depending on your desired variant
    #. For the PyTorch 1.9 GPU variant, use `torch-gpu`
    #. For the PyTorch 1.9 CPU variant, use `torch-cpu`
    #. For the PyTorch 1.13 GPU variant, use `torch-gpu-pt113`
    #. For the PyTorch 1.13 CPU variant, use `torch-cpu-pt113`
    #. For the TensorFlow GPU variant, use `tf-gpu`
    #. For the TensorFlow CPU variant, use `tf-cpu`
    #. For the ONNX GPU variant, use `onnx-gpu`
    #. For the ONNX CPU variant, use `onnx-cpu`

.. code-block::
    export AIMET_VARIANT=<variant_string>


Use prebuilt docker image
~~~~~~~~~~~~~~~~~~~~~~~~~
Follow these instructions to use one of the pre-built docker images:

.. code-block::
    WORKSPACE="<absolute_path_to_workspace>"
    docker_image_name="artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:latest.${AIMET_VARIANT}"
    docker_container_name="aimet-dev-<any_name>"

**NOTE:** Feel free to modify the `docker_container_name` as needed.

Build docker image locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Follow these instructions ONLY if you want to build the docker image locally. If not, skip to the next section.

.. code-block::
    WORKSPACE="<absolute_path_to_workspace>"
    docker_image_name="aimet-dev-docker:<any_tag>"
    docker_container_name="aimet-dev-<any_name>"
    docker build -t ${docker_image_name} -f $WORKSPACE/aimet/Jenkins/Dockerfile.${AIMET_VARIANT} .

**NOTE:** Feel free to modify the `docker_image_name` and `docker_container_name` as needed.

Start docker container 
~~~~~~~~~~~~~~~~~~~~~~~
Ensure that a docker named `$docker_container_name` is not already running; otherwise remove the existing container and then start a new container as follows:

.. code-block::
    docker ps -a | grep ${docker_container_name} && docker kill ${docker_container_name}

    docker run --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -v ${HOME}:${HOME} -v ${WORKSPACE}:${WORKSPACE} \
    -v "/local/mnt/workspace":"/local/mnt/workspace" \
    --entrypoint /bin/bash -w ${WORKSPACE} --hostname ${docker_container_name} ${docker_image_name}


**NOTE:**
    #. Feel free to modify the above `docker run` command based on the environment and filesystem on your host machine.
    #. If nvidia-docker 2.0 is installed, then add `--gpus all` to the `docker run` commands in order to enable GPU access inside the docker container.
    #. If nvidia-docker 1.0 is installed, then replace `docker run` with `nvidia-docker run` in order to enable GPU access inside the docker container. 
    #. Port forwarding needs to be done in order to run the Visualization APIs from docker container. This can be achieved by running the docker container as follows:

.. code-block::
    port_id="<any-port-number>"

    docker run -p ${port_id}:${port_id} --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -v ${HOME}:${HOME} -v ${WORKSPACE}:${WORKSPACE} \
    -v "/local/mnt/workspace":"/local/mnt/workspace" \
    --entrypoint /bin/bash -w ${WORKSPACE} --hostname ${docker_container_name} ${docker_image_name} 

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

Replace <release_tag> in the steps below with the appropriate tag:

.. code-block::

    export release_tag=<release_tag>

Set the package download URL as follows:

.. code-block::

    export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"

Set the common suffix for the package files as follows:

.. code-block::

    export wheel_file_suffix="cp38-cp38-linux_x86_64.whl"

Install the AIMET packages in the order specified below:

**NOTE:**
    #. Please pre-pend the "apt-get install" and "pip3 install" commands with "sudo -H" as appropriate.
    #. These instructions assume that pip packages will be installed in the path: /usr/local/lib/python3.8/dist-packages. If that is not the case, please modify it accordingly.
    #. Python dependencies will automatically get installed.

.. code-block::

    python3 -m pip install ${download_url}/AimetCommon-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

    # Install ONE of the following depending on the variant
    python3 -m pip install ${download_url}/AimetTorch-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix} -f https://download.pytorch.org/whl/torch_stable.html
    # OR
    python3 -m pip install ${download_url}/AimetTensorflow-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

    python3 -m pip install ${download_url}/Aimet-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

Environment setup
~~~~~~~~~~~~~~~~~

Set the common environment variables as follows:

.. code-block::

    source /usr/local/lib/python3.8/dist-packages/aimet_common/bin/envsetup.sh

