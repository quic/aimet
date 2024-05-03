AIMET Docker creation
=====================

This page provides instructions to build a docker image with AIMET packages and start the development docker container.

Setup workspace
---------------

```console
WORKSPACE="<absolute_path_to_workspace>"
mkdir $WORKSPACE && cd $WORKSPACE
git clone https://github.com/quic/aimet.git
cd aimet/packaging/dockers
```

Make sure no wheel file is present in present working directory
```console
rm -rf *.whl
```

Set variant
------------

Set the *<variant_string>* to ONE of the following depending on your desired variant

* For the PyTorch 1.13 GPU variant, use **torch_gpu**
* For the PyTorch 1.13 CPU variant, use **torch_cpu**
* For the PyTorch 2.1.2 GPU variant, use **torch_gpu_pt21**
* For the PyTorch 2.1.2 CPU variant, use **torch_cpu_pt21**
* For the TensorFlow GPU variant, use **tf_gpu**
* For the TensorFlow CPU variant, use **tf_cpu**
* For the ONNX GPU variant, use **onnx_gpu**
* For the ONNX CPU variant, use **onnx_cpu**

```console
export AIMET_VARIANT=<variant_string>
```

Download AIMET packages
------------------------

Go to https://github.com/quic/aimet/releases and identify the release tag of the package you want to install.


Replace <release_tag> in the steps below with the appropriate tag:

```console
export release_tag=<release_tag>
```

Set the package download URL as follows:

```console
export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"
```

Set the common suffix for the package files as follows:

```console
export wheel_file_suffix="cp310-cp310-linux_x86_64.whl"
```

Download the AIMET packages in the order specified below:

```console
wget ${download_url}/AimetCommon-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}


# Download ONE of the following depending on the variant
wget ${download_url}/AimetTorch-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

# OR

wget ${download_url}/AimetTensorflow-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

# OR

wget ${download_url}/AimetOnnx-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}


wget ${download_url}/Aimet-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}
```

Build docker image
------------------

Follow these instructions in order to build the docker image locally. If not, skip to the next section.

```console
docker_image_name="aimet-prod-docker-${AIMET_VARIANT}:<any_tag>"
docker_container_name="aimet-prod-${AIMET_VARIANT}-<any_name>"

docker build -t ${docker_image_name} -f Dockerfile.${AIMET_VARIANT} .
```

**NOTE:** Feel free to modify the *docker_image_name* and *docker_container_name* as needed.

Start docker container
-----------------------

Ensure that a docker named *$docker_container_name* is not already running; otherwise remove the existing container and then start a new container as follows:

```console
docker ps -a | grep ${docker_container_name} && docker kill ${docker_container_name}

docker run --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \
-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
-v ${HOME}:${HOME} -v ${WORKSPACE}:${WORKSPACE} \
-v "/local/mnt/workspace":"/local/mnt/workspace" \
--entrypoint /bin/bash -w ${WORKSPACE} --hostname ${docker_container_name} ${docker_image_name}
```

**NOTE:**
* Feel free to modify the above *docker run* command based on the environment and filesystem on your host machine.
* If nvidia-docker 2.0 is installed, then add *--gpus all* to the *docker run* commands in order to enable GPU access inside the docker container.
* If nvidia-docker 1.0 is installed, then replace *docker run* with *nvidia-docker run* in order to enable GPU access inside the docker container.
* Port forwarding needs to be done in order to run the Visualization APIs from docker container. This can be achieved by running the docker container as follows:

```console

port_id="<any-port-number>"

docker run -p ${port_id}:${port_id} --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \
-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
-v ${HOME}:${HOME} -v ${WORKSPACE}:${WORKSPACE} \
-v "/local/mnt/workspace":"/local/mnt/workspace" \
--entrypoint /bin/bash -w ${WORKSPACE} --hostname ${docker_container_name} ${docker_image_name}
```

Environment setup
------------------

Set the common environment variables as follows:

```console
source /usr/local/lib/python3.10/dist-packages/aimet_common/bin/envsetup.sh
```
