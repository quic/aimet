AIMET Installation in Docker
============================

This page provides instructions to install AIMET package inside a development docker container.

Set variant
------------

Set the *<variant_string>* to ONE of the following depending on your desired variant

* For the PyTorch 1.13 GPU variant, use **torch-gpu**
* For the PyTorch 1.13 CPU variant, use **torch-cpu**
* For the PyTorch 1.9 GPU variant, use **torch-gpu-pt19**
* For the PyTorch 1.9 CPU variant, use **torch-cpu-pt19**
* For the TensorFlow GPU variant, use **tf-gpu**
* For the TensorFlow CPU variant, use **tf-cpu**
* For the ONNX GPU variant, use **onnx-gpu**
* For the ONNX CPU variant, use **onnx-cpu**

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
export wheel_file_suffix="cp38-cp38-linux_x86_64.whl"
```

Download the AIMET packages in the order specified below:

```console
wget -O $WORKSPACE/aimet/packaging/dockers ${download_url}/AimetCommon-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

# Download ONE of the following depending on the variant
wget -O $WORKSPACE/aimet/packaging/dockers ${download_url}/AimetTorch-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix} -f https://download.pytorch.org/whl/torch_stable.html
# OR
wget -O $WORKSPACE/aimet/packaging/dockers ${download_url}/AimetTensorflow-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}
# OR
wget -O $WORKSPACE/aimet/packaging/dockers ${download_url}/AimetOnnx-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

wget -O $WORKSPACE/aimet/packaging/dockers ${download_url}/Aimet-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}
```

Build docker image locally
---------------------------

Follow these instructions ONLY if you want to build the docker image locally. If not, skip to the next section.

```console
WORKSPACE="<absolute_path_to_workspace>"
docker_image_name="aimet-prod-docker:<any_tag>"
docker_container_name="aimet-prod-<any_name>"

docker build -t ${docker_image_name} -f $WORKSPACE/aimet/packaging/dockers/Dockerfile.${AIMET_VARIANT} .
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
source /usr/local/lib/python3.8/dist-packages/aimet_common/bin/envsetup.sh
```
