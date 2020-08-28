# AIMET Installation and Usage in Docker
This page provides instructions to build, install and use the AIMET software in docker. Please follow the instructions in the order provided, unless specified otherwise.

- [Requirements](#requirements)
- [Get the code](#get-the-code)
- [Setup the environment](#setup-the-environment)
  - [Build docker image](#build-docker-image)
  - [Start docker container](#start-docker-container)
- [Build code and install](#build-code-and-install)
- [Set package and library paths](#set-package-and-library-paths)
- [Usage examples and documentation](#usage-examples-and-documentation)
- [Docker information](#docker-information)
  - [Build docker image manually](#build-docker-image-manually)
  - [Start docker container manually](#start-docker-container-manually)
  - [Build and launch docker using script](#build-and-launch-docker-using-script)

## Requirements
The AIMET package requires the following host platform setup:

- 64-bit Intel x86-compatible processor
- Nvidia GPU card
- Linux Ubuntu: 18.04 LTS
- nvidia-docker (optional) - Installation instructions: https://github.com/NVIDIA/nvidia-docker
- bash command shell

To use the GPU accelerated training modules an Nvidia CUDA enabled GPU with a minimum Nvidia driver version of 361+ is required. Using the latest driver is always recommended, especially if using a newer GPU. Both CUDA and cuDNN (the more advanced CUDA interface) enabled GPUs are supported.

Recommended host system hardware requirements:

- Intel i7 multicore CPU w/hyperthreading
- GPU: Nvidia GeForce GTX 1080 or Tesla K80
- 16+ GB RAM
- 500GB+ SSD hard drive

While these are not minimum requirements, they are recommended for good performance when training large networks.

## Get the code
To obtain the code, first define a workspace and follow these instructions:

```bash
WORKSPACE="<absolute_path_to_workspace>"
mkdir $WORKSPACE && cd $WORKSPACE
git clone https://github.com/quic/aimet.git
```
Clone the google test repo as follows:
```
cd aimet
mkdir -p ./ThirdParty/googletest
pushd ./ThirdParty/googletest
git clone https://github.com/google/googletest.git -b release-1.8.0 googletest-release-1.8.0
popd
```
## Setup the environment
In order to build and run AIMET code, several dependencies are required (such as python, cmake, tensorflow, pytorch, etc). A docker file with all prerequisites and dependencies is available [here](Jenkins/Dockerfile). Either install the dependencies on your machine using [this Dockerfile](Jenkins/Dockerfile) as a guide, or just build and launch the docker using the instructions [here](#docker-information).

## Build code and install
Follow these instructions to build the AIMET code:

> NOTE: **If you are inside the docker, set `WORKSPACE="<absolute_path_to_workspace>"` again.**
```bash
cd $WORKSPACE 
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../aimet && make -j8 
```

After a successful build, AIMET package can be installed using the following instructions:

```bash
cd $WORKSPACE/build
make install
```

## Set package and library paths
Once the installation step is complete, AIMET package would be available at `$WORKSPACE/build/staging/lib/`, which should get reflected in some environment variables:

```bash
export PYTHONPATH=$WORKSPACE/build/staging/lib/x86_64-linux-gnu:$WORKSPACE/build/staging/lib/python:$PYTHONPATH
export LD_LIBRARY_PATH=$WORKSPACE/build/staging/lib/x86_64-linux-gnu:$WORKSPACE/build/staging/lib/python:$LD_LIBRARY_PATH
```
At this point, we are all set to use AIMET!

## Usage examples and documentation
The following steps would generate AIMET documentation including the user guide, examples and API documentation at `$WORKSPACE/build/staging/Docs`:

```bash
cd $WORKSPACE/build
make doc
```

To begin navigating the documentation, open the page `$WORKSPACE/build/staging/Docs/user_guide/index.html` on any browser.

## Docker information
Code may *optionally* be developed inside a development docker container. This section describes how to build a docker image and launch a container using the provided [Dockerfile](Jenkins/Dockerfile).

### Build docker image manually
Follow these instructions to build the docker:
```bash
WORKSPACE="<absolute_path_to_workspace>"
docker_image_name="aimet-dev-docker:<any_tag>"
docker_container_name="aimet-dev-<any_name>"
docker build -t ${docker_image_name} -f $WORKSPACE/aimet/Jenkins/Dockerfile .
```

> NOTE: Feel free to modify the `docker_image_name` and `docker_container_name` as needed.

### Start docker container manually
Ensure that a docker named `$docker_container_name` is not already running; otherwise remove the existing container and then start a new container as follows:
```bash
docker ps -a | grep ${docker_container_name} && docker kill ${docker_container_name}

docker run --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v ${HOME}:${HOME} -v ${WORKSPACE}:${WORKSPACE} \
  -v "/local/mnt/workspace":"/local/mnt/workspace" \
  --entrypoint /bin/bash -w ${WORKSPACE} --hostname aimet-dev ${docker_image_name}
```

> **NOTE**
* Feel free to modify the above `docker run` command based on the environment and filesystem on your host machine.
* If nvidia-docker 2.0 is installed, then add `--gpus all` to the `docker run` commands in order to enable GPU access inside the docker container.
* If nvidia-docker 1.0 is installed, then replace `docker run` with `nvidia-docker run` in order to enable GPU access inside the docker container. 

### Build and launch docker using script
The development docker may also be built and launched in interactive mode using the provided script as follows:
```
cd aimet
./buildntest.sh -i
```
If additional directories need to be mounted, use `-m` option with list of targeted directories separated by space **surrounded by double quotes `""`**
```
cd aimet
./buildntest.sh -i -m "sample_dir_1 sample_dir2"
```

To help construct user-specific docker commands, the dry-run option (`-n`) can be used with the above script which prints out the equivalent docker command(s):
```
cd aimet
./buildntest.sh -i -n
# OR
./buildntest.sh -i -n -m "sample_dir_1 sample_dir2"
```
