# AIMET Build, Installation and Usage in Google Colab
This page provides instructions to build, install and use the AIMET software in Google colab environment. Please follow the instructions in the order provided, unless specified otherwise. 

> **_NOTE:_** These instructions are *out of date* and may NOT work with the latest code.

- [Google colab set up](#google-colab-set-up)
- [Install package dependencies](#install-package-dependencies)
- [Reset Google colab envrionment](#reset-google-colab-environment)
- [AIMET build and installation](#aimet-build-and-installation)
- [Configure LD_LIBRARY_PATH and PYTHONPATH](#configure-LD_LIBRARY_PATH-and-PYTHONPATH)
- [Run unit tests](#run-unit-tests)

## Google colab set up

- Please go to Google Colab website: https://colab.research.google.com/
- Open a new notebook from main menu option: File -> New notebook
- Select Hardware Accelerator as GPU in below Google Colab Menu option:
  Runtime -> Change runtime -> Hardware Accelerator(GPU)


## Install package dependencies
Google colab comes with a list of pre-installed packages. AIMET depends on specific versions of some of these packages. You would notice in following instructions that we are downgrading some of the packages. If you see warning message mentioned below during the installation steps, please ignore it.

```bash
WARNING: The following packages were previously imported in this runtime:
[pkg_resources]
You must restart the runtime in order to use newly installed versions.
Certain packages would take effect
```

Please run below commands to install dependencies to build AIMET:

```
!pip3 uninstall protobuf

!pip3 uninstall tensorflow

!apt-get update

!apt-get install python3.8

!apt-get install python3-dev

!apt-get install python3-pip

!apt-get install liblapacke liblapacke-dev

!apt-get install wget

!pip3 install numpy==1.16.4

!apt-get install libgtest-dev build-essential

%cd /content/
!wget https://github.com/Kitware/CMake/releases/download/v3.19.3/cmake-3.19.3-Linux-x86_64.sh
!sh cmake-3.19.3-Linux-x86_64.sh  --skip-license
%rm -rf /usr/local/bin/cmake
%rm -rf /usr/local/bin/cpack
%rm -rf /usr/local/bin/ctest


!ln -s /content/bin/cmake /usr/local/bin/cmake
!ln -s /content/bin/ctest /usr/local/bin/ctest
!ln -s /content/bin/cpack /usr/local/bin/cpack

!pip3 --no-cache-dir install opencv-python==4.1.0.25

!pip3 --no-cache-dir install pillow==9.3.0

!pip3 install pytorch-ignite==0.1.2

!wget -q https://github.com/Itseez/opencv/archive/3.1.0.tar.gz -O /tmp/3.1.0.tar.gz > /dev/null

!tar -C /tmp -xvf /tmp/3.1.0.tar.gz > /dev/null

%cd /tmp/opencv-3.1.0

%mkdir release

%cd release

!cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=release -DWITH_FFMPEG=OFF -DBUILD_TESTS=OFF -DWITH_CUDA=OFF -DBUILD_PERF_TESTS=OFF -DWITH_IPP=OFF -DENABLE_PRECOMPILED_HEADERS=OFF .. > /dev/null

!make -j16 > /dev/null

!make -j16 install > /dev/null

!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

!apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

!dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

!apt-get update

!wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

!apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

!apt-get update

!apt install cuda-cublas-10-0 cuda-cufft-10-0 cuda-curand-10-0 cuda-cusolver-10-0

!apt-get update && apt install cuda-cusparse-10-0 libcudnn7=7.6.2.24-1+cuda10.0 libnccl2=2.4.8-1+cuda10.0  cuda-command-line-tools-10.0

!pip3 install scipy==1.2.1

!pip3 install protobuf==3.7.1

!pip3 install scikit-learn==0.21.0

!pip3 install tensorboardX==2.4

!pip3 install https://download.pytorch.org/whl/cu100/torch-1.4.0%2Bcu100-cp36-cp36m-linux_x86_64.whl

!pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.5.0%2Bcu100-cp36-cp36m-linux_x86_64.whl

!pip3 install --upgrade pip

!pip3 install tensorflow-gpu==1.15.0

!pip3 install future==0.17.1

!pip3 install tensorboard==1.15

!pip3 install bokeh==1.2.0

!pip3 install pandas==1.4.3

!pip3 install holoviews==1.12.7

!pip3 install --no-deps bokeh==1.2.0 hvplot==0.4.0

!pip3 install jsonschema==3.1.1

!pip3 install osqp onnx

!ln -s /usr/local/cuda-10.0 /usr/local/cuda

!apt-get update && apt-get install -y libjpeg8-dev

!ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib

!apt install zlib1g-dev

!pip3 uninstall Pillow && pip3 install Pillow-SIMD==6.2.2.post1

!pip3 uninstall pytest

!pip3 install pytest

!pip3 install setuptools==41.0.1

!pip3 install keras==2.2.4

%rm -rf /usr/local/bin/python

!ln -s /usr/bin/python3 /usr/local/bin/python
```
## Reset Google colab environment
Please restart Google runtime environment from below menu option:

Runtime -> Restart runtime

## AIMET build and installation
Please run below commands to fetch AIMET, and googletest from github repo, and compile AIMET.

```
%cd /content/

!mkdir aimet_code

%cd aimet_code

!git clone https://github.com/quic/aimet.git

%cd aimet

%mkdir -p ./ThirdParty/googletest

%pushd ./ThirdParty/googletest

!git clone https://github.com/google/googletest.git -b release-1.8.0 googletest-release-1.8.0

%popd

%cd /content/aimet_code

%mkdir build

%cd build

!cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../aimet

!make -j 8
```
### To install locally
```
!make install
```
### To create whl packages
```
!make packageaimet
```


## Configure LD_LIBRARY_PATH and PYTHONPATH

```python
import sys

sys.path.append(r'/content/aimet_code/build/staging/universal/lib/python')

sys.path.append(r'/content/aimet_code/build/staging/universal/lib/x86_64-linux-gnu')

sys.path.append(r'/usr/local/lib/python3.8/dist-packages')

sys.path.append(r'/content/aimet_code/build/artifacts')

import os

os.environ['LD_LIBRARY_PATH']+= ":/content/aimet_code/build/artifacts"
```

## Run unit tests
You can run unit tests to make sure AIMET installation was successful.
Please run below commands to run unit tests:

```
%cd /content/aimet_code/build/

!ctest
```
