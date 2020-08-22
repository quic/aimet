=======================================
AIMET installation on Google Colab
=======================================

This document covers the instructions to install AIMET, and to run AIMET unit tests in Google Colab environment.
The steps are explained in more detail in the following sub-sections.

Google Code Colab Setting Update
================================

Select Hardware Accelerator as GPU in below Google Colab Menu option:

| Runtime->Change runtime->Hardware Accelerator(GPU)

Supporting Package Installation
===============================
Please run below commands to install supporting packages for AIMET:

  .. code-block::

    !pip3 uninstall protobuf

    !pip3 uninstall tensorflow

    !apt-get update

    !apt-get install python3.6

    !apt-get install python3-dev

    !apt-get install python3-pip

    !apt-get install liblapacke liblapacke-dev

    !apt-get install wget

    !pip3 install numpy==1.16.4

    !apt-get install libgtest-dev build-essential cmake

    !pip3 --no-cache-dir install opencv-python==4.1.0.25

    !pip3 --no-cache-dir install pillow==6.2.1

    !pip3 install pytorch-ignite==0.1.0

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

    !pip3 install scipy==1.1.0

    !pip3 install protobuf==3.7.1

    !pip3 install scikit-learn==0.19.1

    !pip3 install tb-nightly==1.14.0a20190517

    !pip3 install tensorboardX==1.7

    !pip3 install torch==1.1.0

    !pip3 install torchvision==0.3.0

    !pip3 install --upgrade pip

    !pip3 install tensorflow-gpu==1.15.0

    !pip3 install future==0.17.1

    !pip3 uninstall tb-nightly==1.14.0a20190517

    !pip3 install tb-nightly==1.14.0a20190517

    !pip3 install bokeh==1.2.0

    !pip3 install pandas==0.22.0

    !pip3 install holoviews==1.12.7

    !pip3 install --no-deps bokeh==1.2.0 hvplot==0.4.0

    !pip3 install jsonschema==3.1.1

    !pip3 install osqp onnx

    !ln -s /usr/local/cuda-10.0 /usr/local/cuda

    !apt-get update && apt-get install -y libjpeg8-dev

    !ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib

    !apt install zlib1g-dev

    !pip3 uninstall Pillow && pip3 install Pillow-SIMD==6.0.0.post0

    !pip3 uninstall pytest

    !pip3 install pytest

    !pip3 install setuptools==41.0.1

    !pip3 install keras==2.2.4

    %rm -rf /usr/local/bin/python

    !ln -s /usr/bin/python3 /usr/local/bin/python

Resetting of Code Colab Environment
===================================
Google Code Colab comes with comprehensive list of pre-installed packages, and for certain packages
AIMET uses specific non-latest versions of packages for better functional and/or performance support.
After some of those packages are installed, google runtime environment has to be restarted from below menu option:

| Runtime->Restart runtime

| This is to handle below warning which occurs after installation of some packages listed in above step.

| "WARNING: The following packages were previously imported in this runtime:

| [pkg_resources]

| You must restart the runtime in order to use newly installed versions.

| Certain packages would take effect"

AIMET Installation
===================

Please run below commands to fetch AIMET, and googletest from github repo, and
compile, and install AIMET.

  .. code-block::

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

    !make install

Configuring LD_LIBRARY_PATH, and PYTHONPATH configuration
=================================================================

  .. code-block::

    import sys

    sys.path.append(r'/content/aimet_code/build/staging/universal/lib/python')

    sys.path.append(r'/content/aimet_code/build/staging/universal/lib/x86_64-linux-gnu')

    sys.path.append(r'/usr/local/lib/python3.6/dist-packages')

    sys.path.append(r'/content/aimet_code/build/artifacts')

    import os

    os.environ['LD_LIBRARY_PATH']+= ":/content/aimet_code4/build/artifacts"

Running Unit Tests
===================
ctest shall run Quantization, and Compression tests for both Pytorch, and Tensorflow implementation.

| Please run below commands to run unit tests

 .. code-block::

   %cd /content/aimet_code/build/

   !ctest
