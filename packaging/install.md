=============================================================================
 
  @@-COPYRIGHT-START-@@

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

  @@-COPYRIGHT-END-@@
  
=============================================================================


##### Installation instructions for AI Model Efficienty Toolkit package and dependencies

 NOTE:
 1. Please pre-pend the "apt-get install" and "pip3 install" commands with "sudo -H" as appropriate
 2. These instructions that pip packages will be installed in the path: /usr/local/lib/python3.6/dist-packages. If that is not the case, please modify it accordingly.

##### Install the following pre-requisite packages
apt-get install python3.6 python3.6-dev python3-pip

##### Install AIMET packages

Go to https://github.com/quic/aimet/releases and identify the release tag of the package you want to install
 
release_tag=<release_tag>
pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/AimetCommon-${release_tag}-py3-none-any.whl
pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/AimetTorch-${release_tag}-py3-none-any.whl
pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/AimetTensorflow-${release_tag}-py3-none-any.whl
pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/Aimet-${release_tag}-py3-none-any.whl

##### Install common debian packages from the packages_common.txt file

apt-get update
cat /usr/local/lib/python3.6/dist-packages/aimet_common/bin/packages_common.txt | xargs apt-get --assume-yes install

##### GPU preparation steps
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
apt-get --assume-yes install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
apt-get update

##### Install GPU packages from the packages_common.txt file

cat /usr/local/lib/python3.6/dist-packages/aimet_common/bin/packages_gpu.txt | xargs apt-get --assume-yes install

##### Install python dependencies from the requirements.txt file

pip3 install --upgrade pip
pip3 install -r /usr/local/lib/python3.6/dist-packages/aimet_common/bin/requirements.txt

##### Replace Pillow with Pillow-SIMD
pip3 uninstall -y pillow && pip3 install --no-cache-dir Pillow-SIMD==6.0.0.post0

##### Post installation steps
ln -s /usr/local/cuda-10.0 /usr/local/cuda
ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib

##### Setup the environment
###### Set the common environment variables as follows
source /usr/local/lib/python3.6/dist-packages/aimet_common/bin/envsetup.sh 

##### Add AIMET packages to the environment paths
export LD_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/aimet_common/x86_64-linux-gnu:/usr/local/lib/python3.6/dist-packages/aimet_common:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/lib/python3.6/dist-packages/aimet_common/x86_64-linux-gnu:/usr/local/lib/python3.6/dist-packages/aimet_common:$LD_LIBRARY_PATH
