#!/bin/bash
#==============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2021-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

###############################################################################
# Install script to build various aimet varients into an empty Ubuntu 
# container to test replicability of documentation.
###############################################################################

set -e

# Set Package Root

export PACKAGE_ROOT="/usr/local/lib/python3.8/dist-packages"

# Install packages    
apt-get update 
apt-get install python3.8 python3.8-dev python3-pip -y
python3 -m pip install --upgrade pip
apt-get install --assume-yes wget gnupg2

update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
update-alternatives --set python3 /usr/bin/python3.8

# GPU varients
if [[ "$AIMET_VARIANT" == *"gpu"* ]]; then

    #. PyTorch 1.13 GPU variant: `CUDA Toolkit 11.6.2 <https://developer.nvidia.com/cuda-11-6-2-download-archive>`_
    if [[ "$AIMET_VARIANT" == *"torch"* || "$AIMET_VARIANT" == *"onnx"* ]]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.2-510.47.03-1_amd64.deb
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
        dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.2-510.47.03-1_amd64.deb
        echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list
        echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
        apt-get update

        apt-get install cuda-nvrtc-11-6 \
                cuda-nvrtc-dev-11-6 \
                cuda-nvprune-11-6 \
                cuda-compat-11-6 \
                libcufft-dev-11-6 \
                libcurand-dev-11-6 \
                libcusolver-dev-11-6 \
                libcusparse-dev-11-6 \
                libcublas-11-6 \
                cuda-libraries-11-6 -y --no-install-recommends \
                && rm -rf /var/lib/apt/lists/*

        wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
        dpkg -i nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
        apt-get update
    fi

    # Tensorflow GPU varaint
    if [[ "$AIMET_VARIANT" == *"tf"* ]]; then
        echo "*** Tensorflow GPU ***"
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb   
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
        dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb
        echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list
        echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
        apt-get update

        apt-get install cuda-nvrtc-11-2 \
                cuda-nvrtc-dev-11-2 \
                cuda-nvprune-11-2 \
                cuda-compat-11-2 \
                libcufft-dev-11-2 \
                libcurand-dev-11-2 \
                libcusolver-dev-11-2 \
                libcusparse-dev-11-2=11.4.1.1152-1 \
                libcublas-11-2=11.4.1.1043-1 \
                cuda-libraries-11-2=11.2.2-1 -y --no-install-recommends \
                && rm -rf /var/lib/apt/lists/*

        wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
        dpkg -i nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
        apt-get update

    fi
fi

# Aimet install
export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"
export wheel_file_suffix="cp38-cp38-linux_x86_64.whl"

python3 -m pip install ${download_url}/AimetCommon-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

# Install ONE of the following depending on the variant
if [[ "$AIMET_VARIANT" == *"torch"* ]]; then
    python3 -m pip install ${download_url}/AimetTorch-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix} -f https://download.pytorch.org/whl/torch_stable.html
fi

if [[ "$AIMET_VARIANT" == *"tf"* ]]; then
    python3 -m pip install ${download_url}/AimetTensorflow-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}
fi

if [[ "$AIMET_VARIANT" == *"onnx"* ]]; then
    python3 -m pip install ${download_url}/AimetOnnx-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}
fi

python3 -m pip install ${download_url}/Aimet-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

cat ${PACKAGE_ROOT}/aimet_common/bin/reqs_deb_common.txt | xargs apt-get --assume-yes install


if [[ "$AIMET_VARIANT" == *"torch"* ]]; then
    cat ${PACKAGE_ROOT}/aimet_onnx/bin/reqs_deb_torch_common.txt | xargs apt-get --assume-yes install
    if [[ "$AIMET_VARIANT" == *"gpu"* ]]; then
        cat ${PACKAGE_ROOT}/aimet_torch/bin/reqs_deb_torch_gpu.txt | xargs apt-get --assume-yes install
    fi
fi

if [[ "$AIMET_VARIANT" == *"onnx"* ]]; then
    cat ${PACKAGE_ROOT}/aimet_onnx/bin/reqs_deb_onnx_common.txt | xargs apt-get --assume-yes install
    if [[ "$AIMET_VARIANT" == *"gpu"* ]]; then
        cat ${PACKAGE_ROOT}/aimet_onnx/bin/reqs_deb_onnx_gpu.txt | xargs apt-get --assume-yes install
    fi
fi

if [[ "$AIMET_VARIANT" == *"tf"* ]]; then
    if [[ "$AIMET_VARIANT" == *"gpu"* ]]; then
        cat ${PACKAGE_ROOT}/aimet_tensorflow/bin/reqs_deb_tf_gpu.txt | xargs apt-get --assume-yes install
    fi
fi

python3 -m pip uninstall -y pillow
python3 -m pip install --no-cache-dir Pillow-SIMD==9.0.0.post1

if [[ "$AIMET_VARIANT" == *"torch"* ]]; then
    if [[ "$AIMET_VARIANT" == *"gpu"* ]]; then
        export ONNXRUNTIME_VER=$(python3 -c 'import onnxruntime; print(onnxruntime.__version__)')
        python3 -m pip uninstall -y onnxruntime
        python3 -m pip install --no-cache-dir onnxruntime-gpu==$ONNXRUNTIME_VER
    fi
fi

ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib

if [[ "$AIMET_VARIANT" == "tf_gpu" ]]; then
    ln -s /usr/local/cuda-11.2 /usr/local/cuda
elif [[ "$AIMET_VARIANT" == "torch_gpu" ]]; then
    ln -s /usr/local/cuda-11.6 /usr/local/cuda
fi

source ${PACKAGE_ROOT}/aimet_common/bin/envsetup.sh

