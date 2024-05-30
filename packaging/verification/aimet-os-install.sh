#!/bin/bash
#==============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2021-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

export PACKAGE_ROOT="/usr/local/lib/python3.10/dist-packages"

# Install packages    
apt-get update 
apt-get install python3.10 python3.10-dev python3-pip -y
python3 -m pip install --upgrade pip
apt-get install --assume-yes wget gnupg2

update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
update-alternatives --set python3 /usr/bin/python3.10

# GPU varients
if [[ "$AIMET_VARIANT" == *"gpu"* ]]; then

    #. PyTorch 1.13 GPU variant: `CUDA Toolkit 11.7.1 <https://developer.nvidia.com/cuda-11-7-1-download-archive>`_
    if [[ ( "$AIMET_VARIANT" == *"torch"* || "$AIMET_VARIANT" == *"onnx"* ) && "$AIMET_VARIANT" != *"pt21"* ]]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
        mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
        wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-ubuntu2204-11-7-local_11.7.1-515.65.01-1_amd64.deb
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
        dpkg -i cuda-repo-ubuntu2204-11-7-local_11.7.1-515.65.01-1_amd64.deb
        cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
        echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list
        apt-get update

        apt-get install cuda-nvrtc-11-7 \
                cuda-nvrtc-dev-11-7 \
                cuda-nvprune-11-7 \
                cuda-compat-11-7 \
                libcufft-dev-11-7 \
                libcurand-dev-11-7 \
                libcusolver-dev-11-7 \
                libcusparse-dev-11-7 \
                libcublas-11-7 -y --no-install-recommends \
                && rm -rf /var/lib/apt/lists/*
    fi

    # Tensorflow GPU varaint
    if [[ "$AIMET_VARIANT" == *"tf"* || "$AIMET_VARIANT" == *"pt21"* ]]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
        mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
        wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
        dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
        cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
        echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list
        apt-get update

        apt-get install cuda-nvrtc-11-8 \
                cuda-nvrtc-dev-11-8 \
                cuda-nvprune-11-8 \
                cuda-compat-11-8 \
                libcufft-dev-11-8 \
                libcurand-dev-11-8 \
                libcusolver-dev-11-8 \
                libcusparse-dev-11-8 \
                libcublas-11-8 --no-install-recommends \
                && rm -rf /var/lib/apt/lists/*
    fi
fi

# Aimet install
export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"
export wheel_file_suffix="cp310-cp310-linux_x86_64.whl"

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

cat ${PACKAGE_ROOT}/aimet_common/bin/reqs_deb_common.txt | xargs apt-get --assume-yes --allow-change-held-packages install


if [[ "$AIMET_VARIANT" == *"torch"* ]]; then
    cat ${PACKAGE_ROOT}/aimet_onnx/bin/reqs_deb_torch_common.txt | xargs apt-get --assume-yes --allow-change-held-packages install
    if [[ "$AIMET_VARIANT" == *"gpu"* ]]; then
        cat ${PACKAGE_ROOT}/aimet_torch/bin/reqs_deb_torch_gpu.txt | xargs apt-get --assume-yes --allow-change-held-packages install
    fi
fi

if [[ "$AIMET_VARIANT" == *"onnx"* ]]; then
    cat ${PACKAGE_ROOT}/aimet_onnx/bin/reqs_deb_onnx_common.txt | xargs apt-get --assume-yes --allow-change-held-packages install
    if [[ "$AIMET_VARIANT" == *"gpu"* ]]; then
        cat ${PACKAGE_ROOT}/aimet_onnx/bin/reqs_deb_onnx_gpu.txt | xargs apt-get --assume-yes --allow-change-held-packages install
    fi
fi

if [[ "$AIMET_VARIANT" == *"tf"* ]]; then
    if [[ "$AIMET_VARIANT" == *"gpu"* ]]; then
        cat ${PACKAGE_ROOT}/aimet_tensorflow/bin/reqs_deb_tf_gpu.txt | xargs apt-get --assume-yes --allow-change-held-packages install
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

if [[ -f !/usr/lib/libjpeg.so ]]; then
	ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
fi


if [[ "$AIMET_VARIANT" == "tf_gpu" || "$AIMET_VARIANT" == "torch-gpu-pt21" ]]; then
  if [[ -f !/usr/local/cuda ]]; then 
      ln -s /usr/local/cuda-11.8 /usr/local/cuda
  fi
elif [[ "$AIMET_VARIANT" == "torch_gpu" || "$AIMET_VARIANT" == "onnx-gpu"  ]]; then
    if [[ -f !/usr/local/cuda ]]; then
        ln -s /usr/local/cuda-11.7 /usr/local/cuda
    fi
fi

apt-get install liblapacke-dev

./${PACKAGE_ROOT}/aimet_common/bin/envsetup.sh

