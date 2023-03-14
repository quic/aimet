# AIMET Installation and Setup
This page provides instructions to install AIMET package on ***Ubuntu 18.04 LTS with Nvidia GPU*** (see [system requirements]( docker_install.md#requirements)). Please follow the instructions in the order provided, unless specified otherwise.

- [Installation](#installation)
    - [Install prerequisite packages](#install-prerequisite-packages)
    - [Install AIMET packages](#install-aimet-packages)
    - [Install common debian packages](#install-common-debian-packages)
    - [GPU packages](#install-GPU-packages)
    - [Post installation steps](#post-installation-steps)
- [Environment Setup](#environment-setup)

## Installation

> **_NOTE:_**  
 1. Please pre-pend the "apt-get install" and "pip3 install" commands with "sudo -H" as appropriate.
 2. These instructions assume that pip packages will be installed in the path: /usr/local/lib/python3.8/dist-packages. If that is not the case, please modify it accordingly.

### Install prerequisite packages
Install the basic pre-requisite packages as follows:
```bash
apt-get update
apt-get install python3.8 python3.8-dev python3-pip
python3 -m pip install --upgrade pip
apt-get install --assume-yes wget gnupg2
```

### Install GPU packages
> _NOTE:_ Do this section **ONLY** for the PyTorch or Tensorflow *GPU* packages.

Prepare the environment for installation of GPU packages as follows:
> _NOTE:_ Please visit [this page](https://developer.nvidia.com/cuda-11.1.1-download-archive) to obtain the exact and up-to-date installation instructions for your environment.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list
echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
apt-get update
apt-get -y install cuda

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
apt-get --assume-yes install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
apt-get update
```

### Install AIMET packages
Go to https://github.com/quic/aimet/releases and identify the release tag of the package you want to install. 

Set the `<variant_string>` to ONE of the following depending on your desired variant
- For the PyTorch GPU variant, use `"torch_gpu"`
- For the PyTorch CPU variant, use `"torch_cpu"`
- For the TensorFlow GPU variant, use `"tf_gpu"`
- For the TensorFlow CPU variant, use `"tf_cpu"`
```bash
export AIMET_VARIANT=<variant_string>
```

Replace `<release_tag>` in the steps below with the appropriate tag:
```bash
export release_tag=<release_tag>
```

Set the package download URL as follows:
```bash
export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"
```

Set the common suffix for the package files as follows:
> _NOTE:_ Set wheel_file_suffix to `cp38-cp38-linux_x86_64.whl` OR `cp36-cp36m-linux_x86_64` OR `cp37-cp37m-linux_x86_64` OR `py3-none-any` as appropriate depending on the actual wheel filename(s) on the [releases page](https://github.com/quic/aimet/releases).
```bash
export wheel_file_suffix="cp38-cp38-linux_x86_64.whl"
```

Install the AIMET packages in the order specified below:
> _NOTE:_ Python dependencies will automatically get installed.
```bash
python3 -m pip install ${download_url}/AimetCommon-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

# Install ONE of the following depending on the variant
python3 -m pip install ${download_url}/AimetTorch-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix} -f https://download.pytorch.org/whl/torch_stable.html
# OR
python3 -m pip install ${download_url}/AimetTensorflow-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

python3 -m pip install ${download_url}/Aimet-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}
```

### Install common debian packages
Install the common debian packages as follows:
```bash
cat /usr/local/lib/python3.8/dist-packages/aimet_common/bin/reqs_deb_common.txt | xargs apt-get --assume-yes install
```

### Install tensorflow GPU debian packages
> _NOTE:_ Do this section **ONLY** for the TensorFlow GPU package.

Install the tensorflow GPU debian packages as follows:
```bash
cat /usr/local/lib/python3.8/dist-packages/aimet_tensorflow/bin/reqs_deb_tf_gpu.txt | xargs apt-get --assume-yes install
```

### Install torch GPU debian packages
> _NOTE:_ Do this section **ONLY** for the PyTorch GPU package.

Install the torch GPU debian packages as follows:
```bash
cat /usr/local/lib/python3.8/dist-packages/aimet_torch/bin/reqs_deb_torch_gpu.txt | xargs apt-get --assume-yes install
```

#### Replace Pillow with Pillow-SIMD
*Optional*: Replace the Pillow package with Pillow-SIMD as follows:
```bash
python3 -m pip uninstall -y pillow
python3 -m pip install --no-cache-dir Pillow-SIMD==7.0.0.post3
```

#### Replace onnxruntime with onnxruntime-gpu
> _NOTE:_ Do this section **ONLY** for the PyTorch GPU package.

Replace the onnxruntime package with onnxruntime-gpu as follows:
```bash
python3 -m pip uninstall -y onnxruntime
python3 -m pip install --no-cache-dir onnxruntime-gpu==1.10.0 
```

### Post installation steps
Perform the following post-installation steps:
```bash
ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
```

> _NOTE:_ Do the following step **ONLY** for the PyTorch or Tensorflow GPU packages.
```bash
# If you installed the CUDA 11.x drivers
ln -s /usr/local/cuda-11.0 /usr/local/cuda
# OR if you installed the CUDA 10.x drivers
ln -s /usr/local/cuda-10.0 /usr/local/cuda
```

## Environment setup
Set the common environment variables as follows:
```bash
source /usr/local/lib/python3.8/dist-packages/aimet_common/bin/envsetup.sh
```

