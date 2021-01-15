# AIMET Installation in Google Colab
This page provides instructions to install AIMET package in Google colab environment. Please follow the instructions in the order provided, unless specified otherwise. 

- [Google colab set up](#google-colab-set-up)
- [Install AIMET packages](#Install-AIMET-packages)
    - [Install Dependencies](#Install-Dependency-packages)
- [Configure](#Configure)

## Google colab set up

- Please go to Google Colab website: https://colab.research.google.com/
- Open a new notebook from main menu option: File -> New notebook
- Select Hardware Accelerator as GPU in below Google Colab Menu option:
  Runtime -> Change runtime -> Hardware Accelerator(GPU)


## Install AIMET packages
Please run below commands to install dependencies to build AIMET:

```bash
release_tag=<release_tag>
!pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/AimetCommon-${release_tag}-py3-none-any.whl -f https://download.pytorch.org/whl/torch_stable.html
!pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/AimetTorch-${release_tag}-py3-none-any.whl
!pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/AimetTensorflow-${release_tag}-py3-none-any.whl
```

### Install Dependency packages
```bash
!cat /usr/local/lib/python3.6/dist-packages/aimet_common/bin/packages_common.txt | xargs apt-get --assume-yes install
!cat /usr/local/lib/python3.6/dist-packages/aimet_common/bin/packages_gpu.txt | xargs apt-get --assume-yes install 
```

Please **restart** Google runtime environment when prompted or from below menu option:

Runtime -> Restart runtime


## Configure

```python
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages/aimet_common')
sys.path.append('/usr/local/lib/python3.6/dist-packages/aimet_torch')
sys.path.append('/usr/local/lib/python3.6/dist-packages/aimet_tensorflow')
sys.path.append('/usr/local/lib/python3.6/dist-packages/aimet_common/x86_64-linux-gnu')

import os
os.environ['LD_LIBRARY_PATH'] +=':/usr/local/lib/python3.6/dist-packages/aimet_common/x86_64-linux-gnu'
```

## Usage
You should be able to import the required packages from aimet_common, aimet_torch and aimet_tensorflow to incorporate aimet packages, for additional usage suggestion please refer to the examples from the documentation.
