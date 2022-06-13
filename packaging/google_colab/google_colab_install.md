# AIMET Installation in Google Colab
This page provides instructions to install AIMET package in Google colab environment. Please follow the instructions in the order provided, unless specified otherwise. 

> **_NOTE:_** These instructions are *out of date* and may NOT work with the latest releases.
 
- [Google colab set up](#google-colab-set-up)
- [Install Dependencies](#Install-Dependency-packages)
- [Install AIMET packages](#Install-AIMET-packages)
- [Configure](#Configure)
- [Usage](#Usage)
- [Validation](#Validation)

## Google colab set up

- Please go to Google Colab website: https://colab.research.google.com/
- Open a new notebook from main menu option: File -> New notebook
- Select Hardware Accelerator as GPU in below Google Colab Menu option:
  Runtime -> Change runtime -> Hardware Accelerator(GPU)

### Install Dependency packages
```bash
import os
os.environ['SRC_URL'] = 'https://raw.githubusercontent.com/quic/aimet/develop/packaging/'
!curl ${SRC_URL}packages_common.txt | xargs apt-get --assume-yes install
!curl ${SRC_URL}packages_gpu.txt | xargs apt-get --assume-yes --allow-change-held-packages install 
!wget ${SRC_URL}requirements.txt
!pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Install AIMET packages
Go to https://github.com/quic/aimet/releases and identify the release tag of the packages you want to install. Replace `<RELEASE_TAG>` in the steps below with the appropriate tag (ex. "1.14.0"). Then run the below commands to install the AIMET packages:

```bash
import os
os.environ['release_tag']=<RELEASE_TAG>
!pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/AimetCommon-${release_tag}-py3-none-any.whl 
!pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/AimetTorch-${release_tag}-py3-none-any.whl
!pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/AimetTensorflow-${release_tag}-py3-none-any.whl
```

Please **restart** Google runtime environment when prompted or from below menu option:

Runtime -> Restart runtime


## Configure

```python
import sys
sys.path.append('/usr/local/lib/python3.8/dist-packages/aimet_common')

import os
os.environ['LD_LIBRARY_PATH'] +=':/usr/local/lib/python3.8/dist-packages/aimet_common'
```

## Usage
You should be able to import the required packages from aimet_common, aimet_torch and aimet_tensorflow to incorporate aimet packages, for additional usage suggestion please refer to the examples from the documentation.


## Validation
The install could be validated by executing a snippet of code that instantiates a AIMET quantization simulator
```
import torch
from torchvision import models
from aimet_torch.quantsim import QuantizationSimModel
m = models.resnet18()
sim = QuantizationSimModel(m, (1, 3, 224, 224))
```
**Sample output**
```
print(sim)
-------------------------
Quantized Model Report
-------------------------
Layer: conv1
    Input: bw=8, encoding-present=False
    Params:
        weight: bw=8, encoding-present=False
    Output: bw=8, encoding-present=False
Layer: bn1
    Input: Unquantized
    Params:
        weight: bw=8, encoding-present=False
        bias: Unquantized
    Output: bw=8, encoding-present=False
Layer: relu
    Input: Unquantized
    Params:
    Output: bw=8, encoding-present=False
Layer: maxpool
    Input: Unquantized
    Params:
    Output: bw=8, encoding-present=False
Layer: layer1.0.conv1
    Input: Unquantized
    Params:
        weight: bw=8, encoding-present=False
    Output: bw=8, encoding-present=False
Layer: layer1.0.bn1
...
```