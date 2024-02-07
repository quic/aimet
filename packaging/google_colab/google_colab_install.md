# AIMET Installation in Google Colab
This page provides instructions to install AIMET package in Google colab environment. Please follow the instructions in the order provided, unless specified otherwise. 

> **_NOTE:_** These instructions are *out of date* and may NOT work with the latest releases.
 
- [Google colab set up](#google-colab-set-up)
- [Google colab environment](#google-colab-environment)
- [Install AIMET packages](#Install-AIMET-packages)

- [Usage](#Usage)
- [Validation](#Validation)

## Google colab set up

- Please go to Google Colab website: https://colab.research.google.com/
- Open a new notebook from main menu option: File -> New notebook
- Optionally you can use the provided colab notebook located at aimet/packaging/google_colab/Install_AIMET_GoogleColab.ipynb and open it in colab. For convience here is a link to the notebook in the aimet repo: [Aimet Google Colab](https://github.com/quic/aimet/tree/develop/packaging/google_colab/Install_AIMET_GoogleColab.ipynb)
- Select Hardware Accelerator as GPU in below Google Colab Menu option:
  Runtime -> Change runtime -> Hardware Accelerator(GPU)

## Google colab environment

Please note there are limitations in Google Colab and this is only documented for short quick, aimet tests and not intended to be leveraged for any production purpose. 
To start we need to align the python version of colab with our requirement of python 3.8
```python
python_ver = !python --version
print(python_ver)
if('Python 3.8' not in python_ver[0]):
  print(python_ver)
  !wget -O mini.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh
  !chmod +x mini.sh
  !bash ./mini.sh -b -f -p /usr/local
  !conda install -q -y jupyter
  !conda install -q -y google-colab -c conda-forge
  !python -m ipykernel install --user --name="py38"
```

## Prepare for AIMET installation
Go to https://github.com/quic/aimet/releases and identify the release tag of the packages you want to install. Replace `<RELEASE_TAG>` in the steps below with the appropriate tag (ex. "1.14.0"). Then run the aimet-os installation script.

```python
%env PYTHONPATH="/env/python:/usr/local/lib/python3.8/site-packages"
```
## Export variant and release tags variables
Here we will set the aimet variant and release_tag we are using.
Variants can be found in the Aimet documentation: https://quic.github.io/aimet-pages/releases/latest/install/index.html

Release tags can be found here: https://github.com/quic/aimet/releases

```python
import os
os.environ["AIMET_VARIANT"] = "torch_gpu"
os.environ["release_tag"] = "1.28.0"
```
We will also remove the existing nvidia-ml and nvidia-machine-learning apt lists

```python
!rm -rf /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/nvidia-machine-learning.list
```
```python
!apt-get update
```

Clone aimet to the workspace
```python
!git clone https://github.com/quic/aimet.git
```

Run the aimet installation script

```python 
!/bin/bash aimet/develop/packaging/verification/aimet-os-install.sh
```



## Usage
You should be able to import the required packages from aimet_common, aimet_torch and aimet_tensorflow to incorporate aimet packages, for additional usage suggestion please refer to the examples from the documentation.

```python

%%bash
sh /usr/local/lib/python3.8/site-packages/aimet_common/bin/envsetup.sh
python -c "

from aimet_common.defs import QuantScheme
import aimet_common.defs as aimet_common_defs
import aimet_common.libpymo as libpymo

import torch
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.compress import ModelCompressor

from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.quantsim import QuantizationSimModel

"
```

## Validation
The install could be validated by executing a snippet of code that instantiates a AIMET quantization simulator
```
!pip install pytest 
!pytest test_quantizer_GPU.py
```
