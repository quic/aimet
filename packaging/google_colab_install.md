# AIMET Build, Installation and Usage in Google Colab
This page provides instructions to build, install and use the AIMET software in Google colab environment. Please follow the instructions in the order provided, unless specified otherwise. 

- [Google colab set up](#google-colab-set-up)
- [Install AIMET packages](#install-AIMET-packages)
- [AIMET build and installation](#aimet-build-and-installation)
- [Usage](#Usage)

## Google colab set up

- Please go to Google Colab website: https://colab.research.google.com/
- Open a new notebook from main menu option: File -> New notebook
- Select Hardware Accelerator as GPU in below Google Colab Menu option:
  Runtime -> Change runtime -> Hardware Accelerator(GPU)


## Install AIMET packages
Please run below commands to install dependencies to build AIMET:

```bash
release_tag=<release_tag>
!pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/AimetCommon-${release_tag}-py3-none-any.whl
!pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/AimetTorch-${release_tag}-py3-none-any.whl
!pip3 install https://github.com/quic/aimet/releases/download/${release_tag}/AimetTensorflow-${release_tag}-py3-none-any.whl
```
Please **restart** Google runtime environment when prompted or from below menu option:

Runtime -> Restart runtime


## Configure LD_LIBRARY_PATH and PYTHONPATH

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
