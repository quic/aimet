![Qualcomm Innovation Center, Inc.](../Docs/images/logo-quic-on@h68.png)

# AIMET Examples
AIMET Examples provide reference code to learn how to load models, apply AIMET quantization and compression features, fine tune and save your models. It is also a quick way to become familiar with AIMET usage and APIs. For more details on each of the features and APIs please reference the _[user guide](https://quic.github.io/aimet-pages/releases/1.16.2/user_guide/index.html#api-documentation-and-usage-examples)_.

## Table of Contents
- [Installation](#installation-instructions)
- [Code Layout](#code-layout)
- [Supported Examples](#supported-examples)
- [Running AIMET Examples](#running-aimet-examples-on-the-command-line)

## Installation Instructions
Please install AIMET and its dependencies using the instructions in [this section](../README.md#installation-instructions).
The dataloader, evaluator, and trainer utilized in the examples is for the ImageNet dataset. To run the examples end-to-end, please download it here: https://www.image-net.org/download.php 


## Code Layout:
The code for AIMET Examples shares a common structure:
```
Examples/
common/
  image_net_config.py
  utils.py
torch/
  utils/
    image_net_data_loader.py
    image_net_evaluator.py
    image_net_trainer.py
  quantization/
    adaround.py
    cle_bc.py
    quantization_aware_training.py
  compression/
    channel_pruning.py
    spatial_svd.py
    spatial_svd_cp.py
    weight_svd.py
tensorflow/
  utils/
  quantization/
  compression/

```


## Overview
This section describes how to apply the various quantization and compression techniques.

### Post Training Quantization Examples 
-   _[Cross Layer Equalization and Bias Correction](https://github.com/quic/aimet/blob/develop/Examples/torch/quantization/cle_bc.py)_: 
      - Cross Layer Equalization performs BatchNorm Folding, Cross Layer Scaling, and High Bias Fold
      - Bias Correction corrects shift in layer outputs introduced due to quantization
-   _[Adaround(Adaptive Rounding)](https://github.com/quic/aimet/blob/develop/Examples/torch/quantization/adaround.py)_: 
      -  AdaRound is a weight-rounding mechanism for post-training quantization (PTQ) that adapts to the data and the task loss. AdaRound is computationally fast, needs only a small number of unlabeled examples (which may even be for a different dataset in the same domain), optimizes a local loss, does not require end-to-end finetuning, requires very little or no hyperparameter tuning for different networks and tasks, and can be applied to convolutional or fully connected layers without any modification. It complementary to most other post-training quantization techniques such as CLE, batch-normalization folding and high bias absorption. 
### Pre-Training Quantization Examples
-   _[Quantization-aware Training](https://github.com/quic/aimet/blob/develop/Examples/torch/quantization/quantization_aware_training.py)_:  
      -  Simulate on-target quantized inference. Use quantization simulation to train the model further to improve accuracy. 

### Compression Examples

-   _[Spatial SVD](https://github.com/quic/aimet/blob/develop/Examples/torch/compression/spatial_svd.py)_: 
    - Spatial SVD is a tensor decomposition technique which decomposes one large layer (in terms of mac or memory) into two smaller layers.
    - Given a conv layer, with a given kernel size, Spatial SVD decomposes it into two kernels of smaller rank, which represents the degree of compression achieved.
-   _[Channel Pruning](https://github.com/quic/aimet/blob/develop/Examples/torch/compression/channel_pruning.py)_: 
    -  Removes redundant input channels from a layer and reconstructs layer weights. Once one or more input channels for a layer are removed, then it means corresponding output channels of a upstream layer could also be removed to get further compression gains. Note that the presence of skip-connections or residuals sometimes prevents upstream layers from getting output-pruned.
-   _[Weight SVD](https://github.com/quic/aimet/blob/develop/Examples/torch/compression/weight_svd.py)_: 
    - Weight SVD is a tensor decomposition technique which decomposes one large layer (in terms of mac or memory) into two smaller layers. Given a neural network layer, with kernel (m,n,h,w) where m is the input channels, n the output channels, and h, w giving the height and width of the kernel itself, Weight SVD will decompose the kernel into one of size (m,k,1,1) and another of size (k,n,h,w), where k is called the rank. The smaller the value of k the larger the degree of compression achieved.

## Running AIMET Examples on the Command Line
Step by step to run AIMET examples. Here is how you would run an AIMET example:
```
python example_name.py --dataset_dir path/to/dataset/ --use_cuda 
``` 
For example, to run the channel pruning example run the following: 
```
python channel_pruning.py --dataset_dir path/to/dataset/ --use_cuda --epochs 15 --learning_rate 1e-2 --learning_rate_schedule [5, 10]
``` 
Setting the hyperparameters epochs, learning rate, and learning rate scheduler is optional. If the values are not given, the default values will be used.



