# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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
# =============================================================================

"""
This file demonstrates the use of compression using AIMET weight SVD
technique followed by fine tuning.
"""

import argparse
import logging
import os
from datetime import datetime
from decimal import Decimal
from typing import Tuple
from torchvision import models
import torch

# imports for AIMET
import aimet_common.defs
import aimet_torch.defs
from aimet_torch.compress import ModelCompressor

# imports for data pipelines
from Examples.common import image_net_config
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_trainer import ImageNetTrainer

logger = logging.getLogger('TorchWeightSVD')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format=formatter)


###
# This script utilize AIMET do perform weight svd compression (50% ratio) on a resnet18 pretrained model
# with the ImageNet data set. It should re-create the same performance numbers as published in the
# AIMET release for the particular scenario as described below.

# Scenario parameters:
#    - AIMET Weight SVD compression using auto mode
#    - Ignored model.conv1 (this is the first layer of the model)
#    - Target compression ratio: 0.5 (or 50%)
#    - Number of compression ration candidates: 10
#    - Input shape: [1, 3, 224, 224]
#    - Learning rate: 0.01
#    - Learning rate schedule: [5,10]
###

class ImageNetDataPipeline:
    """
    Provides APIs for model compression using AIMET weight SVD, evaluation and finetuning.
    """

    def __init__(self, _config: argparse.Namespace):
        """
        :param _config:
        """
        self._config = _config

    def evaluate(self, model: torch.nn.Module, iterations: int = None, use_cuda: bool = False) -> float:
        """
        Evaluate the specified model using the specified number of samples from the validation set.
        AIMET's compress_model() expects the function with this signature to its eval_callback
        parameter.

        :param model: The model to be evaluated.
        :param iterations: The number of batches of the dataset.
        :param use_cuda: If True then use a GPU for inference.
        :return: The accuracy for the sample with the maximum accuracy.
        """

        # your code goes here instead of the example from below

        evaluator = ImageNetEvaluator(self._config.dataset_dir, image_size=image_net_config.dataset['image_size'],
                                      batch_size=image_net_config.evaluation['batch_size'],
                                      num_workers=image_net_config.evaluation['num_workers'])

        return evaluator.evaluate(model, iterations, use_cuda)

    def finetune(self, model: torch.nn.Module):
        """
        Finetunes the model.  The implemtation provided here is just an example,
        provide your own implementation if needed.

        :param model: The model to finetune.
        """

        # Your code goes here instead of the example from below

        trainer = ImageNetTrainer(self._config.dataset_dir, image_size=image_net_config.dataset['image_size'],
                                  batch_size=image_net_config.train['batch_size'],
                                  num_workers=image_net_config.train['num_workers'])

        trainer.train(model, max_epochs=self._config.epochs, learning_rate=self._config.learning_rate,
                      learning_rate_schedule=self._config.learning_rate_schedule, use_cuda=self._config.use_cuda)

        torch.save(model, os.path.join(self._config.logdir, 'finetuned_model.pth'))


def aimet_weight_svd(model: torch.nn.Module,
                     evaluator: aimet_common.defs.EvalFunction) -> Tuple[torch.nn.Module,
                                                                         aimet_common.defs.CompressionStats]:
    """
    Compresses the model using AIMET's Weight SVD auto mode compression scheme.

    :param model: The model to compress.
    :param evaluator: Evaluator used during compression.
    :return: A tuple of compressed model and its statistics
    """

    # Please refer to the API documentation for more details.

    # Desired target compression ratio using Weight SVD
    # This value denotes the desired compression % of the original model.
    # To compress the model to 20% of original model, use 0.2. This would
    # compress the model by 80%.
    # We are compressing the model by 50% here.
    target_comp_ratio = Decimal(0.5)

    # Number of compression ratio used by the API at each layer
    # API will evaluate 0.1, 0.2, ..., 0.9, 1.0 ratio (total 10 candidates)
    # at each layer
    num_comp_ratio_candidates = 10

    # Creating Greedy selection parameters:
    greedy_params = aimet_torch.defs.GreedySelectionParameters(target_comp_ratio=target_comp_ratio,
                                                               num_comp_ratio_candidates=num_comp_ratio_candidates)

    # Selecting 'greedy' for rank select scheme:
    rank_select_scheme = aimet_common.defs.RankSelectScheme.greedy

    # Ignoring first convolutional layer of the model for compression
    modules_to_ignore = [model.conv1]

    # Creating Auto mode Parameters:
    auto_params = aimet_torch.defs.WeightSvdParameters.AutoModeParams(rank_select_scheme=rank_select_scheme,
                                                                      select_params=greedy_params,
                                                                      modules_to_ignore=modules_to_ignore)

    # Creating Weight SVD parameters with Auto Mode:
    params = aimet_torch.defs.WeightSvdParameters(aimet_torch.defs.WeightSvdParameters.Mode.auto,
                                                  auto_params)

    # Scheme is Weight SVD:
    scheme = aimet_common.defs.CompressionScheme.weight_svd

    # Cost metric is MAC, it can be MAC or Memory
    cost_metric = aimet_common.defs.CostMetric.mac

    # Input image shape
    image_shape = (1, image_net_config.dataset['image_channels'],
                   image_net_config.dataset['image_width'], image_net_config.dataset['image_height'])

    # Calling model compression using Weight SVD:
    # Here evaluator is passed which is used by the API to evaluate the
    # accuracy for various compression ratio of each layer. To speed up
    # the process, only 10 batches of data is being used inside evaluator
    # (by passing eval_iterations=10) instead of running evaluation on
    # complete dataset.
    results = ModelCompressor.compress_model(model=model,
                                             eval_callback=evaluator,
                                             eval_iterations=10,
                                             input_shape=image_shape,
                                             compress_scheme=scheme,
                                             cost_metric=cost_metric,
                                             parameters=params)

    return results


def weight_svd_example(config: argparse.Namespace):
    """
    1. Instantiate Data Pipeline for evaluation and training
    2. Load the pretrained resnet18 model
    3. Calculate floating point accuracy
    4. Compression
        4.1. Compress the model using AIMET Weight SVD
        4.2. Log the statistics
        4.3. Save the compressed model
        4.4. Calculate and log the accuracy of compressed model
    5. Finetuning
        5.1 Finetune the compressed model
        5.2 Calculate and log the accuracy of compressed-finetuned model

    :param config: This argparse.Namespace config expects following parameters:
                   dataset_dir: Path to a directory containing ImageNet dataset.
                                This folder should conatin at least 2 subfolders:
                                'train': for training dataset and 'val': for validation dataset.
                   use_cuda: A boolean var to indicate to run the test on GPU.
                   logdir: Path to a directory for logging.
                   epochs: Number of epochs (type int) for finetuning.
                   learning_rate: A float type learning rate for model finetuning
                   learning_rate_schedule: A list of epoch indices for learning rate schedule used in finetuning. Check
                                           https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#MultiStepLR
                                           for more details.
    """

    # Instantiate Data Pipeline for evaluation and training
    data_pipeline = ImageNetDataPipeline(config)

    # Load the pretrained resnet18 model
    model = models.resnet18(pretrained=True)
    if config.use_cuda:
        model.to(torch.device('cuda'))
    model.eval()

    # Calculate floating point accuracy
    accuracy = data_pipeline.evaluate(model, use_cuda=config.use_cuda)
    logger.info("Original Model top-1 accuracy = %.2f", accuracy)

    # Compress the model using AIMET Weight SVD
    logger.info("Starting Weight SVD")
    compressed_model, stats = aimet_weight_svd(model=model, evaluator=data_pipeline.evaluate)

    logger.info(stats)
    with open(os.path.join(config.logdir, 'log.txt'), "w") as outfile:
        outfile.write("%s\n\n" % (stats))

    # Calculate and log the accuracy of compressed model
    accuracy = data_pipeline.evaluate(compressed_model, use_cuda=config.use_cuda)
    logger.info("Compressed Model top-1 accuracy = %.2f", accuracy)

    logger.info("Weight SVD Complete")

    # Finetune the compressed model
    logger.info("Starting Model Finetuning")
    data_pipeline.finetune(compressed_model)

    # Calculate and log the accuracy of compressed-finetuned model
    accuracy = data_pipeline.evaluate(compressed_model, use_cuda=config.use_cuda)
    logger.info("After Weight SVD, Model top-1 accuracy = %.2f", accuracy)

    logger.info("Model Finetuning Complete")

    # Save the compressed model
    torch.save(compressed_model, os.path.join(config.logdir, 'compressed_model.pth'))


if __name__ == '__main__':
    default_logdir = os.path.join("benchmark_output", "weight_svd_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    parser = argparse.ArgumentParser(
        description='Apply Weight SVD on pretrained ResNet18 model and finetune it for ImageNet dataset')

    parser.add_argument('--dataset_dir', type=str,
                        required=True,
                        help="Path to a directory containing ImageNet dataset.\n\
                              This folder should conatin at least 2 subfolders:\n\
                              'train': for training dataset and 'val': for validation dataset")
    parser.add_argument('--use_cuda', action='store_true',
                        required=True,
                        help='Add this flag to run the test on GPU.')

    parser.add_argument('--logdir', type=str,
                        default=default_logdir,
                        help="Path to a directory for logging.\
                              Default value is 'benchmark_output/weight_svd_<Y-m-d-H-M-S>'")

    parser.add_argument('--epochs', type=int,
                        default=15,
                        help="Number of epochs for finetuning.\n\
                              Default is 15")
    parser.add_argument('--learning_rate', type=float,
                        default=1e-2,
                        help="A float type learning rate for model finetuning.\n\
                              Default is 0.01")
    parser.add_argument('--learning_rate_schedule', type=list,
                        default=[5, 10],
                        help="A list of epoch indices for learning rate schedule used in finetuning.\n\
                              Check https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#MultiStepLR for more details.\n\
                              Default is [5, 10]")

    _config = parser.parse_args()

    os.makedirs(_config.logdir, exist_ok=True)

    fileHandler = logging.FileHandler(os.path.join(_config.logdir, "test.log"))
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    if _config.use_cuda and not torch.cuda.is_available():
        logger.error('use_cuda is selected but no cuda device found.')
        raise RuntimeError("Found no CUDA Device while use_cuda is selected")

    weight_svd_example(_config)
