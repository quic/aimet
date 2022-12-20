# =============================================================================
#
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
#
# =============================================================================

"""
This file demonstrates the use of quantization using AIMET
quantization aware training.
"""

import argparse
import logging
import os
from datetime import datetime
from functools import partial
from typing import Tuple
from torchvision import models
import torch
import torch.utils.data as torch_data

# imports for AIMET
import aimet_common
from aimet_torch import bias_correction
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantParams, QuantizationSimModel

# imports for data pipelines
from Examples.common import image_net_config
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_trainer import ImageNetTrainer

logger = logging.getLogger('TorchQAT')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format=formatter)


###
# This script utilizes AIMET to perform Quantization Aware Training on a resnet18
# pretrained model with the ImageNet data set. This is intended as a working example
# to show how AIMET APIs can be invoked. Please check the documentation/code to
# further understand how to fine tune and achieve higher accuracy

# Scenario parameters:
#    - AIMET quantization aware training using simulation model
#    - Quant Scheme: 'tf'
#    - rounding_mode: 'nearest'
#    - default_output_bw: 8, default_param_bw: 8
#	 - Encoding computation using 5 batches of data
#    - Input shape: [1, 3, 224, 224]
#    - Learning rate: 0.001
#    - Decay Steps: 5
###

class ImageNetDataPipeline:
    """
    Provides APIs for model quantization using evaluation and finetuning.
    """

    def __init__(self, _config: argparse.Namespace):
        """
        :param _config:
        """
        self._config = _config

    def evaluate(self, model: torch.nn.Module, iterations: int = None, use_cuda: bool = False) -> float:
        """
        Evaluate the specified model using the specified number of samples from the validation set.

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
        Fine tunes the model. The implementation provided here is just an example,
        provide your own implementation if needed.
        :param model: The model to fine tune
        """

        # Your code goes here instead of the example from below

        trainer = ImageNetTrainer(self._config.dataset_dir, image_size=image_net_config.dataset['image_size'],
                                  batch_size=image_net_config.train['batch_size'],
                                  num_workers=image_net_config.train['num_workers'])

        trainer.train(model, max_epochs=self._config.epochs, learning_rate=self._config.learning_rate,
                      learning_rate_schedule=self._config.learning_rate_schedule, use_cuda=self._config.use_cuda)

        torch.save(model, os.path.join(self._config.logdir, 'finetuned_model.pth'))


def apply_cross_layer_equalization(model: torch.nn.Module, input_shape: tuple):
    """
    Applying CLE on the model inplace consists of:
        Batch Norm Folding
        Cross Layer Scaling
        High Bias Fold
        Converts any ReLU6 into ReLU.

    :param model: the loaded model
    :param input_shape: the shape of the input to the model
    :return:
    """

    equalize_model(model, input_shape)


def apply_bias_correction(model: torch.nn.Module, data_loader: torch_data.DataLoader):
    """
    Applies Bias-Correction on the model.
    :param model: The model to quantize
    :param evaluator: Evaluator used during quantization
    :param dataloader: DataLoader used during quantization
    :param logdir: Log directory used for storing log files
    :return: None
    """
    # Rounding mode can be 'nearest' or 'stochastic'
    rounding_mode = 'nearest'

    # Number of samples used during quantization
    num_quant_samples = 16

    # Number of samples used for bias correction
    num_bias_correct_samples = 16

    params = QuantParams(weight_bw=8, act_bw=8, round_mode=rounding_mode, quant_scheme='tf_enhanced')

    # Perform Bias Correction
    bias_correction.correct_bias(model.to(device="cuda"), params, num_quant_samples=num_quant_samples,
                                 data_loader=data_loader, num_bias_correct_samples=num_bias_correct_samples)


def calculate_quantsim_accuracy(model: torch.nn.Module, evaluator: aimet_common.defs.EvalFunction,
                                use_cuda: bool = False, logdir: str = '') -> Tuple[torch.nn.Module, float]:
    """
    Calculates model accuracy on quantized simulator and returns quantized model with accuracy.

    :param model: the loaded model
    :param evaluator: the Eval function to use for evaluation
    :param iterations: No of batches to use in computing encodings.
                       Not used in image net dataset
    :param num_val_samples_per_class: No of samples to use from every class in
                                      computing encodings. Not used in pascal voc
                                      dataset
    :param use_cuda: the cuda device.
    :return: a tuple of quantsim and accuracy of model on this quantsim
    """

    input_shape = (1, image_net_config.dataset['image_channels'],
                   image_net_config.dataset['image_width'],
                   image_net_config.dataset['image_height'],)
    if use_cuda:
        model.to(torch.device('cuda'))
        dummy_input = torch.rand(input_shape).cuda()
    else:
        dummy_input = torch.rand(input_shape)

    # Number of batches to use for computing encodings
    # Only 5 batches are used here to speed up the process, also the
    # number of images in these 5 batches should be sufficient for
    # compute encodings
    iterations = 5

    quantsim = QuantizationSimModel(model=model, quant_scheme='tf_enhanced',
                                    dummy_input=dummy_input, rounding_mode='nearest',
                                    default_output_bw=8, default_param_bw=8, in_place=False)

    quantsim.compute_encodings(forward_pass_callback=partial(evaluator, use_cuda=use_cuda),
                               forward_pass_callback_args=iterations)

    quantsim.export(path=logdir, filename_prefix='resnet_encodings', dummy_input=dummy_input.cpu())
    accuracy = evaluator(quantsim.model, use_cuda=use_cuda)

    return quantsim, accuracy


def quantization_aware_training_example(config: argparse.Namespace):
    """
    1. Instantiates Data Pipeline for evaluation
    2. Loads the pretrained resnet18 Pytorch model
    3. Calculates Model accuracy
        3.1. Calculates floating point accuracy
        3.2. Calculates Quant Simulator accuracy
    4. Applies AIMET CLE and BC
        4.1. Applies AIMET CLE and calculates QuantSim accuracy
        4.2. Applies AIMET BC and calculates QuantSim accuracy


    :param config: This argparse.Namespace config expects following parameters:
                   tfrecord_dir: Path to a directory containing ImageNet TFRecords.
                                This folder should conatin files starting with:
                                'train*': for training records and 'validation*': for validation records
                   use_cuda: A boolean var to indicate to run the test on GPU.
                   logdir: Path to a directory for logging.
    """
    # Instantiate Data Pipeline for evaluation and training
    data_pipeline = ImageNetDataPipeline(config)

    # Load the pretrained resnet18 model
    model = models.resnet18(pretrained=True)
    if config.use_cuda:
        model.to(torch.device('cuda'))
    model = model.eval()

    # Calculate FP32 accuracy
    accuracy = data_pipeline.evaluate(model, use_cuda=config.use_cuda)
    logger.info("Original Model top-1 accuracy = %.2f", accuracy)

    logger.info("Starting Model Quantization")

    # Quantize the model using AIMET QAT (quantization aware training) and calculate accuracy on Quant Simulator
    quantsim, accuracy = calculate_quantsim_accuracy(model=model, evaluator=data_pipeline.evaluate,
                                                     use_cuda=config.use_cuda,
                                                     logdir=config.logdir)

    logger.info("Quantized Model top-1 accuracy = %.2f", accuracy)

    # For good initialization apply, apply Post Training Quantization (PTQ) methods
    # such as Cross Layer Equalization (CLE) and Bias Correction (BC) (optional)
    data_loader = ImageNetDataLoader(is_training=False, images_dir=config.dataset_dir,
                                     image_size=image_net_config.dataset['image_size']).data_loader
    apply_cross_layer_equalization(model=model, input_shape=(1, 3, 224, 224))
    apply_bias_correction(model=model, data_loader=data_loader)

    quantsim, _ = calculate_quantsim_accuracy(model=model, evaluator=data_pipeline.evaluate, use_cuda=config.use_cuda,
                                              logdir=config.logdir)

    logger.info("Post Training Quantization (PTQ) Complete")

    # Finetune the quantized model
    logger.info("Starting Model Finetuning")
    data_pipeline.finetune(quantsim.model)

    # Calculate and log the accuracy of quantized-finetuned model
    accuracy = data_pipeline.evaluate(quantsim.model, use_cuda=config.use_cuda)
    logger.info("After Quantization Aware Training, top-1 accuracy = %.2f", accuracy)

    logger.info("Quantization Aware Training Complete")

    input_shape = (1, 3, 224, 224)
    dummy_input = torch.rand(input_shape)

    # Save the quantized model
    quantsim.export(path=config.logdir, filename_prefix='QAT_resnet', dummy_input=dummy_input.cpu())


if __name__ == '__main__':
    default_logdir = os.path.join("benchmark_output", "QAT" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    parser = argparse.ArgumentParser(
        description='Apply Quantization Aware Training (QAT) on pretrained ResNet18 model and evaluate on ImageNet dataset')

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

    quantization_aware_training_example(_config)
