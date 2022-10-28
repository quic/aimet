# !/usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
Creates Evaluator for Image-Net dataset
"""
import os
import logging

from tensorflow.keras.utils import Progbar
import tensorflow as tf
import numpy as np

from Examples.common import image_net_config
from Examples.tensorflow.utils.keras.image_net_dataset import ImageNetDataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger = logging.getLogger("Eval")


class ImageNetEvaluator:
    """
    For validation of a trained model using the ImageNet dataset.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=dangerous-default-value
    def __init__(self, dataset_dir: str, image_size: int = 224, batch_size: int = 128, model_type: str = "resnet50"):
        """
        Constructor
        :param dataset_dir: The directory path to the data
        :param image_size: Required size for images. Images will be resized to image_size x image_size
        :param batch_size: The batch size to use for validation
        :param model_type: Used to choose pre-processing function for one of
                           the "resnet50" or "mobilenetv1" model types
        """

        if not dataset_dir:
            raise ValueError("dataset_dir cannot not be None")
        self._dataset_dir = dataset_dir
        self._batch_size = batch_size
        self._model_type = model_type
        self._val_dataset = ImageNetDataset(dataset_dir, image_size, batch_size).dataset

    def evaluate(self, model: tf.keras.Model, iterations: int = None):
        """
        Evaluates the model on the validation dataset
        :param model: Model to be evaluated
        :param iterations: The number of iterations to run. If None, all the data will be used
        """
        # Get specific model's preprocessing and decode functions
        if self._model_type == "resnet50":
            from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
        elif self._model_type == "mobilenetv1":
            from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
        else:
            raise ValueError(
                "This notebook only support ResNet50 or MobileNet")

        # If no iterations specified, set to full validation set
        if iterations is None or iterations > len(self._val_dataset):
            logger.info("Iterations is None or greater than the number of batches in the validation set. "
                        "Using full validation set.")
            iterations = image_net_config.dataset["val_images_len"]
        else:
            iterations *= self._batch_size

        top1 = 0
        top5 = 0
        total = 0
        curr_iter = 0

        for (img, label) in self._val_dataset:
            progbar = Progbar(iterations, stateful_metrics=["Top1", "Top5"])
            preds = model.predict(preprocess_input(
                img), batch_size=self._batch_size)
            label = np.where(label)[1]
            label = [self._val_dataset.class_names[int(i)] for i in label]
            cnt = sum([1 for a, b in zip(label, decode_predictions(
                preds, top=1)) if str(a) == b[0][0]])
            top1 += cnt
            cnt = sum([1 for a, b in zip(label, decode_predictions(
                preds, top=5)) if str(a) in [i[0] for i in b]])
            top5 += cnt
            total += len(label)

            curr_iter += 1

            progbar.update(
                total, values=[("Top1", top1 / total), ("Top5", top5 / total)])
            if total >= iterations:
                break
