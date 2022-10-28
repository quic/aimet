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
Creates data-loader for Image-Net dataset
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class ImageNetDataset:
    """ Dataset for ImageNet with Keras """

    def __init__(self, dataset_dir: str, image_size: int = 224, batch_size: int = 128, is_training: bool = False):
        """
        :param dataset_dir: The path to the dataset's directory
        :param image_size: Required size for images. Images will be resized to image_size x image_size
        :param batch_size: The batch size to use for training and validation
        :param is_training: If the dataset returned should be training or validation dataset
        """

        if not dataset_dir:
            raise ValueError("dataset_dir cannot be None")

        self._dataset = image_dataset_from_directory(directory=dataset_dir,
                                                     labels="inferred",
                                                     label_mode="categorical",
                                                     batch_size=batch_size,
                                                     shuffle=is_training,
                                                     image_size=(image_size, image_size))

        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        """
        Returns the batch size
        :return: a value representing batch size
        """
        return self._batch_size

    @property
    def dataset(self) -> tf.data.Dataset:
        """
        Returns the dataset
        :return: a tf.data.Dataset representing dataset
        """
        return self._dataset
