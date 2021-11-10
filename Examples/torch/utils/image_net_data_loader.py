# !/usr/bin/env python
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
Creates data-loader for Image-Net dataset
"""
import logging
import os

from torchvision import transforms
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
from torch.utils.data import Dataset
import torch.utils.data as torch_data


from Examples.common import image_net_config

logger = logging.getLogger('Dataloader')

IMG_EXTENSIONS = '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif'


def make_dataset(directory: str, class_to_idx: dict, extensions: tuple, num_samples_per_class: int) -> list:
    """
    Creates a dataset of images with num_samples_per_class images in each class

    :param directory: The string path to the data directory.
    :param class_to_idx: A dictionary mapping the name of the class to the index (label)
    :param extensions: list of valid extensions to load data
    :param num_samples_per_class: Number of samples to use per class.

    :return: list of images containing the entire dataset.
    """
    images = []
    num_classes = 0
    directory = os.path.expanduser(directory)
    for class_name in sorted(class_to_idx.keys()):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            class_idx = class_to_idx[class_name]
            class_images = add_images_for_class(class_path, extensions, num_samples_per_class, class_idx)
            images.extend(class_images)
            num_classes += 1

    logger.info("Dataset consists of %d images in %d classes", len(images), num_classes)
    return images


def add_images_for_class(class_path: str, extensions: tuple, num_samples_per_class: int, class_idx: int) -> list:
    """
    For a given class, adds num_samples_per_class images to a list.

    :param class_path: The string path to the class directory.
    :param extensions: List of valid extensions to load data
    :param num_samples_per_class: Number of samples to use per class.
    :param class_idx: numerical index of class.

    :return: list of images for given class.
    """
    class_images = []
    count = 0
    for file_name in os.listdir(class_path):
        if num_samples_per_class and count >= num_samples_per_class:
            break
        if has_file_allowed_extension(file_name, extensions):
            image_path = os.path.join(class_path, file_name)
            item = (image_path, class_idx)
            class_images.append(item)
            count += 1

    return class_images


class ImageFolder(Dataset):
    """
    Dataset class inspired by torchvision.datasets.folder.DatasetFolder for images organized as
        individual files grouped by category.
    """

    def __init__(self, root: str, transform=None, target_transform=None,
                 num_samples_per_class: int = None):

        """
        :param root: The path to the data directory.
        :param transform: The required processing to be applied on the sample.
        :param target_transform:  The required processing to be applied on the target.
        :param num_samples_per_class: Number of samples to use per class.

        """
        Dataset.__init__(self)
        classes, class_to_idx = self._find_classes(root)
        self.samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS, num_samples_per_class)
        if not self.samples:
            raise (RuntimeError(
                "Found 0 files in sub folders of: {}\nSupported extensions are: {}".format(
                    root, ",".join(IMG_EXTENSIONS))))

        self.root = root
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in self.samples]

        self.transform = transform
        self.target_transform = target_transform

        self.imgs = self.samples

    @staticmethod
    def _find_classes(directory: str):
        classes = [d for d in os.listdir(directory) if
                   os.path.isdir(os.path.join(directory, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


class ImageNetDataLoader:
    """
    For loading Validation data from the ImageNet dataset.
    """

    def __init__(self, images_dir: str, image_size: int, batch_size: int = 128,
                 is_training: bool = False, num_workers: int = 8, num_samples_per_class: int = None):
        """
        :param images_dir: The path to the data directory
        :param image_size: The length of the image
        :param batch_size: The batch size to use for training and validation
        :param is_training: Indicates whether to load the training or validation data
        :param num_workers: Indiicates to the data loader how many sub-processes to use for data loading.
        :param num_samples_per_class: Number of samples to use per class.
        """

        # For normalization, mean and std dev values are calculated per channel
        # and can be found on the web.
        normalize = transforms.Normalize(mean=image_net_config.dataset['images_mean'],
                                         std=image_net_config.dataset['images_std'])

        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        self.val_transforms = transforms.Compose([
            transforms.Resize(image_size + 24),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize])

        if is_training:
            data_set = ImageFolder(
                root=os.path.join(images_dir, 'train'), transform=self.train_transforms,
                num_samples_per_class=num_samples_per_class)
        else:
            data_set = ImageFolder(
                root=os.path.join(images_dir, 'val'), transform=self.val_transforms,
                num_samples_per_class=num_samples_per_class)

        self._data_loader = torch_data.DataLoader(
            data_set, batch_size=batch_size, shuffle=is_training,
            num_workers=num_workers, pin_memory=True)

    @property
    def data_loader(self) -> torch_data.DataLoader:
        """
        Returns the data-loader
        """
        return self._data_loader
