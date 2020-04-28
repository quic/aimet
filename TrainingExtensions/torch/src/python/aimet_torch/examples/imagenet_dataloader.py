# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Data loader for ImageNet """

import os
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as torch_data


class ImageNetDataLoader:
    """ ImageNet Data Loader """

    def __init__(self, images_dir, size, batch_size, num_workers):

        self.images_dir = images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.train_transforms = transforms.Compose([transforms.RandomResizedCrop(size),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    normalize])

        self.val_transforms = transforms.Compose([transforms.Resize(size + 24),
                                                  transforms.CenterCrop(size),
                                                  transforms.ToTensor(),
                                                  normalize])
        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self):
        """ Property that exposes a data loader for training samples """

        root = os.path.join(self.images_dir, 'train')
        if not self._train_loader:
            train_set = torchvision.datasets.ImageFolder(root=root, transform=self.train_transforms)
            self._train_loader = torch_data.DataLoader(train_set, batch_size=self.batch_size, shuffle=False,
                                                       num_workers=self.num_workers, pin_memory=True)
        return self._train_loader

    @property
    def val_loader(self):
        """ Property that exposes a data loader for validation samples """

        root = os.path.join(self.images_dir, 'val')
        if not self._val_loader:
            val_set = torchvision.datasets.ImageFolder(root=root, transform=self.val_transforms)
            self._val_loader = torch_data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False,
                                                     num_workers=self.num_workers, pin_memory=True)
        return self._val_loader
