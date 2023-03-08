# /usr/bin/env python3.8
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


def _get_cifar10_data_loaders(batch_size=64, num_workers=4, drop_last=True):
    train_set = torchvision.datasets.CIFAR10("./data/CIFAR10", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
    val_set = torchvision.datasets.CIFAR10("./data/CIFAR10", train=False, download=True,
                                           transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last)
    return train_loader, val_loader


def model_train(model: torch.nn.Module, train_loader: DataLoader, epochs: int, optimizer: optim.Optimizer, scheduler):
    """
    Trains the given torch model for the specified number of epochs

    :param model: model
    :param train_loader: Dataloader containing the training data
    :param epochs: number of training
    :param optimizer: Optimizer object for training
    :param scheduler: Learning rate scheduler
    """
    use_cuda = next(model.parameters()).is_cuda
    model.train()
    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            scheduler.step()


def train_cifar10(model: torch.nn.Module, epochs):
    """
    Trains a PyTorch model on CIFAR-10 for the specified number of epochs

    :param model: PyTorch model to train
    :param epochs: Number of epochs to train
    """
    train_loader, _ = _get_cifar10_data_loaders()
    base_lr = 0.0001
    max_lr = 0.06
    momentum = 0.9
    steps = int(len(train_loader) * epochs / 2.0)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=steps)
    model_train(model, train_loader, epochs, optimizer, scheduler)


def model_eval_torch(model: torch.nn.Module, val_loader: DataLoader):
    """
    Measures the accuracy of a PyTorch model over a given validation dataset

    :param model: model to be evaluated
    :param val_loader: Dataloader containing the validation dataset
    :return: top_1_accuracy on validation data
    """

    use_cuda = next(model.parameters()).is_cuda
    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model.eval()

    corr = 0
    total = 0
    for (i, batch) in enumerate(val_loader):
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x)
        corr += torch.sum(torch.argmax(out, dim=1) == y)
        total += x.shape[0]
    return corr / total
