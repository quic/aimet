# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import torchaudio
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
            ' 0
            <SPACE> 1
            a 2
            b 3
            c 4
            d 5
            e 6
            f 7
            g 8
            h 9
            i 10
            j 11
            k 12
            l 13
            m 14
            n 15
            o 16
            p 17
            q 18
            r 19
            s 20
            t 21
            u 22
            v 23
            w 24
            x 25
            y 26
            z 27
            """
        self.char_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer array """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence


def librispeech_data_processing(data, data_type="train"):
    assert (data_type=='train' or data_type=='valid'), "data_type needs to be train/valid"
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    )

    valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

    text_transform = TextTransform()
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def get_cifar10_data_loaders(batch_size=64, num_workers=4, drop_last=True):
    train_set = torchvision.datasets.CIFAR10("./data/CIFAR10", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
    val_set = torchvision.datasets.CIFAR10("./data/CIFAR10", train=False, download=True,
                                           transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last)
    return train_loader, val_loader


def get_librispeech_data_loaders(batch_size=64, num_workers=4, drop_last=True):
    train_set = torchaudio.datasets.LIBRISPEECH("./data/LIBRISPEECH", url='train-clean-100', download=True)
    val_set = torchaudio.datasets.LIBRISPEECH("./data/LIBRISPEECH", url='dev-clean', download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=lambda x: librispeech_data_processing(x, 'train'))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=lambda x: librispeech_data_processing(x, 'valid'))
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
    train_loader, _ = get_cifar10_data_loaders()
    base_lr = 0.0001
    max_lr = 0.06
    momentum = 0.9
    steps = int(len(train_loader) * epochs / 2.0)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=steps)
    model_train(model, train_loader, epochs, optimizer, scheduler)


def train_librispeech(model: torch.nn.Module, epochs, max_batches):
    """
    Trains a PyTorch model on LIBRISPEECH for the specified number of epochs

    :param model: PyTorch model to train
    :param epochs: Number of epochs to train
    """
    use_cuda = next(model.parameters()).is_cuda
    model.train()
    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    train_loader, _ = get_librispeech_data_loaders(batch_size=16)
    lr = 0.01
    steps = int(len(train_loader))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps, epochs=epochs)

    for epoch in range(epochs):
        for (i, batch) in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = batch
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(spectrograms)
            output = output.transpose(0, 1)
            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i+1 >= max_batches:
                break


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
