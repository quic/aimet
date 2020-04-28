# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Models for use in unit testing """

import torch
import torch.nn as nn


# pylint: disable=too-many-instance-attributes
class SingleResidual(nn.Module):
    """ A model with a single residual connection.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=10):
        super(SingleResidual, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # All layers above are same as ResNet
        # The output of the MaxPool2d is used as a residual.

        # The following layers are considered as single block.
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2, bias=False)

        # The output of Conv2d layer above(conv3) is added with the the residual from
        # MaxPool2d and then fed to the relu layer below.
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.conv4 = nn.Conv2d(32, 8, kernel_size=2, stride=2, padding=2, bias=True)
        self.ada = nn.AdaptiveAvgPool2d(5)
        self.fc = nn.Linear(72, num_classes)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Save the output of MaxPool as residual.
        residual = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # Add the residual
        # AdaptiveAvgPool2d is used to get the desired dimension before adding.
        residual = self.conv4(residual)
        residual = self.ada(residual)
        x += residual
        x = self.relu3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultiInput(nn.Module):
    """ A model with multiple inputs.
        Use this model for unit testing purposes. """

    def __init__(self, num_classes=3):
        super(MultiInput, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc = nn.Linear(288, num_classes)

    def forward(self, *inputs):
        x1 = self.conv1(inputs[0])
        x1 = self.conv2(x1)
        x2 = self.conv3(inputs[1])
        x = x1 + x2
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConcatModel(nn.Module):
    """ A model with concat op.
        Use this model for unit testing purposes.
        Expected inputs: 3 inputs, all of size (1, 3, 8, 8) """

    def __init__(self, num_classes=3):
        super(ConcatModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=2, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(3, 4, kernel_size=2, stride=2, padding=2)
        self.conv3 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2)
        self.fc = nn.Linear(504, num_classes)

    def forward(self, *inputs):
        c1 = self.conv1(inputs[0])
        c2 = self.conv2(inputs[1])
        c3 = self.conv3(inputs[2])
        cat_inputs = [c1, c2, c3]
        x = torch.cat(cat_inputs, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModuleListModel(nn.Module):
    """ A model with modules defined using ModuleLists.
        Use this model for unit testing purposes.
        Expected inputs: 3 inputs, all of size (1, 3, 8, 8) """

    def __init__(self, num_classes=3):
        super(ModuleListModel, self).__init__()
        self.mod_list = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),       # use 4th
            nn.ReLU(inplace=True),      # use 3rd
            nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2),                # use 5th
            nn.ReLU(),      # dummy unused op
            nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=2, bias=False)        # use 1st
        ])
        self.seq_list = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2),                # use 6th
            nn.ReLU(),      # dummy unused op
            nn.BatchNorm2d(16),     # use 2nd
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, *inputs):
        x = self.mod_list[4](inputs[0])
        x = self.seq_list[2](x)
        x = self.mod_list[1](x)
        x = self.mod_list[0](x)
        x = self.mod_list[2](x)
        x = self.seq_list[0](x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TinyModel(nn.Module):
    """ Use this model for unit testing purposes. """

    def __init__(self):
        super(TinyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=2, stride=2, padding=2, bias=True)
        self.fc = nn.Linear(36, 12)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.avgpool(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModelWithDropouts(nn.Module):
    """ Use this model for unit testing purposes. """

    def __init__(self):
        super(ModelWithDropouts, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=.4)
        self.dropout2 = nn.Dropout2d(p=.6)
        self.fc = nn.Linear(2592, 10)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ModelWithReusedNodes(nn.Module):
    """ Model that reuses a relu module """

    def __init__(self):
        super(ModelWithReusedNodes, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2592, 10)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
