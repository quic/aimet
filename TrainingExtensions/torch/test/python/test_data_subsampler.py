# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020 Qualcomm Innovation Center, Inc. All rights reserved.
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

import unittest.mock
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as functional

from aimet_torch.utils import create_fake_data_loader
from aimet_torch.data_subsampler import DataSubSampler


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.fc1 = nn.Linear(160, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = functional.relu(functional.max_pool2d(self.conv1(x), 2))
        x = functional.relu(functional.max_pool2d(self.conv2(x), 2))
        x = x.view(x.view(0), -1)
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return functional.log_softmax(x, dim=1)


class TestDataSubSampler(unittest.TestCase):

    @unittest.mock.patch('numpy.random.choice')
    def test_subsampled_output_data(self, np_choice_function):

        """ Test to collect activations (input from model_copy and output from model for conv2 layer) and compare
            with sub sampled output data
        """
        # hardcoded mocked 10 sample locations
        # (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (6, 6), (5, 5)
        np_choice_function.return_value = heights = widths = [0, 1, 2, 3, 4, 5, 6, 7, 6, 5]

        orig_model = TestNet().cuda()
        comp_model = copy.deepcopy(orig_model)
        # only one image and from that 10 samples
        dataset_size = 1
        batch_size = 1
        num_reconstruction_samples = 10

        # create fake data loader with image size (1, 28, 28)
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size, image_size=(1, 28, 28))

        _, conv2_output_data = DataSubSampler.get_sub_sampled_data(orig_layer=orig_model.conv2,
                                                                   pruned_layer=comp_model.conv2,
                                                                   orig_model=orig_model,
                                                                   comp_model=comp_model,
                                                                   data_loader=data_loader,
                                                                   num_reconstruction_samples=
                                                                   num_reconstruction_samples)

        # collect the output data of conv2 from original model using same data loader
        iterator = data_loader.__iter__()
        images_in_one_batch, _ = iterator.__next__()
        conv1_output = orig_model.conv1(images_in_one_batch.cuda())
        conv2_input = conv1_output
        conv2_output = orig_model.conv2(functional.relu(functional.max_pool2d(conv2_input, 2))).\
            detach().cpu().numpy()

        # compare output data with sub sampled output data
        for sample in range(num_reconstruction_samples):

            self.assertTrue(np.array_equal(conv2_output_data[sample, :],
                                           conv2_output[0, :, heights[sample], widths[sample]]))

    @unittest.mock.patch('numpy.random.choice')
    def test_subsampled_input_data(self, np_choice_function):

        """ Test to collect activations (input from model_copy and output from model for conv2 layer) and compare
            with sub sampled input data
        """
        # hardcoded mocked 10 sample locations
        # (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (6, 6), (5, 5)
        np_choice_function.return_value = heights = widths = [0, 1, 2, 3, 4, 5, 6, 7, 6, 5]

        orig_model = TestNet().cuda()
        comp_model = copy.deepcopy(orig_model)
        # only one image and from that 10 samples
        dataset_size = 1
        batch_size = 1
        num_reconstruction_samples = 10

        # create fake data loader with image size (1, 28, 28)
        data_loader = create_fake_data_loader(dataset_size=dataset_size, batch_size=batch_size, image_size=(1, 28, 28))

        conv2_input_data, _ = DataSubSampler.get_sub_sampled_data(orig_layer=orig_model.conv2,
                                                                  pruned_layer=comp_model.conv2,
                                                                  orig_model=orig_model,
                                                                  comp_model=comp_model,
                                                                  data_loader=data_loader,
                                                                  num_reconstruction_samples=num_reconstruction_samples)

        # collect the input data of conv2 from compressed model using same data loader

        iterator = data_loader.__iter__()
        images_in_one_batch, _ = iterator.__next__()
        conv1_output = comp_model.conv1(images_in_one_batch.cuda())
        conv2_input = functional.relu(functional.max_pool2d(conv1_output, 2))

        kernel_size_h, kernel_size_w = comp_model.conv2.kernel_size

        for sample in range(num_reconstruction_samples):

            self.assertTrue(np.array_equal(conv2_input_data[sample, :, :, :],
                                           conv2_input[0, :, heights[sample]:heights[sample] + kernel_size_h,
                                           widths[sample]:widths[sample] + kernel_size_w].detach().cpu().numpy()))
