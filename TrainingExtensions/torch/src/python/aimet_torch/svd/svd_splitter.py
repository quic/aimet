# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Implementation of layer splitting logic for spatial and weight svd schemes """
import numpy as np
import torch
from torch.nn import Conv2d, Linear

from aimet_torch.winnow.winnow_utils import to_numpy
from aimet_common.utils import AimetLogger
from aimet_common.svd_pruner import SpatialSvdPruner

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class SpatialSvdModuleSplitter:
    """ Spatial SVD module splitter"""

    @staticmethod
    def split_module(module: Conv2d, rank: int):
        """
        :param module: Module to be split
        :param rank: rank for splitting
        :return: Two split modules
        """
        assert isinstance(module, Conv2d)
        assert module.dilation == (1, 1)

        weight_tensor = to_numpy(module.weight)  # n c h w
        out_channels, in_channels, height, width = weight_tensor.shape

        h, v = SpatialSvdPruner.lingalg_spatial_svd(weight_tensor, rank, in_channels, out_channels,
                                                    height, width)

        first_module = torch.nn.Conv2d(in_channels=module.in_channels,
                                       out_channels=rank, kernel_size=(height, 1),
                                       stride=(module.stride[0], 1),
                                       padding=(module.padding[0], 0), dilation=1, bias=False)
        first_module.weight.data = torch.FloatTensor(v).to(device=module.weight.device)

        second_module = torch.nn.Conv2d(in_channels=rank,
                                        out_channels=module.out_channels, kernel_size=(1, width),
                                        stride=(1, module.stride[1]),
                                        padding=(0, module.padding[1]), dilation=1, bias=module.bias is not None)
        second_module.weight.data = torch.FloatTensor(h).to(device=module.weight.device)
        if module.bias is not None:
            second_module.bias.data = module.bias.data

        return first_module, second_module


class WeightSvdModuleSplitter:
    """ Weight SVD module splitter """

    @classmethod
    def split_module(cls, module, name, rank, svd_lib_ref):
        """
        Split a given module using weight svd
        :param module: Module to be split
        :param name: Name of the module
        :param rank: Rank to use to split with
        :param svd_lib_ref: Reference to pymo
        :return: Two split modules
        """

        if isinstance(module, Conv2d):
            split_modules = cls.split_conv_module(module, name, rank, svd_lib_ref)

        elif isinstance(module, Linear):
            split_modules = cls.split_fc_module(module, name, rank, svd_lib_ref)

        else:
            raise AssertionError('Weight SVD only supports Conv2d and FC modules currently')

        return split_modules

    @classmethod
    def split_conv_module(cls, module, name, rank, svd_lib_ref):
        """
        Split a given Conv2D module using weight svd
        :param module: Module to be split
        :param name: Name of the module
        :param rank: Rank to use to split with
        :param svd_lib_ref: Reference to pymo
        :return: Two split modules
        """

        split_weights, weight_sizes = [], []

        conv_a_weight_shape = (rank, module.in_channels, 1, 1)
        conv_a_weight = np.zeros(conv_a_weight_shape)

        split_weights.append(conv_a_weight.flatten().tolist())
        weight_sizes.append(conv_a_weight.size)

        conv_b_weight_shape = (module.out_channels, rank, *module.kernel_size)
        conv_b_weight = np.zeros(conv_b_weight_shape)

        split_weights.append(conv_b_weight.flatten().tolist())
        weight_sizes.append(conv_b_weight.size)

        split_weights = svd_lib_ref.SplitLayerWeights(str(name), split_weights, weight_sizes,
                                                      [rank])

        logger.debug("Splitting conv module weight of shape %r into %r and %r",
                     module.weight.shape, conv_a_weight.shape, conv_b_weight.shape)

        # Todo: add sanity check for length of split_weights
        conv_a = torch.nn.Conv2d(module.in_channels, rank, kernel_size=(1, 1),
                                 stride=(1, 1), dilation=module.dilation)
        conv_b = torch.nn.Conv2d(rank, module.out_channels, kernel_size=module.kernel_size,
                                 stride=module.stride, padding=module.padding, dilation=module.dilation)

        conv_a.weight = torch.nn.Parameter(torch.from_numpy(np.array(split_weights[0],
                                                                     dtype=np.float32).reshape(conv_a_weight_shape)))
        conv_b.weight = torch.nn.Parameter(torch.from_numpy(np.array(split_weights[1],
                                                                     dtype=np.float32).reshape(conv_b_weight_shape)))

        if module.weight.is_cuda:
            conv_a.weight = torch.nn.Parameter(conv_a.weight.cuda())
            conv_b.weight = torch.nn.Parameter(conv_b.weight.cuda())

        cls._split_conv_bias(conv_a, conv_b, module, name, rank, svd_lib_ref)

        return conv_a, conv_b

    @staticmethod
    def _split_conv_bias(conv_a, conv_b, module, name, rank, svd_lib_ref):
        if module.bias is not None:
            split_biases, bias_sizes = [], []

            conv_a_bias = np.zeros(rank)
            split_biases.append(conv_a_bias.flatten().tolist())
            bias_sizes.append(conv_a_bias.size)

            conv_b_bias = np.zeros(module.out_channels)
            split_biases.append(conv_b_bias.flatten().tolist())
            bias_sizes.append(conv_b_bias.size)

            split_biases = svd_lib_ref.SplitLayerBiases(str(name), split_biases, bias_sizes,
                                                        [rank])

            conv_a.bias = torch.nn.Parameter(torch.from_numpy(np.array(split_biases[0], dtype=np.float32)))
            conv_b.bias = torch.nn.Parameter(torch.from_numpy(np.array(split_biases[1], dtype=np.float32)))

            if module.bias.is_cuda:
                conv_a.bias = torch.nn.Parameter(conv_a.bias.cuda())
                conv_b.bias = torch.nn.Parameter(conv_b.bias.cuda())

        else:
            conv_a.bias = None
            conv_b.bias = None

    @classmethod
    def split_fc_module(cls, module, name, rank, svd_lib_ref):
        """
        Split a given Linear module using weight svd
        :param module: Module to be split
        :param name: Name of the module
        :param rank: Rank to use to split with
        :param svd_lib_ref: Reference to pymo
        :return: Two split modules
        """

        split_weights, weight_sizes = [], []

        fc_a_weight_shape = (rank, module.in_features)
        fc_a_weight = np.zeros(fc_a_weight_shape)

        split_weights.append(fc_a_weight.flatten().tolist())
        weight_sizes.append(fc_a_weight.size)

        fc_b_weight_shape = (module.out_features, rank)
        fc_b_weight = np.zeros(fc_b_weight_shape)

        split_weights.append(fc_b_weight.flatten().tolist())
        weight_sizes.append(fc_b_weight.size)

        split_weights = svd_lib_ref.SplitLayerWeights(str(name), split_weights, weight_sizes,
                                                      [rank])

        # Todo: add sanity check for length of split_weights
        fc_a = torch.nn.Linear(module.in_features, rank)
        fc_b = torch.nn.Linear(rank, module.out_features)

        fc_a.weight = torch.nn.Parameter(torch.from_numpy(np.array(split_weights[0],
                                                                   dtype=np.float32).reshape(fc_a_weight_shape)))
        fc_b.weight = torch.nn.Parameter(torch.from_numpy(np.array(split_weights[1],
                                                                   dtype=np.float32).reshape(fc_b_weight_shape)))

        if module.weight.is_cuda:
            fc_a.weight = torch.nn.Parameter(fc_a.weight.cuda())
            fc_b.weight = torch.nn.Parameter(fc_b.weight.cuda())

        cls._split_fc_bias(fc_a, fc_b, module, name, rank, svd_lib_ref)

        return fc_a, fc_b

    @staticmethod
    def _split_fc_bias(fc_a, fc_b, module, name, rank, svd_lib_ref):
        if module.bias is not None:
            split_biases, bias_sizes = [], []

            fc_a_bias = np.zeros(rank)
            split_biases.append(fc_a_bias.flatten().tolist())
            bias_sizes.append(fc_a_bias.size)

            fc_b_bias = np.zeros(module.out_features)
            split_biases.append(fc_b_bias.flatten().tolist())
            bias_sizes.append(fc_b_bias.size)

            split_biases = svd_lib_ref.SplitLayerBiases(str(name), split_biases, bias_sizes,
                                                        [rank])

            fc_a.bias = torch.nn.Parameter(torch.from_numpy(np.array(split_biases[0], dtype=np.float32)))
            fc_b.bias = torch.nn.Parameter(torch.from_numpy(np.array(split_biases[1], dtype=np.float32)))

            if module.bias.is_cuda:
                fc_a.bias = torch.nn.Parameter(fc_a.bias.cuda())
                fc_b.bias = torch.nn.Parameter(fc_b.bias.cuda())

        else:
            fc_a.bias = None
            fc_b.bias = None
