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

""" Cross Layer Equalization implementation using Model Optimization (c++) """

from typing import Dict
import torch
import numpy as np
import aimet_common.libpymo as libpymo

from aimet_common.cross_layer_equalization import ClsSetInfo
from aimet_torch.cle.impl import ClsSet, CLSImpl, HBFImpl


class MOCLSImpl(CLSImpl):
    """
    This class implements the CLS algorithm using MO version while following the base Implementation interface.
    """
    def scale_cls_set_with_depthwise_layers(self, cls_set: ClsSet) -> [np.ndarray, np.ndarray]:
        """
        API to invoke equalize layer params for depth wise separable layers(update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers whose weights and biases need to be equalized.
                        Second Conv layer is a depth-wise conv and third conv layer is point-wise conv
        :return: Scaling factors S_12 and S_23 : numpy arrays
        """
        # Create structs for holding layer weights and bias parameters
        prev_layer_params = libpymo.EqualizationParams()
        curr_layer_params = libpymo.EqualizationParams()
        next_layer_params = libpymo.EqualizationParams()

        # Prepare and pack data structures for cls set.
        self._pack_params_for_depthwise_conv(cls_set, prev_layer_params, curr_layer_params, next_layer_params)

        # Scales weights and bias for consecutive layers and updates data structures in-place.
        scaling_params = libpymo.scaleDepthWiseSeparableLayer(prev_layer_params, curr_layer_params, next_layer_params)

        # Update weight and biases for cls set using updated data structures.
        self._update_params_for_depthwise_conv(cls_set, prev_layer_params, curr_layer_params, next_layer_params)

        return scaling_params.scalingMatrix12, scaling_params.scalingMatrix23

    def scale_cls_set_with_conv_layers(self, cls_set: ClsSet) -> np.ndarray:
        """
        API to invoke equalize layer params for regular conv layers (update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized
        :return: Scaling factor S_12 for each conv layer pair: numpy array
        """
        # Create structs for holding layer weights and bias parameters
        prev_layer_params = libpymo.EqualizationParams()
        curr_layer_params = libpymo.EqualizationParams()

        # Prepare and pack data structures for cls set.
        self._pack_params_for_conv(cls_set, prev_layer_params, curr_layer_params)

        # Scales weights and bias for consecutive layers and updates data structures in-place.
        scaling_factor = libpymo.scaleLayerParams(prev_layer_params, curr_layer_params)

        # Update weight and biases for cls set using updated data structures.
        self._update_params_for_conv(cls_set, prev_layer_params, curr_layer_params)

        return scaling_factor

    @staticmethod
    def _pack_params_for_conv(cls_set: ClsSet,
                              prev_layer_params: libpymo.EqualizationParams,
                              curr_layer_params: libpymo.EqualizationParams):
        """
        Prepare and pack data structure for previous and current layer in given cls set.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        """
        weight_set_0 = cls_set[0].weight

        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        if isinstance(cls_set[0], torch.nn.ConvTranspose2d):
            weight_set_0 = weight_set_0.permute(1, 0, 2, 3)
        if isinstance(cls_set[0], torch.nn.ConvTranspose1d):
            weight_set_0 = weight_set_0.permute(1, 0, 2)

        prev_layer_params.weight = weight_set_0.detach().numpy().reshape(-1)
        prev_layer_params.weightShape = np.array(weight_set_0.shape)
        if len(prev_layer_params.weightShape) == 3:
            prev_layer_params.weightShape = prev_layer_params.weightShape + [1]

        weight_set_1 = cls_set[1].weight

        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        if isinstance(cls_set[1], torch.nn.ConvTranspose2d):
            weight_set_1 = weight_set_1.permute(1, 0, 2, 3)
        if isinstance(cls_set[1], torch.nn.ConvTranspose1d):
            weight_set_1 = weight_set_1.permute(1, 0, 2)

        curr_layer_params.weight = weight_set_1.detach().numpy().reshape(-1)
        curr_layer_params.weightShape = np.array(weight_set_1.shape)
        if len(curr_layer_params.weightShape) == 3:
            curr_layer_params.weightShape = curr_layer_params.weightShape + [1]

        if cls_set[0].bias is not None:
            prev_layer_params.bias = cls_set[0].bias.detach().numpy()
        else:
            prev_layer_params.isBiasNone = True

    @staticmethod
    def _update_params_for_conv(cls_set: ClsSet,
                                prev_layer_params: libpymo.EqualizationParams,
                                curr_layer_params: libpymo.EqualizationParams):
        """
        Update weight and biases for cls set using updated data structures.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        """
        if isinstance(cls_set[0], (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            prev_layer_params.weightShape = prev_layer_params.weightShape[:-1]
        cls_set[0].weight.data = torch.from_numpy(np.reshape(prev_layer_params.weight,
                                                             prev_layer_params.weightShape))
        cls_set[0].weight.data = cls_set[0].weight.data.type(torch.FloatTensor)

        # Transpose weight back to N, C, H, W for transposed Conv2D
        if isinstance(cls_set[0], torch.nn.ConvTranspose2d):
            cls_set[0].weight.data = cls_set[0].weight.data.permute(1, 0, 2, 3).contiguous()
        if isinstance(cls_set[0], torch.nn.ConvTranspose1d):
            cls_set[0].weight.data = cls_set[0].weight.data.permute(1, 0, 2).contiguous()

        if isinstance(cls_set[1], (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            curr_layer_params.weightShape = curr_layer_params.weightShape[:-1]
        cls_set[1].weight.data = torch.from_numpy(np.reshape(curr_layer_params.weight,
                                                             curr_layer_params.weightShape))
        cls_set[1].weight.data = cls_set[1].weight.data.type(torch.FloatTensor)

        # Transpose weight back to N, C, H, W for transposed Conv2D
        if isinstance(cls_set[1], torch.nn.ConvTranspose2d):
            cls_set[1].weight.data = cls_set[1].weight.data.permute(1, 0, 2, 3).contiguous()
        if isinstance(cls_set[1], torch.nn.ConvTranspose1d):
            cls_set[1].weight.data = cls_set[1].weight.data.permute(1, 0, 2).contiguous()

        if cls_set[0].bias is not None:
            cls_set[0].bias.data = torch.from_numpy(np.reshape(prev_layer_params.bias,
                                                               prev_layer_params.weightShape[0]))
            cls_set[0].bias.data = cls_set[0].bias.data.type(torch.FloatTensor)

    @staticmethod
    def _pack_params_for_depthwise_conv(cls_set: ClsSet,
                                        prev_layer_params: libpymo.EqualizationParams,
                                        curr_layer_params: libpymo.EqualizationParams,
                                        next_layer_params: libpymo.EqualizationParams):
        """
        Prepare and pack data structure for previous, current and next layer in given cls set.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        :param next_layer_params: Data structure holding weight and bias for next layer in cls set.
        """
        if isinstance(cls_set[0], torch.nn.ConvTranspose2d):
            cls_set[0].weight.data = cls_set[0].weight.data.permute(1, 0, 2, 3).contiguous()
        if isinstance(cls_set[0], torch.nn.ConvTranspose1d):
            cls_set[0].weight.data = cls_set[0].weight.data.permute(1, 0, 2).contiguous()

        if isinstance(cls_set[2], torch.nn.ConvTranspose2d):
            cls_set[2].weight.data = cls_set[2].weight.data.permute(1, 0, 2, 3).contiguous()
        if isinstance(cls_set[2], torch.nn.ConvTranspose1d):
            cls_set[2].weight.data = cls_set[2].weight.data.permute(1, 0, 2).contiguous()

        assert cls_set[1].groups > 1

        prev_layer_params.weight = cls_set[0].weight.detach().numpy().flatten()
        prev_layer_params.weightShape = np.array(cls_set[0].weight.shape)
        if len(prev_layer_params.weightShape) == 3:
            prev_layer_params.weightShape = prev_layer_params.weightShape + [1]

        curr_layer_params.weight = cls_set[1].weight.detach().numpy().flatten()
        curr_layer_params.weightShape = np.array(cls_set[1].weight.shape)
        if len(curr_layer_params.weightShape) == 3:
            curr_layer_params.weightShape = curr_layer_params.weightShape + [1]

        next_layer_params.weight = cls_set[2].weight.detach().numpy().flatten()
        next_layer_params.weightShape = np.array(cls_set[2].weight.shape)
        if len(next_layer_params.weightShape) == 3:
            next_layer_params.weightShape = next_layer_params.weightShape + [1]

        if cls_set[0].bias is not None:
            prev_layer_params.bias = cls_set[0].bias.detach().numpy()
        else:
            prev_layer_params.isBiasNone = True

        if cls_set[1].bias is not None:
            curr_layer_params.bias = cls_set[1].bias.detach().numpy()
        else:
            curr_layer_params.isBiasNone = True

    @staticmethod
    def _update_params_for_depthwise_conv(cls_set: ClsSet,
                                          prev_layer_params: libpymo.EqualizationParams,
                                          curr_layer_params: libpymo.EqualizationParams,
                                          next_layer_params: libpymo.EqualizationParams):
        """
        Update weight and biases for cls set using updated data structures.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        :param next_layer_params: Data structure holding weight and bias for next layer in cls set.
        """
        if isinstance(cls_set[0], (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            prev_layer_params.weightShape = prev_layer_params.weightShape[:-1]
        cls_set[0].weight.data = torch.from_numpy(np.reshape(prev_layer_params.weight,
                                                             prev_layer_params.weightShape))
        cls_set[0].weight.data = cls_set[0].weight.data.type(torch.FloatTensor)
        if isinstance(cls_set[0], torch.nn.ConvTranspose2d):
            cls_set[0].weight.data = cls_set[0].weight.data.permute(1, 0, 2, 3).contiguous()
        if isinstance(cls_set[0], torch.nn.ConvTranspose1d):
            cls_set[0].weight.data = cls_set[0].weight.data.permute(1, 0, 2).contiguous()

        if isinstance(cls_set[1], (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            curr_layer_params.weightShape = curr_layer_params.weightShape[:-1]
        cls_set[1].weight.data = torch.from_numpy(np.reshape(curr_layer_params.weight,
                                                             curr_layer_params.weightShape))
        cls_set[1].weight.data = cls_set[1].weight.data.type(torch.FloatTensor)

        if isinstance(cls_set[2], (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            next_layer_params.weightShape = next_layer_params.weightShape[:-1]

        cls_set[2].weight.data = torch.from_numpy(np.reshape(next_layer_params.weight,
                                                             next_layer_params.weightShape))
        cls_set[2].weight.data = cls_set[2].weight.data.type(torch.FloatTensor)
        if isinstance(cls_set[2], torch.nn.ConvTranspose2d):
            cls_set[2].weight.data = cls_set[2].weight.data.permute(1, 0, 2, 3).contiguous()
        if isinstance(cls_set[2], torch.nn.ConvTranspose1d):
            cls_set[2].weight.data = cls_set[2].weight.data.permute(1, 0, 2).contiguous()

        if cls_set[0].bias is not None:
            cls_set[0].bias.data = torch.from_numpy(np.reshape(prev_layer_params.bias,
                                                               prev_layer_params.weightShape[0]))
            cls_set[0].bias.data = cls_set[0].bias.data.type(torch.FloatTensor)

        if cls_set[1].bias is not None:
            cls_set[1].bias.data = torch.from_numpy(np.reshape(curr_layer_params.bias,
                                                               curr_layer_params.weightShape[0]))
            cls_set[1].bias.data = cls_set[1].bias.data.type(torch.FloatTensor)


class MOHBFImpl(HBFImpl):
    """
    This class implements the HBF algorithm using MO version while following the base Implementation interface.
    """
    def bias_fold(self, cls_pair_info, bn_layers):
        """
        Bias fold implementation using Model optimization (c++) version.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param bn_layers: Dictionary with Key being Conv/Linear layer and value being corresponding folded BN layer.
        """
        # Create data structures for holding layer weights and bias parameters.
        prev_layer_params = libpymo.LayerParams()
        curr_layer_params = libpymo.LayerParams()
        prev_layer_bn_params = libpymo.BNParamsHighBiasFold()

        # Prepare and pack data structures for high bias fold.
        self._pack_bn_layer_params(cls_pair_info, bn_layers, prev_layer_bn_params)
        self._pack_previous_and_current_layer_params(cls_pair_info, prev_layer_params, curr_layer_params)

        # Update bias for previous and current layer and data structures in-place.
        libpymo.updateBias(prev_layer_params, curr_layer_params, prev_layer_bn_params)

        # Set updated biases for previous and current layer.
        self._update_previous_and_current_layer_bias(cls_pair_info, prev_layer_params, curr_layer_params)

    @staticmethod
    def _pack_bn_layer_params(cls_pair_info: ClsSetInfo.ClsSetLayerPairInfo,
                              bn_layers: Dict[torch.nn.Module, torch.nn.BatchNorm2d],
                              prev_layer_bn_params: libpymo.BNParamsHighBiasFold):
        """
        Helper method to pack batch norm layer parameter for high bias fold.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param bn_layers: Dictionary with Key being Conv/Linear layer and value being corresponding folded BN layer.
        :param prev_layer_bn_params: Data structure to pack batch norm parameter.
        """
        scaling_parameter = cls_pair_info.scale_factor

        # Scaling gamma and beta parameter of batch norm layer
        prev_layer_bn_params.gamma = bn_layers[cls_pair_info.layer1].weight.detach().numpy().reshape(-1)
        prev_layer_bn_params.beta = bn_layers[cls_pair_info.layer1].bias.detach().numpy().reshape(-1)

        if len(scaling_parameter) != len(prev_layer_bn_params.gamma) or \
                len(scaling_parameter) != len(prev_layer_bn_params.beta):
            raise ValueError("High Bias absorption is not supported for networks with fold-forward BatchNorms")
        prev_layer_bn_params.gamma = np.divide(prev_layer_bn_params.gamma, scaling_parameter)
        prev_layer_bn_params.beta = np.divide(prev_layer_bn_params.beta, scaling_parameter)

    @staticmethod
    def _pack_previous_and_current_layer_params(cls_pair_info, prev_layer_params, curr_layer_params):
        """
        Helper method to pack information of previous and current layer.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param prev_layer_params: Data structure to pack previous layer parameters.
        :param curr_layer_params: Data structure to pack current layer parameters.
        """
        prev_layer_params.activationIsRelu = cls_pair_info.relu_activation_between_layers
        prev_layer_params.bias = cls_pair_info.layer1.bias.detach().numpy()

        weight = cls_pair_info.layer2.weight

        if isinstance(cls_pair_info.layer2, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            weight = torch.unsqueeze(weight, dim=-1)

        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        if isinstance(cls_pair_info.layer2, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)) and \
                cls_pair_info.layer2.groups == 1:
            weight = weight.permute(1, 0, 2, 3)

        curr_layer_params.bias = cls_pair_info.layer2.bias.detach().numpy()
        curr_layer_params.weight = weight.detach().numpy().reshape(-1)
        curr_layer_params.weightShape = np.array(weight.shape)

    @staticmethod
    def _update_previous_and_current_layer_bias(cls_pair_info: ClsSetInfo.ClsSetLayerPairInfo,
                                                prev_layer_params: libpymo.LayerParams,
                                                curr_layer_params: libpymo.LayerParams):
        """
        Update biases for previous and current layer.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        """
        prev_layer_bias_shape = cls_pair_info.layer1.weight.shape[0]
        if (isinstance(cls_pair_info.layer1, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d))) and \
                (cls_pair_info.layer1.groups == 1):
            prev_layer_bias_shape = cls_pair_info.layer1.weight.shape[1]

        cls_pair_info.layer1.bias.data = torch.from_numpy(np.reshape(prev_layer_params.bias,
                                                                     prev_layer_bias_shape))
        cls_pair_info.layer1.bias.data = cls_pair_info.layer1.bias.data.type(torch.FloatTensor)

        cls_pair_info.layer2.bias.data = torch.from_numpy(np.reshape(curr_layer_params.bias,
                                                                     curr_layer_params.weightShape[0]))
        cls_pair_info.layer2.bias.data = cls_pair_info.layer2.bias.data.type(torch.FloatTensor)
