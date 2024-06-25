# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Cross Layer Equalization implementation using python """

import torch
import numpy as np

from aimet_common.cross_layer_equalization import ClsImpl, HbfImpl


class PythonClsImpl(ClsImpl):
    """
    This class implements the CLS algorithm using Python version while following the base Implementation interface.
    """
    def scale_cls_set_with_depthwise_layers(self, cls_set) -> [np.ndarray, np.ndarray]:
        """
        API to invoke equalize layer params for depth wise separable layers(update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers whose weights and biases need to be equalized.
                        Second Conv layer is a depth-wise conv and third conv layer is point-wise conv
        :return: Scaling factors S_12 and S_23 : numpy arrays
        """
        weight_0 = cls_set[0].weight.detach()
        weight_0 = self._transpose_tensor_in_common_format(cls_set[0], weight_0)
        weight_0 = self._make_4d_tensor(cls_set[0], weight_0)

        weight_1 = cls_set[1].weight.detach()
        assert cls_set[1].groups > 1
        weight_1 = self._make_4d_tensor(cls_set[1], weight_1)

        weight_2 = cls_set[2].weight.detach()
        weight_2 = self._transpose_tensor_in_common_format(cls_set[2], weight_2)
        weight_2 = self._make_4d_tensor(cls_set[2], weight_2)

        bias_0 = None
        if cls_set[0].bias is not None:
            bias_0 = cls_set[0].bias.detach()
        bias_1 = None
        if cls_set[1].bias is not None:
            bias_1 = cls_set[1].bias.detach()

        max_0 = torch.amax(torch.abs(weight_0), dim=(1, 2, 3))
        max_1 = torch.amax(torch.abs(weight_1), dim=(1, 2, 3))
        max_2 = torch.amax(torch.abs(weight_2), dim=(0, 2, 3))
        s_12 = max_0 / torch.pow(max_0 * max_1 * max_2, 1.0 / 3)
        s_23 = torch.pow(max_0 * max_1 * max_2, 1.0 / 3) / max_2

        # Avoid 'divide by zero' by using a value that does no scaling. i.e., 1.
        s_12 = torch.nan_to_num(s_12, nan=1.0)
        s_23 = torch.nan_to_num(s_23, nan=1.0)

        # inplace modifications on detached tensor(s) will update the original tensor(s).
        weight_0 *= (1.0 / s_12[:, None, None, None])
        weight_1 *= s_12[:, None, None, None] * (1.0 / s_23[:, None, None, None])
        weight_2 *= s_23[None, :, None, None]
        if bias_0 is not None:
            bias_0 *= (1.0 / s_12)
        if bias_1 is not None:
            bias_1 *= (1.0 / s_23)

        return s_12.numpy(), s_23.numpy()

    def scale_cls_set_with_conv_layers(self, cls_set) -> np.ndarray:
        """
        API to invoke equalize layer params for regular conv layers (update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized
        :return: Scaling factor S_12 for each conv layer pair: numpy array
        """
        weight_0 = cls_set[0].weight.detach()
        weight_0 = self._transpose_tensor_in_common_format(cls_set[0], weight_0)
        weight_0 = self._make_4d_tensor(cls_set[0], weight_0)

        weight_1 = cls_set[1].weight.detach()
        weight_1 = self._transpose_tensor_in_common_format(cls_set[1], weight_1)
        weight_1 = self._make_4d_tensor(cls_set[1], weight_1)

        bias_0 = None
        if cls_set[0].bias is not None:
            bias_0 = cls_set[0].bias.detach()

        max_0 = torch.amax(torch.abs(weight_0), dim=(1, 2, 3))
        max_1 = torch.amax(torch.abs(weight_1), dim=(0, 2, 3))
        scale_factor = max_0 / torch.pow(max_0 * max_1, 1. / 2)

        # Avoid 'divide by zero' by using a value that does no scaling. i.e., 1.
        scale_factor = torch.nan_to_num(scale_factor, nan=1.0)

        # inplace modifications on detached tensor(s) will update the original tensor(s).
        weight_0 *= (1.0 / scale_factor[:, None, None, None])
        weight_1 *= scale_factor[None, :, None, None]
        if bias_0 is not None:
            bias_0 *= (1.0 / scale_factor)

        return scale_factor.numpy()

    @staticmethod
    def _transpose_tensor_in_common_format(module: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        """
        Transpose tensor in the common format [Noc, Nin, Kh, Kw].
        For transposed conv, weight are stored in [Nin, Noc, Kh, Kw] format.

        :param module: Module.
        :param tensor: Input tensor.
        :return: Output tensor.
        """
        # Transpose weights to Noc, Nin, Kh, Kw from Nin, Noc, Kh, kw since axis are flipped for transposed conv2d
        if isinstance(module, torch.nn.ConvTranspose2d) and module.groups == 1:
            tensor = tensor.permute(1, 0, 2, 3)
        # Transpose weights to Noc, Nin, K from Nin, Noc, K since axis are flipped for transposed conv1d
        if isinstance(module, torch.nn.ConvTranspose1d) and module.groups == 1:
            tensor = tensor.permute(1, 0, 2)
        return tensor

    @staticmethod
    def _make_4d_tensor(module: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        """
        Return 4 dimensional tensor by adding a dimension on the end if the tensor is not 4d.

        :param module: Module.
        :param tensor: Input tensor.
        :return: Output tensor.
        """
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            assert len(tensor.shape) == 3, "Module should have 3d weight tensor."
            tensor = torch.unsqueeze(tensor, dim=-1)
        return tensor


class PythonHbfImpl(HbfImpl):
    """
    This class implements the HBF algorithm using python version while following the base Implementation interface.
    """
    # pylint: disable=no-self-use
    def bias_fold(self, cls_pair_info, bn_layers):
        """
        Bias fold implementation using python version.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param bn_layers: Dictionary with Key being Conv/Linear layer and value being corresponding folded BN layer.
        """
        activation_is_relu = cls_pair_info.relu_activation_between_layers
        beta = bn_layers[cls_pair_info.layer1].bias.detach() / torch.Tensor(cls_pair_info.scale_factor)
        gamma = bn_layers[cls_pair_info.layer1].weight.detach() / torch.Tensor(cls_pair_info.scale_factor)
        weight = cls_pair_info.layer2.weight.detach().cpu()

        if isinstance(cls_pair_info.layer2, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            weight = torch.unsqueeze(weight, dim=-1)

        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        if isinstance(cls_pair_info.layer2, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)) and \
                cls_pair_info.layer2.groups == 1:
            weight = weight.permute(1, 0, 2, 3)

        bias_prev_layer = cls_pair_info.layer1.bias.detach()
        bias_curr_layer = cls_pair_info.layer2.bias.detach()

        if not activation_is_relu:
            # No activation function, absorb whole bias
            absorb_bias = beta
        else:
            # Only absorb bias part that is more than 'min_std' standard deviations
            abs_gamma = torch.abs(gamma)
            absorb_bias = np.maximum(0, beta - 3 * abs_gamma)

        # Calculate correction term for next layer
        weight_matrix = weight.sum(3).sum(2)
        if weight_matrix.shape[1] == 1:
            weight_matrix = weight_matrix.reshape(weight_matrix.shape[0])
            bias_correction = torch.multiply(weight_matrix, absorb_bias)
        else:
            bias_correction = torch.matmul(weight_matrix, absorb_bias)

        # Update bias for previous and current layers.
        # inplace modifications on detached tensor(s) will update the original tensor(s).
        bias_prev_layer -= absorb_bias
        bias_curr_layer += bias_correction
