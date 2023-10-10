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

""" Utilities for working with ModelOptimization C++ library """
from typing import List
import torch

import aimet_common.libpymo as pymo
from aimet_common.defs import CostMetric
from aimet_common.utils import AimetLogger
from aimet_torch.layer_database import Layer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class PymoSvdUtils:
    """ Utilities for working with SVD ModelOptimization C++ library """

    @classmethod
    def configure_layers_in_pymo_svd(cls, layers: List[Layer], cost_metric: CostMetric, svd_lib_ref):
        """
        Configure layers with the pymo svd library
        :param layers: List of layers to configure
        :param cost_metric: Cost metric to use
        :param svd_lib_ref: Reference to pymo instance
        :return:
        """

        # Set the pymo cost metric
        svd_lib_ref.SetCostMetric(cls._get_pymo_cost_metric(cost_metric))

        # Set up the layer attributes for each Conv2D/Linear layer
        for layer in layers:
            attr = pymo.LayerAttributes()
            attr.layerType = cls._get_pymo_layer_type(layer.module)
            attr.mode = pymo.TYPE_SINGLE

            # copy weights and bias to cpu and then convert to numpy
            weights = layer.module.weight.cpu().detach().numpy()
            params = [weights.flatten()]

            if layer.module.bias is not None:
                bias = layer.module.bias.cpu().detach().numpy()
                params = [weights.flatten(), bias.flatten()]

            attr.blobs = params

            if isinstance(layer.module, torch.nn.Conv2d):
                # weight shape : [output_channels, input_channels, kernel_height, kernel_width]
                attr.shape = list(layer.module.weight.size())
                # activation shape : [height, width]
                attr.activation_dims = (layer.output_shape[2], layer.output_shape[3])
                # SVD expects weight matrix of the form [output_channels, input_channels, kernel_height, kernel_width]

            elif isinstance(layer.module, torch.nn.Linear):
                # weight shape : [output_channels, input_channels, 1, 1]
                shape = list(layer.module.weight.size())
                shape.extend([1, 1])
                attr.shape = shape
                attr.activation_dims = [1, 1]

            else:
                # raise error if layer is not Con2D or Linear
                raise AssertionError

            # Save the attributes for this layer
            logger.debug("Storing attributes for layer: %s", format(layer.name))
            svd_lib_ref.StoreLayerAttributes(str(layer.name), attr)

    @staticmethod
    def _get_pymo_layer_type(module: torch.nn.Module):

        if isinstance(module, torch.nn.Conv2d):
            return pymo.LAYER_TYPE_CONV

        if isinstance(module, torch.nn.Linear):
            return pymo.LAYER_TYPE_FC

        raise AssertionError("Unsupported layer_type. Only Linear and Conv2D layers are currently supported")

    @staticmethod
    def _get_pymo_cost_metric(metric: CostMetric):

        if metric == CostMetric.memory:
            return pymo.COST_TYPE_MEMORY

        if metric == CostMetric.mac:
            return pymo.COST_TYPE_MAC

        raise AssertionError('Unknown cost metric')
