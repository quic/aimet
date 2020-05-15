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

""" Prunes layers using Spatial SVD schemes """

from aimet_common.utils import AimetLogger
import aimet_common.svd_pruner

from aimet_tensorflow.utils.op.conv import get_output_activation_shape
from aimet_tensorflow.layer_database import LayerDatabase, Layer
from aimet_tensorflow.svd_spiltter import SpatialSvdModuleSplitter

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class SpatialSvdPruner(aimet_common.svd_pruner.SpatialSvdPruner):
    """
    Pruner for Spatial-SVD method
    """

    def _perform_svd_and_split_layer(self, layer: Layer, rank: int, comp_layer_db: LayerDatabase):

        """
        Performs spatial svd and splits given layer into two layers
        :param layer: Layer to split
        :param rank: Rank to use for spatial svd splitting
        :param comp_layer_db: Compressed layer db to update with the split layers
        :return: None
        """

        # Split module using Spatial SVD
        module_a, module_b = SpatialSvdModuleSplitter.split_module(layer, rank)

        # get the output activation shape for first conv op
        output_shape_a = get_output_activation_shape(sess=layer.model, op=module_a,
                                                     input_op_names=comp_layer_db.starting_ops,
                                                     input_shape=comp_layer_db.input_shape)

        # get the output activation shape for second conv op
        output_shape_b = get_output_activation_shape(sess=layer.model, op=module_b,
                                                     input_op_names=comp_layer_db.starting_ops,
                                                     input_shape=comp_layer_db.input_shape)

        # Create two new layers and return them
        layer_a = Layer(model=layer.model, op=module_a, output_shape=output_shape_a)
        layer_b = Layer(model=layer.model, op=module_b, output_shape=output_shape_b)

        comp_layer_db.replace_layer_with_sequential_of_two_layers(layer, layer_a, layer_b)
