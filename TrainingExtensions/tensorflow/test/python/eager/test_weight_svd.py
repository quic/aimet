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

import pytest
import numpy as np

from aimet_common import cost_calculator
from aimet_common.defs import CostMetric
from aimet_tensorflow.keras.layer_database import *
from aimet_tensorflow.keras.svd_spiltter import WeightSvdModuleSplitter
from aimet_tensorflow.keras.svd_pruner import WeightSvdPruner
from aimet_tensorflow.keras.utils import pymo_utils
import aimet_common.libpymo as pymo


def get_model(model_type="Sequential"):
    tf.keras.backend.clear_session()
    if model_type == "Sequential":
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=(2, 2), name='conv1', padding='same', input_shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(64, 5, name='conv2', padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, name='linear')
        ])
    elif model_type == "Functional":
        inp = tf.keras.Input((28, 28, 3))
        x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), name='conv1', padding='same')(inp)
        x = tf.keras.layers.Conv2D(64, 5, name='conv2', padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        out = tf.keras.layers.Dense(10, name = 'linear')(x)

        return tf.keras.Model(inp, out)


def _get_layers(model, model_type="Sequential"):
    # Drop first layer (Input layer) of Functional model
    if model_type == "Functional":
        return model.layers[1:]
    elif model_type == "Sequential":
        return model.layers


class TestWeightSvdLayerSplitandSVDPrunner:

    @pytest.mark.parametrize("model_type", ["Sequential", "Functional"])
    @pytest.mark.parametrize("rank", [12, 20])
    @pytest.mark.parametrize("cost_metric", [CostMetric.mac, CostMetric.memory])
    def test_split_layer(self, model_type, rank, cost_metric):
        """
        test the output after and before the split_module call
        """
        model = get_model(model_type)
        orig_conv_op = _get_layers(model, model_type)[1]

        org_conv_op_shape = orig_conv_op.output_shape

        layer1 = Layer(orig_conv_op, orig_conv_op.name, output_shape=org_conv_op_shape)

        svd_lib_ref = pymo.GetSVDInstance()
        pymo_utils.PymoSvdUtils.configure_layers_in_pymo_svd([layer1], cost_metric, svd_lib_ref, pymo.TYPE_SINGLE)

        split_conv_op1, split_conv_op2 = WeightSvdModuleSplitter.split_module(model, layer1.module, rank, svd_lib_ref)

        split_conv_output = split_conv_op2.output_shape

        assert org_conv_op_shape == split_conv_output

        # check the bias value after split.

        assert len(split_conv_op2.get_weights()) == len(orig_conv_op.get_weights())
        if len(orig_conv_op.get_weights()) > 1:

            orig_bias_out = orig_conv_op.get_weights()[1]
            split_bias_out = split_conv_op2.get_weights()[1]

            assert np.allclose(orig_bias_out, split_bias_out, atol=1e-4)

        # First split conv op should also have bias
        assert len(split_conv_op1.get_weights()) == 2

        # Length of the bias of first conv split should be equal to rank
        assert len(split_conv_op1.get_weights()[1]) == rank

    @pytest.mark.parametrize("model_type", ["Sequential", "Functional"])
    @pytest.mark.parametrize("cost_metric", [CostMetric.mac, CostMetric.memory])
    def test_split_layer_with_stride(self, model_type, cost_metric):
        """
        test the conv2d split after and before split_module call with stride
        """
        model = get_model(model_type)
        orig_conv_op = _get_layers(model, model_type)[0]

        org_conv_op_shape = orig_conv_op.output_shape

        layer1 = Layer(orig_conv_op, orig_conv_op.name, output_shape=org_conv_op_shape)
        rank = cost_calculator.WeightSvdCostCalculator.calculate_rank_given_comp_ratio(layer1, 0.5, cost_metric)

        svd_lib_ref = pymo.GetSVDInstance()
        pymo_utils.PymoSvdUtils.configure_layers_in_pymo_svd([layer1], cost_metric, svd_lib_ref, pymo.TYPE_SINGLE)

        split_conv_op1, split_conv_op2 = WeightSvdModuleSplitter.split_module(model, layer1.module, rank, svd_lib_ref)

        split_conv_output = split_conv_op2.output_shape

        assert org_conv_op_shape == split_conv_output

        # check the bias value after split.

        assert len(split_conv_op2.get_weights()) == len(orig_conv_op.get_weights())
        if len(orig_conv_op.get_weights()) > 1:

            orig_bias_out = orig_conv_op.get_weights()[1]
            split_bias_out = split_conv_op2.get_weights()[1]

            assert np.allclose(orig_bias_out, split_bias_out, atol=1e-4)

        # First split conv op should also have bias
        assert len(split_conv_op1.get_weights()) == 2

        # Length of the bias of first conv split should be equal to rank
        assert len(split_conv_op1.get_weights()[1]) == rank


    @pytest.mark.parametrize("model_type", ["Sequential", "Functional"])
    @pytest.mark.parametrize("cmp_ratio", [0.4, 0.75])
    @pytest.mark.parametrize("cost_metric", [CostMetric.mac, CostMetric.memory])
    @pytest.mark.parametrize("layer_index", [1, 3]) # 2 points to conv and 4 points to FC
    def test_perform_svd_and_split_layer(self, model_type, cmp_ratio, cost_metric, layer_index):

        model = get_model(model_type)
        layer_db = LayerDatabase(model)

        # Copy the db
        comp_layer_db = copy.deepcopy(layer_db)

        layer = comp_layer_db.find_layer_by_name(_get_layers(model, model_type)[layer_index].name)
        org_count = len(list(comp_layer_db._compressible_layers.values()))
        splitter = WeightSvdPruner()

        splitter._prune_layer(layer_db, comp_layer_db, layer, 0.5, cost_metric)

        # orginal layer will be replaced by the two new layers
        assert layer not in list(comp_layer_db._compressible_layers.values())

        after_split_count = len(list(comp_layer_db._compressible_layers.values()))
        assert (org_count + 1) == after_split_count

