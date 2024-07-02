# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Unit tests for Auto Quant Keras """
import contextlib
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Union
from unittest.mock import MagicMock, patch

import pytest
import tensorflow as tf

from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_tensorflow.keras.auto_quant import AutoQuant
from aimet_tensorflow.keras.connectedgraph import ConnectedGraph
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper
from aimet_tensorflow.keras.quantsim import QuantizationSimModel

from aimet_common.utils import AimetLogger
import logging

AimetLogger.set_level_for_all_areas(logging.DEBUG)

def get_tracking_layer(model: tf.keras.Model) -> bool:
    return model.layers[-1]._layer_to_wrap if isinstance(model.layers[-1], QcQuantizeWrapper) else model.layers[-1]

class TrackingLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(TrackingLayer, self).__init__()

        self.applied_bn_folding = self.add_weight('applied_bn_folding',
                                                    dtype=tf.bool,
                                                    initializer=tf.constant_initializer(False),
                                                    trainable=False)

        self.applied_cle = self.add_weight('applied_cle',
                                            dtype=tf.bool,
                                            initializer=tf.constant_initializer(False),
                                            trainable=False)

        self.applied_adaround= self.add_weight('applied_adaround',
                                                dtype=tf.bool,
                                                initializer=tf.constant_initializer(False),
                                                trainable=False)

    def get_config(self):
        return super().get_config()

    def call(self, inputs, *args, **kwargs):
        return inputs

@pytest.fixture(scope="function")
def model():
    inputs = tf.keras.Input(shape=(32, 32, 3,))
    x = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)
    outputs = TrackingLayer()(x)
    functional_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return functional_model


@pytest.fixture(scope="session")
def dataset_length():
    return 2


@pytest.fixture(scope="session")
def unlabeled_dataset(dataset_length):
    dummy_inputs = tf.random.normal((dataset_length, 16, 16, 3))
    dataset = tf.data.Dataset.from_tensor_slices(dummy_inputs)
    dataset = dataset.batch(1)
    return dataset


def assert_applied_techniques(
        output_model, acc, encoding_path,
        target_acc, bn_folded_acc, cle_acc, adaround_acc
):
    tracking_layer = get_tracking_layer(output_model)
    # Batchnorm folding is always applied.
    assert tracking_layer.applied_bn_folding

    # If accuracy is good enough after batchnorm folding
    if bn_folded_acc >= target_acc:
        assert acc == bn_folded_acc
        assert encoding_path.endswith("batchnorm_folding.encodings")
        assert not tracking_layer.applied_cle
        assert not tracking_layer.applied_adaround
        return

    # If accuracy is good enough after cle
    if cle_acc >= target_acc:
        assert acc == cle_acc
        assert encoding_path.endswith("cross_layer_equalization.encodings")
        assert tracking_layer.applied_cle
        assert not tracking_layer.applied_adaround
        return

    # CLE should be applied if and only if it brings accuracy gain
    assert tracking_layer.applied_cle == (bn_folded_acc < cle_acc)

    # If accuracy is good enough after adaround
    if adaround_acc >= target_acc:
        assert acc == adaround_acc
        assert encoding_path.endswith("adaround.encodings")
        assert tracking_layer.applied_adaround
        return

    assert acc == max(bn_folded_acc, cle_acc, adaround_acc)

    if max(bn_folded_acc, cle_acc, adaround_acc) == bn_folded_acc:
        assert encoding_path.endswith("batchnorm_folding.encodings")
    elif max(bn_folded_acc, cle_acc, adaround_acc) == cle_acc:
        assert encoding_path.endswith("cross_layer_equalization.encodings")
    else:
        assert encoding_path.endswith("adaround.encodings")


FP32_ACC = 80.0

def _set_attr_to_copied_model(copied_model: tf.keras.Model,
                              model: tf.keras.Model):
    for key in ["applied_bn_folding", "applied_cle", "applied_adaround"]:
        if hasattr(model, key):
            setattr(copied_model, key, getattr(model, key))


@contextlib.contextmanager
def patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc):
    def bn_folding(model: tf.keras.Model, *_, **__):
        tracking_layer = get_tracking_layer(model)
        tracking_layer.applied_bn_folding.assign(True)

        return model, tuple()

    def cle(model: tf.keras.Model, *_, **__):
        copied_model = tf.keras.models.clone_model(model)

        copied_model.layers[-1].set_weights(model.layers[-1].get_weights())
        tracking_layer = get_tracking_layer(copied_model)
        tracking_layer.applied_bn_folding.assign(True)
        tracking_layer.applied_cle.assign(True)

        return copied_model

    def adaround(model: tf.keras.Model, *_, **__):
        copied_model = tf.keras.models.clone_model(model)

        tracking_layer = get_tracking_layer(copied_model)
        tracking_layer.set_weights(get_tracking_layer(model).get_weights())
        tracking_layer.applied_adaround.assign(True)

        return copied_model

    class _QuantizationSimModel(QuantizationSimModel):
        def __init__(self, model, quant_scheme: Union[QuantScheme, str] = 'tf_enhanced', rounding_mode: str = 'nearest',
                     default_output_bw: int = 8, default_param_bw: int = 8, in_place: bool = False,
                     config_file: str = None, default_data_type: QuantizationDataType = QuantizationDataType.int):

            super(_QuantizationSimModel, self).__init__(model, quant_scheme, rounding_mode, default_output_bw,
                                                        default_param_bw, in_place, config_file, default_data_type)

            self._model_without_wrappers = model
            if not in_place:
                self._model_without_wrappers = tf.keras.models.clone_model(model)
                n_weights = len(self._model_without_wrappers.weights)
                self._model_without_wrappers.set_weights(model.get_weights()[:n_weights])

                for key in ["applied_bn_folding", "applied_cle", "applied_adaround"]:
                    if hasattr(model, key):
                        setattr(self._model_without_wrappers, key, getattr(model, key))

            self._layer_name_to_quant_wrapper = {}
            self._validate_model()
            self.connected_graph = ConnectedGraph(self._model_without_wrappers)
            self._quantsim_configurator = self._initialize_quantsim_configurator(quant_scheme, rounding_mode,
                                                                                 default_output_bw, default_param_bw,
                                                                                 config_file)
            self.quant_scheme = quant_scheme
            self.per_channel_quantization_enabled = self._quantsim_configurator.per_channel_quantization_flag
            self.model = self._add_quantization_wrappers(quant_scheme, rounding_mode, default_output_bw,
                                                         default_param_bw, QuantizationDataType.int)
            self._disable_quantizers_in_folded_batchnorm()

        def compute_encodings(self, *_):
            pass

        def set_and_freeze_param_encodings(self, _):
            pass

        def _add_quantization_wrappers(self, quant_scheme, rounding_mode, default_output_bw,
                                       default_param_bw, default_data_type):
            model = super()._add_quantization_wrappers(quant_scheme, rounding_mode, default_output_bw,
                                                       default_param_bw, default_data_type)
            _set_attr_to_copied_model(model, self._model_without_wrappers)
            return model

    def mock_eval_callback(model, _):
        tracking_layer = get_tracking_layer(model)
        if tracking_layer.applied_adaround:
            return adaround_acc
        if tracking_layer.applied_cle:
            return cle_acc
        if tracking_layer.applied_bn_folding:
            return bn_folded_acc

        return FP32_ACC

    @dataclass
    class Mocks:
        eval_callback: Callable
        QuantizationSimModel: MagicMock
        fold_all_batch_norms: MagicMock
        equalize_model: MagicMock
        apply_adaround: MagicMock

    with patch("aimet_tensorflow.keras.auto_quant.QuantizationSimModel", side_effect=_QuantizationSimModel) as mock_qsim, \
            patch("aimet_tensorflow.keras.auto_quant.AutoQuant._apply_batchnorm_folding", side_effect=bn_folding) as mock_bn_folding, \
            patch("aimet_tensorflow.keras.auto_quant.equalize_model", side_effect=cle) as mock_cle, \
            patch("aimet_tensorflow.keras.auto_quant.Adaround.apply_adaround", side_effect=adaround) as mock_adaround:
        try:
            yield Mocks(eval_callback=mock_eval_callback,
                        QuantizationSimModel=mock_qsim,
                        fold_all_batch_norms=mock_bn_folding,
                        equalize_model=mock_cle,
                        apply_adaround=mock_adaround)
        finally:
            pass


@pytest.fixture(autouse=True)
def patch_dependencies():
    def render(*_, **__):
        return ""

    with patch("aimet_tensorflow.keras.auto_quant.jinja2.environment.Template.render", side_effect=render):
        yield


class TestAutoQuant:
    @pytest.mark.parametrize(
        "bn_folded_acc, cle_acc, adaround_acc",
        [(50., 60., 70.), (50., 70., 60.), (70., 50., 60.)]
    )
    @pytest.mark.parametrize("allowed_accuracy_drop", [5., 15.])
    def test_auto_quant(self, model, unlabeled_dataset, allowed_accuracy_drop,
                        bn_folded_acc, cle_acc, adaround_acc):
        with patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc) as mocks:
            auto_quant = AutoQuant(allowed_accuracy_drop=allowed_accuracy_drop,
                                   eval_callback=mocks.eval_callback,
                                   unlabeled_dataset=unlabeled_dataset)
            self._do_test_apply_auto_quant(auto_quant, model, allowed_accuracy_drop,
                                           bn_folded_acc, cle_acc, adaround_acc)

    @staticmethod
    def _do_test_apply_auto_quant(auto_quant, model, allowed_accuracy_drop,
                                  bn_folded_acc, cle_acc, adaround_acc):
        with tempfile.TemporaryDirectory() as results_dir:
            target_acc = FP32_ACC - allowed_accuracy_drop

            output_model, acc, encoding_path = auto_quant.apply(model,
                                                                results_dir=results_dir,
                                                                custom_objects={'TrackingLayer': TrackingLayer})
            assert_applied_techniques(output_model, acc, encoding_path, target_acc, bn_folded_acc, cle_acc, adaround_acc)

    def test_auto_quant_invalid_input(self):
        # Allowed accuracy drop < 0
        with pytest.raises(ValueError):
            _ = AutoQuant(-1.0, MagicMock(), MagicMock(), MagicMock())

        # Bitwidth < 4 or bitwidth > 32
        with pytest.raises(ValueError):
            _ = AutoQuant(0, MagicMock(), MagicMock(), default_param_bw=2)

        with pytest.raises(ValueError):
            _ = AutoQuant(0, MagicMock(), MagicMock(), default_param_bw=64)

        with pytest.raises(ValueError):
            _ = AutoQuant(0, MagicMock(), MagicMock(), default_output_bw=2)

        with pytest.raises(ValueError):
            _ = AutoQuant(0, MagicMock(), MagicMock(), default_output_bw=64)

    def test_auto_quant_caching(self, model, unlabeled_dataset):
        allowed_accuracy_drop = 0.0
        bn_folded_acc, cle_acc, adaround_acc = 40., 50., 60.
        with patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc) as mocks:
            auto_quant = AutoQuant(allowed_accuracy_drop=allowed_accuracy_drop,
                                   eval_callback=mocks.eval_callback,
                                   unlabeled_dataset=unlabeled_dataset)

            with tempfile.TemporaryDirectory() as results_dir:
                cache_id = "unittest"
                cache_files = [
                    os.path.join(results_dir, ".auto_quant_cache", cache_id, key)
                    for key in ("cle", "adaround")
                ]

                # No previously cached results
                auto_quant.apply(model, results_dir=results_dir, cache_id=cache_id, custom_objects={'TrackingLayer': TrackingLayer})

                for cache_file in cache_files:
                    assert os.path.exists(cache_file)

                assert mocks.equalize_model.call_count == 1
                assert mocks.apply_adaround.call_count == 1

                # Load cached result
                auto_quant.apply(model, results_dir=results_dir, cache_id=cache_id, custom_objects={'TrackingLayer': TrackingLayer})

                assert mocks.equalize_model.call_count == 1
                assert mocks.apply_adaround.call_count == 1
