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
from dataclasses import dataclass
import os
import tempfile
from typing import Callable, Union
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup
import pytest
import tensorflow as tf
import itertools
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_tensorflow.keras.auto_quant_v2 import AutoQuant
from aimet_tensorflow.keras.auto_quant_v2 import PtqResult
from aimet_tensorflow.keras.connectedgraph import ConnectedGraph
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.adaround_weight import AdaroundParameters
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper
from aimet_common.utils import AimetLogger
import logging

AimetLogger.set_level_for_all_areas(logging.DEBUG)

@pytest.fixture(scope="function")
def model():
    inputs = tf.keras.Input(shape=(32, 32, 3,))
    x = tf.keras.layers.Conv2D(32, (3, 3))(inputs)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    functional_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    functional_model.__setattr__("applied_bn_folding",False)
    functional_model.__setattr__("applied_cle",False)
    functional_model.__setattr__("applied_adaround",False)

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

def assert_html(html_parsed, properties):
    for id_, prop in properties.items():
        elem = html_parsed.find(id=id_)
        assert elem is not None
        for prop_name, prop_val in prop.items():
            if prop_val is None:
                assert prop_name not in elem.attrs
            else:
                assert elem[prop_name] == prop_val


_VISITED = { 'data-visited': 'true', }
_NOT_VISITED = { 'data-visited': None, }
_SUCCESS = {
    'data-visited': 'true',
    'data-stage-result': 'success'
}
_DISCARDED = {
    'data-visited': 'true',
    'data-stage-result': 'discarded'
}
_ERROR_IGNORED = {
    'data-visited': 'true',
    'data-stage-result': 'error-ignored'
}
_ERROR_FAILED = {
    'data-visited': 'true',
    'data-stage-result': 'error-failed'
}

def assert_applied_techniques(
        output_model, acc, encoding_path,
        target_acc, bn_folded_acc, cle_acc, adaround_acc, results_dir,
):
    html_path = os.path.join(results_dir, 'diagnostics.html')
    with open(html_path) as f:
        html_parsed = BeautifulSoup(f.read(), features="html.parser")

    # Batchnorm folding is always applied.
    assert output_model.applied_bn_folding
    assert_html(html_parsed, {
        'node_batchnorm_folding': _SUCCESS,
        'node_test_batchnorm_folding': _VISITED,
    })

    # If accuracy is good enough after batchnorm folding
    if bn_folded_acc >= target_acc:
        assert acc == bn_folded_acc
        assert encoding_path.endswith("batchnorm_folding.encodings")
        assert not output_model.applied_cle
        assert not output_model.applied_adaround

        assert_html(html_parsed, {
            'node_cross_layer_equalization': _NOT_VISITED,
            'node_test_cross_layer_equalization': _NOT_VISITED,
            'node_adaround': _NOT_VISITED,
            'node_test_adaround': _NOT_VISITED,
            'node_result_fail': _NOT_VISITED,
            'node_result_success': _VISITED,
        })
        return

    # If accuracy is good enough after cle
    if cle_acc >= target_acc:
        assert acc == cle_acc
        assert encoding_path.endswith("cross_layer_equalization.encodings")
        assert output_model.applied_cle
        assert not output_model.applied_adaround

        assert_html(html_parsed, {
            'node_adaround': _NOT_VISITED,
            'node_test_adaround': _NOT_VISITED,
            'node_result_fail': _NOT_VISITED,
            'node_result_success': _VISITED,
        })
        return

    assert output_model.applied_adaround == (adaround_acc >= max(bn_folded_acc, cle_acc))

    assert_html(html_parsed, {
        'node_adaround': _SUCCESS if output_model.applied_adaround else _DISCARDED,
        'node_test_adaround': _VISITED,
    })

    # If accuracy is good enough after adaround
    if adaround_acc >= target_acc:
        assert acc == adaround_acc
        assert encoding_path.endswith("adaround.encodings")
        assert output_model.applied_adaround

        assert_html(html_parsed, {
            'node_result_fail': _NOT_VISITED,
            'node_result_success': _VISITED,
        })
        return
    assert_html(html_parsed, {
        'node_result_fail': _VISITED,
        'node_result_success': _NOT_VISITED,
    })
    assert acc == max(bn_folded_acc, cle_acc, adaround_acc)

    if max(bn_folded_acc, cle_acc, adaround_acc) == bn_folded_acc:
        assert encoding_path.endswith("batchnorm_folding.encodings")
    elif max(bn_folded_acc, cle_acc, adaround_acc) == cle_acc:
        assert encoding_path.endswith("cross_layer_equalization.encodings")
    else:
        assert encoding_path.endswith("adaround.encodings")


FP32_ACC = .8
W32_ACC = 0.8 # Assume W32 accuracy is equal to FP32 accuracy
RAW_QUANTSIM_ACC = 0.1

def _set_attr_to_copied_model(copied_model: tf.keras.Model,
                              model: tf.keras.Model):
    for key in ["applied_bn_folding", "applied_cle", "applied_adaround"]:
        if hasattr(model, key):
            setattr(copied_model, key, getattr(model, key))
        else:
            setattr(copied_model, key, False)


@contextlib.contextmanager
def patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc,fp32_acc=None, w32_acc=None, raw_quantsim_acc=None):
    if fp32_acc is None:
        fp32_acc = FP32_ACC

    if w32_acc is None:
        w32_acc = W32_ACC

    if raw_quantsim_acc is None:
        raw_quantsim_acc = RAW_QUANTSIM_ACC

    def bn_folding(model: tf.keras.Model, *_, **__):
        copied_model = tf.keras.models.clone_model(model)
        _set_attr_to_copied_model(copied_model, model)
        copied_model.applied_bn_folding = True
        return copied_model, tuple()

    def cle(model: tf.keras.Model, *_, **__):
        copied_model = tf.keras.models.clone_model(model)
        _set_attr_to_copied_model(copied_model, model)

        copied_model.applied_bn_folding = True
        copied_model.applied_cle = True
        return copied_model

    def adaround(model: tf.keras.Model, *_, **__):
        copied_model = tf.keras.models.clone_model(model)
        _set_attr_to_copied_model(copied_model, model)

        copied_model.applied_adaround = True
        return copied_model

    class _PtqResult(PtqResult):

        def load_model(self) -> tf.keras.Model:
            model = super().load_model()
            bnf_val = True if "batchnorm_folding" in self.applied_techniques else 0
            cle_val = True if "cross_layer_equalization" in self.applied_techniques else 0
            if cle_val:
                bnf_val = True
            ada_val = True if "adaround" in self.applied_techniques else 0
            model.__setattr__("applied_bn_folding",bnf_val)
            model.__setattr__("applied_cle",cle_val)
            model.__setattr__("applied_adaround",ada_val)
            return model


    class _QuantizationSimModel(QuantizationSimModel):
        def __init__(self, model, quant_scheme: Union[QuantScheme, str] = 'tf_enhanced', rounding_mode: str = 'nearest',
                     default_output_bw: int = 8, default_param_bw: int = 8, in_place: bool = False,
                     config_file: str = None, default_data_type: QuantizationDataType = QuantizationDataType.int):

            super(_QuantizationSimModel, self).__init__(model, quant_scheme, rounding_mode, default_output_bw,
                                                        default_param_bw, in_place, config_file, default_data_type)

            self._model_without_wrappers = model
            if not in_place:
                self._model_without_wrappers = tf.keras.models.clone_model(model)
                self._model_without_wrappers.set_weights(model.get_weights()[:4])

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

        if not isinstance(model.layers[1], QcQuantizeWrapper):
            # Not quantized: return fp32 accuracy
            return fp32_acc
        if model.layers[1].param_quantizers[0].bitwidth == 32:
            # W32 evaluation for early exit. Return W32 accuracy
            return w32_acc
        acc = raw_quantsim_acc
        if model.applied_bn_folding:
            acc = bn_folded_acc
        if model.applied_cle:
            acc = cle_acc
        if model.applied_adaround:
            acc = adaround_acc
        return acc

    @dataclass
    class Mocks:
        eval_callback: Callable
        QuantizationSimModel: MagicMock
        PtqResult: MagicMock
        fold_all_batch_norms: MagicMock
        equalize_model: MagicMock
        apply_adaround: MagicMock

    with patch("aimet_tensorflow.keras.auto_quant_v2.QuantizationSimModel", side_effect=_QuantizationSimModel) as mock_qsim, \
            patch("aimet_tensorflow.keras.auto_quant_v2.PtqResult", side_effect=_PtqResult) as mock_ptq, \
            patch("aimet_tensorflow.keras.auto_quant_v2.AutoQuant._apply_batchnorm_folding", side_effect=bn_folding) as mock_bn_folding, \
            patch("aimet_tensorflow.keras.auto_quant_v2.AutoQuant._apply_cross_layer_equalization", side_effect=cle) as mock_cle, \
            patch("aimet_tensorflow.keras.auto_quant_v2.Adaround.apply_adaround", side_effect=adaround) as mock_adaround:
        try:
            yield Mocks(eval_callback=mock_eval_callback,
                        QuantizationSimModel=mock_qsim,
                        PtqResult=mock_ptq,
                        fold_all_batch_norms=mock_bn_folding,
                        equalize_model=mock_cle,
                        apply_adaround=mock_adaround)
        finally:
            pass

class TestAutoQuant:
    def test_auto_quant_run_inference(self, model, unlabeled_dataset):
        bn_folded_acc = .5
        with patch_ptq_techniques(bn_folded_acc, None, None) as mocks:
            with tempfile.TemporaryDirectory() as results_dir:
                auto_quant = AutoQuant(model = model,
                                       eval_callback=mocks.eval_callback,
                                       dataset = unlabeled_dataset,
                                       results_dir=results_dir)
                auto_quant.run_inference()

    @pytest.mark.parametrize(
        "bn_folded_acc, cle_acc, adaround_acc",
        [(0.50, 0.60, 0.70), (0.50, 0.70, 0.60), (0.70, 0.50, 0.60)]
    )

    def _test_auto_quant(
            self, model, unlabeled_dataset,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        with patch_ptq_techniques(
                bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            with tempfile.TemporaryDirectory() as results_dir:
                auto_quant = AutoQuant(model = model,
                                       eval_callback = mocks.eval_callback,
                                       dataset = unlabeled_dataset,
                                       results_dir=results_dir)
                self._do_test_optimize_auto_quant(
                    auto_quant, allowed_accuracy_drop,
                    bn_folded_acc, cle_acc, adaround_acc
                )

    def _do_test_optimize_auto_quant(self, auto_quant,allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,):
        target_acc = FP32_ACC - allowed_accuracy_drop
        output_model, acc, encoding_path = auto_quant.optimize(allowed_accuracy_drop)
        assert_applied_techniques(
            output_model, acc, encoding_path,
            target_acc, bn_folded_acc, cle_acc, adaround_acc,
            auto_quant.results_dir,
        )

    def test_auto_quant_invalid_input(self,model, unlabeled_dataset):

        with pytest.raises(ValueError):
            AutoQuant(None, lambda: None, unlabeled_dataset)
        with pytest.raises(ValueError):
            AutoQuant(model, None, unlabeled_dataset)
        with pytest.raises(ValueError):
            AutoQuant(model, lambda: None, None)
        with pytest.raises(ValueError):
            AutoQuant(model, lambda: None, unlabeled_dataset, results_dir=None)
        with pytest.raises(ValueError):
            AutoQuant(model, lambda: None, unlabeled_dataset, strict_validation=None)

        with pytest.raises(ValueError):
            AutoQuant(model, lambda: None, unlabeled_dataset, param_bw=2)
        with pytest.raises(ValueError):
            AutoQuant(model, lambda: None, unlabeled_dataset, param_bw=64)
        with pytest.raises(ValueError):
            AutoQuant(model, lambda: None, unlabeled_dataset, output_bw=2)
        with pytest.raises(ValueError):
            AutoQuant(model, lambda: None, unlabeled_dataset, output_bw=64)

        auto_quant = AutoQuant(model, lambda: None, unlabeled_dataset)
        # Allowed accuracy drop < 0
        with pytest.raises(ValueError):
            _ = auto_quant.optimize(-1.0)

    def test_auto_quant_early_exit(self, model, unlabeled_dataset):
        allowed_accuracy_drop = 0.1
        w32_acc = FP32_ACC - (allowed_accuracy_drop * 2)

        with tempfile.TemporaryDirectory() as results_dir:
            with patch_ptq_techniques(
                    bn_folded_acc=0, cle_acc=0, adaround_acc=0, w32_acc=w32_acc
            ) as mocks:
                auto_quant = AutoQuant(model = model,
                          eval_callback = mocks.eval_callback,
                          dataset = unlabeled_dataset,
                          results_dir=results_dir)
                output_model, acc, encoding_path = auto_quant.optimize(allowed_accuracy_drop)


            assert output_model is None
            assert acc is None
            assert encoding_path is None

            with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                html_parsed = BeautifulSoup(f.read(), features="html.parser")
                assert_html(html_parsed, {
                    'node_test_w32_eval_score': _VISITED,
                    'node_batchnorm_folding': _NOT_VISITED,
                    'node_cross_layer_equalization': _NOT_VISITED,
                    'node_adaround': _NOT_VISITED,
                    'node_result_fail': _VISITED,
                })

    @pytest.mark.parametrize(
        "bn_folded_acc, cle_acc, adaround_acc",
        itertools.permutations([.5, .6, .7])
    )
    @pytest.mark.parametrize("allowed_accuracy_drop", [.05, .15])
    def test_auto_quant_cpu(
            self, model, unlabeled_dataset,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):

        self._test_auto_quant(
            model, unlabeled_dataset,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
        )


    def test_auto_quant_scheme_selection(
            self, model, unlabeled_dataset,
    ):
        allowed_accuracy_drop = 0.0
        bn_folded_acc, cle_acc, adaround_acc = .4, .5, .6
        with patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc) as mocks:
            def eval_callback(model, _):
                # Assumes the model's eval score drops to zero
                # unless param_quant_scheme == tfe and output_quant_scheme == tf
                if isinstance(model.layers[1], QcQuantizeWrapper):
                    if model.layers[1].param_quantizers[0].quant_scheme != QuantScheme.post_training_tf_enhanced:
                        return 0.0
                    if model.layers[1].output_quantizers[0].quant_scheme != QuantScheme.post_training_tf:
                        return 0.0
                return mocks.eval_callback(model, _)

            _optimize = AutoQuant.optimize
            def optimize(self, *args, **kwargs):
                # Since all the other candidates (tf-tf, tfe-tfe, and tfe-percentile) yields zero accuracy,
                # it is expected that tf-tfe is selected as the quant scheme for AutoQuant.
                ret = _optimize(self, *args, **kwargs)
                assert self._quantsim_params["quant_scheme"].param_quant_scheme == QuantScheme.post_training_tf_enhanced
                assert self._quantsim_params["quant_scheme"].output_quant_scheme == QuantScheme.post_training_tf
                return ret

            with patch("aimet_tensorflow.keras.auto_quant_v2.AutoQuant.optimize", optimize):
                auto_quant = AutoQuant(model = model,
                                       eval_callback = eval_callback,
                                       dataset = unlabeled_dataset)
                auto_quant.optimize(allowed_accuracy_drop)

    def test_set_additional_params(self, model, unlabeled_dataset):
        allowed_accuracy_drop = 0
        bn_folded_acc = .1
        cle_acc = .2
        adaround_acc = .3
        with patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc) as mocks:
            auto_quant = AutoQuant(model = model,
                      eval_callback = mocks.eval_callback,
                                   dataset = unlabeled_dataset)
            adaround_params = AdaroundParameters(unlabeled_dataset, 1)
            auto_quant.set_adaround_params(adaround_params)
            self._do_test_optimize_auto_quant(
                auto_quant, allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc
            )
            adaround_args, _ = mocks.apply_adaround.call_args
            _, actual_adaround_params = adaround_args
            assert adaround_params == actual_adaround_params
