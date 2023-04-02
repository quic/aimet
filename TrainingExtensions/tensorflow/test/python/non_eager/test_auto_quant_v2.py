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

import contextlib
from dataclasses import dataclass
import itertools
from unittest.mock import patch, MagicMock
import os

import numpy as np
from bs4 import BeautifulSoup

from aimet_tensorflow.utils.common import deepcopy_tf_session
import pytest
import shutil
from typing import Callable, cast
import tensorflow as tf
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.auto_quant_v2 import AutoQuant, PtqResult, _QuantSchemePair
from aimet_tensorflow.adaround.adaround_weight import AdaroundParameters
from aimet_tensorflow.quantsim import QuantizationSimModel


tf.compat.v1.disable_eager_execution()


BNF = 'bnf'
CLE = 'cle'
ADA = 'ada'


def session(device):
    with tf.device(device):
        graph = tf.Graph()
        with graph.as_default():
            model = tf.keras.Sequential((
                tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'),
                tf.keras.layers.Conv2D(64, kernel_size=3),
            ))

            init = tf.compat.v1.global_variables_initializer()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.compat.v1.Session(graph=graph, config=config)
    sess.run(init)

    sess.__setattr__(BNF, 0)
    sess.__setattr__(CLE, 0)
    sess.__setattr__(ADA, 0)

    return sess

@pytest.fixture(scope="session")
def sess():
    return session('/cpu:0')


@pytest.fixture(scope="session")
def gpu_session():
    return session('/gpu:0')


@pytest.fixture(scope="session")
def unlabeled_data_loader():
    dummy_inputs = tf.random.normal((4, 28, 28, 3))
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices(dummy_inputs)
    dataset = dataset.batch(2)
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
        target_acc, bn_folded_acc, cle_acc, adaround_acc,
        results_dir,
):
    html_path = os.path.join(results_dir, 'diagnostics.html')
    with open(html_path) as f:
        html_parsed = BeautifulSoup(f.read(), features="html.parser")

    # Batchnorm folding is always applied.
    assert output_model.__getattribute__(BNF)
    assert_html(html_parsed, {
        'node_batchnorm_folding': _SUCCESS,
        'node_test_batchnorm_folding': _VISITED,
    })

    # If accuracy is good enough after batchnorm folding
    if bn_folded_acc >= target_acc:
        assert acc == bn_folded_acc
        assert encoding_path.endswith("batchnorm_folding.encodings")
        assert not output_model.__getattribute__(CLE)
        assert not output_model.__getattribute__(ADA)

        assert_html(html_parsed, {
            'node_cross_layer_equalization': _NOT_VISITED,
            'node_test_cross_layer_equalization': _NOT_VISITED,
            'node_adaround': _NOT_VISITED,
            'node_test_adaround': _NOT_VISITED,
            'node_result_fail': _NOT_VISITED,
            'node_result_success': _VISITED,
        })
        return

    # CLE should be applied if and only if it brings accuracy gain
    assert output_model.__getattribute__(CLE) == (bn_folded_acc < cle_acc)

    assert_html(html_parsed, {
        'node_cross_layer_equalization': _SUCCESS if output_model.__getattribute__(CLE) else _DISCARDED,
        'node_test_cross_layer_equalization': _VISITED,
    })

    # If accuracy is good enough after cle
    if cle_acc >= target_acc:
        assert acc == cle_acc
        assert encoding_path.endswith("cross_layer_equalization.encodings")
        assert output_model.__getattribute__(CLE)
        assert not output_model.__getattribute__(ADA)

        assert_html(html_parsed, {
            'node_adaround': _NOT_VISITED,
            'node_test_adaround': _NOT_VISITED,
            'node_result_fail': _NOT_VISITED,
            'node_result_success': _VISITED,
        })
        return

    assert output_model.__getattribute__(ADA) == (adaround_acc >= max(bn_folded_acc, cle_acc))

    assert_html(html_parsed, {
        'node_adaround': _SUCCESS if output_model.__getattribute__(ADA) else _DISCARDED,
        'node_test_adaround': _VISITED,
    })

    # If accuracy is good enough after adaround
    if adaround_acc >= target_acc:
        assert acc == adaround_acc
        assert encoding_path.endswith("adaround.encodings")
        assert output_model.__getattribute__(ADA)

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

def assert_same_device(graph_a: tf.Graph, graph_b: tf.Graph):
    ops_a = graph_a.get_operations()
    ops_b = graph_b.get_operations()
    for op_a, op_b in zip(ops_a, ops_b):
        assert op_a.device == op_b.device

    with graph_a.as_default():
        vars_a = tf.compat.v1.global_variables()

    with graph_b.as_default():
        vars_b = tf.compat.v1.global_variables()

    for var_a, var_b in zip(vars_a, vars_b):
        assert var_a.device == var_b.device


FP32_ACC = .8
W32_ACC = FP32_ACC # Assume W32 accuracy is equal to FP32 accuracy
RAW_QUANTSIM_ACC = 0.1


@contextlib.contextmanager
def patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc, fp32_acc=None, w32_acc=None, raw_quantsim_acc=None):
    if fp32_acc is None:
        fp32_acc = FP32_ACC

    if w32_acc is None:
        w32_acc = W32_ACC

    if raw_quantsim_acc is None:
        raw_quantsim_acc = RAW_QUANTSIM_ACC


    def bn_folding(model, *_, **__):
        model.__setattr__(BNF, 1)
        return model, _

    def cle(model, *_, **__):
        model.__setattr__(BNF, 1)
        model.__setattr__(CLE, 1)
        return model

    def adaround(model, *_, **__):
        model.__setattr__(ADA, 1)
        return model

    def tf_deep_copy_session(model: tf.compat.v1.Session, *_, **__):
        bnf_val = model.__getattribute__(BNF) if hasattr(model, BNF) else 0
        cle_val = model.__getattribute__(CLE) if hasattr(model, CLE) else 0
        ada_val = model.__getattribute__(ADA) if hasattr(model, ADA) else 0

        new_model = deepcopy_tf_session(model)

        new_model.__setattr__(BNF, bnf_val)
        new_model.__setattr__(CLE, cle_val)
        new_model.__setattr__(ADA, ada_val)

        return new_model.as_default()

    class _QuantizationSimModel(QuantizationSimModel):
        def compute_encodings(self, *_):
            pass

        def set_and_freeze_param_encodings(self, _):
            pass

        def _save_and_load_sim_model(self, *_):
            # First copy the present attribute
            bnf_val = self.session.__getattribute__(BNF) if hasattr(self.session, BNF) else 0
            cle_val = self.session.__getattribute__(CLE) if hasattr(self.session, CLE) else 0
            ada_val = self.session.__getattribute__(ADA) if hasattr(self.session, ADA) else 0

            super()._save_and_load_sim_model()

            self.session.__setattr__(BNF, bnf_val)
            self.session.__setattr__(CLE, cle_val)
            self.session.__setattr__(ADA, ada_val)

    class _PtqResult(PtqResult):
        def load_model(self) -> tf.compat.v1.Session:
            sess = super().load_model()
            bnf_val = 1 if "batchnorm_folding" in self.applied_techniques else 0
            cle_val = 1 if "cross_layer_equalization" in self.applied_techniques else 0
            if cle_val:
                bnf_val = 1
            ada_val = 1 if "adaround" in self.applied_techniques else 0

            sess.__setattr__(BNF, bnf_val)
            sess.__setattr__(CLE, cle_val)
            sess.__setattr__(ADA, ada_val)

            return sess

    def mock_eval_callback(model: tf.compat.v1.Session, _):
        try:
            qc_op = model.graph.get_operation_by_name("conv2d_1/BiasAdd_quantized")
        except KeyError:
            # Quantize op not present so model is unquantized
            return fp32_acc

        param_quantize_op = model.graph.get_operation_by_name('conv2d/Conv2D/ReadVariableOp_quantized')
        op_var_tensor = param_quantize_op.inputs[5]
        param_bw = model.run(op_var_tensor)

        if param_bw == 32:
            # W32 evaluation for early exit. Return W32 accuracy
            return w32_acc

        acc = raw_quantsim_acc

        if hasattr(model, BNF) and  model.__getattribute__(BNF):
            acc = bn_folded_acc

        if hasattr(model, CLE) and  model.__getattribute__(CLE):
            acc = cle_acc

        if hasattr(model, ADA) and model.__getattribute__(ADA):
            acc = adaround_acc

        return acc

    @dataclass
    class Mocks:
        eval_callback: Callable
        QuantizationSimModel: MagicMock
        fold_all_batch_norms: MagicMock
        equalize_model: MagicMock
        apply_adaround: MagicMock
        deepcopy_tf_session: MagicMock
        ptq_result: MagicMock

    with patch("aimet_tensorflow.auto_quant_v2.QuantizationSimModel", side_effect=_QuantizationSimModel) as mock_qsim, \
            patch("aimet_tensorflow.auto_quant_v2.fold_all_batch_norms", side_effect=bn_folding) as mock_bn_folding, \
            patch("aimet_tensorflow.auto_quant_v2.equalize_model", side_effect=cle) as mock_cle, \
            patch("aimet_tensorflow.auto_quant_v2.deepcopy_tf_session", side_effect=tf_deep_copy_session) as mock_session_deepcopy, \
            patch("aimet_tensorflow.auto_quant_v2.Adaround.apply_adaround", side_effect=adaround) as mock_adaround, \
            patch("aimet_tensorflow.auto_quant_v2.PtqResult", side_effect=_PtqResult) as mock_ptq_result:
        try:
            yield Mocks(
                eval_callback=mock_eval_callback,
                QuantizationSimModel=mock_qsim,
                fold_all_batch_norms=mock_bn_folding,
                equalize_model=mock_cle,
                apply_adaround=mock_adaround,
                deepcopy_tf_session=mock_session_deepcopy,
                ptq_result=mock_ptq_result,
            )
        finally:
            pass


starting_ops = ['conv2d_input']
ending_ops =['conv2d_1/BiasAdd']


class TestAutoQuant:

    def test_auto_quant_run_inference(self, sess, unlabeled_data_loader):
        bn_folded_acc = .5

        with patch_ptq_techniques(
                bn_folded_acc, None, None
        ) as mocks:
            with create_tmp_directory() as results_dir:
                auto_quant = AutoQuant(sess,
                                       starting_ops,
                                       ending_ops,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir)
                auto_quant.run_inference()

    @pytest.mark.parametrize(
        "bn_folded_acc, cle_acc, adaround_acc",
        itertools.permutations([.5, .6, .7])
    )
    @pytest.mark.parametrize("allowed_accuracy_drop", [.05, .15])
    def test_auto_quant_cpu(
            self, sess, unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        self._test_auto_quant(
            sess, unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
        )

    @pytest.mark.cuda
    def test_auto_quant_gpu(self, gpu_session, dummy_input, unlabeled_data_loader):
        bn_folded_acc, cle_acc, adaround_acc = .5, .6, .7
        allowed_accuracy_drop = .15

        self._test_auto_quant(
            gpu_session, dummy_input.cuda(), unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
        )

    def test_consecutive_calls(self, sess, unlabeled_data_loader):
        bn_folded_acc, cle_acc, adaround_acc = .5, .6, .7

        with patch_ptq_techniques(
                bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            with create_tmp_directory() as results_dir:
                auto_quant = AutoQuant(sess,
                                       starting_ops,
                                       ending_ops,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir)

                # Should return proper model & summary report
                # regardless of consecutive calls
                for allowed_accuracy_drop in (.5, .4, .3, .2, .1, .05):
                    self._do_test_optimize_auto_quant(
                        auto_quant, sess,
                        allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc
                    )

        with patch_ptq_techniques(
                bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            with create_tmp_directory() as results_dir:
                auto_quant = AutoQuant(sess,
                                       starting_ops,
                                       ending_ops,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir)

                # When run_inference() and optimize() are called in back-to-back,
                # Currently we are relying on the default cache.mark for caching the results.

                auto_quant.run_inference()
                auto_quant.optimize()
                assert mocks.fold_all_batch_norms.call_count == 2
                assert mocks.equalize_model.call_count == 1

                auto_quant.optimize()
                assert mocks.fold_all_batch_norms.call_count == 3
                assert mocks.equalize_model.call_count == 2

                self._do_test_optimize_auto_quant(
                    auto_quant, sess,
                    0.0, bn_folded_acc, cle_acc, adaround_acc
                )
                assert mocks.fold_all_batch_norms.call_count == 4
                assert mocks.equalize_model.call_count == 3

    def _test_auto_quant(
            self, sess, unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        with patch_ptq_techniques(
                bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            with create_tmp_directory() as results_dir:
                auto_quant = AutoQuant(sess,
                                       starting_ops,
                                       ending_ops,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir)
                self._do_test_optimize_auto_quant(
                    auto_quant, sess, allowed_accuracy_drop,
                    bn_folded_acc, cle_acc, adaround_acc
                )

    def _do_test_optimize_auto_quant(
            self, auto_quant, input_model,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        target_acc = FP32_ACC - allowed_accuracy_drop

        output_model, acc, encoding_path = auto_quant.optimize(allowed_accuracy_drop)

        assert_same_device(output_model.graph, input_model.graph)
        assert_applied_techniques(
            output_model, acc, encoding_path,
            target_acc, bn_folded_acc, cle_acc, adaround_acc,
            auto_quant.results_dir,
        )

    def test_auto_quant_invalid_input(self, sess, unlabeled_data_loader):
        with pytest.raises(ValueError):
            AutoQuant(None,  starting_ops, ending_ops, unlabeled_data_loader, lambda: None)

        with pytest.raises(ValueError):
            AutoQuant(sess, None, None, unlabeled_data_loader, lambda: None)

        with pytest.raises(ValueError):
            AutoQuant(sess,  starting_ops, ending_ops, None, lambda: None)

        with pytest.raises(ValueError):
            AutoQuant(sess,  starting_ops, ending_ops, unlabeled_data_loader, None)

        with pytest.raises(ValueError):
            AutoQuant(sess,  starting_ops, ending_ops, unlabeled_data_loader, lambda: None, results_dir=None)

        with pytest.raises(ValueError):
            AutoQuant(sess,  starting_ops, ending_ops, unlabeled_data_loader, lambda: None, strict_validation=None)

        # Bitwidth < 4 or bitwidth > 32
        with pytest.raises(ValueError):
            AutoQuant(sess,  starting_ops, ending_ops, unlabeled_data_loader, lambda: None, param_bw=2)

        with pytest.raises(ValueError):
            AutoQuant(sess,  starting_ops, ending_ops, unlabeled_data_loader, lambda: None, param_bw=64)

        with pytest.raises(ValueError):
            AutoQuant(sess,  starting_ops, ending_ops, unlabeled_data_loader, lambda: None, output_bw=2)

        with pytest.raises(ValueError):
            AutoQuant(sess,  starting_ops, ending_ops, unlabeled_data_loader, lambda: None, output_bw=64)

        auto_quant = AutoQuant(sess,  starting_ops, ending_ops, unlabeled_data_loader, lambda: None)
        # Allowed accuracy drop < 0
        with pytest.raises(ValueError):
            _ = auto_quant.optimize(-1.0)

    def test_auto_quant_inference_fallback(
            self, sess, unlabeled_data_loader,
    ):
        class _Exception(Exception):
            pass

        def error_fn(*_, **__):
            raise _Exception

        bn_folded_acc = .4
        raw_quantsim_acc = bn_folded_acc + 1e-5
        with patch_ptq_techniques(
                bn_folded_acc, None, None, raw_quantsim_acc=raw_quantsim_acc
        ) as mocks:
            with create_tmp_directory() as results_dir:
                auto_quant = AutoQuant(sess,
                                       starting_ops,
                                       ending_ops,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)

                with patch("aimet_tensorflow.auto_quant_v2.fold_all_batch_norms", side_effect=error_fn):
                    # If all of prepare_model, validate_model, and BN folding fail, should return raw quantsim model
                    _, acc = auto_quant.run_inference()
                    assert np.allclose(acc, raw_quantsim_acc)

    def test_auto_quant_optimize_fallback(
            self, sess, unlabeled_data_loader,
    ):
        class _Exception(Exception):
            pass

        def error_fn(*_, **__):
            raise _Exception

        bn_folded_acc, cle_acc, adaround_acc = .4, .5, .6
        with patch_ptq_techniques(
                bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            with create_tmp_directory() as results_dir:
                auto_quant = AutoQuant(sess,
                                       starting_ops,
                                       ending_ops,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)

                with patch("aimet_tensorflow.auto_quant_v2.fold_all_batch_norms", side_effect=error_fn):
                    # If batchnorm folding fails, should return Adaround results
                    _, acc, _ = auto_quant.optimize()
                    assert acc == adaround_acc

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _ERROR_IGNORED,
                            'node_cross_layer_equalization': _SUCCESS,
                            'node_adaround': _SUCCESS,
                        })

                auto_quant = AutoQuant(sess,
                                       starting_ops,
                                       ending_ops,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                with patch("aimet_tensorflow.auto_quant_v2.equalize_model", side_effect=error_fn):
                    # If CLE fails, should return Adaround results
                    _, acc, _ = auto_quant.optimize()
                    assert acc == adaround_acc

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _SUCCESS,
                            'node_cross_layer_equalization': _ERROR_IGNORED,
                            'node_adaround': _SUCCESS,
                        })

                auto_quant = AutoQuant(sess,
                                       starting_ops,
                                       ending_ops,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                with patch("aimet_tensorflow.auto_quant_v2.Adaround.apply_adaround", side_effect=error_fn):
                    # If adaround fails, should return CLE results
                    _, acc, _ = auto_quant.optimize()
                    assert acc == cle_acc

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _SUCCESS,
                            'node_cross_layer_equalization': _SUCCESS,
                            'node_adaround': _ERROR_IGNORED,
                        })

                auto_quant = AutoQuant(sess,
                                       starting_ops,
                                       ending_ops,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=False)
                with patch("aimet_tensorflow.auto_quant_v2.fold_all_batch_norms", side_effect=error_fn), \
                        patch("aimet_tensorflow.auto_quant_v2.equalize_model", side_effect=error_fn), \
                        patch("aimet_tensorflow.auto_quant_v2.Adaround.apply_adaround", side_effect=error_fn):
                    # If everything fails, should raise an error
                    with pytest.raises(RuntimeError):
                        auto_quant.optimize()

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _ERROR_IGNORED,
                            'node_cross_layer_equalization': _ERROR_IGNORED,
                            'node_adaround': _ERROR_IGNORED,
                        })

                auto_quant = AutoQuant(sess,
                                       starting_ops,
                                       ending_ops,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
                                       results_dir=results_dir,
                                       strict_validation=True)
                with patch("aimet_tensorflow.auto_quant_v2.equalize_model", side_effect=error_fn):
                    # Hard stop if strict_validation=True
                    with pytest.raises(_Exception):
                        auto_quant.optimize()

                    with open(os.path.join(results_dir, 'diagnostics.html')) as f:
                        html_parsed = BeautifulSoup(f.read(), features="html.parser")
                        assert_html(html_parsed, {
                            'node_batchnorm_folding': _SUCCESS,
                            'node_cross_layer_equalization': _ERROR_FAILED,
                            'node_adaround': _NOT_VISITED,
                        })

    def test_auto_quant_early_exit(self, sess, unlabeled_data_loader):
        allowed_accuracy_drop = 0.1
        w32_acc = FP32_ACC - (allowed_accuracy_drop * 2)

        with create_tmp_directory() as results_dir:
            with patch_ptq_techniques(
                    bn_folded_acc=0, cle_acc=0, adaround_acc=0, w32_acc=w32_acc
            ) as mocks:
                auto_quant = AutoQuant(sess,
                                       starting_ops,
                                       ending_ops,
                                       unlabeled_data_loader,
                                       mocks.eval_callback,
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



    def test_set_additional_params(self, sess, unlabeled_data_loader):
        allowed_accuracy_drop = 0
        bn_folded_acc = .1
        cle_acc = .2
        adaround_acc = .3
        with patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc) as mocks:

            auto_quant = AutoQuant(sess,
                                   starting_ops,
                                   ending_ops,
                                   unlabeled_data_loader,
                                   mocks.eval_callback)
            adaround_params = AdaroundParameters(unlabeled_data_loader, 1)
            auto_quant.set_adaround_params(adaround_params)


            self._do_test_optimize_auto_quant(
                auto_quant, sess,
                allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc
            )
            adaround_args, _ = mocks.apply_adaround.call_args
            _, _, _, actual_adaround_params = adaround_args
            assert adaround_params == actual_adaround_params


@contextlib.contextmanager
def create_tmp_directory(dirname: str = "/tmp/.aimet_unittest"):
    success = False
    try:
        os.makedirs(dirname, exist_ok=True)
        success = True
    except FileExistsError:
        raise

    try:
        yield dirname
    finally:
        if success:
            shutil.rmtree(dirname)
