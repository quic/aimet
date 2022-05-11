# /usr/bin/env python3.6
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

import contextlib
from dataclasses import dataclass
from unittest.mock import patch, MagicMock
import os
import pytest
import shutil
from typing import Callable
import tensorflow as tf

from aimet_tensorflow.auto_quant import AutoQuant
from aimet_tensorflow.adaround.adaround_weight import AdaroundParameters
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils.common import deepcopy_tf_session


tf.compat.v1.disable_eager_execution()


@pytest.fixture
def starting_op_names():
    return ['conv2d_input']


@pytest.fixture
def output_op_names():
    return ['keras_model/Softmax']


def session(device):
    from aimet_tensorflow.examples.test_models import keras_model
    with tf.device(device):
        graph = tf.Graph()
        with graph.as_default():
            _ = keras_model()

            # flags for test
            applied_bn_folding = tf.compat.v1.get_variable(name="applied_bn_folding", shape=(), dtype=tf.bool)
            _ = tf.compat.v1.assign(applied_bn_folding, True, name="set_applied_bn_folding")
            _ = tf.compat.v1.identity(applied_bn_folding, name="get_applied_bn_folding")

            applied_cle = tf.compat.v1.get_variable(name="applied_cle", shape=(), dtype=tf.bool)
            _ = tf.compat.v1.assign(applied_cle, True, name="set_applied_cle")
            _ = tf.compat.v1.identity(applied_cle, name="get_applied_cle")

            applied_adaround = tf.compat.v1.get_variable(name="applied_adaround", shape=(), dtype=tf.bool)
            _ = tf.compat.v1.assign(applied_adaround, True, name="set_applied_adaround")
            _ = tf.compat.v1.identity(applied_adaround, name="get_applied_adaround")

            init = tf.compat.v1.global_variables_initializer()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.compat.v1.Session(graph=graph, config=config)
    sess.run(init)
    return sess


@pytest.fixture(scope="session")
def cpu_session():
    return session('/cpu:0')


@pytest.fixture(scope="session")
def gpu_session():
    return session('/gpu:0')


def _tf_session_set_flag(sess: tf.compat.v1.Session, var_name: str) -> None:
    assign_op_name = f"set_{var_name}"
    assign_op = sess.graph.get_operation_by_name(assign_op_name)
    sess.run(assign_op)


def _tf_session_get_flag(sess: tf.compat.v1.Session, var_name: str) -> bool:
    identity_op_name = f"get_{var_name}"
    identity_op = sess.graph.get_operation_by_name(identity_op_name)
    assert len(identity_op.outputs) == 1
    ret = sess.run(identity_op.outputs)
    return ret[0]


@pytest.fixture(scope="session")
def dataset_length():
    return 2


@pytest.fixture(scope="session")
def unlabeled_dataset(dataset_length):
    graph = tf.Graph()
    with graph.as_default():
        dummy_inputs = tf.random.normal((dataset_length, 16, 16, 3))
        dataset = tf.compat.v1.data.Dataset.from_tensor_slices(dummy_inputs)
        dataset = dataset.batch(1)
        return dataset


def assert_applied_techniques(
        output_session, acc, encoding_path,
        target_acc, bn_folded_acc, cle_acc, adaround_acc,
):
    applied_bn_folding = _tf_session_get_flag(output_session, "applied_bn_folding")
    applied_cle = _tf_session_get_flag(output_session, "applied_cle")
    applied_adaround = _tf_session_get_flag(output_session, "applied_adaround")

    # Batchnorm folding is always applied.
    assert applied_bn_folding

    # If accuracy is good enough after batchnorm folding
    if bn_folded_acc >= target_acc:
        assert acc == bn_folded_acc
        assert encoding_path.endswith("batchnorm_folding.encodings")
        assert not applied_cle
        assert not applied_adaround
        return

    # If accuracy is good enough after cle
    if cle_acc >= target_acc:
        assert acc == cle_acc
        assert encoding_path.endswith("cross_layer_equalization.encodings")
        assert applied_cle
        assert not applied_adaround
        return

    # CLE should be applied if and only if it brings accuracy gain
    assert applied_cle == (bn_folded_acc < cle_acc)

    # If accuracy is good enough after adaround
    if adaround_acc >= target_acc:
        assert acc == adaround_acc
        assert encoding_path.endswith("adaround.encodings")
        assert applied_adaround
        return

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



FP32_ACC = 80.0


@contextlib.contextmanager
def patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc):
    def bn_folding(session, *_, **__):
        session = deepcopy_tf_session(session)
        _tf_session_set_flag(session, "applied_bn_folding")
        return session, list()

    def cle(session, *_, **__):
        session = deepcopy_tf_session(session)
        _tf_session_set_flag(session, "applied_bn_folding")
        _tf_session_set_flag(session, "applied_cle")
        return session

    def adaround(session, *_, **__):
        session = deepcopy_tf_session(session)
        _tf_session_set_flag(session, "applied_adaround")
        return session

    class _QuantizationSimModel(QuantizationSimModel):
        def _add_and_configure_quant_nodes(self, *_, **__):
            pass

        def compute_encodings(self, forward_pass_callback, args):
            def _forward_pass_callback(sess, args):
                _run = sess.run
                sess.run = lambda *_, **__: None
                ret = forward_pass_callback(sess, args)
                sess.run = _run
                return ret
            return super().compute_encodings(_forward_pass_callback, args)

        def set_and_freeze_param_encodings(self, _):
            pass

    def mock_eval_callback(session, _):
        if _tf_session_get_flag(session, "applied_adaround"):
            return adaround_acc
        if _tf_session_get_flag(session, "applied_cle"):
            return cle_acc
        if _tf_session_get_flag(session, "applied_bn_folding"):
            return bn_folded_acc
        return FP32_ACC

    @dataclass
    class Mocks:
        eval_callback: Callable
        QuantizationSimModel: MagicMock
        fold_all_batch_norms: MagicMock
        equalize_model: MagicMock
        apply_adaround: MagicMock

    with patch("aimet_tensorflow.auto_quant.QuantizationSimModel", side_effect=_QuantizationSimModel) as mock_qsim,\
            patch("aimet_tensorflow.auto_quant.fold_all_batch_norms", side_effect=bn_folding) as mock_bn_folding,\
            patch("aimet_tensorflow.auto_quant.equalize_model", side_effect=cle) as mock_cle,\
            patch("aimet_tensorflow.auto_quant.Adaround.apply_adaround", side_effect=adaround) as mock_adaround:
        try:
            yield Mocks(
                eval_callback=mock_eval_callback,
                QuantizationSimModel=mock_qsim,
                fold_all_batch_norms=mock_bn_folding,
                equalize_model=mock_cle,
                apply_adaround=mock_adaround,
            )
        finally:
            pass


@pytest.fixture(autouse=True)
def patch_dependencies():
    def render(*_, **__):
        return ""

    with patch("aimet_tensorflow.auto_quant.jinja2.environment.Template.render", side_effect=render):
         yield


class TestAutoQuant:
    def test_auto_quant_default_values(
            self, cpu_session, starting_op_names, output_op_names, 
    ):
        graph = tf.Graph()
        with graph.as_default():
            dataset_length = 2
            dummy_inputs = tf.random.normal((dataset_length, 16, 16, 3))
            dataset = tf.compat.v1.data.Dataset.from_tensor_slices(dummy_inputs)
            dataset = dataset.batch(1)

        with create_tmp_directory() as results_dir:
            with patch_ptq_techniques(
                bn_folded_acc=50.0, cle_acc=60.0, adaround_acc=70.0
            ) as mocks:
                auto_quant = AutoQuant(
                    allowed_accuracy_drop=0.0,
                    unlabeled_dataset=dataset,
                    eval_callback=mocks.eval_callback,
                )

                auto_quant.apply(cpu_session,
                                 starting_op_names,
                                 output_op_names,
                                 results_dir=results_dir)

        expected_adaround_params = AdaroundParameters(dataset, dataset_length)
        args, kwargs = mocks.apply_adaround.call_args
        _, _, _, adaround_params = args

        assert adaround_params.data_set == expected_adaround_params.data_set

    @pytest.mark.parametrize(
        "bn_folded_acc, cle_acc, adaround_acc",
        [(50., 60., 70.), (50., 70., 60.), (70., 50., 60.)]
    )
    @pytest.mark.parametrize("allowed_accuracy_drop", [5., 15.])
    def test_auto_quant_cpu(
            self, cpu_session, starting_op_names, output_op_names, unlabeled_dataset,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        self._test_auto_quant(
            cpu_session, starting_op_names, output_op_names, unlabeled_dataset,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
        )

    @pytest.mark.parametrize(
        "bn_folded_acc, cle_acc, adaround_acc",
        [(50., 60., 70.), (50., 70., 60.), (70., 50., 60.)]
    )
    @pytest.mark.parametrize("allowed_accuracy_drop", [5., 15.])
    def test_auto_quant_gpu(
            self, gpu_session, starting_op_names, output_op_names, unlabeled_dataset,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        self._test_auto_quant(
            gpu_session, starting_op_names, output_op_names, unlabeled_dataset,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
        )

    def _test_auto_quant(
            self, session, starting_op_names, output_op_names, unlabeled_dataset,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            auto_quant = AutoQuant(
                allowed_accuracy_drop=allowed_accuracy_drop,
                unlabeled_dataset=unlabeled_dataset,
                eval_callback=mocks.eval_callback,
            )
            self._do_test_apply_auto_quant(
                auto_quant, session, starting_op_names, output_op_names,
                allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc
            )

    def _do_test_apply_auto_quant(
            self, auto_quant, input_session, starting_op_names, output_op_names,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        with create_tmp_directory() as results_dir:
            target_acc = FP32_ACC - allowed_accuracy_drop

            output_session, acc, encoding_path =\
                auto_quant.apply(input_session,
                                 starting_op_names,
                                 output_op_names,
                                 results_dir=results_dir)

            assert_same_device(output_session.graph, input_session.graph)
            assert_applied_techniques(
                output_session, acc, encoding_path,
                target_acc, bn_folded_acc, cle_acc, adaround_acc,
            )

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

    def test_auto_quant_caching(
        self, cpu_session, starting_op_names, output_op_names, unlabeled_dataset,
    ):
        allowed_accuracy_drop = 0.0
        bn_folded_acc, cle_acc, adaround_acc = 40., 50., 60.
        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            auto_quant = AutoQuant(
                allowed_accuracy_drop=allowed_accuracy_drop,
                unlabeled_dataset=unlabeled_dataset,
                eval_callback=mocks.eval_callback,
            )

            with create_tmp_directory() as results_dir:
                cache_id = "unittest"
                cache_files  = [
                    os.path.join(results_dir, ".auto_quant_cache", cache_id, f"{key}.meta")
                    for key in ("cle", "adaround")
                ]

                # No previously cached results
                auto_quant.apply(cpu_session, starting_op_names, output_op_names,
                                 results_dir=results_dir, cache_id=cache_id)
                for cache_file in cache_files:
                    assert os.path.exists(cache_file)

                assert mocks.equalize_model.call_count == 1
                assert mocks.apply_adaround.call_count == 1

                # Load cached result
                auto_quant.apply(cpu_session, starting_op_names, output_op_names,
                                 results_dir=results_dir, cache_id=cache_id)

                # PTQ functions should not be called twice.
                assert mocks.equalize_model.call_count == 1
                assert mocks.apply_adaround.call_count == 1


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
