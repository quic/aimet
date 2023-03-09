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
import copy
from dataclasses import dataclass
import itertools
from unittest.mock import patch, MagicMock
import os
import pytest
import shutil
from typing import Callable
import torch
from torch.utils.data import Dataset, DataLoader

from aimet_torch import utils
from aimet_torch.auto_quant import AutoQuant
from aimet_torch.adaround.adaround_weight import AdaroundParameters, Adaround
from aimet_torch.quantsim import QuantizationSimModel, OnnxExportApiArgs


class Model(torch.nn.Module):
    """
    Model
    """

    def __init__(self):
        super(Model, self).__init__()
        self._conv_0 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1)
        self._relu = torch.nn.ReLU()

        self.applied_bn_folding = False
        self.applied_cle = False
        self.applied_adaround = False

    def forward(self, x: torch.Tensor):
        assert utils.get_device(self) == x.device
        return self._relu(self._conv_0(x))


@pytest.fixture(scope="session")
def cpu_model():
    return Model().cpu()


@pytest.fixture(scope="session")
def gpu_model():
    return Model().cuda()


@pytest.fixture(scope="session")
def dummy_input():
    return torch.randn((1, 3, 8, 8))


@pytest.fixture(scope="session")
def unlabeled_data_loader(dummy_input):
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    dataset = MyDataset([dummy_input[0, :] for _ in range(10)])
    return DataLoader(dataset)


def assert_applied_techniques(
        output_model, acc, encoding_path,
        target_acc, bn_folded_acc, cle_acc, adaround_acc,
):
    # Batchnorm folding is always applied.
    assert output_model.applied_bn_folding

    # If accuracy is good enough after batchnorm folding
    if bn_folded_acc >= target_acc:
        assert acc == bn_folded_acc
        assert encoding_path.endswith("batchnorm_folding.encodings")
        assert not output_model.applied_cle
        assert not output_model.applied_adaround
        return

    # If accuracy is good enough after cle
    if cle_acc >= target_acc:
        assert acc == cle_acc
        assert encoding_path.endswith("cross_layer_equalization.encodings")
        assert output_model.applied_cle
        assert not output_model.applied_adaround
        return

    # CLE should be applied if and only if it brings accuracy gain
    assert output_model.applied_cle == (bn_folded_acc < cle_acc)

    # If accuracy is good enough after adaround
    if adaround_acc >= target_acc:
        assert acc == adaround_acc
        assert encoding_path.endswith("adaround.encodings")
        assert output_model.applied_adaround
        return

    assert acc == max(bn_folded_acc, cle_acc, adaround_acc)

    if max(bn_folded_acc, cle_acc, adaround_acc) == bn_folded_acc:
        assert encoding_path.endswith("batchnorm_folding.encodings")
    elif max(bn_folded_acc, cle_acc, adaround_acc) == cle_acc:
        assert encoding_path.endswith("cross_layer_equalization.encodings")
    else:
        assert encoding_path.endswith("adaround.encodings")


FP32_ACC = 80.0


@contextlib.contextmanager
def patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc):
    def bn_folding(model: Model, *_, **__):
        model.applied_bn_folding = True
        return tuple()

    def cle(model: Model, *_, **__):
        model.applied_bn_folding = True
        model.applied_cle = True

    def adaround(model: Model, *_, **__):
        model = copy.deepcopy(model)
        model.applied_adaround = True
        return model

    class _QuantizationSimModel(QuantizationSimModel):
        def compute_encodings(self, *_):
            pass

        def set_and_freeze_param_encodings(self, _):
            pass

    def mock_eval_callback(model, _):
        if model.applied_adaround:
            return adaround_acc
        if model.applied_cle:
            return cle_acc
        if model.applied_bn_folding:
            return bn_folded_acc

        return FP32_ACC

    @dataclass
    class Mocks:
        eval_callback: Callable
        QuantizationSimModel: MagicMock
        fold_all_batch_norms: MagicMock
        equalize_model: MagicMock
        apply_adaround: MagicMock

    with patch("aimet_torch.auto_quant.QuantizationSimModel", side_effect=_QuantizationSimModel) as mock_qsim,\
            patch("aimet_torch.auto_quant.fold_all_batch_norms", side_effect=bn_folding) as mock_bn_folding,\
            patch("aimet_torch.auto_quant.equalize_model", side_effect=cle) as mock_cle,\
            patch("aimet_torch.auto_quant.Adaround.apply_adaround", side_effect=adaround) as mock_adaround:
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

    with patch("aimet_torch.auto_quant.jinja2.environment.Template.render", side_effect=render):
         yield


class TestAutoQuant:
    def test_auto_quant_default_values(self, unlabeled_data_loader):
        auto_quant = AutoQuant(
            allowed_accuracy_drop=0.0,
            unlabeled_dataset_iterable=unlabeled_data_loader,
            eval_callback=MagicMock(),
        )
        assert auto_quant.adaround_params.data_loader is unlabeled_data_loader
        assert auto_quant.adaround_params.num_batches is len(unlabeled_data_loader)

    @pytest.mark.parametrize(
        "bn_folded_acc, cle_acc, adaround_acc",
        itertools.permutations([50., 60., 70.])
    )
    @pytest.mark.parametrize("allowed_accuracy_drop", [5., 15.])
    def test_auto_quant_cpu(
            self, cpu_model, dummy_input, unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        self._test_auto_quant(
            cpu_model, dummy_input, unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
        )

    @pytest.mark.cuda
    def test_auto_quant_gpu(
            self, gpu_model, dummy_input, unlabeled_data_loader,
    ):
        bn_folded_acc, cle_acc, adaround_acc = 50., 60., 70.
        allowed_accuracy_drop = 15.

        self._test_auto_quant(
            gpu_model, dummy_input, unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
        )

    def _test_auto_quant(
            self, model, dummy_input, unlabeled_data_loader,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            auto_quant = AutoQuant(
                allowed_accuracy_drop=allowed_accuracy_drop,
                unlabeled_dataset_iterable=unlabeled_data_loader,
                eval_callback=mocks.eval_callback,
            )
            self._do_test_apply_auto_quant(
                auto_quant, model, dummy_input,
                allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc
            )

    def _do_test_apply_auto_quant(
            self, auto_quant, input_model, dummy_input,
            allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc,
    ):
        with create_tmp_directory() as results_dir:
            target_acc = FP32_ACC - allowed_accuracy_drop

            if utils.get_device(input_model) == torch.device("cpu"):
                output_model, acc, encoding_path =\
                    auto_quant.apply(input_model,
                                     dummy_input_on_cpu=dummy_input.cpu(),
                                     results_dir=results_dir)
            else:
                output_model, acc, encoding_path =\
                    auto_quant.apply(input_model,
                                     dummy_input_on_cpu=dummy_input.cpu(),
                                     dummy_input_on_gpu=dummy_input.cuda(),
                                     results_dir=results_dir)

            assert utils.get_device(output_model) == utils.get_device(input_model)
            assert_applied_techniques(
                output_model, acc, encoding_path,
                target_acc, bn_folded_acc, cle_acc, adaround_acc,
            )

    def test_auto_quant_invalid_input(self, unlabeled_data_loader):
        # Allowed accuracy drop < 0
        with pytest.raises(ValueError):
            _ = AutoQuant(-1.0, unlabeled_data_loader, MagicMock(), MagicMock())

        # Bitwidth < 4 or bitwidth > 32
        with pytest.raises(ValueError):
            _ = AutoQuant(0, unlabeled_data_loader, MagicMock(), default_param_bw=2)

        with pytest.raises(ValueError):
            _ = AutoQuant(0, unlabeled_data_loader, MagicMock(), default_param_bw=64)

        with pytest.raises(ValueError):
            _ = AutoQuant(0, unlabeled_data_loader, MagicMock(), default_output_bw=2)

        with pytest.raises(ValueError):
            _ = AutoQuant(0, unlabeled_data_loader, MagicMock(), default_output_bw=64)

    @pytest.mark.cuda
    def test_auto_quant_invalid_input_gpu(self, unlabeled_data_loader):
        auto_quant = AutoQuant(0, unlabeled_data_loader, MagicMock())
        with pytest.raises(ValueError):
            auto_quant.apply(Model().cuda(), unlabeled_data_loader)

    def test_auto_quant_caching(
        self, cpu_model, dummy_input, unlabeled_data_loader,
    ):
        allowed_accuracy_drop = 0.0
        bn_folded_acc, cle_acc, adaround_acc = 40., 50., 60.
        with patch_ptq_techniques(
            bn_folded_acc, cle_acc, adaround_acc
        ) as mocks:
            auto_quant = AutoQuant(
                allowed_accuracy_drop=allowed_accuracy_drop,
                unlabeled_dataset_iterable=unlabeled_data_loader,
                eval_callback=mocks.eval_callback,
            )

            with create_tmp_directory() as results_dir:
                cache_id = "unittest"
                cache_files  = [
                    os.path.join(results_dir, ".auto_quant_cache", cache_id, f"{key}.pkl")
                    for key in ("batchnorm_folding", "cle", "adaround")
                ]

                # No previously cached results
                auto_quant.apply(cpu_model, dummy_input, results_dir=results_dir, cache_id=cache_id)

                for cache_file in cache_files:
                    assert os.path.exists(cache_file)

                assert mocks.fold_all_batch_norms.call_count == 1
                assert mocks.equalize_model.call_count == 1
                assert mocks.apply_adaround.call_count == 1

                # Load cached result
                auto_quant.apply(cpu_model, dummy_input, results_dir=results_dir, cache_id=cache_id)

                # PTQ functions should not be called twice.
                assert mocks.fold_all_batch_norms.call_count == 1
                assert mocks.equalize_model.call_count == 1
                assert mocks.apply_adaround.call_count == 1

    def test_set_additional_params(self, cpu_model, dummy_input, unlabeled_data_loader):
        allowed_accuracy_drop = 0
        bn_folded_acc = 0
        cle_acc = 0
        adaround_acc = 0
        with patch_ptq_techniques(bn_folded_acc, cle_acc, adaround_acc) as mocks:
            export = QuantizationSimModel.export

            def export_wrapper(*args, **kwargs):
                assert kwargs["onnx_export_args"].opset_version == 10
                assert kwargs["propagate_encodings"]
                return export(*args, **kwargs)

            try:
                setattr(QuantizationSimModel, "export", export_wrapper)
                auto_quant = AutoQuant(
                    allowed_accuracy_drop=0,
                    unlabeled_dataset_iterable=unlabeled_data_loader,
                    eval_callback=mocks.eval_callback,
                )
                adaround_params = AdaroundParameters(unlabeled_data_loader, 1)
                auto_quant.set_adaround_params(adaround_params)

                auto_quant.set_export_params(OnnxExportApiArgs(10), True)

                self._do_test_apply_auto_quant(
                    auto_quant, cpu_model, dummy_input,
                    allowed_accuracy_drop, bn_folded_acc, cle_acc, adaround_acc
                )
                adaround_args, _ = Adaround.apply_adaround.call_args
                _, _, actual_adaround_params = adaround_args
                assert adaround_params == actual_adaround_params
            finally:
                setattr(QuantizationSimModel, "export", export)


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
