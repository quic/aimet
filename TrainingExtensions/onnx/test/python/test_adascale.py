# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import copy
import shutil
from unittest.mock import patch
import numpy as np
import torch
from onnx import numpy_helper, load_model
import onnx_ir
import onnx_ir.passes.common
import tempfile
import pytest

import aimet_onnx
from aimet_onnx import QuantizationSimModel
from aimet_onnx.experimental.adascale.adascale_optimizer import (
    AdaScale,
    adascale_model_config_dict,
)

from aimet_onnx.experimental.adascale.quantizer import (
    add_qlinear_layers,
    QuantizedLinear,
    AdaScaleLinearWeightQdq,
    AdaScaleConvWeightQdq,
    WeightQdq,
    get_adascale_trainable_params,
    replace_with_adascale_quantizers,
    QuantizedConv2d,
)
from aimet_onnx.experimental.adascale.model_converter import (
    resolve_block_residual_name,
    required_extra_block_inputs,
)
from .utils import add_genai_tests_path, force_random_weight_init

# TODO: Move block definitions to a util file
from .test_llm_topology_integration import (
    _block_topology_params,
    _build_block_topology_model,
    _detect_from_config,
    _export_onnx,
    _sample_inputs,
)


class ModelWithLinears(torch.nn.Module):
    def __init__(self):
        super(ModelWithLinears, self).__init__()

        self.layer1 = torch.nn.Linear(64, 32)
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.layer2 = torch.nn.Linear(32, 64)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)


class ModelWithConvs(torch.nn.Module):
    def __init__(self):
        super(ModelWithConvs, self).__init__()

        self.layer1 = torch.nn.Conv2d(64, 32, (3, 3))
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.layer2 = torch.nn.Conv2d(32, 64, (3, 3))

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)


class ModelWithConsecutiveLinearBlocks(torch.nn.Module):
    def __init__(self):
        super(ModelWithConsecutiveLinearBlocks, self).__init__()
        self.blocks = torch.nn.ModuleList(ModelWithLinears() for _ in range(2))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        for linear_block in self.blocks:
            x = linear_block(x)
        x = self.softmax(x)
        return x


class ModelWithConsecutiveConvBlocks(torch.nn.Module):
    def __init__(self):
        super(ModelWithConsecutiveConvBlocks, self).__init__()
        self.blocks = torch.nn.ModuleList(ModelWithConvs() for _ in range(2))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        for linear_block in self.blocks:
            x = linear_block(x)
        x = self.softmax(x)
        return x


class TestAdascaleQuantizer:
    @pytest.mark.parallel
    def test_quantized_conv2d(self):
        x = torch.randn(1, 4, 32, 32)
        module = torch.nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=2,
        )
        enc_shape = (module.weight.shape[0], 1, 1, 1)
        qmodule = QuantizedConv2d(
            module,
            enc_shape=enc_shape,
            bitwidth=4,
            block_size=None,
            zero_point_shift=None,
        )
        replace_with_adascale_quantizers(qmodule)
        # Check to run the op and see if it runs without failures
        out = qmodule(x)
        attrs = [
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "bias",
        ]
        for attr in attrs:
            val1 = getattr(module, attr)
            val2 = getattr(qmodule, attr)
            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                assert torch.equal(val1, val2)
            else:
                assert val1 == val2

    @pytest.mark.parallel
    def test_quantizer_backprop(self):
        class TwoLayerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # input_size is hardcoded to 10
                self.linear1 = torch.nn.Linear(10, 20)
                self.relu = torch.nn.ReLU()
                # hidden_size is hardcoded to 20, output_size is hardcoded to 5
                self.linear2 = torch.nn.Linear(20, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        model = TwoLayerModel()
        input_shape = (10, 10)
        input_tensor = torch.rand(*input_shape)
        orig_out = model(input_tensor).detach()

        model = add_qlinear_layers(model)
        replace_with_adascale_quantizers(model)
        temp = model(input_tensor)

        all_beta_gamma_parameters, all_scale_parameters = get_adascale_trainable_params(
            model
        )

        for m in model.parameters():
            m.requires_grad = False

        for p in all_scale_parameters + all_beta_gamma_parameters:
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(all_beta_gamma_parameters + all_scale_parameters)

        for epoch in range(5):
            quant_out = model(input_tensor)
            loss = torch.nn.functional.mse_loss(orig_out, quant_out)
            loss.backward()
            optimizer.step()

            if epoch < 4:
                optimizer.zero_grad()

        # All scale and beta, gamma params should have a grad
        for p in all_scale_parameters + all_beta_gamma_parameters:
            assert p.grad is not None

        new_out = model(input_tensor)
        assert not torch.equal(new_out, orig_out)

    @pytest.mark.parallel
    def test_qlinear_layer_replacement(self):
        model = ModelWithConsecutiveLinearBlocks().eval()
        model_copy = copy.deepcopy(model)
        input_shape = (1, 3, 32, 64)
        torch.random.manual_seed(1)
        dummy_input = torch.rand(input_shape)
        out_1 = model(copy.deepcopy(dummy_input))

        add_qlinear_layers(model)
        out_2 = model(copy.deepcopy(dummy_input))

        # verify weights have not changed and the classes are swapped correctly
        for linear_block_1, linear_block_2 in zip(model.blocks, model_copy.blocks):
            assert torch.equal(
                linear_block_1.layer1.weight, linear_block_2.layer1.weight
            )
            assert torch.equal(
                linear_block_1.layer2.weight, linear_block_2.layer2.weight
            )

            assert isinstance(linear_block_1.layer1, QuantizedLinear)
            assert isinstance(linear_block_1.layer2, QuantizedLinear)

        # multiple calls show no change in model parameters (no attrs set to train mode)
        out_2_a = model(copy.deepcopy(dummy_input))
        assert torch.equal(out_2, out_2_a)

        for linear_block in model.blocks:
            linear_block.layer1.param_quantizers["weight"] = None
            linear_block.layer2.param_quantizers["weight"] = None

        # with params removed, we should get the un-quantized output
        out_3 = model(copy.deepcopy(dummy_input))
        assert torch.equal(out_3, out_1)

    @pytest.mark.parallel
    def test_single_quantizer_backprop(self):
        """
        Given:
        - Create QDQ module, store initial scale and create adascale equivalent with the QDQ module
        - Set Adascale params requires_grad to True
        When:
        - Train with random data
        - Save S2, S3
        Then:
        - S2, S3 Should not be zeros
        - Compare original scale with new scale
        """

        weight_shape, qdq_shape = (30, 20), (30, 1)
        torch.manual_seed(0)
        weight_tensor = torch.rand(*weight_shape)

        torch.manual_seed(1)
        expected_tensor = torch.rand(*weight_shape)

        qdq = WeightQdq(weight_tensor, qdq_shape, 4)

        adascale_qdq = AdaScaleLinearWeightQdq(weight_tensor, qdq_shape, 4)
        assert torch.equal(adascale_qdq.min, qdq.min)
        assert torch.equal(adascale_qdq.max, qdq.max)
        assert torch.equal(qdq(weight_tensor), adascale_qdq(weight_tensor))

        beta_gamma, scale_params = adascale_qdq.get_adascale_trainable_parameters()
        for p in beta_gamma + scale_params:
            assert p.requires_grad

        orig_output = adascale_qdq(weight_tensor)
        prev_loss = None
        optimizer = torch.optim.Adam(beta_gamma + scale_params)
        for epoch in range(5):
            quant_out = adascale_qdq(weight_tensor)
            loss = torch.nn.functional.mse_loss(expected_tensor, quant_out)
            assert prev_loss != loss
            prev_loss = loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        adascale_out = adascale_qdq(weight_tensor)
        # verify training is changing the output
        assert not torch.equal(adascale_out, orig_output)

        # verify adascale_qdq can be converted to regular qdq
        weight_after_adascale_fold = adascale_qdq.get_folded_weight(weight_tensor)

        new_qdq = WeightQdq(weight_after_adascale_fold, qdq_shape, 4)
        new_qdq.set_range(adascale_qdq.get_min(), adascale_qdq.get_max())

        assert torch.allclose(adascale_qdq.get_max(), new_qdq.get_max())
        assert torch.allclose(adascale_qdq.get_min(), new_qdq.get_min())

        modified_out = new_qdq(weight_after_adascale_fold)
        assert torch.allclose(modified_out, adascale_out)

    @pytest.mark.parallel
    def test_get_adascale_trainable_params_linear(self):
        model = ModelWithConsecutiveLinearBlocks().eval()
        add_qlinear_layers(model)
        replace_with_adascale_quantizers(model)
        all_beta_gamma_parameters, all_scale_parameters = get_adascale_trainable_params(
            model
        )
        assert (
            len(all_beta_gamma_parameters) == 8
        )  # 2 blocks * 2 linear layers * 2 params(beta, gamma)
        assert (
            len(all_scale_parameters) == 8
        )  # 2 blocks * 2 linear layers * 2 params(s2, s3)

    @pytest.mark.parallel
    def test_get_adascale_trainable_params_conv(self):
        model = ModelWithConsecutiveConvBlocks().eval()
        add_qlinear_layers(model)
        replace_with_adascale_quantizers(model)
        all_beta_gamma_parameters, all_scale_parameters = get_adascale_trainable_params(
            model
        )
        assert (
            len(all_beta_gamma_parameters) == 8
        )  # 2 blocks * 2 conv layers * 2 params(beta, gamma)
        assert (
            len(all_scale_parameters) == 12
        )  # 2 blocks * 2 conv layers * 3 params(s2, s3, s4)

    @pytest.mark.parallel
    def test_adascale_forward_linear(self):
        weight_shape, qdq_shape = (3, 10), (3, 1)
        out_channels_dim = 0
        torch.manual_seed(0)
        bw = 4

        weight_tensor = torch.rand(*weight_shape)

        # torch.rand returns random values in [0, 1)
        # here is the math for finding min, max, scale, offset for symmetric quantization
        expected_max = torch.max(
            weight_tensor.view(weight_shape[0], -1), dim=1
        ).values.reshape(qdq_shape)
        expected_scale = expected_max / float(
            2 ** (bw - 1) - 1
        )  # 2^(bits-1)-1 = 7 for 4 bits
        expected_min = -1 * expected_max - expected_scale

        adascale_qdq = AdaScaleLinearWeightQdq(weight_tensor, qdq_shape, 4)

        # At construction, min, max, scale, offset should match expected values, since the learnable scales are 0
        assert torch.allclose(adascale_qdq.get_max(), expected_max)
        assert torch.allclose(adascale_qdq.get_min(), expected_min)
        assert torch.allclose(adascale_qdq.get_scale(), expected_scale)
        assert torch.equal(adascale_qdq.get_offset(), torch.zeros(qdq_shape))

        def simple_ada_qdq(weight, max, min, s2, s3, gamma, beta):
            # simple adascale forward that mimics the one in AdaScaleLinearWeightQdq
            scaled_weight = (weight / torch.exp(s2)) / torch.exp(s3)
            max = max * torch.exp(gamma)  # new max
            min = min * torch.exp(beta)  # new min
            scale = (max - min) / float(2 ** (bw) - 1)  # new scale

            # Regular qdq
            quantized = torch.clamp(
                torch.round(scaled_weight / scale), -(2 ** (bw - 1)), 2 ** (bw - 1) - 1
            )
            dequantized = quantized * scale

            return dequantized

        # With s2, s3 = 0, beta, gamma = 0, output should match simple_ada_qdq output
        test_s2 = torch.full(weight_shape, 0.0)
        test_s3 = torch.full(qdq_shape, 0.0)
        test_gamma = torch.full(qdq_shape, 0.0)
        test_beta = torch.full(qdq_shape, 0.0)

        out_1 = adascale_qdq(weight_tensor)
        out_2 = simple_ada_qdq(
            weight_tensor,
            expected_max,
            expected_min,
            test_s2,
            test_s3,
            test_gamma,
            test_beta,
        )
        assert torch.allclose(out_1, out_2)

        # With s2 = 1, s3 = 0, beta, gamma = 0, output should match simple_ada_qdq output
        test_s2 = torch.full(weight_shape, 1.0)
        test_s3 = torch.full(qdq_shape, 0.0)
        test_gamma = torch.full(qdq_shape, 0.0)
        test_beta = torch.full(qdq_shape, 0.0)

        adascale_qdq.s2.data = test_s2

        out_1 = adascale_qdq(weight_tensor)
        out_2 = simple_ada_qdq(
            weight_tensor,
            expected_max,
            expected_min,
            test_s2,
            test_s3,
            test_gamma,
            test_beta,
        )
        assert torch.allclose(out_1, out_2)

        # With s2 = 1, s3 = 1, beta, gamma = 1, output should match simple_ada_qdq output
        test_s2 = torch.full(weight_shape, 1.0)
        test_s3 = torch.full(qdq_shape, 1.0)
        test_gamma = torch.full(qdq_shape, 1.0)
        test_beta = torch.full(qdq_shape, 1.0)

        adascale_qdq.s2.data = test_s2
        adascale_qdq.s3.data = test_s3
        adascale_qdq.gamma.data = test_gamma
        adascale_qdq.beta.data = test_beta

        out_1 = adascale_qdq(weight_tensor)
        out_2 = simple_ada_qdq(
            weight_tensor,
            expected_max,
            expected_min,
            test_s2,
            test_s3,
            test_gamma,
            test_beta,
        )
        assert torch.allclose(out_1, out_2)

    @pytest.mark.parallel
    def test_adascale_forward_conv(self):
        weight_shape, qdq_shape = (3, 10, 5, 5), (3, 1, 1, 1)
        s4_shape = (1, 10, 1, 1)
        out_channels_dim = 0
        torch.manual_seed(0)
        bw = 4

        weight_tensor = torch.rand(*weight_shape)

        # torch.rand returns random values in [0, 1)
        # here is the math for finding min, max, scale, offset for symmetric quantization
        expected_max = torch.max(
            weight_tensor.view(weight_shape[0], -1), dim=1
        ).values.reshape(qdq_shape)
        expected_scale = expected_max / float(
            2 ** (bw - 1) - 1
        )  # 2^(bits-1)-1 = 7 for 4 bits
        expected_min = -1 * expected_max - expected_scale

        adascale_qdq = AdaScaleConvWeightQdq(weight_tensor, qdq_shape, 4)

        # At construction, min, max, scale, offset should match expected values, since the learnable scales are 0
        assert torch.allclose(adascale_qdq.get_max(), expected_max)
        assert torch.allclose(adascale_qdq.get_min(), expected_min)
        assert torch.allclose(adascale_qdq.get_scale(), expected_scale)
        assert torch.equal(adascale_qdq.get_offset(), torch.zeros(qdq_shape))

        def simple_ada_qdq(weight, max, min, s2, s3, s4, gamma, beta):
            # simple adascale forward that mimics the one in AdaScaleLinearWeightQdq
            scaled_weight = ((weight / torch.exp(s2)) / torch.exp(s3)) / torch.exp(s4)
            max = max * torch.exp(gamma)  # new max
            min = min * torch.exp(beta)  # new min
            scale = (max - min) / float(2 ** (bw) - 1)  # new scale

            # Regular qdq
            quantized = torch.clamp(
                torch.round(scaled_weight / scale), -(2 ** (bw - 1)), 2 ** (bw - 1) - 1
            )
            dequantized = quantized * scale

            return dequantized

        # With s2, s3 = 0, beta, gamma = 0, output should match simple_ada_qdq output
        test_s2 = torch.full(weight_shape, 0.0)
        test_s3 = torch.full(qdq_shape, 0.0)
        test_s4 = torch.full(s4_shape, 0.0)
        test_gamma = torch.full(qdq_shape, 0.0)
        test_beta = torch.full(qdq_shape, 0.0)

        out_1 = adascale_qdq(weight_tensor)
        out_2 = simple_ada_qdq(
            weight_tensor,
            expected_max,
            expected_min,
            test_s2,
            test_s3,
            test_s4,
            test_gamma,
            test_beta,
        )
        assert torch.allclose(out_1, out_2)

        # With s2 = 1, s3 = 0, beta, gamma = 0, output should match simple_ada_qdq output
        test_s2 = torch.full(weight_shape, 1.0)
        test_s3 = torch.full(qdq_shape, 0.0)
        test_s4 = torch.full(s4_shape, 0.0)
        test_gamma = torch.full(qdq_shape, 0.0)
        test_beta = torch.full(qdq_shape, 0.0)

        adascale_qdq.s2.data = test_s2

        out_1 = adascale_qdq(weight_tensor)
        out_2 = simple_ada_qdq(
            weight_tensor,
            expected_max,
            expected_min,
            test_s2,
            test_s3,
            test_s4,
            test_gamma,
            test_beta,
        )
        assert torch.allclose(out_1, out_2)

        # With s2 = 1, s3 = 1, beta, gamma = 1, output should match simple_ada_qdq output
        test_s2 = torch.full(weight_shape, 1.0)
        test_s3 = torch.full(qdq_shape, 1.0)
        test_s4 = torch.full(s4_shape, 1.0)
        test_gamma = torch.full(qdq_shape, 1.0)
        test_beta = torch.full(qdq_shape, 1.0)

        adascale_qdq.s2.data = test_s2
        adascale_qdq.s3.data = test_s3
        adascale_qdq.s4.data = test_s4
        adascale_qdq.gamma.data = test_gamma
        adascale_qdq.beta.data = test_beta

        out_1 = adascale_qdq(weight_tensor)
        out_2 = simple_ada_qdq(
            weight_tensor,
            expected_max,
            expected_min,
            test_s2,
            test_s3,
            test_s4,
            test_gamma,
            test_beta,
        )
        assert torch.allclose(out_1, out_2)

    @pytest.mark.parallel
    def test_block_level_api(self):
        model = ModelWithConsecutiveLinearBlocks().eval()
        input_shape = (1, 3, 32, 64)
        torch.random.manual_seed(1)
        dummy_input = [torch.rand(input_shape), torch.rand(input_shape)]
        weight_names = [
            "onnx::MatMul_24",
            "onnx::MatMul_25",
            "onnx::MatMul_26",
            "onnx::MatMul_27",
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            torch.onnx.export(
                model,
                dummy_input[0],
                tempdir + "/model.onnx",
                input_names=["input"],
                output_names=["output"],
                dynamo=False,
            )
            model_onnx = load_model(tempdir + "/model.onnx")
            sim = QuantizationSimModel(
                model_onnx,
                [dummy_input],
                config_file="htp_v73",
            )
            sim._compute_param_encodings(overwrite=False)
            qt_input = []
            for t in dummy_input:
                qt_input.append(
                    t * 0.3
                )  # making quantized input different from fp input

            original_weights = {}
            for initializer in sim.model.model.graph.initializer:
                if initializer.name in weight_names:
                    weight_array = numpy_helper.to_array(initializer)
                    original_weights[initializer.name] = weight_array.copy()

            orig_enc = {}
            for quantizer_name in weight_names:
                orig_enc[quantizer_name] = sim.qc_quantize_op_dict[
                    quantizer_name
                ].get_encodings()

            for i in range(len(model.blocks)):
                block_input_output_names = [
                    (["input"], ["/blocks.0/layer2/Add_output_0"]),
                    (["/blocks.0/layer2/Add_output_0"], ["output"]),
                ]
                sim_model: onnx_ir.Model = onnx_ir.from_proto(sim.model.model)
                onnx_ir.passes.common.TopologicalSortPass().call(sim_model)
                AdaScale.optimize_adascale_block(
                    sim_model,
                    sim.qc_quantize_op_dict,
                    dummy_input,
                    qt_input,
                    block_input_output_names=block_input_output_names[i],
                    beta_gamma_lr=1e-3,
                    scales_lr=5e-4,
                    num_iterations=100,
                )
                sim.model.model.CopyFrom(onnx_ir.to_proto(sim_model))

            updated_weights = {}
            for initializer in sim.model.model.graph.initializer:
                if initializer.name in weight_names:
                    weight_array = numpy_helper.to_array(initializer)
                    updated_weights[initializer.name] = weight_array.copy()

            for weight in weight_names:
                assert not np.all(original_weights[weight] == updated_weights[weight])

            for quantizer_name in weight_names:
                updated_enc = sim.qc_quantize_op_dict[quantizer_name].get_encodings()
                consolidated_delta_updated_enc = [
                    updated_enc[i].delta for i in range(len(updated_enc))
                ]
                consolidated_delta_orig_enc = [
                    orig_enc[quantizer_name][i].delta
                    for i in range(len(orig_enc[quantizer_name]))
                ]
                assert consolidated_delta_updated_enc != consolidated_delta_orig_enc

    @pytest.mark.parallel
    @pytest.mark.parametrize("seq_len", [8, 32, 2048])
    def test_mse_loss_fn(self, seq_len):
        """lp_loss equals MSE scaled by the sequence length S (dim 1)."""
        from aimet_onnx.experimental.adascale import adascale_optimizer as opt

        torch.manual_seed(0)
        fp_out = torch.rand(4, seq_len, 16)  # [B, S, H]
        qt_out = torch.rand(4, seq_len, 16)

        lp = opt._mse_loss_fn(fp_out, qt_out)

        mse = torch.nn.functional.mse_loss(fp_out, qt_out)
        assert torch.allclose(lp, mse * seq_len)

    @pytest.mark.parallel
    def test_block_level_adascale_early_stopping(self):
        """Integration test for the _EARLY_STOPPING flag using the real factory and
        _EarlyStopping."""
        from aimet_onnx.experimental.adascale import adascale_optimizer as opt
        from aimet_onnx.common.early_stopping import _EarlyStoppingConfig

        model = ModelWithConsecutiveLinearBlocks().eval()
        input_shape = (1, 3, 32, 64)
        torch.random.manual_seed(1)
        dummy_input = [torch.rand(input_shape), torch.rand(input_shape)]
        qt_input = [t * 0.3 for t in dummy_input]

        num_iterations = 20
        block_input_output_names = (["input"], ["/blocks.0/layer2/Add_output_0"])

        real_loss_fn = opt._mse_loss_fn

        def make_counting_loss_fn(counter):
            def loss_fn(fp_out, qt_out, data_idx):
                counter[0] += 1
                return real_loss_fn(fp_out, qt_out)

            return loss_fn

        def run_block(counter):
            with tempfile.TemporaryDirectory() as tempdir:
                torch.onnx.export(
                    model,
                    dummy_input[0],
                    tempdir + "/model.onnx",
                    input_names=["input"],
                    output_names=["output"],
                    dynamo=False,
                )
                model_onnx = load_model(tempdir + "/model.onnx")
                sim = QuantizationSimModel(
                    model_onnx,
                    [dummy_input],
                    config_file="htp_v73",
                )
                sim._compute_param_encodings(overwrite=False)

                sim_model = onnx_ir.from_proto(sim.model.model)
                onnx_ir.passes.common.TopologicalSortPass().call(sim_model)
                AdaScale.optimize_adascale_block(
                    sim_model,
                    sim.qc_quantize_op_dict,
                    dummy_input,
                    qt_input,
                    block_input_output_names=block_input_output_names,
                    beta_gamma_lr=1e-3,
                    scales_lr=5e-4,
                    num_iterations=num_iterations,
                    loss_fn=make_counting_loss_fn(counter),
                )

        # Early stopping ON.
        cfg = _EarlyStoppingConfig(check_interval=1, rel_threshold=1e9, window=1)
        on_count = [0]
        with patch.object(opt, "_EARLY_STOPPING", cfg):
            run_block(on_count)
        assert 0 < on_count[0] < num_iterations

        # Early stopping OFF: the loop should run the full num_iterations = 20 iterations.
        assert opt._EARLY_STOPPING is None
        off_count = [0]
        run_block(off_count)
        assert off_count[0] == num_iterations

    @pytest.mark.cuda
    def test_adascale_gpu_memory_leak(self):
        """
        Test that GPU memory doesn't leak during AdaScale optimization loop.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        class ModelWithTwoInputs(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(1024, 512)
                self.linear2 = torch.nn.Linear(512, 256)

            def forward(self, x1, x2):
                combined = x1 + x2
                out = self.linear1(combined)
                out = torch.nn.functional.relu(out)
                return self.linear2(out)

        model = ModelWithTwoInputs().eval()
        input_shape = (2, 512, 1024)
        torch.random.manual_seed(1)

        # Each sample is a LIST of tensors (multiple inputs)
        fp_inputs = [
            [torch.rand(input_shape), torch.rand(input_shape)],
            [torch.rand(input_shape), torch.rand(input_shape)],
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            torch.onnx.export(
                model,
                (fp_inputs[0][0], fp_inputs[0][1]),
                tempdir + "/model.onnx",
                input_names=["input1", "input2"],
                output_names=["output"],
                dynamo=False,
            )
            onnx_model = load_model(tempdir + "/model.onnx")
            sim = QuantizationSimModel(
                onnx_model,
                fp_inputs[0],
            )
            sim._compute_param_encodings(overwrite=False)

            quantized_inputs = []
            for inputs in fp_inputs:
                quantized_inputs.append([inp * 0.3 for inp in inputs])

            # Clear memory before test
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Monitor memory during optimization by monkey-patching
            memory_samples = []
            original_adam_step = torch.optim.Adam.step
            iteration_counter = [0]

            def step_with_memory_tracking(self, *args, **kwargs):
                result = original_adam_step(self, *args, **kwargs)
                iteration_counter[0] += 1
                if iteration_counter[0] in [1, 5, 10, 15]:
                    torch.cuda.synchronize()
                    memory_samples.append(
                        {
                            "iteration": iteration_counter[0],
                            "memory_mb": torch.cuda.memory_allocated() / 1e6,
                        }
                    )
                return result

            torch.optim.Adam.step = step_with_memory_tracking
            try:
                block_input_output_names = (["input1", "input2"], ["output"])
                sim_model = onnx_ir.from_proto(sim.model.model)
                onnx_ir.passes.common.TopologicalSortPass().call(sim_model)
                AdaScale.optimize_adascale_block(
                    sim_model,
                    sim.qc_quantize_op_dict,
                    fp_inputs,
                    quantized_inputs,
                    block_input_output_names=block_input_output_names,
                    beta_gamma_lr=1e-3,
                    scales_lr=5e-4,
                    num_iterations=15,
                    device=torch.device("cuda:0"),
                )
                sim.model.model.CopyFrom(onnx_ir.to_proto(sim_model))
            finally:
                torch.optim.Adam.step = original_adam_step

            assert len(memory_samples) == 4

            # Check that memory is stable across all iterations stamps
            mem_at_iter_1 = memory_samples[0]["memory_mb"]
            mem_at_iter_5 = memory_samples[1]["memory_mb"]
            mem_at_iter_10 = memory_samples[2]["memory_mb"]
            mem_at_iter_15 = memory_samples[3]["memory_mb"]

            # Check each iteration against baseline
            max_allowed_diff_pct = 0.05

            for idx, (iteration, memory) in enumerate(
                [(5, mem_at_iter_5), (10, mem_at_iter_10), (15, mem_at_iter_15)]
            ):
                if idx == 0:
                    continue
                diff_pct = abs(memory - mem_at_iter_1) / mem_at_iter_1
                assert diff_pct < max_allowed_diff_pct


@pytest.mark.parallel
class TestBlockResidualResolution:
    """Unit tests for the fp16 block-boundary handling that ``apply_adascale``
    relies on: ``resolve_block_residual_name`` (walks past leading ``Cast``
    producers) and ``required_extra_block_inputs`` (collects unbounded graph
    inputs the subgraph still depends on).
    """

    @staticmethod
    def _graph(text: str) -> onnx_ir.Graph:
        return onnx_ir.from_onnx_text(text).graph

    def test_resolve_walks_past_leading_cast(self):
        """fp16 pattern: RMSNorm anchor's input is post-Cast; the true residual is upstream."""
        graph = self._graph(
            """
            <ir_version: 8, opset_import: ["": 18]>
            g (float16[N, 4] residual) => (float[N, 4] anchor_in) {
                anchor_in = Cast<to=1>(residual)
            }
            """
        )
        assert resolve_block_residual_name(graph, "anchor_in") == "residual"

    def test_resolve_walks_past_chained_casts(self):
        """Multiple Casts in a row (e.g., fp16 -> fp32 -> fp16) all get stripped."""
        graph = self._graph(
            """
            <ir_version: 8, opset_import: ["": 18]>
            g (float16[N, 4] residual) => (float16[N, 4] anchor_in) {
                mid = Cast<to=1>(residual)
                anchor_in = Cast<to=10>(mid)
            }
            """
        )
        assert resolve_block_residual_name(graph, "anchor_in") == "residual"

    def test_resolve_stops_at_non_cast_producer(self):
        """Non-Cast producer (Add) is not stepped through — the anchor is the true residual."""
        graph = self._graph(
            """
            <ir_version: 8, opset_import: ["": 18]>
            g (float[N, 4] a, float[N, 4] b) => (float[N, 4] residual) {
                residual = Add(a, b)
            }
            """
        )
        assert resolve_block_residual_name(graph, "residual") == "residual"

    def test_resolve_unknown_name_returns_input_unchanged(self):
        """Names not in the graph pass through — this is the fallback path for
        callers that may pass an already-resolved name."""
        graph = self._graph(
            """
            <ir_version: 8, opset_import: ["": 18]>
            g (float[N, 4] x) => (float[N, 4] y) {
                y = Identity(x)
            }
            """
        )
        assert resolve_block_residual_name(graph, "not_in_graph") == "not_in_graph"

    def test_required_extras_collects_unbounded_graph_input(self):
        """When the subgraph reachable from ``output_names`` depends on a graph
        input not listed in ``input_names``, that input is returned as an extra."""
        graph = self._graph(
            """
            <ir_version: 8, opset_import: ["": 18]>
            g (float[N, 4] residual, float[N, 4] side) => (float[N, 4] out) {
                out = Add(residual, side)
            }
            """
        )
        extras = required_extra_block_inputs(graph, ["residual"], ["out"])
        assert extras == ["side"]

    def test_required_extras_empty_when_fully_declared(self):
        """When every producer chain terminates at an already-declared input,
        no extras are needed."""
        graph = self._graph(
            """
            <ir_version: 8, opset_import: ["": 18]>
            g (float[N, 4] residual) => (float[N, 4] out) {
                out = Relu(residual)
            }
            """
        )
        assert required_extra_block_inputs(graph, ["residual"], ["out"]) == []

    def test_required_extras_ignores_initializers(self):
        """Initializers (constants baked into the graph) are not counted as
        unbounded inputs — only true graph inputs are."""
        graph = self._graph(
            """
            <ir_version: 8, opset_import: ["": 18]>
            g (float[N, 4] residual) => (float[N, 4] out)
            <float[4] bias = {0.0, 0.0, 0.0, 0.0}>
            {
                out = Add(residual, bias)
            }
            """
        )
        assert required_extra_block_inputs(graph, ["residual"], ["out"]) == []


# MoE models whose torchscript export bakes a fixed expert-routing Split
# (sized to the experts hit while tracing).
# These fail basic ORT inference when routed to a different number of experts
_UNEXPORTABLE_MOE = {
    "Qwen2MoeConfig",
    "Glm4MoeConfig",
    "Qwen3NextConfig",
    "OlmoeConfig",
}


@pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
@pytest.mark.skip_on_windows_amd64(
    "insufficient disk for large ONNX export on Windows AMD64 runner"
)
@pytest.mark.parametrize(
    "config_attr, backend, detect_kwargs, _",
    list(_block_topology_params()),
)
def test_adascale_block_topologies(config_attr, backend, detect_kwargs, _):
    """Run AdaScale over the decoder blocks detected on each transformer topology."""
    if config_attr in _UNEXPORTABLE_MOE:
        pytest.skip(
            f"{config_attr}: export bakes a fixed expert-routing for MoE, fails ORT inference"
        )
    np.random.seed(0)
    torch.manual_seed(0)
    model, cfg = _build_block_topology_model(config_attr)
    onnx_model = _export_onnx(model, _sample_inputs(cfg), backend)
    end_points, *_ = _detect_from_config(config_attr, backend, detect_kwargs)
    inputs = [
        {
            t.name: tensor.numpy()
            for t, tensor in zip(onnx_model.graph.input, _sample_inputs(cfg))
        }
    ]
    sim = aimet_onnx.QuantizationSimModel(
        onnx_model, param_type="int4", activation_type="int16"
    )
    sim.compute_encodings(inputs)
    output = sim.session.run(None, inputs[0])[0]
    # Smoke test for sampler, converter, reloading
    AdaScale._apply_adascale(
        sim, inputs, end_points, num_iterations=1, beta_gamma_lr=0.0, scales_lr=0.0
    )
    # Weights/encodings should be loaded back in original state
    output_after_adascale = sim.session.run(None, inputs[0])[0]
    assert np.allclose(output, output_after_adascale, rtol=1e-4, atol=5e-4)


@pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
@pytest.mark.skip_on_windows_amd64(
    "insufficient disk for large ONNX export on Windows AMD64 runner"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_adascale_e2e(add_genai_tests_path, dtype, small_model: bool = True):
    from transformers import AutoConfig
    from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX
    from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
        get_model_checkpoint_path,
    )
    import random

    context_length = 32
    sequence_length = 16
    model_id = "Qwen/Qwen2-0.5B"
    model_cls = LLM_ONNX

    SEED = 20
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if small_model:
        llm_config.num_hidden_layers = 2

    cache_dir = get_model_checkpoint_path(model_id)
    try:
        with force_random_weight_init(vocab_size=1024):
            entry = model_cls.instantiate_float_model(
                model_id,
                context_length,
                sequence_length,
                small_model=small_model,
                dtype=dtype,
            )
        collection = model_cls.instantiate_quantsim(entry)
        sim = collection.backbone

        onnx_weights_min_max = {}
        for initializer in sim.model.model.graph.initializer:
            weight_array = numpy_helper.to_array(initializer)
            onnx_weights_min_max[initializer.name] = {
                "min": float(np.min(weight_array)),
                "max": float(np.max(weight_array)),
            }
        adascale_model_config_dict["qwen2"].model_config = llm_config

        # Float inputs must match the exported graph's float dtype (fp16 when
        # ``dtype=torch.float16``); ORT rejects a dtype mismatch on session run.
        float_np_dtype = np.float16 if dtype == torch.float16 else np.float32
        inputs = {
            "input_ids": np.random.randint(0, 100, size=(1, 16), dtype=np.int32),
            "attention_mask": np.random.randint(0, 100, size=(1, 1, 16, 32)).astype(
                float_np_dtype
            ),
            "position_ids": np.arange(0, 16).reshape(1, 16).astype(np.int32),
            "past_key_0_in": np.zeros((1, 2, 16, 64)).astype(float_np_dtype),
            "past_value_0_in": np.zeros((1, 2, 16, 64)).astype(float_np_dtype),
            "past_key_1_in": np.zeros((1, 2, 16, 64)).astype(float_np_dtype),
            "past_value_1_in": np.zeros((1, 2, 16, 64)).astype(float_np_dtype),
        }

        # Create a copy of the weights before applying AdaScale
        original_weights = {}
        for initializer in sim.model.model.graph.initializer:
            weight_array = numpy_helper.to_array(initializer)
            original_weights[initializer.name] = weight_array.copy()

        AdaScale.apply_adascale(
            sim,
            [inputs],
            adascale_model_config_dict["qwen2"],
            num_iterations=2,
        )

        linear_list = [
            key for key in sim.qc_quantize_op_dict.keys() if "onnx::MatMul" in key
        ]

        # Dropping the last linear layers since that is always the LM head, which is not modified by adascale
        param_list = linear_list[:-1]

        # Verify that the encodings are frozen for the parameters modified by AdaScale
        for param in param_list:
            assert sim.qc_quantize_op_dict[param]._is_encoding_frozen

        for initializer in sim.model.model.graph.initializer:
            if initializer.name in param_list:
                weight_array = numpy_helper.to_array(initializer)
                assert not np.all(original_weights[initializer.name] == weight_array)
            else:
                weight_array = numpy_helper.to_array(initializer)
                assert np.all(original_weights[initializer.name] == weight_array)

        assert len(sim.model.model.graph.output)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
@pytest.mark.skip(reason="Too long to run in CI")
def test_qwen_adascale_e2e_ppl(add_genai_tests_path, small_model=False):
    """AdaScale test pipeline for qwen model"""
    from unittest.mock import patch

    with patch(
        "aimet_onnx.experimental.adascale.adascale_optimizer._DEBUG_NUM_PARTIAL_ITERATIONS",
        new=2,
    ):
        from transformers import AutoConfig
        from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX
        from GenAILab.qai_hub_lm.models.generator import Generator
        from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import (
            TorchONNXInterface,
        )
        from GenAILab.bench.onnx.quant_recipes import _prefill_inputs
        from GenAILab.bench.datasets import Wikitext
        from GenAILab.bench.metrics import PPL

        context_length = 512
        sequence_length = 512
        model_id = "Qwen/Qwen2.5-0.5B"
        model_cls = LLM_ONNX

        llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if small_model:
            llm_config.num_hidden_layers = 2

        entry = model_cls.instantiate_float_model(
            model_id, context_length, sequence_length, small_model=small_model
        )
        collection = model_cls.instantiate_quantsim(entry)
        sim = collection.backbone

        tokenizer = LLM_ONNX.instantiate_tokenizer(model_id)

        train_dataset = Wikitext.load_encoded_dataset(
            tokenizer, context_length, "train"
        )
        quantsim_with_torch_interface = TorchONNXInterface(sim, llm_config)
        generator = Generator(
            quantsim_with_torch_interface, tokenizer, sequence_length, context_length
        )

        inputs = _prefill_inputs(sim, generator, train_dataset, num_iterations=20)

        adascale_model_config_dict[
            generator.config.model_type
        ].model_config = llm_config

        for name in sim.activation_names:
            sim.qc_quantize_op_dict[name].enabled = False
        sim.compute_encodings(inputs)

        ppl_score_before_ada = PPL.evaluate(
            generator, tokenizer, context_length, num_iterations=50
        )
        print("PPL before Adascale: ", ppl_score_before_ada)

        AdaScale.apply_adascale(
            sim,
            inputs,
            adascale_model_config_dict[generator.config.model_type],
            num_iterations=1500,
        )

        sim.compute_encodings(inputs)
        ppl_score_after_ada = PPL.evaluate(
            generator, tokenizer, context_length, num_iterations=50
        )
        print("Computed PPL score after applying AdaScale", ppl_score_after_ada)
        assert ppl_score_before_ada > ppl_score_after_ada
