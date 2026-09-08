# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Module for checking QAT on quantsim v2"""

import os
import copy
import json
import tempfile
import pytest
import torch
from torchvision import models
from unittest.mock import patch

from aimet_torch.v2.quantization import affine
from aimet_torch.v2.quantization._utils import concretize_block_size, interleave
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.utils import is_leaf_module


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *x):
        return torch.round(*x)

    @staticmethod
    def backward(ctx, *output_grad):
        return output_grad


def autograd_based_qdq(
    tensor, scale, offset, qmin, qmax, block_size=None, zero_point_shift=0.0
) -> torch.Tensor:
    orig_tensor_shape = tensor.shape

    if block_size:
        block_size = concretize_block_size(tensor.shape, scale.shape, block_size)
        tensor = tensor.reshape(-1, *interleave(scale.shape, block_size))
        scale = scale.view(interleave(scale.shape, 1))
        offset = offset.view(interleave(offset.shape, 1))

    x_round = STE.apply(tensor / scale) - offset
    # PyTorch 2.14 clamp has different graidient in the clipping boundaries.
    # To keep the boundary gradient alive as in PyTorch 2.13,
    # we add a small margin to the clamp range and round it back to integer grid
    # See https://github.com/pytorch/pytorch/issues/195931
    x_quant = STE.apply(torch.clamp(x_round, qmin - 0.1, qmax + 0.1))
    return ((x_quant + offset) * scale).view(orig_tensor_shape)


# Wrap resnet function to mimic model class instantiation
def resnet18():
    return models.resnet18(weights=None)


# Test util functions
def get_quantized_modules(model: torch.nn.Module):
    module_names = []
    for _, module in model.named_children():
        if isinstance(module, BaseQuantizationMixin):
            module_names.append(module)

        elif not is_leaf_module(module):
            module_names = module_names + get_quantized_modules(module)

    return module_names


def get_enabled_quantizers_from_quantized_module(module: BaseQuantizationMixin):
    quantizers = (
        module.input_quantizers
        + module.output_quantizers
        + list(module.param_quantizers.values())
    )
    return [quantizer for quantizer in quantizers if quantizer is not None]


@pytest.fixture
def config_path():
    config_json = {
        "defaults": {
            "ops": {"is_output_quantized": "True", "is_symmetric": "False"},
            "params": {"is_quantized": "True", "is_symmetric": "True"},
        },
        "params": {},
        "op_type": {},
        "supergroups": [],
        "model_input": {"is_input_quantized": "True"},
        "model_output": {},
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config_path = os.path.join(temp_dir, "quantsim_config.json")
        with open(temp_config_path, "w") as temp_config_file:
            json.dump(config_json, temp_config_file)
        yield temp_config_path


@pytest.mark.parametrize("model_cls, input_shape", [(resnet18, (1, 3, 224, 224))])
class TestQuantsimV2QAT:
    # Based on https://github.com/qualcomm/aimet/blob/75131de478af7ead6a8676dc36f4c46a6d3390ea/TrainingExtensions/torch/test/python/test_range_learning.py#L81
    def test_grad_correctness(self, model_cls, input_shape, config_path):
        model = model_cls()
        dummy_input = torch.randn(input_shape)

        aimetgrad_qsim = QuantizationSimModel(
            model,
            dummy_input,
            config_file=config_path,
            default_param_bw=8,
            default_output_bw=4,
        )
        autograd_qsim = QuantizationSimModel(
            model,
            dummy_input,
            config_file=config_path,
            default_param_bw=8,
            default_output_bw=4,
        )

        aimetgrad_qsim.compute_encodings(
            lambda sim_model, _: sim_model(dummy_input), forward_pass_callback_args=None
        )
        autograd_qsim.compute_encodings(
            lambda sim_model, _: sim_model(dummy_input), forward_pass_callback_args=None
        )

        aimetgrad_qsim.model(dummy_input).sum().backward()
        # Patch backend of auto_grad_qsim to autograd-based backend
        with patch(
            "aimet_torch.quantization.affine.quantizer.quantize_dequantize",
            autograd_based_qdq,
        ):
            autograd_qsim.model(dummy_input).sum().backward()

        # Make sure that all the assertions are not being bypassed
        quantized_modules = get_quantized_modules(aimetgrad_qsim.model)
        assert len(quantized_modules) > 0

        for aimetgrad_param, autograd_param in zip(
            aimetgrad_qsim.model.parameters(), autograd_qsim.model.parameters()
        ):
            if not aimetgrad_param.requires_grad and not autograd_param.requires_grad:
                continue

            # Compare gradient between aimet-grad and autograd
            assert torch.allclose(
                aimetgrad_param.grad, autograd_param.grad, rtol=1e-3, atol=1e-3
            )

    def test_grad_accumulation(self, model_cls, input_shape, config_path):
        model = model_cls()
        dummy_input = torch.randn(input_shape)
        sim = QuantizationSimModel(
            model,
            dummy_input,
            config_file=config_path,
            default_param_bw=8,
            default_output_bw=4,
        )
        sim.compute_encodings(
            lambda sim_model, _: sim_model(dummy_input), forward_pass_callback_args=None
        )

        # First backward
        sim.model(dummy_input).sum().backward()
        grad_after_first_backward = {}
        for param_name, param in sim.model.named_parameters():
            if param.requires_grad:
                grad_after_first_backward[param_name] = param.grad.clone()

        # Second backward without zero_grad()
        sim.model(dummy_input).sum().backward()
        for param_name, param in sim.model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(
                    grad_after_first_backward[param_name] * 2, param.grad
                )

    def _test_weight_update(self, model_cls, input_shape, config_path, device):
        model = model_cls().to(device)
        dummy_input = torch.randn(input_shape).to(device)
        sim = QuantizationSimModel(
            model,
            dummy_input,
            config_file=config_path,
            default_param_bw=8,
            default_output_bw=4,
        )
        sim.model.to(device)
        sim.compute_encodings(
            lambda sim_model, _: sim_model(dummy_input), forward_pass_callback_args=None
        )

        # Store initial parameter values
        initial_param_value = {}
        for param_name, param in sim.model.named_parameters():
            if param.requires_grad:
                initial_param_value[param_name] = param.clone()

        # Update weights using gradient
        learning_rate = 0.01
        optimizer = torch.optim.SGD(sim.model.parameters(), lr=learning_rate)
        sim.model(dummy_input).sum().backward()

        # Store gradient of parameters
        grad_after_backward = {}
        for param_name, param in sim.model.named_parameters():
            if param.requires_grad:
                grad_after_backward[param_name] = param.grad.clone()
        optimizer.step()

        # Check weight update
        for param_name, param in sim.model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(
                    param,
                    initial_param_value[param_name]
                    - grad_after_backward[param_name] * learning_rate,
                )

    def test_qat_on_cpu(self, model_cls, input_shape, config_path):
        self._test_weight_update(
            model_cls,
            input_shape,
            config_path,
            device=torch.device("cpu"),
        )

    @pytest.mark.cuda
    def test_qat_on_gpu(self, model_cls, input_shape, config_path):
        self._test_weight_update(
            model_cls,
            input_shape,
            config_path,
            device=torch.device("cuda"),
        )

    # Based on https://github.com/qualcomm/aimet/blob/75131de478af7ead6a8676dc36f4c46a6d3390ea/TrainingExtensions/torch/test/python/test_quantizer.py#L1050
    @pytest.mark.cuda
    def test_multi_gpu(self, model_cls, input_shape, config_path):
        device = torch.device("cuda")
        model = model_cls().to(device)
        dummy_input = torch.randn(input_shape).to(device)
        sim = QuantizationSimModel(
            model,
            dummy_input,
            config_file=config_path,
            default_param_bw=8,
            default_output_bw=4,
        )
        sim.model.to(device)
        sim.compute_encodings(
            lambda sim_model, _: sim_model(dummy_input), forward_pass_callback_args=None
        )

        # Get single GPU outputs and gradients
        single_gpu_output = sim.model(copy.deepcopy(dummy_input))
        single_gpu_output.sum().backward()
        single_gpu_grad = {}
        for param_name, param in sim.model.named_parameters():
            if param.requires_grad:
                single_gpu_grad[param_name] = param.grad.clone()

        # Get multi-GPU outputs and gradients
        sim.model.zero_grad()
        sim.model = torch.nn.DataParallel(sim.model)
        multi_gpu_output = sim.model(copy.deepcopy(dummy_input))
        multi_gpu_output.sum().backward()

        # Compare single-gpu results and multi-gpu results
        assert torch.allclose(single_gpu_output, multi_gpu_output)
        for param_name, param in sim.model.named_parameters():
            if param_name.startswith("module."):
                param_name = ".".join(param_name.split(".")[1:])
            if param.requires_grad:
                assert torch.allclose(
                    param.grad, single_gpu_grad[param_name], rtol=1e-3, atol=1e-3
                )


def test_autograd_based_qdq():
    torch.manual_seed(0)
    tensor = torch.randn(4, 8, 12)
    block_size = [2, 4, 3]
    scale = torch.randn(2, 2, 4)
    offset = torch.randint(low=-128, high=127, size=(2, 2, 4), dtype=scale.dtype)
    autograd_based_qdq_out = autograd_based_qdq(
        tensor, scale, offset, 0, 255, block_size
    )
    affine_qdq_out = affine.quantize_dequantize(
        tensor, scale, offset, bitwidth=8, signed=False, block_size=block_size
    )
    assert torch.equal(autograd_based_qdq_out, affine_qdq_out)
