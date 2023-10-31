# /usr/bin/env python
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

""" Sequential MSE implementation """

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Callable
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from aimet_common.defs import QuantScheme
import aimet_common.libpymo as libpymo
from aimet_torch.utils import CachedDataset, get_ordered_list_of_modules, in_eval_mode, StopForwardException,\
    change_tensor_device_placement, get_device
from aimet_torch.adaround.activation_sampler import create_modulelist_for_group_modules,\
    get_block_inputs, get_block_outputs
from aimet_torch.qc_quantize_op import QcQuantizeWrapper, QcQuantizeOpMode
from aimet_torch.tensor_quantizer import TensorQuantizer, StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer
from aimet_torch.quantsim import QuantizationSimModel

# The following modules with weights are supported
SUPPORTED_MODULES = (torch.nn.Linear, )


def default_forward_fn(model, inputs):
    """
    Default forward function.
    :param model: pytorch model
    :param inputs: model inputs
    """
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
    return model(*inputs)


@dataclass
class SeqMseParams:
    """
    Sequential MSE parameters

    :param num_batches: Number of batches.
    :param num_candidates: Number of candidates to perform grid search. Default 20.
    :param inp_symmetry: Input symmetry. Default 'symqt'.
    :param loss_fn: Loss function. Default 'mse'.
    :param forward_fn: Optional adapter function that performs forward pass given a model and inputs
     yielded from the data loader. The function expects model as first argument and inputs to model as second argument.
    """
    num_batches: int
    num_candidates: int = 20
    inp_symmetry: str = 'symqt'
    loss_fn: str = 'mse'
    forward_fn: Callable = default_forward_fn


def apply_seq_mse(model: torch.nn.Module,
                  sim: QuantizationSimModel,
                  data_loader: DataLoader,
                  params: SeqMseParams,
                  modules_to_exclude: Optional[List[torch.nn.Module]] = None,
                  module_classes_to_exclude: Optional[List[torch.nn.Module]] = None,
                  checkpoints_config: Optional[str] = None):
    """
    Apply sequential MSE - find and freze optimal parameter encodings candidate
        1 Disable all input/output quantizers, param quantizers from exclusion list
	    2 Find and feeze optimal parameter encodings candidate for remaining supported modules
	    3 Re-enable disabled quantizers from step 1

	NOTE: module reference(s) passed to module_to_exclude list should be from sim.model.

    :param model: Original fp32 model
    :param sim: Corresponding QuantizationSimModel object
    :param data_loader: Data loader
    :param params: Sequential MSE parameters
    :param modules_to_exclude: List of supported modules to exclude when applying Sequential MSE
    :param module_classes_to_exclude: List of supported module classes to exclude when applying Sequential MSE
    :param checkpoints_config: Config files to split fp32/quant model by checkpoints to speedup activations sampling
    """
    # pylint: disable=protected-access
    assert sim._quant_scheme == QuantScheme.post_training_tf, "Use TF quant-scheme with sequential MSE."

    # disable all input/output activation quantizers and
    # parameter quantizers corresponding to modules from exclusion list
    quantizers = get_quantizers_to_be_disabled(sim, modules_to_exclude, module_classes_to_exclude)
    enable_disable_quantizers(quantizers, enabled=False)

    # Initialize all remaining parameters' encodings
    compute_all_param_encodings(sim)

    # Find and freeze optimal parameter encodings candidate
    with tempfile.TemporaryDirectory() as tempdir:
        cached_dataset = CachedDataset(data_loader, params.num_batches, os.path.join(tempdir, 'cached_dataset'))
        if checkpoints_config:
            apply_seq_mse_using_opt_sampling(checkpoints_config, model, sim, cached_dataset, params, tempdir)
        else:
            dummy_input = change_tensor_device_placement(next(iter(data_loader)), get_device(model))
            fp32_modules = get_ordered_list_of_modules(model, dummy_input)
            fp32_modules = [(name, module) for name, module in fp32_modules if isinstance(module, SUPPORTED_MODULES)]
            run_seq_mse(fp32_modules, model, sim.model, params, params.forward_fn,
                        cached_dataset, None)

    # re-enable disabled quantizers
    enable_disable_quantizers(quantizers, enabled=True)


def apply_seq_mse_using_opt_sampling(checkpoints_config: str,
                                     model: torch.nn.Module,
                                     sim: QuantizationSimModel,
                                     cached_dataset: CachedDataset,
                                     params: SeqMseParams,
                                     tempdir: str):
    """
    Apply sequential MSE using optimized sampling of intermediate data. When checkpoints_config file is provided,
     intermediate activations from breakpoint are treated as model inputs for next blocks.

    NOTE: Assumption is that the outputs from the current block are fed directly to following block
     and there are no funciotnal operations in-between.

    :param checkpoints_config: Config files to split fp32/quant model by checkpoints to speedup activations sampling
    :param model: Original fp32 model
    :param sim: Corresponding QuantizationSimModel object
    :param cached_dataset: Cached dataset
    :param params: Sequential MSE parameters
    :param tempdir: temporary working directory
    """
    # pylint: disable=too-many-locals
    ckpts_file = json.load(open(checkpoints_config))
    assert 'grouped_modules' in ckpts_file.keys(), \
        "Please provide a dictionary of grouped_modules in the file to define checkpoints"
    assert 'include_static_inputs' in ckpts_file.keys(), \
        "Please provide a dictionary of include_static_inputs in the file to define checkpoints"
    assert 'cache_on_cpu' in ckpts_file.keys(), \
        "Please define cache_on_cpu to determine whether to cache intermediate tensors on CPU"

    grouped_modules = ckpts_file['grouped_modules']
    breakpoint_module_name = ckpts_file['grouped_modules'][list(grouped_modules.keys())[0]][0]
    include_static_inputs = ckpts_file['include_static_inputs']
    cache_on_cpu = ckpts_file['cache_on_cpu']
    cached_fp_dataset, cached_quant_dataset = get_block_inputs(model, sim,
                                                               breakpoint_module_name,
                                                               cached_dataset, cache_on_cpu,
                                                               params.forward_fn, params.num_batches,
                                                               tempdir)
    # Get the device of model to latter be used to place input tensor on the same device
    device = get_device(model)
    model.cpu()
    sim.model.cpu()

    # Forward function for the ModuleList object
    def fwd_fn_modulelist(modulelists, x):
        for mod in modulelists:
            x = mod(*x) if isinstance(x, (tuple, list)) else mod(x)
        return x

    sub_fp_models, sub_sim_models = create_modulelist_for_group_modules(model, sim, grouped_modules)
    for i, (fp_block, quant_sim_block, static_input) in enumerate(zip(sub_fp_models,
                                                                      sub_sim_models,
                                                                      include_static_inputs)):
        fp32_modules = get_ordered_list_of_modules(fp_block, cached_fp_dataset[0], fwd_fn_modulelist)
        fp32_modules = [(name, module) for name, module in fp32_modules if isinstance(module, SUPPORTED_MODULES)]
        run_seq_mse(fp32_modules, fp_block, quant_sim_block, params, fwd_fn_modulelist,
                    cached_fp_dataset, cached_quant_dataset)

        # Get the outputs from the current block and assign to be the inputs for next block
        # except for the last block
        if i < len(sub_fp_models) - 1:
            get_block_outputs(fp_block, quant_sim_block, static_input,
                              cached_fp_dataset, cached_quant_dataset, cache_on_cpu,
                              fwd_fn_modulelist, device, tempdir)
    sim.model.to(device)

def run_seq_mse(fp32_modules: List[Tuple[str, torch.nn.Module]],
                model: torch.nn.Module,
                quant_model: torch.nn.Module,
                params: SeqMseParams,
                forward_fn: Callable,
                cached_fp_dataset: CachedDataset,
                cached_quant_dataset: Optional[CachedDataset] = None,
                ):
    """
    Run Sequential MSE

    :param fp32_modules: List of FP32 candidate modules in order of occurence
    :param model: FP32 model
    :param quant_model: QuantizationSimModel object
    :param params: Sequential MSE parameters
    :param forward_fn: Optional adapter function that performs forward pass given a model and inputs
     yielded from the data loader. The function expects model as first argument and inputs to model as second argument.
    :param cached_fp_dataset: Cached dataset object
    :param cached_quant_dataset: Cached dataset object
    """
    name_to_quant_module = {}
    for name, quant_module in quant_model.named_modules():
        name_to_quant_module[name] = quant_module

    if not cached_quant_dataset:
        cached_quant_dataset = cached_fp_dataset

    for module_qualified_name, fp32_module in fp32_modules:
        try:
            quant_module = name_to_quant_module[module_qualified_name]
        except KeyError:
            continue

        print("Finding optimal parameter encodings candidate: ", module_qualified_name)
        if params.inp_symmetry == "asym":
            fp32_inp_acts = get_module_inp_acts(fp32_module, model, params, forward_fn, cached_fp_dataset)
            quant_inp_acts = get_module_inp_acts(quant_module, quant_model, params, forward_fn, cached_quant_dataset)
            optimize_module(quant_module, fp32_inp_acts, quant_inp_acts, params)
        elif params.inp_symmetry == "symfp":
            fp32_inp_acts = get_module_inp_acts(fp32_module, model, params, forward_fn, cached_fp_dataset)
            optimize_module(quant_module, fp32_inp_acts, fp32_inp_acts, params)
        elif params.inp_symmetry == "symqt":
            quant_inp_acts = get_module_inp_acts(quant_module, quant_model, params, forward_fn, cached_quant_dataset)
            optimize_module(quant_module, quant_inp_acts, quant_inp_acts, params)
        else:
            raise ValueError(f"Invalid inp_symmetry: {params.inp_symmetry}")


def get_module_inp_acts(module: torch.nn.Module,
                        model: torch.nn.Module,
                        params: SeqMseParams,
                        forward_fn: Callable,
                        cached_dataset: CachedDataset,
                        ) -> torch.Tensor:
    """
    For given module, get inputs to the module.

    :param module: FP32/quant module
    :param model: FP32/quant model
    :param params: Sequential MSE parameters
    :param forward_fn: Optional adapter function that performs forward pass given a model and inputs
     yielded from the data loader. The function expects model as first argument and inputs to model as second argument.
    :param cached_dataset: Cached dataset
    :return: Concatenated inputs
    """
    inp_acts = []
    def hook_fn(_, inp, __):
        if isinstance(inp, tuple):
            inp_acts.append(inp[0])
        raise StopForwardException
    handle = module.register_forward_hook(hook_fn)

    iterator = iter(cached_dataset)
    for _ in range(params.num_batches):
        batch = change_tensor_device_placement(next(iterator), get_device(model))
        try:
            with in_eval_mode(model), torch.no_grad():
                forward_fn(model, batch)
        except StopForwardException:
            pass
    handle.remove()

    inp_acts = torch.stack(inp_acts)
    return inp_acts


def get_quantizers_to_be_disabled(sim: QuantizationSimModel,
                                  modules_to_exclude: Optional[List[torch.nn.Module]],
                                  module_classes_to_exclude: Optional[List[torch.nn.Module]])\
        -> List[TensorQuantizer]:
    """
    For given quantsim model, get all quantizers to be disabled before applying sequential MSE.

    :param sim: QuantizationSimModel object
    :param modules_to_exclude: List of supported modules to exclude when applying Sequential MSE
    :param module_classes_to_exclude: List of supported module classes to exclude when applying Sequential MSE
    :return: List of quantizers to be disabled.
    """
    # pylint: disable=protected-access
    # pylint: disable=unidiomatic-typecheck
    quantizers_to_be_disabled = []
    for _, quant_wrapper in sim.quant_wrappers():
        for quantizer in quant_wrapper.input_quantizers:
            if quantizer.enabled:
                quantizers_to_be_disabled.append(quantizer)
        for quantizer in quant_wrapper.output_quantizers:
            if quantizer.enabled:
                quantizers_to_be_disabled.append(quantizer)

    for _, quant_wrapper in sim.quant_wrappers():
        if modules_to_exclude and quant_wrapper in modules_to_exclude:
            for quantizer in quant_wrapper.param_quantizers.values():
                if quantizer.enabled:
                    quantizers_to_be_disabled.append(quantizer)
        if module_classes_to_exclude and type(quant_wrapper._module_to_wrap) in module_classes_to_exclude:
            for quantizer in quant_wrapper.param_quantizers.values():
                if quantizer.enabled:
                    quantizers_to_be_disabled.append(quantizer)
    return quantizers_to_be_disabled


def enable_disable_quantizers(quantizers: List[TensorQuantizer], enabled: bool):
    """
    For given list of quantizers, set (enable/disable) quantizer's 'enabled' attribute.

    :param quantizers: List of quantizers.
    :param enabled: Enabled flag.
    """
    for quantizer in quantizers:
        quantizer.enabled = enabled


def compute_all_param_encodings(sim: QuantizationSimModel):
    """
    Compute encodings for all parameters, needed for initializing Sequential MSE

    :param sim: Quant sim
    """
    for _, quant_wrapper in sim.quant_wrappers():
        for name, quantizer in quant_wrapper.param_quantizers.items():
            quantizer.reset_encoding_stats()
            quantizer.update_encoding_stats(getattr(quant_wrapper, name).data)
            quantizer.compute_encoding()

        # Wrapper mode must be set to ACTIVE because the wrapper's quantize_dequantize_params() will only call
        # into the param tensor quantizer's quantize_dequantize() if the mode isn't PASSTHROUGH.
        quant_wrapper.set_mode(QcQuantizeOpMode.ACTIVE)


def get_candidates(num_candidates: int,
                   per_channel_max: torch.Tensor,
                   per_channel_min: Optional[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Perform grid search.

    :param num_candidates: Number of candidates
    :param per_channel_max: Per channel max values
    :param per_channel_min: Per channel min values
    :return: candidates
    """
    candidates = []
    if per_channel_min is not None:
        for cand in range(num_candidates):
            cand_max = torch.tensor(per_channel_max / num_candidates * (cand + 1))
            cand_min = torch.tensor(per_channel_min / num_candidates * (cand + 1))
            candidates.append((cand_max, cand_min))
    else:
        for cand in range(num_candidates):
            cand_max = torch.tensor(per_channel_max / num_candidates * (cand + 1))
            cand_min = -cand_max
            candidates.append((cand_max, cand_min))
    return candidates


def optimize_module(quant_module: QcQuantizeWrapper,
                    x: torch.Tensor,
                    xq: torch.Tensor,
                    params: SeqMseParams):
    """
    Find and freeze optimal parameter encodings candidate for given module.

    :param quant_module: Quant module to be optimized
    :param x: Inputs to module from FP32 model
    :param xq: Inputs to module from QuantSim model
    :param params: Sequenial MSE parameters
    """
    # pylint: disable=too-many-locals
    if quant_module.param_quantizers["weight"].use_symmetric_encodings:
        per_channel_max = torch.max(quant_module.weight.abs(), dim=1)[0].detach()
        per_channel_min = None
    else:
        per_channel_max = torch.max(quant_module.weight, dim=1)[0].detach()
        per_channel_min = torch.min(quant_module.weight, dim=1)[0].detach()
    candidates = get_candidates(params.num_candidates, per_channel_max, per_channel_min)

    total_loss = []
    for cand_max, cand_min in candidates:
        compute_param_encodings(quant_module.param_quantizers['weight'], cand_min, cand_max)
        w = quant_module.weight
        wq = quant_module.param_quantizers['weight'].quantize_dequantize(w, libpymo.RoundingMode.ROUND_NEAREST)
        loss = torch.zeros(len(cand_max), device=w.device)
        with torch.no_grad():
            for batch_idx in range(params.num_batches):
                xqwq, xw = compute_outputs(quant_module, x[batch_idx], xq[batch_idx], w, wq)
                loss += compute_recon_loss(xqwq, xw, params)
            total_loss.append(loss)

    best_indices = torch.stack(total_loss).min(0, keepdim=True)[1]
    print(best_indices.squeeze(0)[:params.num_candidates])
    best_max = torch.stack([cand_max for cand_max, _ in candidates]).gather(0, best_indices)[0]
    best_min = torch.stack([cand_min for _, cand_min in candidates]).gather(0, best_indices)[0]

    # Compute and freeze parameter encodings using best candidate
    compute_param_encodings(quant_module.param_quantizers['weight'], best_min, best_max)
    quant_module.param_quantizers['weight'].freeze_encoding()


def compute_param_encodings(quantizer: Union[StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer],
                            x_min: torch.Tensor,
                            x_max: torch.Tensor):
    """
    Compute encodings for parameter quantizer using given x_min and x_max values.

    :param quantizer: Tensor quantizer
    :param x_min: min values
    :param x_max: max values
    """
    tensor = torch.stack([x_min, x_max], dim=-1)
    quantizer.reset_encoding_stats()
    quantizer.update_encoding_stats(tensor)
    quantizer.compute_encoding()


def compute_outputs(quant_module: QcQuantizeWrapper,
                    x: torch.Tensor,
                    xq: torch.Tensor,
                    w: torch.Tensor,
                    wq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute X^W^ and XW output acitvations.

    :param quant_module: Wrapper module to be optimized
    :param x: Inputs from FP32 model
    :param xq: Inputs from QuantSim model
    :param w: FP32 weights
    :param wq: Quantized-dequantized weights
    :return: xqwq, xw
    """
    # pylint: disable=protected-access
    module = quant_module._module_to_wrap

    if isinstance(module, torch.nn.Linear):
        xqwq = functional.linear(xq, wq, module.bias)
        xw = functional.linear(x, w, module.bias)
    else:
        raise ValueError('Unsupported module: ', module)
    return xqwq, xw


def compute_recon_loss(xqwq: torch.Tensor, xw: torch.Tensor, params: SeqMseParams):
    """
    Compute reconsturction loss

    :param xqwq: X^Q^ quantized-dequantized values
    :param xw: XW FP32 values
    :param params: Sequenial MSE parameters
    :return: loss
    """
    if params.loss_fn == "mse":
        loss_fn = functional.mse_loss
    elif params.loss_fn == "l1":
        loss_fn = functional.l1_loss
    else:
        loss_fn = neg_sqnr
    loss = loss_fn(xqwq, xw, reduction="none").sum((0, 1))
    return loss


def neg_sqnr(pred: torch.Tensor, target: torch.Tensor, eps=1e-10, reduction="none"):
    """
    Loss function to minimize negative SQNR which is equivalent to maximizing SQNR.

    :param pred: X^Q^ quantized-dequantized values
    :param target: XW FP32 values
    :param eps: epsilon
    :param reduction: unused arg
    :return: Negative SQNR
    """
    # pylint: disable=unused-argument
    quant_error = target - pred
    exp_noise = torch.mean(quant_error ** 2, (0, 1), keepdim=True) + eps
    exp_signal = torch.mean(target ** 2, (0, 1), keepdim=True)
    sqnr = exp_signal / exp_noise
    sqnr_db = 10 * torch.log10(sqnr)
    return -sqnr_db
