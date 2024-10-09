# =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
#
# =============================================================================
# pylint: disable=all
import os
import json
from packaging import version
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import autocast

from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer
from datasets import load_dataset

import deepspeed as ds
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.inference.engine import InferenceEngine
from tqdm import tqdm

from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.v1.qc_quantize_op import enable_recompute, QcQuantizeWrapper, LearnedGridQuantWrapper
from aimet_common.defs import QuantScheme
from aimet_torch.utils import get_all_quantizers, in_eval_mode, in_train_mode

import utils


##############################
###### dist parameters #######
##############################
RANK = int(os.getenv('RANK'))
LOCAL_RANK = int(os.getenv('LOCAL_RANK'))
WORLD_SIZE = int(os.getenv('WORLD_SIZE'))
MASTER_ADDR = os.getenv('MASTER_ADDR')
MASTER_PORT = int(os.getenv('MASTER_PORT', 19101))


##############################
##### model parameters #######
##############################
STUDENT_MODEL_NAME = 'huggyllama/llama-7b' # choice: ['gpt2', 'gpt2-{medium, large, xl}', 'huggyllama/llama-7b']
TEACHER_MODEL_NAME = 'huggyllama/llama-7b' # choice: ['gpt2', 'gpt2-{medium, large, xl}', 'huggyllama/llama-7b']
SEQUENCE_LENGTH = 1024


##############################
#### training parameters #####
##############################
NUM_EPOCHS = 1
TRAIN_MICRO_BATCH_SIZE_PER_GPU = 1
TRAIN_BATCH_SIZE = 8
if TRAIN_BATCH_SIZE % (TRAIN_MICRO_BATCH_SIZE_PER_GPU * WORLD_SIZE) != 0:
    raise ValueError(
        "Invalid Arguments: Expected TRAIN_BATCH_SIZE % (TRAIN_MICRO_BATCH_SIZE_PER_GPU * WORLD_SIZE) == 0"
    )
NUM_BATCHES_TO_TRAIN = None # If None, use all batches
TRAIN_DATALOADER_SEED = 1017


##############################
###### eval parameters #######
##############################
EVAL_MICRO_BATCH_SIZE_PER_GPU = 4
NUM_BATCHES_TO_EVAL = None # If None, use all batches


##############################
## quantization parameters ###
##############################
WEIGHT_BITWIDTH = 4
ACTIVATION_BITWIDTH = 16
NUM_BATCHES_FOR_CALIBRATION = 10


def get_student_model() -> torch.nn.Module:
    return _get_model(STUDENT_MODEL_NAME)


def get_teacher_model() -> torch.nn.Module:
    return _get_model(TEACHER_MODEL_NAME)


def _get_model(model_name) -> torch.nn.Module:
    # NOTE: Modify this function as necessary

    config = AutoConfig.from_pretrained(model_name)
    config.return_dict = False

    if model_name in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'):
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
        model_cls = GPT2LMHeadModel
    elif model_name == 'huggyllama/llama-7b':
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        model_cls = LlamaForCausalLM
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model_cls.from_pretrained(model_name, config=config)


def get_tokenizer():
    # NOTE: Modify this function as necessary

    if STUDENT_MODEL_NAME in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'):
        tokenizer_cls = AutoTokenizer
    elif STUDENT_MODEL_NAME == 'huggyllama/llama-7b':
        tokenizer_cls = LlamaTokenizer
    else:
        raise ValueError(f"Unsupported model name: {STUDENT_MODEL_NAME}")

    return tokenizer_cls.from_pretrained(STUDENT_MODEL_NAME)


def get_dataset(dataset, subset, split) -> Dataset:
    # NOTE: Modify this function as necessary
    return load_dataset(dataset, subset, split=split)


def evaluate(model, dataloader, num_batches):
    """
    Evaluates cross entropy loss over the given dataset
    """
    with in_eval_mode(model):
        return _evaluate(model, dataloader, num_batches)


@autocast()
@torch.no_grad()
def _evaluate(model, dataloader, num_batches):
    if num_batches is None:
        num_batches = len(dataloader)

    if num_batches > len(dataloader):
        raise RuntimeError(
            f'num_batchs ({num_batches}) is larger than '
            f'the length of data loader ({len(dataloader)}).'
        )

    device = f"cuda:{LOCAL_RANK}"

    loss = 0
    for batch_id, batch in enumerate(tqdm(dataloader, total=num_batches, desc="evaluate")):
        if batch_id >= num_batches:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids = position_ids.clip(0, SEQUENCE_LENGTH - 1).to(device)
        labels = batch.get("labels", input_ids).to(device)

        # Shift inputs and labels
        labels = labels[..., 1:]
        input_ids = input_ids[..., :-1]
        attention_mask = attention_mask[..., :-1]
        position_ids = position_ids[..., :-1]

        logits, *_ = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           position_ids=position_ids)
        loss += F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1))

    loss = loss / num_batches
    dist.all_reduce(loss)
    return loss / dist.get_world_size()


def train_kd(student_engine: DeepSpeedEngine,
             teacher_engine: InferenceEngine,
             dataloader: DataLoader,
             num_batches: Optional[int]):
    with in_train_mode(student_engine), in_eval_mode(teacher_engine):
        return _train_kd(student_engine, teacher_engine, dataloader, num_batches)


@autocast()
def _train_kd(student_engine: DeepSpeedEngine,
              teacher_engine: InferenceEngine,
              dataloader: DataLoader,
              num_batches: Optional[int]):
    if num_batches is None:
        num_batches = len(dataloader)

    if num_batches > len(dataloader):
        raise RuntimeError(
            f'num_batchs ({num_batches}) is larger than '
            f'the length of data loader ({len(dataloader)}).'
        )

    device = f"cuda:{LOCAL_RANK}"

    for _ in range(NUM_EPOCHS):
        for batch_id, batch in enumerate(tqdm(dataloader, total=num_batches, desc="train_kd")):
            if batch_id >= num_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_ids = attention_mask.cumsum(dim=1) - 1
            position_ids = position_ids.clip(0, SEQUENCE_LENGTH - 1).to(device)
            labels = batch.get("labels", input_ids).to(device)

            # Shift inputs and labels
            labels = labels[..., 1:]
            input_ids = input_ids[..., :-1]
            attention_mask = attention_mask[..., :-1]
            position_ids = position_ids[..., :-1]

            with torch.no_grad():
                teacher_logits, *_ = teacher_engine(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             position_ids=position_ids)

            with enable_recompute():
                student_logits, *_ = student_engine(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             position_ids=position_ids)

            teacher_prob = F.softmax(teacher_logits, dim=-1)
            student_log_prob = F.log_softmax(student_logits, dim=-1)
            kl_div = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
            ce_loss = F.cross_entropy(student_logits.flatten(0, 1), labels.flatten(0,1))
            loss = 0.5 * kl_div + 0.5 * ce_loss

            student_engine.backward(loss)
            student_engine.step()


@torch.no_grad()
def create_quant_sim(model, config_file, train_dataloader):
    dummy_input = next(iter(train_dataloader))['input_ids'].to(LOCAL_RANK)

    with ds.zero.GatheredParameters(model.parameters()):
        sim = QuantizationSimModel(model,
                                   dummy_input,
                                   default_output_bw=ACTIVATION_BITWIDTH,
                                   default_param_bw=WEIGHT_BITWIDTH,
                                   in_place=True,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                   config_file=config_file)

        for name, module in model.named_modules():
            if not isinstance(module, QcQuantizeWrapper):
                continue

            if isinstance(module._module_to_wrap, torch.nn.Embedding):
                # NOTE: Embedding weights are quantized in asymmetric 16 bit.
                #       We disable weight quantizer of embedding, assuming that
                #       float16 is a good enough proxy of int16.

                # module.param_quantizers['weight'].bitwidth = 16
                module.param_quantizers['weight'].enabled = False
                module.param_quantizers['weight'].use_symmetric_encodings = False

            if isinstance(module._module_to_wrap, torch.nn.LayerNorm) or\
                    name.endswith('norm'):
                # NOTE: Layernorm weights are quantized in asymmetric 16 bit.
                #       We disable weight quantizer of layernorms, assuming that
                #       float16 is a good enough proxy of int16.

                # NOTE: Modify the if condition as necessary since not all
                #       normalization layers may satisfy this condition.

                # module.param_quantizers['weight'].bitwidth = 16
                module.param_quantizers['weight'].enabled = False
                module.param_quantizers['weight'].use_symmetric_encodings = False

        # NOTE: LM head is quantized in 8 bit
        model.lm_head.param_quantizers['weight'].bitwidth = 8

        # NOTE: Disable all input/output quantizers, assuming that
        #       float16 is a good enough proxy of int16.
        _, input_quantizers, output_quantizers = get_all_quantizers(model)
        for quantizer in input_quantizers + output_quantizers:
            quantizer.enabled = False

        print(f'Computing encodings...')
        sim.compute_encodings(lambda model, _: evaluate(model, train_dataloader, NUM_BATCHES_FOR_CALIBRATION), None)

    handles = []
    for _, wrapper in sim.quant_wrappers():
        for encoding in wrapper.parameters(recurse=False):
            handle = dist.broadcast(encoding, src=0, async_op=True)
            handles.append(handle)

    for handle in handles:
        handle.wait()

    if version.parse(ds.__version__) >= version.parse("0.13.0"):
        # Deepspeed >= 0.13.0 requires users to make explicit the modules to be considered as leaf.
        # In our case, we want deepspeed to consider all quant wrappers as leaf.

        for _, wrapper in sim.quant_wrappers():
            if not isinstance(wrapper._module_to_wrap, torch.nn.Embedding):
                # For practical issues, deepspeed currently doesn't support setting modules as leaf
                # if it takes inputs that don't requires_grad.
                # Here, we exclude the quant wrappers of torch.nn.Embedding which takes int64 indices as
                # input which always doesn't require gradient.

                # NOTE: Additionally exclude modules that take inputs that don't requires_grad as necessary
                #       (e.g. nn.EmbeddingBag, modules that take leaf input, etc)
                ds.utils.set_z3_leaf_modules(wrapper, [LearnedGridQuantWrapper])

    return sim


def load_deepspeed_config_student():
    with open("deepspeed_config_student.json", "r") as f:
        config = json.load(f)

    if "train_batch_size" in config:
        raise RuntimeError(
            '"train_batch_size" in deepspeed config file will be ignored and '
            'overwritten by TRAIN_BATCH_SIZE. Please remove this from deepspeed config file.'
        )

    if "train_micro_batch_size_per_gpu" in config:
        raise RuntimeError(
            '"train_micro_batch_size_per_gpu" in deepspeed config file will be ignored and '
            'overwritten by TRAIN_MICRO_BATCH_SIZE_PER_GPU. Please remove this from deepspeed config file.'
        )

    if "gradient_accumulation_steps" in config:
        raise RuntimeError(
            '"gradient_accumulation_steps" in deepspeed config file will be ignored and '
            'calculated based on TRAIN_BATCH_SIZE and TRAIN_MICRO_BATCH_SIZE_PER_GPU. '
            'Please remove this from deepspeed config file.'
        )

    config.update({
        "train_batch_size": TRAIN_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": TRAIN_MICRO_BATCH_SIZE_PER_GPU,
        "gradient_accumulation_steps": TRAIN_BATCH_SIZE // (TRAIN_MICRO_BATCH_SIZE_PER_GPU * dist.get_world_size()),
    })
    return config


def load_deepspeed_config_teacher():
    with open("deepspeed_config_teacher.json", "r") as f:
        return json.load(f)


def main():
    ds.init_distributed(rank=RANK, world_size=WORLD_SIZE, distributed_port=MASTER_PORT)

    tokenizer = get_tokenizer()

    train_dataset = get_dataset('wikitext', 'wikitext-2-raw-v1', 'train')
    train_dataset = utils.preprocess_split(train_dataset, tokenizer, SEQUENCE_LENGTH)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=TRAIN_MICRO_BATCH_SIZE_PER_GPU,
                                  sampler=DistributedSampler(train_dataset, seed=TRAIN_DATALOADER_SEED),
                                  collate_fn=utils.default_data_collator)

    test_dataset = get_dataset('wikitext', 'wikitext-2-raw-v1', 'test')
    test_dataset = utils.preprocess_split(test_dataset, tokenizer, SEQUENCE_LENGTH)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=EVAL_MICRO_BATCH_SIZE_PER_GPU,
                                 sampler=DistributedSampler(test_dataset, shuffle=False),
                                 collate_fn=utils.default_data_collator)

    student = get_student_model().to(LOCAL_RANK)
    print(f"evaluating student model: {STUDENT_MODEL_NAME}")
    ce_loss = evaluate(student, test_dataloader, NUM_BATCHES_TO_EVAL)
    print(f"fp32 perplexity: {ce_loss.exp()}")

    sim = create_quant_sim(student, "default_per_channel_quantsim_config.json", train_dataloader)
    student_engine, optimizer, _, lr_scheduler = ds.initialize(model=sim.model, config=load_deepspeed_config_student())

    teacher = get_teacher_model().to(LOCAL_RANK)
    teacher_engine = ds.init_inference(model=teacher, config=load_deepspeed_config_teacher())

    print(f"evaluating quantsim student model.")
    ce_loss_before = evaluate(student_engine, test_dataloader, NUM_BATCHES_TO_EVAL)
    print(f"quantized perplexity (before training): {ce_loss_before.exp()}")

    student_engine.train()
    teacher_engine.eval()
    train_kd(student_engine, teacher_engine, train_dataloader, NUM_BATCHES_TO_TRAIN)

    ce_loss_after = evaluate(student_engine, test_dataloader, NUM_BATCHES_TO_EVAL)
    print(f"quantized perplexity (before training): {ce_loss_before.exp()}")
    print(f"quantized perplexity (after training): {ce_loss_after.exp()}")


if __name__ == "__main__":
    main()
