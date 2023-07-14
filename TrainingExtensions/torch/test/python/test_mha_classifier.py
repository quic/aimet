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

""" Unit tests to validate mha classifier """

from typing import Optional, List
import torch
from transformers import GPT2Config, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from aimet_torch.utils import replace_modules_of_type1_using_constructor
from aimet_torch.transformers.activation import QuantizableMultiheadAttention, create_quantizable_multihead_attention
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.transformers.mha_classifier import find_mha_variant


def _create_torch_mha_pattern(embed_dim, num_heads, seq_size, batch_size) -> List[str]:
    """
    Create pattern for torch MHA variant.
    A pattern is list of connected graph op types in order of occurence.

    :param embed_dim:
    :param num_heads:
    :param seq_size:
    :param batch_size:
    :return: List of op types in order of occurence.
    """
    key = torch.rand(seq_size, batch_size, embed_dim)
    query = torch.rand(seq_size, batch_size, embed_dim)
    value = torch.rand(seq_size, batch_size, embed_dim)
    dummy_input = (query, key, value)
    torch_mha = QuantizableMultiheadAttention(embed_dim, num_heads, bias=False)
    wrapped_torch_mha = TorchMhaWrapper(torch_mha, need_weights=False)
    conn_graph = ConnectedGraph(wrapped_torch_mha, dummy_input)
    pattern = [op.type for op in conn_graph.ordered_ops]
    return pattern


def _create_gpt2_mha_pattern() -> List[str]:
    """
    Create pattern for GPT2 MHA variant.
    A pattern is list of connected graph op types in order of occurence.
    :return: List of op types in order of occurence.
    """
    dummy_input = torch.randn(1, 1, 768)
    gpt2_mha = GPT2Attention(GPT2Config())
    wrapped_gpt2_mha = Gpt2MhaWrapper(gpt2_mha)
    conn_graph = ConnectedGraph(wrapped_gpt2_mha, dummy_input)
    pattern = [op.type for op in conn_graph.ordered_ops]
    return pattern


class TorchMhaWrapper(torch.nn.Module):
    """
    Torch MHA variant Wrapper which allows following:
    1) to be torch.jit.traceable because few output atrributes might be None.
    2) generate multiple patterns for dynamic control flow inside forward pass.
    """
    def __init__(self,
                 multihead_attn,
                 need_weights: bool = True,
                 attn_mask: Optional[torch.Tensor] = None,
                 average_attn_weights: bool = True):
        super().__init__()
        self.multihead_attn = multihead_attn
        self.need_weights = need_weights
        self.attn_mask = attn_mask
        self.average_attn_weights = average_attn_weights

    def forward(self, *arg, **kwargs):
        kwargs["need_weights"] = self.need_weights
        kwargs["attn_mask"] = self.attn_mask
        kwargs["average_attn_weights"] = self.average_attn_weights
        outputs = self.multihead_attn(*arg, **kwargs)
        outputs = [out for out in outputs if out is not None]
        return tuple(outputs)


class Gpt2MhaWrapper(torch.nn.Module):
    """
    GPT2 MHA variant Wrapper which allows following:
    1) to be torch.jit.traceable because few output atrributes might be None.
    2) generate multiple patterns for dynamic control flow inside forward pass.
    """
    def __init__(self,
                 multihead_attn,
                 layer_past=None,
                 attention_mask=None,
                 head_mask=None,
                 encoder_hidden_states=None,
                 encoder_attention_mask=None,
                 use_cache=False,
                 output_attentions=False
                 ):
        super(Gpt2MhaWrapper, self).__init__()
        self.multihead_attn = multihead_attn
        self.layer_past = layer_past
        self.attention_mask = attention_mask
        self.head_mask = head_mask
        self.encoder_hidden_states = encoder_hidden_states
        self.encoder_attention_mask = encoder_attention_mask
        self.use_cache = use_cache
        self.output_attentions = output_attentions

    def forward(self, *args, **kwargs):
        kwargs["layer_past"] = self.layer_past
        kwargs["attention_mask"] = self.attention_mask
        kwargs["head_mask"] = self.head_mask
        kwargs["encoder_hidden_states"] = self.encoder_hidden_states
        kwargs["encoder_attention_mask"] = self.encoder_attention_mask
        kwargs["use_cache"] = self.use_cache
        kwargs["output_attentions"] = self.output_attentions
        outputs = self.multihead_attn(*args, **kwargs)
        outputs = [out for out in outputs if out is not None]
        return tuple(outputs)


class TestMhaClassifier:

    def test_torch_mha_variant(self):
        """ find torch variant in given model """
        embed_dim = 128
        num_heads = 2
        batch_size = 1
        seq_size = 20
        num_encoder_layers = 6
        num_decoder_layers = 6
        pattern = _create_torch_mha_pattern(embed_dim, num_heads, seq_size, batch_size)
        model = torch.nn.Transformer(d_model=embed_dim, nhead=num_heads,
                                     num_encoder_layers=num_encoder_layers,
                                     num_decoder_layers=num_decoder_layers).eval()\
        # replace with Quantizable MHA
        replace_modules_of_type1_using_constructor(model,
                                                   torch.nn.MultiheadAttention,
                                                   create_quantizable_multihead_attention)
        src = torch.rand((seq_size, batch_size, embed_dim))
        tgt = torch.rand((seq_size, batch_size, embed_dim))
        dummy_input = (src, tgt)

        mha_info = find_mha_variant(model, dummy_input, pattern=pattern)

        # Verify number of found MHAs
        assert len(mha_info) == num_encoder_layers + 2 * num_decoder_layers
        for mha in mha_info:
            assert type(model.get_submodule(mha.module_qualified_name)) == mha.type
            assert isinstance(model.get_submodule(mha.module_qualified_name), QuantizableMultiheadAttention)

    def test_gpt2_mha_variant(self):
        """ find GPT2 variant in given model """
        config = GPT2Config()
        pattern = _create_gpt2_mha_pattern()
        model = GPT2Model(config)
        input_shape = (1, 768)
        dummy_input = torch.randint(1, input_shape)

        class Wrapper(torch.nn.Module):
            """
            Wrapper which allows
            1) GPT2Model() returns BaseModelOutputWithPastAndCrossAttentions() which has nested tuples.
            2) GPT2Model() few output atrributes might be None.
            """
            def __init__(self, model):
                super(Wrapper, self).__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                outputs = self.model(*args, **kwargs)
                hidden_states = outputs.last_hidden_state
                presents = outputs.past_key_values
                _ = outputs.hidden_states,
                _= outputs.attentions,
                _ = outputs.cross_attentions
                outputs = (hidden_states, presents)
                outputs = [out for out in outputs if out is not None]
                return tuple(outputs)

        wrapped_model = Wrapper(model)

        mha_info = find_mha_variant(wrapped_model, dummy_input, pattern=pattern)

        # Verify number of found MHAs
        assert len(mha_info) == config.n_layer
        for mha in mha_info:
            assert type(wrapped_model.get_submodule(mha.module_qualified_name)) == mha.type
            assert isinstance(wrapped_model.get_submodule(mha.module_qualified_name), GPT2Attention)
