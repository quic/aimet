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

import unittest
import pytest
import os

import tensorflow as tf
from packaging import version

import aimet_common.libpymo as libpymo
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.utils.constants import QuantizeOpIndices


if version.parse(tf.version.VERSION) >= version.parse("2.0"):
    import transformers
    from transformers import BertTokenizer, TFBertModel, BertConfig, DistilBertTokenizer, DistilBertConfig, \
        TFDistilBertModel

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@pytest.mark.tf2
class TransformerQuantizationAcceptanceTests(unittest.TestCase):
    def test_hf_bert_with_tokenizer(self):
        tf.compat.v1.reset_default_graph()

        tokenizer = BertTokenizer.from_pretrained('./data/huggingface/bert-base-uncased')

        configuration = BertConfig(num_hidden_layers=1)
        model = TFBertModel(configuration)

        input_ids = tf.keras.Input([512], dtype=tf.int32, name='input_ids')
        encoded = transformers.BatchEncoding({
            'input_ids': input_ids,
        })
        outputs = model(encoded)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        sim = QuantizationSimModel(sess, [input_ids.op.name], [outputs['pooler_output'].op.name], use_cuda=False)

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('tf_bert_model/bert/pooler/dense/Tanh_quantized:0')
            encoded_value = tokenizer('Hello World', return_tensors='np', padding='max_length')
            model_inputs = {
                sess.graph.get_tensor_by_name('input_ids:0'): encoded_value['input_ids']
            }
            sess.run(model_output, model_inputs)

        sim.compute_encodings(dummy_forward_pass, None)

        quant_ops = {op.name: op for op in sim.session.graph.get_operations() if op.type == 'QcQuantize'}
        quant_ops_to_check = []

        # Add Embedding quant ops
        embedding_path = 'tf_bert_model/bert/embeddings'
        embedding_add_1_quant = ('{}/add/add_1_quantized'.format(embedding_path), True)
        embedding_add_quant = ('{}/add/add_quantized'.format(embedding_path), True)
        embedding_token_quant = ('{}/Gather_2_quantized'.format(embedding_path), True)
        embedding_position_quant = ('{}/Identity_1_quantized'.format(embedding_path), True)
        embedding_word_quant = ('{}/Gather_quantized'.format(embedding_path), True)

        quant_ops_to_check += [embedding_add_1_quant, embedding_add_quant, embedding_word_quant, embedding_token_quant,
                               embedding_position_quant]

        # Add LayerNorm quant ops
        layernorm_paths = ['tf_bert_model/bert/embeddings', 'tf_bert_model/bert/encoder/layer_._0/attention/output',
                           'tf_bert_model/bert/encoder/layer_._0/output']

        for layernorm_path in layernorm_paths:
            output_quant_op = ('{}/LayerNorm/batchnorm/add_1_quantized'.format(layernorm_path), True)
            beta_quant_op = ('{}/LayerNorm/batchnorm/ReadVariableOp_quantized'.format(layernorm_path), False)
            gamma_quant_op = ('{}/LayerNorm/batchnorm/mul/ReadVariableOp_quantized'.format(layernorm_path), True)

            quant_ops_to_check += [output_quant_op, beta_quant_op, gamma_quant_op]

        # Add GeLU quant ops
        gelu_path = 'tf_bert_model/bert/encoder/layer_._0/intermediate'
        output_quant_op = ('{}/Gelu/mul_1_quantized'.format(gelu_path), True)

        quant_ops_to_check += [output_quant_op]

        # Add Query, Key, and Value quant ops
        self_attention_path = 'tf_bert_model/bert/encoder/layer_._0/attention/self'
        for dense_type in ['query', 'key', 'value']:
            output_quant_op = ('{}/{}/BiasAdd_quantized'.format(self_attention_path, dense_type), True)
            parameter_quant_op = ('{}/{}/Tensordot/ReadVariableOp_quantized'
                                  .format(self_attention_path, dense_type), True)
            bias_quant_op = ('{}/{}/BiasAdd/ReadVariableOp_quantized'.format(self_attention_path, dense_type), False)

            # No need to check input_quant_op as those are output quantize ops of layer norm
            quant_ops_to_check += [output_quant_op, parameter_quant_op, bias_quant_op]

        # Check if quant op exists and check encoding
        # Pop quantizer from quant_ops list if it is found
        for quant_op_name, enabled in quant_ops_to_check:
            quant_op = quant_ops.pop(quant_op_name)
            self.assertTrue(quant_op)
            self.assertTrue(check_encoding(quant_op, sim, enabled))

        # quant_ops should not contain any LayerNorm, GeLU, and QKV quant ops as those should have been popped
        for quant_op_name in quant_ops.keys():
            self.assertTrue(all(x not in quant_op_name for x in ['LayerNorm', 'Gelu', 'query', 'key', 'value']))

        del sim
        sess.close()

    def test_hf_distilbert_with_tokenizer(self):
        tf.compat.v1.reset_default_graph()

        tokenizer = DistilBertTokenizer.from_pretrained('./data/huggingface/distilbert-base-uncased')

        configuration = DistilBertConfig(n_layers=1)
        model = TFDistilBertModel(configuration)

        input_ids = tf.keras.Input([512], dtype=tf.int32, name='input_ids')
        encoded = transformers.BatchEncoding({
            'input_ids': input_ids,
        })
        outputs = model(encoded)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        sim = QuantizationSimModel(sess, [input_ids.op.name], [outputs['last_hidden_state'].op.name], use_cuda=False)

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name(outputs['last_hidden_state'].op.name + '_quantized:0')
            encoded_value = tokenizer('Hello World', return_tensors='np', padding='max_length')
            model_inputs = {
                sess.graph.get_tensor_by_name(input_ids.op.name + ':0'): encoded_value['input_ids']
            }
            sess.run(model_output, model_inputs)

        sim.compute_encodings(dummy_forward_pass, None)

        quant_ops = {op.name: op for op in sim.session.graph.get_operations() if op.type == 'QcQuantize'}
        quant_ops_to_check = []

        # Add LayerNorm quant ops
        layernorm_paths = ['tf_distil_bert_model/distilbert/embeddings/LayerNorm',
                           'tf_distil_bert_model/distilbert/transformer/layer_._0/sa_layer_norm',
                           'tf_distil_bert_model/distilbert/transformer/layer_._0/output_layer_norm']

        for layernorm_path in layernorm_paths:
            output_quant_op = ('{}/batchnorm/add_1_quantized'.format(layernorm_path), True)
            beta_quant_op = ('{}/batchnorm/ReadVariableOp_quantized'.format(layernorm_path), False)
            gamma_quant_op = ('{}/batchnorm/mul/ReadVariableOp_quantized'.format(layernorm_path), True)

            quant_ops_to_check += [output_quant_op, beta_quant_op, gamma_quant_op]

        # Add GeLU quant ops
        gelu_path = 'tf_distil_bert_model/distilbert/transformer/layer_._0/ffn/Gelu'
        output_quant_op = ('{}/mul_1_quantized'.format(gelu_path), True)

        quant_ops_to_check += [output_quant_op]

        # Add Query, Key, and Value quant ops
        self_attention_path = 'tf_distil_bert_model/distilbert/transformer/layer_._0/attention'
        for dense_type in ['q_lin', 'k_lin', 'v_lin']:
            output_quant_op = ('{}/{}/BiasAdd_quantized'.format(self_attention_path, dense_type), True)
            parameter_quant_op = ('{}/{}/Tensordot/ReadVariableOp_quantized'
                                  .format(self_attention_path, dense_type), True)
            bias_quant_op = ('{}/{}/BiasAdd/ReadVariableOp_quantized'.format(self_attention_path, dense_type), False)

            # No need to check input_quant_op as those are output quantize ops of layer norm
            quant_ops_to_check += [output_quant_op, parameter_quant_op, bias_quant_op]

        # Check if quant op exists and check encoding
        # Pop quantizer from quant_ops list if it is found
        for quant_op_name, enabled in quant_ops_to_check:
            quant_op = quant_ops.pop(quant_op_name)
            self.assertTrue(quant_op)
            self.assertTrue(check_encoding(quant_op, sim, enabled))

        # quant_ops should not contain any LayerNorm, GeLU, and QKV quant ops as those should have been popped
        for quant_op_name in quant_ops.keys():
            self.assertTrue(all(x not in quant_op_name for x in ['layer_norm', 'Gelu', 'q_lin', 'k_lin', 'v_lin']))

        del sim
        sess.close()

def check_quant_info(quant_info, enabled):
    # Check settings
    if quant_info.enabled != enabled or quant_info.rounding_mode != libpymo.RoundingMode.ROUND_NEAREST or \
            quant_info.quant_scheme != libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED or quant_info.bitwidth != 8:
        return False
    else:
        return True


def check_encoding(quant_op, sim, enabled):
    # Check encoding
    encoding_min = sim.session.run(quant_op.inputs[QuantizeOpIndices.encoding_min])
    encoding_max = sim.session.run(quant_op.inputs[QuantizeOpIndices.encoding_max])
    bit_width = sim.session.run(quant_op.inputs[QuantizeOpIndices.bit_width])

    if bit_width != 8:
        return False
    if (encoding_min != 0.0 or encoding_max != 0.0) == enabled:
        return True
    else:
        return False
