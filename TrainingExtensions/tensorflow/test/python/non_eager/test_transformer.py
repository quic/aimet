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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import tensorflow as tf
import numpy as np
from packaging import version

import aimet_common.libpymo as libpymo
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.common.graph_eval import initialize_uninitialized_vars
from aimet_tensorflow.utils import transformer_utils
from aimet_tensorflow.utils.constants import QuantizeOpIndices
from aimet_tensorflow.utils.graph_saver import load_model_from_meta
from aimet_tensorflow.quantsim import save_checkpoint, load_checkpoint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.compat.v1.disable_eager_execution()

if version.parse(tf.version.VERSION) >= version.parse("2.0"):
    import transformers
    from transformers import BertConfig, TFBertModel, DistilBertConfig, TFDistilBertModel, activations_tf


@pytest.mark.tf2
class TransformerQuantizationUnittests(unittest.TestCase):
    def test_hf_bert(self):
        tf.compat.v1.reset_default_graph()

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

        sim.compute_encodings(random_input_forward_pass, {
            'input_tensor': 'input_ids:0',
            'input_shape': (1, 512),
            'output_tensor': 'tf_bert_model/bert/pooler/dense/Tanh_quantized:0',
            'int': True
        })

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

    def test_hf_bert_all_inputs(self):
        tf.compat.v1.reset_default_graph()

        configuration = BertConfig(num_hidden_layers=1)
        model = TFBertModel(configuration)

        input_ids = tf.keras.Input([512], dtype=tf.int32, name='input_ids')
        position_ids = tf.keras.Input([512], dtype=tf.int32, name='position_ids')
        token_type_ids = tf.keras.Input([512], dtype=tf.int32, name='token_type_ids')
        attention_mask = tf.keras.Input([512], dtype=tf.int32, name='attention_mask')
        encoded = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        outputs = model(encoded)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        sim = QuantizationSimModel(sess, [input_ids.op.name, position_ids.op.name, token_type_ids.op.name,
                                          attention_mask.op.name], [outputs['pooler_output'].op.name], use_cuda=False)

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('tf_bert_model/bert/pooler/dense/Tanh_quantized:0')
            np.random.seed(0)
            model_inputs = {
                sess.graph.get_tensor_by_name('input_ids:0'): np.random.randint(16384, size=[1, 512]),
                sess.graph.get_tensor_by_name('position_ids:0'): np.random.randint(256, size=[1, 512]),
                sess.graph.get_tensor_by_name('token_type_ids:0'): np.random.randint(1, size=[1, 512]),
                sess.graph.get_tensor_by_name('attention_mask:0'): np.random.randint(1, size=[1, 512]),
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

    def test_export_hf_bert(self):
        tf.compat.v1.reset_default_graph()

        configuration = BertConfig(num_hidden_layers=1)
        model = TFBertModel(configuration)

        input_tensor = tf.keras.Input([512], dtype=tf.int32, name='input_ids')
        encoded = transformers.BatchEncoding({
            'input_ids': input_tensor,
        })
        output_tensor = model(encoded)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        sim = QuantizationSimModel(sess, [input_tensor.op.name], [output_tensor['pooler_output'].op.name], use_cuda=False)

        sim.compute_encodings(random_input_forward_pass, {
            'input_tensor': 'input_ids:0',
            'input_shape': (1, 512),
            'output_tensor': 'tf_bert_model/bert/pooler/dense/Tanh_quantized:0',
            'int': True
        })

        sim.export('/tmp', 'tf_hf_bert_model')

        with open('/tmp/tf_hf_bert_model.encodings') as json_file:
            encoding_data = json.load(json_file)

        activation_tensors = []
        parameter_tensors = []
        bias_tensors = []

        # Add Embedding quant ops
        embedding_path = 'tf_bert_model/bert/embeddings'
        embedding_add_1_tensor = '{}/add/add_1:0'.format(embedding_path)
        embedding_add_tensor = '{}/add/add:0'.format(embedding_path)
        embedding_token_tensor = '{}/Gather_2:0'.format(embedding_path)
        embedding_position_tensor = '{}/Identity_1:0'.format(embedding_path)
        embedding_word_tensor = '{}/Gather:0'.format(embedding_path)

        activation_tensors += [embedding_add_tensor, embedding_add_1_tensor]
        parameter_tensors += [embedding_token_tensor, embedding_word_tensor, embedding_position_tensor]

        # Add LayerNorm quant ops
        layernorm_paths = ['tf_bert_model/bert/embeddings', 'tf_bert_model/bert/encoder/layer_._0/attention/output',
                           'tf_bert_model/bert/encoder/layer_._0/output']

        for layernorm_path in layernorm_paths:
            output_tensor = '{}/LayerNorm/batchnorm/add_1:0'.format(layernorm_path)
            beta_tensor = '{}/LayerNorm/batchnorm/ReadVariableOp:0'.format(layernorm_path)
            gamma_tensor = '{}/LayerNorm/batchnorm/mul/ReadVariableOp:0'.format(layernorm_path)

            activation_tensors += [output_tensor]
            parameter_tensors += [gamma_tensor]
            bias_tensors += [beta_tensor]

        # Add GeLU quant ops
        gelu_path = 'tf_bert_model/bert/encoder/layer_._0/intermediate'
        output_tensor = '{}/Gelu/mul_1:0'.format(gelu_path)

        activation_tensors += [output_tensor]

        # Add Query, Key, and Value quant ops
        self_attention_path = 'tf_bert_model/bert/encoder/layer_._0/attention/self'
        for dense_type in ['query', 'key', 'value']:
            output_tensor = '{}/{}/BiasAdd:0'.format(self_attention_path, dense_type)
            weight_tensor = '{}/{}/Tensordot/ReadVariableOp:0'.format(self_attention_path, dense_type)
            bias_tensor = '{}/{}/BiasAdd/ReadVariableOp:0'.format(self_attention_path, dense_type)

            # No need to check input_quant_op as those are output quantize ops of layer norm
            activation_tensors += [output_tensor]
            parameter_tensors += [weight_tensor]
            bias_tensors += [bias_tensor]

        for activation_tesnor in activation_tensors:
            self.assertTrue(activation_tesnor in encoding_data['activation_encodings'])
            act_encoding_keys = encoding_data['activation_encodings'][activation_tesnor][0].keys()
            self.assertTrue('bitwidth' in act_encoding_keys)
            self.assertTrue('is_symmetric' in act_encoding_keys)
            self.assertTrue('max' in act_encoding_keys)
            self.assertTrue('min' in act_encoding_keys)
            self.assertTrue('offset' in act_encoding_keys)
            self.assertTrue('scale' in act_encoding_keys)

        for parameter_tensor in parameter_tensors:
            self.assertTrue(parameter_tensor in encoding_data['param_encodings'])
            param_encoding_keys = encoding_data['param_encodings'][parameter_tensor][0].keys()
            self.assertTrue('bitwidth' in param_encoding_keys)
            self.assertTrue('is_symmetric' in param_encoding_keys)
            self.assertTrue('max' in param_encoding_keys)
            self.assertTrue('min' in param_encoding_keys)
            self.assertTrue('offset' in param_encoding_keys)
            self.assertTrue('scale' in param_encoding_keys)

        new_sess = load_model_from_meta('/tmp/tf_hf_bert_model.meta')

        all_op_types = [op.type for op in new_sess.graph.get_operations()]
        self.assertNotIn('QcQuantize', all_op_types)

        del sim
        sess.close()

        if os.path.exists('/tmp/tf_hf_bert_model'):
            os.remove('/tmp/tf_hf_bert_model')

    def test_save_load_chkpt_hf_bert(self):
        tf.compat.v1.reset_default_graph()

        configuration = BertConfig(num_hidden_layers=1)
        model = TFBertModel(configuration)

        input_tensor = tf.keras.Input([512], dtype=tf.int32, name='input_ids')
        encoded = transformers.BatchEncoding({
            'input_ids': input_tensor,
        })
        output_tensor = model(encoded)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        sim = QuantizationSimModel(sess, [input_tensor.op.name], [output_tensor['pooler_output'].op.name], use_cuda=False)

        save_checkpoint(sim, '/tmp', 'orig_tf_hf_bert_model')
        new_quantsim = load_checkpoint('/tmp', 'orig_tf_hf_bert_model')

        self.assertNotEqual(sim, new_quantsim)
        self.assertTrue(new_quantsim.session)
        self.assertEqual(new_quantsim._quant_scheme, sim._quant_scheme)
        self.assertEqual(new_quantsim._rounding_mode, sim._rounding_mode)
        self.assertEqual(new_quantsim._use_cuda, sim._use_cuda)
        self.assertEqual(len(new_quantsim._param_quantizers), len(sim._param_quantizers))
        self.assertEqual(len(new_quantsim._activation_quantizers), len(sim._activation_quantizers))

        for quantize_op in new_quantsim._param_quantizers:
            self.assertNotEqual(sim._param_quantizers[quantize_op].session,
                                new_quantsim._param_quantizers[quantize_op].session)
            self.assertEqual(sim._param_quantizers[quantize_op].tensor_quantizer.getQuantScheme(),
                             new_quantsim._param_quantizers[quantize_op].tensor_quantizer.getQuantScheme())
            self.assertEqual(sim._param_quantizers[quantize_op].tensor_quantizer.roundingMode,
                             new_quantsim._param_quantizers[quantize_op].tensor_quantizer.roundingMode)
            self.assertFalse(sim._param_quantizers[quantize_op].tensor_quantizer.isEncodingValid)
            self.assertFalse(new_quantsim._param_quantizers[quantize_op].tensor_quantizer.isEncodingValid)

        for quantize_op in new_quantsim._activation_quantizers:
            self.assertNotEqual(sim._activation_quantizers[quantize_op].session,
                                new_quantsim._activation_quantizers[quantize_op].session)
            self.assertEqual(sim._activation_quantizers[quantize_op].tensor_quantizer.getQuantScheme(),
                             new_quantsim._activation_quantizers[quantize_op].tensor_quantizer.getQuantScheme())
            self.assertEqual(sim._activation_quantizers[quantize_op].tensor_quantizer.roundingMode,
                             new_quantsim._activation_quantizers[quantize_op].tensor_quantizer.roundingMode)
            self.assertFalse(sim._activation_quantizers[quantize_op].tensor_quantizer.isEncodingValid)
            self.assertFalse(new_quantsim._activation_quantizers[quantize_op].tensor_quantizer.isEncodingValid)

        del sim
        del new_quantsim
        sess.close()

        if os.path.exists('/tmp/orig_tf_hf_bert_model'):
            os.remove('/tmp/orig_tf_hf_bert_model')

    def test_custom_hw_config_hf_bert(self):
        tf.compat.v1.reset_default_graph()

        configuration = BertConfig(num_hidden_layers=1)
        model = TFBertModel(configuration)

        input_tensor = tf.keras.Input([512], dtype=tf.int32, name='input_ids')
        encoded = transformers.BatchEncoding({
            'input_ids': input_tensor,
        })
        output_tensor = model(encoded)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        file_path = "/tmp/tf_hf_custom_quantsim_config.json"
        generate_custom_quantsim_config(file_path)

        sim = QuantizationSimModel(sess, [input_tensor.op.name], [output_tensor['pooler_output'].op.name],
                                   use_cuda=False, config_file=file_path)

        sim.compute_encodings(random_input_forward_pass, {
            'input_tensor': 'input_ids:0',
            'input_shape': (1, 512),
            'output_tensor': 'tf_bert_model/bert/pooler/dense/Tanh_quantized:0',
            'int': True
        })

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
            output_quant_op = ('{}/LayerNorm/batchnorm/add_1_quantized'.format(layernorm_path), False)
            beta_quant_op = ('{}/LayerNorm/batchnorm/ReadVariableOp_quantized'.format(layernorm_path), False)
            gamma_quant_op = ('{}/LayerNorm/batchnorm/mul/ReadVariableOp_quantized'.format(layernorm_path), False)

            quant_ops_to_check += [output_quant_op, beta_quant_op, gamma_quant_op]

        # Add GeLU quant ops
        gelu_path = 'tf_bert_model/bert/encoder/layer_._0/intermediate'
        output_quant_op = ('{}/Gelu/mul_1_quantized'.format(gelu_path), False)

        quant_ops_to_check += [output_quant_op]

        # Add Query, Key, and Value quant ops
        self_attention_path = 'tf_bert_model/bert/encoder/layer_._0/attention/self'
        for dense_type in ['query', 'key', 'value']:
            output_quant_op = ('{}/{}/BiasAdd_quantized'.format(self_attention_path, dense_type), False)
            parameter_quant_op = ('{}/{}/Tensordot/ReadVariableOp_quantized'
                                  .format(self_attention_path, dense_type), False)
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

        if os.path.exists(file_path):
            os.remove(file_path)

    def test_batching_hf_bert(self):
        tf.compat.v1.reset_default_graph()

        configuration = BertConfig(num_hidden_layers=1)
        model = TFBertModel(configuration)

        input_tensor = tf.keras.Input([512], dtype=tf.int32, name='input_ids')
        encoded = transformers.BatchEncoding({
            'input_ids': input_tensor,
        })
        output_tensor = model(encoded)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        attention_tensor = sess.graph.get_tensor_by_name(
            'tf_bert_model/bert/encoder/layer_._0/attention/output/LayerNorm/batchnorm/add_1:0')
        matmul_tensor = sess.graph.get_tensor_by_name('tf_bert_model/bert/encoder/layer_._0/attention/self/MatMul_1:0')
        np.random.seed(0)
        random_input = np.random.randint(16384, size=(2, 512))

        attention_result, matmul_result, result = sess.run([attention_tensor, matmul_tensor, output_tensor], {
            input_tensor: random_input
        })

        config_batch, config_max_seq_len = random_input.shape
        config_hidden = configuration.hidden_size
        config_num_attention = configuration.num_attention_heads
        config_attention_head_size = int(config_hidden / config_num_attention)

        self.assertEqual(attention_result.shape, (config_batch, config_max_seq_len, config_hidden))
        self.assertEqual(matmul_result.shape,
                         (config_batch, config_num_attention, config_max_seq_len, config_attention_head_size))

        sess.close()

    def test_hf_bert_mask(self):
        tf.compat.v1.reset_default_graph()

        configuration = BertConfig(num_hidden_layers=1)
        model = TFBertModel(configuration)

        input_ids_tensor = tf.keras.Input([512], dtype=tf.int32, name='input_ids')
        input_masks_tensor = tf.keras.Input([512], dtype=tf.int32, name='input_masks')

        encoded = transformers.BatchEncoding({
            'input_ids': input_ids_tensor,
            'attention_mask': input_masks_tensor
        })
        output_tensor = model(encoded)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        sim = QuantizationSimModel(sess, [input_ids_tensor.op.name, input_masks_tensor.op.name],
                                   [output_tensor['pooler_output'].op.name], use_cuda=False)

        def dummy_forward_pass(sess, args):
            model_output = sess.graph.get_tensor_by_name('tf_bert_model/bert/pooler/dense/Tanh_quantized:0')
            np.random.seed(0)
            model_inputs = {
                sess.graph.get_tensor_by_name('input_ids:0'): np.random.randint(16384, size=[1, 512]),
                sess.graph.get_tensor_by_name('input_masks:0'): np.random.randint(1, size=[1, 512])
            }
            sess.run(model_output, model_inputs)

        sim.compute_encodings(dummy_forward_pass, None)

        mask_add = 'tf_bert_model/bert/encoder/layer_._0/attention/self/Add_quantized'
        mask_quantizer = sim.quantizer_config(mask_add)
        self.assertTrue(mask_quantizer)
        encoding_min = mask_quantizer.get_variable_from_op(QuantizeOpIndices.encoding_min)
        self.assertAlmostEqual(encoding_min, transformer_utils.MASK_OVERRIDE_VALUE, places=1)
        mask_bitwidth = mask_quantizer.get_variable_from_op(QuantizeOpIndices.bit_width)
        self.assertEqual(mask_bitwidth, sim._default_output_bw)

        del sim
        sess.close()

    def test_hf_distilbert(self):
        tf.compat.v1.reset_default_graph()

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

        sim.compute_encodings(random_input_forward_pass, {
            'input_tensor': 'input_ids:0',
            'input_shape': (1, 512),
            'output_tensor': outputs['last_hidden_state'].op.name + '_quantized:0',
            'int': True
        })

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

    def test_layernorm_quantization(self):
        tf.compat.v1.reset_default_graph()

        inputs = tf.keras.Input([128])
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-5)(inputs)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, [inputs.op.name], [outputs.op.name])

        sim.compute_encodings(random_input_forward_pass, {
            'input_tensor': 'input_1:0',
            'input_shape': (1, 128),
            'output_tensor': 'layer_normalization/batchnorm/add_1_quantized:0',
            'int': False
        })

        ln_output_name = 'layer_normalization/batchnorm/add_1_quantized'
        beta_name = 'layer_normalization/batchnorm/ReadVariableOp_quantized'
        gamma_name = 'layer_normalization/batchnorm/mul/ReadVariableOp_quantized'

        for name, quant_info in sim._activation_quantizers.items():
            if name == ln_output_name:
                self.assertTrue(check_quant_info(quant_info, True))
            else:
                self.assertEqual(name, 'input_1_quantized')

        for name, quant_info in sim._param_quantizers.items():
            if name == gamma_name:
                self.assertTrue(check_quant_info(quant_info, True))
            elif name == beta_name:
                self.assertTrue(check_quant_info(quant_info, False))
            else:
                self.assertTrue(0)

        del sim
        sess.close()

    def test_huggingface_gelu_quantization(self):
        tf.compat.v1.reset_default_graph()

        inputs = tf.keras.Input([28, 28, 3])
        x = tf.keras.layers.Conv2D(32, kernel_size=3)(inputs)

        with tf.name_scope('gelu'):
            outputs = activations_tf._gelu(x)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)
        sim = QuantizationSimModel(sess, [inputs.op.name], [outputs.op.name])

        sim.compute_encodings(random_input_forward_pass, {
            'input_tensor': 'input_1:0',
            'input_shape': (1, 28, 28, 3),
            'output_tensor': 'gelu/mul_1_quantized:0',
            'int': False
        })

        gelu_output_name = 'gelu/mul_1_quantized'
        for name, quant_info in sim._activation_quantizers.items():
            if name == gelu_output_name:
                self.assertTrue(check_quant_info(quant_info, True))
            else:
                self.assertTrue(name in ['input_1_quantized', 'conv2d/BiasAdd_quantized'])

        del sim
        sess.close()

    def test_hf_bert_embedding(self):
        tf.compat.v1.reset_default_graph()

        input_ids = tf.keras.Input([512], dtype=tf.int32, name='input_ids')
        token_type_ids = tf.constant(0, dtype=tf.int32, shape=(1, 512))
        embedding = transformers.models.bert.modeling_tf_bert.TFBertEmbeddings(BertConfig())

        outputs = embedding(input_ids=input_ids, token_type_ids=token_type_ids)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        sim = QuantizationSimModel(sess, [input_ids.op.name, token_type_ids.op.name], [outputs.op.name], use_cuda=False)

        sim.compute_encodings(random_input_forward_pass, {
            'input_tensor': 'input_ids:0',
            'input_shape': (1, 512),
            'output_tensor': 'tf_bert_embeddings/dropout/Identity:0',
            'int': True
        })

        quant_ops = {op.name: op for op in sim.session.graph.get_operations() if op.type == 'QcQuantize'}
        quant_ops_to_check = []

        layernorm_quant = 'tf_bert_embeddings/LayerNorm/batchnorm/add_1_quantized'
        add_1_quant = 'tf_bert_embeddings/add/add_1_quantized'
        add_quant = 'tf_bert_embeddings/add/add_quantized'

        quant_ops_to_check += [(layernorm_quant, True), (add_1_quant, True), (add_quant, True)]

        token_quant = 'tf_bert_embeddings/Gather_2_quantized'
        position_quant = 'tf_bert_embeddings/Identity_1_quantized'
        word_quant = 'tf_bert_embeddings/Gather_quantized'
        beta_quant = 'tf_bert_embeddings/LayerNorm/batchnorm/ReadVariableOp_quantized'
        gamma_quant = 'tf_bert_embeddings/LayerNorm/batchnorm/mul/ReadVariableOp_quantized'

        quant_ops_to_check += [(token_quant, True), (position_quant, True), (word_quant, True), (gamma_quant, True),
                               (beta_quant, False)]

        # Check if quant op exists and check encoding
        # Pop quantizer from quant_ops list if it is found
        for quant_op_name, enabled in quant_ops_to_check:
            quant_op = quant_ops.pop(quant_op_name)
            self.assertTrue(quant_op)
            self.assertTrue(check_encoding(quant_op, sim, enabled))

        self.assertTrue(len(quant_ops) == 0)

        del sim
        sess.close()

    def test_hf_bert_embedding_custom_config(self):
        tf.compat.v1.reset_default_graph()

        input_ids = tf.keras.Input([512], dtype=tf.int32, name='input_ids')
        token_type_ids = tf.constant(0, dtype=tf.int32, shape=(1, 512))
        embedding = transformers.models.bert.modeling_tf_bert.TFBertEmbeddings(BertConfig())

        outputs = embedding(input_ids=input_ids, token_type_ids=token_type_ids)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        file_path = "/tmp/tf_hf_custom_quantsim_config.json"
        generate_custom_quantsim_config(file_path)

        sim = QuantizationSimModel(sess, [input_ids.op.name, token_type_ids.op.name], [outputs.op.name], use_cuda=False, config_file=file_path)

        sim.compute_encodings(random_input_forward_pass, {
            'input_tensor': 'input_ids:0',
            'input_shape': (1, 512),
            'output_tensor': 'tf_bert_embeddings/dropout/Identity:0',
            'int': True
        })

        quant_ops = {op.name: op for op in sim.session.graph.get_operations() if op.type == 'QcQuantize'}
        quant_ops_to_check = []

        layernorm_quant = 'tf_bert_embeddings/LayerNorm/batchnorm/add_1_quantized'
        add_1_quant = 'tf_bert_embeddings/add/add_1_quantized'
        add_quant = 'tf_bert_embeddings/add/add_quantized'

        quant_ops_to_check += [(layernorm_quant, False), (add_1_quant, True), (add_quant, True)]

        token_quant = 'tf_bert_embeddings/Gather_2_quantized'
        position_quant = 'tf_bert_embeddings/Identity_1_quantized'
        word_quant = 'tf_bert_embeddings/Gather_quantized'
        beta_quant = 'tf_bert_embeddings/LayerNorm/batchnorm/ReadVariableOp_quantized'
        gamma_quant = 'tf_bert_embeddings/LayerNorm/batchnorm/mul/ReadVariableOp_quantized'

        quant_ops_to_check += [(token_quant, True), (position_quant, True), (word_quant, True), (gamma_quant, False),
                               (beta_quant, False)]

        # Check if quant op exists and check encoding
        # Pop quantizer from quant_ops list if it is found
        for quant_op_name, enabled in quant_ops_to_check:
            quant_op = quant_ops.pop(quant_op_name)
            self.assertTrue(quant_op)
            self.assertTrue(check_encoding(quant_op, sim, enabled))

        self.assertTrue(len(quant_ops) == 0)

        del sim
        sess.close()

    @pytest.mark.skip
    def test_hf_bert_embedding_multiple_readvar(self):
        tf.compat.v1.reset_default_graph()

        input_ids = tf.keras.Input([512], dtype=tf.int32, name='input_ids')
        token_type_ids = tf.constant(0, dtype=tf.int32, shape=(1, 512))
        embedding = transformers.models.bert.modeling_tf_bert.TFBertEmbeddings(BertConfig())

        outputs = embedding(input_ids=input_ids, token_type_ids=token_type_ids)
        additional_op1 = tf.math.add(embedding.weight, 1)
        additional_op2 = tf.math.add(embedding.weight, 1)

        sess = tf.compat.v1.Session()
        initialize_uninitialized_vars(sess)

        sim = QuantizationSimModel(sess, [input_ids.op.name, token_type_ids.op.name], [outputs.op.name], use_cuda=False)

        sim.compute_encodings(random_input_forward_pass, {
            'input_tensor': 'input_ids:0',
            'input_shape': (1, 512),
            'output_tensor': 'tf_bert_embeddings/dropout/Identity:0',
            'int': True
        })

        quant_ops = {op.name: op for op in sim.session.graph.get_operations() if op.type == 'QcQuantize'}
        quant_ops_to_check = []

        layernorm_quant = 'tf_bert_embeddings/LayerNorm/batchnorm/add_1_quantized'
        add_1_quant = 'tf_bert_embeddings/add/add_1_quantized'
        add_quant = 'tf_bert_embeddings/add/add_quantized'

        quant_ops_to_check += [(layernorm_quant, True), (add_1_quant, True), (add_quant, True)]

        token_quant = 'tf_bert_embeddings/Gather_2_quantized'
        position_quant = 'tf_bert_embeddings/Identity_1_quantized'
        word_quant = 'tf_bert_embeddings/Gather_quantized'
        beta_quant = 'tf_bert_embeddings/LayerNorm/batchnorm/ReadVariableOp_quantized'
        gamma_quant = 'tf_bert_embeddings/LayerNorm/batchnorm/mul/ReadVariableOp_quantized'

        quant_ops_to_check += [(token_quant, True), (position_quant, True), (word_quant, True), (gamma_quant, True),
                               (beta_quant, False)]

        # Check if quant op exists and check encoding
        # Pop quantizer from quant_ops list if it is found
        for quant_op_name, enabled in quant_ops_to_check:
            quant_op = quant_ops.pop(quant_op_name)
            self.assertTrue(quant_op)
            self.assertTrue(check_encoding(quant_op, sim, enabled))

        self.assertTrue(len(quant_ops) == 0)

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


def random_input_forward_pass(sess, args):
    np.random.seed(0)
    model_output = sess.graph.get_tensor_by_name(args['output_tensor'])
    data = np.random.randint(16384, size=args['input_shape']) if args['int'] else np.random.rand(*args['input_shape'])
    model_inputs = {
        sess.graph.get_tensor_by_name(args['input_tensor']): data
    }
    sess.run(model_output, model_inputs)


def generate_custom_quantsim_config(file_path: str):
    """
    Helper method to generate custom config for transformer models
    :return:
    """

    custom_quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True"
            },
            "params": {
                "is_quantized": "True"
            }
        },
        "params": {},
        "op_type": {
            "LayerNorm": {
                "is_output_quantized": "False",
                "params": {
                    "weight": {
                        "is_quantized": "False"
                    },
                    "bias": {
                        "is_quantized": "False"
                    }
                }
            },
            "GeLU": {
                "is_output_quantized": "False"
            },
            "Gemm": {
                "is_output_quantized": "False",
                "params": {
                    "weight": {
                        "is_quantized": "False"
                    },
                    "bias": {
                        "is_quantized": "False"
                    }
                }
            }
        },
        "supergroups": [],
        "model_input": {},
        "model_output": {}
    }

    with open(file_path, 'w') as f:
        json.dump(custom_quantsim_config, f)
