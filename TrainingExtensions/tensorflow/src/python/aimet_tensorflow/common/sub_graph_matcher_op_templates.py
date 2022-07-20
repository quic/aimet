# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" This file contains op pattern templates that are used to generate reference
sub-graph patterns to detect Op sub-graphs in a given Tensorflow model graph.
(sub-graph module detection described above is used to generate an intermediate
representation called Connected graph) """

# This dictionary is used to generate various op type pattern sub-graphs used for module detection.
# Dictionary mapping names of op types identified by a unique string
# to a tuple of input shape into the op, and the constructor for the op.
# Note that 'inputs' is the name of an input op that is instantiated with shape of the input shape.
# 'Constants' is the name of a constant op that is instantiated with shape of the input shape.
op_type_templates = {
    'Conv2D': {
        'input_shape': (1, 10, 10, 3),
        'op_type': 'Conv2D',
        'constructor': "tf.keras.layers.Conv2D(10, (1, 1), use_bias=False)(constants)",
        'module_regex': ['(.+/Conv2D)$', '(.+/separable_conv2d)$', '(.+/convolution)$'],
        'associated_op_regex': ['Conv2D$', 'separable_conv2d$', 'convolution$'],
        'supported_tf_versions': [1, 2]
    },
    'Conv2D_with_bias': {
        'input_shape': (1, 10, 10, 3),
        'op_type': 'Conv2D',
        'constructor': "tf.keras.layers.Conv2D(10, (1, 1), use_bias=True)(constants)",
        'module_regex': ['(.+/Conv2D)$', '(.+/separable_conv2d)$', '(.+/convolution)$'],
        'associated_op_regex': ['Conv2D$', 'separable_conv2d$', 'convolution$'],
        'supported_tf_versions': [1, 2]
    },
    'DepthwiseConv2dNative': {
        'input_shape': (1, 10, 10, 3),
        'op_type': 'DepthwiseConv2dNative',
        'constructor': "tf.keras.layers.DepthwiseConv2D(3, (1, 1))(constants)",
        'module_regex': ['(.+/depthwise)$', '(.+/DepthwiseConv2dNative)$'],
        'associated_op_regex': ['depthwise$', 'DepthwiseConv2dNative$'],
        'supported_tf_versions': [1, 2]
    },
    'Dense': {
        'input_shape': (1, 10),
        'op_type': 'Dense',
        'constructor': "tf.keras.layers.Dense(10, activation=None)(constants)",
        'module_regex': ['(.+/MatMul)$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1, 2]
    },
    'Dense_tensordot_known_shape': {
        'input_shape': (1, 10, 10),
        'op_type': 'Dense',
        'constructor': "tf.keras.layers.Dense(10, activation=None)(constants)",
        'module_regex': ['(.+/MatMul)$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1, 2]
    },
    'Dense_tensordot_unknown_shape': {
        'input_shape': (10, 10),
        'op_type': 'Dense',
        'constructor': "tf.keras.layers.Dense(10, activation=None)(inputs)",
        'module_regex': ['(.+/MatMul)$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1, 2]
    },
    'BN_keras_with_training_tensor': {
        'input_shape': (10, 10, 3,),
        'op_type': 'FusedBatchNormV3',
        'constructor': "tf.keras.layers.BatchNormalization()(inputs)",
        'module_regex': ['(.+)/cond/FusedBatchNormV3_1$', '(.+)/cond/Identity'],
        'associated_op_regex': ['FusedBatchNormV3_1$', 'cond/Identity$'],
        'supported_tf_versions': [1, 2]
    },
    'BN_keras_with_training_True': {
        'input_shape': (10, 10, 3,),
        'op_type': 'FusedBatchNormV3',
        'constructor': "tf.keras.layers.BatchNormalization()(inputs, training=True)",
        'module_regex': ['(.+)/FusedBatchNormV3$'],
        'associated_op_regex': ['FusedBatchNormV3$'],
        'supported_tf_versions': [1, 2]
    },
    'BN_keras_with_training_False': {
        'input_shape': (10, 10, 3,),
        'op_type': 'FusedBatchNormV3',
        'constructor': "tf.keras.layers.BatchNormalization()(inputs, training=False)",
        'module_regex': ['(.+)/FusedBatchNormV3$'],
        'associated_op_regex': ['FusedBatchNormV3$'],
        'supported_tf_versions': [1, 2]
    },
    'BN_non_fused_keras_with_training_tensor': {
        'input_shape': (10, 10, 3,),
        'op_type': 'BatchNorm',
        'constructor': "tf.keras.layers.BatchNormalization(fused=False)(inputs)",
        'module_regex': ['(.+)/batchnorm/mul_1$'],
        'associated_op_regex': ['batchnorm/mul_1$'],
        'supported_tf_versions': [1, 2]
    },
    'BN_non_fused_keras_with_training_True': {
        'input_shape': (10, 10, 3,),
        'op_type': 'BatchNorm',
        'constructor': "tf.keras.layers.BatchNormalization(fused=False)(inputs, training=True)",
        'module_regex': ['(.+)/batchnorm/mul_1$'],
        'associated_op_regex': ['batchnorm/mul_1$'],
        'supported_tf_versions': [1, 2]
    },
    'BN_non_fused_keras_with_training_False': {
        'input_shape': (10, 10, 3,),
        'op_type': 'BatchNorm',
        'constructor': "tf.keras.layers.BatchNormalization(fused=False)(inputs, training=False)",
        'module_regex': ['(.+)/batchnorm/mul_1$'],
        'associated_op_regex': ['batchnorm/mul_1$'],
        'additional_starting_ops': ['batch_normalization/batchnorm/mul'],
        'supported_tf_versions': [1, 2]
    },
    'BN_slim_with_training_tensor': {
        'input_shape': (10, 10, 3,),
        'op_type': 'FusedBatchNormV3',
        'constructor': "slim.batch_norm(inputs, is_training=is_training)",
        'module_regex': ['(.+)/cond/FusedBatchNormV3_1$'],
        'associated_op_regex': ['FusedBatchNormV3_1$'],
        'supported_tf_versions': [1]
    },
    'BN_slim_with_training_True': {
        'input_shape': (10, 10, 3,),
        'op_type': 'FusedBatchNormV3',
        'constructor': "slim.batch_norm(inputs, is_training=True)",
        'module_regex': ['(.+)/FusedBatchNormV3$'],
        'associated_op_regex': ['FusedBatchNormV3$'],
        'supported_tf_versions': [1]
    },
    'BN_slim_with_training_False': {
        'input_shape': (10, 10, 3,),
        'op_type': 'FusedBatchNormV3',
        'constructor': "slim.batch_norm(inputs, is_training=False)",
        'module_regex': ['(.+)/FusedBatchNormV3$'],
        'associated_op_regex': ['FusedBatchNormV3$'],
        'supported_tf_versions': [1]
    },
    'Softmax_slim': {
        'input_shape': (1, 10),
        'op_type': 'Softmax',
        'constructor': "slim.softmax(constants)",
        'module_regex': ['(.+)/Softmax$'],
        'associated_op_regex': ['Softmax$'],
        'supported_tf_versions': [1]
    },
    'Softmax_slim_with_unknown_shape': {
        'input_shape': (10,),
        'op_type': 'Softmax',
        'constructor': "slim.softmax(inputs)",
        'module_regex': ['(.+)/Softmax$'],
        'associated_op_regex': ['Softmax$'],
        'supported_tf_versions': [1]
    },
    'Dropout_with_training_tensor': {
        'input_shape': (1, 10, 10, 3),
        'op_type': 'Dropout',
        'constructor': "tf.keras.layers.Dropout(rate=.4)(inputs)",
        'module_regex': ['(.+)/cond/dropout/mul_1$'],
        'associated_op_regex': ['cond/dropout/mul_1$'],
        'supported_tf_versions': [1, 2]
    },
    'Dropout_training_True': {
        'input_shape': (1, 10, 10, 3),
        'op_type': 'Dropout',
        'constructor': "tf.keras.layers.Dropout(rate=.4)(inputs, training=True)",
        'module_regex': ['(.+)/.+/mul_1$'],
        'associated_op_regex': ['/.+/mul_1$'],
        'supported_tf_versions': [1, 2]
    },
    'Dropout_with_training_tensor_unknown_shape': {
        'input_shape': (1, 10, 10, 3),
        'op_type': 'Dropout',
        'constructor': "tf.keras.layers.Dropout(rate=.4)(constants)",
        'module_regex': ['(.+)/cond/dropout/mul_1$'],
        'associated_op_regex': ['cond/dropout/mul_1$'],
        'supported_tf_versions': [1, 2]
    },
    'Dropout_training_True_unknown_shape': {
        'input_shape': (1, 10, 10, 3),
        'op_type': 'Dropout',
        'constructor': "tf.keras.layers.Dropout(rate=.4)(constants, training=True)",
        'module_regex': ['(.+)/.+/mul_1$'],
        'associated_op_regex': ['/.+/mul_1$'],
        'supported_tf_versions': [1, 2]
    },
    'Flatten': {
        'input_shape': (10, 10, 3,),
        'op_type': 'Flatten',
        'constructor': "tf.keras.layers.Flatten()(inputs)",
        'module_regex': ['(.+/Reshape)$'],
        'associated_op_regex': ['Reshape$'],
        'supported_tf_versions': [1, 2]
    },
    'Reshape_to_3D': {
        'input_shape': (300,),
        'op_type': 'Reshape',
        'constructor': "tf.keras.layers.Reshape(target_shape=[10, 10, 3])(inputs)",
        'module_regex': ['(.+)/Reshape$'],
        'associated_op_regex': ['Reshape$'],
        'supported_tf_versions': [1, 2]
    },
    'Upsample2D': {
        'input_shape': (10, 10, 3,),
        'op_type': 'Upsample2D',
        'constructor': "tf.keras.layers.UpSampling2D(size=(2, 3))(inputs)",
        'module_regex': ['(.+)/Shape$'],
        'associated_op_regex': ['Shape$'],
        'supported_tf_versions': [1, 2]
    },
    'GlobalMaxPool2D': {
        'input_shape': (10, 10, 3,),
        'op_type': 'GlobalMaxPool2D',
        'constructor': "tf.keras.layers.GlobalMaxPool2D()(inputs)",
        'module_regex': ['(.+)/Max$'],
        'associated_op_regex': ['Max$'],
        'supported_tf_versions': [1, 2]
    },
    'SimpleRNN': {
        'input_shape': (3, 100),
        'op_type': 'SimpleRNN',
        'constructor': "tf.keras.layers.SimpleRNN(10)(inputs)",
        'module_regex': ['(.+)/while/MatMul$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1]
    },
    'SimpleRNNWithRelu': {
        'input_shape': (3, 100),
        'op_type': 'SimpleRNN',
        'constructor': "tf.keras.layers.SimpleRNN(10, activation='relu')(inputs)",
        'module_regex': ['(.+)/while/MatMul$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1]
    },
    'SimpleRNNWithSequencesReturned': {
        'input_shape': (3, 100),
        'op_type': 'SimpleRNN',
        'constructor': "tf.keras.layers.SimpleRNN(10, return_sequences=True)(inputs)",
        'module_regex': ['(.+)/while/MatMul$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1]
    },
    'SimpleRNNWithSequencesReturnedRelu': {
        'input_shape': (3, 100),
        'op_type': 'SimpleRNN',
        'constructor': "tf.keras.layers.SimpleRNN(10, activation='relu', return_sequences=True)(inputs)",
        'module_regex': ['(.+)/while/MatMul$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1]
    },
    'LSTM': {
        'input_shape': (3, 100),
        'op_type': 'LSTM',
        'constructor': "tf.keras.layers.LSTM(10)(inputs)",
        'module_regex': ['(.+)/while/MatMul$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1]
    },
    'LSTM_TimeMajor_True': {
        'input_shape': (3, 100),
        'op_type': 'LSTM',
        'constructor': "tf.keras.layers.LSTM(10, time_major=True)(inputs)",
        'module_regex': ['(.+)/while/MatMul$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1]
    },
    'LSTM_Sigmoid': {
        'input_shape': (3, 100),
        'op_type': 'LSTM',
        'constructor': "tf.keras.layers.LSTM(10, recurrent_activation='sigmoid')(inputs)",
        'module_regex': ['(.+)/while/MatMul$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1]
    },
    'LSTM_Stacked_TimeMajor_True': {
        'input_shape': (3, 100),
        'op_type': 'LSTM',
        'constructor': "tf.keras.layers.LSTM(10, time_major=True, "
                       "return_sequences=True)(inputs)",
        'module_regex': ['(.+)/while/MatMul$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1]
    },
    'LSTM_Stacked_TimeMajor_True_Sigmoid': {
        'input_shape': (3, 100),
        'op_type': 'LSTM',
        'constructor': "tf.keras.layers.LSTM(10, recurrent_activation='sigmoid',"
                       "time_major=True, "
                       "return_sequences=True)(inputs)",
        'module_regex': ['(.+)/while/MatMul$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1]
    },
    'LSTM_Stacked': {
        'input_shape': (3, 100),
        'op_type': 'LSTM',
        'constructor': "tf.keras.layers.LSTM(10, return_sequences=True)(inputs)",
        'module_regex': ['(.+)/while/MatMul$'],
        'associated_op_regex': ['MatMul$'],
        'supported_tf_versions': [1]
    },
    'PReLU': {
        'input_shape': (1, 10),
        'op_type': 'PReLU',
        'constructor': "tf.keras.layers.PReLU()(inputs)",
        'module_regex': ['(.+/Relu)$'],
        'associated_op_regex': ['Relu$'],
        'supported_tf_versions': [1, 2]
    },
    'InstanceNormalization': {
        'input_shape': (10, 10, 3,),
        'op_type': 'InstanceNormalization',
        'constructor': "tf.contrib.layers.instance_norm(inputs)",
        'module_regex': ['(.+)/instancenorm/add_1$'],
        'associated_op_regex': ['/instancenorm/add_1$'],
        'supported_tf_versions': [1]
    },
    'Conv2DTranspose': {
        'input_shape': (7, 7, 1),
        'op_type': 'Conv2DTranspose',
        'constructor': "tf.keras.layers.Conv2DTranspose(1, (4, 4), use_bias=False)(inputs)",
        'module_regex': ['(.+/conv2d_transpose)$'],
        'associated_op_regex': ['/conv2d_transpose$'],
        'supported_tf_versions': [1, 2]
    },
    'Conv2DTranspose_with_bias': {
        'input_shape': (7, 7, 1),
        'op_type': 'Conv2DTranspose',
        'constructor': "tf.keras.layers.Conv2DTranspose(1, (4, 4), use_bias=True)(inputs)",
        'module_regex': ['(.+/conv2d_transpose)$'],
        'associated_op_regex': ['/conv2d_transpose$'],
        'supported_tf_versions': [1, 2]
    },
    'LayerNorm_non_fused': {
        'input_shape': (10, 10, 3,),
        'op_type': 'LayerNorm',
        # There is no explicit option to create non fused LayerNorm
        # But setting epsilon to a value smaller than 1e-5 creates non fused version
        'constructor': "tf.keras.layers.LayerNormalization(epsilon=1e-12)(inputs)",
        'module_regex': ['(.+)/batchnorm/mul_1$'],
        'associated_op_regex': ['batchnorm/mul_1$'],
        'supported_tf_versions': [2]
    },
    'GeLU': {
        'input_shape': (1, 10),
        'op_type': 'GeLU',
        'constructor': "tf.keras.activations.gelu(inputs, approximate=False)",
        'module_regex': ['(.+/mul_1)$'],
        'associated_op_regex': ['mul_1$'],
        'supported_tf_versions': [2]
    },
    'GeLU_approximate': {
        'input_shape': (1, 10),
        'op_type': 'GeLU',
        'constructor': "tf.keras.activations.gelu(inputs, approximate=True)",
        'module_regex': ['(.+/mul_1)$'],
        'associated_op_regex': ['mul_1$'],
        'supported_tf_versions': [2]
    },
    'HuggingFace_GeLU': {
        'input_shape': (1, 10),
        'op_type': 'GeLU',
        'constructor': "transformers.activations_tf._gelu(inputs)",
        'module_regex': ['(.+/mul_1)$'],
        'associated_op_regex': ['mul_1$'],
        'supported_tf_versions': [2]
    },
}
