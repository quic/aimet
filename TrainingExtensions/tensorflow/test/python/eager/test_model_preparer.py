# /usr/bin/env python3.5
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

""" Unit tests for keras model preparer """
import logging
import os
import pytest
import numpy as np
import tensorflow as tf
from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.connectedgraph import ConnectedGraph
from aimet_tensorflow.keras.model_preparer import prepare_model, _get_original_models_weights_in_functional_model_order
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops, get_all_output_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
AimetLogger.set_level_for_all_areas(logging.DEBUG)


def conv_functional():
    input_shape = (128, 28, 28, 1)
    inp = tf.keras.Input(shape=input_shape[1:])
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inp)
    x = tf.keras.layers.Conv2DTranspose(
        32, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=1, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5, trainable=False)(x)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs=x, name='conv_functional')
    return model


class TwoConvs(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TwoConvs, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(32,
                                           kernel_size=(3, 3),
                                           activation='relu',
                                           name='class_conv')

        self.conv_transpose = tf.keras.layers.Conv2DTranspose(64,
                                                              kernel_size=(3, 3),
                                                              activation='relu',
                                                              name='class_conv_transpose')

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.conv_transpose(x)
        return x


class ConvTimesThree(tf.keras.layers.Layer):
    def __init__(self, **kwargs):

        super(ConvTimesThree, self).__init__(**kwargs)
        self.depth_conv = tf.keras.layers.DepthwiseConv2D(depth_multiplier=1,
                                                          kernel_size=(3, 3),
                                                          activation='relu',
                                                          name='class_conv_depth')
        self.two_convs = TwoConvs()

    def call(self, x, **kwargs):
        return self.depth_conv(self.two_convs(x))


def conv_sub_class():
    input_shape = (128, 28, 28, 1)
    inp = tf.keras.Input(batch_shape=input_shape)
    x = ConvTimesThree()(inp)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x, training=False)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs=x, name='conv_classes')
    return model

###########################################################################################################
#                                                                                                         #
# Below models are based on Deep Learning with Python by Francois Chollet Second Edition (page 182 - 185) #
#                                                                                                         #
###########################################################################################################

# Only Subclassing
class CustomerTicketModel(tf.keras.Model):

    def __init__(self, num_departments):
        super().__init__()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.mixing_layer = tf.keras.layers.Dense(64, activation="relu")
        self.priority_scorer = tf.keras.layers.Dense(1, activation="sigmoid")
        self.department_classifier = tf.keras.layers.Dense(num_departments, activation="softmax")

    def call(self, inputs, **kwargs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department

# Functional model that includes subclassed layers
class Classifier(tf.keras.Model):

    def __init__(self, num_classes=4):
        super().__init__()
        if num_classes == 2:
            num_units = 1
            activation = "sigmoid"
        else:
            num_units = num_classes
            activation = "softmax"
        self.dense = tf.keras.layers.Dense(num_units, activation=activation)

    def call(self, inputs, **kwargs):
        return self.dense(inputs)


def functional_model_with_subclassed_layers():
    inputs = tf.keras.layers.Input(shape=(3,))
    features = tf.keras.layers.Dense(64, activation="relu")(inputs)
    outputs = Classifier(num_classes=10)(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Subclass model that includes functional layers
def subclass_model_with_functional_layers():
    inputs = tf.keras.Input(shape=(64,))
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(inputs)
    binary_classifier = tf.keras.Model(inputs=inputs, outputs=outputs)

    class MyFunctionalModel(tf.keras.Model):
        def __init__(self):
            super().__init__(name='my_functional_model')
            self.dense = tf.keras.layers.Dense(64, activation="relu")
            self.classifier = binary_classifier

        def call(self, inputs, **kwargs):
            features = self.dense(inputs)
            return self.classifier(features)

    model = MyFunctionalModel()
    return model

# Layer with multiple math operations in call
class MultiMathOperations(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MultiMathOperations, self).__init__(**kwargs)
        self.r = tf.constant(np.random.rand(1, 1), dtype=tf.float32)
        self.g = tf.constant(np.random.rand(1, 1), dtype=tf.float32)
        self.b = tf.constant(np.random.rand(1, 1), dtype=tf.float32)

    def call(self, inputs, **kwargs):
        r = tf.math.multiply(inputs[:, :, :, 0:1], self.r)
        g = tf.math.multiply(inputs[:, :, :, 1:2], self.g)
        b = tf.math.multiply(inputs[:, :, :, 2:3], self.b)
        return tf.concat([r, g, b], axis=3)


    #####################################################
    #                                                   #
    #    QA models for testing the conversion script    #
    #                                                   #
    #####################################################

# Text classification Transformer model from https://keras.io/examples/nlp/text_classification_with_transformer/
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training, **kwargs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x, **kwargs):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        x = x + positions
        return x



def get_keras_text_classification_example_model_and_data_input():
    # Download and prepare dataset
    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review

    random_input = np.random.random((10, 200))

    # Create classifier model using transformer layer
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = tf.keras.layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(20, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    _ = model(random_input)
    # Train and Evaluate
    return model, random_input


    ################################
    #                              #
    #  Start of testing functions  #
    #                              #
    ################################

def compare_weights(original_weights, functional_weights):
    """
    Helper function to compare the weights of two models. This function is used to test the conversion script.
    :param original_weights: the original model's weights 
    :param functional_weights: the model's weights that was converted from the original model
    """

    for i, _ in enumerate(original_weights):
        np.testing.assert_array_equal(original_weights[i], functional_weights[i])


def verify_functional_model(functional_model, original_model,
                            random_input, number_of_layers_in_model):
    """
    Helper function to verify that the functional model is ready for AIMET. 
    This function is used to test the conversion script.
    :param functional_model: the functional model that was converted from the original model
    :param original_model: the original model
    :param random_input: a random input to the model
    """
    # Verify the functional model produces the same output as the original model
    np.testing.assert_array_equal(functional_model(random_input).numpy(), original_model(random_input).numpy())
    assert len(functional_model.layers) == number_of_layers_in_model, \
        f"Expected {number_of_layers_in_model} layers in functional model, but got {len(functional_model.layers)}"


@pytest.mark.skip("Need to verify how functional model should look")
def test_full_subclass_to_functional():
    vocabulary_size = 10000
    num_tags = 100
    num_departments = 4
    num_samples = 1280

    title_data = np.random.randint(0, 2, size=(1, num_samples, vocabulary_size))
    text_body_data = np.random.randint(0, 2, size=(1, num_samples, vocabulary_size))
    tags_data = np.random.randint(0, 2, size=(1, num_samples, num_tags))

    model = CustomerTicketModel(num_departments=num_departments)
    _ = model({"title": title_data,
               "text_body": text_body_data,
               "tags": tags_data})
    # Since this model is fully subclassed, specifically at the beginning, we call prepare model with
    # the inputs to have Keras symoblic tensor fit the rest of the layers correctly.
    input_layers = [tf.keras.Input(shape=(num_samples, vocabulary_size,), name="title"),
                    tf.keras.Input(shape=(num_samples, vocabulary_size,), name="text_body"),
                    tf.keras.Input(shape=(num_samples, num_tags,), name="tags")]
    functional_model = prepare_model(model, input_layers)
    assert functional_model.count_params() == model.count_params()


def test_functional_model_with_subclassed_layers_to_functional():
    original_model = functional_model_with_subclassed_layers()
    random_input = np.random.rand(32, 3)
    _ = original_model(random_input)

    functional_model = prepare_model(original_model)
    assert functional_model.count_params() == original_model.count_params()

    compare_weights(original_model.get_weights(), functional_model.get_weights())
    np.testing.assert_array_equal(functional_model(random_input).numpy(), original_model(random_input).numpy())
    verify_functional_model(functional_model,
                            original_model,
                            random_input,
                            number_of_layers_in_model=3)

    connected_graph = ConnectedGraph(functional_model)
    assert "Unknown" not in connected_graph.ordered_ops

    product_dict = connected_graph.get_all_products()
    assert "Gemm_0_to_Gemm_1" in product_dict

    input_ops = get_all_input_ops(connected_graph)
    assert len(input_ops) == 1
    assert input_ops[0].get_module() == functional_model.layers[1]
    assert isinstance(input_ops[0].get_module(), type(original_model.layers[1]))

    output_ops = get_all_output_ops(connected_graph)
    assert len(output_ops) == 1
    assert output_ops[0].get_module() == functional_model.layers[-1]
    assert isinstance(output_ops[0].get_module(), type(original_model.layers[-1].dense))


def test_input_layer_missing():
    model = subclass_model_with_functional_layers()
    input_shape = (32, 64)
    random_input = np.random.rand(*input_shape)
    _ = model(random_input)
    with pytest.raises(ValueError):
        prepare_model(model)


def test_subclass_model_with_subclassed_layers_to_functional():
    original_model = subclass_model_with_functional_layers()
    input_shape = (32, 64)
    random_input = np.random.rand(*input_shape)
    _ = original_model(random_input)

    functional_model = prepare_model(original_model, tf.keras.Input(shape=input_shape[1:]))
    assert functional_model.count_params() == original_model.count_params()
    compare_weights(original_model.get_weights(), functional_model.get_weights())
    verify_functional_model(functional_model,
                            original_model,
                            random_input,
                            number_of_layers_in_model=3)

    connected_graph = ConnectedGraph(functional_model)
    assert "Unknown" not in connected_graph.ordered_ops

    product_dict = connected_graph.get_all_products()
    assert "Gemm_0_to_Gemm_1" in product_dict

    input_ops = get_all_input_ops(connected_graph)
    assert len(input_ops) == 1
    assert input_ops[0].get_module() == functional_model.layers[1]
    assert isinstance(input_ops[0].get_module(), type(original_model.layers[0]))

    output_ops = get_all_output_ops(connected_graph)
    assert len(output_ops) == 1
    assert output_ops[0].get_module() == functional_model.layers[-1]
    assert isinstance(output_ops[0].get_module(), type(original_model.layers[-1].layers[-1]))


def test_conv_times_three_subclass_to_functional():
    original_model = conv_sub_class()
    input_shape = (32, 28, 28, 1)
    random_input = np.random.rand(*input_shape)
    _ = original_model(random_input)

    functional_model = prepare_model(original_model)
    assert functional_model.count_params() == original_model.count_params()

    # NOTE: Since ConvTimesThree has the internal layers out of order compared to the call method,
    # the weights are not in the order of what the actual architecture is (this is a Keras design).
    # Therefore, we get the original model's weights and sort them in the order of the actual
    # architecture and use those weights to compare to the functional model's weights.
    model_weights_in_correct_order = _get_original_models_weights_in_functional_model_order(
        original_model, functional_model, class_names=["conv_times_three", "two_convs"])

    compare_weights(model_weights_in_correct_order, functional_model.get_weights())
    verify_functional_model(functional_model,
                            original_model,
                            random_input,
                            number_of_layers_in_model=7)

    connected_graph = ConnectedGraph(functional_model)
    assert "Unknown" not in connected_graph.ordered_ops

    product_dict = connected_graph.get_all_products()
    assert "Conv_0_to_ConvTranspose_1" in product_dict
    assert "ConvTranspose_1_to_Conv_2" in product_dict

    input_ops = get_all_input_ops(connected_graph)
    assert len(input_ops) == 1
    assert input_ops[0].get_module() == functional_model.layers[1]
    assert isinstance(input_ops[0].get_module(), type(original_model.layers[1].two_convs.conv))

    output_ops = get_all_output_ops(connected_graph)
    assert len(output_ops) == 1
    assert output_ops[0].get_module() == functional_model.layers[-1]
    assert isinstance(output_ops[0].get_module(), type(original_model.layers[-1]))



def test_multi_math_operations_subclass_to_functional():
    input_layer = tf.keras.Input(shape=(32, 32, 1))
    output = MultiMathOperations()(input_layer)
    original_model = tf.keras.Model(inputs=input_layer, outputs=output)

    functional_model = prepare_model(original_model)
    assert functional_model.count_params() == original_model.count_params()
    compare_weights(original_model.get_weights(), functional_model.get_weights())

    # NOTE: No testing of the ConnectedGraph since Lambda layers are not supported by the ConnectedGraph at this time.


def test_keras_text_classification_example_model_to_functional():
    original_model, random_input = get_keras_text_classification_example_model_and_data_input()
    functional_model = prepare_model(original_model)
    assert functional_model.count_params() == original_model.count_params()

    # NOTE: Since TextClassification Model has the internal layers out of order compared to the call method,
    # the weights are not in the order of what the actual architecture is (this is a Keras design).
    # Therefore, we get the original model's weights and sort them in the order of the actual
    # architecture and use those weights to compare to the functional model's weights.
    model_weights_in_correct_order = _get_original_models_weights_in_functional_model_order(
        original_model, functional_model, class_names=["token_and_position_embedding", "transformer_block"])
    compare_weights(model_weights_in_correct_order, functional_model.get_weights())
    verify_functional_model(functional_model,
                            original_model,
                            random_input,
                            number_of_layers_in_model=20)

    # NOTE: No testing of the ConnectedGraph since Embedding, MultiHeadAttention, and Lambda layers are not
    # supported by the ConnectedGraph at this time.
