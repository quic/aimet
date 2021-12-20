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
"""
Connected graph class and utilities
"""
import typing

import tensorflow as tf

from aimet_common.connected_graph.connectedgraph import (
    ConnectedGraph as AimetCommonConnectedGraph,
)
from aimet_common.connected_graph.operation import Op
from aimet_common.connected_graph.product import Product
from aimet_common.model_module import KerasModelModule


map_keras_types_to_onnx = {
    tf.keras.layers.Conv1D: ["Conv"],
    tf.keras.layers.Conv2D: ["Conv"],
    tf.keras.layers.DepthwiseConv2D: ["Conv"],
    tf.keras.layers.ZeroPadding1D: ["Pad"],
    tf.keras.layers.ZeroPadding2D: ["Pad"],
    tf.keras.layers.Dropout: ["Dropout"],
    tf.keras.layers.BatchNormalization: ["BatchNormalization"],
    tf.keras.layers.ReLU: ["Relu"],
    tf.keras.layers.MaxPool2D: ["MaxPool"],
    tf.keras.layers.GlobalAveragePooling2D: ["GlobalAveragePool"],
    tf.keras.layers.Reshape: ["Reshape"],
    tf.keras.layers.Dense: ["Gemm", "MatMul"],
    tf.keras.layers.AveragePooling2D: ["AveragePool"],
    tf.keras.layers.RNN: ["RNN"],
    tf.keras.layers.LSTM: ["LSTM"],
    tf.keras.layers.GRU: ["GRU"],
    tf.keras.layers.Conv2DTranspose: ["ConvTranspose"],
    tf.keras.layers.PReLU: ["PRelu"],
    tf.keras.layers.LeakyReLU: ["LeakyRelu"],
    tf.keras.layers.ELU: ["Elu"],
    tf.keras.layers.Flatten: ["Flatten"],
    tf.keras.layers.Add: ["Add"],
    tf.keras.layers.Subtract: ["Sub"],
    tf.keras.layers.Multiply: ["Mul"],
    tf.keras.layers.Concatenate: ["Concat"],
}


def parse_activation_layer(
        activation_layer: tf.keras.layers.Activation,
) -> typing.List[str]:
    """
    Parse generic tf.keras.layers.Activation and convert it to corresponding onnx type

    :param activation_layer: tf.keras.layers.Activation
    :return: list of converted onnx type str
    """
    activation_name = tf.keras.activations.serialize(activation_layer.activation)

    keras_to_onnx_dict = {
        "softmax": "Softmax",
        "relu": "Relu",
        "selu": "Selu",
        "tanh": "Tanh",
        "sigmoid": "Sigmoid",
        "leaky_relu": "LeakyRelu",
        "softplus": "Softplus",
        "softsign": "Softsign",
        "elu": "Elu",
    }
    onnx_activation = keras_to_onnx_dict.get(activation_name)
    if onnx_activation is None:
        return ["Unknown"]

    return [onnx_activation]


class ConnectedGraph(AimetCommonConnectedGraph):
    """
    Connected Graph class
    """

    def __init__(
            self,
            model: tf.keras.Model,
            input_shapes: typing.Union[
                None, typing.Tuple, typing.List[typing.Tuple]
            ] = None,
    ):
        """
        If the model object is implemented in a subclassing manner, resulting object is different from
        the original object because this method is converting to Functional manner

        :param model: Keras Model (Sequential, Functional, Subclassing)
        :param input_shapes: Input shape tuple or list of input tuple shape
        """
        super(ConnectedGraph, self).__init__()

        self._name_to_layer = dict()
        self._layer_to_op = dict()
        self.ordered_ops = []
        self._ops_index = 0

        if model.built:
            self._model = model
        else:
            if input_shapes is None:
                raise RuntimeError(
                    "input_shapes should be provided if model was not built"
                )
            self._model = self._build_model(model, input_shapes)

        # Generate Ops by parsing layer information
        self._parse_layers(self._model.layers)

        # Generate Products by parsing layer connection information
        self._parse_layer_connections()
        self._fill_op_params()

    @staticmethod
    def _build_model(
            model: tf.keras.Model,
            input_shapes: typing.Union[typing.Tuple, typing.List[typing.Tuple]],
    ) -> tf.keras.Model:
        """
        Build tf.keras.model if it was not built. After building layer connection information is set

        :param model: Keras Model (Sequential, Subclassing)
        :param input_shapes: Input shape tuple or list of input tuple shape
        :return: Keras Model with layer connection information
        """
        if isinstance(model, tf.keras.Sequential):
            if not isinstance(input_shapes, typing.Tuple):
                raise RuntimeError(
                    "Sequential model can only receive one input, multiple input is not supported"
                )

            model.build((None,) + input_shapes)
            return model

        # Subclassing model
        if isinstance(input_shapes, typing.Tuple):
            # Received input shape tuple, it's a single input case
            inputs = tf.keras.Input(shape=input_shapes)
        else:
            # Received list of input shape tuple, it's a multiple input case
            inputs = [tf.keras.Input(shape=input_shape) for input_shape in input_shapes]

        return tf.keras.Model(inputs=inputs, outputs=model.call(inputs))

    def _parse_layers(self, layers: typing.List[tf.keras.layers.Layer]):
        """
        Parse layers iteratively to obtain Ops

        :param layers: list of Keras layers
        """
        for layer in layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            self._parse_layer(layer)

    def _parse_layer(self, layer: tf.keras.layers.Layer):
        """
        Parse a layer and convert it to Op with recursive manner if it is nested

        :param layer: Keras layer
        """
        if hasattr(layer, "layers"):
            self._parse_layers(layer.layers)
        else:
            if isinstance(layer, tf.keras.layers.Activation):
                op_type = parse_activation_layer(layer)
            else:
                op_type = map_keras_types_to_onnx.get(type(layer))

            if op_type is None:
                op_type = "Unknown"
            else:
                op_type = op_type[0]

            op = self._generate_op(op_type, layer)

            self._name_to_layer[layer.output.name] = layer
            self._layer_to_op[layer] = op
            self._ops[op.name] = op
            self.ordered_ops.append(op)
            self._ops_index += 1

    def _generate_op(self, op_type: str, layer: tf.keras.layers.Layer) -> Op:
        """
        Generate operation object using operation type and keras layer

        :param op_type: Operation type compatible with ONNX
        :param layer: Keras layer
        :return: Operation object
        """
        op_name = f"{op_type}_{self._ops_index}"
        dotted_name = f"{self._model.name}.{layer.name}"
        output_shape = layer.output_shape

        op = Op(
            name=op_name,
            dotted_name=dotted_name,
            output_shape=output_shape,
            is_anonymous=False,
            op_type=op_type,
        )
        op.model_module = KerasModelModule(layer)

        if op.type == "Conv" and hasattr(layer, "groups"):
            op.groups = layer.groups

        return op

    def _parse_layer_connections(self):
        """
        Obtain layer connection by parsing layer information and generate product
        """
        for op in self.ordered_ops:
            self._generate_product(op.get_module())

    def _generate_product(self, target_layer: tf.keras.layers.Layer):
        """
        Generate products while traversing the inbound layer of the target layer

        :param target_layer: tf.keras.layer related to consumer Op
        """
        for _, node in enumerate(target_layer.inbound_nodes):
            for inbound_layer, _, _, _ in node.iterate_inbound():
                if isinstance(inbound_layer, tf.keras.layers.InputLayer):
                    self._generate_input_product(target_layer)
                else:
                    self._generate_usual_product(target_layer, inbound_layer)

    def _generate_input_product(self, target_layer):
        """
        Generate product which is producer is model input

        :param target_layer: tf.keras.layer related to consumer Op
        """
        consumer_op = self._layer_to_op.get(target_layer)
        product_name = f"input_to_{consumer_op.name}"

        if product_name not in self._products:
            product = Product(product_name, None)
            product.is_model_input = True
            product.add_consumer(consumer_op)
            consumer_op.add_input(product)
            self._products[product_name] = product

    def _generate_usual_product(self, target_layer, inbound_layer):
        """
        Generate usual product which is producer is not model input

        :param target_layer: tf.keras.layer related to consumer Op
        :param inbound_layer: tf.keras.layer related to producer Op
        """
        consumer_op = self._layer_to_op.get(target_layer)
        producer_op = self.get_op_from_module_name(inbound_layer.output.name)

        if producer_op is None:
            raise RuntimeError("Producer Op must exist")

        product_name = f"{producer_op.name}_to_{consumer_op.name}"
        if product_name not in self._products:
            product = Product(product_name, producer_op.output_shape)
            product.producer = producer_op
            self._products[product_name] = product
        else:
            product = self._products[product_name]

        product.add_consumer(consumer_op)
        consumer_op.add_input(product)
        producer_op.output = product

    def _fill_op_params(self):
        """
        For certain ops like convolution, batch norm, and linear,
        create products for their parameters if they don't exist yet.
        """

        for op in self._ops.values():
            layer = op.get_module()
            layer_name = layer.name

            if not layer.built:
                raise RuntimeError("Layer should be built before executing this method")

            if op.type in ["Conv", "ConvTranspose", "Gemm"]:
                weight_tensors = layer.get_weights()

                self._create_and_add_param_product_if_not_exists(
                    op, f"{layer_name}.weight", list(weight_tensors[0].shape)
                )
                if layer.use_bias:
                    self._create_and_add_param_product_if_not_exists(
                        op, f"{layer_name}.bias", list(weight_tensors[1].shape)
                    )

            if op.type == "BatchNormalization":
                tensor_descriptions = ["weight", "bias", "running_mean", "running_var"]
                weight_tensors = layer.get_weights()

                for weight_tensor, tensor_description in zip(weight_tensors, tensor_descriptions):
                    self._create_and_add_param_product_if_not_exists(
                        op,
                        f"{layer_name}.{tensor_description}",
                        list(weight_tensor.shape),
                    )

    def _create_and_add_param_product_if_not_exists(
            self, op: Op, product_name: str, shape: typing.List[int]
    ):
        """
        Given a name of a product, create it if it doesn't exist, and attach it to the specified op as a parameter.
        :param op: Op to connect the parameter product to.
        :param product_name: Name of the product to create.
        :param shape: Shape of the product to create.
        """
        if product_name not in self._products.keys():
            product = Product(product_name, shape)
            product.is_parm = True
            product.add_consumer(op)
            op.add_input(product)
            self._products[product_name] = product

    def get_op_from_module_name(self, name: str) -> typing.Union[Op, None]:
        """
        Given the name of a operation/module, return the corresponding op in ops dict
        :param name: tf.keras.layer name
        :return: Connected graph operation corresponding to tf.keras.layer name. Returns None if not found
        """

        layer = self._name_to_layer.get(name, None)
        if layer:
            return self._layer_to_op.get(layer, None)
        return None

    def get_all_ops(self) -> typing.Dict[str, Op]:
        """
        Returns the ops dictionary
        :return: Ops dictionary
        """
        return self._ops

    def get_all_products(self) -> typing.Dict[str, Product]:
        """
        Returns the products dictionary
        :return: Product dictionary
        """
        return self._products
