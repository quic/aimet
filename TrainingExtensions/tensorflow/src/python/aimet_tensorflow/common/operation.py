# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Tf Operation class and utilities """

import tensorflow as tf
import aimet_common.connected_graph.operation
from aimet_tensorflow.common.product import Product


class OpWithMetaInfoType:
    """ Data type to hold info on connected graph op as tf.Operation with input and output tensors as tf.Tensor types"""
    def __init__(self, conn_op, in_tensor: tf.Tensor, out_tensor: tf.Tensor):
        self.op = conn_op.get_module()
        self.in_tensor = in_tensor
        self.out_tensor = out_tensor


class Op(aimet_common.connected_graph.operation.Op):
    """ Subclass Op inherited from aimet_common.connected_graph.operation.Op """

    def __init__(self, name, dotted_name, output_shape, is_anonymous, op_type, pattern_type):
        """
        Initializer for Op
        :param name: namme of the operation
        :param dotted_name: dotted name of the operation
        :param output_shape: shape of the output product of the operation
        :param is_anonymous: whether this is an anonymous operation
        :param op_type: type of the operation
        :param pattern_type: pattern type used to match the operation
        """
        super().__init__(name, dotted_name, output_shape, is_anonymous, op_type)
        self._output_op_node = None
        self._parameters = {}
        self._attributes = {}
        self._pattern_type = pattern_type

    @property
    def output_op_node(self):
        """ Get the output op node for this operation """
        return self._output_op_node

    @output_op_node.setter
    def output_op_node(self, op: tf.Operation):
        """ Set the output op node for this operation """
        self._output_op_node = op

    @property
    def pattern_type(self):
        """ Get the pattern type matched for this operation """
        return self._pattern_type

    def get_attribute(self, attribute_name: str):
        """ Get an attribute for this operation, returns None if attribute doesn't exist """
        return self._attributes.get(attribute_name, None)

    def add_attribute(self, attribute_name: str, attribute):
        """ Add an attribute for this operation """
        self._attributes[attribute_name] = attribute

    def add_param(self, param: str, product: Product):
        """ Add a parameter product to parameters dictionary """
        self._parameters[param] = product

    def get_param_product(self, param: str):
        """ Get the product corresponding to a particular parameter.  If no such product exists, return None """
        return self._parameters.get(param)

    def get_tf_op_with_io_tensor(self) -> OpWithMetaInfoType:
        """
        For a given connected graph op, this returns info as OpWithMetaInfoType
        (returns tf.Operation type along with input and output tensor as tf.Tensor)
        :return: OpWithMetaInfoType type
        """

        # get input tensor
        in_product = self.get_input_products()[0]
        in_tensor = in_product.tensor_dict[self]

        # handles single output at the moment (check if there is a branch op]
        # conn graph explicitly inserts a branch op when op output is fed to two tf outputs ops (>1 consumer).
        # get product/link/tensor
        output = self.output

        assert len(output.consumers) == 1
        # product goes to this consumer
        # get output tensor
        out_tensor = output.tensor_dict[output.consumers[0]]

        return OpWithMetaInfoType(self, in_tensor, out_tensor)

    def get_input_product_index_of_parent(self, parent_op) -> int:
        """
        Get the index of the input product that connects parent_op to this op.
        :param parent_op: Parent op
        :return: input product index, or None if not found.
        """
        input_product_index = None
        op_input_products = self.get_input_products()
        for index, product in enumerate(op_input_products):
            if product.producer == parent_op:
                input_product_index = index
                break
        return input_product_index
