#!/usr/bin/env python3.5

#  =============================================================================
#
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
#
#  =============================================================================

#  =============================================================================
#
#  @@-COPYRIGHT-START-@@
#  From PyTorch:
#
#  Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
#  Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
#  Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
#  Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
#  Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
#  Copyright (c) 2011-2013 NYU                      (Clement Farabet)
#  Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
#  Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
#  Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
#  From Caffe2:
#
#  Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
#  All contributions by Facebook:
#  Copyright (c) 2016 Facebook Inc.
#
#  All contributions by Google:
#  Copyright (c) 2015 Google Inc.
#  All rights reserved.
#
#  All contributions by Yangqing Jia:
#  Copyright (c) 2015 Yangqing Jia
#  All rights reserved.
#
#  All contributions from Caffe:
#  Copyright(c) 2013, 2014, 2015, the respective contributors
#  All rights reserved.
#
#  All other contributions:
#  Copyright(c) 2015, 2016 the respective contributors
#  All rights reserved.
#
#  Caffe2 uses a copyright model similar to Caffe: each contributor holds
#  copyright over their contributions to Caffe2. The project versioning records
#  all such contribution and copyright details. If a contributor wants to further
#  mark their specific copyright on a particular contribution, they should
#  indicate their copyright solely in the commit message of the change when it is
#  committed.
#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright
#  notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#
#  3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#  and IDIAP Research Institute nor the names of its contributors may be
#  used to endorse or promote products derived from this software without
#  specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#  © 2019 GitHub, Inc.
#
#  @@-COPYRIGHT-END-@@
#
#  =============================================================================

"""For constructing a uniform representation of the computational graph for a PyTorch model,
that is easy to navigate and stores information for the purpose of winnowing.
The representation graph consists of nodes that are either 'operation' or 'product';
operations represent a module or a function that generates a tensor, while products represent
the tensors that are either input to the model (input, constant or parameter) or the
result of an operation. Furthermore the graph representation is bi-directional."""


import tempfile
from typing import Tuple, Union
from distutils.version import LooseVersion

import torch
import torch.utils.tensorboard._pytorch_graph

from aimet_common.utils import AimetLogger, ModelApi
from aimet_common.model_module import PytorchModelModule
from aimet_common.connected_graph.connectedgraph import ConnectedGraph as AimetCommonConnectedGraph
from aimet_common.connected_graph.product import Product
from aimet_common.connected_graph.operation import Op, determine_preceding_op_input_product_index_in_multi_input_op

# pylint: disable=no-member
OperatorExportTypes = torch._C._onnx.OperatorExportTypes # pylint: disable=protected-access
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Winnow)


class ConnectedGraph(AimetCommonConnectedGraph):
    """For construction of a graph that connects operations together (
    either module or functional) as producers and consumers of tensors.
    Note that the graph has two kinds of nodes: operations and products."""

    def __init__(self, model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple[torch.Tensor]],
                 mask_propagation: bool = False):
        super().__init__()
        self._model_name = type(model).__name__
        self._mask_propagation = mask_propagation
        self._generate_module_lookup_table(model)
        self._split_count = 0  # Use it in the name of split Ops getting added to the connected graph.

        if torch_version_okay():
            self._construct_graph(model, dummy_input)

    def __del__(self):
        """
        Destructor of ConnectedGraph class
        break the dependencies of Ops with Product
        """
        for product in self._products.values():
            product.producer = None
            product.set_consumers_to_null()

    def get_leaf_operations(self):
        """Operations with output not used by other operation(s)"""
        return [op for op in self._ops.values() if not op.output]

    def get_op_from_module_name(self, name):
        """ Returns the Op associated with the module name """
        return next((op for op in self._ops.values() if op.dotted_name == name), None)

    @property
    def num_operations(self):
        """Total number of operations: named and anonymous modules, and functions"""
        return len(self._ops)

    @property
    def num_products(self):
        """Total number of products: inputs, constants, parameters and
        products transfered between operations."""
        return len(self._products)

    def get_product(self, name):
        """ Returns the product with the name passed in the argument """
        return self._products[name]

    def _construct_graph(self, model, x_input):
        list_of_xnodes = trace_and_parse(model, x_input)
        list_of_module_names = []

        for name, module in model.named_modules(prefix=self._model_name):
            if 'torch.nn.module' in str(type(module)):
                list_of_module_names.append(name)
        for xnode in list_of_xnodes:
            self._parse_xnode(xnode, list_of_module_names)
        for xnode in list_of_xnodes:
            self._connect_xnode(xnode)

        # tensorboardX doesn't detect splits (due to residual connections) in the model.
        # For each split in the model, insert a corresponding split Op in the connected graph.
        ops_list = [op for op in self._ops.values()]
        for op in ops_list:
            self._determine_split_behavior_for_op_and_insert_split_op_in_connected_graph(op)

        # Now that the graph is constructed, for each Op, create the Mask.
        for op in self._ops.values():
            if not op.is_anonymous:
                module = self._modules_by_name[op.dotted_name]
                _fill_op_info(op, module)

            if self._mask_propagation:
                op.create_mask_info(ModelApi.pytorch)

        self._report_on_graph()

    def _report_on_graph(self):
        logger.debug("%d operations registered", len(self._ops))
        logger.debug("%d products registered", len(self._products))

        producers = set()
        consumers = set()
        num_products_wo_producer = 0
        for product in self._products.values():
            if product.producer:
                producers.add(product.producer.name)
            else:
                num_products_wo_producer += 1
            for consumer in product.consumers:
                consumers.add(consumer.name)
        logger.debug("%d unique producers referenced", len(producers))
        logger.debug("%d unique consumers referenced", len(consumers))
        logger.debug("%d products without producer", num_products_wo_producer)

        inputs = set()
        outputs = set()
        num_leaf_ops = 0
        for op in self._ops.values():
            if op.output:
                outputs.add(op.output.name)
            else:
                num_leaf_ops += 1
            for inp in op.inputs:
                inputs.add(inp.name)
        logger.debug("%d unique inputs referenced", len(inputs))
        logger.debug("%d unique outputs referenced", len(outputs))
        logger.debug("%d operations without output", num_leaf_ops)


    def _add_product(self, name, shape):
        assert name not in self._products
        product = Product(name, shape)
        self._products[name] = product
        return product

    def _parse_xnode(self, xnode: dict, list_of_module_names: list):
        """Uses information from the specified node definition to create nodes
        in our graph representation."""

        xname = xnode.name
        output_size = get_output_shape_from_xnode(xnode)

        if xname.find('input/0') == 0:
            assert not xnode.input
            self._add_product(xname, output_size).is_model_input = True

        if xname.find('input.') != -1:
            assert not xnode.input
            self._add_product(xname, output_size).is_model_input = True

        elif xnode.op == 'IO Node':
            if xname.find('input/') == 0:
                self._add_product(xname, output_size).is_model_input = True

        elif xnode.op == 'Parameter':
            self._add_product(xname, output_size).is_parm = True

        elif xnode.op in 'onnx::Constant':
            # # Devel code for a case (which?) where this assert didn't hold:
            # attr_str = xnode['attr'].replace('\n', ' ')
            # # e.g. '{ value : tensor([ 0.0108, -0.1134, -0.1086, ... , 0.1751,  0.0436])}'
            # # Note 1: After substring 'tensor(' may also first occur e.g.: '1.00000e-02 *'
            # # Note 2: The string containing the values may contain '...' (Ellipsis)
            # tensor_start_pos = attr_str.find('tensor(') + len('tensor(')
            # tensor_end_pos = attr_str.rfind(')')
            # tensor_str = attr_str[tensor_start_pos:tensor_end_pos]
            # shape = torch.Tensor(eval(tensor_str)).shape
            if output_size:
                self._add_product(xname, output_size).is_const = True

        elif xnode.op in ('onnx::Shape', 'onnx::Unsqueeze', 'onnx::Transpose'):
            pass

        else:
            dotted_name, op_type, is_anonymous = parse_xname(xname, xop=xnode.op,
                                                             list_of_module_names=list_of_module_names)

            # For MNIST, the Droput2d (conv2_drop) and the Conv2d (conv2) are defined in the model definition but
            # a separate NodeDef is NOT returned for the Conv2d module. PyTorch/ONNX , fuses the Dropout2d  and Conv2d
            # Ops in to a single Op. This creates an inconsistency between the model definition and the constructed
            # graph. Since Dropout2d doesn't play a role in Winnowing, replace it with the Conv2d.
            # Using the Conv2d's dotted_name helpsin obtaining the Conv2d module later, which is needed for
            # reducing it.
            if op_type == 'Dropout2d' and xnode.op == 'onnx::Conv':
                op_type, dotted_name = \
                    _temporary_fix_for_handling_dropouts_in_mnist_fix_me(dotted_name, list_of_module_names)

            op = Op(xname, dotted_name, output_size, is_anonymous, op_type)
            self._ops[xname] = op

    def _connect_xnode(self, xnode: dict):
        """Uses information from the specified tensorboardX node to connect nodes
        in our graph representation."""

        xname = xnode.name

        if xname.find('input/0') == 0:
            assert not self._products[xname].producer
            # nothing else to do

        elif xnode.op == 'Parameter':
            product = self._products[xname]
            assert not product.producer
            assert len(product.consumers) <= 1
            # nothing else to do

        elif xnode.op in 'onnx::Constant':
            pass

        elif xnode.op in ('output', 'onnx::Shape', 'onnx::Unsqueeze', 'onnx::Dropout',
                          'onnx::Transpose'):
            # Ignore the final output of the model and other operrations that are
            # not required for winnowing purposes.
            pass

        else:
            # so it's an operation
            add_consumer = True
            for input_xname in xnode.input:
                if input_xname in self._ops:
                    # op-to-op: no Product created yet
                    producer = self._ops[input_xname]
                    product_shape = producer.output_shape
                    product_name = input_xname + '__to__' + xname
                    product = Product(product_name, product_shape)
                    self._products[product_name] = product
                    producer.output = product
                    product.producer = producer
                elif input_xname not in self._products:
                    add_consumer = False
                else:
                    product = self._products[input_xname]

                if add_consumer:
                    consumer = self._ops[xname]
                    consumer.add_input(product)
                    product.add_consumer(consumer)

    def get_product_names_from_dotted_name(self, dotted_name):
        """ Returns all Products whose Producer's dotted name matches the argument dotted name.
            For Residual models, same producer will have multiple products. """

        matched_products = list()
        for product in self._products.values():
            if product.producer:
                if product.producer.dotted_name == dotted_name:
                    matched_products.append(product.name)
        return matched_products

    def _generate_module_lookup_table(self, model):
        """ Generates a look up dictionary for getting modules from their names. """
        self._modules_by_name = dict()
        for name, module in model.named_modules(prefix=self._model_name):
            self._modules_by_name[name] = module

    def _create_split_op_output_product(self, preceding_op, split_op):
        """ """
        split_op_product_name = split_op.name + '__to__' + 'multiple_ops'
        split_op_product_shape = preceding_op.output.shape
        split_op_product = self._add_product(split_op_product_name, split_op_product_shape)
        split_op_product.producer = split_op

        return split_op_product

    def _create_split_op(self, op: Op):
        """ The op's output is split in the forward function. To model it correctly,
        create a Split Op. """
        split_name_parts = ['Split_', str(self._split_count)]
        split_name = ''.join(split_name_parts)
        self._split_count += 1
        split_dotted_name_parts = [self._model_name, split_name]
        split_dotted_name = '.'.join(split_dotted_name_parts)
        is_anonymous = True
        split_op = Op(split_name, split_dotted_name, op.output_shape, is_anonymous, 'Split')
        self._ops[split_name] = split_op
        return split_op

    def _add_consumers_to_split_op_product(self, preceding_op: Op, split_op_product: Product):
        """ A Split Op's output product has multiple consumers. Add them tothe product. """

        dotted_name = preceding_op.dotted_name
        output_product_names = self.get_product_names_from_dotted_name(dotted_name)

        # Important Notes
        # ResNet model uses the same Relu twice in the forward function of ReseNet's BasicBlock.
        # The first Relu feeds in to the BasicBlock's Conv2.
        # The second Relu's output is split with one branch feeding the next BasicBlock's conv1 and the other
        # branch feeding in to the next BasicBlock's Add.
        # The following line filters out the Relu whose output is NOT split :(
        out_product_names = [name for name in output_product_names if preceding_op.name in name]

        num_products = len(out_product_names)
        consumer_index = 0
        for a_product_index in range(num_products):
            a_product = self.get_product(out_product_names[a_product_index])
            a_consumer = a_product.consumers[0]
            split_op_product.consumers.append(a_consumer)
            logger.debug("Insert Split Op: Step 2a. Consumer Op: %s, a_product_index: %s",
                         a_consumer.dotted_name, a_product_index)
            if a_consumer.type in ('Concat', 'Add'):
                # Need to insert the newly created split_op product in the correct input index of the Concat Op :)
                logger.debug("Insert Split Op: Step 2b. Op has multiple input products: %s", a_consumer.inputs)
                input_product_index = determine_preceding_op_input_product_index_in_multi_input_op(preceding_op,
                                                                                                   a_consumer)
                a_consumer.inputs[input_product_index] = split_op_product
                logger.debug("Insert Split Op: Step 2c. For product: %s, split_op input_product_index: %s",
                             split_op_product.name, input_product_index)
            else:
                # There is only one input to this consumer. Add it to the 0th index of inputs.
                logger.debug("Insert Split Op: Step 2d. Op has single input product: %s", a_consumer.inputs)
                input_product_index = 0
                a_consumer.inputs[input_product_index] = split_op_product
                logger.debug("Insert Split Op: Step 2e. For split_op product: %s, input_product_index: %s",
                             split_op_product.name, input_product_index)
            consumer_index += 1

    def _insert_split_op_in_connected_graph(self, preceding_op: Op, split_op: Op):
        """ Insert a Split Op below the preceding Op in the connected graph. """

        # Important Notes
        # Op:
        # An Op class represents a module in a model.
        #
        # Product:
        # In this version of the Winnower, the Product class represents the following entities in a model.
        # 1) a Tensor between two modules (in Winnower, 2 Ops).
        # 2) an input
        # 3) a constant
        # 4) a parameter
        #
        # Considering only the definition #1) above, i.e., Product is a Tensor between 2 Ops,
        # an Op's inputs and output are Products.
        # That means, an Op could have multiple input Products and one output Product.
        # Examples of Op with multiple input products: Add, Concat
        # A Product's Producer and Consumer are Ops.
        # A Product could have only one Producer but could have multiple consumers.
        # For example, a Split Op has one output.  The Split Op's single output isa Product.
        # That product's single Producer is the Split Op and multiple consumers are the 2 Ops in the 2 branches of
        # the Split, that receive the Split output.

        # Steps:
        # 1. Create a new Product for Split Op's output.
        # 2.This product has multiple consumers. Add the consumers to the Product.
        #   Get the consumers from the op's multiple products.
        # 3. Set the the current Op's output Product's consumer to Split Op. The output product's name must be changed.
        # 4. Set the Split Op's input to point to current Op's output. Its name must be changed.

        # 1. Create a new Product for Split Op's output.
        split_op_product = self._create_split_op_output_product(preceding_op, split_op)
        split_op.output = split_op_product

        # 2.This product has multiple consumers. Add the consumers to the Product.
        # Get the consumers from the op's multiple products.

        self._add_consumers_to_split_op_product(preceding_op, split_op_product)

        # 3. Create a new product to connect the preceding Op to the Split Op.
        # Set the the preceding Op's output Product's consumer to Split Op.

        # The preceding Op's output products (products, since it was behaving like a Split) are going to be deleted,
        # since a Split is being inserted in the connected graph.
        # Save the preceding Op's output Product shape.
        # This is needed to create the new product from the preceding Op to the newly being inserted Split Op.
        new_product_shape = preceding_op.output.shape

        # Since the preceding Op was behaving like a Split Op, it  would have 2 products with the preceding Op as the
        # producer. Delete these products from the product dictionary.
        preceding_op_product_names = self.get_product_names_from_dotted_name(preceding_op.dotted_name)
        for name in preceding_op_product_names:
            # Important Notes
            # The following check is needed since ResNet uses the same Relu twice in BasicBlock's forward()
            # Please read the details comments in _add_consumers_to_split_op_product()
            if preceding_op.name in name:
                deleted_product = self._products.pop(name)
                logger.debug("Insert Split Op: Step 3. Deleted product: %s", deleted_product)

        new_product_name = preceding_op.name + '__to__' +  split_op.name
        new_product_shape = preceding_op.output.shape
        new_product = self._add_product(new_product_name, new_product_shape)
        new_product.producer = preceding_op
        preceding_op.output = new_product
        preceding_op.output.consumers.append(split_op)

        # 4. Set the Split Op's input to point to current Op's output.
        #new_name = preceding_op.name + '__to__' + split_op.name
        split_op.inputs.append(preceding_op.output)

    def _determine_split_behavior_for_op_and_insert_split_op_in_connected_graph(self, op: Op):
        """ Determine if an Op's output is used as an input to more than one Op. If it is, create a Split Op and
         insert it in the connected graph, below this Op.

         Note:
             The split is done in the forward() function of a model and is NOT a PyTorch OP.
             For constructing the graph, tensorboardX is being used and correctly, it doesn't detect Split
             as an OP. For mask propagation purposes, it is important to model the split as an OP. """

        name = op.name
        dotted_name = op.dotted_name

        # Get the output product names.
        output_product_names = self.get_product_names_from_dotted_name(dotted_name)

        name_list = []
        for prod_name in output_product_names:
            to_pos = prod_name.find('to')
            first_name = prod_name[:to_pos]
            name_list.append(first_name)

        # Split ops have 2 or more output products
        if len(output_product_names) > 1:
            name_list = [+1 for prod in name_list if name in prod]
            if len(name_list) > 1:
                logger.debug("%s is a split Op", op.dotted_name)

                # Create a Split Op
                split_op = self._create_split_op(op)

                # Insert the Split Op in the connected graph.
                self._insert_split_op_in_connected_graph(op, split_op)


def trace_and_parse(model: torch.nn.Module, x_inputs: Union[torch.Tensor, Tuple[torch.Tensor]]):
    """ Taken from torch.utils.tensorboard._pytorch_graph.py """

    with torch.onnx.set_training(model, False):
        try:
            trace, _ = torch.jit.get_trace_graph(model, x_inputs)
        except RuntimeError:
            logger.error('Error occurs, failed to get jit trace')
            _ = model(*x_inputs)  # don't catch, just logger.debug the error message
            logger.error("Checking if it's onnx problem...")
            try:
                torch.onnx.export(model, x_inputs, tempfile.TemporaryFile(), verbose=True)
            except RuntimeError:
                logger.error("Your model fails onnx too, please report to onnx team")
            return None

    # Only, PyTorch 1.1 and above are supported.
    # assert LooseVersion(torch.__version__) >= LooseVersion("1.1")
    _optimize_trace(trace, OperatorExportTypes.ONNX) # pylint: disable=protected-access

    graph = trace.graph()  # C implementation
    list_of_nodes = torch.utils.tensorboard._pytorch_graph.parse(graph,  # pylint: disable=protected-access
                                                                 args=[],
                                                                 omit_useless_nodes=True)
    return list_of_nodes


def torch_version_okay():
    """ Only, PyTorch 1.1 and above are supported. """

    if LooseVersion(torch.__version__) >= LooseVersion("1.1"):
        return True
    return False


def parse_anonymous_op_xname(xname, xop):
    """ Parses the xname for anonymous operations. """
    dotted_name = xname.replace('/', '.')
    onnx_prefix = 'onnx::'
    aten_prefix = 'aten::'
    prim_prefix = 'prim::'
    assert xop.find(onnx_prefix) == 0 or xop.find(aten_prefix) == 0 or xop.find(prim_prefix) == 0
    if xop.find(onnx_prefix) == 0:
        op_type = xop[len(onnx_prefix):]
    elif xop.find(prim_prefix) == 0:
        op_type = xop[len(prim_prefix):]
    else:
        op_type = xop[len(aten_prefix):]
    return dotted_name, op_type


def parse_named_op_xname(xname):
    """ Parses the xname for named operations."""
    # e.g. VGG / Sequential[features] / Conv2d[0] / Conv_33
    xparts = xname.split('/')
    module_name_parts = []
    op_types = []
    for part in xparts[:-1]:
        bracket_pos = part.find('[')
        if bracket_pos < 0:
            module_name_parts.append(part)
        else:
            op_type = part[:bracket_pos]
            op_types.append(op_type)
            var_name = part[bracket_pos + 1:-1]
            module_name_parts.append(var_name)

    return '.'.join(module_name_parts), op_types[-1]


def get_module_name(xname):
    """ Parses the xname for named operations."""
    # e.g. VGG / Sequential[features] / Conv2d[0] / Conv_33
    xparts = xname.split('/')
    module_name_parts = []
    for part in xparts[:-1]:
        bracket_pos = part.find('[')
        if bracket_pos < 0:
            module_name_parts.append(part)
        else:
            var_name = part[bracket_pos + 1:-1]
            module_name_parts.append(var_name)

    return '.'.join(module_name_parts)


def is_functional(xname, list_of_module_names):
    """ Returns True if a module is a PyTorch Functional module.
    If not, returns False. """

    module_name = get_module_name(xname)
    if module_name in list_of_module_names:
        return False
    return True


def parse_xname(xname, xop, list_of_module_names):
    """ Parses the xname for operations. """
    is_anonymous = is_functional(xname, list_of_module_names)
    # The above check doesn't work for the Add operation.
    # Specific handling is needed for Add which is an anonymous operation.

    if is_anonymous:
        # e.g. MyModel/Flatten_5 ; or MyModel/aten::clamp_min_6 ; or MyModel/Conv2d/Conv_9
        dotted_name, op_type = parse_anonymous_op_xname(xname, xop)
    else:
        # e.g. MyModel/Linear[linear1]/Gemm_7
        dotted_name, op_type = parse_named_op_xname(xname)
    return dotted_name, op_type, is_anonymous


def get_output_shape_from_xnode(xnode):
    """
    Return the output size(s) from the xnode's output shapes attribute.
    It is possible that some modules have more than one output.

    :param xnode:
    :return: list of output shapes
    """

    # The attributes could contain a list of output shapes.
    output_shapes_list = xnode.attr['_output_shapes'].list

    for shape in output_shapes_list.shape:
        for field in shape.DESCRIPTOR.fields:
            if field.name in 'dim':
                value = getattr(shape, "dim")
                if field.label == field.LABEL_REPEATED:
                    output_size = [getattr(element, "size") for element in value]
                    return output_size
    return []


def _temporary_fix_for_handling_dropouts_in_mnist_fix_me(dotted_name, list_of_module_names):
    """
    This is a temporary solution. Please refer the comments _parse_xnode() where this function is
    being called.

    :param dotted_name: The dotted name which represents teh fused Droput and Conv2d modules in MNIST model.
    :param list_of_module_names: list of a model's module names
    :return: the modified op type and dotted name representing teh fused Conv2d module.
    """

    # Since the Conv2d is getting fused and Winnower winnows the Conv2d modules,
    # the modified Op type is set to Conv2d.
    modified_op_type = 'Conv2d'
    modified_dotted_name = get_previous_module_dotted_name(dotted_name, list_of_module_names)
    return modified_op_type, modified_dotted_name


def get_previous_module_dotted_name(dotted_name, list_of_module_names):
    """
    For a given module

    :param dotted_name: the module for which the previous module's dotted name is returned.
    :param list_of_module_names: list of a model's module names
    :return: the previous module's dotted name.
    """
    for index, item in enumerate(list_of_module_names):
        if item == dotted_name:
            return list_of_module_names[(index - 1)]
    return None



#
# The following two functions are copied from torch.utils.tensorboard._pytorch_graph.py
# In the above PyTorch file, these 2 functions are defined as nested functions and so unable
# to call them directly.
#
# pylint: disable=protected-access
def _optimize_trace(trace, operator_export_type):
    trace.set_graph(_optimize_graph(trace.graph(), operator_export_type))


def _optimize_graph(graph, operator_export_type):
    # torch._C._jit_pass_remove_inplace_ops(graph)
    # we record now record some ops like ones/zeros
    # into a trace where we previously recorded constants
    # use constant prop to maintain our current level of onnx support
    # without implementing symbolics for all of them
    torch._C._jit_pass_constant_propagation(graph)
    torch.onnx.utils._split_tensor_list_constants(graph, graph)
    # run dce to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)

    # torch._C._jit_pass_canonicalize_ops(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_lint(graph)

    # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
    torch._C._jit_pass_prepare_division_for_onnx(graph)
    # onnx only supports tensors, so we turn all out number types into tensors
    torch._C._jit_pass_erase_number_types(graph)
    # onnx does not support tuples, so try to remove them

    #### torch._C._jit_pass_lower_all_tuples(graph)
    #### torch._C._jit_pass_peephole(graph, True)

    torch._C._jit_pass_lint(graph)

    if operator_export_type != OperatorExportTypes.RAW:
        graph = torch._C._jit_pass_onnx(graph, operator_export_type)
        torch._C._jit_pass_lint(graph)
        # torch._C._jit_pass_onnx_peephole(graph)
        torch._C._jit_pass_lint(graph)

    #### torch._C._jit_pass_dce(graph)

    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_fixup_onnx_loops(graph)
    torch._C._jit_pass_lint(graph)
    graph = torch._C._jit_pass_canonicalize(graph)
    torch._C._jit_pass_lint(graph)
    return graph


def _fill_op_info(op, module):
    """ Fill in input dimension, output dimension, groups, and module op info """

    if op.type in ('Conv', 'Conv2d'):
        op.in_dimension = module.in_channels
        op.out_dimension = module.out_channels
        op.groups = module.groups
    elif op.type in 'Linear':
        op.in_dimension = module.in_features
        op.out_dimension = module.out_features
    op.model_module = PytorchModelModule(module)
