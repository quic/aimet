#!/usr/bin/env python3.5

#  =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

import re
from typing import Tuple, Union, List, Dict
import torch

from aimet_common.connected_graph.connectedgraph import ConnectedGraph as AimetCommonConnectedGraph, get_ordered_ops
from aimet_common.connected_graph.product import Product
from aimet_common.connected_graph.operation import Op, determine_preceding_op_input_product_index_in_multi_input_op
from aimet_common.model_module import PytorchModelModule
from aimet_common.utils import AimetLogger, ModelApi, api_channel_index_dict
from aimet_torch.utils import run_hook_for_layers, is_leaf_module

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Winnow)


# pylint: disable=too-many-instance-attributes
class ConnectedGraph(AimetCommonConnectedGraph):
    """
    For construction of a graph that connects operations together (
        either module or functional) as producers and consumers of tensors.
        Note that the graph has two kinds of nodes: operations and products."""

    # pylint: disable=unused-argument
    def __init__(self, model: torch.nn.Module, model_input: Tuple[torch.Tensor]):
        """
        Init function for connected graph
        :param model: Pytorch model to create connected graph from
        :param model_input: Example input to model.  Can be a single tensor or a list/tuple of input tensors
        """
        super().__init__()
        self._model = model
        self._model_input = model_input
        self._model_name = type(model).__name__
        # Maps pytorch module names to modules
        self._name_to_module = dict()
        # Maps pytorch modules to module names
        self._module_to_name = dict()
        # Maps pytorch modules to connected graph ops
        self._module_to_op_dict = dict()

        # Parameters dict to hold parameters identified from trace code, to be made into Products
        # Maps parameter name as found in trace code to tuple of (corresponding module, parameter type, parameter shape)
        self._parameters = {}

        # Ops dict to map names of ops as found in trace code to corresponding Ops that are created
        self._named_ops = {}

        # In trace code, a group of operations can show up as a named line to be used later.
        # Ex. _6 = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        # In this case, save this as a named group, and when we parse inputs with split_inputs in a different line,
        # treat each member in the group as a separate input.
        self._named_groups = {}
        self._op_count = 0
        self._split_count = 0  # Use it in the name of split Ops getting added to the connected graph.

        # Operations for which which we will completely ignore and skip processing (will not look at op arguments
        # either).
        self._skip_processing_ops = {
            'annotate'
        }

        # If the following are detected as the outermost function in trace code, ignore it and evaluate what is inside.
        self._patterns_to_ignore = [
            r'torch\.t\(([_A-Za-z0-9.]+)\)',
            r'int\(([_A-Za-z0-9.]+)\)'
        ]


        # Parameter types which we do not care about, and will not be identified as parameters in _parse_expression()
        self._parameter_types_to_ignore = [
            'num_batches_tracked'
        ]

        # List of ops in the order they are traversed using the forward function
        self.ordered_ops = []

        self._generate_module_lookup_table(model)
        self._construct_graph(model, model_input)

    # Map torch module types to normalized names to provide backward compatibility to
    # trace code based construction
    op_type_map = {
        torch.nn.Conv2d: ['convolution'],
        torch.nn.ConvTranspose2d: ['convolution'],
        torch.nn.BatchNorm2d: ['batch_norm'],
        torch.nn.ReLU: ['relu'],
        torch.nn.MaxPool2d: ['max_pool2d'],
        torch.nn.AdaptiveAvgPool2d: ['adaptive_avg_pool2d'],
        torch.nn.AvgPool2d: ['avg_pool2d'],
        torch.nn.Linear: ['addmm', 'matmul'],
        torch.nn.Dropout: ['dropout'],
        torch.nn.Dropout2d: ['feature_dropout']
    }

    # Graph nodes for which which we will completely ignore and skip processing
    ignore_graph_nodes = [
        "prim::Constant",
        "prim::ListConstruct",
        "prim::TupleConstruct",  # TODO PoseModel returns Tensor Tuple - valid usecase?
        "aten::Int",
        "aten::t"
    ]
    def __del__(self):
        """
        Destructor of ConnectedGraph class
        break the dependencies of Ops with Product
        """
        for product in self._products.values():
            product.producer = None
            product.set_consumers_to_null()

    def get_op_from_module_name(self, name: str) -> Union[Op, None]:
        """
        Given the name of a operation/module, return the corresponding op in ops dict
        :param name: Pytorch module name
        :return: Connected graph operation corresponding to pytorch module name.  Returns None if not found
        """
        module = self._name_to_module.get(name, None)
        if module:
            return self._module_to_op_dict.get(module, None)
        return None

    def get_all_ops(self) -> Dict[str, Op]:
        """ Returns the ops dictionary """
        return self._ops

    def get_all_products(self) -> Dict[str, Product]:
        """ Returns the products dictionary """
        return self._products

    def get_product(self, name: str) -> Product:
        """
        Returns the product with the name passed in the argument
        :param name: Product name
        """
        return self._products.get(name, None)

    def _generate_module_lookup_table(self, model: torch.nn.Module):
        """
        Generates a look up dictionary for getting modules from their names.
        :param model: Pytorch model
        """
        for name, module in model.named_modules(prefix=self._model_name):
            self._name_to_module[name] = module
            self._module_to_name[module] = name

    @staticmethod
    def _generate_module_tensors_lookup_table(model: torch.nn.Module, model_input: Tuple[torch.Tensor]) -> \
            Dict[str, Tuple[Tuple[torch.Tensor]]]:
        """
        Generates a look up dictionary for getting modules from their names.
        :param model: Pytorch model
        :return: Map of modules and input and output tensor obtained from a forward pass
        """
        module_tensor_tuples_map = dict()

        def forward_hook(curr_module: torch.nn.Module,
                         input_tensor_tuple: Tuple[torch.Tensor],
                         output_tensor_tuple: Tuple[torch.Tensor]):
            """
            Custom forward hook function to add every module to module-to-tensor dict.
            :param curr_module: Current module being traversed during forward pass.
            :param input_tensor_tuple: tuple of input tensors to the current module
            :param output_tensor_tuple: tuple of output tensors of the current module
            """
            # Currently, we assume that multiple input tensors have the same shape, and likewise for output tensors.
            module_tensor_tuples_map[curr_module] = (input_tensor_tuple, output_tensor_tuple)

        input_shapes = [inp.shape for inp in model_input]
        run_hook_for_layers(model, input_shapes, forward_hook, leaf_node_only=False)

        return module_tensor_tuples_map

    def _construct_graph(self, model: torch.nn.Module, model_input: Tuple[torch.Tensor]):
        """
        Construct connected graph from model and example inputs.
        :param model: Pytorch model to create connected graph from
        :param model_input: Example input to model.  Can be a single tensor or a list/tuple of input tensors
        """
        if torch.__version__ == '1.1.0':
            trace = torch.jit.trace(model, model_input)
            # Parse trace code to create ops and products
            self._parse_trace_code(trace.code)
            # Associate ops in connected graph with corresponding pytorch modules, and fill in shapes if possible
            self._fill_op_modules_and_shapes()
        else:
            module_tensor_tuples_map = ConnectedGraph._generate_module_tensors_lookup_table(model, model_input)
            self._parse_trace_graph(model, model_input, module_tensor_tuples_map)
        # Create parameters for ops such as conv, batchnorm, etc.
        self._fill_op_params()

        # For each split in the model, insert a corresponding split Op in the connected graph.
        ops_list = [op for op in self._ops.values()]
        for op in ops_list:
            self._determine_split_behavior_for_op_and_insert_split_op_in_connected_graph(op)

        self._fill_empty_shapes()

    def _parse_trace_graph(self, model: torch.nn.Module, model_input: Tuple[torch.Tensor],
                           module_tensor_tuples_map: Dict[str, Tuple[Tuple[torch.Tensor]]]):
        # pylint: disable=protected-access
        """
        Implements a depth-first graph extraction to create an equivalent connected graph representation
        with Ops and Products. depth-first extraction is realized using recursion.

        :param node: trace graph node
        :param model: Pytorch model to create connected graph from
        :param model_input: Example input to model.  Can be a single tensor or a list/tuple of input tensors
        :param module_tensor_tuples_map: Map of modules and input and output tensor obtained from a forward pass
        """
        input_tensor_tuple, _ = module_tensor_tuples_map[model]
        trace = torch.jit.trace(model, input_tensor_tuple)

        # graph node equivalent Ops or Products locally constructed or populated via recursive invocation
        # initialized with input products
        model_debug_name, ops = self._create_input_products(model_input, trace.graph)
        # pending sub-graph that have to be explored
        pending_nodes = dict()
        # modules that are being referenced within the sub-graph
        modules = {model_debug_name: model}
        for node in trace.graph.nodes():

            if node.outputsSize() != 1:
                logger.error("multiple output Ops are not supported %s", str(node))
                raise NotImplementedError

            output_name: str = node.output().debugName()

            # retrieving a module reference
            if 'GetAttr' in node.kind():
                subgraph_model = ConnectedGraph._get_module_instance(node, modules)
                if output_name not in modules:
                    modules[output_name] = subgraph_model
                else:
                    raise ValueError("duplicate model for {0} -> {1} and {2}".format(
                        output_name, modules[output_name], subgraph_model))
                if not is_leaf_module(subgraph_model):
                    pending_nodes[output_name] = subgraph_model

            # invoking forward method
            elif 'CallMethod' in node.kind():
                self.parse_callmethod_node(node, module_tensor_tuples_map, modules, ops, pending_nodes)

            # functional operations e.g. cat, size etc
            elif node.kind() not in self.ignore_graph_nodes:
                ops[output_name] = self._create_functional_op(node, ops)

        # return the last op enqueued in the sub-graph forward pass
        return self.ordered_ops[-1]

    def parse_callmethod_node(self, node: torch._C.Node,
                              module_tensor_tuples_map: Dict[str, Tuple[Tuple[torch.Tensor]]],
                              modules: Dict[str, torch.nn.Module],
                              ops: Dict[str, Union[Op, Product]],
                              pending_nodes: Dict[str, torch._C.Node]):
        # pylint: disable=protected-access
        """
        The call method node signifies invocation of the forward method, this method extracts an Op representation of
        the module or alist of Ops in case of module representing a sub-graph. Typically the node has the following construct:
            %output_N : Tensor = prim::CallMethod[name="forward"](%output_L, %output_M)
        :param node: trace graph node i.e. 'CallMethod' node
        :param module_tensor_tuples_map: Map of modules and input and output tensor obtained from a forward pass
        :param modules: dictionary of module indexed by output_name referenced in the sub-graph
        :param ops: dictionary of Ops and Products indexed by output names referenced in the graph
        :param pending_nodes: dictionary of torch graph nodes index of output_name that have not been resolved.
        """
        output_name: str = node.output().debugName()
        inputs = self._resolve_input_nodes(node)
        # 1st input is a reference on which the call method is being invoked.
        input_name: str = inputs[0].debugName()
        if input_name in pending_nodes:
            input_ops = [ops[i.debugName()] for i in inputs[1:]]

            # the op returned on parsing the sub-graph shall be last op in the sub-graph forward pass
            ops[output_name] = self._parse_trace_graph(pending_nodes[input_name], input_ops,
                                                       module_tensor_tuples_map)
        if input_name in modules and is_leaf_module(modules[input_name]):
            # the graph is fully represented by a directional graph of leaf torch modules so the recursion is
            # stopped at this level.
            ops[output_name] = self._create_leaf_module_op(modules[input_name],
                                                           inputs, ops, module_tensor_tuples_map)

    def _create_input_products(self, model_input: Tuple[torch.Tensor], graph: torch._C.Graph) -> \
            Tuple[str, Dict[str, Product]]:
        # pylint: disable=protected-access
        """
        Creates a dictionary of input products index by input name referenced in the sub-graph
        :param model_input: Example input to model.  Can be a single tensor or a list/tuple of input tensors
        :param graph: trace graph representing the model or sub-set of the model
        :return: model input name , dictionary of input products
        """
        products = dict()
        model_debug_name = None
        for i, inp in enumerate(graph.inputs()):
            input_name: str = inp.debugName()
            if i == 0:
                model_debug_name = input_name
            else:
                inp_op = model_input[i - 1]
                if isinstance(inp_op, torch.Tensor):
                    shape = list(inp_op.shape)
                    self._parameters[input_name] = (None, 'input', shape)
                    product = Product(input_name, shape)
                    product.is_model_input = True
                    self._products[product.name] = product
                    products[input_name] = product
                elif isinstance(inp_op, tuple([Product, Op])):
                    products[input_name] = inp_op
                else:
                    logger.warning("ignoring input %s of unknown type %s", str(inp_op), type(inp_op))
        return model_debug_name, products

    @staticmethod
    def _get_attribute_name(node: torch._C.Node) -> List[str]:
        # pylint: disable=protected-access
        """
        Retrieve the attributes associated with the graph node
        :param node: trace graph node
        :return: a dictionary of attributes associated with the node
        """
        attributes = dict()
        # node description has pseudo-code of the form  '... torch_mangle_2.Module = prim::GetAttr[name="fc"](%self.1)'
        # for the above example attributeNames() iterator should return a string 'name'
        node_desc = str(node)
        for attribute_name in node.attributeNames():
            pattern = attribute_name + '="'
            if pattern in node_desc:
                attributes[attribute_name] = node_desc.split(pattern)[1].split('"')[0]
        return attributes

    @staticmethod
    def _get_module_instance(node: torch._C.Node,
                             modules: Dict[str, torch.nn.Module]) -> torch.nn.Module:
        # pylint: disable=protected-access
        # pylint: disable=eval-used
        """
        Get the torch.nn.Module referenced by the node.
        the node are typically of 1st type shown below and in case of nn.ModuleList (2nd case):
            %output_N : __torch__.torch ... Module = prim::GetAttr[name="fc"](%self.1)
            %output_M : __torch__.torch ... Module = prim::GetAttr[name="5"](%output_K)
            2nd type shown about is for nn.ModuleList which reference a prior output '%output_K' instead of '%self.1'
        :param node: trace graph node
        :param modules: dictionary of module index by output_name referenced in the sub-graph
        :return: list of attributes defined with the node
        """
        input_name: str = node.input().debugName()
        attributes = ConnectedGraph._get_attribute_name(node)
        model = modules[input_name]
        if isinstance(model, tuple([torch.nn.modules.container.Sequential, torch.nn.modules.ModuleList])):
            sub_model = eval('model[' + attributes['name'] + ']')
        else:
            sub_model = eval('model.' + attributes['name'])
        return sub_model

    def _resolve_input_nodes(self, node: torch._C.Node) -> List[torch._C.Node]:
        # pylint: disable=protected-access
        """
        recursively aggregate inputs nodes that produce inputs consumed by the node
        recursion is used to aggregate inputs feeding into the input_node if the node belongs to a ignored node list.
        :param node: trace graph node
        :return: list of producer nodes that feed the node
        """
        inputs = []
        for _, inp in enumerate(node.inputs()):
            if inp.node().kind() in self.ignore_graph_nodes:
                inputs.extend(self._resolve_input_nodes(inp.node()))
            else:
                inputs.append(inp)
        return inputs

    def _create_leaf_module_op(self, model: torch.nn.Module,
                               inputs: List[torch._C.Node], ops: Dict[str, Union[Op, Product]],
                               module_tensor_tuples_map: Dict[str, Tuple[Tuple[torch.Tensor]]]) -> Op:
        # pylint: disable=protected-access
        """
        Creates a fully populated Op and along with associated products representing inputs
        :param model: PyTorch Module representing the model or a sub-set of the model
        :param inputs: list of producer graph nodes
        :param ops: dictionary of Ops and Products indexed by output names referenced in the graph
        :return: Ops
        """
        # use nominal Op type if its a known type else use torch defined Module name
        if isinstance(model, tuple(self.op_type_map.keys())):
            op_type = self.op_type_map[type(model)][0]
        else:
            op_type = type(model).__name__

        op = self._create_op_and_products(op_type, inputs[1:], ops)

        # populating module info associated with Op
        op.model_module = PytorchModelModule(model)
        self._module_to_op_dict[model] = op
        op.dotted_name = self._module_to_name[op.get_module()]
        _fill_conv_op_info(op, model)

        # populating input and output shapes obtained via hook and forward pass
        input_tensor_tuple, output_tensor_tuple = module_tensor_tuples_map[model]
        _fill_and_check_op_product_shapes(op, list(input_tensor_tuple[0].shape), list(output_tensor_tuple[0].shape))
        return op

    def _create_functional_op(self, node: torch._C.Node, ops: Dict[str, Union[Op, Product]]) -> Union[Op, None]:
        # pylint: disable=protected-access
        """
        Creates an Op and along with associated products representing inputs. If output shape is available then shapes
        attribute are populated as well.
        :param node: trace graph node
        :param inputs: list of producer graph nodes
        :param ops: dictionary of Ops and Products indexed by output names referenced in the graph
        :return: Ops
        """
        inputs = self._resolve_input_nodes(node)

        # extracting Op type from node.kind string e.g. aten::relu_, aten::size etc
        op_type = node.kind().split("::")[-1].lstrip('_').rstrip('_')
        op = self._create_op_and_products(op_type, inputs, ops)

        # Determine the output_shape based on output_type
        output_type = node.output().type()
        if isinstance(output_type, torch._C.TensorType):
            output_shape = list(output_type.sizes())
        elif isinstance(output_type, torch._C.TupleType) and \
                isinstance(output_type.elements()[0], torch._C.TensorType):
            # the first output_shape is assumed to define the shape of the output tensor
            output_shape = list(output_type.elements()[0].sizes())
        else:
            # output is not of a Tensor type e.g. Int, skip setting shape fpr this op
            return op

        # first input tensor is assumed to define the shape of the input tensor(s)
        inp_op = ops[inputs[0].debugName()]
        if isinstance(inp_op, Product):
            input_shape = inp_op.shape
        else:
            input_shape = inp_op.output_shape
        _fill_and_check_op_product_shapes(op, input_shape, output_shape[1:])
        return op

    def _create_op_and_products(self, op_type: str, inputs: List[torch._C.Node],
                                ops: Dict[str, Union[Op, Product]]) -> Op:
        # pylint: disable=protected-access
        """
        Creates an Op and along with associated products representing inputs.
        :param op_type: string representation fo Op type could be one of the following:-
            > normalized name if mapping exists in op_type_map
            > class name if associated with an instance of torch.nn.Module
            > string extracted from graph node description in case of functional Op
        :param inputs: list of producer graph nodes
        :param ops: dictionary of Ops and Products indexed by output names referenced in the graph
        :return: Ops
        """
        unique_op_name = self._make_unique_op_name(op_type)
        op = Op(unique_op_name, unique_op_name, None, False, op_type)
        self.ordered_ops.append(op)
        self._ops[unique_op_name] = op
        for input_node in inputs:
            input_name = input_node.debugName()
            resolved_inp = ops[input_name]
            if isinstance(resolved_inp, Op):
                # Create a product linking the identified Operation with the current Operation.
                self._create_and_link_inter_op_product(resolved_inp, op)
            elif isinstance(resolved_inp, Product):
                self._associate_op_with_parameter_product(op, resolved_inp)
        return op

# trace code implementation will be deprecated once migration to PyTorch v1.4 is complete
# pylint: disable=too-many-lines
    def _parse_trace_code(self, code: str):
        """
        Given a torch.jit.trace code of the model, parse the code to create ops and products
        :param code: code attribute of torch.jit.trace output of the model
        """
        lines = code.split('\n')
        # Find indices corresponding to the starts of different sections in the code.
        first_parameter_line_index, first_operation_line_index, first_return_line_index = \
            _get_operation_and_return_line_indices(lines)

        # Identify model inputs in the code and create a corresponding Operation for each one.  These are named ops
        # so add them to both self._ops as well as self._named_ops.
        self._parse_input_lines(lines[:first_parameter_line_index])
        self._parse_parameter_lines(lines[first_parameter_line_index:first_operation_line_index])
        self._parse_operation_lines(lines[first_operation_line_index:first_return_line_index])
        self._parse_return_lines(lines[first_return_line_index:])

    def _parse_input_lines(self, lines: List[str]):
        """
        Parse input lines and create an entry in self._parameters for each input
        :param lines: List of trace code lines pertaining to model inputs
        """

        # Input patterns that represent possible inputs to the graph in trace code. Ex. input, input0, input1, or
        # x, x0, x1, etc.
        input_patterns = [
            re.compile(r'(input([0-9]*))[,:]'),
            re.compile(r'(x([0-9]*))[,:]')
        ]
        found_input = False
        for input_pattern in input_patterns:
            for line in lines:
                result = input_pattern.search(line)
                if result:
                    found_input = True

                    # Get shape of the corresponding input
                    if not result.group(2):
                        shape = self._model_input[0].shape
                    else:
                        # First input has no number, second input is '0', third is '1', and so on.
                        # Ex. input, input0, input1, etc.  Need to add 1 to the ending number to get the correct index.
                        shape = self._model_input[int(result.group(2)) + 1].shape
                    self._parameters[result.group(1)] = (None, 'input', list(shape))
            if found_input:
                break

    def _parse_parameter_lines(self, lines: List[str]):
        """
        Parse the lines in the trace code that deal with obtaining parameters for certain modules.
        :param lines: List of trace code lines pertaining to obtaining parameters
        """
        underscore_param_pattern = re.compile(r'(_[0-9]+)')
        for line in lines:
            stripped_line = line.strip()
            split_at_equal = stripped_line.split(' = ')
            result = underscore_param_pattern.match(split_at_equal[1])
            if result:
                assert result.group(1) in self._parameters.keys()
                split_at_equal[1] = split_at_equal[1].replace(result.group(1), self._parameters[result.group(1)][1])

            split_at_equal[1] = split_at_equal[1].replace('self', self._model_name, 1)
            last_period_index = split_at_equal[1].rfind('.')
            param_type = split_at_equal[1][last_period_index+1:]
            corresponding_op = self._name_to_module.get(split_at_equal[1][:last_period_index], None)
            self._parameters[split_at_equal[0]] = (corresponding_op, param_type, None)

    def _parse_operation_lines(self, lines: List[str]):
        """
        Parse the lines in the trace code that deal with torch operations
        :param lines: List of trace code lines pertaining to torch operations
        """
        for line in lines:
            stripped_line = line.strip()
            split_at_equal = stripped_line.split(' = ')
            output_name = split_at_equal[0]
            _ = self._parse_expression(split_at_equal[1], output_name)

    def _parse_return_lines(self, lines: List[str]):
        """
        Parse return lines in the trace code.  Return lines may still contain expressions.
        :param lines: List of trace code lines pertaining to returns
        """
        for line in lines:
            stripped_line = line.strip()
            return_pattern = re.compile(r'return (.+)')
            result = return_pattern.match(stripped_line)
            if result:
                _ = self._parse_expression(result.group(1))

    def _parse_expression(self, expression: str, output_name=None) -> Union[Op, Product, None]:
        """
        Parse a single expression. This could be a torch operation, or name of an op or parameter already created, or an
        input argument to a torch operation.  Basically, this can be any line that is to the right of the '=' for an
        operation line in the code, or any input item of that operation, which could be an operation itself. Together
        with _parse_operation(), this works recursively to create all ops and products in a single line of trace code.
        :param expression: Expression to parse
        :param output_name: If this is an operation with a name, self_named_ops will register the created Op with this
        name.  This only happens for outermost operations in the trace code, and not for internal anonymous operations.
        :return: Either a corresponding Operation or Product for this expression, or None if not applicable
        """
        exp = expression
        # Ignore certain expressions which are not important with regards to connected graph.  Ex. torch.Tensor() or
        # int()
        for pattern in self._patterns_to_ignore:
            regex = re.compile(pattern)
            result = regex.match(exp)
            if result:
                exp = result.group(1)
                break

        # If expression is a parameter, either return the corresponding Product if it exists, or create one.
        if exp in self._parameters.keys() and self._parameters[exp][1] not in self._parameter_types_to_ignore:
            if exp in self._products.keys():
                return self._products[exp]

            product = Product(exp, self._parameters[exp][2])
            if self._parameters[exp][1] == 'input':
                product.is_model_input = True
            else:
                product.is_parm = True
            self._products[product.name] = product
            return product

        # If expression is the name of an operation we have processed earlier, return the corresponding Operation.
        if exp in self._named_ops:
            return self._named_ops[exp]

        # If expression is a torch operation, process its arguments with _parse_operation, and return the corresponding
        # Operation.
        # The below patter matches such expressions like:
        # torch._convolution(...)
        # ops.prim.NumToTensor(...)
        # ^CustomFunctionName()(...)
        function_pattern = re.compile(r'([\^_A-Za-z0-9.]+(\(\))*)\((.+)\)')
        result = function_pattern.match(exp)
        if result:
            # Extract operation name, taking only the rightmost part of function expression as the name, and stripping
            # leading and trailing '_' characters.  Ex. torch.relu_ -> relu, ops.prim.NumToTensor -> NumToTensor,
            # torch._convolution -> convolution, etc.
            operation = result.group(1)
            last_period_index = operation.rfind('.')
            operation = operation[last_period_index+1:]
            operation = operation.lstrip('_^')
            operation = operation.rstrip('_^')
            op = self._parse_operation(operation, result.group(3), output_name)
            return op

        # In trace code, a group of operations can show up as a named line to be used later.
        # Ex. _6 = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        # In this case, save this as a named group, and when we parse inputs with split_inputs in a different line,
        # treat each member in the group as a separate input.
        if exp[0] == '[' and exp[-1] == ']' and output_name:
            self._named_groups[output_name] = exp[1:-1]

        return None

    def _parse_operation(self, op_name: str, inputs: str, output_name=None) -> Union[Op, None]:
        """
        Parse a single operation's inputs, which can recursively be operations as well.  For each input, call
        _parse_expression to parse it.
        :param op_name: name of the torch operation
        :param inputs: inputs to the torch operation
        :param output_name: If this is an operation with a name, self_named_ops will register the created Op with this
        name.  This only happens for outermost operations in the trace code, and not for internal anonymous operations.
        :return: The Operation created as a result of parsing
        """
        # Skip processing this op if it is one of the ops we don't care about.
        if op_name in self._skip_processing_ops:
            return None

        # Parse each input argument of the operation and make a local list of the results.  Will either be Op, Product,
        # or None.
        input_list = _split_inputs(inputs, self._named_groups)
        parsed_inp_list = []
        for inp in input_list:
            parsed_inp = self._parse_expression(inp)
            parsed_inp_list.append(parsed_inp)

        # Setting op to none for now.  If all parsed_inp values are None, no point in creating the op.
        op = None
        for parsed_inp in parsed_inp_list:
            # Currently only do special processing for inputs if it turns out to be an Operation or a parameter.
            if parsed_inp:
                if not op:
                    # Create op object
                    unique_op_name = self._make_unique_op_name(op_name)
                    op = Op(unique_op_name, unique_op_name, None, False, op_name)
                    self.ordered_ops.append(op)
                    if output_name:
                        self._named_ops[output_name] = op
                    self._ops[unique_op_name] = op
                if isinstance(parsed_inp, Op):
                    # Create a product linking the identified Operation with the current Operation.
                    self._create_and_link_inter_op_product(parsed_inp, op)
                else:
                    # If input is a parameter, we are able to identify the actual pytorch module it corresponds to.
                    # Also link the created product to the Operation.
                    self._associate_op_with_parameter_product(op, parsed_inp)
        return op

    def _make_unique_op_name(self, op_name: str) -> str:
        """
        Given an op name, combine it with the self._op_count member to create a unique op name.  Increment
        self._op_count each time.
        :param op_name: Name of the operation to create a unique op name from.
        :return: The unique op name.
        """
        unique_op_name = op_name + '_' + str(self._op_count)
        self._op_count += 1
        return unique_op_name

    def _create_and_link_inter_op_product(self, parent_op: Op, current_op: Op):
        """
        Given a parent op and child op, create a product to link the two if it doesn't yet exist.
        :param parent_op: The parent op.
        :param current_op: The child op.
        """
        product_name = parent_op.name + '_to_' + current_op.name
        input_product = self.get_product(product_name)
        if not input_product:
            input_product = Product(product_name, None)
            self._products[input_product.name] = input_product
        parent_op.output = input_product
        input_product.producer = parent_op
        input_product.add_consumer(current_op)
        current_op.add_input(input_product)

    def _associate_op_with_parameter_product(self, op: Op, parameter_product: Product):
        """
        Given an op and a Product that represents a parameter of the op, link the op to the product.
        :param op: The op to link the parameter product to.
        :param parameter_product: The parameter product associated with the op.
        """
        if not op.model_module:
            param_tuple = self._parameters[parameter_product.name]
            if param_tuple[0]:
                op.model_module = PytorchModelModule(param_tuple[0])
        parameter_product.add_consumer(op)
        op.add_input(parameter_product)

    def _fill_op_modules_and_shapes(self):
        """
        Use a forward pass through the model with a custom hook to obtain the actual pytorch modules corresponding to
        named modules defined in the model.  Then cross reference with the connected graph created earlier to match ops
        with the modules.  This assumes that modules are traversed in the same order in the forward pass as they showed
        up in the trace code.  Since trace code is also generated from a forward pass through the model, this seems to
        be a valid assumption.
        """
        named_module_list = []

        def forward_hook(curr_module, input_tensor_tuple, output_tensor_tuple):
            """
            Custom forward hook function to simply add the current module to named_module_list.
            :param curr_module: Current module being traversed during forward pass.
            :param input_tensor_tuple: tuple of input tensors to the current module
            :param output_tensor_tuple: tuple of output tensors of the current module
            """
            # Currently, we assume that multiple input tensors have the same shape, and likewise for output tensors.
            named_module_list.append((curr_module,
                                      list(input_tensor_tuple[0].shape),
                                      list(output_tensor_tuple[0].shape)))

        input_shapes = [inp.shape for inp in self._model_input]
        run_hook_for_layers(self._model, input_shapes, forward_hook)

        ordered_ops_index = 0
        module_id_index = 0
        # Step through named_module_list as well as self.ordered_ops in order, matching up corresponding ops.
        # self.ordered_ops may have more ops than named_module_list if there are anonymous ops
        while module_id_index < len(named_module_list) and ordered_ops_index < len(self.ordered_ops):
            current_op = self.ordered_ops[ordered_ops_index]
            current_named_module, input_shape, output_shape = named_module_list[module_id_index]

            # If named_module is a type we don't recognize, skip it.
            if not isinstance(current_named_module, tuple(self.op_type_map.keys())):
                module_id_index += 1
                logger.debug('No known mapping for %s in pytorch module types to op types, skipping.',
                             type(current_named_module))
                continue
            # If current module from named_module_list and Op from self.ordered_ops match in type, pair them up.
            # If there is already a module associated with the Op, check that it is identical to the current module
            # from named_module_list.
            if current_op.type in self.op_type_map[type(current_named_module)]:
                if current_op.model_module:
                    if not current_op.get_module() == current_named_module:
                        logger.error('Module %s identified during fill_op_modules does not agree with module %s found'
                                     'from parameters', type(current_op.get_module()), type(current_named_module))
                    self._module_to_op_dict[current_op.get_module()] = current_op
                    current_op.dotted_name = self._module_to_name[current_op.get_module()]
                else:
                    current_op.model_module = PytorchModelModule(current_named_module)
                    self._module_to_op_dict[current_named_module] = current_op
                    current_op.dotted_name = self._module_to_name[current_named_module]

                _fill_and_check_op_product_shapes(current_op, input_shape, output_shape)
                _fill_conv_op_info(current_op, current_op.get_module())

                module_id_index += 1
            ordered_ops_index += 1
        assert module_id_index == len(named_module_list)

    def _fill_op_params(self):
        """
        For certain ops like convolution, batch norm, and linear, create products for their parameters if they don't
        exist yet.
        """
        for op in self._ops.values():
            module = op.get_module()
            name = self._module_to_name.get(module, None)
            if op.type in ['convolution', 'batch_norm', 'addmm', 'matmul']:
                if module.weight is not None:
                    product_name = name + '.weight'
                    self._create_and_add_param_product_if_not_exists(op, product_name, list(module.weight.shape))
                if module.bias is not None:
                    product_name = name + '.bias'
                    self._create_and_add_param_product_if_not_exists(op, product_name, list(module.bias.shape))
            if op.type == 'batch_norm':
                # If batch_norm, fill in rest of bn params
                if module.running_mean is not None:
                    product_name = name + '.running_mean'
                    self._create_and_add_param_product_if_not_exists(op, product_name, list(module.running_mean.shape))
                if module.running_var is not None:
                    product_name = name + '.running_var'
                    self._create_and_add_param_product_if_not_exists(op, product_name, list(module.running_var.shape))

    def _create_and_add_param_product_if_not_exists(self, op: Op, product_name: str, shape):
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

    def _determine_split_behavior_for_op_and_insert_split_op_in_connected_graph(self, op: Op):
        """
        Determine if an Op's output is used as an input to more than one Op. If it is, create a Split Op and
        insert it in the connected graph, below this Op.
        Note that the split is done in the forward() function of a model and is NOT a PyTorch OP.
        :param op: Op to check if output is used as an input to more than one op.
        """

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

    def get_product_names_from_dotted_name(self, dotted_name: str) -> List[str]:
        """
        Returns all names of products whose producer op dotted name matches the argument dotted name.
        For Residual models, same producer will have multiple products.
        During connected graph construction, only one output product can be associated with an op, so previous output
        products are overwritten when a new op is created.  Thus we must search through products dictionary for all
        output products corresponding to an op.
        :param dotted_name: Dotted name for connected graph op to check for output products.
        :return: List of products
        """

        matched_products = list()
        for product in self._products.values():
            if product.producer:
                if product.producer.dotted_name == dotted_name:
                    matched_products.append(product.name)
        return matched_products

    def _create_split_op(self, op: Op) -> Op:
        """
        The op's output is split in the forward function. To model it correctly, create a Split Op.
        :param op: Op to create split op after
        :return: Split op that was created
        """
        split_name_parts = ['Split_', str(self._split_count)]
        split_name = ''.join(split_name_parts)
        self._split_count += 1
        split_dotted_name_parts = [self._model_name, split_name]
        split_dotted_name = '.'.join(split_dotted_name_parts)
        is_anonymous = True
        split_op = Op(split_name, split_dotted_name, op.output_shape, is_anonymous, 'Split')
        self._ops[split_name] = split_op
        return split_op

    def _insert_split_op_in_connected_graph(self, preceding_op: Op, split_op: Op):
        """
        Insert a Split Op below the preceding Op in the connected graph.
        :param preceding_op: Op prior to split op
        :param split_op: Split op to insert
        """

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
        # Examples of Op with multiple input products: add, cat (Concat)
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

        new_product_name = preceding_op.name + '__to__' + split_op.name
        new_product_shape = preceding_op.output.shape
        new_product = self._add_product(new_product_name, new_product_shape)
        new_product.producer = preceding_op
        preceding_op.output = new_product
        preceding_op.output.consumers.append(split_op)

        # 4. Set the Split Op's input to point to current Op's output.
        # new_name = preceding_op.name + '__to__' + split_op.name
        split_op.inputs.append(preceding_op.output)

    def _create_split_op_output_product(self, preceding_op: Op, split_op: Op) -> Product:
        """
        Create output product of the split op and connected it to the split op
        :param preceding_op: Op prior to split op
        :param split_op: Split op to create output product for
        :return: Output product of the split op
        """
        split_op_product_name = split_op.name + '__to__' + 'multiple_ops'
        split_op_product_shape = preceding_op.output.shape
        split_op_product = self._add_product(split_op_product_name, split_op_product_shape)
        split_op_product.producer = split_op
        return split_op_product

    def _add_product(self, name: str, shape: List[int]) -> Product:
        """
        Add product to self._products dictionary
        :param name: Name of product
        :param shape: Shape of product
        :return: Product that was created
        """
        assert name not in self._products
        product = Product(name, shape)
        self._products[name] = product
        return product

    def _add_consumers_to_split_op_product(self, preceding_op: Op, split_op_product: Product):
        """
        A Split Op's output product has multiple consumers. Add them to the product.
        :param preceding_op: Op prior to split op
        :param split_op_product: Output product of split op
        """

        dotted_name = preceding_op.dotted_name
        output_product_names = self.get_product_names_from_dotted_name(dotted_name)

        # Important Notes
        # ResNet model uses the same Relu twice in the forward function of ResNet's BasicBlock.
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
            if a_consumer.type in ('cat', 'add'):
                # Need to insert the newly created split_op product in the correct input index of the cat Op :)
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

    def _fill_empty_shapes(self):
        """ Anonymous ops like add or concat do not have shapes associated with them when their ops are created.
        Traverse through ops in the graph in order and use existing product shape information to try to infer what the
        shapes of ops and products with missing shapes are.  This may not be completely accurate in the face of reshape
        and unknown ops. """

        starting_ops = []
        for product in self.get_all_products().values():
            if product.is_model_input:
                for consumer in product.consumers:
                    if consumer not in starting_ops:
                        starting_ops.append(consumer)

        ops_in_order = get_ordered_ops(starting_ops)
        for op in ops_in_order:
            for inp in op.inputs:
                assert inp.shape is not None

            # If op's output product has a shape already, simply use that shape as op's output_shape as well.
            if op.output_shape is None:
                if op.output and op.output.shape is not None:
                    op.output_shape = op.output.shape
                    continue

                # If op is of type cat, add the number of channels of incoming products and declare that is the number
                # of out channels in the outgoing product.
                if op.type == 'cat':
                    num_channels = 0
                    for inp in op.inputs:
                        num_channels += inp.shape[api_channel_index_dict[ModelApi.pytorch]]
                    concat_shape = op.inputs[0].shape
                    concat_shape[api_channel_index_dict[ModelApi.pytorch]] = num_channels
                    op.output_shape = concat_shape

                # Otherwise, assume that the input shape is unchanged (this may be wrong, but hopefully should not
                # matter in the scheme of mask propagation since ops that matter should not be anonymous).
                else:
                    op.output_shape = op.inputs[0].shape
                if op.output:
                    op.output.shape = op.output_shape


def _fill_and_check_op_product_shapes(op: Op, input_shape: List, output_shape: List):
    """
    Given an Op and input and output shapes obtained from forward pass, fill in the shapes for the op and its input and
    output products.  If these products already have shapes associated with them, check that they match with the given
    shapes; if not, log an error.
    :param op: Current op to fill shape parameter
    :param input_shape: Input shape obtained from forward pass
    :param output_shape: Output shape obtained from forward pass.  A '1' element is prepended to it to fill in the
    missing num_batches index
    """
    op.output_shape = [1] + output_shape
    for inp in op.inputs:
        if not inp.is_parm:
            if inp.shape and inp.shape[1:] != input_shape[1:]:
                logger.error('Mismatch between existing shape %s for product %s and input shape %s for input of op %s',
                             inp.shape,
                             inp,
                             input_shape,
                             op)
            elif not inp.shape:
                inp.shape = input_shape
    if op.output:
        if op.output.shape and op.output.shape[1:] != output_shape[1:]:
            logger.error('Mismatch between existing shape %s for product %s and output shape %s for output of op %s',
                         op.output.shape,
                         output_shape,
                         input_shape,
                         op)
        elif not op.output.shape:
            op.output.shape = op.output_shape


def _fill_conv_op_info(op: Op, module: torch.nn.Module):
    """ Fill in groups info """

    if op.type in 'convolution':
        op.groups = module.groups


def _get_operation_and_return_line_indices(lines: List[str]) -> Tuple[int, int, int]:
    """
    Torch jit trace code is split into 4 sections: forward function signature, parameter getting, operations, and return
    statements.
    Each section can be identified by certain attributes.
    Identify the line indices marking the start of the parameter getting, operations, and return statements sections.
    :param lines: Lines of torch jit trace code.
    :return: A tuple of three integers corresponding to starting indices of the parameter getting, operations and return
    statements sections.
    """
    curr_index = 0
    parameter_index = 0
    operation_index = 0
    return_index = 0
    for line in lines:
        # Forward function signature lines have no '=' signs, so the first line to have one is the start of the
        # parameter getting section.
        if '=' in line and not parameter_index:
            parameter_index = curr_index

        # Parameter getting section only contains parentheses if it uses the 'getattr()' function.  Operation lines
        # don't use the getattr() function, so the first line with parentheses and no getattr() function is the start
        # of the operation section,
        if '(' in line and 'getattr' not in line and 'forward' not in line and not operation_index:
            operation_index = curr_index

        # Return sections lines all have 'return' within.
        if 'return' in line:
            return_index = curr_index
            break
        curr_index += 1
    return parameter_index, operation_index, return_index


def _split_inputs(inputs: str, named_groups: dict) -> List[str]:
    """
    Identify and split inputs for torch operations.
    This assumes that no quotes, apostrophes, or braces show up in the operation string.
    Currently, items in lists that are not within functions will be broken up into separate inputs.
    :param inputs: String of inputs to a torch operation.
    :param named_groups: Dictionary of names of grouped inputs.  If the name of a group shows up in the inputs list,
    Map it to the group in this dictionary and treat each member of the group as a separate input.
    :return: List of strings representing each input to the operation.
    """
    replaced_str = inputs
    parentheses_depth = 0
    bracket_depth = 0

    # Step through each character in the line and see if it is a comma which separates inputs.
    # Keep counters for parentheses, brackets, and braces to only identify commas outside of any such structures.
    for idx, inp in enumerate(inputs):
        curr_char = inp
        # Currently, individual elements in lists are each treated as a separate parameter.  See if we need to keep the
        # lists intact as well (check bracket depth as well)
        if curr_char == ',' and parentheses_depth == 0:
            # Replace comma with '$' for easy splitting later.  Assumes '$' does not normally show up in the code.
            replaced_str = replaced_str[:idx] + '$' + replaced_str[idx+1:]
        elif curr_char == '(':
            parentheses_depth += 1
        elif curr_char == '[':
            bracket_depth += 1
        elif curr_char == ')':
            parentheses_depth -= 1
        elif curr_char == ']':
            bracket_depth -= 1

        # We should never have seen more right parentheses, brackets, or braces than left ones.
        assert parentheses_depth >= 0 and bracket_depth >= 0

    # Split the string using the '$' characters replaced earlier.
    # Also remove leading and trailing list brackets
    split = replaced_str.split('$')
    split = [s.strip('[] ') for s in split]

    # Items in split may be a named group.  If so, replace the item with each item in the group.
    # Call split inputs recursively on the group in case it contains named groups within as well.
    split_inputs = []
    for inp in split:
        if inp in named_groups.keys():
            group_items = _split_inputs(named_groups[inp], named_groups)
            for item in group_items:
                split_inputs.append(item)
        else:
            split_inputs.append(inp)
    return split_inputs
