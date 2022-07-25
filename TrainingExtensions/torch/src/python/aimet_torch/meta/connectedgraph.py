#!/usr/bin/env python3.6

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

"""For constructing a uniform representation of the computational graph for a PyTorch model,
that is easy to navigate and stores information for the purpose of winnowing.
The representation graph consists of nodes that are either 'operation' or 'product';
operations represent a module or a function that generates a tensor, while products represent
the tensors that are either input to the model (input, constant or parameter) or the
result of an operation. Furthermore the graph representation is bi-directional."""

from typing import Tuple, Union, List, Dict, Type
import torch

from aimet_common.connected_graph.connectedgraph import ConnectedGraph as AimetCommonConnectedGraph
from aimet_common.connected_graph.product import Product
from aimet_common.connected_graph.operation import determine_preceding_op_input_product_index_in_multi_input_op
from aimet_common.model_module import PytorchModelModule
from aimet_common.utils import AimetLogger
from aimet_torch.meta.operation import Op
from aimet_torch.utils import is_leaf_module, run_hook_for_layers_with_given_input, in_eval_mode,\
    is_torch_nn_leaf_module, is_custom_leaf_module
from aimet_torch import onnx_utils

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)

# Check trace parameter for torch jit trace
check_trace = False


# pylint: disable=too-many-lines
# pylint: disable=protected-access
class IrNode:
    """
    Representation for a module in torch graph.
    """
    def __init__(self, node_type: str, inputs: List[Union[List, torch._C.TensorType]],
                 outputs: List[Union[List, torch._C.TensorType]], module: Union[torch.nn.Module, None],
                 residing_module: Union[torch.nn.Module, None] = None):
        self.node_type = node_type
        self.inputs = inputs
        self.outputs = outputs
        self.module = module
        self.residing_module = residing_module

    def __str__(self):
        return self.node_type


class GetAttrNodeInfo:
    """
    Holds information for GetAttr node types.
    """
    def __init__(self, node_alias: str, node_name: str, node_input: Union[None, str]):
        """
        Ex. GetAttr node: %1 : ... prim::GetAttr[name="model"](%self.1)
        node_alias: %1, node_name: model, node_input: %self.1
        :param node_alias: String representing an alias for this node (used as input in other nodes)
        :param node_name: Name of this node as it is referred to in trace attributes
        :param node_input: Inputs to this node. Will be node aliases of other nodes. If this node is a direct descendant
            of the current trace graph being parsed, node_input will point to the current module. Otherwise, it points
            to the direct parent of this node in the trace (following parents upwards will eventually lead to the
            current module).
        """
        self.node_alias = node_alias
        self.node_name = node_name
        self.node_input = node_input


ConnectionsToIrDictType = Dict[torch._C.TensorType, List[Union[IrNode, List[IrNode]]]]
ModuleTensorShapeMapType = Dict[torch.nn.Module, Tuple[List[Union[List, torch.Size]], List[Union[List, torch.Size]]]]


class ConnectedGraph(AimetCommonConnectedGraph):
    """
    For construction of a graph that connects operations together (
        either module or functional) as producers and consumers of tensors.
        Note that the graph has two kinds of nodes: operations and products."""

    def __init__(self, model: torch.nn.Module, model_input: Union[torch.Tensor, Tuple]):
        """
        Init function for connected graph
        :param model: Pytorch model to create connected graph from
        :param model_input: Example input to model.  Can be a single tensor or a list/tuple of input tensors
        """
        super().__init__()
        self._model_name = type(model).__name__
        # Maps pytorch module names to modules
        self._name_to_module = {}
        # Maps pytorch modules to module names
        self._module_to_name = {}

        self._split_count = 0  # Use it in the name of split Ops getting added to the connected graph.

        # List of ops in the order they are traversed using the forward function
        self.ordered_ops = []

        self._generate_module_lookup_table(model)
        with in_eval_mode(model), torch.no_grad():
            self._construct_graph(model, model_input)

        # Maps pytorch modules to connected graph ops
        self._module_to_op_dict = _create_module_to_op_dict(self.ordered_ops)

    # List of op types which can remain as functional in pytorch model definition without affecting the outcome of
    # AIMET features.
    functional_ops = {
        'size',
        'NumToTensor',
        'view',
        'add',
        'sub',
        'mul',
        'div',
        'narrow',
        'reshape',
        'mean',
        'index_select',
        'slice',
        'select',
        'unsqueeze',
        'randn',
        'flatten',
        'Split'
    }

    # Graph nodes for which which we will treat as passthrough and not represent with an Op
    passthrough_graph_nodes = [
        "Int",       # aten::Int
        "t",         # aten::t
        "to",        # aten::to
        "detach",    # aten::detach
        "values"     # aten::values
    ]

    # Input graph nodes to ignore
    input_graph_nodes_to_ignore = [
        "Constant",     # prim::Constant
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

    def get_all_nodes(self, module: torch.nn.Module,
                      module_to_jit_trace: Dict[torch.nn.Module, torch.jit.TracedModule]) -> List[torch._C.Node]:
        """
        Given PyTorch module, Find all the valid aten nodes in forward pass for given trace of model or submodule.

        :param module: PyTorch module.
        :param module_to_jit_trace: Dictionary mapping torch modules to their traces
        :return: List of trace graph nodes if node.kind() starts with "aten::".
        """
        try:
            trace = module_to_jit_trace[module]
        except:
            raise KeyError(f"Couldn't find corresponding JIT trace for module : {module}")

        nodes = self._find_aten_nodes_in_forward_pass(trace)
        return nodes

    def _generate_module_lookup_table(self, model: torch.nn.Module):
        """
        Generates a look up dictionary for getting modules from their names.
        :param model: Pytorch model
        """
        for name, module in model.named_modules(prefix=self._model_name):
            self._name_to_module[name] = module
            self._module_to_name[module] = name

    @staticmethod
    def _generate_module_tensor_shapes_lookup_table(model: torch.nn.Module,
                                                    model_input: Union[torch.Tensor, Tuple]) -> \
            ModuleTensorShapeMapType:
        """
        Generates a look up dictionary for getting module input and output shapes from the module.
        :param model: Pytorch model
        :param model_input: Input to run through forward pass
        :return: Map of modules and input and output tensor shapes obtained from a forward pass
        """
        module_tensor_shapes_map = {}

        def forward_hook(curr_module: torch.nn.Module,
                         inputs: Union[torch.Tensor, List, Dict, None],
                         outputs: Union[torch.Tensor, List, Dict, None]):
            """
            Custom forward hook function to add every module to module-to-tensor shapes dict.
            :param curr_module: Current module being traversed during forward pass.
            :param inputs: tuple of input tensors to the current module
            :param outputs: tuple of output tensors of the current module
            """
            input_shapes = _get_module_tensor_shapes_entry(inputs)
            output_shapes = _get_module_tensor_shapes_entry(outputs)
            if not isinstance(input_shapes, List):
                input_shapes = [input_shapes]
            if not isinstance(output_shapes, List):
                output_shapes = [output_shapes]
            module_tensor_shapes_map[curr_module] = (input_shapes, output_shapes)

        run_hook_for_layers_with_given_input(model, model_input, forward_hook, leaf_node_only=False)

        return module_tensor_shapes_map

    def _construct_graph(self, model: torch.nn.Module, model_input: Union[torch.Tensor, Tuple]):
        """
        Construct connected graph from model and example inputs.
        :param model: Pytorch model to create connected graph from
        :param model_input: Example input to model.  Can be a single tensor or a list/tuple of input tensors
        """
        module_tensor_shapes_map = ConnectedGraph._generate_module_tensor_shapes_lookup_table(model, model_input)
        trace = torch.jit.trace(model, model_input, check_trace=check_trace)
        ir_nodes_list, output_map = self._parse_top_level_trace(trace, model)
        self._construct_ops_and_products(ir_nodes_list,
                                         module_tensor_shapes_map,
                                         self.passthrough_graph_nodes,
                                         self.input_graph_nodes_to_ignore,
                                         output_map)

        # Create parameters for ops such as conv, batchnorm, etc.
        self._fill_op_params()

        # For each split in the model, insert a corresponding split Op in the connected graph.
        ops_list = [op for op in self._ops.values()]
        for op in ops_list:
            self._determine_split_behavior_for_op_and_insert_split_op_in_connected_graph(op)

    def _parse_top_level_trace(self, trace: Union[torch.jit.TopLevelTracedModule, torch.jit.TracedModule],
                               model: torch.nn.Module) -> Tuple[List[IrNode], Dict[torch._C.TensorType,
                                                                                   torch._C.TensorType]]:
        """
        Parse the top level trace and return a list of IrNodes and a dictionary mapping equivalent output tensors found
        during trace parsing.
        :param trace: Pytorch JIT trace for model or a submodule
        :param model: Pytorch model to create connected graph from
        :return: Tuple containing a list of IrNodes and a dictionary mapping equivalent output tensors found during
        trace parsing.
        """
        ir_nodes_list = []
        output_map = {}
        module_to_jit_trace = self._generate_trace_lookup_table(model, trace)
        _ = self._parse_trace_graph(trace, model, ir_nodes_list=ir_nodes_list, output_map=output_map,
                                    module_to_jit_trace=module_to_jit_trace)
        return ir_nodes_list, output_map

    def _parse_trace_graph(self, # pylint: disable=too-many-locals
                           trace: Union[torch.jit.TopLevelTracedModule, torch.jit.TracedModule],
                           model: torch.nn.Module,
                           ir_nodes_list: List[IrNode],
                           output_map: Dict[torch._C.TensorType, torch._C.TensorType],
                           module_to_jit_trace: Dict[torch.nn.Module, torch.jit.TracedModule],
                           higher_level_inputs: Union[List, None] = None,
                           inputs_map: Union[Dict, None] = None) -> List[torch._C.TensorType]:
        """
        Implements a depth-first graph extraction to obtain connectivity information in the form of an IrNodes list.
        Depth-first extraction is realized using recursion.

        :param trace: Pytorch JIT trace for model or a submodule
        :param model: Pytorch model to create connected graph from
        :param ir_nodes_list: List of IrNodes created from traversing the trace graph
        :param output_map: Dictionary mapping high recursion level outputs to lower level equivalent outputs
        :param module_to_jit_trace: Dictionary mapping torch modules to their traces
        :param higher_level_inputs: Corresponding inputs from a higher graph level
        :param inputs_map: Dictionary mapping low recursion level inputs to higher level equivalent inputs
        :return: the outputs of the traced module
        """
        # Create a single IrNode if the module is leaf module and the trace has only one node in forward pass.
        # These checks are needed because even though the module is leaf module but forward pass
        # might have more than one node. In that case, we can't handle the module as a single IrNode and further
        # parsing is needed.
        if not self._is_recursive_parsing_needed(model, module_to_jit_trace):
            return self._parse_single_module_model(model, trace.graph, ir_nodes_list)
        if inputs_map is None:
            inputs_map = {}
        curr_inputs = [inp for inp in trace.graph.inputs()]
        if higher_level_inputs is not None:
            _update_inputs_map(curr_inputs, higher_level_inputs, inputs_map)

        # A map of sub-graph models and node name that requires recursive parsing
        node_name_to_subgraph_model = {}
        # modules that are being referenced within the sub-graph
        node_name_to_module = {curr_inputs[0].debugName(): model}
        for node in trace.graph.nodes():
            outputs = [output for output in node.outputs()]

            # retrieving a module reference
            if 'GetAttr' in node.kind():
                # For GetAttr lines, the output name will be referring to the module, and not the module's output(s)
                assert len(outputs) == 1
                getattr_node_info = ConnectedGraph._get_getattr_node_info(node)
                if getattr_node_info.node_input == curr_inputs[0].debugName():
                    getattr_node_info.node_input = None

                subgraph_model = ConnectedGraph._get_module_instance(node, node_name_to_module)
                if isinstance(subgraph_model, torch.Tensor):
                    continue
                if getattr_node_info.node_alias not in node_name_to_module:
                    node_name_to_module[getattr_node_info.node_alias] = subgraph_model
                else:
                    raise ValueError("duplicate model for {0} -> {1} and {2}".format(
                        getattr_node_info.node_alias, node_name_to_module[getattr_node_info.node_alias],
                        subgraph_model))

                # Recursive parsing is not needed 1) if the module is leaf module and
                # module is from torch.nn (Conv2d, Linear etc.) 2) if the module is leaf module and
                # custom module whose forward method has only one functional operation (elementwise_ops.Add()).
                # Recursive parsing is needed 1) if the module is not leaf module.
                # 2) If the module is leaf module but has multiple functional operations in
                # forward method.
                if self._is_recursive_parsing_needed(subgraph_model, module_to_jit_trace):
                    node_name_to_subgraph_model[getattr_node_info.node_alias] = (subgraph_model, getattr_node_info)

            # invoking forward method
            elif 'CallMethod' in node.kind():
                self.parse_callmethod_node(node, trace, node_name_to_module, node_name_to_subgraph_model,
                                           ir_nodes_list, inputs_map, output_map, model, module_to_jit_trace)

            # functional operations e.g. cat, size etc
            else:
                ir_nodes_list.append(_create_functional_ir_node(node, inputs_map, residing_module=model))

        # return output connections
        return [output for output in trace.graph.return_node().inputs()]

    @staticmethod
    def _parse_op_type(node: torch._C.Node) -> str:
        """
        Helper method to extract op type from node info
        :param node: trace graph node
        :return: Op Type string
        """
        # extracting Op type from node.kind string e.g. aten::relu_, aten::size etc
        op_type = node.kind().split("::")[-1].lstrip('_').rstrip('_')
        return op_type

    # pylint: disable=too-many-arguments
    def parse_callmethod_node(self, node: torch._C.Node,
                              trace: Union[torch.jit.TopLevelTracedModule, torch.jit.TracedModule],
                              node_name_to_module: Dict[str, torch.nn.Module],
                              node_name_to_subgraph_model: Dict[str, Tuple[torch.jit.TracedModule, torch._C.Node]],
                              ir_nodes_list: List[IrNode],
                              inputs_map: Dict[torch._C.TensorType, torch._C.TensorType],
                              output_map: Dict[torch._C.TensorType, torch._C.TensorType],
                              residing_module: torch.nn.Module,
                              module_to_jit_trace: Dict[torch.nn.Module, torch.jit.TracedModule]):
        # pylint: disable=too-many-locals
        """
        The call method node signifies invocation of the forward method, this method extracts an IrNode representation
        of the module. Typically the node has the following construct:
            %output_N : Tensor = prim::CallMethod[name="forward"](%output_L, %output_M)
        :param node: trace graph node i.e. 'CallMethod' node
        :param trace: trace of model or submodule
        :param node_name_to_module: dictionary of module indexed by output_name referenced in the sub-graph
        :param node_name_to_subgraph_model: dictionary of torch graph nodes index of output_name that have not been
            resolved
        :param ir_nodes_list: List of IrNodes created from traversing the trace graph
        :param inputs_map: Dictionary mapping low recursion level inputs to higher level equivalent inputs
        :param output_map: Dictionary mapping high recursion level outputs to lower level equivalent outputs
        :param residing_module: Torch module in which the current node is situated
        :param module_to_jit_trace: Dictionary mapping torch modules to their traces
        """
        inputs = [inp for inp in node.inputs()]
        # 1st input is a reference on which the call method is being invoked.
        input_name: str = inputs[0].debugName()
        outputs = [output for output in node.outputs()]
        if input_name in node_name_to_subgraph_model:
            subgraph_model, getattr_node_info = node_name_to_subgraph_model[input_name]
            trace_levels = [getattr_node_info.node_name]
            # If node_input (input to the current GetAttr node) is None, we are at the topmost level, and can call
            # trace.<current node name> to get the trace for the subgraph. Otherwise, compile a list of node names to
            # call into by following the subgraph_input entries up the chain.
            while getattr_node_info.node_input is not None:
                _, getattr_node_info = node_name_to_subgraph_model[getattr_node_info.node_input]
                # Later on trace levels will be processed with most recent level first, so insert each one at the front.
                trace_levels.insert(0, getattr_node_info.node_name)
            subgraph_trace = trace
            for level in trace_levels:
                subgraph_trace = getattr(subgraph_trace, level)

            submodule_outputs = self._parse_trace_graph(subgraph_trace, subgraph_model, ir_nodes_list,
                                                        higher_level_inputs=inputs[1:], inputs_map=inputs_map,
                                                        output_map=output_map, module_to_jit_trace=module_to_jit_trace)
            # Current node is a subgraph. The outputs of the subgraph at this level will correspond to the outputs of
            # the subgraph in the inner recursion level. Update output_map to contain this new mapping.
            assert len(submodule_outputs) == len(outputs)
            for idx, output in enumerate(outputs):
                if submodule_outputs[idx] in output_map:
                    output_map[output] = output_map[submodule_outputs[idx]]
                else:
                    output_map[output] = submodule_outputs[idx]

                # Go through each ir_node and examine the outputs. For any output that was returned from an inner
                # recursion level, replace it with the corresponding output from this level.
                for ir_node in ir_nodes_list:
                    for ir_node_idx, ir_node_output in enumerate(ir_node.outputs):
                        if ir_node_output == submodule_outputs[idx]:
                            ir_node.outputs[ir_node_idx] = outputs[idx]

        elif input_name in node_name_to_module and is_leaf_module(node_name_to_module[input_name]):
            # the graph is fully represented by a directional graph of leaf torch modules so the recursion is
            # stopped at this level. torch.nn.Identity are being ignored because in graph node representation
            # the torch.nn.Identity op generate no output and are not part of inputs for downstream op
            if not isinstance(node_name_to_module[input_name], torch.nn.Identity):
                op_type = self.get_op_type(type(node_name_to_module[input_name]))
                node_inputs = [inp for inp in node.inputs()]
                ir_node = IrNode(node_type=op_type,
                                 inputs=[inputs_map.get(inp, inp) for inp in node_inputs[1:]],
                                 outputs=outputs,
                                 module=node_name_to_module[input_name],
                                 residing_module=residing_module)
                ir_nodes_list.append(ir_node)

    def _parse_single_module_model(self, module: torch.nn.Module,
                                   graph: torch._C.Graph,
                                   ir_nodes_list: List[IrNode]) -> List[torch._C.TensorType]:
        """
        Create a node for the single module model.
        :param module:  Pytorch model composed on single module
        :param graph: trace graph representing the model
        :param ir_nodes_list: List of IrNodes created from traversing the trace graph
        :return: Node created for the module
        """
        op_type = self.get_op_type(type(module))
        node_inputs = [inp for inp in graph.inputs()]
        outputs = [output for output in graph.return_node().inputs()]
        ir_node = IrNode(node_type=op_type,
                         inputs=node_inputs[1:],
                         outputs=[output for output in graph.outputs()],
                         module=module)
        ir_nodes_list.append(ir_node)
        return outputs

    @staticmethod
    def _get_attribute_name(node: torch._C.Node) -> Dict[str, str]:
        """
        Retrieve the attributes associated with the graph node
        :param node: trace graph node
        :return: a dictionary of attributes associated with the node
        """
        attributes = {}
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
                             node_name_to_module: Dict[str, torch.nn.Module]) -> torch.nn.Module:
        """
        Get the torch.nn.Module referenced by the node.
        :param node: trace graph node
        :param node_name_to_module: dictionary of module index by output_name referenced in the sub-graph
        :return: torch module corresponding to the node
        """
        input_name: str = node.input().debugName()
        attributes = ConnectedGraph._get_attribute_name(node)
        model = node_name_to_module[input_name]
        sub_model = getattr(model, attributes['name'])
        return sub_model

    @staticmethod
    def _get_getattr_node_info(node: torch._C.Node) -> GetAttrNodeInfo:
        """
        Obtain node_alias, node_name, and node_input for the given node.
        :param node: GetAttr node to obtain information for
        :return: GetAttrNodeInfo object containing node information
        """
        # Obtain the input node to the GetAttr node. This will usually refer to the current level module.
        # However, in certain cases such as Sequentials, the index level will also show up as a separate GetAttr
        # node.
        # Ex.
        # %1 : ... prim::GetAttr[name="model"](%self.1)
        # %2 : ... prim::GetAttr[name="_layer0"](%1)
        # Here, to call into %2 from the current trace, we must call .model._layer0. Tracking inputs to
        # the GetAttr nodes tells us this path (%2 comes from %1 which comes from %self.1, the current module)
        node_input = [inp for inp in node.inputs()][0].debugName()
        node_alias = [output for output in node.outputs()][0].debugName()
        node_name = ConnectedGraph._get_attribute_name(node).get('name')
        return GetAttrNodeInfo(node_alias, node_name, node_input)

    @staticmethod
    def get_op_type(model_cls: Type[torch.nn.Module]) -> str:
        """
        Get connected graph op type for a pytorch module
        :param model_cls: Pytorch module class to get op type for
        :return: Connected graph op type
        """
        # use nominal Op type if its a known type else use torch defined Module name
        if model_cls in onnx_utils.map_torch_types_to_onnx:
            # Currently always taking first element in the list. Check whether we need extra logic for using other
            # elements in the list.
            op_type = onnx_utils.map_torch_types_to_onnx[model_cls][0]
        else:
            op_type = model_cls.__name__
            logger.debug("unknown op_type -- defaulting to class name %s", op_type)
        return op_type

    def _construct_ops_and_products(self,
                                    ir_nodes_list: List[IrNode],
                                    module_tensor_shapes_map: ModuleTensorShapeMapType,
                                    passthrough_types: List[str],
                                    input_types_to_ignore: List[str],
                                    output_map: Dict[torch._C.TensorType, torch._C.TensorType]):
        """
        Create ops and products from nodes and connections.
        :param ir_nodes_list: List of ir_nodes to create ops for
        :param module_tensor_shapes_map: Dictionary mapping modules to input and output tensor shapes obtained from a
            forward pass
        :param passthrough_types: IrNode types to treat as passthrough (ops will not be created for these ir_nodes)
        :param input_types_to_ignore: Input node types to ignore (do not create products for output connections from
            these ir_nodes)
        :param output_map: Mapping between higher level connections with corresponding lower level recursive
            connections. Only the base level connections carry shape information.
        """
        self._handle_ir_nodes_of_interest(ir_nodes_list, passthrough_types, input_types_to_ignore)
        filtered_ir_nodes = [ir_node for ir_node in ir_nodes_list if ir_node.node_type not in ['TupleConstruct',
                                                                                               'TupleUnpack',
                                                                                               'ListUnpack',
                                                                                               'ListConstruct']
                             and ir_node.node_type not in input_types_to_ignore
                             and ir_node.node_type not in passthrough_types]
        connections_to_nodes_dict = self._create_connections_to_ir_nodes_dict(filtered_ir_nodes)
        ir_node_to_op_dict = self._create_ops_from_ir_nodes_list(filtered_ir_nodes, module_tensor_shapes_map)
        self._create_products_from_connections(connections_to_nodes_dict, ir_node_to_op_dict, output_map,
                                               input_types_to_ignore)

    def _create_ops_from_ir_nodes_list(self, ir_nodes_list: List[IrNode],
                                       module_tensor_shapes_map: ModuleTensorShapeMapType)\
            -> Dict[IrNode, Op]:
        """
        Given a list of nodes, create ops for each one.
        :param ir_nodes_list: List of nodes to create ops for
        :param module_tensor_shapes_map: Dictionary mapping modules to input and output tensor shapes obtained from a
            forward pass
        :return: Dictionary mapping nodes to corresponding ops that were created
        """
        node_to_op_dict = {}
        for idx, node in enumerate(ir_nodes_list):
            op_name = node.node_type + '_' + str(idx)
            op = Op(name=op_name, dotted_name=op_name, output_shape=None, is_anonymous=node.module is None,
                    op_type=node.node_type, residing_module=node.residing_module)
            if node.module is not None:
                op.model_module = PytorchModelModule(node.module)
                if node.module in module_tensor_shapes_map:
                    _, output_tensor_shapes = module_tensor_shapes_map[node.module]
                    # Temporarily not handling dict types for output tensors.
                    if isinstance(output_tensor_shapes, Dict):
                        output_tensor_shapes = None
                    # For now, treat only the first output shape as the shape of the node if there is more than one
                    # entry
                    flattened_shapes = _flatten_lists(output_tensor_shapes)
                    op.output_shape = flattened_shapes[0]
                op.dotted_name = self._module_to_name[node.module]
                _fill_groups_info(op, node.module)
            node_to_op_dict[node] = op
            self._ops[op_name] = op
            self.ordered_ops.append(op)
        return node_to_op_dict

    def _create_products_from_connections(self, connections_to_nodes_dict: ConnectionsToIrDictType,
                                          node_to_op_dict: Dict[IrNode, Op],
                                          output_map: Dict[torch._C.TensorType, torch._C.TensorType],
                                          input_types_to_ignore: List[str]):
        """
        Given connections in a dictionary, create products for each connection if it is not one to ignore.
        :param connections_to_nodes_dict: Dictionary mapping connections to input IrNode and output IrNodes
        :param node_to_op_dict: Dictionary mapping nodes to corresponding ops
        :param output_map: Mapping between higher level connections with corresponding lower level recursive
            connections. Only the base level connections carry shape information.
        :param input_types_to_ignore: Input node types to ignore (do not create products for output connections from
            these nodes)
        """
        for connection, (producer, consumer_list) in connections_to_nodes_dict.items():
            if producer is None:
                # Case of input connection to model
                assert consumer_list
                for consumer in consumer_list:
                    consumer_op = node_to_op_dict[consumer]
                    product_name = 'input_to_' + consumer_op.name
                    if product_name not in self._products:
                        product = _create_product_from_connection(product_name, connection, output_map)
                        product.is_model_input = True
                        product.add_consumer(consumer_op)
                        consumer_op.add_input(product)
                        self._products[product_name] = product
            elif producer.node_type in input_types_to_ignore:
                continue
            else:
                producer_op = node_to_op_dict[producer]
                for consumer in consumer_list:
                    consumer_op = node_to_op_dict[consumer]
                    product_name = producer_op.name + '_to_' + consumer_op.name
                    if product_name not in self._products:
                        product = _create_product_from_connection(product_name, connection, output_map)
                        product.producer = producer_op
                        self._products[product_name] = product
                    else:
                        product = self._products[product_name]
                    # If an op has multiple output products, only the first one will be listed as the op's output here
                    if producer_op.output is None:
                        _update_op_output_with_product(producer_op, product)
                    product.add_consumer(consumer_op)
                    consumer_op.add_input(product)

    def _fill_op_params(self):
        """
        For certain ops like convolution, batch norm, and linear, create products for their parameters if they don't
        exist yet.
        """
        for op in self._ops.values():
            module = op.get_module()
            if module is not None:
                name = self._module_to_name.get(module, None)
                if op.type in ['Conv', 'ConvTranspose', 'BatchNormalization', 'Gemm']:
                    if module.weight is not None:
                        product_name = name + '.weight'
                        self._create_and_add_param_product_if_not_exists(op, product_name, list(module.weight.shape))
                    if module.bias is not None:
                        product_name = name + '.bias'
                        self._create_and_add_param_product_if_not_exists(op, product_name, list(module.bias.shape))
                if op.type == 'BatchNormalization':
                    # If batch_norm, fill in rest of bn params
                    if module.running_mean is not None:
                        product_name = name + '.running_mean'
                        self._create_and_add_param_product_if_not_exists(op, product_name,
                                                                         list(module.running_mean.shape))
                    if module.running_var is not None:
                        product_name = name + '.running_var'
                        self._create_and_add_param_product_if_not_exists(op, product_name,
                                                                         list(module.running_var.shape))

    def _create_and_add_param_product_if_not_exists(self, op: Op, product_name: str, shape: List[int]):
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
        split_op = Op(name=split_name, dotted_name=split_dotted_name, output_shape=op.output_shape,
                      is_anonymous=is_anonymous, op_type='Split', residing_module=None)
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
            # Need to insert the newly created split_op product in the correct input index of the op
            logger.debug("Insert Split Op: Step 2b. Op has multiple input products: %s", a_consumer.inputs)
            input_product_index = determine_preceding_op_input_product_index_in_multi_input_op(preceding_op,
                                                                                               a_consumer)
            a_consumer.inputs[input_product_index] = split_op_product
            logger.debug("Insert Split Op: Step 2c. For product: %s, split_op input_product_index: %s",
                         split_op_product.name, input_product_index)
            consumer_index += 1

    def _create_connections_to_ir_nodes_dict(self, ir_nodes_list: List[IrNode]) -> ConnectionsToIrDictType:
        """
        Create a mapping from connections found in torch graph to input and output IrNodes. Each connection will have one
        input and zero or more outputs IrNodes.
        :param ir_nodes_list: List of IrNodes to extract connections information from
        :return: Dictionary mapping connections to input ir_node and output ir_nodes
        """
        connections_to_ir_nodes_dict = {}
        for ir_node in ir_nodes_list:
            node_inputs = _flatten_lists(ir_node.inputs)
            node_outputs = _flatten_lists(ir_node.outputs)
            for inp in node_inputs:
                if inp in connections_to_ir_nodes_dict:
                    connections_to_ir_nodes_dict[inp][1].append(ir_node)
                else:
                    connections_to_ir_nodes_dict[inp] = [None, [ir_node]]
            for output in node_outputs:
                if output in connections_to_ir_nodes_dict:
                    if connections_to_ir_nodes_dict[output][0] is not None:
                        inp_op_name = None
                        if connections_to_ir_nodes_dict[output][0].module is not None:
                            inp_op_name = self._module_to_name[connections_to_ir_nodes_dict[output][0].module]
                        error_msg = (f'Input of {output} with name {inp_op_name} already exists. Ensure that no '
                                     f'modules are being reused in the model.')
                        logger.error(error_msg)
                        raise AssertionError(error_msg)
                    connections_to_ir_nodes_dict[output][0] = ir_node
                else:
                    connections_to_ir_nodes_dict[output] = [ir_node, []]
        return connections_to_ir_nodes_dict

    def _handle_ir_nodes_of_interest(self, ir_nodes_list: List[IrNode], passthrough_ops: List[str],
                                     input_ops_to_ignore: List[str]):
        """
        Update input and output connections of certain ir_nodes in ir_nodes_list (Tuple/ListConstructs, passthrough,
        input ops)
        :param ir_nodes_list: List of ir_nodes to update connections for
        :param passthrough_ops: List of op types to treat as passthrough
        :param input_ops_to_ignore: List of input op types to ignore
        """
        connections_to_ir_nodes_dict = self._create_connections_to_ir_nodes_dict(ir_nodes_list)
        for ir_node in ir_nodes_list:
            if ir_node.node_type in ['TupleConstruct', 'ListConstruct']:
                _handle_tuple_and_list_construct_ir_node(ir_node, connections_to_ir_nodes_dict)
            elif ir_node.node_type in ['TupleUnpack', 'ListUnpack']:
                _handle_tuple_unpack_ir_node(ir_node, connections_to_ir_nodes_dict)
            elif ir_node.node_type in passthrough_ops:
                _handle_passthrough_ir_node(ir_node, connections_to_ir_nodes_dict)
            elif ir_node.node_type in input_ops_to_ignore:
                _handle_input_ir_node_to_ignore(ir_node, connections_to_ir_nodes_dict)

    def _is_recursive_parsing_needed(self, module: torch.nn.Module,
                                     module_to_jit_trace: Dict[torch.nn.Module, torch.jit.TracedModule]) -> bool:
        """
        Utility to decide whether recursive parsing is needed for given module and it's jit trace.
        Recursive parsing is not needed
        1) if the module is leaf module and from torch.nn class (nn.Conv2d, nn.ReLU, nn.rNN etc.)
        2) if the module is leaf module and has only one aten node inside forward method (elementwise_ops.Add etc.)

        :param module: PyTorch module.
        :param module_to_jit_trace: Dictionary mapping torch modules to their traces
        :return: Boolean whether recursive parsing needed or not. If needed returns True, False otherwise.
        """
        recursive_parsing_needed = True
        if is_torch_nn_leaf_module(module) or is_custom_leaf_module(module, self.get_all_nodes(module,
                                                                                               module_to_jit_trace)):
            recursive_parsing_needed = False

        return recursive_parsing_needed

    @staticmethod
    def _generate_trace_lookup_table(model: torch.nn.Module,
                                     trace: Union[torch.jit.TopLevelTracedModule, torch.jit.TracedModule]):
        """
        Generate pytorch module names to corresponding JIT trace dictionary. There will be always one to one
        mapping between pytorch module and corresponding JIT trace.

        :param model: PyTorch model.
        :param trace: PyTorch JIT trace.
        """
        def _add_jit_trace(model: torch.nn.Module,
                           trace: Union[torch.jit.TopLevelTracedModule, torch.jit.TracedModule]):
            """
            Recursively add jit trace for all the modules to dictionary.
            :param model: PyTorch model or submodule.
            :param trace: PyTorch JIT trace of model.
            """
            for name, module in model.named_children():
                sub_trace = getattr(trace, name)
                module_to_jit_trace[module] = sub_trace

                # recursively call children modules.
                if not is_leaf_module(module):
                    _add_jit_trace(module, sub_trace)

        # Add top level model and corresponding JIT trace.
        module_to_jit_trace = {model: trace}
        # Recursively add children modules and corresponding JIT traces.
        _add_jit_trace(model, trace)
        return module_to_jit_trace

    @staticmethod
    def _find_aten_nodes_in_forward_pass(trace: Union[torch.jit.TopLevelTracedModule, torch.jit.TracedModule]) \
            -> List[torch._C.Node]:
        """
        Find all the valid nodes in forward pass for given trace of model or submodule.
        Three possible outcomes:
        1) When the forward method has multiple functional operations and valid nodes are returned.
        2) When the forward method is not defined which causes RuntimeError and empty list is returned.
        3) When the module is unused, empty list is returned.

        :param trace: trace of model or submodule.
        :return: List of trace graph nodes if node.kind() starts with "aten::".
        """
        # pylint: disable=protected-access
        nodes = []
        try:
            nodes = [node for node in trace.graph.nodes() if "aten::" in node.kind() and
                     ConnectedGraph._parse_op_type(node) not in ConnectedGraph.passthrough_graph_nodes]
        except RuntimeError:
            pass
        return nodes


def _create_module_to_op_dict(ops: List[Op]) -> Dict[torch.nn.Module, Op]:
    """
    Utility to create dictionary mapping pytorch modules to connected graph Ops
    :param ops: List of connected graph Ops
    :return: Dictionary mapping pytorch modules to connected graph Ops
    """
    module_to_op_dict = {}
    for op in ops:
        if op.get_module():
            module_to_op_dict[op.get_module()] = op
    return module_to_op_dict


def _fill_groups_info(op: Op, module: torch.nn.Module):
    """
    Fill in groups info for convolution ops. If op is not a conv op, groups is unchanged.
    :param op: Connected graph op to fill groups info for
    :param module: Pytorch module to check for groups
    """

    if op.type in 'Conv':
        op.groups = module.groups


def _handle_tuple_and_list_construct_ir_node(ir_node: IrNode, connections_to_ir_nodes_dict: ConnectionsToIrDictType):
    """
    Update connections of tuple and list construct ir_nodes, as well as connections to children of the ir_node.
    :param ir_node: Tuple or list construct ir_node
    :param connections_to_ir_nodes_dict: Dictionary mapping connections to input ir_node and output ir_nodes
    """
    # Identify the output connection and consumers of that connection
    assert len(ir_node.outputs) == 1
    output = ir_node.outputs[0]
    consumers = connections_to_ir_nodes_dict[output][1]

    # For each consumer, update their inputs by replacing the connection from Tuple/ListConstruct to the inputs
    # of Tuple/ListConstruct instead
    for consumer in consumers:
        connection_index = consumer.inputs.index(output)
        consumer.inputs[connection_index] = ir_node.inputs

    # Replace the output connection of Tuple/ListConstruct with its input connections
    ir_node.outputs[0] = ir_node.inputs
    del connections_to_ir_nodes_dict[output]


def _handle_tuple_unpack_ir_node(ir_node: IrNode, connections_to_ir_nodes_dict: ConnectionsToIrDictType):
    """
    Update connections of tuple unpack nodes, as well as connections to children of the node.
    :param ir_node: Tuple or list construct node
    :param connections_to_ir_nodes_dict: Dictionary mapping connections to input node and output nodes
    """
    # For tuple unpack, either input came from tuple pack, or an op that had multiple outputs
    assert len(ir_node.inputs) == 1
    tuple_unpack_input = ir_node.inputs[0]
    if isinstance(tuple_unpack_input, List):
        # Case where input came from tuple pack
        # Length of unpacked output should be same as length of input to tuple pack
        assert len(tuple_unpack_input) == len(ir_node.outputs)
        # For each output, find consumers of the output. For the input connection to the consumer that matches
        # the output, replace it with the corresponding input connection to tuple_unpack.
        # Also replace the output of tuple unpack with the corresponding input connection.
        for idx, output in enumerate(ir_node.outputs):
            consumers = connections_to_ir_nodes_dict[output][1]
            for consumer in consumers:
                connection_index = consumer.inputs.index(output)
                consumer.inputs[connection_index] = tuple_unpack_input[idx]
            ir_node.outputs[idx] = tuple_unpack_input[idx]
            del connections_to_ir_nodes_dict[output]
    else:
        # Case where input came from op with multiple outputs
        # Replace the output connection of the op with the outputs of TupleUnpack
        producer = connections_to_ir_nodes_dict[tuple_unpack_input][0]

        # Producer can be None if a ListUnpack follows a values passthrough op for processing dict inputs
        if producer is not None:
            producer.outputs = ir_node.outputs
        ir_node.inputs = ir_node.outputs
        # Remove what used to be the input to the tuple unpack node from connections_to_ir_nodes_dict
        del connections_to_ir_nodes_dict[tuple_unpack_input]


def _handle_passthrough_ir_node(ir_node: IrNode, connections_to_ir_nodes_dict: ConnectionsToIrDictType):
    """
    Update connections of ir_nodes feeding into and following the passthrough ir_node to effectively skip the
    passthrough ir_node.
    :param ir_node: Tuple or list construct ir_node
    :param connections_to_ir_nodes_dict: Dictionary mapping connections to input ir_node and output ir_nodes

    """
    if not ir_node.inputs:
        # This will be the case when input to passthrough node is an input to ignore. In this case, the input connection
        # will have been removed earlier. Treat this node as an input to ignore as well.
        _handle_input_ir_node_to_ignore(ir_node, connections_to_ir_nodes_dict)
    else:
        # For passthrough ops, simply set the input of the consumer ir_nodes of the op's output to be the input of the
        # passthrough op itself.
        # Also clear the inputs and outputs of the passthrough Op to disconnect it from the graph.
        # Below check is not strictly necessary but current logic assumes so. Easy to change in the future if so.
        assert len(ir_node.inputs) == 1
        assert len(ir_node.outputs) == 1
        input_connection = ir_node.inputs[0]
        output_connection = ir_node.outputs[0]
        consumers = connections_to_ir_nodes_dict[output_connection][1]
        for consumer in consumers:
            connection_index = consumer.inputs.index(output_connection)
            consumer.inputs[connection_index] = input_connection
        ir_node.inputs = []
        ir_node.outputs = []
        del connections_to_ir_nodes_dict[output_connection]


def _handle_input_ir_node_to_ignore(ir_node: IrNode, connections_to_ir_nodes_dict: ConnectionsToIrDictType):
    """
    Update the consumers of input ir_nodes to ignore to remove thoe incoming connections.
    :param ir_node: Tuple or list construct ir_node
    :param connections_to_ir_nodes_dict: Dictionary mapping connections to input ir_node and output ir_nodes
    """
    # For input ops like prim::Constant, disconnect it from the graph so it is not represented by ConnectedGraph.
    # For consumers of the input op, remove the input connection corresponding to the input from the input op.
    assert not ir_node.inputs
    for output in ir_node.outputs:
        consumers = connections_to_ir_nodes_dict[output][1]
        for consumer in consumers:
            consumer.inputs.remove(output)
        del connections_to_ir_nodes_dict[output]
    ir_node.outputs = []


def _create_product_from_connection(product_name: str, connection: torch._C.TensorType,
                                    output_map: Dict[torch._C.TensorType, torch._C.TensorType]) -> Product:
    """
    Create connected graph product given a connection.
    :param product_name: Name of product to create
    :param connection: Connection to extract shape information from
    :param output_map: Mapping between higher level connections with corresponding lower level recursive connections.
        Only the base level connections carry shape information.
    :return: Product that was created
    """
    # Create product if it doesn't already exist. There is a limitation here, where two ops with multiple
    # unique connections linking the two will only result in one product being created. (For example, LSTM
    # with output, h, and c feeding into a second LSTM would show 3 unique connections). This behavior
    # matches the existing connected graph behavior, but is an item of possible rework in the future.

    # Get product shape
    shape = None
    if isinstance(connection.type(), torch._C.TensorType):
        shape = connection.type().sizes()
        if shape is None and connection in output_map:
            # Shape may be none if this is an output connection that was replaced from the output of a lower level
            # module. Look up the corresponding lowest level connection in output_map.
            if isinstance(output_map[connection].type(), torch._C.TensorType):
                shape = output_map[connection].type().sizes()

    product = Product(product_name, shape)
    return product


def _update_op_output_with_product(op: Op, product: Product):
    """
    Update the output of an op with the given product if there is not already an associated product. Also reconcile
    shapes of the op and product if possible.
    :param op: Op to update output for
    :param product: Output product of the op
    """
    op.output = product
    if op.output_shape is None:
        # Possible for product's shape to be None here as well. If so, they will both remain None.
        op.output_shape = product.shape
    elif product.shape is None:
        product.shape = op.output_shape
    elif product.shape and op.output_shape != product.shape:
        logger.debug('Mismatch between existing shape %s for product %s and output shape %s for '
                     'output of op %s', product.shape, product.name, op.output_shape,
                     op.name)


def _create_functional_ir_node(node: torch._C.Node, inputs_map: Dict[torch._C.TensorType, torch._C.TensorType],
                               residing_module: torch.nn.Module) -> IrNode:
    """
    Create an IrNode containing input and output connections information given a torch graph node.
    :param node: trace graph node
    :param inputs_map: Mapping
    :param residing_module: Torch module in which the current node is situated
    :return: IrNode created from information in the trace graph node
    """
    outputs = [output for output in node.outputs()]
    op_type = ConnectedGraph._parse_op_type(node)
    # If there is a known mapping to an onnx type for this functional, use the onnx type as the op type
    if op_type in onnx_utils.pytorch_functional_name_to_onnx_dict.keys():
        op_type = onnx_utils.pytorch_functional_name_to_onnx_dict[op_type]
    ir_node = IrNode(node_type=op_type,
                     inputs=[inputs_map.get(inp, inp) for inp in node.inputs()],
                     outputs=outputs,
                     module=None,
                     residing_module=residing_module)
    return ir_node


def _flatten_lists(nested_list: List[List]) -> List:
    """
    Given a list which may contain either an item or a further list of items, generate a list that only contains items
    by bringing all elements of nested lists into the top level list.
    :param nested_list: List containing items or lists of items
    :return: List containing only items (no nested lists)
    """
    flattened_list = []
    for item in nested_list:
        if not isinstance(item, List):
            flattened_list.append(item)
        else:
            flattened_list.extend(_flatten_lists(item))
    return flattened_list


def _update_inputs_map(curr_inputs: List[torch._C.TensorType], higher_level_inputs: List[torch._C.TensorType],
                       inputs_map: Dict[torch._C.TensorType, torch._C.TensorType]):
    """
    Update inputs_map with additional entries mapping higher_level_inputs to inputs of the current graph.
    :param curr_inputs: Inputs to the current graph level
    :param higher_level_inputs: Corresponding inputs from a higher graph level
    :param inputs_map: Dictionary mapping low recursion level inputs to higher level equivalent inputs
    """
    # First index of curr_inputs refers to the current module itself, and not any actual input
    assert len(higher_level_inputs) == len(curr_inputs) - 1
    for idx, inp in enumerate(higher_level_inputs):
        if inp in inputs_map:
            inputs_map[curr_inputs[idx + 1]] = inputs_map[inp]
        else:
            inputs_map[curr_inputs[idx + 1]] = inp


def _get_module_tensor_shapes_entry(tensors: Union[torch.Tensor, List, Dict, None]):
    """
    Given the tensor input or output for a module, extract their shapes and return the shapes in the same list structure
    as the given tensor.
    :param tensors: Input or output tensors for a module
    :return: Shapes of the constituent tensors in tensors
    """
    if tensors is None:
        return None
    if isinstance(tensors, torch.Tensor):
        return tensors.shape
    # Tensors must then be a nested list of tensors, so recursively get the shapes for each item in the list
    if isinstance(tensors, (List, Tuple)):
        return [_get_module_tensor_shapes_entry(entry) for entry in tensors]
    if isinstance(tensors, Dict):
        shapes_dict = {}
        for k, v in tensors.items():
            shapes_dict[k] = _get_module_tensor_shapes_entry(v)
        return shapes_dict
    logger.debug('Unexpected data type for tensor %s. Supported types include tensors, or Lists, Tuples, and Dicts of '
                 'tensors.', type(tensors))
    return None
