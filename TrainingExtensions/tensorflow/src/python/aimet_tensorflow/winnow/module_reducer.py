# /usr/bin/env python3.5
# -*- mode: python -*-
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

""" Contains functionality related to reducing TensorFlow modules.  """

from typing import List, Tuple, Dict
import tensorflow as tf

import aimet_common.winnow.module_reducer
from aimet_common.connected_graph.connectedgraph import get_ordered_ops
import aimet_common.winnow.winnow_utils
from aimet_common.winnow.mask import Mask
from aimet_common.utils import AimetLogger, ModelApi
from aimet_common.winnow.winnow_utils import OpConnectivity, ConnectivityType,\
    get_indices_among_ones_of_overlapping_ones
from aimet_tensorflow import graph_editor
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.common.operation import Op
from aimet_tensorflow.utils.op.fusedbatchnorm import BNUtils
import aimet_tensorflow.winnow.module_reducer_handler as module_reducers


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Winnow)


class ReducedInfo:
    """ Class to hold information about the reduced version of the op if there is one """
    def __init__(self, dotted_name: str, output_op_node: tf.Operation, module: tf.Operation):
        self.dotted_name = dotted_name
        self.output_op_node = output_op_node
        self.module = module


class ModuleReducer(aimet_common.winnow.module_reducer.ModuleReducer):
    """ Class responsible for reducing TensorFlow modules. """

    def __init__(self, conn_graph: ConnectedGraph, sess: tf.compat.v1.Session, using_cuda: bool, reshape: bool,
                 op_to_mask_dict: dict):
        """
        ModuleReducer initialization.
        :param conn_graph: ConnectedGraph associated with the session.
        :param sess: Current session
        :param using_cuda: if True, indicates that the model is on GPU.
        :param reshape: If set to True, add DownSampleLayer/UpSampleLayer  module as needed.
                        If set to False, do not add ownSampleLayer/UpsampleLayer module.
        :param op_to_mask_dict: Dictionary mapping Op to mask
        """

        super().__init__(using_cuda, reshape, op_to_mask_dict)
        self._conn_graph = conn_graph
        self._sess = sess
        self._reduced_modules = {}
        self._reduced_op_info = {}

    def reduce_modules(self, _=None) -> (tf.compat.v1.Session, Dict[str, Tuple[tf.Operation, Mask]]):
        """
        For the Ops in the list,  reduce the corresponding modules.
        Reduce includes reducing the parameter tensors associated with the module
        as well as prepending/appending DownSample/UpSample layer to the module.
        :return: Dictionary mapping original module names to a tuple of (reduced module, op mask)

        """

        ordered_ops = get_ordered_ops(self._conn_graph.starting_ops)

        with self._sess.graph.as_default():
            # Step through the list of ordered ops, and reduce modules on an as needed basis.
            # Since the ops are ordered such that any op's parents will be earlier in the list, we guarantee that we
            # will never reduce a child op before its parent op has been checked for reduction.

            # Boolean to track whether the current op to winnow should be detached from its inputs
            # If the parent op is unwinnowed and this is the first op to winnow, it will be detached.
            # If the op is in the middle of a set of ops to winnow, no need to detach since one of its ancestor
            # ops will already have been detached.
            needs_detach = True
            for op in ordered_ops:
                if op.type == "branch" or op not in self._op_to_mask_dict.keys():
                    # branch input matches the previous op's output, since they actually represent the same thing.
                    # During validation, branch outputs are set equal to branch inputs.
                    # Upshot is that if previous op is winnowed, branch will be winnowed too.  If previous op is not
                    # winnowed, neither will branch op.
                    # An exception is where a concat is followed immediately by a branch.  In this case, concat's mask
                    # could differ from branch's mask due to mask propagation rules.  This case is still handled during
                    # _get_input_tensors() for child ops of the branch.
                    continue
                if not self._op_to_mask_dict[op].are_masks_unchanged():
                    new_op_tensors = self._create_reduced_module(op, needs_detach)
                    needs_detach = self._reroute_if_necessary(op, new_op_tensors)

        return self._sess, self._reduced_modules

    def _create_reduced_module(self, op: Op, needs_detach: bool) -> List[tf.Tensor]:
        """
        Create reduced version of op and update appropriate ConnectedGraph parameters
        :param op: Op to reduce
        :param needs_detach: True if this op's parent was not reduced, making this the top of a string of ops to detach
        from the graph
        :return: List of output tensors of the winnowed op
        """

        input_tensors = self._get_input_tensors_for_winnowed_op(op)
        new_op_tensors = self._create_new_op((op, input_tensors))
        if needs_detach:
            self._detach_op_from_inputs(op)
        return new_op_tensors

    def _create_new_op(self, op_tensor_tuple: Tuple[Op, List[tf.Tensor]]) -> List[tf.Tensor]:
        """
        Given a tuple of an operation to mirror and the parent tensor, create the new op.  Return a list of output
        tensors from the new op.
        :param op_tensor_tuple: Tuple containing (op to winnow, list of input tensors to the op)
        :return: List of output tensors of the winnowed op
        """

        switcher = {
            "Conv2D": module_reducers.reduce_conv2d,
            "DepthwiseConv2dNative": module_reducers.reduce_conv2d,
            "MaxPool": module_reducers.reduce_maxpool,
            "BatchNorm": module_reducers.reduce_batchnorm,
            "FusedBatchNormV3": module_reducers.reduce_batchnorm,
            "Relu": module_reducers.reduce_relu,
            "Relu6": module_reducers.reduce_relu,
            "AvgPool": module_reducers.reduce_avgpool,
            "Tanh": module_reducers.reduce_tanh,
            "Add": module_reducers.reduce_add,
            "AddN": module_reducers.reduce_add,
            "AddV2": module_reducers.reduce_add,
            "Identity": module_reducers.reduce_identity,
            "Dropout": module_reducers.reduce_dropout,
            "Pad": module_reducers.reduce_pad,
            "PadV2": module_reducers.reduce_pad,
            "MirrorPad": module_reducers.reduce_pad,
            "Minimum": module_reducers.reduce_min_max,
            "Maximum": module_reducers.reduce_min_max,
            "Downsample": module_reducers.reduce_downsample,
            "Upsample2D": module_reducers.reduce_upsample2d,
            "LeakyRelu": module_reducers.reduce_leaky_relu
        }

        reducer = switcher.get(op_tensor_tuple[0].type, module_reducers.reduce_default)
        op_mask = self._op_to_mask_dict[op_tensor_tuple[0]]
        name, output_op_node, module = reducer(self._sess, op_tensor_tuple, op_mask)
        self._reduced_op_info[op_tensor_tuple[0]] = ReducedInfo(name, output_op_node, module)
        self._reduced_modules[op_tensor_tuple[0].get_module().name] = (module, op_mask)

        output_tensors = output_op_node.outputs
        if op_tensor_tuple[0].type in ['FusedBatchNormV3', 'Dropout']:
            output_tensors = output_tensors[:1]     # only get first output tensor, but in list form

        # Remove winnowed bn ops from UPDATE_OPS if present
        if op_tensor_tuple[0].type == 'FusedBatchNormV3':
            BNUtils.remove_bn_op_from_update_ops(self._sess, op_tensor_tuple[0].get_module())
        return output_tensors

    def _detach_op_from_inputs(self, op: Op):
        """
        Detach op from its parent operations.
        :param op: Op to detach
        """
        tf_ops_to_detach = []
        input_products = op.get_input_products()
        for product in input_products:
            tensor = product.tensor_dict.get(op, None)
            if tensor is not None:
                for consumer in tensor.consumers():
                    corresponding_op = self._conn_graph.get_op_from_module_name(consumer.name)
                    if corresponding_op == op:
                        tf_ops_to_detach.append(consumer)
        graph_editor.detach_inputs(tf_ops_to_detach)

    def _reroute_if_necessary(self, op: Op, new_op_tensors: List[tf.Tensor]) -> bool:
        """
        Reroute old op and new op outputs if the old op's children's masks are unchanged.  If needed, insert downsample
        or upsample ops.
        :param op: Original unwinnowed op whose winnowed counterpart has output tensors new_op_tensors
        :param new_op_tensors: Output tensors of the newly created winnowed version of op
        :return: True if reroute was performed.
        """
        if len(new_op_tensors) > 1:
            # Len of new_op_tensors should only be greater than one in the case of ending at a split
            raise NotImplementedError

        current_op = op
        child_op = op.output.consumers[0]
        while child_op.type == 'branch' or OpConnectivity.get_op_connectivity(ModelApi.tensorflow, child_op.type) == \
                ConnectivityType.skip:
            # For both cases when child op is of type branch or skip connectivity, go one more level down
            current_op = child_op
            child_op = child_op.output.consumers[0]

        # Op output may have multiple consumers, but in the case of non splits, there will be only one tensor shared
        # among all consumers.  Thus looking at only the first consumer is enough to give us the correct tensor to swap.
        if child_op in self._op_to_mask_dict.keys():
            op_mask = self._op_to_mask_dict[op]
            child_op_mask = self._op_to_mask_dict[child_op]
            if not child_op_mask.are_masks_unchanged():
                return False

            # find the correct child op input mask if it has multiple input masks
            prod_index = child_op.get_input_product_index_of_parent(current_op)
            assert prod_index is not None        # did not find any input product that connects to current op
            new_op_tensor = _insert_downsample_or_upsample_ops_if_needed(new_op_tensors[0],
                                                                         op_mask.output_channel_masks[0],
                                                                         child_op_mask.input_channel_masks[prod_index])
        else:
            prod_index = child_op.get_input_product_index_of_parent(current_op)
            assert prod_index is not None        # did not find any input product that connects to current op
            new_op_tensor = new_op_tensors[0]

        # We have hit the end of a string of ops to reduce, and will now connect the newly reduced ops back to the
        # main graph.  This also detaches the old op's output from its old child op
        old_tensor = child_op.get_input_products()[prod_index].tensor_dict[child_op]
        graph_editor.reroute_ts(ts0=new_op_tensor, ts1=old_tensor)
        return True

    def _get_input_tensors_for_winnowed_op(self, op: Op) -> List[tf.Tensor]:
        """
        Get all of the input tensors to be used when winnowing the op.  If the parent of the op to be reduced is
        also an op that has been reduced, the input tensor will be from the reduced parent op.
        If needed, a downsample or upsample op will be attached to the previous op's output before feeding into the
        winnowed op; if this is the case, input tensors will be the outputs of the downsample or upsample op.
        :param op: Unwinnowed op whose parent tensors will be taken as input tensors for the winnowed version.
        :return: List of input tensors to op
        """
        input_tensors = []
        input_products = op.get_input_products()
        if op.type in ['Add', 'AddN', 'AddV2', 'ConcatV2']:
            assert len(input_products) > 1
        else:
            # if op is not of type add or concat or similar, we only expect to have one incoming tensor
            assert len(input_products) == 1

        for input_product in input_products:
            parent_op = input_product.producer

            if parent_op.type == 'branch':
                # if parent op is a branch op, there could be multiple output tensors coming from the parent op
                # need to find the correct index in parent op's outputs list
                branch_op = parent_op
                parent_op = branch_op.inputs[0].producer
                input_tensor = input_product.tensor_dict[op]

                # loop through each output tensor in parent op's output node to see if it matches input_tensor
                for (idx, tensor) in enumerate(parent_op.output_op_node.outputs):
                    if input_tensor == tensor:
                        if parent_op in self._reduced_op_info:
                            # parent op has a reduced version of itself created already
                            # need to select the output tensor from the reduced op, not the original
                            input_tensor = self._reduced_op_info[parent_op].output_op_node.outputs[idx]
            else:
                # make sure incoming tensor only goes to one op (the current op)
                assert len(input_product.tensor_dict.keys()) == 1
                input_tensor = input_product.tensor_dict[op]
                if parent_op in self._reduced_op_info:
                    # parent op has a reduced version of itself created already
                    # need to select the output tensor from the reduced op, not the original
                    # add output tensor from the reduced op corresponding to the parent op
                    input_tensor = self._reduced_op_info[parent_op].output_op_node.outputs[0]

            # now check the masks of the parent op and the current op to see if a downsample or upsample layer
            # needs to be added
            # First, need to find the correct indices for which masks to use
            # If parent op is of type skip, keep going upwards until we find a non skip op to use its output mask
            while OpConnectivity.get_op_connectivity(ModelApi.tensorflow, parent_op.type) == ConnectivityType.skip:
                parent_op = parent_op.inputs[0].producer
            parent_mask = self._op_to_mask_dict[parent_op]
            parent_output_mask = parent_mask.output_channel_masks[0]
            child_mask = self._op_to_mask_dict[op]
            child_input_mask_index = op.get_input_products().index(input_product)
            child_input_mask = child_mask.input_channel_masks[child_input_mask_index]
            input_tensor = _insert_downsample_or_upsample_ops_if_needed(input_tensor, parent_output_mask,
                                                                        child_input_mask)

            input_tensors.append(input_tensor)

        assert len(input_tensors) == len(input_products)
        return input_tensors


def _insert_downsample_or_upsample_ops_if_needed(input_tensor: tf.Tensor,
                                                 parent_mask: List,
                                                 child_mask: List) -> tf.Tensor:
    """
    :param input_tensor: tensor that may need downsample or upsample op appended after it
    :param parent_mask: mask of parent op
    :param child_mask: mask of child op
    :return input_tensor: tensor that is either the original input_tensor, or the output of a downsample or upsample
    op that has been appended to the input_tensor
    """

    # some sanity checks
    assert len(child_mask) == len(parent_mask)
    parent_mask_sum = sum(parent_mask)
    child_mask_sum = sum(child_mask)
    if parent_mask_sum == child_mask_sum:
        # parent and child masks have same numbers of channels.  Ensure that the two lists match for each index.
        assert parent_mask == child_mask
        # no downsample or upsample is needed. Return the original input tensor
    elif parent_mask_sum > child_mask_sum:
        input_tensor = _insert_downsample_op(input_tensor, parent_mask, child_mask)
    else:
        input_tensor = _insert_upsample_op(input_tensor, parent_mask, child_mask)

    return input_tensor


def _insert_downsample_op(input_tensor: tf.Tensor, parent_mask: List, child_mask: List) -> tf.Tensor:
    """
    Append gather operation to input_tensor, and return the output tensor of the gather operation
    :param input_tensor: Tensor to attach downsample op to
    :param parent_mask: Output mask of the parent op of the tensor
    :param child_mask: Input mask of the child op of the tensor
    :return: Output of the downsample op
    """

    # Ensure that for all indices where child_mask has a 1, parent_mask also has a 1
    assert list(map(lambda x, y: x & y, parent_mask, child_mask)) == child_mask

    gather_list = get_indices_among_ones_of_overlapping_ones(parent_mask, child_mask)
    gather_tensor = module_reducers.create_downsample_op('downsample', input_tensor, gather_list)
    return gather_tensor


def _insert_upsample_op(input_tensor: tf.Tensor, parent_mask: List, child_mask: List) -> tf.Tensor:
    """
    Unstack input_tensor an insert zero elements where necessary, then restack and return resulting tensor
    :param input_tensor: Tensor to attach upsample op to
    :param parent_mask: Output mask of the parent op of the tensor
    :param child_mask: Input mask of the child op of the tensor
    :return: Output of the upsample op
    """

    # Ensure that for all indices where parent_mask has a 1, child_mask also has a 1
    assert list(map(lambda x, y: x & y, parent_mask, child_mask)) == parent_mask

    # Get list of indices for which each channel in the current tensor should map to after upsampling
    index_list = get_indices_among_ones_of_overlapping_ones(child_mask, parent_mask)

    # Channels index is the last index in the tensor
    with tf.name_scope("upsample"):
        unstacked = tf.unstack(input_tensor, axis=-1)
        zeros = tf.zeros_like(unstacked[0])
        current_index = 0

        # loop for inserting zeros at the correct index points
        for index in index_list:
            while current_index < index:
                unstacked.insert(current_index, zeros)
                current_index += 1
            current_index += 1

        # If child mask has additional ones after the last index where parent mask also had a 1, we need to also add
        # rows for these as well, since the additional ones would not be represented in index_list
        # Ex:
        # parent_mask: 1, 1, 1, 0
        #  child_mask: 1, 1, 1, 1
        # index_list would only read [0, 1, 2], but we also need to add a row of zeros for the last one in child mask

        # Simply slice child mask starting at the location after the last entry in index_list (the last matching ones
        # index between parent mask and child).  The sum of the remaining entries will give the number of zeros rows to
        # add.
        for _ in range(sum(child_mask[index_list[-1] + 1:])):
            unstacked.append(zeros)
        stack = tf.stack(unstacked, axis=-1)

    return stack
