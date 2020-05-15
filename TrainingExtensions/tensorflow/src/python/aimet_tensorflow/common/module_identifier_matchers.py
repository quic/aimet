# /usr/bin/env python3.5
# -*- mode: python -*-
#  =============================================================================
#
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
#
#  =============================================================================

""" Functions for matching different types of modules. """

import re
import struct
from typing import Dict
import tensorflow as tf
from aimet_common.utils import AimetLogger
from aimet_tensorflow.common.operation import TfApi

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)


class ModuleIdentifierOpInfo:
    """ Class for summarizing information regarding a tf operation """
    def __init__(self, module_name, op_type, tf_op, tf_api=TfApi.keras):
        self._module_name = module_name
        self._op_type = op_type
        self._tf_op = tf_op
        self._tf_api = tf_api
        self._attributes = {}

    @property
    def module_name(self):
        """ Returns the module name corresponding to this operation. """
        return self._module_name

    @module_name.setter
    def module_name(self, module_name):
        """ Sets the module name of an Operation. """
        self._module_name = module_name

    @property
    def op_type(self):
        """ Returns the op type of the module corresponding to this operation. """
        return self._op_type

    @op_type.setter
    def op_type(self, op_type):
        """ Sets the op type """
        self._op_type = op_type

    @property
    def tf_op(self):
        """ Returns the tf op for the module corresponding to this operation. """
        return self._tf_op

    @property
    def tf_api(self):
        """ Returns the tf api of the module. """
        return self._tf_api

    @tf_api.setter
    def tf_api(self, tf_api):
        """ Sets the tf api of the module. """
        self._tf_api = tf_api

    def add_attribute(self, attribute_name: str, attribute):
        """ Set an attribute of the module identifier op info """
        self._attributes[attribute_name] = attribute

    def get_attributes(self):
        """ Return the attributes dictionary """
        return self._attributes


def match_conv2d_dense_type_ops(op_to_module_dict: Dict[tf.Operation, ModuleIdentifierOpInfo],
                                op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for Conv2d and Dense type ops
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at either the conv2d op or the matmul op
    op = op_info.tf_op
    if op.type == 'MatMul':
        op_info.op_type = 'Dense'
    op_info.module_name = op.name
    op_to_module_dict[op] = op_info
    if len(op.outputs) > 1:
        logger.error('Not expecting Conv2D to ever have more than one output tensor')
        assert False
    if len(op.outputs[0].consumers()) > 1:
        # Hit end of Conv2D if output of current op goes to more than one child op
        return True
    if not op.outputs[0].consumers():
        # Conv op does not lead to any op. This can happen if this Conv op was winnowed, and this is a dangling
        # conv op with no bias. Still represent this as an Op in the Connected Graph.
        return True
    if op.outputs[0].consumers()[0].type == 'BiasAdd':
        op_to_module_dict[op.outputs[0].consumers()[0]] = op_info
        return True
    return False


def match_fusedbatchnorm_pattern_1(op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for FusedBatchNormV3 type ops using tensor to switch between training and non training mode
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at op of type FusedBatchNormV3, and try to match pattern 1 (uses placeholder tensor for switching between
    # training and non training) to op_to_module_dict
    op = op_info.tf_op
    try:
        assert len(op.inputs) == 5
        switch_op = op.inputs[0].op
        assert switch_op.type == 'Switch'
        # FusedBatchNormV3 op has 5 outputs, 3 of which are being used
        assert len(op.outputs) == 6
        merge_op = op.outputs[0].consumers()[0]
        assert merge_op.type == 'Merge'

        # @todo check if there is a way to identify gradient nodes without this string match
        if op.outputs[1].consumers()[0].name.startswith('gradients'):
            merge_1_op = op.outputs[1].consumers()[1]
        else:
            merge_1_op = op.outputs[1].consumers()[0]

        assert merge_1_op.type == 'Merge'

        if op.outputs[2].consumers()[0].name.startswith('gradients'):
            merge_2_op = op.outputs[2].consumers()[1]
        else:
            merge_2_op = op.outputs[2].consumers()[0]

        assert merge_2_op.type == 'Merge'

        # Now look up from merge node to find training fused bn op and parent switch op
        # Both must be added to op_to_module dict since they are in the path of DFS from model input
        training_fused_bn = merge_op.inputs[1].op
        assert training_fused_bn.type == 'FusedBatchNormV3' and training_fused_bn.get_attr('is_training')
        training_switch_op = training_fused_bn.inputs[0].op
        assert training_switch_op.type == 'Switch'
        pred_id_op = training_switch_op.inputs[1].op
        assert pred_id_op.type == 'Identity'
        training_tensor = pred_id_op.inputs[0]

        # This fusedbatchnorm uses a placeholder tensor for determining whether it is in training mode or not
        op_info.add_attribute('training', training_tensor.name)
        # FusedBatchNormV3s of this type always end with /cond/FusedBatchNormV3_1 in the name
        # Everything preceding the cond is the scope name
        match_name = re.match('(.+)/cond/FusedBatchNormV3_1', op.name)
        if match_name:
            op_info.module_name = match_name.group(1)
        op_to_module_dict.update({op: op_info,
                                  switch_op: op_info,
                                  merge_op: op_info,
                                  training_fused_bn: op_info,
                                  training_switch_op: op_info
                                  })
        _add_self_and_descendants_to_module_dict(op_to_module_dict, op_info, merge_1_op)
        _add_self_and_descendants_to_module_dict(op_to_module_dict, op_info, merge_2_op)
        return True
    except:     # pylint: disable=bare-except
        return False


def match_fusedbatchnorm_pattern_2(op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for FusedBatchNormV3 type ops with training set to True
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at op of type FusedBatchNormV3, and try to match pattern 3 (training=True) to op_to_module_dict
    op = op_info.tf_op
    try:
        assert len(op.inputs) == 5
        gamma_op = op.inputs[1].op
        assert gamma_op.type in ['ReadVariableOp', 'Const', 'Identity']
        beta_op = op.inputs[2].op
        assert beta_op.type in ['ReadVariableOp', 'Const', 'Identity']
        moving_mean_op = op.inputs[3].op
        assert moving_mean_op.type == 'Const'
        moving_variance_op = op.inputs[4].op
        assert moving_variance_op.type == 'Const'

        assert len(op.outputs) == 6
        moving_mean_op = op.outputs[1].consumers()[0]
        assert moving_mean_op.type == 'Sub'
        moving_variance_op = op.outputs[2].consumers()[0]
        assert moving_variance_op.type == 'Sub'

        op_info.add_attribute('training', True)
        # FusedBatchNormV3s of this type always end with /FusedBatchNormV3 in the name
        # Everything preceding the cond is the scope name
        match_name = re.match('(.+)/FusedBatchNormV3', op.name)
        if match_name:
            op_info.module_name = match_name.group(1)
        op_to_module_dict.update({op: op_info})
        _add_self_and_descendants_to_module_dict(op_to_module_dict, op_info, moving_mean_op)
        _add_self_and_descendants_to_module_dict(op_to_module_dict, op_info, moving_variance_op)
        return True
    except:     # pylint: disable=bare-except
        return False


def match_fusedbatchnorm_pattern_3(op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for FusedBatchNormV3 type ops with training set to False
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at op of type FusedBatchNormV3, and try to match pattern 2 (training=False) to op_to_module_dict
    op = op_info.tf_op
    try:
        assert len(op.inputs) == 5
        gamma_op = op.inputs[1].op
        # Need to keep consts and identities for tf slim batchnorms
        assert gamma_op.type in ['ReadVariableOp', 'Const', 'Identity']
        beta_op = op.inputs[2].op
        assert beta_op.type in ['ReadVariableOp', 'Const', 'Identity']
        moving_mean_op = op.inputs[3].op
        assert moving_mean_op.type in ['ReadVariableOp', 'Const', 'Identity']
        moving_variance_op = op.inputs[4].op
        assert moving_variance_op.type in ['ReadVariableOp', 'Const', 'Identity']

        op_info.add_attribute('training', False)
        # FusedBatchNormV3s of this type always end with /FusedBatchNormV3 in the name
        # Everything preceding the cond is the scope name
        match_name = re.match('(.+)/FusedBatchNormV3', op.name)
        if match_name:
            op_info.module_name = match_name.group(1)
        op_to_module_dict.update({op: op_info})
        return True
    except:     # pylint: disable=bare-except
        return False


def match_flatten_ops(op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for tf slim flatten type ops
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at op of type reshape and try to match to tf slim flatten pattern
    op = op_info.tf_op
    op_info.op_type = "Flatten"
    op_info.tf_api = TfApi.slim
    try:
        pack_op = op.inputs[1].op
        assert pack_op.type == "Pack"
        strided_slice_op = pack_op.inputs[0].op
        assert strided_slice_op.type == "StridedSlice"

        op_to_module_dict[op] = op_info

        # keras model has previous module feeding into shape_op as well, need to add all these ops to the
        # dictionary since they will be seen in the depth first search
        # shape_op will either be an actual shape op, or a const representing a shape
        shape_op = strided_slice_op.inputs[0].op
        if shape_op.inputs:
            op_info.tf_api = TfApi.keras
            op_to_module_dict[pack_op] = op_info
            op_to_module_dict[strided_slice_op] = op_info
            op_to_module_dict[shape_op] = op_info
        return True

    # if any of the above fails, the structure is not as we expect, declare it as not of type Flatten
    except:     # pylint: disable=bare-except
        return False


def match_dropout_pattern_1(op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for keras dropout type ops
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at op of type RandomUniform and try to match to dropout pattern 1 (keras pattern)
    op = op_info.tf_op
    try:
        mul = op.outputs[0].consumers()[0]
        assert mul.type == "Mul"
        add = mul.outputs[0].consumers()[0]
        assert add.type == "Add"
        greater_equal = add.outputs[0].consumers()[0]
        assert greater_equal.type == "GreaterEqual"
        cast = greater_equal.outputs[0].consumers()[0]
        assert cast.type == "Cast"
        mul_1 = cast.outputs[0].consumers()[0]
        assert mul_1.type == "Mul"
        merge = mul_1.outputs[0].consumers()[0]
        assert merge.type == "Merge"
        identity = merge.inputs[0].op
        assert identity.type == "Identity"
        switch = identity.inputs[0].op
        assert switch.type == "Switch"
        mul_2 = mul_1.inputs[0].op
        assert mul_2.type == "Mul"
        shape = op.inputs[0].op
        assert shape.type == "Shape"
        switch_1 = shape.inputs[0].op
        assert switch_1.type == "Switch"

        # Add ops to the op to module dict
        op_info.op_type = "Dropout"
        op_info.add_attribute('rate_tensor', greater_equal.inputs[1])
        match_name = re.match("(.+)/cond", merge.name)
        if match_name:
            op_info.module_name = match_name.group(1)
        op_to_module_dict.update({op: op_info,
                                  mul: op_info,
                                  add: op_info,
                                  greater_equal: op_info,
                                  cast: op_info,
                                  mul_1: op_info,
                                  merge: op_info,
                                  identity: op_info,
                                  switch: op_info,
                                  mul_2: op_info,
                                  shape: op_info,
                                  switch_1: op_info})
        return True
    except:     # pylint: disable=bare-except
        return False


def match_dropout_pattern_2(op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for tf slim dropout type ops (seen when creating dropout node using tf slim api)
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at op of type RandomUniform and try to match to dropout pattern 2 (tf slim pattern option 1)
    op = op_info.tf_op
    try:
        mul = op.outputs[0].consumers()[0]
        assert mul.type == "Mul"
        add = mul.outputs[0].consumers()[0]
        assert add.type == "Add"
        greater_equal = add.outputs[0].consumers()[0]
        assert greater_equal.type == "GreaterEqual"
        cast = greater_equal.outputs[0].consumers()[0]
        assert cast.type == "Cast"
        mul_1 = cast.outputs[0].consumers()[0]
        assert mul_1.type == "Mul"
        shape = op.inputs[0].op
        assert shape.type == "Shape"
        mul_2 = mul_1.inputs[0].op
        assert mul_2.type == "Mul"
        assert mul_2.inputs[0] == shape.inputs[0]

        # Add ops to the op to module dict
        op_info.op_type = "Dropout"
        op_info.add_attribute('rate_tensor', greater_equal.inputs[1])
        op_info.tf_api = TfApi.slim
        match_name = re.search("(.+)/random_uniform/RandomUniform", op.name)
        if match_name:
            op_info.module_name = match_name.group(1)
        op_to_module_dict.update({op: op_info,
                                  mul: op_info,
                                  add: op_info,
                                  greater_equal: op_info,
                                  cast: op_info,
                                  mul_1: op_info,
                                  shape: op_info,
                                  mul_2: op_info})
        return True
    except:     # pylint: disable=bare-except
        return False


def match_dropout_pattern_3(op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for tf slim dropout type ops (seen in vgg16 slim)
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at op of type Mul and try to match to dropout pattern 3 (tf slim pattern option 2)
    op = op_info.tf_op
    try:
        # Find all expected ops for pattern 3
        truediv = op.inputs[1].op       # Not needed in op dictionary
        assert truediv.type == "RealDiv"
        sub = truediv.inputs[1].op       # Not needed in op dictionary
        assert sub.type == "Sub"
        mul = op.outputs[0].consumers()[0]
        assert mul.type == "Mul"
        cast = mul.inputs[1].op       # Not needed in op dictionary
        assert cast.type == "Cast"
        greater_equal = cast.inputs[0].op       # Not needed in op dictionary
        assert greater_equal.type == "GreaterEqual"
        add = greater_equal.inputs[0].op       # Not needed in op dictionary
        assert add.type == "Add"
        mul_1 = add.inputs[0].op       # Not needed in op dictionary
        assert mul_1.type == "Mul"
        random_uniform = mul_1.inputs[0].op       # Not needed in op dictionary
        assert random_uniform.type == "RandomUniform"
        shape = random_uniform.inputs[0].op
        assert shape.type == "Const"

        # Add ops to the op to module dict
        op_info.op_type = "Dropout"
        op_info.add_attribute('rate_tensor', greater_equal.inputs[1])
        op_info.tf_api = TfApi.slim
        match_name = re.search("(.+)/random_uniform/RandomUniform", random_uniform.name)
        if match_name:
            op_info.module_name = match_name.group(1)
        op_to_module_dict.update({op: op_info,
                                  mul: op_info})
        return True
    except:     # pylint: disable=bare-except
        return False


def match_softmax(op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for tf slim softmax type ops
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at op of type softmax and try to match to tf slim softmax pattern
    op = op_info.tf_op
    op_info.tf_api = TfApi.slim
    try:
        reshape = op.inputs[0].op
        assert reshape.type == "Reshape"
        reshape_1 = op.outputs[0].consumers()[0]
        assert reshape_1.type == "Reshape"
        op_to_module_dict.update({op: op_info,
                                  reshape: op_info,
                                  reshape_1: op_info})
        if len(reshape_1.inputs) == 2 and reshape_1.inputs[1].op.type == "Shape":
            op_to_module_dict.update({reshape_1.inputs[1].op: op_info})
        return True
    except:     # pylint: disable=bare-except
        return False


def match_downsample(op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for downsample type ops
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at op of type GatherV2 and try to match to downsample pattern
    op = op_info.tf_op
    # Add ops to the op to module dict
    match_name = re.search(".*downsample(_[0-9]+)*/GatherV2(_[0-9]+)*", op.name)

    # Only declare downsample if name matches as expected.  Otherwise, GatherV2 can show up in other cases too.
    if not match_name:
        return False
    op_info.module_name = match_name.group(0)
    op_info.op_type = "Downsample"
    op_to_module_dict.update({op: op_info})
    return True


def match_upsample(op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for upsample type ops
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at op of type Unpack and try to match to upsample pattern
    op = op_info.tf_op
    try:
        # For some reason, sometimes the indices of unstack's consumers are switched
        consumer_1 = op.outputs[0].consumers()[0]
        assert consumer_1.type in ['ZerosLike', 'Pack']
        consumer_2 = op.outputs[0].consumers()[1]
        assert consumer_2.type in ['ZerosLike', 'Pack']
        assert consumer_1.type != consumer_2.type

        # Add ops to the op to module dict
        op_info.op_type = "Upsample"
        match_name = re.match("(.+)/unstack", op.name)
        if match_name:
            op_info.module_name = match_name.group(1)
        op_to_module_dict.update({op: op_info,
                                  consumer_1: op_info,
                                  consumer_2: op_info})
        return True
    except:     # pylint: disable=bare-except
        return False


def match_upsample2d(op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for upsample2D type ops
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at op of type Shape and try to match to upsample2d pattern
    op = op_info.tf_op
    try:
        strided_slice = op.outputs[0].consumers()[0]
        assert strided_slice.type == 'StridedSlice'
        mul = strided_slice.outputs[0].consumers()[0]
        assert mul.type == 'Mul'
        resize_nearest_neighbor = mul.outputs[0].consumers()[0]
        assert resize_nearest_neighbor.type == 'ResizeNearestNeighbor'
        prev_op = op.inputs[0].op
        assert prev_op == resize_nearest_neighbor.inputs[0].op

        # Add ops to the op to module dict
        op_info.op_type = "Upsample2D"
        match_name = re.match("(.+)/Shape", op.name)
        if match_name:
            op_info.module_name = match_name.group(1)

        # Fill in size attribute
        const_op = mul.inputs[1].op
        tensor_content_length = const_op.get_attr('value').tensor_shape.dim[0].size
        unpack_string = str(tensor_content_length) + 'i'       # i for int, indices_length tells how many integers to parse out
        upsample_size = struct.unpack(unpack_string, const_op.get_attr('value').tensor_content)
        op_info.add_attribute('size', upsample_size)

        op_to_module_dict.update({op: op_info,
                                  strided_slice: op_info,
                                  mul: op_info,
                                  resize_nearest_neighbor: op_info})
        return True
    except:     # pylint: disable=bare-except
        return False


def match_leaky_relu(op_to_module_dict: dict, op_info: ModuleIdentifierOpInfo) -> bool:
    """
    Matcher for leaky_relu type ops
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :return: True if a valid match was made, False otherwise
    """
    # Begin at op of type Mul and try to match to leaky_relu pattern
    op = op_info.tf_op
    try:
        alpha_op = op.inputs[0].op
        assert alpha_op.type == 'Const'
        maximum = op.outputs[0].consumers()[0]
        assert maximum.type == 'Maximum'
        prev_op = op.inputs[1].op
        assert prev_op == maximum.inputs[1].op

        # Fill in alpha attribute
        alpha = alpha_op.get_attr('value').float_val[0]
        op_info.add_attribute('alpha', alpha)

        # Add ops to the op to module dict
        op_info.op_type = "LeakyRelu"
        match_name = re.match("(.+)/mul", op.name)
        if match_name:
            op_info.module_name = match_name.group(1)

        op_to_module_dict.update({op: op_info,
                                  maximum: op_info})
        return True
    except:     # pylint: disable=bare-except
        return False


def handle_default(*_) -> bool:
    """ Do nothing here """
    return True


def _add_self_and_descendants_to_module_dict(op_to_module_dict: dict,
                                             op_info: ModuleIdentifierOpInfo,
                                             op: tf.Operation):
    """
    Starting at op, add it and all children to the op_to_module_dict, associating op_info with each
    :param op_to_module_dict: Dictionary mapping tf ops to ModuleIdentifierOpInfo objects.  All tf ops belonging to the
    same module will be mapped to the same ModuleIdentifierOpInfo object.
    :param op_info: ModuleIdentifierOpInfo to fill in, for holding information about the module that multiple tf ops
    belong to
    :param op: Op to add self and all descendants to op_to_module_dict
    """
    op_queue = [op]
    while op_queue:
        current_op = op_queue.pop()
        op_to_module_dict[current_op] = op_info
        for product in current_op.outputs:
            for consumer in product.consumers():
                op_queue.append(consumer)
