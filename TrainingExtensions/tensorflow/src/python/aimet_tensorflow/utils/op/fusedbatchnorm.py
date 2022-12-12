# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
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
# =============================================================================
""" utilities for fused batchnorm op """
# pylint: disable=too-many-lines

from typing import Union, List
import numpy as np
import tensorflow as tf

from aimet_common.utils import AimetLogger
from aimet_tensorflow.utils import constants
from aimet_tensorflow import graph_editor

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

_BN_STRUCTURE_ERROR_MSG = "BN op doesn't have the expected structure"


class BNUtils:
    """ Batch Norm/ fused Batch Norm op related utils"""
    # pylint: disable=too-many-public-methods

    # pylint: disable=too-many-locals
    @staticmethod
    def get_bn_params_tf_variable(sess: tf.compat.v1.Session, bn_op: tf.Operation) ->List[tf.compat.v1.Variable]:
        """
        To get_bn_params_tf_variable for read and write.  specific to the bn op pattern in TF 2.x runtime
        :param sess:
        :param bn_op: Batch normalization op that should be worked as passthrough op (no-op)
        :return: tf_variable for gamma, beta, mean, var
        """
        assert bn_op.type in ['FusedBatchNormV3', 'Identity']
        if bn_op.type == 'FusedBatchNormV3':
            bn_gamma_tf_var_name = bn_op.inputs[1].op.inputs[0].name
            bn_beta_tf_var_name = bn_op.inputs[2].op.inputs[0].name
            bn_mean_tf_var_name = bn_op.inputs[3].op.inputs[0].name
            bn_var_tf_var_name = bn_op.inputs[4].op.inputs[0].name
        else:
            bn_gamma_tf_var_name = bn_op.inputs[0].op.inputs[1].name
            bn_beta_tf_var_name = bn_op.inputs[0].op.inputs[2].name
            bn_mean_tf_var_name = bn_op.inputs[0].op.inputs[3].name
            bn_var_tf_var_name = bn_op.inputs[0].op.inputs[4].name

        with sess.graph.as_default():
            tf_global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
            for v in tf_global_vars:
                if v.name == bn_gamma_tf_var_name:
                    bn_gamma_tf_var = v
                if v.name == bn_beta_tf_var_name:
                    bn_beta_tf_var = v
                if v.name == bn_mean_tf_var_name:
                    bn_mean_tf_var = v
                if v.name == bn_var_tf_var_name:
                    bn_var_tf_var = v
        return bn_gamma_tf_var, bn_beta_tf_var, bn_mean_tf_var, bn_var_tf_var

    @staticmethod
    def modify_bn_params_to_make_as_passthrough(sess: tf.compat.v1.Session, bn_op: tf.Operation):
        """
        To change the batch normalization parameters to work as no-op operation
        :param sess:
        :param bn_op: Batch normalization op that should be worked as passthrough op (no-op)
        """
        bn_gamma_tf_var, bn_beta_tf_var, bn_mean_tf_var, bn_var_tf_var = BNUtils.get_bn_params_tf_variable(sess, bn_op)
        with sess.graph.as_default():
            sess.run([tf.compat.v1.assign(bn_gamma_tf_var, np.ones(bn_gamma_tf_var.shape, dtype=bn_gamma_tf_var.dtype.as_numpy_dtype)),
                      tf.compat.v1.assign(bn_beta_tf_var, np.zeros(bn_beta_tf_var.shape, dtype=bn_beta_tf_var.dtype.as_numpy_dtype)),
                      tf.compat.v1.assign(bn_mean_tf_var, np.zeros(bn_mean_tf_var.shape, dtype=bn_mean_tf_var.dtype.as_numpy_dtype)),
                      tf.compat.v1.assign(bn_var_tf_var, np.ones(bn_var_tf_var.shape, dtype=bn_var_tf_var.dtype.as_numpy_dtype))])


    @staticmethod
    def skip_bn_op(sess: tf.compat.v1.Session, bn_op: tf.Operation, in_tensor: tf.Tensor, out_tensor: tf.Tensor):
        """
        Skip given bn op specified (fused batch norm op).
        Note: supports only Fused bn op types.

        :param sess: Tensorflow session
        :param bn_op: Batchnorm op to be skipped
        :param in_tensor: Input tensor to the batchnorm op
        :param out_tensor: Output tensor of the batchnorm op
        """

        if in_tensor is None or out_tensor is None:
            logger.error("Error, input and output tensors must be provided for skipping the op")
            assert False
        else:
            with sess.graph.as_default():
                if bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm', 'Identity']:
                    graph_editor.detach_outputs(in_tensor.op)
                    graph_editor.reroute_ts(in_tensor, out_tensor)
                    BNUtils.remove_bn_op_from_update_ops(sess, bn_op)
                else:
                    logger.error("Error, Unknown BN op")
                    assert False

    @staticmethod
    def _get_tensor_read_var_op_trainable_bn_op(input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Generic helper to find a read op tensor associated with input tensor that can be evaluated, when the bn op is
        marked trainable.

        :param input_tensor: Input tensor to find corresponding read op tensor that can be evaluated
        :return: read var op type tensor as tf.Tensor type.
        """

        logger.debug('Fetching params from trainable BN op type')
        assert input_tensor.op.inputs[0].op.inputs is not None
        # inputs of 0 is beta tensor , get readVarOp associated with it
        var_tensor = input_tensor.op.inputs[0].op.inputs[0]
        assert var_tensor.op.outputs is not None
        assert len(var_tensor.consumers()) >= 3

        tensor_consumers = var_tensor.consumers()

        var_read_tensor = None
        # get read variable op tensor from these consumers
        # do not pick the one with _1 , it is not fetch-able
        for consumer in tensor_consumers:
            if consumer.type == 'ReadVariableOp' and 'ReadVariableOp_1' not in consumer.name:
                assert consumer.outputs is not None
                var_read_tensor = consumer.outputs[0]
                break

        assert var_read_tensor is not None

        return var_read_tensor

    @staticmethod
    def get_beta_read_op(bn_op: tf.Operation) -> tf.Operation:
        """
        Get beta read op from BN op specified.

        :param bn_op: bn_op obtained from connected graph using get_modules (is mul_1 op inside BN scope)
        :return: beta read op
        """
        if bn_op.type in ['Mul']:
            # For regular BN
            # mul_1 -> add_1 <-- sub <-- beta_read
            assert len(bn_op.outputs) >= 1, _BN_STRUCTURE_ERROR_MSG
            add_1 = bn_op.outputs[0].consumers()[0]
            assert len(add_1.inputs) >= 2, _BN_STRUCTURE_ERROR_MSG
            sub = add_1.inputs[1].op
            assert len(sub.inputs) >= 1, _BN_STRUCTURE_ERROR_MSG
            beta_read = sub.inputs[0].op
        elif bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm']:
            assert len(bn_op.inputs) == 5
            beta_read = bn_op.inputs[constants.BN_OP_PARAM_INDICES['beta']].op
            if beta_read.type == 'Switch':      # tf slim bn using training tensor form
                beta_read = beta_read.inputs[0].op
                assert 'read' in beta_read.name
        elif bn_op.type in ['Identity']:
            cond = bn_op.inputs[0].op
            beta_read_tensor = cond.inputs[2]
            beta_read = [node for node in cond.graph.get_operations() if beta_read_tensor in node.inputs][2]
        else:
            logger.error("Error, unknown BN op")
            assert False

        assert beta_read.type in ['ReadVariableOp', 'Identity', 'VarHandleOp']      # Will be VarHandleOp for tf slim BNs in tf2 runtime
        return beta_read

    @staticmethod
    def get_beta_read_var_op_tensor_using_structure(bn_op: tf.Operation) -> tf.Tensor:
        """
        Get beta readVariableOp tensor from BN op specified.

        :param bn_op: FusedBatchNorm as tf.Operation
        :return: tensor associated with bn op beta readVariableOp type, as tf.Tensor
        """
        assert bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm', 'Mul', 'Identity']
        beta_read_tensor = BNUtils.get_beta_read_op(bn_op).outputs[0]

        assert beta_read_tensor is not None
        if beta_read_tensor.op.inputs and beta_read_tensor.op.inputs[0].op.type == 'Switch':
            logger.debug('Fetching params from trainable BN op type')
            beta_read_tensor = BNUtils._get_tensor_read_var_op_trainable_bn_op(beta_read_tensor)

        return beta_read_tensor

    @staticmethod
    def get_beta_read_var_op_tensor(graph: tf.Graph, bn_op: tf.Operation) -> tf.Tensor:
        """
        Get beta readVariableOp tensor from BN op specified.

        :param graph: TensorFlow graph
        :param bn_op: FusedBatchNorm as tf.Operation
        :return: tensor associated with bn op beta readVariableOp type, as tf.Tensor
        """
        try:
            # try name based tensor look up for Keras layers
            beta_read_tensor = BNUtils._get_bn_param_tensor_using_name(graph, bn_op,
                                                                       constants.BNOpParamType.beta)
        except KeyError:
            # if we can't find the tensor name, use structure match
            # to figure out the read tensor for param
            beta_read_tensor = BNUtils.get_beta_read_var_op_tensor_using_structure(bn_op)

        return beta_read_tensor

    @staticmethod
    def get_beta_as_numpy_data(sess: tf.compat.v1.Session, bn_op: tf.Operation) -> np.ndarray:
        """
        Get beta param from BN op specified.

        :param sess: tensorflow session
        :param bn_op: bn_op as tf.Operation
        :return: beta tensor as numpy data
        """

        beta_tensor = BNUtils.get_beta_read_var_op_tensor(sess.graph, bn_op)

        with sess.graph.as_default():
            numpy_data = sess.run(beta_tensor)

        return numpy_data

    @staticmethod
    def get_gamma_as_read_op(bn_op: tf.Operation) -> tf.Operation:
        """
        Get gamma read op from BN op specified.

        :param bn_op: bn_op obtained from connected graph using get_modules (is mul_1 op inside BN scope)
        :return: gamma read op
        """
        if bn_op.type in ['Mul']:
            # For regular BN
            # mul_1 <-- mul <-- gamma_read <-- gamma_tensor
            assert len(bn_op.inputs) >= 2, _BN_STRUCTURE_ERROR_MSG
            mul = bn_op.inputs[1].op
            assert len(mul.inputs) >= 2, _BN_STRUCTURE_ERROR_MSG
            gamma_read = mul.inputs[1].op
        elif bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm']:
            assert len(bn_op.inputs) == 5
            gamma_read = bn_op.inputs[constants.BN_OP_PARAM_INDICES['gamma']].op
            if gamma_read.type == 'Switch':      # tf slim bn using training tensor form
                gamma_read = gamma_read.inputs[0].op
                assert 'read' in gamma_read.name or gamma_read.type == 'Const'
        elif bn_op.type in ['Identity']:
            assert len(bn_op.inputs) == 1
            cond = bn_op.inputs[0].op
            gamma_read_tensor = cond.inputs[1]
            gamma_read = [node for node in cond.graph.get_operations() if gamma_read_tensor in node.inputs][2]
        else:
            logger.error("Error, unknown BN op")
            assert False
        assert gamma_read.type in ['ReadVariableOp', 'Identity', 'Const', 'VarHandleOp']    # Will be identity for tf slim BNs
        return gamma_read

    @staticmethod
    def get_gamma_read_var_op_tensor_using_structure(bn_op: tf.Operation) -> tf.Tensor:
        """
        Get the gamma read var op tensor associated with the batchnorm op.

        :param bn_op: Batchnorm op to get gamma read var op tensor from
        :return: Gamma read var op tensor associated with bn_op
        """
        assert bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm', 'Mul', 'Identity']
        gamma_read_tensor = BNUtils.get_gamma_as_read_op(bn_op).outputs[0]
        assert gamma_read_tensor is not None

        if gamma_read_tensor.op.inputs and gamma_read_tensor.op.inputs[0].op.type == 'Switch':
            logger.debug('Fetching params from trainable BN op type')
            gamma_read_tensor = BNUtils._get_tensor_read_var_op_trainable_bn_op(gamma_read_tensor)

        return gamma_read_tensor

    @staticmethod
    def get_gamma_read_var_op_tensor(graph: tf.Graph, bn_op: tf.Operation) -> tf.Tensor:
        """
        Get the gamma read var op tensor associated with the batchnorm op.

        :param graph: TensorFlow graph
        :param bn_op: Batchnorm op to get gamma read var op tensor from
        :return: Gamma read var op tensor associated with bn_op
        """
        try:
            # try name based tensor look up for Keras layers
            gamma_read_tensor = BNUtils._get_bn_param_tensor_using_name(graph, bn_op,
                                                                        constants.BNOpParamType.gamma)
        except KeyError:
            # if we can't find the tensor name, use structure match
            # to figure out the read tensor for param
            gamma_read_tensor = BNUtils.get_gamma_read_var_op_tensor_using_structure(bn_op)

        return gamma_read_tensor

    @staticmethod
    def get_gamma_as_numpy_data(sess: tf.compat.v1.Session, bn_op: tf.Operation) -> np.ndarray:
        """
        Get gamma param from BN op specified.

        :param sess: tensorflow session
        :param bn_op: bn_op obtained from connected graph using get_modules (is mul_1 op inside BN scope)
        :return: gamma as numpy data
        """

        gamma_tensor = BNUtils.get_gamma_read_var_op_tensor(sess.graph, bn_op)

        with sess.graph.as_default():
            numpy_data = sess.run(gamma_tensor)

        return numpy_data

    @staticmethod
    def _bn_op_var_struct_1(bn_op: tf.Operation) -> Union[tf.Operation, None]:
        """
        Return moving_variance op corresponding to batchnorm with training tensor.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: Read operation for moving_variance
        """
        try:
            mul_op = bn_op.inputs[1].op
            assert mul_op.type == 'Mul'
            rsqrt_op = mul_op.inputs[0].op
            assert rsqrt_op.type == 'Rsqrt'
            add_op = rsqrt_op.inputs[0].op
            assert add_op.type == 'AddV2'
            merge_op = add_op.inputs[0].op
            assert merge_op.type == 'Merge'
            read_op = merge_op.inputs[0].op
            assert read_op.type in ['ReadVariableOp']
            return read_op
        except:     # pylint: disable=bare-except
            return None

    @staticmethod
    def _bn_op_var_struct_2(bn_op: tf.Operation) -> Union[tf.Operation, None]:
        """
        Return moving_variance op corresponding to batchnorm with training=True.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: Read operation for moving_variance
        """
        try:
            mul_op = bn_op.inputs[1].op
            assert mul_op.type == 'Mul'
            rsqrt_op = mul_op.inputs[0].op
            assert rsqrt_op.type == 'Rsqrt'
            add_op = rsqrt_op.inputs[0].op
            assert add_op.type == 'AddV2'
            squeeze_1_op = add_op.inputs[0].op
            assert squeeze_1_op.type == 'Squeeze'
            sub_op = squeeze_1_op.outputs[0].consumers()[0]
            assert sub_op.type == 'Sub'
            read_op = sub_op.inputs[0].op
            assert read_op.type in ['ReadVariableOp']
            return read_op
        except:     # pylint: disable=bare-except
            return None

    @staticmethod
    def _bn_op_var_struct_3(bn_op: tf.Operation) -> Union[tf.Operation, None]:
        """
        Return moving_variance op corresponding to batchnorm with training=False.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: Read operation for moving_variance
        """
        try:
            mul_op = bn_op.inputs[1].op
            assert mul_op.type == 'Mul'
            rsqrt_op = mul_op.inputs[0].op
            assert rsqrt_op.type == 'Rsqrt'
            add_op = rsqrt_op.inputs[0].op
            assert add_op.type == 'AddV2'
            read_op = add_op.inputs[0].op
            assert read_op.type in ['ReadVariableOp']
            return read_op
        except:     # pylint: disable=bare-except
            return None

    @staticmethod
    def get_moving_variance_as_read_op(bn_op: tf.Operation) -> Union[tf.Operation, None]:
        """
        Get moving variance read op from BN op specified.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: moving variance as read op
        """
        # register handlers for different structures
        bn_op_struct_for_variance_handlers = [BNUtils._bn_op_var_struct_1,
                                              BNUtils._bn_op_var_struct_2,
                                              BNUtils._bn_op_var_struct_3]

        if bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm']:
            assert len(bn_op.inputs) == 5
            moving_var_read = bn_op.inputs[constants.BN_OP_PARAM_INDICES['movingvariance']].op
            if moving_var_read.type == 'Switch':      # tf slim bn using training tensor form
                moving_var_read = moving_var_read.inputs[0].op
                assert 'read' in moving_var_read.name
        elif bn_op.type in ['Mul']:
            # For regular BN
            moving_var_read = None
            # try all handlers available
            for handler in bn_op_struct_for_variance_handlers:
                if moving_var_read is None:
                    moving_var_read = handler(bn_op)
                else:
                    break
            assert moving_var_read is not None, _BN_STRUCTURE_ERROR_MSG

        elif bn_op.type in ['Identity']:
            assert len(bn_op.inputs) == 1
            cond = bn_op.inputs[0].op
            moving_var_read_tensor = cond.inputs[4]
            moving_var_read = [node for node in cond.graph.get_operations() if moving_var_read_tensor in node.inputs][2]
        else:
            logger.error("Error, unknown BN op")
            assert False

        if moving_var_read.type == 'Identity':
            assert len(moving_var_read.inputs) == 1, _BN_STRUCTURE_ERROR_MSG

        assert moving_var_read.type in ['ReadVariableOp', 'Const', 'Identity', 'VarHandleOp']

        return moving_var_read

    @staticmethod
    def _get_moving_variance_read_var_op_tensor_using_structure(bn_op: tf.Operation) -> tf.Tensor:
        """
        Get moving variance readVariableOp tensor from BN op specified.

        :param bn_op: FusedBatchNorm as tf.Operation
        :return: tensor associated with bn op moving variance readVariableOp type, as tf.Tensor
        """
        # only support fused BN
        assert bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm', 'Mul', 'Identity']
        moving_var_read_tensor = BNUtils.get_moving_variance_as_read_op(bn_op).outputs[0]
        assert moving_var_read_tensor is not None

        if moving_var_read_tensor.op.type == 'Const':
            logger.debug("BN op has const type op for moving variance")

            # get the sub_1 op associated with moving variance read op
            assert len(bn_op.outputs) >= 2
            moving_avg_1_sub_1 = bn_op.outputs[2].consumers()[0]
            all_inputs = moving_avg_1_sub_1.inputs

            # among inputs figure out the read var op type that can be "evaluated"
            for input_t in all_inputs:
                if input_t.op.type == 'ReadVariableOp':
                    moving_var_read_tensor = input_t
                elif input_t.op.type == 'Identity' and 'read:0' in input_t.name:      # tf slim form
                    moving_var_read_tensor = input_t

        elif moving_var_read_tensor.op.inputs and moving_var_read_tensor.op.inputs[0].op.type == 'Switch':
            logger.debug("Fetch moving var from a trainable BN op structure")
            moving_var_read_tensor = BNUtils._get_tensor_read_var_op_trainable_bn_op(moving_var_read_tensor)

        return moving_var_read_tensor

    @staticmethod
    def get_moving_variance_read_var_op_tensor(graph: tf.Graph, bn_op: tf.Operation) -> tf.Tensor:
        """
        Get moving variance readVariableOp tensor from BN op specified.

        :param graph: TensorFlow graph
        :param bn_op: FusedBatchNorm as tf.Operation
        :return: tensor associated with bn op moving variance readVariableOp type, as tf.Tensor
        """
        try:
            # try name based tensor look up for Keras layers
            moving_var_read_tensor = BNUtils._get_bn_param_tensor_using_name(graph, bn_op,
                                                                             constants.BNOpParamType.moving_variance)
        except KeyError:
            # if we can't find the tensor name, use structure match
            # to figure out the read tensor for param
            moving_var_read_tensor = BNUtils._get_moving_variance_read_var_op_tensor_using_structure(bn_op)

        return moving_var_read_tensor

    @staticmethod
    def get_moving_variance_as_numpy_data(sess: tf.compat.v1.Session, bn_op: tf.Operation) -> np.ndarray:
        """
        Get moving variance param from BN op specified.

        :param sess: tensorflow session
        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: moving variance as numpy data
        """

        moving_var_tensor = BNUtils.get_moving_variance_read_var_op_tensor(sess.graph, bn_op)

        with sess.graph.as_default():
            numpy_data = sess.run(moving_var_tensor)

        return numpy_data

    @staticmethod
    def _bn_op_mean_struct_1(bn_op: tf.Operation) -> Union[tf.Operation, None]:
        """
        Return moving_mean op corresponding to batchnorm with training tensor.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: Read operation for moving_mean
        """
        try:
            mul_op = bn_op.inputs[1].op
            assert mul_op.type == 'Mul'
            mul_2_op = mul_op.outputs[0].consumers()[1]
            assert mul_2_op.type == 'Mul'
            merge_op = mul_2_op.inputs[0].op
            assert merge_op.type == 'Merge'
            read_op = merge_op.inputs[0].op
            assert read_op.type in ['ReadVariableOp']
            return read_op
        except:     # pylint: disable=bare-except
            return None

    @staticmethod
    def _bn_op_mean_struct_2(bn_op: tf.Operation) -> Union[tf.Operation, None]:
        """
        Return moving_mean op corresponding to batchnorm with training=True.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: Read operation for moving_mean
        """
        try:
            mul_op = bn_op.inputs[1].op
            assert mul_op.type == 'Mul'
            mul_2_op = mul_op.outputs[0].consumers()[1]
            assert mul_2_op.type == 'Mul'
            squeeze_op = mul_2_op.inputs[0].op
            assert squeeze_op.type == 'Squeeze'
            sub_op = squeeze_op.outputs[0].consumers()[0]
            assert sub_op.type == 'Sub'
            read_op = sub_op.inputs[0].op
            assert read_op.type in ['ReadVariableOp']
            return read_op
        except:     # pylint: disable=bare-except
            return None

    @staticmethod
    def _bn_op_mean_struct_3(bn_op: tf.Operation) -> Union[tf.Operation, None]:
        """
        Return moving_mean op corresponding to batchnorm with training=False.

        :param bn_op: bn_op obtained from connected graph using get_modules
        a mul_1 op inside BN scope.
        :return: Read operation for moving_mean
        """
        try:
            mul_op = bn_op.inputs[1].op
            assert mul_op.type == 'Mul'
            mul_2_op = mul_op.outputs[0].consumers()[1]
            assert mul_2_op.type == 'Mul'
            read_op = mul_2_op.inputs[0].op
            assert read_op.type in ['ReadVariableOp']
            return read_op
        except:     # pylint: disable=bare-except
            return None

    @staticmethod
    def get_moving_mean_as_read_op(bn_op: tf.Operation) -> Union[tf.Operation, None]:
        """
        Get moving mean read op from BN op specified.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: moving mean read op
        """
        if bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm']:
            assert len(bn_op.inputs) == 5
            moving_mean_read = bn_op.inputs[constants.BN_OP_PARAM_INDICES['movingmean']].op
            if moving_mean_read.type == 'Switch':      # tf slim bn using training tensor form
                moving_mean_read = moving_mean_read.inputs[0].op
                assert 'read' in moving_mean_read.name
        elif bn_op.type in ['Mul']:
            # For regular BN
            # mul_1 << - mul --> mul_2 <-- cond/merge <-- switch2 <-- moving mean read < moving mean tensor
            # inputs[1] is mul .op.inputs[1] is gamma:read op whose input is gamma tensor as variable v2

            # register handlers for different structures
            bn_op_struct_for_mean_handlers = [BNUtils._bn_op_mean_struct_1,
                                              BNUtils._bn_op_mean_struct_2,
                                              BNUtils._bn_op_mean_struct_3]

            moving_mean_read = None
            # try all handlers available
            for handler in bn_op_struct_for_mean_handlers:
                if moving_mean_read is None:
                    moving_mean_read = handler(bn_op)
                else:
                    break
            assert moving_mean_read is not None, _BN_STRUCTURE_ERROR_MSG

        elif bn_op.type in ['Identity']:
            assert len(bn_op.inputs) == 1
            cond = bn_op.inputs[0].op
            moving_mean_read_tensor = cond.inputs[3]
            moving_mean_read = [node for node in cond.graph.get_operations() if moving_mean_read_tensor in node.inputs][2]
        else:
            logger.error("Error, unknown BN op")
            assert False

        if moving_mean_read.type == 'Identity':
            assert len(moving_mean_read.inputs) == 1, _BN_STRUCTURE_ERROR_MSG

        assert moving_mean_read.type in ['ReadVariableOp', 'Const', 'Identity', 'VarHandleOp']
        return moving_mean_read

    @staticmethod
    def _get_moving_mean_read_var_op_tensor_using_structure(bn_op: tf.Operation) -> tf.Tensor:
        """
        Get moving mean readVariableOp tensor from BN op specified.

        :param bn_op: FusedBatchNorm as tf.Operation
        :return: tensor associated with bn op moving mean readVariableOp type, as tf.Tensor
        """

        assert bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm', 'Mul', 'Identity']
        moving_mean_read_tensor = BNUtils.get_moving_mean_as_read_op(bn_op).outputs[0]
        assert moving_mean_read_tensor is not None

        if moving_mean_read_tensor.op.type == 'Const':
            logger.debug("BN op has const type op for moving variance")
            # get the read var type from bn op
            # get the sub_1 op associated with moving mean read op
            assert len(bn_op.outputs) > 1
            moving_avg_sub_1 = bn_op.outputs[1].consumers()[0]
            all_inputs = moving_avg_sub_1.inputs

            # among inputs figure out the read var op type that can be "evaluated"
            for input_t in all_inputs:
                if input_t.op.type == 'ReadVariableOp':
                    moving_mean_read_tensor = input_t
                elif input_t.op.type == 'Identity' and 'read:0' in input_t.name:      # tf slim form
                    moving_mean_read_tensor = input_t

        elif moving_mean_read_tensor.op.inputs and moving_mean_read_tensor.op.inputs[0].op.type == 'Switch':
            logger.debug("Fetch moving var from a trainable BN op structure")
            moving_mean_read_tensor = BNUtils._get_tensor_read_var_op_trainable_bn_op(moving_mean_read_tensor)

        return moving_mean_read_tensor

    @staticmethod
    def get_moving_mean_read_var_op_tensor(graph: tf.Graph, bn_op: tf.Operation) -> tf.Tensor:
        """
        Get moving mean readVariableOp tensor from BN op specified.

        :param graph: TensorFlow graph
        :param bn_op: FusedBatchNorm as tf.Operation
        :return: tensor associated with bn op moving mean readVariableOp type, as tf.Tensor
        """
        try:
            # try name based tensor look up for Keras layers
            moving_mean_read_tensor = BNUtils._get_bn_param_tensor_using_name(graph, bn_op,
                                                                              constants.BNOpParamType.moving_mean)
        except KeyError:
            # if we can't find the tensor name, use structure match
            # to figure out the read tensor for param
            moving_mean_read_tensor = BNUtils._get_moving_mean_read_var_op_tensor_using_structure(bn_op)

        return moving_mean_read_tensor

    @staticmethod
    def get_moving_mean_as_numpy_data(sess: tf.compat.v1.Session, bn_op: tf.Operation) -> np.ndarray:
        """
        Get moving mean param from BN op specified.

        :param sess: tensorflow session
        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: moving mean as numpy data
        """

        moving_mean_tensor = BNUtils.get_moving_mean_read_var_op_tensor(sess.graph, bn_op)

        with sess.graph.as_default():
            numpy_data = sess.run(moving_mean_tensor)

        return numpy_data

    @staticmethod
    def get_epsilon(bn_op: tf.Operation) -> float:
        """
        Returns epsilon extracted from given bn op.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: epsilon value
        """

        if bn_op.type in ['Mul']:
            assert len(bn_op.inputs) >= 2, _BN_STRUCTURE_ERROR_MSG
            mul = bn_op.inputs[1].op
            assert len(mul.inputs) >= 1, _BN_STRUCTURE_ERROR_MSG
            rsqrt = mul.inputs[0].op
            assert len(rsqrt.inputs) >= 1, _BN_STRUCTURE_ERROR_MSG
            add = rsqrt.inputs[0].op
            assert len(add.inputs) >= 2, _BN_STRUCTURE_ERROR_MSG
            epsilon = add.inputs[1].op
            numpy_epsilon = epsilon.get_attr('value').float_val[0]
        elif bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm']:
            # epsilon can be derived as attribute value
            numpy_epsilon = bn_op.get_attr("epsilon")
        else:
            raise RuntimeError(f"Unknown BN op type: {bn_op.type}")

        return numpy_epsilon

    @staticmethod
    def get_assign_moving_avg_op(bn_op: tf.Operation) -> Union[tf.Operation, None]:
        """
        Get assign_moving_avg op corresponding with the bn_op, if it exists.

        :param bn_op: Batchnorm op to search for corresponding assign_moving_avg op
        :return: assign_moving_op corresponding with the bn op, or None if it does not exist.
        """
        assert (bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm'] and len(bn_op.outputs) in [5, 6]) \
               or (bn_op.type == 'Identity' and len(bn_op.outputs) == 1)

        assign_moving_avg_op = None
        if bn_op.type in 'Identity':
            if_op = bn_op.inputs[0].op
            assert if_op.type == 'If'
            identity_1_op = if_op.outputs[1].consumers()[0]
            assert identity_1_op.type == 'Identity'
            sub_op = identity_1_op.outputs[0].consumers()[0]
            assert sub_op.type == 'Sub'
            mul_op = sub_op.outputs[0].consumers()[0]
            assert mul_op.type == 'Mul'
            assign_moving_avg_op = mul_op.outputs[0].consumers()[0]
            assert assign_moving_avg_op.type in ['AssignSub', 'AssignSubVariableOp']
        elif bn_op.outputs[1].consumers():
            child_op = bn_op.outputs[1].consumers()[0]
            if child_op.type == 'Merge':
                sub_op = child_op.outputs[0].consumers()[0]
            else:
                sub_op = child_op
            assert sub_op.type == 'Sub'
            mul_op = sub_op.outputs[0].consumers()[0]
            assert mul_op.type == 'Mul'
            assign_moving_avg_op = mul_op.outputs[0].consumers()[0]
            assert assign_moving_avg_op.type in ['AssignSub', 'AssignSubVariableOp']
        return assign_moving_avg_op

    @staticmethod
    def get_assign_moving_avg_1_op(bn_op: tf.Operation) -> Union[tf.Operation, None]:
        """
        Get assign_moving_avg_1 op corresponding with the bn_op, if it exists.

        :param bn_op: Batchnorm op to search for corresponding assign_moving_avg_1 op
        :return: assign_moving_avg_1 corresponding with the bn op, or None if it does not exist.
        """
        assert (bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm'] and len(bn_op.outputs) in [5, 6]) \
               or (bn_op.type == 'Identity' and len(bn_op.outputs) == 1)

        assign_moving_avg_op = None
        if bn_op.type in 'Identity':
            if_op = bn_op.inputs[0].op
            assert if_op.type == 'If'
            identity_2_op = if_op.outputs[2].consumers()[0]
            assert identity_2_op.type == 'Identity'
            sub_op = identity_2_op.outputs[0].consumers()[0]
            assert sub_op.type == 'Sub'
            mul_op = sub_op.outputs[0].consumers()[0]
            assert mul_op.type == 'Mul'
            assign_moving_avg_op = mul_op.outputs[0].consumers()[0]
            assert assign_moving_avg_op.type in ['AssignSub', 'AssignSubVariableOp']
        elif bn_op.outputs[2].consumers():
            child_op = bn_op.outputs[2].consumers()[0]
            if child_op.type == 'Merge':
                sub_op = child_op.outputs[0].consumers()[0]
            else:
                sub_op = child_op
            assert sub_op.type == 'Sub'
            mul_op = sub_op.outputs[0].consumers()[0]
            assert mul_op.type == 'Mul'
            assign_moving_avg_op = mul_op.outputs[0].consumers()[0]
            assert assign_moving_avg_op.type in ['AssignSub', 'AssignSubVariableOp']
        return assign_moving_avg_op

    @staticmethod
    def remove_bn_op_from_update_ops(sess: tf.compat.v1.Session, bn_op: tf.Operation):
        """
        Remove batchnorm assign_moving_avg and assign_moving_avg_1 ops from update ops.

        :param sess: tf.compat.v1.Session
        :param bn_op: BatchNorm operation whose assign_moving_avg and assign_moving_avg_1 ops should be removed.
        """
        with sess.graph.as_default():
            update_ops = tf.compat.v1.get_collection_ref(tf.compat.v1.GraphKeys.UPDATE_OPS)
            assign_moving_avg_op = BNUtils.get_assign_moving_avg_op(bn_op)
            assign_moving_avg_op_1 = BNUtils.get_assign_moving_avg_1_op(bn_op)
            if assign_moving_avg_op and assign_moving_avg_op in update_ops:
                update_ops.remove(assign_moving_avg_op)
                logger.debug('Removed %s from update ops', assign_moving_avg_op.name)
            if assign_moving_avg_op_1 and assign_moving_avg_op_1 in update_ops:
                update_ops.remove(assign_moving_avg_op_1)
                logger.debug('Removed %s from update ops', assign_moving_avg_op_1.name)

    @staticmethod
    def _get_bn_param_tensor_using_name(graph: tf.Graph, bn_op: tf.Operation, param_type: constants.BNOpParamType):
        """
        Helper to get BN op param read tensor.

        :param graph: TensorFlow graph
        :param bn_op: BN op from which param read tensor is to be extracted
        :param param_type: param type for which param tensor is to be extracted, as constants.BNOpParamType (supported
            types are beta, gamma, moving_mean or moving_variance)
        :return: param read tensor
        """

        if param_type not in vars(constants.BNOpParamType).values():
            assert 0, 'Error, get_bn_param_using_name() invalid param type requested'

        # name of the fused bn contains bn_name/FusedBatchNormV3 or
        # bn_name/cond/FusedBatchNormV3_1
        # we need only the bn_name to make param tensor names
        op_name = bn_op.name.split('/')[0]
        param_tensor_name = op_name + constants.BN_OP_PARAM_NAME_SUFFIX[param_type]
        param_tensor = graph.get_tensor_by_name(param_tensor_name)

        return param_tensor

    @staticmethod
    def _bn_op_momentum_struct_1(bn_op: tf.Operation) -> Union[float, None]:
        """
        Return momentum value corresponding to batchnorm with training tensor.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: momentum value
        """
        try:
            mul_op = bn_op.inputs[1].op
            assert mul_op.type == 'Mul'
            mul_2_op = mul_op.outputs[0].consumers()[1]
            assert mul_2_op.type == 'Mul'
            merge_op = mul_2_op.inputs[0].op
            assert merge_op.type == 'Merge'
            switch_1_op = merge_op.outputs[0].consumers()[0]
            assert switch_1_op.type == 'Switch'
            sub_op = switch_1_op.outputs[1].consumers()[0]
            assert sub_op.type == 'Sub'
            assign_moving_avg_mul_op = sub_op.outputs[0].consumers()[0]
            assert assign_moving_avg_mul_op.type == 'Mul'
            decay_op = assign_moving_avg_mul_op.inputs[1].op
            assert decay_op.type == 'Const'
            decay = decay_op.get_attr('value').float_val[0]
            return 1 - decay
        except:     # pylint: disable=bare-except
            return None

    @staticmethod
    def _bn_op_momentum_struct_2(bn_op: tf.Operation) -> Union[float, None]:
        """
        Return momentum value corresponding to batchnorm with training=True.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: momentum value
        """
        try:
            mul_op = bn_op.inputs[1].op
            assert mul_op.type == 'Mul'
            mul_2_op = mul_op.outputs[0].consumers()[1]
            assert mul_2_op.type == 'Mul'
            squeeze_op = mul_2_op.inputs[0].op
            assert squeeze_op.type == 'Squeeze'
            sub_op = squeeze_op.outputs[0].consumers()[0]
            assert sub_op.type == 'Sub'
            assign_moving_avg_mul_op = sub_op.outputs[0].consumers()[0]
            assert assign_moving_avg_mul_op.type == 'Mul'
            decay_op = assign_moving_avg_mul_op.inputs[1].op
            assert decay_op.type == 'Const'
            decay = decay_op.get_attr('value').float_val[0]
            return 1 - decay
        except:     # pylint: disable=bare-except
            return None

    @staticmethod
    def _fused_bn_op_momentum_struct_1(bn_op: tf.Operation) -> Union[float, None]:
        """
        Return momentum value corresponding to fused batchnorm with training tensor.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: momentum value
        """
        try:
            merge_1_op = bn_op.outputs[1].consumers()[0]
            assert merge_1_op.type == 'Merge'
            sub_op = merge_1_op.outputs[0].consumers()[0]
            assert sub_op.type == 'Sub'
            mul_op = sub_op.outputs[0].consumers()[0]
            assert mul_op.type == 'Mul'
            sub_2_op = mul_op.inputs[1].op
            assert sub_2_op.type == 'Sub'
            merge_op = sub_2_op.inputs[1].op
            assert merge_op.type == 'Merge'
            decay_op = merge_op.inputs[1].op
            assert decay_op.type == 'Const'
            decay = decay_op.get_attr('value').float_val[0]
            return decay
        except:     # pylint: disable=bare-except
            return None

    @staticmethod
    def _fused_bn_op_momentum_struct_2(bn_op: tf.Operation) -> Union[float, None]:
        """
        Return momentum value corresponding to fused batchnorm with training=True.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: momentum value
        """
        try:
            sub_op = bn_op.outputs[1].consumers()[0]
            assert sub_op.type == 'Sub'
            mul_op = sub_op.outputs[0].consumers()[0]
            assert mul_op.type == 'Mul'
            sub_2_op = mul_op.inputs[1].op
            assert sub_2_op.type == 'Sub'
            decay_op = sub_2_op.inputs[1].op
            assert decay_op.type == 'Const'
            decay = decay_op.get_attr('value').float_val[0]
            return decay
        except:     # pylint: disable=bare-except
            return None

    @staticmethod
    def _fused_bn_op_cond_momentum_struct_1(bn_op: tf.Operation) -> Union[float, None, tf.Variable]:
        """
        Return momentum value corresponding to fused batchnorm with training=Variable

        :param bn_op: bn_op obtained from connected graph using get_modules a Identity op inside BN scope.
        :return: momentum value
        """
        try:
            cond_1_identity_op = BNUtils.get_cond_1_identity_op(bn_op)
            cond_1_op = cond_1_identity_op.inputs[0].op
            assert cond_1_op.type == 'If'
            decay_op = cond_1_op.inputs[1].op
            if decay_op.type == 'Const':
                decay = decay_op.get_attr('value').float_val[0]
            else:
                decay = decay_op
            return decay
        except:     # pylint: disable=bare-except
            return None

    @staticmethod
    def get_momentum(bn_op: tf.Operation) -> Union[float, tf.Variable]:
        """
        Returns momentum extracted from given bn op.  If bn op is training=False mode, momentum will be none.

        :param bn_op: bn_op obtained from connected graph using get_modules a mul_1 op inside BN scope.
        :return: momentum value
        """
        # register handlers for different structures
        bn_op_struct_for_momentum_handlers = [BNUtils._bn_op_momentum_struct_1,
                                              BNUtils._bn_op_momentum_struct_2]
        fused_bn_op_struct_for_momentum_handlers = [BNUtils._fused_bn_op_momentum_struct_1,
                                                    BNUtils._fused_bn_op_momentum_struct_2]
        fused_bn_op_cond_struct_for_momentum_handlers = [BNUtils._fused_bn_op_cond_momentum_struct_1]

        decay = None
        if bn_op.type in ['Mul']:
            # try all handlers available
            for handler in bn_op_struct_for_momentum_handlers:
                decay = handler(bn_op)
                if decay is not None:
                    break
        elif bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm']:
            # try all handlers available
            for handler in fused_bn_op_struct_for_momentum_handlers:
                decay = handler(bn_op)
                if decay is not None:
                    break
        elif bn_op.type in ['Identity']:
            for handler in fused_bn_op_cond_struct_for_momentum_handlers:
                decay = handler(bn_op)
                if decay is not None:
                    break
        else:
            logger.error("Error, unknown BN op")
            assert False
        return decay

    @staticmethod
    def get_training(bn_op: tf.Operation) -> Union[None, bool, tf.Tensor, tf.Variable]:
        """
        Returns either a boolean of whether the BN op training mode is True or False, or the is_training tensor
        feeding into the BN op if it is using a tensor to determine the mode dynamically.
        :param bn_op: bn_op obtained in the connected graph
        :return: True or False for training mode, or tf.Tensor that determines the mode dynamically.
        """
        assert bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm', 'Mul', 'Identity']

        if bn_op.type in ['FusedBatchNormV3', 'FusedBatchNorm', 'Identity']:
            if 'FusedBatchNormV3_1' in bn_op.name:
                switch_op = bn_op.inputs[0].op
                pred_id_op = switch_op.inputs[1].op
                training = pred_id_op.inputs[0]
            elif bn_op.type == 'Identity':
                cond_op = bn_op.inputs[0].op
                training = cond_op.inputs[0]
            else:
                training = bn_op.get_attr('is_training')
            return training

        # Non fused batchnorm case
        mul_op = bn_op.inputs[1].op
        assert mul_op.type == 'Mul'
        rsqrt_op = mul_op.inputs[0].op
        assert rsqrt_op.type == 'Rsqrt'
        add_op = rsqrt_op.inputs[0].op
        assert add_op.type == 'AddV2'
        add_input_op = add_op.inputs[0].op
        if add_input_op.type == 'Squeeze':
            return True
        if add_input_op.type == 'ReadVariableOp':
            return False
        if add_input_op.type == 'Merge':
            switch_op = add_input_op.inputs[1].op
            assert switch_op.type == 'Switch'
            pred_id_op = switch_op.inputs[1].op
            assert pred_id_op.type == 'Identity'
            return pred_id_op.inputs[0]
        logger.error('Error, unknown BN structure')
        return None

    @staticmethod
    def get_cond_1_identity_op(bn_op: tf.Operation) -> tf.Operation:
        """
        Returns cond_1 Identity op of bn when training is variable

        :param bn_op: bn_op obtained from connected graph using get_modules a Identity op inside BN scope.
        :return: cond_1 Identity op
        """
        assert bn_op.type == 'Identity'

        assign_moving_avg = BNUtils.get_assign_moving_avg_op(bn_op)
        mul_op = assign_moving_avg.inputs[1].op
        assert mul_op.type == 'Mul'
        sub_op = mul_op.inputs[1].op
        assert sub_op.type == 'Sub'
        cond_1_identity_op = sub_op.inputs[1].op
        assert cond_1_identity_op.type == 'Identity'

        # Make sure fetched op isn't cond op
        assert bn_op != cond_1_identity_op

        return cond_1_identity_op
