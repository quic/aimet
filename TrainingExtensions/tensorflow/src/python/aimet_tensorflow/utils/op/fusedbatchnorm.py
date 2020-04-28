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
""" utilities for fused batchnorm op """

import tensorflow as tf
from tensorflow.contrib import graph_editor as ge
from aimet_common.utils import AimetLogger
from aimet_tensorflow.utils import constants

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

_BN_STRUCTURE_ERROR_MSG = "BN op doesn't have the expected structure"

class BNUtils:
    """ Batch Norm/ fused Batch Norm op related utils"""

    @staticmethod
    def skip_bn_op(sess: tf.Session, bn_op: tf.Operation, in_tensor: tf.Tensor, out_tensor: tf.Tensor):
        """
        skip given bn op specified (fused batch norm op)
        Note : supports only Fused bn op types.
        """

        if in_tensor is None or out_tensor is None:
            logger.error("Error, input and output tensors must be provided for skipping the op")
            assert False
        else:
            with sess.graph.as_default():
                if bn_op.type in ['FusedBatchNormV3']:
                    ge.detach_outputs(in_tensor.op)
                    ge.reroute_ts(in_tensor, out_tensor)
                else:
                    logger.error("Error, Unknown BN op")
                    assert False

    @staticmethod
    def _get_tensor_read_var_op_trainable_bn_op(input_tensor: tf.Tensor) -> tf.Tensor:
        """
         generic helper to find a read op tensor associated with input tensor
         that can be evaluated, when the bn op is marked trainable.
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
    def get_beta_read_op(bn_op: tf.Operation):
        """
        get beta read op from BN op specified
        :param bn_op: bn_op obtained from connected graph using get_modules
         (is mul_1 op inside BN scope)
        :return: beta read op
        """
        # only support fused BN
        assert bn_op.type == 'FusedBatchNormV3'
        assert len(bn_op.inputs) == 5

        beta_read = bn_op.inputs[constants.BN_OP_PARAM_INDICES['beta']].op

        assert beta_read.type == 'ReadVariableOp'
        return beta_read

    @staticmethod
    def get_beta_read_var_op_tensor(bn_op: tf.Operation)->tf.Tensor:
        """
        get beta readVariableOp tensor from BN op specified
        :param bn_op: FusedBatchNorm as tf.Operation
        :return: tensor associated with bn op beta readVariableOp type, as tf.Tensor
        """
        # only support fused BN
        assert bn_op.type == 'FusedBatchNormV3'
        assert len(bn_op.inputs) == 5
        beta_read_tensor = bn_op.inputs[constants.BN_OP_PARAM_INDICES['beta']]

        assert beta_read_tensor is not None

        if beta_read_tensor.op.inputs[0].op.type == 'Switch':
            logger.debug('Fetching params from trainable BN op type')
            beta_read_tensor = BNUtils._get_tensor_read_var_op_trainable_bn_op(beta_read_tensor)
        elif beta_read_tensor.op.type == 'Switch':      # tf slim bn using training tensor form
            beta_read_tensor = beta_read_tensor.op.inputs[0]
            assert 'read:0' in beta_read_tensor.name

        return beta_read_tensor

    @staticmethod
    def get_beta_as_numpy_data(sess: tf.Session, bn_op: tf.Operation):
        """
        get beta param from BN op specified
        :param sess: tensorflow session
        :param bn_op: bn_op as tf.Operation
        :return: beta tensor as numpy data
        """
        try:
            # try name based tensor look up for Keras layers
            beta_tensor = BNUtils._get_bn_param_tensor_using_name(sess, bn_op,
                                                                  constants.BNOpParamType.beta)
        except KeyError:
            # if we can't find the tensor name, use structure match
            # to figure out the read tensor for param
            beta_tensor = BNUtils.get_beta_read_var_op_tensor(bn_op)

        with sess.graph.as_default():
            numpy_data = sess.run(beta_tensor)
        return numpy_data

    @staticmethod
    def get_gamma_as_read_op(bn_op: tf.Operation):
        """
        get gamma read op from BN op specified
        :param bn_op: bn_op obtained from connected graph using get_modules
        (is mul_1 op inside BN scope)
        :return: gamma read op
        """

        # only support fused BN
        assert bn_op.type == 'FusedBatchNormV3'
        assert len(bn_op.inputs) == 5

        gamma_read = bn_op.inputs[constants.BN_OP_PARAM_INDICES['gamma']].op

        return gamma_read

    @staticmethod
    def get_gamma_read_var_op_tensor(bn_op: tf.Operation)->tf.Tensor:
        """

        :param bn_op:
        :return:
        """
        # only support fused BN
        assert bn_op.type == 'FusedBatchNormV3'
        assert len(bn_op.inputs) == 5
        gamma_read_tensor = bn_op.inputs[constants.BN_OP_PARAM_INDICES['gamma']]
        assert gamma_read_tensor is not None

        if gamma_read_tensor.op.inputs and gamma_read_tensor.op.inputs[0].op.type == 'Switch':
            logger.debug('Fetching params from trainable BN op type')
            gamma_read_tensor = BNUtils._get_tensor_read_var_op_trainable_bn_op(gamma_read_tensor)
        # tf slim bn using training tensor form, or regular bn with const gamma
        elif gamma_read_tensor.op.type == 'Switch':
            gamma_read_tensor = gamma_read_tensor.op.inputs[0]
            assert 'read:0' in gamma_read_tensor.name or gamma_read_tensor.op.type == 'Const'

        return gamma_read_tensor

    @staticmethod
    def get_gamma_as_numpy_data(sess: tf.Session, bn_op: tf.Operation):
        """
        get gamma param from BN op specified
        :param sess: tensorflow session
        :param bn_op: bn_op obtained from connected graph using get_modules
        (is mul_1 op inside BN scope)
        :return: gamma as numpy data
        """
        try:
            # try name based tensor look up for Keras layers
            gamma_tensor = BNUtils._get_bn_param_tensor_using_name(sess, bn_op,
                                                                   constants.BNOpParamType.gamma)
        except KeyError:
            # if we can't find the tensor name, use structure match
            # to figure out the read tensor for param
            gamma_tensor = BNUtils.get_gamma_read_var_op_tensor(bn_op)

        with sess.graph.as_default():
            numpy_data = sess.run(gamma_tensor)

        return numpy_data

    @staticmethod
    def get_moving_variance_as_read_op(bn_op: tf.Operation):
        """
        get moving variance read op from BN op specified
        :param bn_op: bn_op obtained from connected graph using get_modules
        a mul_1 op inside BN scope.
        :return: moving variance as read op
        """

        # only support fused BN
        assert bn_op.type == 'FusedBatchNormV3'
        assert len(bn_op.inputs) == 5

        moving_var_read = bn_op.inputs[constants.BN_OP_PARAM_INDICES['movingvariance']].op
        assert moving_var_read.type == 'ReadVariableOp'

        return moving_var_read

    @staticmethod
    def get_moving_variance_read_var_op_tensor(bn_op: tf.Operation)->tf.Tensor:
        """
        get moving variance readVariableOp tensor from BN op specified
        :param bn_op: FusedBatchNorm as tf.Operation
        :return: tensor associated with bn op moving variance readVariableOp type, as tf.Tensor
        """
        # only support fused BN
        assert bn_op.type == 'FusedBatchNormV3'
        assert len(bn_op.inputs) == 5
        moving_var_read_tensor = bn_op.inputs[constants.BN_OP_PARAM_INDICES['movingvariance']]
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

        elif moving_var_read_tensor.op.inputs[0].op.type == 'Switch':
            logger.debug("Fetch moving var from a trainable BN op structure")
            moving_var_read_tensor = BNUtils._get_tensor_read_var_op_trainable_bn_op(moving_var_read_tensor)

        elif moving_var_read_tensor.op.type == 'Switch':      # tf slim bn using training tensor form
            moving_var_read_tensor = moving_var_read_tensor.op.inputs[0]
            assert 'read:0' in moving_var_read_tensor.name

        return moving_var_read_tensor

    @staticmethod
    def get_moving_variance_as_numpy_data(sess: tf.Session, bn_op: tf.Operation):
        """
        get moving variance param from BN op specified
        :param sess: tensorflow session
        :param bn_op: bn_op obtained from connected graph using get_modules
        a mul_1 op inside BN scope.
        :return: moving variance as numpy data
        """

        try:
            # try name based tensor look up for Keras layers
            moving_var_tensor = BNUtils._get_bn_param_tensor_using_name(sess, bn_op,
                                                                        constants.BNOpParamType.moving_variance)
        except KeyError:
            # if we can't find the tensor name, use structure match
            # to figure out the read tensor for param
            moving_var_tensor = BNUtils.get_moving_variance_read_var_op_tensor(bn_op)

        with sess.graph.as_default():
            numpy_data = sess.run(moving_var_tensor)
        return numpy_data

    @staticmethod
    def get_moving_mean_as_read_op(bn_op: tf.Operation):
        """
        get moving mean read op from BN op specified
        :param bn_op: bn_op obtained from connected graph using get_modules
        a mul_1 op inside BN scope.
        :return: moving mean read op
        """
        # only support fused BN
        assert bn_op.type == 'FusedBatchNormV3'
        assert len(bn_op.inputs) == 5

        moving_mean_read = bn_op.inputs[constants.BN_OP_PARAM_INDICES['movingmean']].op
        assert moving_mean_read.type == 'ReadVariableOp'
        return moving_mean_read

    @staticmethod
    def get_moving_mean_read_var_op_tensor(bn_op: tf.Operation)->tf.Tensor:
        """
        get moving mean readVariableOp tensor from BN op specified
        :param bn_op: FusedBatchNorm as tf.Operation
        :return: tensor associated with bn op moving mean readVariableOp type, as tf.Tensor
        """
        # only support fused BN
        assert bn_op.type == 'FusedBatchNormV3'
        assert len(bn_op.inputs) == 5
        moving_mean_read_tensor = bn_op.inputs[constants.BN_OP_PARAM_INDICES['movingmean']]
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

        elif moving_mean_read_tensor.op.inputs[0].op.type == 'Switch':
            logger.debug("Fetch moving var from a trainable BN op structure")
            moving_mean_read_tensor = BNUtils._get_tensor_read_var_op_trainable_bn_op(moving_mean_read_tensor)

        elif moving_mean_read_tensor.op.type == 'Switch':      # tf slim bn using training tensor form
            moving_mean_read_tensor = moving_mean_read_tensor.op.inputs[0]
            assert 'read:0' in moving_mean_read_tensor.name

        return moving_mean_read_tensor

    @staticmethod
    def get_moving_mean_as_numpy_data(sess: tf.Session, bn_op: tf.Operation):
        """
        get moving mean param from BN op specified
        :param sess: tensorflow session
        :param bn_op: bn_op obtained from connected graph using get_modules
        a mul_1 op inside BN scope.
        :return: moving mean as numpy data
        """

        try:
            # try name based tensor look up for Keras layers
            moving_mean_tensor = BNUtils._get_bn_param_tensor_using_name(sess, bn_op,
                                                                         constants.BNOpParamType.moving_mean)
        except KeyError:
            # if we can't find the tensor name, use structure match
            # to figure out the read tensor for param
            moving_mean_tensor = BNUtils.get_moving_mean_read_var_op_tensor(bn_op)

        with sess.graph.as_default():
            numpy_data = sess.run(moving_mean_tensor)
        return numpy_data

    @staticmethod
    def get_epsilon(bn_op: tf.Operation):
        """
        Returns epsilon extracted from given bn ip
        :param bn_op: bn_op obtained from connected graph using get_modules
        a mul_1 op inside BN scope.
        :return: epsilon value
        """

        # only support fused BNs
        assert bn_op.type == 'FusedBatchNormV3'
        # epsilon can be derived as attribute value
        numpy_epsilon = bn_op.get_attr("epsilon")

        return numpy_epsilon

    @staticmethod
    def _get_bn_param_tensor_using_name(sess: tf.Session, bn_op: tf.Operation, param_type: constants.BNOpParamType):
        """
        Helper to get BN op param read tensor
        :param sess: TensorFlow session tf.Session
        :param bn_op: BN op from which param read tensor is to be extracted
        :param param_type: param type for which param tensor is to be
        extracted, as constants.BNOpParamType (supported types are beta,
        gamma, moving_mean or moving_variance)
        :return: param read tensor
        """

        if param_type not in vars(constants.BNOpParamType).values():
            assert 0, 'Error, get_bn_param_using_name() invalid param type requested'

        # name of the fused bn contains bn_name/FusedBatchNormV3 or
        # bn_name/cond/FusedBatchNormV3_1
        # we need only the bn_name to make param tensor names
        op_name = bn_op.name.split('/')[0]
        param_tensor_name = op_name + constants.BN_OP_PARAM_NAME_SUFFIX[param_type]
        param_tensor = sess.graph.get_tensor_by_name(param_tensor_name)

        return param_tensor
