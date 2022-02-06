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

""" Handler functions for reducing different types of modules. """

from typing import Tuple, List, Union
import struct
import numpy as np
import tensorflow as tf

from aimet_common.winnow.mask import Mask
from aimet_common.winnow.winnow_utils import get_zero_positions_in_binary_mask
from aimet_common.utils import AimetLogger
from aimet_tensorflow.common.operation import Op
from aimet_tensorflow.common.product import Product
from aimet_tensorflow.utils.op.conv import WeightTensorUtils, BiasUtils
from aimet_tensorflow.utils.op.fusedbatchnorm import BNUtils

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Winnow)


# pylint: disable=too-many-locals
def reduce_conv2d(sess: tf.compat.v1.Session,
                  op_tensor_tuple: Tuple[Op, List[tf.Tensor]], op_mask) -> (str, tf.Operation, tf.Operation):
    """
    Conv2D module reducer
    :param sess: current tf.compat.v1.Session
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    :param op_mask: Mask containing information on input and output channels to winnow
    """

    tf_op = op_tensor_tuple[0].get_module()
    padding = str(tf_op.get_attr("padding"), "utf-8")
    strides = tf_op.get_attr("strides")
    # Remove last part of conv op's name, or else we end up with names like Conv2D/Conv2D/Conv2D... if the same conv op
    # is reduced multiple times
    last_slash = op_tensor_tuple[0].dotted_name.rfind('/')
    name = "reduced_" + op_tensor_tuple[0].dotted_name[:last_slash]
    kernel_product = op_tensor_tuple[0].get_param_product('kernel')
    kernel_size = kernel_product.shape.as_list()[:2]

    # Depthwise Conv2d always has output dim of 1.  Only slice out input channel dimensions.
    if op_tensor_tuple[0].type == 'DepthwiseConv2dNative':
        # Check to make sure input channel and output channel sizes are identical
        # Format is expected to be NHWC, so channels is the last index in shape array
        if tf_op.inputs[0].shape.as_list()[-1] != tf_op.outputs[0].shape.as_list()[-1]:
            raise NotImplementedError('Reducing depthwise conv2d with differing input and output channel sizes not '
                                      'supported')
        output_dim = None
    else:
        output_dim = 3

    reduced_weights = _get_reduced_params(sess=sess,
                                          product=kernel_product,
                                          mask=op_mask,
                                          input_dim=2,
                                          output_dim=output_dim)
    reduced_weights_init = tf.compat.v1.constant_initializer(reduced_weights, verify_shape=True)
    bias_product = op_tensor_tuple[0].get_param_product('bias')
    reduced_bias = None
    if bias_product:
        use_bias = True
        reduced_bias = _get_reduced_params(sess=sess,
                                           product=bias_product,
                                           mask=op_mask,
                                           input_dim=None,
                                           output_dim=0)
        reduced_bias_init = tf.compat.v1.constant_initializer(reduced_bias, verify_shape=True)
    else:
        use_bias = False
        reduced_bias_init = 'zeros'

    output_ch_masks = op_mask.output_channel_masks
    output_ch_indices_to_reduce = get_zero_positions_in_binary_mask(output_ch_masks[0])
    filters = len(output_ch_masks[0]) - len(output_ch_indices_to_reduce)

    # Check for regularization in the kernel
    kernel_tensor = kernel_product.tensor_dict[kernel_product.consumers[0]]
    kernel_regularizer = _get_kernel_regularizer(kernel_tensor)

    if op_tensor_tuple[0].type == 'DepthwiseConv2dNative':
        new_tensor = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                                     strides=strides[1:3],
                                                     padding=padding,
                                                     use_bias=use_bias,
                                                     bias_initializer=reduced_bias_init,
                                                     kernel_initializer=reduced_weights_init,
                                                     kernel_regularizer=kernel_regularizer,
                                                     name=name)(op_tensor_tuple[1][0])
    else:
        new_tensor = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=strides[1:3],
                                            padding=padding,
                                            use_bias=use_bias,
                                            bias_initializer=reduced_bias_init,
                                            kernel_initializer=reduced_weights_init,
                                            kernel_regularizer=kernel_regularizer,
                                            name=name)(op_tensor_tuple[1][0])
    module = new_tensor.op
    if use_bias:
        module = module.inputs[0].op
    WeightTensorUtils.update_tensor_for_op(sess, module, reduced_weights)
    if use_bias:
        BiasUtils.update_bias_for_op(sess, module, reduced_bias)

    return name, new_tensor.op, module


def reduce_maxpool(sess: tf.compat.v1.Session,
                   op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _) -> (str, tf.Operation, tf.Operation):
    """
    Maxpool module reducer
    :param sess: current tf.compat.v1.Session
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    """

    tf_op = op_tensor_tuple[0].get_module()
    padding = str(tf_op.get_attr("padding"), "utf-8")
    strides = tf_op.get_attr("strides")[1:3]
    ksize = tf_op.get_attr("ksize")[1:3]
    name = "reduced_" + op_tensor_tuple[0].dotted_name

    # Hardcoding [:-8] below to remove '/MaxPool' from the dotted name
    # Otherwise this leads to a doubly nested op
    new_tensor = tf.keras.layers.MaxPool2D(pool_size=ksize,
                                           strides=strides,
                                           padding=padding,
                                           name=name[:-8])(op_tensor_tuple[1][0])
    module = sess.graph.get_operation_by_name(name)

    return name, new_tensor.op, module


def reduce_avgpool(sess: tf.compat.v1.Session,
                   op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _) -> (str, tf.Operation, tf.Operation):
    """
    Avgpool module reducer
    :param sess: current tf.compat.v1.Session
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    """

    tf_op = op_tensor_tuple[0].get_module()
    padding = str(tf_op.get_attr("padding"), "utf-8")
    strides = tf_op.get_attr("strides")[1:3]
    ksize = tf_op.get_attr("ksize")[1:3]
    name = "reduced_" + op_tensor_tuple[0].dotted_name

    # Hardcoding [:-8] below to remove '/AvgPool' from the dotted name
    # Otherwise this leads to a doubly nested op
    new_tensor = tf.keras.layers.AvgPool2D(pool_size=ksize,
                                           strides=strides,
                                           padding=padding,
                                           name=name[:-8])(op_tensor_tuple[1][0])
    module = sess.graph.get_operation_by_name(name)

    return name, new_tensor.op, module


def reduce_batchnorm(sess: tf.compat.v1.Session,
                     op_tensor_tuple: Tuple[Op, List[tf.Tensor]], op_mask) -> (str, tf.Operation, tf.Operation):
    """
    Fused and non fused batchnorm module reducer
    :param sess: current tf.compat.v1.Session
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    :param op_mask: Mask containing information on input and output channels to winnow
    """

    beta_product = op_tensor_tuple[0].get_param_product('beta')
    if beta_product:
        use_beta = True
        reduced_beta_init = tf.compat.v1.constant_initializer(_get_reduced_params(sess=sess,
                                                                                  product=beta_product,
                                                                                  mask=op_mask,
                                                                                  input_dim=0,
                                                                                  output_dim=None),
                                                              verify_shape=True)
    else:
        use_beta = False
        reduced_beta_init = 'zeros'

    gamma_product = op_tensor_tuple[0].get_param_product('gamma')
    if gamma_product:
        use_gamma = True
        reduced_gamma_init = tf.compat.v1.constant_initializer(_get_reduced_params(sess=sess,
                                                                                   product=gamma_product,
                                                                                   mask=op_mask,
                                                                                   input_dim=0,
                                                                                   output_dim=None),
                                                               verify_shape=True)
    else:
        use_gamma = False
        reduced_gamma_init = 'ones'

    moving_mean_product = op_tensor_tuple[0].get_param_product('moving_mean')
    reduced_mov_mean_init = tf.compat.v1.constant_initializer(_get_reduced_params(sess=sess,
                                                                                  product=moving_mean_product,
                                                                                  mask=op_mask,
                                                                                  input_dim=0,
                                                                                  output_dim=None),
                                                              verify_shape=True)
    moving_variance_product = op_tensor_tuple[0].get_param_product('moving_variance')
    reduced_mov_variance_init = tf.compat.v1.constant_initializer(_get_reduced_params(sess=sess,
                                                                                      product=moving_variance_product,
                                                                                      mask=op_mask,
                                                                                      input_dim=0,
                                                                                      output_dim=None),
                                                                  verify_shape=True)

    name = "reduced_" + op_tensor_tuple[0].dotted_name
    # Get training attribute
    # This will either be True, False, or a string representing a training_placeholder the original BN was using
    training = BNUtils.get_training(op_tensor_tuple[0].get_module())
    assert training is not None
    is_fused = op_tensor_tuple[0].type == 'FusedBatchNormV3'
    epsilon = BNUtils.get_epsilon(op_tensor_tuple[0].get_module())
    momentum = BNUtils.get_momentum(op_tensor_tuple[0].get_module())
    if momentum is not None:
        new_tensor = tf.keras.layers.BatchNormalization(center=use_beta,
                                                        scale=use_gamma,
                                                        epsilon=epsilon,
                                                        momentum=momentum,
                                                        beta_initializer=reduced_beta_init,
                                                        gamma_initializer=reduced_gamma_init,
                                                        moving_mean_initializer=reduced_mov_mean_init,
                                                        moving_variance_initializer=reduced_mov_variance_init,
                                                        fused=is_fused,
                                                        name=name)(op_tensor_tuple[1][0], training=training)
    else:
        new_tensor = tf.keras.layers.BatchNormalization(center=use_beta,
                                                        scale=use_gamma,
                                                        epsilon=epsilon,
                                                        beta_initializer=reduced_beta_init,
                                                        gamma_initializer=reduced_gamma_init,
                                                        moving_mean_initializer=reduced_mov_mean_init,
                                                        moving_variance_initializer=reduced_mov_variance_init,
                                                        fused=is_fused,
                                                        name=name)(op_tensor_tuple[1][0], training=training)
    module = new_tensor.op.inputs[0].op

    return name, new_tensor.op, module


def reduce_relu(sess: tf.compat.v1.Session,
                op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _) -> (str, tf.Operation, tf.Operation):
    """
    Relu module reducer
    :param sess: current tf.compat.v1.Session
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    """

    # In TF 1.7, no ReLU exists in tf.python.keras.layers API
    # need to use lower level tf.nn API here
    # pylint: disable=no-member
    name = "reduced_" + op_tensor_tuple[0].dotted_name
    if op_tensor_tuple[0].type == 'Relu6':
        new_tensor = tf.nn.relu6(op_tensor_tuple[1][0],
                                 name=name)
    else:
        new_tensor = tf.nn.relu(op_tensor_tuple[1][0],
                                name=name)
    module = sess.graph.get_operation_by_name(name)

    return name, new_tensor.op, module


def reduce_tanh(sess: tf.compat.v1.Session,
                op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _) -> (str, tf.Operation, tf.Operation):
    """
    Tanh module reducer
    :param sess: current tf.compat.v1.Session
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    """

    name = "reduced_" + op_tensor_tuple[0].dotted_name
    new_tensor = tf.nn.tanh(op_tensor_tuple[1][0],
                            name=name)
    module = sess.graph.get_operation_by_name(name)

    return name, new_tensor.op, module


def reduce_dropout(_, op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _op_mask) -> (str, tf.Operation, tf.Operation):
    """
    Dropout module reducer
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    :param _op_mask: unused parameter
    """

    name = "reduced_" + op_tensor_tuple[0].dotted_name
    # Get rate tensor
    cast_op = op_tensor_tuple[0].get_module().inputs[1].op
    assert cast_op.type == 'Cast'
    greater_equal_op = cast_op.inputs[0].op
    assert greater_equal_op.type == 'GreaterEqual'
    rate_tensor = greater_equal_op.inputs[1]
    rate = rate_tensor.op.get_attr('value').float_val[0]
    if op_tensor_tuple[0].pattern_type == 'Dropout_with_training_tensor':
        new_tensor = tf.keras.layers.Dropout(rate, name=name)(op_tensor_tuple[1][0])
    else:
        new_tensor = tf.keras.layers.Dropout(rate, name=name)(op_tensor_tuple[1][0], training=True)
    module = new_tensor.op

    return name, new_tensor.op, module


def reduce_identity(sess: tf.compat.v1.Session,
                    op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _) -> (str, tf.Operation, tf.Operation):
    """
    Identity module reducer
    :param sess: current tf.compat.v1.Session
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    """

    name = "reduced_" + op_tensor_tuple[0].dotted_name
    new_tensor = tf.identity(op_tensor_tuple[1][0],
                             name=name)
    module = sess.graph.get_operation_by_name(name)

    return name, new_tensor.op, module


def reduce_pad(sess: tf.compat.v1.Session,
               op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _) -> (str, tf.Operation, tf.Operation):
    """
    Pad module reducer
    :param sess: current tf.compat.v1.Session
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    """

    name = "reduced_" + op_tensor_tuple[0].dotted_name
    pad_op = op_tensor_tuple[0].get_module()

    # Get padding tensor dimensions
    # Padding dimension information is captured in an input tensor to the pad op, index 1 of pad op inputs
    # Dimensions of this tensor are always (N, 2), where N is the dimensionality of the input tensor coming into pad.
    # The value of padding[N][0] gives the amount to pad in dimension N prior to the contents of the input to pad, while
    # padding[N][1] gives the amount to pad in dimension N after the contents of the input.
    # Currently we do not support reducing a pad op that modifies the channel dimension, which is the last dimension,
    # indexed by -1 below.  So check to make sure that indices [-1][0] and [-1][1] remain 0 (no padding).
    padding_tensor_eval = sess.run(pad_op.inputs[1])
    if padding_tensor_eval[-1][0] != 0 or padding_tensor_eval[-1][1] != 0:
        raise NotImplementedError("Attempting to reduce pad operation that modifies channel size, not supported.")
    new_padding_tensor = tf.constant(padding_tensor_eval)       # No need to actually modify padding tensor

    # Get constant value for padding
    # If pad op takes a non default constant value (default = 0), it appears as a third input tensor to pad op, index 2
    const_val = 0
    if len(pad_op.inputs) > 2:
        const_val = sess.run(pad_op.inputs[2])

    # Get mode
    # Mode can be 'CONSTANT', 'SYMMETRIC', or 'REFLECT'.  'CONSTANT' is default, and will not appear as a mode attribute
    # if it is the case.
    try:
        mode = pad_op.get_attr('mode')
        mode = mode.decode('utf-8')
    except ValueError:
        mode = 'CONSTANT'

    new_tensor = tf.pad(op_tensor_tuple[1][0],
                        new_padding_tensor,
                        constant_values=const_val,
                        mode=mode,
                        name=name)
    module = sess.graph.get_operation_by_name(name)

    return name, new_tensor.op, module


def reduce_min_max(sess: tf.compat.v1.Session,
                   op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _) -> (str, tf.Operation, tf.Operation):
    """
    Minimum and maximum module reducer
    :param sess: current tf.compat.v1.Session
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    """

    name = "reduced_" + op_tensor_tuple[0].dotted_name
    rate = op_tensor_tuple[0].get_module().inputs[1].op.get_attr('value').float_val[0]

    if op_tensor_tuple[0].type == 'Minimum':
        new_tensor = tf.minimum(op_tensor_tuple[1][0],
                                rate,
                                name=name)
    else:
        new_tensor = tf.maximum(op_tensor_tuple[1][0],
                                rate,
                                name=name)
    module = sess.graph.get_operation_by_name(name)

    return name, new_tensor.op, module


def reduce_add(sess: tf.compat.v1.Session,
               op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _) -> (str, tf.Operation, tf.Operation):
    """
    Add module reducer
    :param sess: current tf.compat.v1.Session
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    """

    name = "reduced_" + op_tensor_tuple[0].dotted_name
    if len(op_tensor_tuple[1]) > 2:
        new_tensor = tf.add_n(op_tensor_tuple[1],
                              name=name)
    else:
        new_tensor = tf.add(op_tensor_tuple[1][0],
                            op_tensor_tuple[1][1],
                            name=name)
    module = sess.graph.get_operation_by_name(name)

    return name, new_tensor.op, module


def reduce_concat(sess: tf.compat.v1.Session, op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _):
    """
    Concat module reducer
    :param sess: current tf.compat.v1.Session
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    """
    raise AssertionError("Should not need to call this function unless mask propagation rules for concat change")


def reduce_downsample(_, op_tensor_tuple: Tuple[Op, List[tf.Tensor]], op_mask) -> (str, tf.Operation, tf.Operation):
    """
    Downsample module reducer
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    :param op_mask: Mask of downsample op
    """

    # Get tuple of channels which downsample is currently keeping
    indices_op = op_tensor_tuple[0].get_module().inputs[1].op
    indices_length = indices_op.get_attr('value').tensor_shape.dim[0].size
    unpack_string = str(indices_length) + 'i'       # i for int, indices_length tells how many integers to parse out
    child_indices = struct.unpack(unpack_string, indices_op.get_attr('value').tensor_content)

    # Get list of remaining child indices after the existing tuple of channels is further pruned according to new mask
    child_mask = op_mask.output_channel_masks[0]
    new_child_indices = [index for i, index in enumerate(child_indices) if child_mask[i]]

    # Create new downsample op applying new list of child indices
    new_tensor = create_downsample_op('downsample', op_tensor_tuple[1][0], new_child_indices)
    module = new_tensor.op

    return new_tensor.op.name, new_tensor.op, module


def reduce_upsample2d(_, op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _op_mask) -> (str, tf.Operation,
                                                                                    tf.Operation):
    """
    Upsample2D module reducer
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    :param _op_mask: unused parameter
    """
    name = "reduced_" + op_tensor_tuple[0].dotted_name
    # Get size attribute
    strided_slice = op_tensor_tuple[0].get_module().outputs[0].consumers()[0]
    assert strided_slice.type == 'StridedSlice'
    mul = strided_slice.outputs[0].consumers()[0]
    const_op = mul.inputs[1].op
    tensor_content_length = const_op.get_attr('value').tensor_shape.dim[0].size
    # i for int, tensor_content_length tells how many integers to parse out
    unpack_string = str(tensor_content_length) + 'i'
    size = struct.unpack(unpack_string, const_op.get_attr('value').tensor_content)
    new_tensor = tf.keras.layers.UpSampling2D(size=size, name=name)(op_tensor_tuple[1][0])
    module = new_tensor.op

    return name, new_tensor.op, module


def reduce_leaky_relu(_, op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _op_mask) -> (str, tf.Operation,
                                                                                    tf.Operation):
    """
    Leaky_relu module reducer
    :param op_tensor_tuple: tuple containing the op to reduce, and a list of input tensors to the op
    :param _op_mask: unused parameter
    """
    name = "reduced_" + op_tensor_tuple[0].dotted_name
    alpha = op_tensor_tuple[0].get_module().get_attr('alpha')
    assert alpha is not None
    new_tensor = tf.nn.leaky_relu(op_tensor_tuple[1][0], alpha=alpha, name=name)
    module = new_tensor.op

    return name, new_tensor.op, module


def reduce_default(_, op_tensor_tuple: Tuple[Op, List[tf.Tensor]], _op_mask):
    """ Default reducer handler """
    raise NotImplementedError("reduction for op type %s not implemented" % op_tensor_tuple[0].type)


def _get_kernel_regularizer(kernel_tensor: tf.Tensor) -> Union[None, tf.Tensor]:
    """
    Get a kernel regularizer of the same kind as attached to kernel_tensor
    :param kernel_tensor: Kernel tensor to check for regularization
    :return: A new kernel regularizer if kernel_tensor has regularization, None otherwise
    """
    kernel_regularizer = None
    for consumer in kernel_tensor.consumers():
        if consumer.type == 'L2Loss':
            # Try to see if there is a scale value associated with it
            try:
                l2_regularizer_mul = consumer.outputs[0].consumers()[0]
                scale_op = l2_regularizer_mul.inputs[0].op
                scale_val = scale_op.get_attr('value').float_val[0]
                kernel_regularizer = tf.contrib.layers.l2_regularizer(scale_val)
            except:     # pylint: disable=bare-except
                kernel_regularizer = tf.nn.l2_loss      # pylint: disable=no-member
    return kernel_regularizer


def _get_reduced_params(sess: tf.compat.v1.Session, product: Product, mask: Mask, input_dim: Union[int, None],
                        output_dim: Union[int, None]) -> np.ndarray:
    """
    Reduce parameter tensor shapes
    :param sess: Current session
    :param product: represents the tensor which we want to reduce, using information contained in mask
    :param mask: contains information regarding input and output channels to reduce
    :param input_dim: dimension corresponding to input channels in product.  User specifies -1 to mean no input
    dim is present
    :param output_dim: dimension corresponding to output channels in product.  User specifies -1 to mean no output
    dim is present
    """

    input_ch_masks, output_ch_masks = mask.input_channel_masks, mask.output_channel_masks
    if len(input_ch_masks) == 1:
        input_ch_indices_to_reduce = get_zero_positions_in_binary_mask(input_ch_masks[0])
    else:
        input_ch_indices_to_reduce = []

    if output_ch_masks:
        output_ch_indices_to_reduce = get_zero_positions_in_binary_mask(output_ch_masks[0])
    else:
        output_ch_indices_to_reduce = []

    # get values of old parameter tensor as ndarray
    assert len(product.tensor_dict.keys()) == 1
    assert len(product.consumers) == 1
    old_tensor = product.tensor_dict[product.consumers[0]]
    values = old_tensor.eval(session=sess)

    # use np.delete to slice out specific elements of the ndarray
    # ensure that dimension size is large enough to delete all specified channels
    if input_ch_indices_to_reduce and input_dim is not None:
        assert max(input_ch_indices_to_reduce) < values.shape[input_dim]
        values = np.delete(values, input_ch_indices_to_reduce, input_dim)
    if output_ch_indices_to_reduce and output_dim is not None:
        assert max(output_ch_indices_to_reduce) < values.shape[output_dim]
        values = np.delete(values, output_ch_indices_to_reduce, output_dim)

    return values


def create_downsample_op(name, input_tensor, index_list) -> tf.Tensor:
    """
    Create a downsample op and return the output tensor
    :param name: scope name to create the op under
    :param input_tensor: Input tensor to the downsample op
    :param index_list: Indices to drop out
    :return: Output tensor of the downsample op
    """
    with tf.name_scope(name):
        gather_tensor = tf.gather(input_tensor, index_list, axis=-1)
    return gather_tensor
