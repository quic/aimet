# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Adaround optimizer """

from typing import Union, Callable, Tuple
from functools import reduce
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_tensorflow.adaround.adaround_loss import AdaroundLoss, AdaroundHyperParameters
from aimet_tensorflow.adaround.adaround_wrapper import AdaroundWrapper, BATCH_SIZE

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class AdaroundOptimizer:
    """
    Optimizes the weight rounding
    """
    def __init__(self):
        """
        Constructor creates session for optimization
        """
        self._warm_start_tensor = tf.compat.v1.placeholder(dtype=tf.bool, shape=[])
        self._beta_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[])
        self._inp_tensor = tf.compat.v1.placeholder(dtype=tf.float32, name='inp_tensor')
        self._out_tensor = tf.compat.v1.placeholder(dtype=tf.float32, name='out_tensor')
        self._indices_tensor = tf.compat.v1.placeholder(dtype=tf.int32)
        self._optimizer_session = tf.compat.v1.Session()

    def __del__(self):
        """
        Close session used for optimization
        """
        self._optimizer_session.close()

    def adaround_wrapper(self, wrapper: AdaroundWrapper, act_func: Callable, all_inp_data: np.ndarray,
                         all_orig_out_data: np.ndarray, opt_params: AdaroundHyperParameters) -> (np.ndarray, np.ndarray):
        """
        Adaround wrapper
        :param wrapper: Adaround wrapper
        :param act_func: Activation function
        :param all_inp_data: Input activation data
        :param all_orig_out_data: Original output activation data
        :param opt_params: Adaround hyper parameters
        :return: hard_rounded_weight, soft_rounded_weight
        """
        # Initialize alpha and get reconstruction error tensor
        self._optimizer_session.run(wrapper.alpha.initializer)
        recons_err_tensor = self._get_recons_err_tensor(wrapper, act_func, self._inp_tensor, self._out_tensor)

        # Single test with batch size of activation data
        inp_data = all_inp_data[:BATCH_SIZE]
        orig_out_data = all_orig_out_data[:BATCH_SIZE]

        # Reconstruction error using hard and soft rounding before optimization
        recons_err_hard,\
        recons_err_soft = self._eval_recons_err_metrics(wrapper, recons_err_tensor, inp_data, orig_out_data)
        logger.debug("Before opt, Recons. error metrics using soft rounding=%f and hard rounding=%f", recons_err_soft,
                     recons_err_hard)

        hard_rounded_weight, soft_rounded_weight = self.optimize_rounding(wrapper, act_func, all_inp_data,
                                                                          all_orig_out_data, opt_params)

        # Reconstruction error using hard and soft rounding after optimization
        recons_err_hard,\
        recons_err_soft = self._eval_recons_err_metrics(wrapper, recons_err_tensor, inp_data, orig_out_data)
        logger.debug("After opt, Recons. error metrics using soft rounding=%f and hard rounding=%f", recons_err_soft,
                     recons_err_hard)

        return hard_rounded_weight, soft_rounded_weight

    @staticmethod
    def _get_recons_err_tensor(wrapper: AdaroundWrapper, act_func, inp_tensor: tf.Tensor, orig_out_tensor: tf.Tensor) \
            -> tf.Tensor:
        """
        Gets reconstruction error tensor
        :param wrapper: Adaround wrapper
        :param inp_tensor: Input activation data tensor
        :param orig_out_tensor: Original output activation data tensor
        :return: Reconstruction error tensor
        """
        # Forward pass through wrapper
        adaround_out_tensor = wrapper(inp_tensor)

        # If followed by an activation function
        if act_func is not None:
            adaround_out_tensor = act_func(adaround_out_tensor)
            orig_out_tensor = act_func(orig_out_tensor)

        recons_error_tensor = tf.reduce_mean(tf.math.squared_difference(adaround_out_tensor, orig_out_tensor))
        return recons_error_tensor

    def _eval_recons_err_metrics(self, wrapper: AdaroundWrapper, recons_error_tensor: tf.Tensor, inp_data: np.ndarray,
                                 out_data: np.ndarray) -> (float, float):
        """
        Evaluates reconstruction error tensor using hard and soft rounding
        :param wrapper: Adaround wrapper
        :param recons_error_tensor: Reconstruction error tensor to be evaluated
        :param inp_data: Input activation data
        :param out_data: Original output activation data
        :return: Reconstruction error using hard rounding and soft rounding
        """
        feed_dict = {wrapper.use_soft_rounding: False, self._inp_tensor: inp_data, self._out_tensor: out_data}
        recons_err_hard = self._optimizer_session.run(recons_error_tensor, feed_dict=feed_dict)

        feed_dict = {wrapper.use_soft_rounding: True, self._inp_tensor: inp_data, self._out_tensor: out_data}
        recons_err_soft = self._optimizer_session.run(recons_error_tensor, feed_dict=feed_dict)

        return float(recons_err_hard), float(recons_err_soft)

    def optimize_rounding(self, wrapper: AdaroundWrapper, act_func, all_inp_data: np.ndarray,
                          all_orig_out_data: np.ndarray, opt_params: AdaroundHyperParameters) \
            -> (np.ndarray, np.ndarray):
        """
        Optimizes the weight rounding of Adaround wrapper layer and returns soft and hard rounded weight
        :param wrapper: Adaround wrapper
        :param act_func: Activation function
        :param all_inp_data: Input activation data
        :param all_orig_out_data: Original output activation data
        :param opt_params: Adaround hyper parameters
        :return: hard_rounded_weight, soft_rounded_weight
        """
        # pylint: disable=too-many-locals
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # Create tf.constant() operation for intermediate activation data to speed up the optimization
        # if possible otherwise fallback to original implementation which incurs CPU-GPU transfers.
        tensors_pinned, inp_tensor, out_tensor = self._pin_inp_and_out_tensors(all_inp_data, all_orig_out_data)

        # Get the channels index for 'channels_last' data format to compute reconstruction loss
        # across the channels index. 'channels_last' data format is only supported format.
        channels_index = len(all_orig_out_data.shape) - 1

        for cur_iteration in tqdm(range(opt_params.num_iterations)):

            # During warm start period, rounding loss is zero
            warm_start = cur_iteration < opt_params.num_iterations * opt_params.warm_start

            # Compute beta parameter used in regularization function using cosine decay
            beta = AdaroundLoss.compute_beta(opt_params.num_iterations, cur_iteration, opt_params.beta_range,
                                             opt_params.warm_start)

            # Get random indices of batch size
            indices = np.random.permutation(all_inp_data.shape[0])[:BATCH_SIZE]

            # Graph is created only once
            if cur_iteration == 0:
                train_op, loss_tensors = self.train_step(wrapper, act_func, optimizer, inp_tensor,
                                                         out_tensor, opt_params.reg_param,
                                                         self._warm_start_tensor, self._beta_tensor, channels_index)
                self._optimizer_session.run(tf.compat.v1.global_variables_initializer())
                total_loss_tensor, recon_loss_tensor, round_loss_tensor = loss_tensors

            # Create feed_dict of inputs for session run
            feed_dict = {self._warm_start_tensor: warm_start, self._beta_tensor: beta}
            # If tensors are created and can be fit entirely in memory, then pass only indices, else
            # pass both input and output data.
            if tensors_pinned:
                feed_dict.update({self._indices_tensor: indices})
            else:
                feed_dict.update({inp_tensor: all_inp_data[indices], out_tensor: all_orig_out_data[indices]})

            _, total_loss, recon_loss, round_loss = self._optimizer_session.run([train_op, total_loss_tensor,
                                                                                 recon_loss_tensor, round_loss_tensor],
                                                                                feed_dict=feed_dict)
            if cur_iteration == 0 or cur_iteration % 100 == 0:
                logger.debug("After iterations=%d, Total loss=%5f, Recons. loss=%5f, Rounding loss=%5f",
                             cur_iteration, float(total_loss), float(recon_loss), float(round_loss))

        hard_rounded_weight = self._optimizer_session.run(wrapper.adaround_weights(),
                                                          feed_dict={wrapper.use_soft_rounding: False})
        soft_rounded_weight = self._optimizer_session.run(wrapper.adaround_weights(),
                                                          feed_dict={wrapper.use_soft_rounding: True})

        return hard_rounded_weight, soft_rounded_weight

    @staticmethod
    def train_step(wrapper: AdaroundWrapper, act_func: Callable, optimizer, inp_tensor: tf.Tensor,
                   orig_out_tensor: tf.Tensor, reg_param: float, warm_start: Union[bool, tf.Tensor],
                   beta: Union[float, tf.Tensor], channels_index: int):
        """
        Common implementation between TensorFlow eager and graph mode
        :param wrapper: Adaround wrapper
        :param act_func: Activation function
        :param optimizer: Optimizer
        :param inp_tensor: Input activation data tensor
        :param orig_out_tensor: Original output activation data tensor
        :param reg_param: Regularization parameter, trading off between rounding loss vs reconstruction loss
        :param warm_start: Warm up period, during which rounding loss has zero effect
        :param beta: Beta parameter
        :param channels_index: channels_index across which reconstruction loss will be computed
        :return: train_op, total_loss, recon_loss, round_loss
        """
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-arguments
        with tf.GradientTape() as tape:

            # Forward pass through wrapper
            adaround_out_tensor = wrapper(inp_tensor)

            # If followed by an activation function
            if act_func is not None:
                adaround_out_tensor = act_func(adaround_out_tensor)
                orig_out_tensor = act_func(orig_out_tensor)

            # Calculate total loss
            recon_loss = AdaroundLoss.compute_recon_loss(adaround_out_tensor, orig_out_tensor, channels_index)
            round_loss = AdaroundLoss.compute_round_loss(wrapper.alpha, reg_param, warm_start, beta)
            total_loss = recon_loss + round_loss

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, [wrapper.alpha])
        train_op = optimizer.apply_gradients(zip(gradients, [wrapper.alpha]))

        return train_op, (total_loss, recon_loss, round_loss)

    @staticmethod
    def _is_tensor_pinned(numpy_data: np.ndarray, name: str) -> bool:
        """
        NOTE: there is hard limit of 2GB to create tensorflow protobuf.

        Utility to see if the given numpy data can be entirely fit in the device (GPU if available) memory
        by creating tf.constant() operation in the graph.

        :param numpy_data: Numpy data.
        :param name: Name of tensor to be created.
        :return: True if tensor is created and pinned in memory, False otherwise.
        """
        is_tensor_pinned = True
        try:
            tf.constant(numpy_data, dtype=tf.float32, name=name)
        except ValueError:
            is_tensor_pinned = False
        return is_tensor_pinned

    def _pin_inp_and_out_tensors(self, all_inp_data: np.ndarray, all_orig_out_data: np.ndarray) -> \
            Tuple[bool, tf.Tensor, tf.Tensor]:
        """
        NOTE: there is hard limit of 2GB to create tensorflow protobuf.

        Create tf.constant() operations for op's intermediate activation data in the graph which will avoid
        CPU-GPU data transfers and speeds up the optimization. If we can't pin the activation data in device memory,
        then we fall back to original implementation which returns placeholders for input and output tensors
        to transfer Numpy arrays to device memory.

        :param all_inp_data: Input activation data.
        :param all_orig_out_data: Original output activation data.
        :return: True if the tensors are pinned successfully in device memory (False otherwise),
         input tensor, output tensor.
        """
        all_inp_data_dim = reduce(lambda x, y: x * y, all_inp_data.shape)
        all_orig_out_data_dim = reduce(lambda x, y: x * y, all_orig_out_data.shape)

        if all_inp_data_dim >= all_orig_out_data_dim:
            larger_act_data, larger_act_data_name = all_inp_data, "all_inp_data"
            smaller_act_data, smaller_act_data_name = all_orig_out_data, "all_orig_out_data"
        else:
            larger_act_data, larger_act_data_name = all_orig_out_data, "all_orig_out_data"
            smaller_act_data, smaller_act_data_name = all_inp_data, "all_inp_data"

        inp_tensor = self._inp_tensor
        out_tensor = self._out_tensor
        tensors_pinned = False

        # Try to fit larger activation data in memory first. If it succeeds, try to fit the other
        # activation data in memory. If it fails in either of two the cases, fall back to original implementation.
        if self._is_tensor_pinned(larger_act_data, name=larger_act_data_name):
            if self._is_tensor_pinned(smaller_act_data, name=smaller_act_data_name):
                inp_tensor = self._optimizer_session.graph.get_tensor_by_name("all_inp_data:0")
                out_tensor = self._optimizer_session.graph.get_tensor_by_name("all_orig_out_data:0")
                inp_tensor = tf.gather(inp_tensor, self._indices_tensor)
                out_tensor = tf.gather(out_tensor, self._indices_tensor)
                tensors_pinned = True

        return tensors_pinned, inp_tensor, out_tensor
