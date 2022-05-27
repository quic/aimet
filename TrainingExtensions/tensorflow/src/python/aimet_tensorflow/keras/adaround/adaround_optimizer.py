# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
from typing import Callable
import numpy as np
import tensorflow as tf

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_tensorflow.adaround.adaround_loss import AdaroundLoss, AdaroundHyperParameters
from aimet_tensorflow.adaround.adaround_wrapper import AdaroundWrapper
from aimet_tensorflow.adaround.adaround_optimizer import AdaroundOptimizer as TfAdaroundOptimizer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
BATCH_SIZE = 32


class AdaroundOptimizer:
    """
    Optimizes the weight rounding
    """
    @staticmethod
    def adaround_wrapper(wrapper: AdaroundWrapper, act_func: Callable, all_inp_data: np.ndarray,
                         all_orig_out_data: np.ndarray, opt_params: AdaroundHyperParameters) -> \
            (np.ndarray, np.ndarray):
        """
        Adaround wrapper
        :param wrapper: Adaround wrapper
        :param act_func: Activation function
        :param all_inp_data: Input activation data
        :param all_orig_out_data: Original output activation data
        :param opt_params: Adaround hyper parameters
        :return: hard_rounded_weight, soft_rounded_weight
        """
        # Single test with batch size of activation data
        inp_data = all_inp_data[:BATCH_SIZE]
        orig_out_data = all_orig_out_data[:BATCH_SIZE]

        # Reconstruction error using hard and soft rounding before optimization
        recons_err_hard,\
        recons_err_soft = AdaroundOptimizer._eval_recons_err_metrics(wrapper, act_func, inp_data, orig_out_data)
        logger.debug("Before opt, Recons. error metrics using soft rounding=%f and hard rounding=%f", recons_err_soft,
                     recons_err_hard)

        hard_rounded_weight, soft_rounded_weight = AdaroundOptimizer.optimize_rounding(wrapper, act_func, all_inp_data,
                                                                                       all_orig_out_data, opt_params)

        # Reconstruction error using hard and soft rounding after optimization
        recons_err_hard,\
        recons_err_soft = AdaroundOptimizer._eval_recons_err_metrics(wrapper, act_func, inp_data, orig_out_data)
        logger.debug("After opt, Recons. error metrics using soft rounding=%f and hard rounding=%f", recons_err_soft,
                     recons_err_hard)

        return hard_rounded_weight, soft_rounded_weight

    @staticmethod
    def _eval_recons_err_metrics(wrapper: AdaroundWrapper, act_func, inp_data: np.ndarray, out_data: np.ndarray) -> \
            (float, float):
        """
        Evaluates reconstruction error tensor using hard and soft rounding
        :param wrapper: Adaround wrapper
        :param act_func: Activation function
        :param inp_data: Input activation data
        :param out_data: Original output activation data
        :return: Reconstruction error using hard rounding and soft rounding
        """
        wrapper.use_soft_rounding.assign(False)
        adaround_out_tensor = wrapper(inp_data)
        orig_out_tensor = out_data
        # If followed by an activation function
        if act_func is not None:
            adaround_out_tensor = act_func(adaround_out_tensor)
            orig_out_tensor = act_func(out_data)
        recons_error_hard = tf.reduce_mean(tf.math.squared_difference(adaround_out_tensor, orig_out_tensor))

        wrapper.use_soft_rounding.assign(True)
        adaround_out_tensor = wrapper(inp_data)
        orig_out_tensor = out_data
        # If followed by an activation function
        if act_func is not None:
            adaround_out_tensor = act_func(adaround_out_tensor)
            orig_out_tensor = act_func(out_data)
        recons_error_soft = tf.reduce_mean(tf.math.squared_difference(adaround_out_tensor, orig_out_tensor))

        return float(recons_error_hard), float(recons_error_soft)

    @staticmethod
    def optimize_rounding(wrapper: AdaroundWrapper, act_func, all_inp_data: np.ndarray, all_orig_out_data: np.ndarray,
                          opt_params: AdaroundHyperParameters) -> (np.ndarray, np.ndarray):
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
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        wrapper.use_soft_rounding.assign(True)

        for cur_iteration in range(opt_params.num_iterations):
            # During warm start period, rounding loss is zero
            warm_start = cur_iteration < opt_params.num_iterations * opt_params.warm_start

            # Compute beta parameter used in regularization function using cosine decay
            beta = AdaroundLoss.compute_beta(opt_params.num_iterations, cur_iteration, opt_params.beta_range,
                                             opt_params.warm_start)

            # Get random indices of batch size and get original output and input activation data of batch size
            indices = np.random.permutation(all_inp_data.shape[0])[:BATCH_SIZE]
            # inp_data = tf.gather(all_inp_data, indices)
            # orig_out_data = tf.gather(all_orig_out_data, indices)
            inp_data = all_inp_data[indices]
            orig_out_data = all_orig_out_data[indices]

            # Get the channels index for 'channels_last' data format to compute reconstruction loss
            # across the channels index
            channels_index = len(orig_out_data.shape) - 1
            _, (total_loss, recon_loss, round_loss) = TfAdaroundOptimizer.train_step(wrapper, act_func, optimizer,
                                                                                     inp_data, orig_out_data,
                                                                                     opt_params.reg_param, warm_start,
                                                                                     beta, channels_index,
                                                                                     orig_out_data.shape)
            if cur_iteration == 0 or cur_iteration % 100 == 0:
                logger.debug("After iterations=%d, Total loss=%5f, Recons. loss=%5f, Rounding loss=%5f",
                             cur_iteration, float(total_loss), float(recon_loss), float(round_loss))

        wrapper.use_soft_rounding.assign(False)
        hard_rounded_weight = wrapper.adaround_weights()

        wrapper.use_soft_rounding.assign(True)
        soft_rounded_weight = wrapper.adaround_weights()

        return hard_rounded_weight, soft_rounded_weight
