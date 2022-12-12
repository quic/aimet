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
from tensorflow import keras
import tensorflow.keras.backend as K
# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_tensorflow.adaround.adaround_loss import AdaroundLoss, AdaroundHyperParameters
from aimet_tensorflow.adaround.adaround_wrapper import AdaroundWrapper
from aimet_tensorflow.keras.adaround.adaround_loss import compute_beta
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

BATCH_SIZE = 32

#pylint: disable=too-many-ancestors
#pylint: disable=abstract-method
class CustomModel(keras.Model):
    """
    Custom model to train adaraound wrapper
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_count = K.variable(0)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {}

    def calculate_loss_wrapper(self, opt_params: AdaroundHyperParameters, wrapper: AdaroundWrapper, channels_index: int):
        """
        Wrapper function to compute loss
        :param opt_params: Adaround hyper parameters
        :param wrapper: Adaround wrapper
        :param channels_index: channels_index across which reconstruction loss will be computed
        """
        def calculate_loss(orig_out_tensor, adaround_out_tensor):
            """
            :param inp_tensor: Input activation data tensor
            :param orig_out_tensor: Original output activation data tensor
            """
            warm_start = self.epoch_count < opt_params.num_iterations * opt_params.warm_start

            # Compute beta parameter used in regularization function using cosine decay
            beta = compute_beta(opt_params.num_iterations, self.epoch_count, opt_params.beta_range, opt_params.warm_start)
            recon_loss = AdaroundLoss.compute_recon_loss(adaround_out_tensor, orig_out_tensor, channels_index=channels_index)
            round_loss = AdaroundLoss.compute_round_loss(alpha=wrapper.alpha, reg_param=opt_params.reg_param, warm_start=tf.cast(warm_start, tf.bool), beta=beta)
            total_loss = recon_loss + round_loss
            return total_loss
        return calculate_loss

class AccessEpochNumber(keras.callbacks.Callback):
    """
    Class to access and set epoch number during training.
    This epoch number will be used for calculating loss
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    def on_epoch_begin(self, epoch, logs=None):
        K.set_value(self.model.epoch_count, epoch)

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

        # Get random indices of batch size and get original output and input activation data of batch size
        indices = np.random.permutation(all_inp_data.shape[0])[:BATCH_SIZE]
        inp_data = all_inp_data[indices]
        orig_out_data = all_orig_out_data[indices]

        # Get the channels index for 'channels_last' data format to compute reconstruction loss
        # across the channels index
        channels_index = len(orig_out_data.shape) - 1

        inp_shape = inp_data.shape[1:]
        inp_layer = tf.keras.Input(shape=inp_shape)
        adaround_out_tensor = wrapper(inp_layer)

        # If followed by an activation function
        if act_func is not None:
            adaround_out_tensor = act_func(adaround_out_tensor)
            orig_out_data = act_func(orig_out_data)

        # Create custom model and training phase
        model = CustomModel(inp_layer, adaround_out_tensor)
        model.compile(optimizer=optimizer, loss=model.calculate_loss_wrapper(opt_params, wrapper, channels_index))
        epochNumberCallback = AccessEpochNumber(model=model)
        model.fit(inp_data, orig_out_data, epochs=opt_params.num_iterations,
                  callbacks=[epochNumberCallback], verbose=0)

        wrapper.use_soft_rounding.assign(False)
        hard_rounded_weight = wrapper.adaround_weights()

        wrapper.use_soft_rounding.assign(True)
        soft_rounded_weight = wrapper.adaround_weights()

        return hard_rounded_weight, soft_rounded_weight
