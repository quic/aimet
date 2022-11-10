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

""" Top level API for Adaptive Rounding - Post-Training Quantization (PTQ) """
from typing import Dict, List, Union, Iterable
import tensorflow as tf
from tensorflow.keras.utils import Progbar

import aimet_common.libpymo as libpymo

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_tensorflow.adaround.adaround_weight import AdaroundParameters
from aimet_tensorflow.adaround.adaround_weight import Adaround as TfAdaround
from aimet_tensorflow.adaround.adaround_loss import AdaroundHyperParameters
from aimet_tensorflow.keras.adaround.activation_sampler import ActivationSampler
from aimet_tensorflow.keras.adaround.adaround_wrapper import AdaroundWrapper
from aimet_tensorflow.keras.adaround.adaround_optimizer import AdaroundOptimizer
from aimet_tensorflow.keras.connectedgraph import ConnectedGraph, map_keras_types_to_onnx

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

AdaroundSupportedOps = (tf.keras.layers.Conv2D, tf.keras.layers.Dense)


class Adaround:
    """
    Weight-rounding mechanism for Post Training Quantization (PTQ)
    """

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    @classmethod
    def apply_adaround(cls, model: tf.keras.Model, params: AdaroundParameters, path: str, filename_prefix: str,
                       default_param_bw: int = 4,
                       default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                       config_file: str = None) -> tf.keras.Model:
        """
        Returns model with optimized weight rounding of every op (Conv and Linear) and also saves the
        corresponding quantization encodings to a separate JSON-formatted file that can then be imported by
        QuantSim for inference or QAT

        :param model: Model to adaround
        :param params: Parameters for adaround
        :param path: path where to store parameter encodings
        :param filename_prefix: Prefix to use for filename of the encodings file
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters. Default 4
        :param default_quant_scheme:  Quantization scheme. Supported options are QuantScheme.post_training_tf or
         QuantScheme.post_training_tf_enhanced. Default QuantScheme.post_training_tf_enhanced
        :param config_file: Configuration file for model quantizers
        :return: Model with Adarounded weights
        """

        # Get parameters from config file. To allow one central place for Adaround and Quantsim
        configs, strict_symmetric, unsigned_symmetric, per_channel_enabled = TfAdaround.get_config_dict_keys(config_file)

        # Optimization Hyper parameters
        opt_params = AdaroundHyperParameters(params.num_iterations, params.reg_param, params.beta_range,
                                             params.warm_start)

        # Activation sampler
        act_sampler = ActivationSampler(params.data_set, params.num_batches)

        hard_rounded_model = tf.keras.models.clone_model(model)
        hard_rounded_model.set_weights(model.get_weights())
        soft_rounded_model = tf.keras.models.clone_model(model)
        soft_rounded_model.set_weights(model.get_weights())

        ordered_layer_indices = cls._get_ordered_adaround_layer_indices(model)
        module_act_func_pair = cls._get_module_act_func_pair(model)
        param_encodings = {}

        progbar = Progbar(len(ordered_layer_indices))
        for idx in ordered_layer_indices:
            use_symmetric_encodings = TfAdaround.get_is_symmetric_flag_for_op_param(configs, model.layers[idx],
                                                                                    param_name='weight',
                                                                                    framework_to_onnx_type_dict=map_keras_types_to_onnx)
            cls.adaround_layer(act_sampler, use_symmetric_encodings, strict_symmetric, unsigned_symmetric,
                               default_param_bw, default_quant_scheme, model, hard_rounded_model, soft_rounded_model,
                               idx, module_act_func_pair, opt_params, param_encodings, per_channel_enabled)
            progbar.add(1)

        # Export quantization encodings to JSON-formatted file at provided path
        TfAdaround.export_encoding_to_json(path, filename_prefix, param_encodings)
        return soft_rounded_model

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    @classmethod
    def adaround_layer(cls, act_sampler, is_symmetric, strict_symmetric, unsigned_symmetric, default_param_bw,
                       default_quant_scheme, orig_model, hard_rounded_model, soft_rounded_model, idx,
                       module_act_func_pair, opt_params, param_encodings, per_channel_enabled):
        """
        Perform adaround on a specific layer.
        :param act_sampler: Activation sampler
        :param is_symmetric: True if symmetric encodings is used, else asymmetric encodings
        :param strict_symmetric: Taken from config file, False by default
        :param unsigned_symmetric: Taken from config file, True by default
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters
        :param default_quant_scheme: Quantization scheme. Supported options are QuantScheme.post_training_tf or
            QuantScheme.post_training_tf_enhanced
        :param orig_model: Original model
        :param hard_rounded_model: Model with hard rounded weights
        :param soft_rounded_model: Model with soft rounded weights
        :param idx: Index of layer in model to Adaround
        :param module_act_func_pair: Dictionary mapping modules to subsequent activation functions
        :param opt_params: Adaround hyperparameters
        :param param_encodings: Dictionary holding parameter encodings information
        :param per_channel_enabled: Flag for per channel quantization
        """
        # Collect input and output activations data
        all_inp_data, all_out_data = act_sampler.sample_activation(orig_model.layers[idx], orig_model,
                                                                   hard_rounded_model.layers[idx],
                                                                   hard_rounded_model)
        # Get module's next following activation function
        act_func = module_act_func_pair[orig_model.layers[idx]]

        output_height, output_width, output_channels = None, None, None
        if isinstance(orig_model.layers[idx], tf.keras.layers.Conv2DTranspose):
            data_format = 'NHWC' if orig_model.layers[idx].data_format == 'channels_last' else 'NCHW'
            output_height, output_width, output_channels = \
                TfAdaround.get_conv2d_transpose_output_tensor_shape(data_format, all_out_data)

        wrapper = AdaroundWrapper(orig_model.layers[idx], default_param_bw, default_quant_scheme,
                                  is_symmetric, strict_symmetric, unsigned_symmetric, per_channel_enabled,
                                  output_height, output_width, output_channels)
        hard_rounded_weight, soft_rounded_weight = AdaroundOptimizer.adaround_wrapper(wrapper, act_func,
                                                                                      all_inp_data, all_out_data,
                                                                                      opt_params)
        # Update param encodings dictionary
        Adaround._update_param_encodings_dict(param_encodings, orig_model.layers[idx], wrapper.encoding,
                                              is_symmetric)
        hard_rounded_model.layers[idx].set_weights([hard_rounded_weight] +
                                                   hard_rounded_model.layers[idx].get_weights()[1:])
        soft_rounded_model.layers[idx].set_weights([soft_rounded_weight] +
                                                   soft_rounded_model.layers[idx].get_weights()[1:])

    @classmethod
    def _get_module_act_func_pair(cls, model) -> Dict[tf.keras.layers.Layer, Union[None, tf.keras.layers.Layer]]:
        """
        Get dictionary mapping modules to subsequent activation functions. If the activation function is built into the
        layer, map None to that layer.
        :param model: Model to obtain module to activation info from
        :return: Dictionary mapping modules to subsequent activation functions
        """
        conn_graph = ConnectedGraph(model)
        module_act_func_pair = {}
        activation_types = (tf.keras.layers.ELU, tf.keras.layers.PReLU, tf.keras.layers.Softmax, tf.keras.layers.ReLU,
                            tf.keras.layers.LeakyReLU, tf.keras.layers.ThresholdedReLU, tf.keras.layers.Activation)
        for op in conn_graph.get_all_ops().values():
            # Get module associated with op
            cur_module = op.get_module()
            if cur_module:
                module_act_func_pair[cur_module] = None
                if hasattr(cur_module, 'activation'):
                    if tf.keras.activations.serialize(cur_module.activation) != 'linear':
                        # If the activation is not passthrough, it is built into the layer. Use None for the associated
                        # activation since running forward pass on the layer will already include the activation.
                        continue
                if op.output:
                    assert op.output.consumers, 'op output should have at least one consumer op.'
                    # Get the next op
                    next_op = op.output.consumers[0]
                    # Get module associated with next op
                    next_module = next_op.get_module()

                    # Get the appropriate activation function
                    if isinstance(next_module, activation_types):
                        module_act_func_pair[cur_module] = next_module
                        _logger.debug("Module: %s is followed by activation function: %s", op.dotted_name,
                                      next_op.dotted_name)

        return module_act_func_pair

    @staticmethod
    def _update_param_encodings_dict(encoding_dict: Dict, layer: tf.keras.layers.Layer,
                                     encoding: Union[libpymo.TfEncoding, List[libpymo.TfEncoding]],
                                     is_symmetric: bool):
        """
        Add layer's parameter encoding to dictionary to be used for exporting.
        :param encoding_dict: Encoding dictionary
        :param layer: Layer to obtain parameter encoding information for
        :param encoding: Encoding
        :param is_symmetric: Symmetric vs Asymmetric boolean
        """
        tensor_name = layer.weights[0].name
        if not isinstance(encoding, Iterable):
            encoding = [encoding]
        encoding_dict[tensor_name] = [{'min': enc.min,
                                       'max': enc.max,
                                       'scale': enc.delta,
                                       'offset': enc.offset,
                                       'bitwidth': enc.bw,
                                       'is_symmetric': is_symmetric} for enc in encoding]

    @staticmethod
    def _get_ordered_adaround_layer_indices(model: tf.keras.Model) -> List[int]:
        """
        Get a list of ordered layer indices corresponding to layers to Adaround.
        :param model: Model to find indices for
        :return: List of ordered layer indices to Adaround
        """
        return [idx for idx, layer in enumerate(model.layers) if isinstance(layer, AdaroundSupportedOps)]
