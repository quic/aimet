# /usr/bin/env python3.5
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
""" Quantsim for Keras """

import json
import os
from typing import Union, Dict
import tensorflow as tf

from aimet_common.defs import QuantScheme
from aimet_common.utils import AimetLogger, save_json_yaml
from aimet_common.quantsim import encoding_version
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper, QuantizerSettings
from aimet_tensorflow.keras.quant_sim.tensor_quantizer import TensorQuantizer

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

unquantizable_modules = (tf.keras.layers.InputLayer, QcQuantizeWrapper)

class QuantizationSimModel:
    """
    Implements mechanism to add quantization simulations ops to a model. This allows for off-target simulation of
    inference accuracy. Also allows the model to be fine-tuned to counter the effects of quantization.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=unused-argument
    def __init__(self, model, quant_scheme: Union[QuantScheme, str] = 'tf_enhanced', rounding_mode: str = 'nearest',
                 default_output_bw: int = 8, default_param_bw: int = 8, in_place: bool = False,
                 config_file: str = None):
        """
        :param model: Model to quantize
        :param quant_scheme: Quantization Scheme, currently supported schemes are post_training_tf and
               post_training_tf_enhanced, defaults to post_training_tf_enhanced
        :param rounding_mode: The round scheme to used. One of: 'nearest' or 'stochastic', defaults to 'nearest'.
        :param default_output_bw: bitwidth to use for activation tensors, defaults to 8
        :param default_param_bw: bitwidth to use for parameter tensors, defaults to 8
        :param in_place: If True, then the given 'model' is modified in-place to add quant-sim nodes.
                Only suggested use of this option is when the user wants to avoid creating a copy of the model
        :param config_file: Path to a config file to use to specify rules for placing quant ops in the model
        """
        self._model_without_wrappers = model
        if not in_place:
            self._model_without_wrappers = tf.keras.models.clone_model(model)
            self._model_without_wrappers.set_weights(model.get_weights())
        self._layer_name_to_quant_wrapper = {}
        self._validate_model()
        self.model = self._add_quantization_wrappers(quant_scheme, rounding_mode, default_output_bw, default_param_bw)

    def _validate_model(self):
        """
        Check that model is appropriate for quantsim.
        """
        multiple_inbound_node_layers = []

        for layer in self._model_without_wrappers.layers:
            if len(layer.inbound_nodes) > 1:
                multiple_inbound_node_layers.append(layer.name)

        if multiple_inbound_node_layers:
            _logger.error('Layers with more than one inbound nodes are unsupported. This may occur if a layer is '
                          'reused multiple times in the model definition.')
            _logger.error('Layers with multiple inbound nodes: {%s}', multiple_inbound_node_layers)
            raise NotImplementedError

    def _add_quantization_wrappers(self, quant_scheme, rounding_mode, default_output_bw, default_param_bw):
        """
        Add quantization wrappers to the model and return a new model with the wrappers inserted.
        :param quant_scheme: Quantization scheme to use
        :param rounding_mode: Rounding mode to use
        :param default_output_bw: Default bitwidth for activation quantizers
        :param default_param_bw: Default bitwidth for param quantizers
        """
        def wrap_layer(layer) -> tf.keras.layers.Layer:
            """
            Function to wrap layers with QcQuantizeWrappers, used by keras clone_model()
            :param layer: Layer to wrap
            :return: Wrapped layer, or original layer if layer is not to be wrapped
            """
            activation_quant_settings = QuantizerSettings(default_output_bw, rounding_mode,
                                                          quant_scheme, False, False, False)
            param_quant_settings = QuantizerSettings(default_param_bw, rounding_mode,
                                                     quant_scheme, False, False, False)
            if isinstance(layer, unquantizable_modules) or layer.submodules:
                return layer

            wrapper = QcQuantizeWrapper(layer, activation_quant_settings, param_quant_settings,
                                        num_inputs=len(layer.inbound_nodes[0].keras_inputs))
            self._layer_name_to_quant_wrapper[layer.name] = wrapper
            return wrapper

        return tf.keras.models.clone_model(self._model_without_wrappers, clone_function=wrap_layer)

    @staticmethod
    def _get_encoding_dict_for_quantizer(quantizer: TensorQuantizer) -> Dict[str, Union[str, int, float]]:
        """
        Get encoding dict for a tensor quantizer.
        :param quantizer: Quantizer to get encoding info from
        :return: Dictionary containing encodings info for the tensor quantizer
        """
        encoding_dict = {}
        encoding_dict['min'] = quantizer.encoding.min
        encoding_dict['max'] = quantizer.encoding.max
        encoding_dict['scale'] = quantizer.encoding.delta
        encoding_dict['offset'] = int(quantizer.encoding.offset)
        encoding_dict['bitwidth'] = quantizer.encoding.bw
        encoding_dict['is_symmetric'] = str(quantizer.is_symmetric)
        encoding_dict['dtype'] = 'int'
        return encoding_dict

    def _get_encodings_dict(self) -> Dict[str, Union[str, Dict]]:
        """
        Get encodings dict containing all activation and parameter encodings info in the model
        :return: Dictionary containing all activation and parameter encodings info in the model
        """
        # pylint: disable=protected-access
        activation_encodings = {}
        param_encodings = {}
        for wrapper in self.quant_wrappers():
            for idx, input_quantizer in enumerate(wrapper.input_quantizers):
                if input_quantizer.encoding is not None:
                    tensor_name = wrapper._layer_to_wrap.inbound_nodes[0].keras_inputs[idx].name
                    encoding_dict = self._get_encoding_dict_for_quantizer(input_quantizer)
                    activation_encodings[tensor_name] = encoding_dict
            for idx, param_quantizer in enumerate(wrapper.param_quantizers):
                if param_quantizer.encoding is not None:
                    param_name = wrapper._layer_to_wrap.weights[idx].name
                    encoding_dict = self._get_encoding_dict_for_quantizer(param_quantizer)
                    param_encodings[param_name] = encoding_dict
            for idx, output_quantizer in enumerate(wrapper.output_quantizers):
                if output_quantizer.encoding is not None:
                    tensor_name = wrapper._layer_to_wrap.output.name
                    encoding_dict = self._get_encoding_dict_for_quantizer(output_quantizer)
                    activation_encodings[tensor_name] = encoding_dict
        encodings_dict = {'version': encoding_version,
                          'activation_encodings': activation_encodings,
                          'param_encodings': param_encodings}
        return encodings_dict

    def compute_encodings(self, forward_pass_callback, forward_pass_callback_args):
        """
        Computes encodings for all quantization sim nodes in the model.

        :param forward_pass_callback: A callback function that is expected to runs forward passes on a model.
               This callback function should use representative data for the forward pass, so the calculated
               encodings work for all data samples.
        :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
               the user to determine the type of this parameter. E.g. could be simply an integer representing the number
               of data samples to use. Or could be a tuple of parameters or an object representing something more
               complex.
        """
        forward_pass_callback(self.model, forward_pass_callback_args)
        for quant_wrapper in self.quant_wrappers():
            quant_wrapper.compute_encoding()

    def export(self, path, filename_prefix):
        """
        This method exports out the quant-sim model so it is ready to be run on-target.

        Specifically, the following are saved

        1. The sim-model is exported to a regular Keras model without any simulation ops
        2. The quantization encodings are exported to a separate JSON-formatted file that can
           then be imported by the on-target runtime (if desired)

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        """
        model_path = os.path.join(path, filename_prefix)
        self._model_without_wrappers.save(model_path)
        self._model_without_wrappers.save(model_path + '.h5', save_format='h5')
        encodings_dict = self._get_encodings_dict()
        encoding_file_path = os.path.join(path, filename_prefix + '.encodings')
        save_json_yaml(encoding_file_path, encodings_dict)

    def set_and_freeze_param_encodings(self, encoding_path: str):
        """
        Set and freeze parameter encodings from encodings JSON file
        :param encoding_path: path from where to load parameter encodings file
        """
        # Load parameter encodings file
        with open(encoding_path) as json_file:
            param_encodings = json.load(json_file)

        for quant_wrapper in self.quant_wrappers():
            quant_wrapper.set_and_freeze_param_encoding(param_encodings)

    def quant_wrappers(self):
        """
        Generator for yielding all quantization wrappers
        """
        for layer in self.model.layers:
            if isinstance(layer, QcQuantizeWrapper):
                yield layer

    def get_quant_wrapper_for_layer_name(self, layer_name: str) -> QcQuantizeWrapper:
        """
        Return qc quant wrapper corresponding to a layer name
        :param layer_name: Layer name to get quantize wrapper for
        :return: Qc quant wrapper corresponding to a layer name
        """
        return self._layer_name_to_quant_wrapper.get(layer_name)
