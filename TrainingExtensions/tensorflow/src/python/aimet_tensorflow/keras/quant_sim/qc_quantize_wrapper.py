# /usr/bin/env python3.5
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
""" Qc Quantize wrapper for tf 2 keras """

from typing import Union, List, Dict
import tensorflow as tf
from packaging import version

from aimet_common.utils import AimetLogger
from aimet_common.defs import MAP_QUANT_SCHEME_TO_PYMO, MAP_ROUND_MODE_TO_PYMO, QuantScheme
import libpymo
# Remove version check when we upgrade to tf 2.0
if version.parse(tf.version.VERSION) >= version.parse("2.00"):
    # pylint: disable=no-name-in-module
    from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

def _load_ops():
    """
    Function which loads the quantization op library. In order to load a graph with
    custom quantization ops this must be called first as this provides tensorflow with
    the required op definitions.

    :return: Loaded library
    """
    return tf.load_op_library('libaimet_tf_ops.so')

# Load the aimet ops
qcops = _load_ops()

class QuantizerSettings:
    """ Class holding quantizer settings """
    def __init__(self, bitwidth: int, round_mode: str, quant_scheme: Union[str, QuantScheme], is_symmetric: bool,
                 use_unsigned_symmetric: bool, use_strict_symmetric: bool):
        self._bitwidth = bitwidth
        self._round_mode = round_mode
        if isinstance(quant_scheme, str):
            if quant_scheme == 'tf':
                quant_scheme = QuantScheme.post_training_tf
            elif quant_scheme == 'tf_enhanced':
                quant_scheme = QuantScheme.post_training_tf_enhanced
            else:
                _logger.error('Unsupported quant scheme: {%s}', quant_scheme)
                raise AssertionError
        self._quant_scheme = quant_scheme
        self._is_symmetric = is_symmetric
        self._use_unsigned_symmetric = use_unsigned_symmetric
        self._use_strict_symmetric = use_strict_symmetric

    @property
    def quant_scheme(self):
        """ Quant scheme getter """
        return self._quant_scheme

    @property
    def round_mode(self):
        """ Round mode getter """
        return self._round_mode

    @property
    def bitwidth(self):
        """ Bitwidth getter """
        return self._bitwidth

    @bitwidth.setter
    def bitwidth(self, bitwidth: int):
        """ Bitwidth setter """
        self._bitwidth = bitwidth

    @property
    def is_symmetric(self):
        """ Is symmetric getter """
        return self._is_symmetric

    @is_symmetric.setter
    def is_symmetric(self, is_symmetric: bool):
        """ Bitwidth setter """
        self._is_symmetric = is_symmetric

    @property
    def use_unsigned_symmetric(self):
        """ Use unsigned symmetric getter """
        return self._use_unsigned_symmetric

    @use_unsigned_symmetric.setter
    def use_unsigned_symmetric(self, use_unsigned_symmetric: bool):
        """ Use unsigned symmetric setter """
        self.use_unsigned_symmetric = use_unsigned_symmetric

    @property
    def use_strict_symmetric(self):
        """ Use strict symmetric getter """
        return self._use_strict_symmetric

    @use_strict_symmetric.setter
    def use_strict_symmetric(self, use_strict_symmetric: bool):
        """ Use strict symmetric setter """
        self._use_strict_symmetric = use_strict_symmetric


class QuantizeWrapperTransform(transforms.Transform):
    """ Transform for inserting quantize wrapper """

    def __init__(self, layer_type: str, activation_quant_settings: QuantizerSettings,
                 param_quant_settings: QuantizerSettings, name_to_module_map: Dict[str, tf.keras.layers.Layer]):
        super(QuantizeWrapperTransform, self).__init__()
        self._name_to_module_map = name_to_module_map
        self._layer_type = layer_type
        self._activation_quant_settings = activation_quant_settings
        self._param_quant_settings = param_quant_settings

    def pattern(self):
        """ Layer pattern to search for replacement """
        return transforms.LayerPattern(self._layer_type)

    def replacement(self, match_layer):
        """ Replacement method to create quant wrapper layer node """
        keras_module = self._name_to_module_map.get(match_layer.layer['config']['name'])
        if keras_module is not None:
            wrapper = QcQuantizeWrapper(keras_module, self._activation_quant_settings, self._param_quant_settings)
            wrapper_layer_config = tf.keras.layers.serialize(wrapper)
            wrapper_layer_config['name'] = wrapper.name
            wrapper_layer_node = transforms.LayerNode(wrapper_layer_config, weights=match_layer.weights)
            return wrapper_layer_node
        _logger.error('Layer to replace does not have associated keras module')
        raise AssertionError

    # pylint: disable=no-self-use
    def custom_objects(self):
        """ List of custom objects used in replacement method """
        return {'QcQuantizeWrapper': QcQuantizeWrapper}

class TensorQuantizer(tf.keras.layers.Layer):
    """ Tensor quantizer class containing cpp tensor quantizer and associated attributes """
    # pylint: disable=too-many-arguments
    def __init__(self, name, op_mode, quant_scheme, round_mode, bitwidth, is_symmetric, use_strict_symmetric,
                 use_unsigned_symmetric):
        super(TensorQuantizer, self).__init__()
        self._tensor_quantizer = libpymo.TensorQuantizer(quant_scheme, round_mode)
        self._tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
        self._tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)
        self._bitwidth = bitwidth
        self._is_symmetric = is_symmetric

        self._encoding_min = self.add_weight(name + '.encoding_min', dtype=tf.float64,
                                             initializer=tf.constant_initializer(0.))
        self._encoding_max = self.add_weight(name + '.encoding_max', dtype=tf.float64,
                                             initializer=tf.constant_initializer(0.))
        self._quantizer_mode = self.add_weight(name + '.op_mode', dtype=tf.int32, trainable=False,
                                               initializer=tf.constant_initializer(int(op_mode)))

    @property
    def tensor_quantizer(self):
        """ Tensor quantizer getter """
        return self._tensor_quantizer

    def compute_encoding(self):
        """ Compute encoding for the tensor quantizer """
        # TODO: remove last two parameters after fixing PyModelOptimizations
        encoding = self._tensor_quantizer.computeEncoding(self._bitwidth, self._is_symmetric, False, False)
        return encoding

    # pylint: disable=arguments-differ
    def call(self, tensor):
        """ Forward pass for tensor quantizer """
        return qcops.qc_quantize(name='qc_quantize_op', in_tensor=tensor,
                                 op_mode=self._quantizer_mode,
                                 tensor_quantizer_reference=libpymo.PtrToInt64(self._tensor_quantizer),
                                 encoding_min=self._encoding_min,
                                 encoding_max=self._encoding_max,
                                 bit_width=self._bitwidth,
                                 use_symmetric_encoding=self._is_symmetric)

class QcQuantizeWrapper(tf.keras.layers.Layer):
    """ Wrapper for simulating quantization noise """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 layer_to_wrap: tf.keras.layers.Layer,
                 activation_quant_settings: QuantizerSettings,
                 param_quant_settings: QuantizerSettings,
                 num_inputs: int = 1,
                 num_outputs: int = 1,
                 input_quantizers: Union[None, List[TensorQuantizer]] = None,
                 output_quantizers: Union[None, List[TensorQuantizer]] = None,
                 param_quantizers: Union[None, List[TensorQuantizer]] = None,
                 **kwargs):
        super(QcQuantizeWrapper, self).__init__(**kwargs)
        self._layer_to_wrap = layer_to_wrap
        self._activation_quant_settings = activation_quant_settings
        self._param_quant_settings = param_quant_settings

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self.input_quantizers = input_quantizers
        self.output_quantizers = output_quantizers
        self.param_quantizers = param_quantizers

        # Create quantizer variables and quantizers for input quantizers if not yet existing
        if self.input_quantizers is None:
            self.input_quantizers = []
            for i in range(self._num_inputs):
                self.input_quantizers.append(TensorQuantizer(self._layer_to_wrap.name + '_input_quantizer_' + str(i),
                                                             libpymo.TensorQuantizerOpMode.updateStats,
                                                             MAP_QUANT_SCHEME_TO_PYMO[
                                                                 self._activation_quant_settings.quant_scheme],
                                                             MAP_ROUND_MODE_TO_PYMO[
                                                                 self._activation_quant_settings.round_mode],
                                                             self._activation_quant_settings.bitwidth,
                                                             self._activation_quant_settings.is_symmetric,
                                                             self._activation_quant_settings.use_strict_symmetric,
                                                             self._activation_quant_settings.use_unsigned_symmetric))

        # Create quantizer variables and quantizers for outputs if not yet existing
        if self.output_quantizers is None:
            self.output_quantizers = []
            for i in range(self._num_outputs):
                self.output_quantizers.append(TensorQuantizer(self._layer_to_wrap.name + '_output_quantizer_' + str(i),
                                                              libpymo.TensorQuantizerOpMode.updateStats,
                                                              MAP_QUANT_SCHEME_TO_PYMO[
                                                                  self._activation_quant_settings.quant_scheme],
                                                              MAP_ROUND_MODE_TO_PYMO[
                                                                  self._activation_quant_settings.round_mode],
                                                              self._activation_quant_settings.bitwidth,
                                                              self._activation_quant_settings.is_symmetric,
                                                              self._activation_quant_settings.use_strict_symmetric,
                                                              self._activation_quant_settings.use_unsigned_symmetric))

        # Create quantizer variables and quantizers for params if not yet existing
        if self.param_quantizers is None:
            self.param_quantizers = []
            for weight in self._layer_to_wrap.weights:
                weight_name = weight.name.split(':')[0]
                self.param_quantizers.append(TensorQuantizer(weight_name,
                                                             libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize,
                                                             MAP_QUANT_SCHEME_TO_PYMO[
                                                                 self._activation_quant_settings.quant_scheme],
                                                             MAP_ROUND_MODE_TO_PYMO[
                                                                 self._activation_quant_settings.round_mode],
                                                             self._param_quant_settings.bitwidth,
                                                             self._param_quant_settings.is_symmetric,
                                                             self._param_quant_settings.use_strict_symmetric,
                                                             self._param_quant_settings.use_unsigned_symmetric))

    def get_config(self):
        """ Override get_config """
        return {"layer_to_wrap": self._layer_to_wrap,
                "activation_quant_settings": self._activation_quant_settings,
                "param_quant_settings": self._param_quant_settings,
                "num_inputs": self._num_inputs,
                "num_outputs": self._num_outputs,
                "name": self.name,
                "input_quantizers": self.input_quantizers,
                "output_quantizers": self.output_quantizers,
                "param_quantizers": self.param_quantizers}

    # pylint: disable=arguments-differ
    def call(self, inputs):
        """
        Forward-pass routine. This quantizes the weights before delegating to the wrapped module and then quantizes the
        output before returning the same
        :param inputs: Inputs passed to the module in the forward pass
        :return: Quantized output from the wrapped module
        """
        shadow_params = [tf.keras.backend.get_value(param) for param in self._layer_to_wrap.weights]
        self._quantize_params()
        inputs = self._quantize_activation(inputs, self.input_quantizers)
        outputs = self._layer_to_wrap(inputs)
        outputs = self._quantize_activation(outputs, self.output_quantizers)
        self._restore_shadow_params(shadow_params)
        return outputs

    def _quantize_params(self):
        """ Quantize parameters """
        for idx, param in enumerate(self._layer_to_wrap.weights):
            param_val = tf.keras.backend.get_value(param)
            quantized_param = self.param_quantizers[idx](param_val)
            self._layer_to_wrap.weights[idx].assign(quantized_param)

    @staticmethod
    def _quantize_activation(activation: Union[tf.Tensor, List], quantizers: List[TensorQuantizer]) -> \
            Union[tf.Tensor, List]:
        """
        Quantize activation
        :param activation: Activation tensor(s) to quantize
        :param quantizers: List of quantizers to use for quantizing activation
        :return: Quantized tensor(s)
        """
        if isinstance(activation, tf.Tensor):
            activation = [activation]

        if len(activation) != len(quantizers):
            _logger.error('Mismatch between number of tensors {%s} and number of quantizers {%s}', len(activation),
                          len(quantizers))
            raise AssertionError
        quantized_activations = []
        for idx, tensor in enumerate(activation):
            quantized_tensor = quantizers[idx](tensor)
            quantized_activations.append(quantized_tensor)
        if len(quantized_activations) == 1:
            quantized_activations = quantized_activations[0]
        return quantized_activations

    def _restore_shadow_params(self, shadow_params):
        """
        Restore saved parameters
        :param shadow_params: Original parameters to restore
        """
        for idx, param in enumerate(shadow_params):
            self._layer_to_wrap.weights[idx].assign(param)
