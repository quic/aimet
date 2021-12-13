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
from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms

from aimet_common.utils import AimetLogger
from aimet_common.defs import MAP_QUANT_SCHEME_TO_PYMO, MAP_ROUND_MODE_TO_PYMO, QuantScheme
import libpymo

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

class TensorQuantizer():
    """ Tensor quantizer class containing cpp tensor quantizer and associated attributes """
    def __init__(self, quant_scheme, round_mode, quantizer_mode, bitwidth, is_symmetric):
        self._tensor_quantizer = libpymo.TensorQuantizer(quant_scheme, round_mode)
        self._quantizer_mode = quantizer_mode
        self.bitwidth = bitwidth
        self.is_symmetric = is_symmetric

    @property
    def quantizer_mode(self):
        """ Quantizer mode getter """
        return self._quantizer_mode

    @property
    def tensor_quantizer(self):
        """ Tensor quantizer getter """
        return self._tensor_quantizer

    def compute_encoding(self):
        """ Compute encoding for the tensor quantizer """
        encoding = self._tensor_quantizer.computeEncoding(self.bitwidth, self.is_symmetric, False, False)
        return encoding

class QuantizerVars:
    """ Object holding quantizer variables """
    def __init__(self, encoding_min_var: tf.Variable, encoding_max_var: tf.Variable, op_mode_var: tf.Variable):
        self.encoding_min_var = encoding_min_var
        self.encoding_max_var = encoding_max_var
        self.op_mode_var = op_mode_var

class QcQuantizeWrapper(tf.keras.layers.Layer):
    """ Wrapper for simulating quantization noise """
    def __init__(self,
                 layer_to_wrap: tf.keras.layers.Layer,
                 activation_quant_settings: QuantizerSettings,
                 param_quant_settings: QuantizerSettings,
                 num_inputs: int = 1,
                 num_outputs: int = 1,
                 **kwargs):
        super(QcQuantizeWrapper, self).__init__(**kwargs)
        self._layer_to_wrap = layer_to_wrap
        self._activation_quant_settings = activation_quant_settings
        self._param_quant_settings = param_quant_settings

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self.input_quantizers = []
        self.param_quantizers = {}
        self.output_quantizers = []
        self.quantizer_info_dict = NoDependency({})

        for i in range(num_inputs):
            encoding_min = self.add_weight('inputs.' + str(i) + '.encoding_min', dtype=tf.float64,
                                           initializer=tf.constant_initializer(0.))
            encoding_max = self.add_weight('inputs.' + str(i) + '.encoding_max', dtype=tf.float64,
                                           initializer=tf.constant_initializer(0.))
            quantizer_mode = self.add_weight('inputs.' + str(i) + '.op_mode', dtype=tf.int32,
                                             trainable=False,
                                             initializer=tf.constant_initializer(
                                                 int(libpymo.TensorQuantizerOpMode.updateStats)))
            tensor_quantizer = TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[self._activation_quant_settings.quant_scheme],
                                               MAP_ROUND_MODE_TO_PYMO[self._activation_quant_settings.round_mode],
                                               libpymo.TensorQuantizerOpMode.updateStats,
                                               self._activation_quant_settings.bitwidth,
                                               self._activation_quant_settings.is_symmetric)
            # pylint: disable=unsupported-assignment-operation
            self.quantizer_info_dict[tensor_quantizer] = QuantizerVars(encoding_min, encoding_max, quantizer_mode)
            self.input_quantizers.append(tensor_quantizer)

        for i in range(num_outputs):
            encoding_min = self.add_weight('outputs.' + str(i) + '.encoding_min', dtype=tf.float64,
                                           initializer=tf.constant_initializer(0.))
            encoding_max = self.add_weight('outputs.' + str(i) + '.encoding_max', dtype=tf.float64,
                                           initializer=tf.constant_initializer(0.))
            quantizer_mode = self.add_weight('outputs.' + str(i) + '.op_mode', dtype=tf.int32,
                                             trainable=False,
                                             initializer=tf.constant_initializer(
                                                 int(libpymo.TensorQuantizerOpMode.updateStats)))
            tensor_quantizer = TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[self._activation_quant_settings.quant_scheme],
                                               MAP_ROUND_MODE_TO_PYMO[self._activation_quant_settings.round_mode],
                                               libpymo.TensorQuantizerOpMode.updateStats,
                                               self._activation_quant_settings.bitwidth,
                                               self._activation_quant_settings.is_symmetric)
            # pylint: disable=unsupported-assignment-operation
            self.quantizer_info_dict[tensor_quantizer] = QuantizerVars(encoding_min, encoding_max, quantizer_mode)
            self.output_quantizers.append(tensor_quantizer)

    def get_config(self):
        """ Override get_config """
        return {"layer_to_wrap": self._layer_to_wrap,
                "activation_quant_settings": self._activation_quant_settings,
                "param_quant_settings": self._param_quant_settings,
                "num_inputs": self._num_inputs,
                "num_outputs": self._num_outputs,
                "name": self.name}

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
            # Perform parameter quantization here
            quantized_param = param_val
            self._layer_to_wrap.weights[idx].assign(quantized_param)

    def _quantize_activation(self, activation: Union[tf.Tensor, List], quantizers: List[TensorQuantizer]) -> \
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
            # pylint: disable=unsubscriptable-object
            quantizer_info = self.quantizer_info_dict[quantizers[idx]]
            quantized_tensor = qcops.qc_quantize(name='qc_quantize_op', in_tensor=tensor,
                                                 op_mode=quantizer_info.op_mode_var,
                                                 tensor_quantizer_reference=libpymo.PtrToInt64(
                                                     quantizers[idx].tensor_quantizer),
                                                 encoding_min=quantizer_info.encoding_min_var,
                                                 encoding_max=quantizer_info.encoding_max_var,
                                                 bit_width=quantizers[idx].bitwidth,
                                                 use_symmetric_encoding=quantizers[idx].is_symmetric)
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
