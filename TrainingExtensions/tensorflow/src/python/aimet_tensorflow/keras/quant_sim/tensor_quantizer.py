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
""" Tensor quantizer for tf 2 keras """
import abc
import tensorflow as tf

from aimet_common.defs import MAP_QUANT_SCHEME_TO_PYMO, MAP_ROUND_MODE_TO_PYMO, QuantScheme
from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.quant_sim.quantsim_straight_through_grad import qc_straight_through_estimator_grad, \
    quantsim_custom_grad_learned_grid
import libpymo  # pylint: disable=import-error

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

class TensorQuantizer(tf.keras.layers.Layer, abc.ABC):
    """ Tensor quantizer class containing cpp tensor quantizer and associated attributes """
    # pylint: disable=too-many-arguments
    # pylint: disable=unused-argument
    def __init__(self, name: str, op_mode: libpymo.TensorQuantizerOpMode, quant_scheme: libpymo.QuantizationMode,
                 round_mode: libpymo.RoundingMode, bitwidth: int, is_symmetric: bool, use_strict_symmetric: bool,
                 use_unsigned_symmetric: bool, **kwargs):
        super(TensorQuantizer, self).__init__(name=name)
        self._quant_scheme = quant_scheme
        self._tensor_quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[quant_scheme],
                                                         MAP_ROUND_MODE_TO_PYMO[round_mode])
        self._tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
        self._tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)
        self._bitwidth = self.add_weight(name + '.bitwidth', dtype=tf.int8,
                                         initializer=tf.constant_initializer(bitwidth), trainable=False)
        self._is_symmetric = self.add_weight(name + '.is_symmetric', dtype=tf.bool,
                                             initializer=tf.constant_initializer(is_symmetric), trainable=False)

        self._encoding = None

        self._encoding_min = self.add_weight(name + '.encoding_min', dtype=tf.float64, trainable=True,
                                             initializer=tf.constant_initializer(0.))
        self._encoding_max = self.add_weight(name + '.encoding_max', dtype=tf.float64, trainable=True,
                                             initializer=tf.constant_initializer(0.))
        self._quantizer_mode = self.add_weight(name + '.op_mode', dtype=tf.int32, trainable=False,
                                               initializer=tf.constant_initializer(int(op_mode)))

    @property
    def quant_scheme(self):
        """ Quant scheme getter """
        return self._quant_scheme

    @quant_scheme.setter
    def quant_scheme(self, quant_scheme: QuantScheme):
        """ Quant scheme setter """
        self._tensor_quantizer.setQuantScheme(MAP_QUANT_SCHEME_TO_PYMO[quant_scheme])
        self._quant_scheme = quant_scheme
        self.reset_encoding()

    @property
    def round_mode(self):
        """ Quant scheme getter """
        return self._tensor_quantizer.roundingMode

    @round_mode.setter
    def round_mode(self, round_mode: str):
        """ Round mode setter """
        self._tensor_quantizer.roundingMode = MAP_ROUND_MODE_TO_PYMO[round_mode]
        self.reset_encoding()

    @property
    def bitwidth(self):
        """ Bitwidth getter """
        return tf.keras.backend.get_value(self._bitwidth)

    @bitwidth.setter
    def bitwidth(self, bitwidth: int):
        """ Bitwidth setter """
        self._bitwidth.assign(bitwidth)
        self.reset_encoding()

    @property
    def is_symmetric(self):
        """ Is symmetric getter """
        return tf.keras.backend.get_value(self._is_symmetric)

    @is_symmetric.setter
    def is_symmetric(self, is_symmetric: bool):
        """ Is symmetric setter """
        self._is_symmetric.assign(is_symmetric)
        self.reset_encoding()

    @property
    def use_strict_symmetric(self):
        """ Use strict symmetric getter """
        return self._tensor_quantizer.getStrictSymmetric()

    @use_strict_symmetric.setter
    def use_strict_symmetric(self, use_strict_symmetric: bool):
        """ Use strict symmetric setter """
        self._tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
        self.reset_encoding()

    @property
    def use_unsigned_symmetric(self):
        """ Use unsigned symmetric getter """
        return self._tensor_quantizer.getUnsignedSymmetric()

    @use_unsigned_symmetric.setter
    def use_unsigned_symmetric(self, use_unsigned_symmetric: bool):
        """ Use unsigned symmetric setter """
        self._tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)
        self.reset_encoding()

    @property
    def encoding(self):
        """ Encoding getter """
        return self._encoding

    @property
    def quant_mode(self):
        """ Get quant mode"""
        return tf.keras.backend.get_value(self._quantizer_mode)

    @abc.abstractmethod
    def enable(self):
        """ Enable the tensor quantizer """

    def disable(self):
        """ Disable the tensor quantizer """
        self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.passThrough))

    def is_enabled(self) -> bool:
        """ Return True if the tensor quantizer is enabled, False otherwise """
        return self.quant_mode != int(libpymo.TensorQuantizerOpMode.passThrough)

    def compute_encoding(self):
        """ Compute encoding for the tensor quantizer """
        if self.quant_mode != int(libpymo.TensorQuantizerOpMode.passThrough):
            # TODO: remove last two parameters after fixing PyModelOptimizations
            encoding = self._tensor_quantizer.computeEncoding(self.bitwidth, self.is_symmetric, False, False)
            if self._tensor_quantizer.isEncodingValid:
                self._encoding = encoding
                self._encoding_min.assign(self._encoding.min)
                self._encoding_max.assign(self._encoding.max)
                if self.quant_mode == int(libpymo.TensorQuantizerOpMode.updateStats):
                    self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.quantizeDequantize))
            else:
                _logger.info('Tensor quantizer %s did not have a valid encoding calculated, and has been set to '
                             'passThrough mode.', self.name)
                self._encoding = None
                self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.passThrough))

    def reset_encoding(self):
        """ Reset the encoding to None, and reset quantizer mode if applicable """
        self._encoding = None
        if self.quant_mode == int(libpymo.TensorQuantizerOpMode.quantizeDequantize):
            self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.updateStats))

    # pylint: disable=arguments-differ
    def call(self, tensor):
        """
        Forward pass for the quantizer
        """
        if self.quant_scheme in [QuantScheme.training_range_learning_with_tf_init,
                                 QuantScheme.training_range_learning_with_tf_enhanced_init]:
            return self.call_quantsim_custom_grad_learned_grid(tensor)
        return self.call_quantize_straight_through_estimator_grad(tensor)

    @tf.custom_gradient
    def call_quantize_straight_through_estimator_grad(self, tensor):
        """
        Quantizes tensor with straight through estimator grad
        :param tensor: Tensor to quantize
        """
        def grad(upstream, variables):
            """
            Straight through estimator grad function
            :param upstream: Gradient from child layers
            :param variables: Variables used in forward pass to return gradients for
            """
            assert len(variables) == 2, 'len variables is ' + str(len(variables))
            assert 'encoding_min' in variables[0].name
            return qc_straight_through_estimator_grad(tensor, self._encoding_min, self._encoding_max,
                                                      self._quantizer_mode, upstream)

        return qcops.qc_quantize(name='qc_quantize_op', in_tensor=tensor,
                                 op_mode=self._quantizer_mode,
                                 tensor_quantizer_reference=libpymo.PtrToInt64(self._tensor_quantizer),
                                 encoding_min=self._encoding_min,
                                 encoding_max=self._encoding_max,
                                 bit_width=self._bitwidth,
                                 use_symmetric_encoding=self._is_symmetric), grad

    @tf.custom_gradient
    def call_quantsim_custom_grad_learned_grid(self, tensor):
        """
        Quantizes tensor with range learning grad
        :param tensor: Tensor to quantize
        """
        def grad(upstream, variables):
            """
            Range learning grad function
            :param upstream: Gradient from child layers
            :param variables: Variables used in forward pass to return gradients for
            """
            assert len(variables) == 2, 'len variables is ' + str(len(variables))
            assert 'encoding_min' in variables[0].name
            return quantsim_custom_grad_learned_grid(tensor, self._encoding_min, self._encoding_max,
                                                     self._quantizer_mode, self._bitwidth, self._is_symmetric,
                                                     upstream)

        return qcops.qc_quantize(name='qc_quantize_op', in_tensor=tensor,
                                 op_mode=self._quantizer_mode,
                                 tensor_quantizer_reference=libpymo.PtrToInt64(self._tensor_quantizer),
                                 encoding_min=self._encoding_min,
                                 encoding_max=self._encoding_max,
                                 bit_width=self._bitwidth,
                                 use_symmetric_encoding=self._is_symmetric), grad

# pylint: disable=too-many-ancestors
class ActivationTensorQuantizer(TensorQuantizer):
    """ Activation tensor quantizer definition """
    # pylint: disable=too-many-arguments
    def __init__(self, name: str, quant_scheme: libpymo.QuantizationMode,
                 round_mode: libpymo.RoundingMode, bitwidth: int, is_symmetric: bool, use_strict_symmetric: bool,
                 use_unsigned_symmetric: bool, enabled: bool):
        if enabled:
            op_mode = libpymo.TensorQuantizerOpMode.updateStats
        else:
            op_mode = libpymo.TensorQuantizerOpMode.passThrough
        super(ActivationTensorQuantizer, self).__init__(name, op_mode, quant_scheme, round_mode, bitwidth, is_symmetric,
                                                        use_strict_symmetric, use_unsigned_symmetric)

    def enable(self):
        """ Enable the activation tensor quantizer """
        if self._encoding is not None:
            self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.quantizeDequantize))
        else:
            self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.updateStats))


# pylint: disable=too-many-ancestors
class ParamTensorQuantizer(TensorQuantizer):
    """ Parameter tensor quantizer definition """
    # pylint: disable=too-many-arguments
    def __init__(self, name: str, quant_scheme: libpymo.QuantizationMode,
                 round_mode: libpymo.RoundingMode, bitwidth: int, is_symmetric: bool, use_strict_symmetric: bool,
                 use_unsigned_symmetric: bool, enabled: bool):
        if enabled:
            op_mode = libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize
        else:
            op_mode = libpymo.TensorQuantizerOpMode.passThrough
        super(ParamTensorQuantizer, self).__init__(name, op_mode, quant_scheme, round_mode, bitwidth, is_symmetric,
                                                   use_strict_symmetric, use_unsigned_symmetric)
    def enable(self):
        """ Enable the parameter tensor quantizer """
        self._quantizer_mode.assign(int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize))
