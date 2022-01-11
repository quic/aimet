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

""" Qunatizer Info """

import io
from enum import Enum
import tensorflow as tf

from aimet_common.defs import QuantScheme
from aimet_common.quantsim import calculate_delta_offset
from aimet_tensorflow.utils.constants import QuantizeOpIndices
import libpymo


quant_scheme_to_libpymo = {QuantScheme.post_training_tf: libpymo.QuantizationMode.QUANTIZATION_TF,
                           QuantScheme.post_training_tf_enhanced:
                               libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                           QuantScheme.training_range_learning_with_tf_init:
                               libpymo.QuantizationMode.QUANTIZATION_TF,
                           QuantScheme.training_range_learning_with_tf_enhanced_init:
                               libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED}


class QuantizerType(Enum):
    """ Enum for quantize op types """
    param = 0
    activation = 1


class PickleableTensorQuantizerState:
    """
    State variables to be saved while pickling tensor quantizer
    """
    def __init__(self, quant_op_name, tensor_quantizer_ref, quantizer_type):
        """
        class type to save pickle-able info pertaining to tensor quantizer
        :param quant_op_name: name of the quantize op
        :param tensor_quantizer_ref: TensorQuantizer reference
        :param quantizer_type : param or activation quantizer
        """

        self.quant_op_name = quant_op_name
        self.quantizer_type = quantizer_type
        self.num_channels = 0
        if isinstance(tensor_quantizer_ref, list):
            for tensor_quantizer in tensor_quantizer_ref:
                self.quant_scheme = tensor_quantizer.getQuantScheme()
                self.use_strict_symmetric = tensor_quantizer.getStrictSymmetric()
                self.use_unsigned_symmetric = tensor_quantizer.getUnsignedSymmetric()
                self.rounding_mode = tensor_quantizer.roundingMode
                self.is_encoding_valid = tensor_quantizer.isEncodingValid
                self.num_channels = len(tensor_quantizer_ref)
                break
        else:
            self.quant_scheme = tensor_quantizer_ref.getQuantScheme()
            self.use_strict_symmetric = tensor_quantizer_ref.getStrictSymmetric()
            self.use_unsigned_symmetric = tensor_quantizer_ref.getUnsignedSymmetric()
            self.rounding_mode = tensor_quantizer_ref.roundingMode
            self.is_encoding_valid = tensor_quantizer_ref.isEncodingValid


class QuantizerInfo:
    """
    Holds information about a given MO Quantizer object and active session
    """
    __slots__ = ['session', 'tensor_quantizer', 'quant_op_name', 'quantizer_type', '_is_encoding_frozen']

    def __init__(self, session: tf.compat.v1.Session, tensor_quantizer: libpymo.TensorQuantizer,
                 quant_op_name: str, quantizer_type: QuantizerType):
        self.session = session
        self.tensor_quantizer = tensor_quantizer
        self.quant_op_name = quant_op_name
        self.quantizer_type = quantizer_type
        self._is_encoding_frozen = False

    def set_variable(self, var_name, value):
        """
        sets Quantize op variable with value passed
        :param var_name: Name of the variable to be updated
        :param value: value to be assigned to the variable
        """
        with self.session.graph.as_default():
            vars_with_given_name = [var for var in tf.compat.v1.global_variables()
                                    if var.op.name == var_name]
        var_to_be_updated = vars_with_given_name[0]
        var_to_be_updated.load(value, self.session)

    def get_variable_from_op(self, var_index):
        """
        Reads variable from Quantize op
        :param var_index: Quantize op input param index corresponding to the variable to be read
        :return: variable value read from the Quantize op
        """
        quantize_op = self.session.graph.get_operation_by_name(self.quant_op_name)
        op_var_tensor = quantize_op.inputs[var_index]
        return self.session.run(op_var_tensor)

    @property
    def bitwidth(self) -> int:
        """
        Reads bitwidth from the Quantize op
        :return: returns the bitiwdth associated with Quantize op
        """
        # return the variable value from op
        return self.get_variable_from_op(QuantizeOpIndices.bit_width)

    @bitwidth.setter
    def bitwidth(self, bitwidth: int):
        """
        Sets the bitwidth in the Quantize op
        :param bitwidth: value to be assigned to bitwidth variable
        """
        var_name = self.quant_op_name + '_bit_width'
        self.set_variable(var_name, bitwidth)
        if isinstance(self.tensor_quantizer, list):
            for tensor_quantizer in self.tensor_quantizer:
                tensor_quantizer.isEncodingValid = False
        else:
            self.tensor_quantizer.isEncodingValid = False

    @property
    def use_symmetric_encoding(self) -> bool:
        """
        Reads use_symmetric_encoding flag in the Quantize op
        :return: use_symmetric_encoding config as bool
        """
        return self.get_variable_from_op(QuantizeOpIndices.use_symmetric_encoding)

    @use_symmetric_encoding.setter
    def use_symmetric_encoding(self, use_symmetric_encoding: bool):
        """
        Sets the use_symmetric_encoding flag in the Quantize op
        :param use_symmetric_encoding: value to be assigned to use_symmetric_encoding flag
        """
        var_name = self.quant_op_name + '_use_symmetric_encoding'
        self.set_variable(var_name, use_symmetric_encoding)
        if isinstance(self.tensor_quantizer, list):
            for tensor_quantizer in self.tensor_quantizer:
                tensor_quantizer.isEncodingValid = False
        else:
            self.tensor_quantizer.isEncodingValid = False

    @property
    def quant_scheme(self) -> libpymo.QuantizationMode:
        """
        Reads the quant_scheme associated with the Quantize op
        :return: quant_scheme as libpymo.QuantizationMode type
        """
        if isinstance(self.tensor_quantizer, list):
            for tensor_quantizer in self.tensor_quantizer:
                return tensor_quantizer.getQuantScheme()
        else:
            return self.tensor_quantizer.getQuantScheme()

    @quant_scheme.setter
    def quant_scheme(self, quant_scheme: libpymo.QuantizationMode):
        """
        Sets the quant_scheme associated with the Quantize op
        :param quant_scheme: value to be assigned to quant_scheme param in Quantizer
        """
        if isinstance(self.tensor_quantizer, list):
            for tensor_quantizer in self.tensor_quantizer:
                tensor_quantizer.setQuantScheme(quant_scheme_to_libpymo[quant_scheme])
        else:
            self.tensor_quantizer.setQuantScheme(quant_scheme_to_libpymo[quant_scheme])

    @property
    def rounding_mode(self) -> libpymo.RoundingMode:
        """
        Reads rounding_mode associated with the Quantize op
        :return: rounding_mode value as libpymo.RoundingMode type
        """
        if isinstance(self.tensor_quantizer, list):
            for tensor_quantizer in self.tensor_quantizer:
                return tensor_quantizer.roundingMode
        else:
            return self.tensor_quantizer.roundingMode

    @rounding_mode.setter
    def rounding_mode(self, rounding_mode: libpymo.RoundingMode):
        """
        Sets the rounding_mode associated with the Quantize op
        :param rounding_mode: value to be assigned to rounding_mode param in Quantizer
        """
        if isinstance(self.tensor_quantizer, list):
            for tensor_quantizer in self.tensor_quantizer:
                tensor_quantizer.isEncodingValid = False
                tensor_quantizer.roundingMode = rounding_mode
        else:
            self.tensor_quantizer.isEncodingValid = False
            self.tensor_quantizer.roundingMode = rounding_mode

    @property
    def use_strict_symmetric(self) -> bool:
        """
        Reads useStrictSymmetric config from Tensor Quantizer
        :return: True if strict symmetric mode is to be used, False otherwise
        """
        if isinstance(self.tensor_quantizer, list):
            for tensor_quantizer in self.tensor_quantizer:
                return tensor_quantizer.getStrictSymmetric()
        else:
            return self.tensor_quantizer.getStrictSymmetric()

    @use_strict_symmetric.setter
    def use_strict_symmetric(self, use_strict_symmetric: bool):
        """
        Sets the useStrictSymmetric associated with the Tensor Quantizer
        :param use_strict_symmetric: True if strict symmetric mode is to be used, False otherwise
        """
        if isinstance(self.tensor_quantizer, list):
            for tensor_quantizer in self.tensor_quantizer:
                tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
        else:
            self.tensor_quantizer.setStrictSymmetric(use_strict_symmetric)

    @property
    def use_unsigned_symmetric(self) -> bool:
        """
        Reads useStrictSymmetric config from Tensor Quantizer
        :return: True if unsigned symmetric mode is to be used, False otherwise
        """
        if isinstance(self.tensor_quantizer, list):
            for tensor_quantizer in self.tensor_quantizer:
                return tensor_quantizer.getUnsignedSymmetric()
        else:
            return self.tensor_quantizer.getUnsignedSymmetric()

    @use_unsigned_symmetric.setter
    def use_unsigned_symmetric(self, use_unsigned_symmetric: bool):
        """
        Sets the useUnsignedSymmetric associated with the Tensor Quantizer
        :param use_unsigned_symmetric: True if unsigned symmetric mode is to be used, False otherwise
        """
        if isinstance(self.tensor_quantizer, list):
            for tensor_quantizer in self.tensor_quantizer:
                tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)
        else:
            self.tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)

    def get_op_mode(self) -> libpymo.TensorQuantizerOpMode:
        """
        Reads op mode variable from Quantize op
        :return: Op mode as pymo.TensorQuantizerOpMode type
        """
        op = self.session.graph.get_operation_by_name(self.quant_op_name)
        op_mode_tensor = op.inputs[QuantizeOpIndices.op_mode]
        return self.session.run(op_mode_tensor)

    def set_op_mode(self, op_mode: libpymo.TensorQuantizerOpMode):
        """
        Set op mode for Quantize op
        :param op_mode: Op mode as pymo.TensorQuantizerOpMode type
        """
        if not self._is_encoding_frozen:
            var_name = self.quant_op_name + '_op_mode'
            self.set_variable(var_name, int(op_mode))

    @property
    def enabled(self) -> bool:
        """
        Reads Quantize op flag that indicates if op is enabled or disabled
        :return: bool
        """
        is_enabled = True
        # return the variable value from op
        if self.get_op_mode() == int(libpymo.TensorQuantizerOpMode.passThrough):
            is_enabled = False
        return is_enabled

    @enabled.setter
    def enabled(self, enabled: bool):
        """
         Enables or disables given Quantize op if enabled is False
        :param enabled: boolean flag to indicate enable or disable
        """
        # if disable is requested on the op and this op was not already in "passThrough" mode,
        # we will disable the op by marking it as "passThrough"
        if not enabled and self.get_op_mode() != int(libpymo.TensorQuantizerOpMode.passThrough):
            op_mode = int(libpymo.TensorQuantizerOpMode.passThrough)
            # update the isEncodingValid state to False
            if isinstance(self.tensor_quantizer, list):
                for tensor_quantizer in self.tensor_quantizer:
                    tensor_quantizer.isEncodingValid = False
            else:
                self.tensor_quantizer.isEncodingValid = False
        # if enable is requested and this op was previously disabled
        # we enable the op by setting the initial op_mode that depends on the Quantizer type
        elif enabled and self.get_op_mode() == int(libpymo.TensorQuantizerOpMode.passThrough):
            if self.quantizer_type is QuantizerType.param:
                op_mode = int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize)
            elif self.quantizer_type is QuantizerType.activation:
                op_mode = int(libpymo.TensorQuantizerOpMode.updateStats)
            # update the isEncodingValid state to False
            if isinstance(self.tensor_quantizer, list):
                for tensor_quantizer in self.tensor_quantizer:
                    tensor_quantizer.isEncodingValid = False
            else:
                self.tensor_quantizer.isEncodingValid = False

        var_name = self.quant_op_name + '_op_mode'
        self.set_variable(var_name, op_mode)

    def compute_encoding(self, bitwidth: int, use_symmetric_encodings: bool) -> libpymo.TfEncoding:
        """
        Compute the quantization encoding for this tensor
        :param bitwidth: Quantization bitwidth
        :param use_symmetric_encodings: True if symmetric encoding is used. False otherwise.
        :return: Encoding
        """
        if not self._is_encoding_frozen:
            encoding = []
            if isinstance(self.tensor_quantizer, list):
                for tensor_quantizer in self.tensor_quantizer:
                    encoding.append(tensor_quantizer.computeEncoding(bitwidth, use_symmetric_encodings, False, False))
            else:
                encoding.append(self.tensor_quantizer.computeEncoding(bitwidth, use_symmetric_encodings, False, False))
                encoding = encoding[0]
        else:
            encoding = self.get_encoding()

        return encoding

    def set_encoding(self, encoding: libpymo.TfEncoding):
        """
        Set encoding min and max variable and update isEncodingValid state to True
        :param encoding: Encoding
        """
        if not self._is_encoding_frozen:
            encoding_min_var = self.quant_op_name + '_encoding_min'
            encoding_max_var = self.quant_op_name + '_encoding_max'

            # update the isEncodingValid state to True as well as encoding variable in the TF op
            if isinstance(self.tensor_quantizer, list):
                encoding_min = []
                encoding_max = []
                for index, tensor_quantizer in enumerate(self.tensor_quantizer):
                    tensor_quantizer.isEncodingValid = True
                    encoding_min.append(encoding[index].min)
                    encoding_max.append(encoding[index].max)
                self.set_variable(encoding_min_var, encoding_min)
                self.set_variable(encoding_max_var, encoding_max)
            else:
                self.tensor_quantizer.isEncodingValid = True
                self.set_variable(encoding_min_var, encoding.min)
                self.set_variable(encoding_max_var, encoding.max)


    def get_encoding(self) -> libpymo.TfEncoding:
        """
        Get encoding if valid else raise error
        :return: encoding
        """
        if self.is_encoding_valid():
            encoding_min = self.get_variable_from_op(QuantizeOpIndices.encoding_min)
            encoding_max = self.get_variable_from_op(QuantizeOpIndices.encoding_max)
            bitwidth = self.bitwidth

            # If per channel quantization is enabled then we need to create a list of TF encoding objects
            if isinstance(encoding_min, list):
                encoding = []
                for i, encoding_min_val in enumerate(encoding_min):
                    _encoding = libpymo.TfEncoding()
                    _encoding.min = encoding_min_val
                    _encoding.max = encoding_max[i]
                    _encoding.bw = bitwidth
                    _encoding.delta, _encoding.offset = calculate_delta_offset(encoding_min_val, encoding_max[i],
                                                                               bitwidth)
                    encoding.append(_encoding)
            else:
                encoding = libpymo.TfEncoding()
                encoding.min = encoding_min
                encoding.max = encoding_max
                encoding.bw = bitwidth
                encoding.delta, encoding.offset = calculate_delta_offset(encoding_min, encoding_max, bitwidth)
        else:
            raise AssertionError('Compute encoding or Set encoding must be invoked before')

        return encoding

    def freeze_encoding(self):
        """
        Set is_encoding_frozen flag to True
        """
        self._is_encoding_frozen = True

    def set_and_freeze_encoding_and_op_mode(self, encoding: libpymo.TfEncoding, op_mode: libpymo.TensorQuantizerOpMode):
        """
        Set encoding min and max variable, op_mode and freezes it
        :param encoding: Encoding
        :param op_mode: Op mode as pymo.TensorQuantizerOpMode type
        """
        self.set_encoding(encoding)
        self.set_op_mode(op_mode)
        self.freeze_encoding()

    def is_encoding_valid(self) -> bool:
        """
        Return bool if encoding is valid or not
        :return: Boolean
        """
        if isinstance(self.tensor_quantizer, list):
            for tensor_quantizer in self.tensor_quantizer:
                return tensor_quantizer.isEncodingValid
        else:
            return self.tensor_quantizer.isEncodingValid

    def __getstate__(self):
        # convert tensor quantizer state to pickle-able form
        state = PickleableTensorQuantizerState(self.quant_op_name,
                                               self.tensor_quantizer,
                                               self.quantizer_type)
        return state

    def __setstate__(self, state):
        self.session = None
        # Create the cpp tensor quantizer reference
        self.quant_op_name = state.quant_op_name
        self.quantizer_type = state.quantizer_type
        # If per channel quantization is enabled for a parameter
        if state.num_channels > 0:
            self.tensor_quantizer = []
            for _ in range(state.num_channels):
                tensor_quantizer = libpymo.TensorQuantizer(state.quant_scheme,
                                                           state.rounding_mode)
                tensor_quantizer.setStrictSymmetric(state.use_strict_symmetric)
                tensor_quantizer.setUnsignedSymmetric(state.use_unsigned_symmetric)
                tensor_quantizer.isEncodingValid = state.is_encoding_valid
                self.tensor_quantizer.append(tensor_quantizer)
        else:
            self.tensor_quantizer = libpymo.TensorQuantizer(state.quant_scheme,
                                                            state.rounding_mode)
            self.tensor_quantizer.setStrictSymmetric(state.use_strict_symmetric)
            self.tensor_quantizer.setUnsignedSymmetric(state.use_unsigned_symmetric)
            self.tensor_quantizer.isEncodingValid = state.is_encoding_valid

    def __str__(self):
        stream = io.StringIO(newline='\n')
        stream.write('Quantizer Info:\n')
        stream.write(' quantize_op_name:{}\n quantizer_type:{}\n bitwidth={}\n use_symmetric_encoding={}\n'
                     ' round_mode={}\n quant_scheme={}\n use_strict_symmetric={}\n use_unsigned_symmetric={}\n'
                     ' enabled:{}\n'.format(self.quant_op_name,
                                            self.quantizer_type,
                                            self.bitwidth,
                                            self.use_symmetric_encoding,
                                            self.rounding_mode,
                                            self.quant_scheme,
                                            self.use_strict_symmetric,
                                            self.use_unsigned_symmetric,
                                            self.enabled))

        return stream.getvalue()
