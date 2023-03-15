# /usr/bin/env python3.8
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
""" Custom QcQuantizeOp to quantize weights and activations using ONNXRuntime """

from typing import Union
import aimet_common.libpymo as libpymo
from aimet_common.libpymo import TensorQuantizerOpMode
from aimet_common.defs import QuantScheme, MAP_QUANT_SCHEME_TO_PYMO, MAP_ROUND_MODE_TO_PYMO, QuantizationDataType
from aimet_common import libquant_info


OpMode = TensorQuantizerOpMode


class QcQuantizeOp:
    """ A custom quantization operation to perform using ONNXRuntime """

    # pylint: disable=too-many-arguments
    def __init__(self, quant_info: libquant_info.QcQuantizeInfo,
                 quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                 rounding_mode: str = 'nearest',
                 encodings: Union[libpymo.TfEncoding, None] = None,
                 op_mode: Union[OpMode, None] = None,
                 bitwidth: int = 8, use_symmetric_encodings: bool = False):
        """
        Args:
            quant_info: libquant_info.QcQuantizeInfo object holding quantization parameters passed to the C++ op
            quant_scheme: Quantization scheme (e.g. QuantScheme.post_training_tf)
            rounding_mode: Rounding mode (e.g. nearest)
            encodings: libpymo.TfEncoding object with min, max, offset, delta, bw
            op_mode: QcQuantizeOp mode (e.g. update_stats)
            bitwidth: Quantization bitwidth
            use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
        """
        self.quant_info = quant_info
        self.quant_scheme = quant_scheme
        self.rounding_mode = rounding_mode
        self.set_tensor_quantizer(self._build_tensor_quantizer())
        self.op_mode = op_mode
        self.bitwidth = bitwidth
        self.encodings = encodings
        self.use_symmetric_encodings = use_symmetric_encodings
        self.enabled = True
        self._data_type = QuantizationDataType.int

    @property
    def data_type(self) -> QuantizationDataType:
        """
        Returns the data type for quantization

        :return: Quantization data type
        """
        return self._data_type

    @data_type.setter
    def data_type(self, data_type: QuantizationDataType):
        """
        Sets the quantization data type field in the op and sets isIntDataType inside quantizer_info to true or false
        based on the data type

        :param data_type: Quantization data type
        """
        self._data_type = data_type
        self.quant_info.isIntDataType = False
        if data_type == QuantizationDataType.int:
            self.quant_info.isIntDataType = True

    def _build_tensor_quantizer(self):
        return libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[self.quant_scheme],
                                       MAP_ROUND_MODE_TO_PYMO[self.rounding_mode])

    def set_tensor_quantizer(self, tensor_quantizer: libpymo.TensorQuantizer):
        """
        Stores tensor_quantizer in self._tensor_quantizer and passes a pointer to the object
        to the C++ op's QcQuantInfo object
        :param tensor_quantizer: The libpymo.TensorQuantizer object to give to the C++ op
        """
        self._tensor_quantizer = tensor_quantizer
        self.quant_info.tensorQuantizerRef = libpymo.PtrToInt64(tensor_quantizer)

    @property
    def enabled(self) -> bool:
        """
        If False, quant_info.OpMode will be overriden with OpMode.passThrough to prevent quantization
        :return: True if the quantizer is to be utilized, False otherwise
        """
        return self.quant_info.enabled

    @enabled.setter
    def enabled(self, enable: bool):
        """
        Set the value of enabled to be accessed by the C++ op
        :param enable: True if the op is to be utilized, False will override the OpMode with passThrough
        """
        self.quant_info.enabled = enable

    @property
    def use_symmetric_encodings(self) -> bool:
        """
        Reads useSymmetricEncoding from the node's QcQuantizeInfo object
        :return: True if the node is to use symmetric encodings
        """
        return self.quant_info.useSymmetricEncoding

    @use_symmetric_encodings.setter
    def use_symmetric_encodings(self, use_symmetric_encodings: bool):
        """
        Sets the useSymmetricEncoding attribute of the nodes QcQuantizeInfo object
        :param use_symmetric_encodings: True if the node is to use symmetric encodings
        """
        self.quant_info.useSymmetricEncoding = use_symmetric_encodings

    @property
    def use_strict_symmetric(self) -> bool:
        """
        Reads useStrictSymmetric config from Tensor Quantizer
        :return: True if strict symmetric mode is to be used, False otherwise
        """
        return self._tensor_quantizer.getStrictSymmetric()

    @use_strict_symmetric.setter
    def use_strict_symmetric(self, use_strict_symmetric: bool):
        """
        Sets the useStrictSymmetric associated with the Tensor Quantizer
        :param use_strict_symmetric: True if strict symmetric mode is to be used, False otherwise
        """
        self._tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
        self.encodings = None

    @property
    def use_unsigned_symmetric(self) -> bool:
        """
        Reads useStrictSymmetric config from Tensor Quantizer
        :return: True if unsigned symmetric mode is to be used, False otherwise
        """
        return self._tensor_quantizer.getUnsignedSymmetric()

    @use_unsigned_symmetric.setter
    def use_unsigned_symmetric(self, use_unsigned_symmetric: bool):
        """
        Sets the useUnsignedSymmetric associated with the Tensor Quantizer
        :param use_unsigned_symmetric: True if unsigned symmetric mode is to be used, False otherwise
        """
        self._tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)
        self.encodings = None

    @property
    def encodings(self) -> libpymo.TfEncoding:
        """
        Reads the encodings object from the node's QcQuantizeInfo
        :return: The libpymo.TfEncoding object used to store the node's quantization encoding
        """
        return self.quant_info.encoding

    @encodings.setter
    def encodings(self, encoding: Union[libpymo.TfEncoding, None]):
        """
        Stores encoding in self._encoding to prevent deletion and sets self.quant_info.encoding to point to encoding.
        If encoding is None, creates an empty encoding to prevent seg faults
        :param encoding: The libpymo.TfEncoding object to be used by the C++ op
        """
        if encoding is None:
            encoding = libpymo.TfEncoding()
            self._tensor_quantizer.isEncodingValid = False
        else:
            self._tensor_quantizer.isEncodingValid = True
        self._encoding = encoding
        self.quant_info.encoding = encoding
        self.quant_info.encoding.bw = self.bitwidth

    @property
    def op_mode(self) -> OpMode:
        """
        Reads the OpMode from the node's quant_info object
        :return: The node's current mode of operation
        """
        return self.quant_info.opMode

    @op_mode.setter
    def op_mode(self, op_mode: OpMode):
        """
        Sets the opMode field in the node's quant_info
        :param op_mode: The OpMode to be used
        """
        self.quant_info.opMode = op_mode

    def reset_encoding_stats(self):
        """
        reset the stats of tensor quantizer
        """
        self.encodings = None
        self._tensor_quantizer.resetEncodingStats()

    def set_bitwidth(self, bitwidth: int):
        """
        Set bitwidth for quantization
        """
        self.bitwidth = bitwidth
        self.reset_encoding_stats()

    def set_quant_scheme(self, quant_scheme: QuantScheme):
        """
        Set QcQuantizeOp as given quant scheme
        """
        self.quant_scheme = quant_scheme
        self.set_tensor_quantizer(self._build_tensor_quantizer())
        self.reset_encoding_stats()

    def compute_encodings(self) -> libpymo.TfEncoding:
        """
        Compute and return encodings of each tensor quantizer
        """
        self.encodings = self._tensor_quantizer.computeEncoding(self.bitwidth,
                                                               self.use_symmetric_encodings)
        return self.encodings
