# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

from typing import Union, List, Optional
import aimet_common.libpymo as libpymo
from aimet_common.libpymo import TensorQuantizerOpMode
from aimet_common.defs import QuantScheme, MAP_QUANT_SCHEME_TO_PYMO, MAP_ROUND_MODE_TO_PYMO, QuantizationDataType
from aimet_common import libquant_info
from aimet_common.utils import deprecated


OpMode = TensorQuantizerOpMode


class TensorQuantizerParams:
    """
    Per channel quantization parameters
    """
    def __init__(self, tensor_shape, channel_axis: int = None, block_axis: int = None):
        """

        :param tensor_shape: Shape of the input tensor
        :param channel_axis: Axis along which per channel quantization is performed
        :param block_axis: Axis along which blockwise quantization is performed
        """
        self.tensor_shape = tensor_shape
        self.channel_axis = channel_axis
        self.block_axis = block_axis


class QcQuantizeOp:
    """ A custom quantization operation to perform using ONNXRuntime """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, quant_info: libquant_info.QcQuantizeInfo,
                 quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                 rounding_mode: str = 'nearest',
                 op_mode: Union[OpMode, None] = None,
                 bitwidth: int = 8, use_symmetric_encodings: bool = False,
                 tensor_quantizer_params: Union[TensorQuantizerParams, None] = None):
        """
        Args:
            quant_info: libquant_info.QcQuantizeInfo object holding quantization parameters passed to the C++ op
            quant_scheme: Quantization scheme (e.g. QuantScheme.post_training_tf)
            rounding_mode: Rounding mode (e.g. nearest)
            op_mode: QcQuantizeOp mode (e.g. update_stats)
            bitwidth: Quantization bitwidth
            use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
            tensor_quantizer_params: Parameters like number of output channels, axis if per channel quantization is performed
        """
        self.quant_info = quant_info
        self.quant_scheme = quant_scheme
        self.rounding_mode = rounding_mode
        self._is_encoding_frozen = False
        self._tensor_quantizer = None
        self._set_tensor_quantizer(self._build_tensor_quantizer())
        self.op_mode = op_mode
        self.bitwidth = bitwidth
        self.use_symmetric_encodings = use_symmetric_encodings
        self.enabled = True
        self._data_type = QuantizationDataType.int
        self.tensor_quantizer_params = tensor_quantizer_params
        self._reset_encodings()

    def is_encoding_frozen(self) -> bool:
        """ Returns is_encoding_frozen var """
        return self._is_encoding_frozen

    def freeze_encodings(self):
        """ Sets encodings to frozen """
        self._is_encoding_frozen = True

    def enable_per_channel_quantization(self):
        """
        Enables per channel quantization for qc_quantize_op
        """
        assert self.tensor_quantizer_params is not None
        assert self.tensor_quantizer_params.channel_axis is not None
        channel_axis = self.tensor_quantizer_params.channel_axis
        self.quant_info.usePerChannelMode = True
        num_channels = self.tensor_quantizer_params.tensor_shape[channel_axis]
        self._create_tensor_quantizers(num_channels)
        self.quant_info.channelAxis = channel_axis if channel_axis >= 0 else channel_axis + len(self.tensor_quantizer_params.tensor_shape)

    def _create_tensor_quantizers(self, num: int):
        tensor_quantizers = []
        for _ in range(num):
            tensor_quantizer = self._build_tensor_quantizer()
            tensor_quantizer.setStrictSymmetric(self.use_strict_symmetric)
            tensor_quantizer.setUnsignedSymmetric(self.use_unsigned_symmetric)
            tensor_quantizer.isEncodingValid = False
            tensor_quantizers.append(tensor_quantizer)

        self._tensor_quantizer = tensor_quantizers
        self.quant_info.tensorQuantizerRef = [libpymo.PtrToInt64(tensor_quantizer)
                                              for tensor_quantizer in tensor_quantizers]

        self.reset_encoding_stats()

    def _enable_blockwise_quantization(self, block_size):
        assert self.tensor_quantizer_params is not None
        tensor_shape = self.tensor_quantizer_params.tensor_shape
        block_axis = self.tensor_quantizer_params.block_axis
        channel_axis = self.tensor_quantizer_params.channel_axis
        assert channel_axis is not None
        assert block_axis is not None
        assert block_axis != channel_axis
        if tensor_shape[block_axis] % block_size != 0:
            raise ValueError(f"Input shape {tensor_shape} not divisible by block size {block_size} at axis {block_axis}")

        num_quantizers = tensor_shape[channel_axis] * tensor_shape[block_axis] // block_size
        self._create_tensor_quantizers(num_quantizers)

        self.quant_info.usePerChannelMode = True
        self.quant_info.channelAxis = channel_axis if channel_axis >= 0 else channel_axis + len(tensor_shape)
        self.quant_info.blockAxis = block_axis if block_axis >= 0 else block_axis + len(tensor_shape)
        self.quant_info.blockSize = block_size

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
        if not self._is_encoding_frozen:
            self._data_type = data_type
            self.quant_info.isIntDataType = False
            if data_type == QuantizationDataType.int:
                self.quant_info.isIntDataType = True

    def _build_tensor_quantizer(self):
        return libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[self.quant_scheme],
                                       MAP_ROUND_MODE_TO_PYMO[self.rounding_mode])

    def _set_tensor_quantizer(self, tensor_quantizer: libpymo.TensorQuantizer):
        """
        Stores tensor_quantizer in self._tensor_quantizer and passes a pointer to the object
        to the C++ op's QcQuantInfo object
        :param tensor_quantizer: The libpymo.TensorQuantizer object to give to the C++ op
        """
        self._tensor_quantizer = [tensor_quantizer]
        self.quant_info.tensorQuantizerRef = [libpymo.PtrToInt64(tensor_quantizer)]

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
        if not self._is_encoding_frozen:
            self.quant_info.useSymmetricEncoding = use_symmetric_encodings

    @property
    def use_strict_symmetric(self) -> bool:
        """
        Reads useStrictSymmetric config from Tensor Quantizer
        :return: True if strict symmetric mode is to be used, False otherwise
        """
        return self._tensor_quantizer[0].getStrictSymmetric()

    @use_strict_symmetric.setter
    def use_strict_symmetric(self, use_strict_symmetric: bool):
        """
        Sets the useStrictSymmetric associated with the Tensor Quantizer
        :param use_strict_symmetric: True if strict symmetric mode is to be used, False otherwise
        """
        for tensor_quantizer in self._tensor_quantizer:
            tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
        self._reset_encodings()

    @property
    def use_unsigned_symmetric(self) -> bool:
        """
        Reads useStrictSymmetric config from Tensor Quantizer
        :return: True if unsigned symmetric mode is to be used, False otherwise
        """
        return self._tensor_quantizer[0].getUnsignedSymmetric()

    @use_unsigned_symmetric.setter
    def use_unsigned_symmetric(self, use_unsigned_symmetric: bool):
        """
        Sets the useUnsignedSymmetric associated with the Tensor Quantizer
        :param use_unsigned_symmetric: True if unsigned symmetric mode is to be used, False otherwise
        """
        for tensor_quantizer in self._tensor_quantizer:
            tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)
        self._reset_encodings()

    @property
    def encodings(self) -> Optional[List[libpymo.TfEncoding]]:
        """
        Reads the encodings object from the node's QcQuantizeInfo
        :return: The libpymo.TfEncoding object used to store the node's quantization encoding
        """
        if not self.is_initialized() or self.data_type == QuantizationDataType.float:
            return None
        return self.quant_info.encoding

    def update_quantizer_and_load_encodings(self, encoding: List[libpymo.TfEncoding], is_symmetric: Optional[bool],
                                            is_strict_symmetric: Optional[bool], is_unsigned_symmetric: Optional[bool],
                                            data_type: QuantizationDataType):
        """
        Update quantizer settings and load pre-existing encodings to quantizer which can be used during
        quantize-dequantize.

        :param encoding: The libpymo.TfEncoding object to be used by the C++ op
        :param is_symmetric: True if encoding is symmetric, False otherwise
        :param is_strict_symmetric: True if encoding is strict symmetric, False otherwise
        :param is_unsigned_symmetric: True if encoding is unsigned symmetric, False otherwise
        :param data_type: Data type of encoding
        """
        self.enabled = True
        self.bitwidth = encoding[0].bw
        self.data_type = data_type
        if self.data_type == QuantizationDataType.int:
            assert self.use_symmetric_encodings is not None
            assert self.use_strict_symmetric is not None
            assert self.use_unsigned_symmetric is not None

            self.use_symmetric_encodings = is_symmetric
            if self.use_symmetric_encodings:
                self.use_strict_symmetric = is_strict_symmetric
            # is_unsigned_symmetric is a special case since the flag could be enabled but the encoding can be signed
            # if the observed tensor had negative values.
            # To err on the side of caution, only set self.use_unsigned_symmetric if we know for sure that the encodings
            # were unsigned.
            if self.use_symmetric_encodings and is_unsigned_symmetric:
                self.use_unsigned_symmetric = is_unsigned_symmetric

            self.load_encodings(encoding)

    def load_encodings(self, encoding: List[libpymo.TfEncoding]):
        """
        Load pre-existing encodings to the quantizer and sets the op-mode to quantize-dequantize

        :param encoding: The list of libpymo.TfEncoding objects to be used by the C++ op
        """
        assert isinstance(encoding, (list, tuple))
        assert len(encoding) == len(self._tensor_quantizer)
        if self.data_type == QuantizationDataType.float:
            raise RuntimeError(f"`load_encodings` is not supported for floating-point quantizers.")
        for tensor_quantizer in self._tensor_quantizer:
            tensor_quantizer.isEncodingValid = True
        self.op_mode = OpMode.quantizeDequantize
        # pylint: disable=attribute-defined-outside-init
        self._encoding = encoding
        self.quant_info.encoding = encoding

    @encodings.setter
    @deprecated(f"Use {load_encodings.__qualname__} instead.")
    def encodings(self, encoding: Union[List[libpymo.TfEncoding], None]):
        """Deprecated. Use :meth:`load_encodings` to set the quantizer encodings.

        Stores encoding in self._encoding to prevent deletion and sets self.quant_info.encoding to point to encoding.
        If encoding is None, creates an empty encoding to prevent seg faults
        :param encoding: The libpymo.TfEncoding object to be used by the C++ op
        """
        if encoding is None:
            self._reset_encodings()

        else:
            self.load_encodings(encoding)

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
        if not self._is_encoding_frozen:
            for tensor_quantizer in self._tensor_quantizer:
                tensor_quantizer.resetEncodingStats()
            self._reset_encodings()

    def _reset_encodings(self):
        """
        Resets the quantizer's encodings
        """
        if self.is_encoding_frozen():
            return

        encodings = []
        for tensor_quantizer in self._tensor_quantizer:
            encoding = libpymo.TfEncoding()
            encoding.bw = self.bitwidth
            encodings.append(encoding)
            tensor_quantizer.isEncodingValid = False

        self._encoding = encodings
        self.quant_info.encoding = encodings

    def set_bitwidth(self, bitwidth: int):
        """
        Set bitwidth for quantization
        """
        if not self._is_encoding_frozen:
            self.bitwidth = bitwidth
            self._reset_encodings()

    def set_quant_scheme(self, quant_scheme: QuantScheme):
        """
        Set QcQuantizeOp as given quant scheme
        """
        self.quant_scheme = quant_scheme
        if self.quant_info.usePerChannelMode:
            self.enable_per_channel_quantization()
        else:
            self._set_tensor_quantizer(self._build_tensor_quantizer())
        self.reset_encoding_stats()

    def compute_encodings(self) -> Optional[List[libpymo.TfEncoding]]:
        """
        Compute and return encodings of each tensor quantizer
        """
        if not self._is_encoding_frozen:
            if self.enabled:
                encodings = []
                for tensor_quantizer in self._tensor_quantizer:
                    encodings.append(tensor_quantizer.computeEncoding(self.bitwidth, self.use_symmetric_encodings))
                self.load_encodings(encodings)
            else:
                encodings = None

            return encodings
        return None

    def get_stats_histogram(self) -> List[List]:
        """
        NOTE: Not to invoke when quantization scheme is not TF-Enhanced.

        Get histogram of statistics. Returns list of buckets where each bucket is
        tuple of two values - the float value representing the left edge of the
        bucket and a PDF of the values in this bucket relative to all the values
        seen across all buckets.

        :return: List of buckets where each bucket is (xLeft, PDF).
        """
        if self.quant_scheme != QuantScheme.post_training_tf_enhanced:
            raise RuntimeError("get_stats_histogram() can be invoked only when quantization scheme is TF-Enhanced.")

        if not self.encodings:
            raise RuntimeError("get_stats_histogram() can be invoked only when encoding is computed.")

        histogram = []
        for tensor_quantizer in self._tensor_quantizer:
            histogram.append(tensor_quantizer.getStatsHistogram())

        return histogram

    def is_initialized(self) -> bool:
        """
        Returns True if all quantizers have been initialized, False otherwise
        """
        if self.data_type == QuantizationDataType.float:
            # Fp16 quantizers do not need to be initialized
            return True

        if all(tq.isEncodingValid for tq in self._tensor_quantizer):
            return True

        return False

    def export_encodings(self, encoding_version: str = "0.6.1"):
        """
        Exports the quantizer's encodings in the selected format.

        :param encoding_version: Version string indicated the encoding export format.
        """
        if encoding_version == '0.6.1':
            return self._export_legacy_encodings()

        raise RuntimeError(f"Unsupported encoding export version: {encoding_version}")

    def _export_legacy_encodings(self) -> Union[List, None]:
        """
        Create encoding dictionary from encoding object

        :return: List of encoding dictionaries in 0.6.1 encoding format
        """
        if not self.enabled or not self.is_initialized():
            return None

        if self.data_type == QuantizationDataType.float:
            return [{'bitwidth': self.bitwidth, 'dtype': "float"}]

        if self.data_type == QuantizationDataType.int:
            encodings = []
            for encoding in self.encodings:
                enc_dict = dict(min=encoding.min,
                                max=encoding.max,
                                scale=encoding.delta,
                                offset=int(encoding.offset),
                                bitwidth=encoding.bw,
                                is_symmetric=str(self.use_symmetric_encodings),
                                dtype="int")
                encodings.append(enc_dict)
            return encodings

        raise RuntimeError(f"Exporting data type {self.data_type} not supported")
