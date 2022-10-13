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

import os
from typing import Union
from enum import Enum
import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef
import aimet_common.libpymo as libpymo
from aimet_common.defs import QuantScheme, MAP_QUANT_SCHEME_TO_PYMO, MAP_ROUND_MODE_TO_PYMO


qc_quantize_op_dict = {}
TF_ENHANCED_USE_DOWNSAMPLING = bool(int(os.environ.get("AIMET_TFE_USE_DOWNSAMPLING", "0")))
TF_ENHANCED_OFFSET_FACTOR = 0
TF_ENHANCED_STRIDE_FACTOR = 2


@onnx_op(op_type='QcQuantizeOp',
         inputs=[PyCustomOpDef.dt_float],
         outputs=[PyCustomOpDef.dt_float],
         attrs=['op_name'],
         )
def custom_onnxruntime_op(x, **kwargs):
    """
    Custom ONNXRuntime operations
    """
    op_name = kwargs['op_name']
    qc_op = qc_quantize_op_dict[op_name]

    # TODO
    # For non-four-dimensional tensors, set op_mode to update_stats, libpymo.tensorQunatizer doesn't support.
    if len(x.shape) != 4:
        qc_op.set_mode(OpMode.update_stats)

    return qc_op.compute(x)


def reset_qc_quantize_op_dict():
    """
    Reset qc_quantize_op dict to prevent overwrite
    """
    # pylint: disable=global-statement
    global qc_quantize_op_dict
    qc_quantize_op_dict.clear()


class OpMode(Enum):
    """
    op modes for tensor quantizer
    """
    update_stats = 1
    quantize_dequantize = 2
    one_shot_quantize_dequantize = 3


class QcQuantizeOp:
    """ A custom quantization operation to perform using ONNXRuntime """

    # pylint: disable=too-many-arguments
    def __init__(self, quant_scheme: QuantScheme = QuantScheme.post_training_tf,
                 rounding_mode: str = 'nearest',
                 encodings: Union[libpymo.TfEncoding, None] = None,
                 op_mode: Union[OpMode, None] = None,
                 bitwidth: int = 8, use_symmetric_encodings: bool = False,
                 use_cuda: bool = False):
        """
        Args:
            quant_scheme: Quantization scheme (e.g. QuantScheme.post_training_tf)
            rounding_mode: Rounding mode (e.g. nearest)
            encodings: libpymo.TfEncoding object with min, max, offset, delta, bw
            op_mode: QcQuantizeOp mode (e.g. update_stats)
            bitwidth: Quantization bitwidth
            use_symmetric_encodings: True if symmetric encoding is used.  False otherwise.
            use_cuda: True if using CUDA to run quantization op. False otherwise.
        """
        self.quant_scheme = quant_scheme
        self.rounding_mode = rounding_mode
        self.tensor_quantizer = self._build_tensor_quantizer()
        self.encodings = encodings
        self.op_mode = op_mode
        self.bitwidth = bitwidth
        self.use_symmetric_encodings = use_symmetric_encodings
        self.use_cuda = use_cuda
        self.enabled = True

    def _build_tensor_quantizer(self):
        return libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[self.quant_scheme],
                                       MAP_ROUND_MODE_TO_PYMO[self.rounding_mode])

    @property
    def use_strict_symmetric(self) -> bool:
        """
        Reads useStrictSymmetric config from Tensor Quantizer
        :return: True if strict symmetric mode is to be used, False otherwise
        """
        return self.tensor_quantizer.getStrictSymmetric()

    @use_strict_symmetric.setter
    def use_strict_symmetric(self, use_strict_symmetric: bool):
        """
        Sets the useStrictSymmetric associated with the Tensor Quantizer
        :param use_strict_symmetric: True if strict symmetric mode is to be used, False otherwise
        """
        self.tensor_quantizer.setStrictSymmetric(use_strict_symmetric)
        self.encodings = None

    @property
    def use_unsigned_symmetric(self) -> bool:
        """
        Reads useStrictSymmetric config from Tensor Quantizer
        :return: True if unsigned symmetric mode is to be used, False otherwise
        """
        return self.tensor_quantizer.getUnsignedSymmetric()

    @use_unsigned_symmetric.setter
    def use_unsigned_symmetric(self, use_unsigned_symmetric: bool):
        """
        Sets the useUnsignedSymmetric associated with the Tensor Quantizer
        :param use_unsigned_symmetric: True if unsigned symmetric mode is to be used, False otherwise
        """
        self.tensor_quantizer.setUnsignedSymmetric(use_unsigned_symmetric)
        self.encodings = None

    def set_encodings(self, encodings: libpymo.TfEncoding):
        """
        set encodings of tensor quantizer by given encodings
        """
        self.encodings = encodings
        self.tensor_quantizer.isEncodingValid = True

    def get_encodings(self):
        """
        return the encodings of tensor quantizer
        """
        return self.encodings

    def set_mode(self, op_mode: OpMode):
        """
        set the mode of QcQuantizeOp
        """
        self.op_mode = op_mode

    def reset_encoding_stats(self):
        """
        reset the stats of tensor quantizer
        """
        self.encodings = None
        self.tensor_quantizer.resetEncodingStats()

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
        self.tensor_quantizer = self._build_tensor_quantizer()
        self.reset_encoding_stats()

    def compute_encodings(self):
        """
        Compute and return encodings of each tensor quantizer
        """
        self.encodings = self.tensor_quantizer.computeEncoding(self.bitwidth,
                                                               self.use_symmetric_encodings)
        return self.encodings

    def compute(self, in_tensor: Union[None, np.array] = None):
        """
        forward function called by ONNXRuntime
        """
        if self.enabled:
            if self.op_mode == OpMode.update_stats:
                output = in_tensor
                if TF_ENHANCED_USE_DOWNSAMPLING and self.quant_scheme == QuantScheme.post_training_tf_enhanced:
                    in_tensor_flatten = in_tensor.reshape(-1)
                    in_tensor = \
                        in_tensor_flatten[TF_ENHANCED_OFFSET_FACTOR::TF_ENHANCED_STRIDE_FACTOR].astype(np.float32)
                self.tensor_quantizer.updateStats(in_tensor, self.use_cuda)

            elif self.op_mode == OpMode.one_shot_quantize_dequantize:
                out_tensor = np.zeros(in_tensor.shape).astype(np.float32)
                self.reset_encoding_stats()
                self.tensor_quantizer.updateStats(in_tensor, self.use_cuda)
                self.encodings = self.tensor_quantizer.computeEncoding(self.bitwidth,
                                                                       self.use_symmetric_encodings)
                self.tensor_quantizer.quantizeDequantize(in_tensor, out_tensor, self.encodings.min,
                                                         self.encodings.max, self.bitwidth, self.use_cuda)
                output = out_tensor

            elif self.op_mode == OpMode.quantize_dequantize:
                out_tensor = np.zeros(in_tensor.shape).astype(np.float32)
                self.tensor_quantizer.quantizeDequantize(in_tensor, out_tensor, self.encodings.min,
                                                         self.encodings.max, self.bitwidth, self.use_cuda)
                output = out_tensor
            else:
                raise Exception("Please initialize the op_mode before calling compute function")

        else:
            output = in_tensor

        return output
