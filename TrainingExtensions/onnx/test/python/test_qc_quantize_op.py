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
import numpy as np
from aimet_common import libpymo
from aimet_common.defs import QuantScheme
from aimet_onnx.qc_quantize_op import QcQuantizeOp, OpMode


class TestQcQuantizeOp:

    def test_update_stats_with_pymo(self):

        input_arr = np.random.rand(1, 3, 4, 4).astype(np.float32)
        qc_op = QcQuantizeOp(rounding_mode='stochastic', quant_scheme=QuantScheme.post_training_tf, bitwidth=8, op_mode=OpMode.update_stats)
        qc_op.compute(input_arr)

        encodings = qc_op.compute_encodings()
        print("Encoding returned: min={}, max={}, offset={}. delta={}, bw={}"
              .format(encodings.min, encodings.max, encodings.offset, encodings.delta, encodings.bw))
        assert encodings is not None
        assert qc_op.tensor_quantizer.isEncodingValid is True

    def test_quantize_dequantize_with_pymo(self):

        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        qc_op = QcQuantizeOp(quant_scheme=QuantScheme.post_training_tf, rounding_mode='nearest', bitwidth=8,
                             use_symmetric_encodings=False, use_cuda=False)

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 1
        encodings.min = -5.0

        qc_op.set_encodings(encodings)

        qc_op.set_mode(OpMode.quantize_dequantize)
        output = qc_op.compute(input_arr)

        print(output)
        assert np.max(output) <= 1.1
        assert np.min(output) >= -5


    def test_one_shot_quantize_dequantize_asymmetric_cpu(self):
        qc_op = QcQuantizeOp(quant_scheme=QuantScheme.post_training_tf, rounding_mode='nearest',
                             op_mode=OpMode.one_shot_quantize_dequantize, bitwidth=8,
                             use_symmetric_encodings=False, use_cuda=False)
        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        output_oneshot = qc_op.compute(input_arr)

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 2.5
        encodings.min = -7
        encodings.offset = -188
        qc_op.set_encodings(encodings)
        qc_op.set_mode(OpMode.quantize_dequantize)

        output_qdq = qc_op.compute(input_arr)

        assert np.allclose(output_oneshot, output_qdq)

    def test_one_shot_quantize_dequantize_symmetric_signed_cpu(self):
        qc_op = QcQuantizeOp(quant_scheme=QuantScheme.post_training_tf, rounding_mode='nearest',
                             op_mode=OpMode.one_shot_quantize_dequantize, bitwidth=8,
                             use_symmetric_encodings=True, use_cuda=False)
        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        output_oneshot = qc_op.compute(input_arr)

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 7
        encodings.min = -7
        encodings.offset = -128
        qc_op.set_encodings(encodings)
        qc_op.set_mode(OpMode.quantize_dequantize)

        output_qdq = qc_op.compute(input_arr)

        assert np.allclose(output_oneshot, output_qdq)

    def test_one_shot_quantize_dequantize_symmetric_unsigned_cpu(self):
        qc_op = QcQuantizeOp(quant_scheme=QuantScheme.post_training_tf, rounding_mode='nearest',
                             op_mode=OpMode.one_shot_quantize_dequantize, bitwidth=8,
                             use_symmetric_encodings=True, use_cuda=False)
        qc_op.use_unsigned_symmetric = True
        input_arr = np.asarray([[[[0, 1.2, 1.5, 4.0, 4.9, 5.3]]]]).astype(np.float32)
        output_oneshot = qc_op.compute(input_arr)

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.3
        encodings.min = 0.0
        encodings.offset = 0
        qc_op.set_encodings(encodings)
        qc_op.set_mode(OpMode.quantize_dequantize)

        output_qdq = qc_op.compute(input_arr)

        assert np.allclose(output_oneshot, output_qdq)

    def test_set_get_properties(self):
        qc_op = QcQuantizeOp(quant_scheme=QuantScheme.post_training_tf, rounding_mode='nearest',
                             op_mode=OpMode.one_shot_quantize_dequantize, bitwidth=8,
                             use_symmetric_encodings=True, use_cuda=False)
        qc_op.use_strict_symmetric = True
        assert qc_op.use_strict_symmetric == True

        qc_op.use_unsigned_symmetric = False
        assert qc_op.use_unsigned_symmetric == False