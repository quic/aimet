# /usr/bin/env python3.8
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
import os
import onnx
import onnxruntime as ort
from onnx import helper
from aimet_common import libpymo
from aimet_common.defs import QuantScheme, MAP_QUANT_SCHEME_TO_PYMO, MAP_ROUND_MODE_TO_PYMO
from aimet_onnx.qc_quantize_op import QcQuantizeOp, OpMode
from aimet_onnx import libquant_info
from aimet_common.libpymo import TensorQuantizerOpMode as PyMoOpMode


library_name = "libaimet_ort_ops.so"
op_domain = "aimet.customop"
op_name = "QcQuantizeOp"

shared_library = '/'.join(libquant_info.__file__.split('/')[0:-1])
shared_library = os.path.join(shared_library, library_name)

def create_quant_info(encoding,
                    tensor_quantizer,
                    opMode,
                    useSymmetricEncoding=True,
                    enabled=True,
                    bitwidth=8):
    quant_info = libquant_info.QcQuantizeInfo()
    encoding.bw = bitwidth
    quant_info.encoding = encoding
    quant_info.opMode = opMode
    quant_info.useSymmetricEncoding = useSymmetricEncoding
    quant_info.enabled = enabled
    quant_info.tensorQuantizerRef = libpymo.PtrToInt64(tensor_quantizer)
    return quant_info


def create_model_from_node(quant_node, shape):
    input_info = helper.make_tensor_value_info(name=quant_node.input[0], elem_type=helper.TensorProto.FLOAT,
                                               shape=shape)

    output_info = helper.make_tensor_value_info(name=quant_node.output[0], elem_type=helper.TensorProto.FLOAT,
                                                shape=shape)
    onnx_graph = helper.make_graph([quant_node],
                                   'dummy_graph', [input_info], [output_info],
                                   [])

    model = helper.make_model(onnx_graph, opset_imports=[onnx.helper.make_opsetid(op_domain, 13)])
    return model


def build_session(model, providers):
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(shared_library)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        path_or_bytes=model.SerializeToString(),
        sess_options=sess_options,
        providers=providers,
    )
    return session


def onnx_callback(session, inputs):
    in_tensor = {'input': inputs}
    session.run(None, in_tensor)


class TestQcQuantizeOp:

    def test_compare_update_stats(self):

        input_arr = np.random.rand(1, 3, 4, 4).astype(np.float32)
        qc_op = QcQuantizeOp(rounding_mode='stochastic', quant_scheme=QuantScheme.post_training_tf, bitwidth=8, op_mode=OpMode.update_stats)
        qc_op.compute(input_arr)
        encodings = qc_op.compute_encodings()

        assert encodings is not None
        assert qc_op.tensor_quantizer.isEncodingValid is True
        tensor_quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                    MAP_ROUND_MODE_TO_PYMO['stochastic'])
        cxx_encodings = libpymo.TfEncoding()
        quant_info = create_quant_info(cxx_encodings, tensor_quantizer, PyMoOpMode.updateStats, useSymmetricEncoding=False)
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, ['CPUExecutionProvider'])
        session.run(None, {'input':input_arr})
        cxx_encodings = tensor_quantizer.computeEncoding(cxx_encodings.bw,
                                         quant_info.useSymmetricEncoding)

        assert quant_info.tensorQuantizerRef.isEncodingValid is True
        assert cxx_encodings.max == encodings.max
        assert cxx_encodings.min == encodings.min
        assert cxx_encodings.delta == encodings.delta
        assert cxx_encodings.offset == encodings.offset


    def test_compare_quantize_dequantize(self):

        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        qc_op = QcQuantizeOp(quant_scheme=QuantScheme.post_training_tf, rounding_mode='nearest', bitwidth=8,
                             use_symmetric_encodings=False, use_cuda=False)

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 1.
        encodings.min = -5.0

        qc_op.set_encodings(encodings)

        qc_op.set_mode(OpMode.quantize_dequantize)
        output = qc_op.compute(input_arr)

        tensor_quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                    MAP_ROUND_MODE_TO_PYMO['nearest'])
        tensor_quantizer.isEncodingValid = True
        quant_info = create_quant_info(encodings, tensor_quantizer, PyMoOpMode.quantizeDequantize,
                                       useSymmetricEncoding=False)
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, ['CPUExecutionProvider'])
        cxx_output = session.run(None, {'input':input_arr})
        assert np.alltrue(output == cxx_output)


    def test_one_shot_quantize_dequantize_asymmetric_cpu(self):
        qc_op = QcQuantizeOp(quant_scheme=QuantScheme.post_training_tf, rounding_mode='nearest',
                             op_mode=OpMode.one_shot_quantize_dequantize, bitwidth=8,
                             use_symmetric_encodings=False, use_cuda=False)
        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        output_oneshot = qc_op.compute(input_arr)

        tensor_quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                    MAP_ROUND_MODE_TO_PYMO['nearest'])
        encodings = libpymo.TfEncoding()
        quant_info = create_quant_info(encodings, tensor_quantizer, PyMoOpMode.oneShotQuantizeDequantize,
                                       useSymmetricEncoding=False)
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, ['CPUExecutionProvider'])
        cxx_output = session.run(None, {'input':input_arr})

        assert np.allclose(output_oneshot, cxx_output)

    def test_one_shot_quantize_dequantize_symmetric_signed_cpu(self):
        qc_op = QcQuantizeOp(quant_scheme=QuantScheme.post_training_tf, rounding_mode='nearest',
                             op_mode=OpMode.one_shot_quantize_dequantize, bitwidth=8,
                             use_symmetric_encodings=True, use_cuda=False)
        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        output_oneshot = qc_op.compute(input_arr)

        tensor_quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                    MAP_ROUND_MODE_TO_PYMO['nearest'])

        encodings = libpymo.TfEncoding()
        quant_info = create_quant_info(encodings, tensor_quantizer, PyMoOpMode.oneShotQuantizeDequantize,
                                       useSymmetricEncoding=True)
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, ['CPUExecutionProvider'])
        cxx_output = session.run(None, {'input':input_arr})

        assert np.alltrue(output_oneshot == cxx_output)

    def test_one_shot_quantize_dequantize_symmetric_unsigned_cpu(self):
        qc_op = QcQuantizeOp(quant_scheme=QuantScheme.post_training_tf, rounding_mode='nearest',
                             op_mode=OpMode.one_shot_quantize_dequantize, bitwidth=8,
                             use_symmetric_encodings=True, use_cuda=False)
        qc_op.use_unsigned_symmetric = True
        input_arr = np.asarray([[[[0, 1.2, 1.5, 4.0, 4.9, 5.3]]]]).astype(np.float32)
        output_oneshot = qc_op.compute(input_arr)

        tensor_quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                    MAP_ROUND_MODE_TO_PYMO['nearest'])
        tensor_quantizer.setUnsignedSymmetric(True)
        encodings = libpymo.TfEncoding()
        quant_info = create_quant_info(encodings, tensor_quantizer, PyMoOpMode.oneShotQuantizeDequantize,
                                       useSymmetricEncoding=True)
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, ['CPUExecutionProvider'])
        cxx_output = session.run(None, {'input':input_arr})

        assert np.alltrue(output_oneshot == cxx_output)
