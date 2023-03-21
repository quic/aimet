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
import onnx
import math
import onnxruntime as ort
from onnx import helper
import os
import pytest
from aimet_common import libpymo
from aimet_common.defs import QuantScheme, MAP_QUANT_SCHEME_TO_PYMO, MAP_ROUND_MODE_TO_PYMO, QuantizationDataType
from aimet_onnx.qc_quantize_op import QcQuantizeOp, OpMode
from aimet_common import libquant_info


shared_library = os.path.dirname(libquant_info.__file__)
shared_library = os.path.join(shared_library, "libaimet_onnxrt_ops.so")

available_providers = [
    provider
    for provider in ort.get_available_providers()
    if provider not in {"TvmExecutionProvider", "TensorrtExecutionProvider"}
]

if "CUDAExecutionProvider" in available_providers:
    op_domain = "aimet.customop.cuda"
else:
    op_domain = "aimet.customop.cpu"
op_name = "QcQuantizeOp"

def create_quant_info(encoding,
                    tensor_quantizer,
                    opMode,
                    useSymmetricEncoding=False,
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

    model = helper.make_model(onnx_graph)
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

class TestQcQuantizeOp:

    def test_update_stats_with_pymo(self):

        input_arr = np.random.rand(1, 3, 4, 4).astype(np.float32)

        tensor_quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                                   MAP_ROUND_MODE_TO_PYMO['stochastic'])
        cpp_encodings = libpymo.TfEncoding()
        quant_info = create_quant_info(cpp_encodings, tensor_quantizer, OpMode.updateStats,
                                       useSymmetricEncoding=False)
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        session.run(None, {'input': input_arr})
        encodings = tensor_quantizer.computeEncoding(cpp_encodings.bw,
                                                         quant_info.useSymmetricEncoding)
        print("Encoding returned: min={}, max={}, offset={}. delta={}, bw={}"
              .format(encodings.min, encodings.max, encodings.offset, encodings.delta, encodings.bw))
        assert encodings is not None
        assert quant_info.tensorQuantizerRef.isEncodingValid is True

    def test_quantize_dequantize_with_pymo(self):

        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                     quant_scheme=QuantScheme.post_training_tf,
                     rounding_mode='nearest',
                     encodings=None,
                     op_mode=OpMode.oneShotQuantizeDequantize,
                     bitwidth=8,
                     use_symmetric_encodings=False,
                     )

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 1
        encodings.min = -5.0

        qc_op.encodings = encodings

        qc_op.op_mode = OpMode.quantizeDequantize
        output = session.run(None, {'input': input_arr})[0]

        assert np.max(output) <= 1.1
        assert np.min(output) >= -5.1

    def test_update_stats_quantize_dequantize(self):

        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        input_arr2 = np.random.randn(*input_arr.shape).astype(np.float32) * 10
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                     quant_scheme=QuantScheme.post_training_tf,
                     rounding_mode='nearest',
                     encodings=None,
                     op_mode=OpMode.updateStats,
                     bitwidth=8,
                     use_symmetric_encodings=False,
                     )

        session.run(None, {'input': input_arr})[0]
        qc_op.compute_encodings()
        assert math.isclose(qc_op.encodings.max, 2.5, rel_tol=1e-2)
        assert math.isclose(qc_op.encodings.min, -7, rel_tol=1e-2)

        qc_op.op_mode = OpMode.quantizeDequantize
        output = session.run(None, {'input': input_arr2})[0]
        assert np.max(output) <= 2.6
        assert np.min(output) >= -7.1
        assert not np.allclose(output, input_arr2)

    def test_compare_one_shot_with_pymo(self):

        input_arr = np.random.randn(2, 3, 5, 1).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                     quant_scheme=QuantScheme.post_training_tf,
                     rounding_mode='nearest',
                     encodings=None,
                     op_mode=OpMode.oneShotQuantizeDequantize,
                     bitwidth=8,
                     use_symmetric_encodings=False,
                     )

        quantizer = libpymo.TensorQuantizer(MAP_QUANT_SCHEME_TO_PYMO[QuantScheme.post_training_tf],
                                       MAP_ROUND_MODE_TO_PYMO['nearest'])
        out_tensor = np.zeros(input_arr.shape).astype(np.float32)
        # Perform one-shot quant-dequant in python
        quantizer.updateStats(input_arr, False)
        enc = quantizer.computeEncoding(8, False)
        quantizer.quantizeDequantize(input_arr.copy(), out_tensor, enc.min,
                                                 enc.max, 8, False)

        output = session.run(None, {'input': input_arr})[0]
        assert quant_info.encoding.max == enc.max
        assert quant_info.encoding.min == enc.min
        assert np.allclose(output, out_tensor)

    def test_one_shot_quantize_dequantize_asymmetric_cpu(self):
        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)

        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                     quant_scheme=QuantScheme.post_training_tf,
                     rounding_mode='nearest',
                     encodings=None,
                     op_mode=OpMode.oneShotQuantizeDequantize,
                     bitwidth=8,
                     use_symmetric_encodings=False,
                     )

        output_oneshot = session.run(None, {'input': input_arr})[0]

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 2.5
        encodings.min = -7
        encodings.offset = -188
        qc_op.encodings = encodings
        qc_op.op_mode = OpMode.quantizeDequantize

        output_qdq = session.run(None, {'input': input_arr})


        assert np.allclose(output_oneshot, output_qdq)

    def test_one_shot_quantize_dequantize_symmetric_signed_cpu(self):
        input_arr = np.asarray([[[[-7, -5, -3, 0, .1, 2.5]]]]).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                     quant_scheme=QuantScheme.post_training_tf,
                     rounding_mode='nearest',
                     op_mode=OpMode.oneShotQuantizeDequantize,
                     bitwidth=8,
                     use_symmetric_encodings=True,
                     )
        output_oneshot = session.run(None, {'input': input_arr})

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 7
        encodings.min = -7
        encodings.offset = -128
        qc_op.encodings = encodings
        qc_op.op_mode = OpMode.quantizeDequantize

        output_qdq = session.run(None, {'input': input_arr})

        assert np.allclose(output_oneshot, output_qdq)

    def test_one_shot_quantize_dequantize_symmetric_unsigned_cpu(self):
        input_arr = np.asarray([[[[0, 1.2, 1.5, 4.0, 4.9, 5.3]]]]).astype(np.float32)
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        model = create_model_from_node(quant_node, input_arr.shape)
        session = build_session(model, available_providers)
        qc_op = QcQuantizeOp(quant_info=quant_info,
                     quant_scheme=QuantScheme.post_training_tf,
                     rounding_mode='nearest',
                     op_mode=OpMode.oneShotQuantizeDequantize,
                     bitwidth=8,
                     use_symmetric_encodings=True,
                     )

        qc_op.use_unsigned_symmetric = True

        output_oneshot = session.run(None, {'input': input_arr})

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.3
        encodings.min = 0.0
        encodings.offset = 0
        qc_op.encodings = encodings
        qc_op.op_mode = OpMode.quantizeDequantize

        output_qdq = session.run(None, {'input': input_arr})

        assert np.allclose(output_oneshot, output_qdq)

    @pytest.mark.cuda
    def test_one_shot_quantize_dequantize_cpu_vs_gpu(self):
        input_arr = np.asarray([[[[0, 1.2, 1.5, 4.0, 4.9, 5.3]]]]).astype(np.float32)
        quant_info_cpu = libquant_info.QcQuantizeInfo()
        quant_node_cpu = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain="aimet.customop.cpu", quant_info=libpymo.PtrToInt64(quant_info_cpu))
        model_cpu = create_model_from_node(quant_node_cpu, input_arr.shape)
        session_cpu = build_session(model_cpu, available_providers)
        qc_op_cpu = QcQuantizeOp(quant_info=quant_info_cpu,
                     quant_scheme=QuantScheme.post_training_tf,
                     rounding_mode='nearest',
                     op_mode=OpMode.oneShotQuantizeDequantize,
                     bitwidth=8,
                     use_symmetric_encodings=True)

        output_cpu = session_cpu.run(None, {'input': input_arr})

        quant_info_gpu = libquant_info.QcQuantizeInfo()
        quant_node_gpu = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info_gpu))
        model_gpu = create_model_from_node(quant_node_gpu, input_arr.shape)
        session_gpu = build_session(model_gpu, available_providers)
        qc_op_gpu = QcQuantizeOp(quant_info=quant_info_gpu,
                     quant_scheme=QuantScheme.post_training_tf,
                     rounding_mode='nearest',
                     op_mode=OpMode.oneShotQuantizeDequantize,
                     bitwidth=8,
                     use_symmetric_encodings=True)


        output_gpu = session_gpu.run(None, {'input': input_arr})


        assert np.alltrue(output_gpu[0] == output_cpu[0])

    def test_set_get_properties(self):
        quant_info = libquant_info.QcQuantizeInfo()
        quant_node = helper.make_node(op_name, inputs=['input'], outputs=['output'],
                                      domain=op_domain, quant_info=libpymo.PtrToInt64(quant_info))
        qc_op = QcQuantizeOp(quant_info=quant_info,
                     quant_scheme=QuantScheme.post_training_tf,
                     rounding_mode='nearest',
                     op_mode=OpMode.oneShotQuantizeDequantize,
                     bitwidth=8,
                     use_symmetric_encodings=True,
                     )
        qc_op.use_strict_symmetric = True
        assert quant_info.tensorQuantizerRef.getStrictSymmetric() == True

        qc_op.use_unsigned_symmetric = False
        assert quant_info.tensorQuantizerRef.getUnsignedSymmetric()== False

        qc_op.use_unsigned_symmetric = True
        assert quant_info.tensorQuantizerRef.getUnsignedSymmetric() == True

        qc_op.data_type = QuantizationDataType.float
        assert qc_op.data_type == QuantizationDataType.float
        assert qc_op.quant_info.isIntDataType == False
