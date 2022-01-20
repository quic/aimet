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
import tensorflow as tf
from packaging import version

from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper
import libpymo

def test_quantize_resnet50_keras():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        rand_inp = np.random.randn(1, 224, 224, 3)
        model = tf.keras.applications.resnet50.ResNet50()
        orig_out = model(rand_inp)
        qsim = QuantizationSimModel(model)
        for wrapper in qsim.quant_wrappers():
            for input_q in wrapper.input_quantizers:
                input_q.disable()
        qsim.compute_encodings(lambda m, _: m(rand_inp), None)
        quant_out = qsim.model(rand_inp)
        assert not np.array_equal(orig_out, quant_out)
        for wrapper in qsim.quant_wrappers():
            for input_q in wrapper.input_quantizers:
                assert input_q.quant_mode == int(libpymo.TensorQuantizerOpMode.passThrough)
                assert input_q.encoding is None
            for param_q in wrapper.param_quantizers:
                assert param_q.quant_mode == int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize)
                assert param_q.encoding is not None
            for output_q in wrapper.output_quantizers:
                assert output_q.quant_mode == int(libpymo.TensorQuantizerOpMode.quantizeDequantize)
                assert output_q.encoding is not None
        for idx, layer in enumerate(model.layers):
            if isinstance(qsim.model.layers[idx], QcQuantizeWrapper):
                assert layer.name == qsim.model.layers[idx]._layer_to_wrap.name
