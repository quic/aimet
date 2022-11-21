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
""" Unit tests for Keras tensor quantizer """
import tensorflow as tf
import numpy as np
import aimet_common.libpymo as libpymo
from aimet_tensorflow.keras.quant_sim.tensor_quantizer import ActivationTensorQuantizer, ParamPerTensorQuantizer
from aimet_common.defs import QuantScheme, QuantizationDataType


def test_set_encodings():
    tf.keras.backend.clear_session()
    quantizer = ActivationTensorQuantizer(tf.keras.layers.Layer(), name='quantizer',
                                          quant_scheme=QuantScheme.post_training_tf,
                                          round_mode='nearest', bitwidth=4, data_type=QuantizationDataType.int,
                                          is_symmetric=True, use_unsigned_symmetric=False,
                                          use_strict_symmetric=True, enabled=True)
    assert not quantizer._is_encoding_valid

    # Create encoding and set
    encoding = libpymo.TfEncoding()
    encoding.min = 0.0
    encoding.max = 30.0
    encoding.bw = 4
    quantizer.encoding = encoding
    quant_encoding = quantizer.encoding

    assert quantizer._is_encoding_valid
    assert quant_encoding.min == 0.0
    assert quant_encoding.max == 30.0
    assert np.allclose(quant_encoding.delta, 2.142857142857143, rtol=0.01)
    assert quant_encoding.offset == 0


def test_tensor_quantizer_freeze_encodings():
    tf.keras.backend.clear_session()
    quantizer = ParamPerTensorQuantizer(tf.keras.layers.Layer(), name='quantizer',
                                        quant_scheme=QuantScheme.post_training_tf, round_mode='nearest',
                                        bitwidth=4, data_type=QuantizationDataType.int, is_symmetric=True,
                                        use_unsigned_symmetric=False, use_strict_symmetric=True, enabled=True)
    # Create encoding and set
    encoding = libpymo.TfEncoding()
    encoding.min = 0.0
    encoding.max = 30.0
    encoding.bw = 4
    quantizer.encoding = encoding

    quantizer.freeze_encoding()
    assert quantizer._is_encoding_frozen
    quantizer.bitwidth = 8
    assert quantizer.bitwidth == 4
    quantizer.quant_scheme = QuantScheme.post_training_tf_enhanced
    assert quantizer.quant_scheme == QuantScheme.post_training_tf
    quantizer.round_mode = 'stochastic'
    assert quantizer.round_mode == libpymo.ROUND_NEAREST
    quantizer.is_symmetric = False
    assert quantizer.is_symmetric
    quantizer.use_unsigned_symmetric = True
    assert not quantizer.use_unsigned_symmetric
    quantizer.use_strict_symmetric = False
    assert quantizer.use_strict_symmetric
    quantizer.disable()
    assert quantizer.quant_mode == int(libpymo.TensorQuantizerOpMode.oneShotQuantizeDequantize)
    quantizer.reset_quant_mode()
    assert quantizer._is_encoding_valid
