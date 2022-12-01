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
from aimet_tensorflow.examples.test_models import keras_functional_conv_net
from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.utils.quantizer_utils import get_enabled_activation_quantizers, get_enabled_param_quantizers


class TestQuantizerUtils:
    """Test Quantizer Utils"""
    def test_get_enabled_activation_quantizers(self):
        model = keras_functional_conv_net()
        fold_all_batch_norms(model)
        sim = QuantizationSimModel(model)

        enabled_quantizers = get_enabled_activation_quantizers(sim)
        # total 9 activation quantizers, one model input quantizer and 8 layer output quantizers
        #   are enabled as per default config file.
        assert len(enabled_quantizers) == 7

    def test_get_enabled_param_quantizers(self):
        model = keras_functional_conv_net()
        fold_all_batch_norms(model)
        sim = QuantizationSimModel(model)

        enabled_quantizers = get_enabled_param_quantizers(sim)
        # total 5 parameter quantizers, 4 conv kernel quantizers and 1 PReLU alpha quantizer
        #   are enabled as per default config file.
        assert len(enabled_quantizers) == 5
