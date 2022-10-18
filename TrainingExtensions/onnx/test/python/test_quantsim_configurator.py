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
import json
import os
import numpy as np
from onnx import load_model
from aimet_onnx.quantsim import QuantizationSimModel
import test_models


class TestQuantSimConfig:
    """Tests for applying config to QuantizationSimModel"""
    def test_qs_config_dummy_model(self):
        model = test_models.build_dummy_model()
        sim = QuantizationSimModel(model)
        assert sim.qc_quantize_op_dict['conv_w'].enabled == False
        assert sim.qc_quantize_op_dict['conv_b'].enabled == False
        assert sim.qc_quantize_op_dict['fc_w'].enabled == False
        assert sim.qc_quantize_op_dict['fc_b'].enabled == False
        assert sim.qc_quantize_op_dict['input'].enabled == False
        assert sim.qc_quantize_op_dict['3'].enabled == False
        assert sim.qc_quantize_op_dict['4'].enabled == False
        assert sim.qc_quantize_op_dict['5'].enabled == False
        assert sim.qc_quantize_op_dict['6'].enabled == False
        assert sim.qc_quantize_op_dict['output'].enabled == False